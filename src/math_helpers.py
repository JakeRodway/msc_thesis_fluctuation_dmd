"""
Mathematical principles and data-handling for the fluctuation-diagnostic analysis.

This module provides the foundational operations used across the analytical pipeline, including:
- Experimental data loading
- Reference STFT spectrogram generation with Viterbi ridge tracking
- Time-window fetching and Fourier-based spectral seeding
- The two-pass seeded BOPDMD fitting routine, with Gavish-Donoho rank selection
- Continuous and discrete eigenvalue-to-physics conversions
"""

import numpy as np
import scipy.signal as sg
import matplotlib as mpl

from .diag_lib import submodule

from .config import (PID, DET, CH_REF, SEARCH_BAND, T_MIN_BUFFER, T_MAX_BUFFER, STFT_MAX_DF, MARGIN, RANK_CAP, EVAL_TREE_CH)

mpl.rcParams.update(mpl.rcParamsDefault)


def load_data(pid=PID, det=DET):
    """
    Loads data for the selected PID and DET, and returns a dictionary with the following keys:
    - d: ProcessedData object for the selected PID and DET
    - fs: Sampling frequency of the data
    - dx_m: Channel spacing in meters
    - n_ch: Number of channels in the data
    - prg: Integer program ID derived from the PID
    - t_on: ECRH heating start time in seconds
    - t_off: ECRH heating end time in seconds
    - et: ECRH time array in seconds
    - ep: ECRH power array in kW
    """
    ProcessedData = submodule("get").ProcessedData
    qp = submodule("query_programs")

    d = ProcessedData(pid, verbose=False, programlogs=False)
    fs = float(d.freq[det - 1])
    dx_m = d._channel_spacing(det)
    n_ch = 32
    
    # Get the integer program ID
    prg = int(pid[2:].replace('.', ''))
    
    # Get the ECRH heating times to define the valid plasma window (t_on, t_off)
    try:
        v, t, _ = qp.get_archive_byname('ECRH_tot', prg, nsamples=5000)
        et = np.asarray(t, dtype=float) / 1e9
        ep = np.asarray(v, dtype=float) / 1e3
        on = ep > 0.05 * np.nanmax(ep)
        t_on, t_off = float(et[on][0]), float(et[on][-1])
    except Exception as e:
        raise RuntimeError(f"CRITICAL: Could not fetch ECRH heating times from the archive. "
                           f"Cannot determine valid plasma window. Archive error: {e}")
        
    return dict(d=d, fs=fs, dx_m=dx_m, n_ch=n_ch, prg=prg, 
                t_on=t_on, t_off=t_off, et=et, ep=ep)


def _viterbi_ridge(score, f_b, t_sp, dt_col, max_df_per_s, smooth_s, jump_pen, valid=None):
    """
    Viterbi ridge tracker for a 2D spectrogram score. Finds the optimal path through the score that 
    maximises the sum of scores while penalising frequency jumps between consecutive time points.

    Parameters:
    - score: 2D array of spectrogram pixel intensities (frequency x time)
    - f_b: frequency bins corresponding to the rows of score
    - t_sp: time points corresponding to the columns of score
    - dt_col: time difference between consecutive columns in score
    - max_df_per_s: maximum allowed frequency drift in Hz/s for the ridge
    - smooth_s: smoothing window length in seconds for the score
    - jump_pen: penalty factor for frequency jumps between consecutive time points
    - valid: optional boolean array indicating valid time points for the ridge

    Returns:
    - ridge_f: frequency values of the detected ridge at each time point
    - ridge_idx: indices of the detected ridge in the frequency bins
    """
    ncol = max(1, int(round(smooth_s / dt_col)) | 1) # number of columns to smooth over, odd for symmetric convolution
    ker = np.ones(ncol) / ncol
    Sm = np.vstack([np.convolve(r, ker, mode='same') for r in score]) # spectrogram smoothed along time
    if valid is not None:
        Sm[:, ~valid] = 0.0 # allows the Viterbi algorithm to ignore invalid columns without affecting the valid ones
    n_f, n_t = Sm.shape
    dfb = f_b[1] - f_b[0]
    allowed = max(max_df_per_s * dt_col / dfb, 1e-9) # number of frequency bins allowed to jump per time step
    jump = np.arange(n_f)[:, None] - np.arange(n_f)[None, :] # generates a matrix of frequency bin differences between all pairs of bins
    pen = jump_pen * (jump / allowed) ** 2 # penalty matrix for frequency jumps, scaled by the allowed jump size

    best = Sm[:, 0].copy() # initialise the best score for the first column
    ptr = np.zeros((n_f, n_t), dtype=np.int32) # pointer matrix to store the index of the best previous frequency bin for each current bin
    for j in range(1, n_t): # iterate over time columns (forward pass)
        tot = best[None, :] - pen # compute the total score for each possible jump from the previous column to the current column
        arg = np.argmax(tot, axis=1) # find the index of the best previous frequency bin for each current bin
        best = tot[np.arange(n_f), arg] + Sm[:, j] # update the best score for the current column by adding the current score
        ptr[:, j] = arg # store the index of the best previous frequency bin for each current bin
    idx = np.empty(n_t, dtype=int) # initialise the index array to store the indices of the detected ridge
    idx[-1] = int(np.argmax(best)) # start backtracking from the last column with the best score
    for j in range(n_t - 1, 0, -1): # iterate backwards over time columns (backward pass)
        idx[j - 1] = ptr[idx[j], j] # follow the pointer to the best previous frequency bin for each current bin
    return f_b[idx], idx


def get_reference_spectrogram(ctx):
    """
    Fetches the pre-computed STFT spectrogram and extracts the dominant coherent 
    frequency ridge using a Viterbi tracker over the valid plasma heating window.

    Parameters:
    - ctx: dictionary containing the ProcessedData object and other context information

    Returns:
    - f: frequency bins of the spectrogram
    - t: time points of the spectrogram
    - Sxx: spectrogram intensity values
    - Sxx_db: spectrogram intensity values in dB
    - ridge_t: time points of the detected ridge
    - ridge_f: frequency values of the detected ridge (masked for invalid times)
    - ridge_f_smooth: smoothed frequency values of the detected ridge
    - ridge_t_valid: time points of the detected ridge within the valid plasma window
    - ridge_f_valid: frequency values of the detected ridge within the valid plasma window
    - disp_a: boolean array indicating the frequency range to display in plots
    - early_excl: boolean array indicating time points before the valid plasma window
    - late_excl: boolean array indicating time points after the valid plasma window
    """
    d = ctx['d']

    f, t, Sxx = d.spectrogram(det=DET, ch=EVAL_TREE_CH) # fetch pre-computed STFT for the chosen channel
    Sxx_db = 10 * np.log10(Sxx / np.mean(Sxx[1:])) # convert to dB and normalise by the mean of the positive frequencies

    # Adaptive Ridge Tracker (Viterbi) setup
    band = (f >= SEARCH_BAND[0]) & (f <= SEARCH_BAND[1]) # frequency band of interest
    f_b = f[band] # frequency bins within band
    P = Sxx[band] # spectrogram intensity values within band
    
    # Two-way background normalisation (divide by row median, subtract column median)
    P_row = P / np.median(P, axis=1, keepdims=True)
    logP = np.log(np.clip(P_row, 1e-12, None))
    score = logP - np.median(logP, axis=0, keepdims=True)
    
    dt_col = float(np.median(np.diff(t))) # time difference between consecutive columns in the spectrogram
    valid = (t >= ctx['t_on'] + T_MIN_BUFFER) & (t <= ctx['t_off'] + T_MAX_BUFFER) # valid time points within the plasma heating window, with buffers applied
    
    # 4. Run Viterbi and smooth
    ridge_f, _ = _viterbi_ridge(score, f_b, t, dt_col,
                                 max_df_per_s=STFT_MAX_DF, smooth_s=30e-3, 
                                 jump_pen=0.005, valid=valid)
    ridge_f_smooth = sg.medfilt(ridge_f, kernel_size=5)

    ridge_f_masked = ridge_f_smooth.copy() 
    ridge_f_masked[~valid] = np.nan # mask the ridge frequency values outside the valid time window with NaN
    
    early_excl = t <= ctx['t_on'] + T_MIN_BUFFER # time points before valid time window
    late_excl = t >= ctx['t_off'] + T_MAX_BUFFER # time points after valid time window
    disp_a = (f >= SEARCH_BAND[0] - 20e3) & (f <= SEARCH_BAND[1] + 20e3) 

    return dict(
        f=f, t=t, Sxx=Sxx, Sxx_db=Sxx_db, 
        ridge_t=t, 
        ridge_f=ridge_f_masked, 
        ridge_f_smooth=ridge_f_smooth, 
        ridge_t_valid=t[valid], 
        ridge_f_valid=ridge_f_smooth[valid], 
        disp_a=disp_a, 
        early_excl=early_excl, 
        late_excl=late_excl
    )


def fetch_window(d, tc, win, fs, margin=MARGIN, det=DET):
    """
    Fetches a precisely sliced time window by fetching extra data on either side of the requested window 
    and then slicing it down to the exact requested size.

    Parameters:
    - d: ProcessedData object for the selected PID and DET
    - tc: center time of the window in seconds
    - win: window length in seconds
    - fs: sampling frequency of the data
    - margin: extra data to fetch on either side of the window in seconds (default: MARGIN)
    - det: detector number (default: DET)

    Returns:
    - X: 2D array of shape (n_channels, n_samples) containing the sliced data for the requested window
    - t_local: 1D array of time points corresponding to the samples in X, relative to the center time tc

    """
    raw, traw = d.get_interval(det=det, interval=[tc - win / 2 - margin, tc + win / 2 + margin])
    raw = raw - raw.mean(axis=1, keepdims=True) # remove channel mean to avoid DC offset issues
    n_win = int(round(win * fs))
    i0 = int(round(margin * fs))
    if raw.shape[1] < i0 + n_win:
        raise ValueError(f'short fetch: got {raw.shape[1]} samples, need {i0+n_win}')
    X = raw[:, i0:i0 + n_win]
    t_local = np.arange(n_win) / fs
    return X, t_local


def local_fourier_seed(X, fs, center, hw_search=8e3, pad=8):
    """
    Computes per window periodogram and returns frequency of peak closest to the Viterbi seeded frequency.

    Parameters:
    - X: 2D array of shape (n_channels, n_samples) containing the data for the window
    - fs: sampling frequency of the data
    - center: frequency around which to search for the peak in Hz
    - hw_search: half-width of the search band around the center frequency in Hz (default: 8e3)
    - pad: padding factor for the FFT to increase frequency resolution (default: 8)

    Returns:
    - peak_freq: frequency of the peak closest to the center frequency in Hz  

    """
    n = X.shape[1]
    win = np.hanning(n)
    f = np.fft.rfftfreq(n * pad, 1 / fs) # padded to increase frequency resolution
    s = (np.abs(np.fft.rfft(X * win, n=n * pad, axis=1)) ** 2).mean(axis=0)
    m = (f >= center - hw_search) & (f <= center + hw_search)
    if not m.any():
        return float(center) # fallback is the global frequency guess from Viterbi
    return float(f[m][np.argmax(s[m])])


def rank_selection(X, cap=RANK_CAP):
    """
    Selects the optimal rank for the data matrix X using the Gavish-Donoho method, capped to a specified limit.

    Parameters:
    - X: 2D array of shape (n_channels, n_samples) containing the data
    - cap: maximum rank to consider (default: RANK_CAP)

    Returns:
    - r_used: the selected rank
    - s: the singular values

    """
    from optht import optht
    U, s, Vh = np.linalg.svd(X, full_matrices=False) # compute SVD of the data matrix X
    r = int(optht(X, sv=s, sigma=None)) # compute the optimal hard threshold rank using the Gavish-Donoho method
    r_used = max(2, min(r, cap)) # cap the rank to be at least 2 and at most the specified cap
    return r_used, s


def fit_bopdmd(X, t_local, rank, seed_freq, real_eig_limit=None):
    """
    Two pass of BOPDMD: first pass unseeded to get a sensible guess for the background/turbulence poles,
    second pass seeded with the modified eigenvalues to force the mode of interest to be captured.

    Parameters:
    - X: 2D array of shape (n_channels, n_samples) containing the data
    - t_local: 1D array of time points corresponding to the samples in X
    - rank: rank to use for the BOPDMD fit
    - seed_freq: frequency to seed the BOPDMD fit in Hz
    - real_eig_limit: optional limit for the real part of the eigenvalues (default: None, which sets a safe default based on the time window)

    Returns:
    - m1: fitted BOPDMD model after the second pass

    """
    from pydmd import BOPDMD
    T = t_local[-1] if t_local[-1] > 0 else 1.0
    if real_eig_limit is None:
        real_eig_limit = min(1e5, 0.5 * 709.0 / T) # 709 is the largest float64 exponent, so this is a safe default for the real eigenvalue limit

    # Initial unseeded fit to get a sensible guess for the background/turbulence poles
    m0 = BOPDMD(svd_rank=rank, num_trials=0, use_proj=False,
                eig_constraints={'limited'}, real_eig_limit=real_eig_limit)
    m0.fit(X, t_local)
    alpha0 = np.array(m0.eigs, dtype=complex) # evals

    i = int(np.argmin(np.abs(np.imag(alpha0) - 2 * np.pi * seed_freq))) # find the index of the eigenvalue closest to the seeded frequency
    j = int(np.argmin(np.abs(np.imag(alpha0) + 2 * np.pi * seed_freq))) # find the index of the eigenvalue closest to the negative of the seeded frequency
    alpha_seeded = alpha0.copy()
    alpha_seeded[i] = 2j * np.pi * seed_freq # overwrite the eigenvalue at index i with the seeded frequency
    if j != i:
        alpha_seeded[j] = -2j * np.pi * seed_freq # overwrite the eigenvalue at index j with the negative of the seeded frequency

    # Second pass: seeded fit with the modified eigenvalues
    m1 = BOPDMD(svd_rank=rank, num_trials=0, use_proj=False,
                init_alpha=alpha_seeded, eig_constraints={'limited'},
                real_eig_limit=real_eig_limit)
    m1.fit(X, t_local)
    return m1


def eig_to_f_gamma_continuous(eigs):
    """
    Convert continuous-time eigenvalues to frequency and growth rate.
    """
    f = np.imag(eigs) / (2 * np.pi)
    g = np.real(eigs)
    return f, g


def eig_to_f_gamma_discrete(eigs, dt):
    """
    Convert discrete-time eigenvalues to frequency and growth rate.
    """
    omega = np.log(eigs.astype(complex)) / dt
    f = np.imag(omega) / (2 * np.pi)
    g = np.real(omega)
    return f, g


def energy_fraction(amplitudes):
    """
    Calculate the energy fraction of each amplitude.
    """
    energy = np.abs(amplitudes) ** 2
    tot = energy.sum()
    return energy / tot if tot > 0 else np.zeros_like(energy)