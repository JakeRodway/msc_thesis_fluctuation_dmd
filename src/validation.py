"""
Method validation and estimator benchmarking via synthetic signal injection.

This module evaluates the accuracy of candidate DMD (BOP-DMD/Optimised DMD vs Hankel DMD) and Fourier 
baselines by recovering known parameters from noisy data. It includes:
- Extraction of real experimental turbulence to act as a realistic background instead of Gaussian white-noise
- Injection of synthetic standing waves with known frequencies, spatial envelopes and decay rates (varying SNR)
- Quantification of growth-rate and frequency errors
- LaTeX table generation to summarise benchmark results
"""

import os
import numpy as np
import scipy.signal as sg

from .diag_lib import submodule

from .config import (DET, WIN_PRIMARY, VAL_INTERVAL, VAL_BAND, VAL_F_INJ, VAL_GAMMA_INJ,
                     VAL_SNRS, VAL_LOBE_CENTRES, VAL_LOBE_WIDTHS, VAL_LOBE_SIGNS,
                     VAL_HANKEL_DELAYS)
from .math_helpers import (
    local_fourier_seed, rank_selection, fit_bopdmd,
    eig_to_f_gamma_continuous, eig_to_f_gamma_discrete
)


def prepare_host_chunks(ctx, win=WIN_PRIMARY, interval=VAL_INTERVAL, band=VAL_BAND, det=DET):
    """
    Prepares the chunks of real diagnostic data that the synthetic mode is injected into. 

    Parameters:
    - ctx: context dictionary containing experimental parameters
    - win: window length in seconds (default: WIN_PRIMARY, the production window)
    - interval: [t0, t1] in seconds, stretch of the discharge used as host data
    - band: tuple specifying the band-pass applied to the host data
    - det: detector number (default: DET)

    Returns:
    - chunks: list of 2D arrays of shape (n_channels, n_samples), one per independent window
    - t_ch: 1D array of time points within a window, shared by every chunk

    """
    pr = submodule("process")
    
    d, fs = ctx['d'], ctx['fs']
    lo, hi = band
    raw, _ = d.get_interval(det=det, interval=list(interval))

    # Divide out the frequency response at band centre so channels are comparable
    H = np.array([pr.get_fcal(np.array([0.5 * (lo + hi)]), det, ch)[0] for ch in range(1, ctx['n_ch'] + 1)])
    data = (raw - raw.mean(axis=1, keepdims=True)) / H[:, None]

    filt = pr.bandpass(data, lo, hi, fs, order=4)
    edge = int(3 * fs / (hi - lo)) # drop the filter transient at each end
    dec = filt[:, edge:-edge]

    n_s = int(round(win * fs))
    n_chunks = dec.shape[1] // n_s
    chunks = [dec[:, i * n_s:(i + 1) * n_s] for i in range(n_chunks)]
    return chunks, np.arange(n_s) / fs


def standing_envelope(n_ch, centres=VAL_LOBE_CENTRES, widths=VAL_LOBE_WIDTHS, signs=VAL_LOBE_SIGNS):
    """
    Builds the spatial envelope A(x) of the injected mode: a sum of signed Gaussian lobes across the
    channels, mimicking the alternating-sign radial structure of an eigenmode.

    Parameters:
    - n_ch: number of channels
    - centres: channel positions of the lobe centres
    - widths: Gaussian widths of the lobes in channels
    - signs: sign of each lobe, alternating to place a node between neighbouring antinodes

    Returns:
    - A: 1D array of length n_ch, normalised to a peak magnitude of 1

    """
    x = np.arange(1, n_ch + 1)
    A = np.zeros(n_ch)
    for c, w, s in zip(centres, widths, signs):
        A += s * np.exp(-0.5 * ((x - c) / w) ** 2)
    return A / np.max(np.abs(A))


def inject_standing(X, t, f_hz, snr, gamma=VAL_GAMMA_INJ, envelope=None):
    """
    Adds A(x) exp(-gamma t) cos(2 pi f t) to a window of real data at a given SNR.

    Parameters:
    - X: 2D array of shape (n_channels, n_samples) containing the host data
    - t: 1D array of time points corresponding to the samples in X
    - f_hz: frequency of the injected mode in Hz
    - snr: standard deviation of the injected wave relative to that of the host data
    - gamma: decay rate of the injected mode in 1/s (so the truth is gamma = -gamma)
    - envelope: spatial envelope A(x) (default: None, which injects a flat profile)

    Returns:
    - X_inj: 2D array of the same shape as X, host data plus the injected mode

    """
    if envelope is None:
        envelope = np.ones(X.shape[0])
    wave = (envelope[:, None] * np.exp(-gamma * t)[None, :]
            * np.cos(2 * np.pi * f_hz * t[None, :]))
    return X + snr * np.std(X) / np.std(wave) * wave


def hankel_embed(X, d):
    """
    Stacks d time-delayed copies of X along the channel axis.
    """
    n_ch, n_s = X.shape
    return np.vstack([X[:, i:i + n_s - d + 1] for i in range(d)])


def measure_bopdmd(X, t, target_freq, fs, rank=None):
    """
    BOP-DMD pipeline exactly as used elsewhere: Gavish-Donoho rank, a periodogram seed refined near
    the target, and the two-pass seeded fit. Eigenvalues are continuous-time, so no ln(lambda)/dt
    step is needed. The seed is part of the method being validated.

    Parameters:
    - X: 2D array of shape (n_channels, n_samples) containing the data
    - t: 1D array of time points corresponding to the samples in X
    - target_freq: known injected frequency in Hz, used to select the pole
    - fs: sampling frequency of the data
    - rank: optional truncation rank (default: None, which uses rank_selection)

    Returns:
    - f: recovered frequency in Hz
    - gamma: recovered growth rate in 1/s
    - rank: the truncation rank used

    """
    if rank is None:
        rank, _ = rank_selection(X)
    m = fit_bopdmd(X, t, rank, local_fourier_seed(X, fs, target_freq))
    f, g = eig_to_f_gamma_continuous(m.eigs)
    cand = np.where(f > 0)[0]
    if not cand.size:
        return np.nan, np.nan, rank
    j = cand[np.argmin(np.abs(f[cand] - target_freq))] # pole nearest the known frequency
    return float(f[j]), float(g[j]), rank


def measure_hankel(X, t, target_freq, d=VAL_HANKEL_DELAYS, rank=None):
    """
    Hankel DMD: delay-embed, then exact DMD. Eigenvalues are discrete-time lambda, so the conversion
    omega = ln(lambda)/dt is required, and that step is where the known damping bias enters. The rank
    comes from the same Gavish-Donoho threshold used for BOP-DMD, so the comparison is like-for-like.

    Parameters:
    - X: 2D array of shape (n_channels, n_samples) containing the data
    - t: 1D array of time points corresponding to the samples in X
    - target_freq: known injected frequency in Hz, used to select the pole
    - d: number of time-delay copies to stack (default: VAL_HANKEL_DELAYS)
    - rank: optional truncation rank (default: None, which uses rank_selection)

    Returns:
    - f: recovered frequency in Hz
    - gamma: recovered growth rate in 1/s
    - rank: the truncation rank used

    """
    from pydmd import HankelDMD
    dt = float(np.mean(np.diff(t)))
    if rank is None:
        rank, _ = rank_selection(hankel_embed(X, d))
    m = HankelDMD(svd_rank=rank, d=d, opt=True, reconstruction_method='mean')
    m.fit(X / np.std(X))
    f, g = eig_to_f_gamma_discrete(np.asarray(m.eigs), dt)
    cand = np.where(f > 0)[0]
    if not cand.size:
        return np.nan, np.nan, rank
    j = cand[np.argmin(np.abs(f[cand] - target_freq))]
    return float(f[j]), float(g[j]), rank


def measure_fourier_freq(X, fs, target_freq, hw=8e3, pad=8):
    """
    Non-DMD frequency baseline: the zero-padded periodogram peak with parabolic sub-bin
    interpolation. This is `local_fourier_seed` plus interpolation, i.e. the very estimator that
    seeds BOP-DMD, so the comparison shows whether the variable-projection fit improves on the seed
    it was handed. The interpolation matters: without it the estimator is quantised by the padded bin
    spacing and saturates regardless of SNR, which would unfairly flatter DMD.

    Parameters:
    - X: 2D array of shape (n_channels, n_samples) containing the data
    - fs: sampling frequency of the data
    - target_freq: centre of the search band in Hz
    - hw: half-width of the search band in Hz
    - pad: padding factor for the FFT

    Returns:
    - f: recovered frequency in Hz

    """
    n = X.shape[1]
    f = np.fft.rfftfreq(n * pad, 1 / fs)
    s = (np.abs(np.fft.rfft(X * np.hanning(n), n=n * pad, axis=1)) ** 2).mean(axis=0)
    m = (f >= target_freq - hw) & (f <= target_freq + hw)
    if not m.any():
        return float(target_freq)
    k = int(np.where(m)[0][np.argmax(s[m])])
    if k <= 0 or k >= len(s) - 1:
        return float(f[k])
    y0, y1, y2 = np.log(s[k - 1]), np.log(s[k]), np.log(s[k + 1])
    den = y0 - 2 * y1 + y2
    delta = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return float(f[k] + delta * (f[1] - f[0]))


def measure_fourier_gamma(X, t, target_freq, halfwidth=25e3):
    """
    Non-DMD growth-rate baseline: band-pass around the mode, take the analytic-signal envelope
    averaged over channels, and fit ln(envelope) linearly in time.

    Parameters:
    - X: 2D array of shape (n_channels, n_samples) containing the data
    - t: 1D array of time points corresponding to the samples in X
    - target_freq: centre of the band-pass in Hz
    - halfwidth: half-width of the band-pass in Hz

    Returns:
    - gamma: recovered growth rate in 1/s

    """
    fs = 1.0 / np.mean(np.diff(t))
    lo, hi = max(target_freq - halfwidth, 1e3), min(target_freq + halfwidth, 0.49 * fs)
    b, a = sg.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    env = np.abs(sg.hilbert(sg.filtfilt(b, a, X, axis=1), axis=1)).mean(axis=0)
    n = len(env) // 10 # drop the edges, where the Hilbert transform is unreliable
    env, tt = env[n:-n], t[n:-n]
    A = np.vstack([tt, np.ones_like(tt)]).T
    (slope, _), *_ = np.linalg.lstsq(A, np.log(np.clip(env, 1e-30, None)), rcond=None)
    return float(slope)


def run_snr_sweep(ctx, chunks, t_ch, snrs=VAL_SNRS, f_inj=VAL_F_INJ, gamma_inj=VAL_GAMMA_INJ,
                  checkpoint_path=None, verbose=True):
    """
    Injects the synthetic mode into every chunk at every SNR and measures it with all three
    estimators.

    Parameters:
    - ctx: context dictionary containing experimental parameters
    - chunks: list of host data windows, from prepare_host_chunks
    - t_ch: 1D array of time points within a window, from prepare_host_chunks
    - snrs: array of signal-to-noise ratios to sweep
    - f_inj: injected frequency in Hz
    - gamma_inj: injected decay rate in 1/s
    - checkpoint_path: optional string specifying a path to save/load intermediate results
    - verbose: whether to print a line per SNR as the sweep proceeds

    Returns:
    - dict containing, for each estimator ('bop', 'han', 'fou'), arrays of shape (n_snrs, n_chunks)
      of recovered frequency ('<tag>_f') and growth rate ('<tag>_g'), the ranks used ('rank_bop',
      'rank_han'), and the sweep settings (snrs, f_inj, gamma_inj, n_chunks, t_window)

    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        return dict(np.load(checkpoint_path))

    fs, env = ctx['fs'], standing_envelope(ctx['n_ch'])
    shape = (len(snrs), len(chunks))
    res = {k: np.full(shape, np.nan) for k in
           ('bop_f', 'bop_g', 'han_f', 'han_g', 'fou_f', 'fou_g', 'rank_bop', 'rank_han')}

    for i, snr in enumerate(snrs):
        for j, c in enumerate(chunks):
            inj = inject_standing(c, t_ch, f_inj, snr, gamma=gamma_inj, envelope=env)
            try:
                res['bop_f'][i, j], res['bop_g'][i, j], res['rank_bop'][i, j] = \
                    measure_bopdmd(inj, t_ch, f_inj, fs)
            except Exception:
                pass
            try:
                res['han_f'][i, j], res['han_g'][i, j], res['rank_han'][i, j] = \
                    measure_hankel(inj, t_ch, f_inj)
            except Exception:
                pass
            try:
                res['fou_f'][i, j] = measure_fourier_freq(inj, fs, f_inj)
            except Exception:
                pass
            try:
                res['fou_g'][i, j] = measure_fourier_gamma(inj, t_ch, f_inj)
            except Exception:
                pass
        if verbose:
            print(f'  SNR {snr:.2f}: gamma bias  '
                  + '  '.join(f'{t.upper()} {np.nanmean(res[f"{t}_g"][i]) + gamma_inj:+8.1f}'
                              for t in ('bop', 'han', 'fou'))
                  + '   |  f RMS (Hz)  '
                  + '  '.join(f'{t.upper()} {np.sqrt(np.nanmean((res[f"{t}_f"][i] - f_inj) ** 2)):7.1f}'
                              for t in ('bop', 'han', 'fou')), flush=True)

    res.update(snrs=np.asarray(snrs), f_inj=f_inj, gamma_inj=gamma_inj,
               n_chunks=len(chunks), t_window=t_ch[-1])
    if checkpoint_path:
        np.savez(checkpoint_path, **res)
    return res


def summarise(D):
    """
    Reduces the sweep to the two numbers quoted per estimator: the RMS error in frequency, and the
    bias in growth rate with its standard deviation across windows.

    Parameters:
    - D: sweep results, from run_snr_sweep

    Returns:
    - dict keyed by estimator tag ('fou', 'han', 'bop'), each containing:
        - f_rms: array of RMS frequency errors, one per SNR
        - g_bias: array of mean growth-rate biases relative to the truth, one per SNR
        - g_sd: array of growth-rate standard deviations across windows, one per SNR

    """
    f_inj, g_true = float(D['f_inj']), -float(D['gamma_inj'])
    return {t: dict(f_rms=np.sqrt(np.nanmean((D[f'{t}_f'] - f_inj) ** 2, axis=1)),
                    g_bias=np.nanmean(D[f'{t}_g'], axis=1) - g_true,
                    g_sd=np.nanstd(D[f'{t}_g'], axis=1))
            for t in ('fou', 'han', 'bop')}


def latex_table(D, label='tab:method_validation'):
    """
    Renders the summary as the LaTeX table used in the report (booktabs + siunitx).

    Parameters:
    - D: sweep results, from run_snr_sweep
    - label: LaTeX label for the table

    Returns:
    - str containing the complete table environment

    """
    S = summarise(D)
    snrs, g_true = D['snrs'], -float(D['gamma_inj'])
    rows = '\n'.join(
        f'{s:.2f} & ' + ' & '.join(f'{S[t]["f_rms"][i]:.1f}' for t in ('fou', 'han', 'bop'))
        + ' & ' + ' & '.join(f'$%+d \\pm %d$' % (round(S[t]['g_bias'][i]), round(S[t]['g_sd'][i]))
                             for t in ('fou', 'han', 'bop')) + r' \\'
        for i, s in enumerate(snrs))

    caption = (
        'Recovery of the injected mode parameters against signal-to-noise\n'
        f'ratio. A standing mode of \\SI{{{float(D["f_inj"]) / 1e3:.0f}}}{{\\kilo\\hertz}} with decay rate\n'
        f'\\SI{{{float(D["gamma_inj"]):.0f}}}{{\\per\\second}} was injected into real measured turbulence and fitted over {int(D["n_chunks"])}\n'
        f'independent \\SI{{{float(D["t_window"]) * 1e3:.1f}}}{{\\milli\\second}} windows, matching the window length used\n'
        'for the experimental analysis. Frequencies are quoted as an RMS deviation from\n'
        'the injected value. Growth rates are quoted as a bias relative to the true\n'
        f'value of \\SI{{{g_true:.0f}}}{{\\per\\second}}, with the standard deviation across the\n'
        'windows. The Fourier column gives a non-DMD baseline: a zero-padded\n'
        'periodogram peak with parabolic sub-bin interpolation for the frequency, and a\n'
        'Hilbert-envelope decay fit for the growth rate.')

    return (r'\begin{table}[h!]' '\n'
            r'\centering' '\n'
            r'\small' '\n'
            r'\begin{tabular}{crrrrrr}' '\n'
            r'\toprule' '\n'
            r'& \multicolumn{3}{c}{RMS error in $f$ (Hz)} & \multicolumn{3}{c}{$\gamma$ bias (s$^{-1}$)} \\' '\n'
            r'\cmidrule(lr){2-4} \cmidrule(lr){5-7}' '\n'
            r'SNR & Fourier & Hankel & Opt.\ DMD & Fourier & Hankel & Opt.\ DMD \\' '\n'
            r'\midrule' '\n'
            f'{rows}\n'
            r'\bottomrule' '\n'
            r'\end{tabular}' '\n'
            f'\\caption{{{caption}}}\\label{{{label}}}\n'
            r'\end{table}' '\n')
