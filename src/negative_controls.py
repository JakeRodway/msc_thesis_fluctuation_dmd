"""
Negative controls and statistical validation for spatial mode structures.

This module provides control tests used to verify whether the extracted spatial structures are genuine physical 
phenomena rather than mathematical or instrumental artefacts. It includes:
- Off-band control fitting (applying identical DMD pipeline to a featureless background frequency band)
- Raw channel envelope analysis
- Channel permutation tests with scrambled channel order to demonstrate that the node/dip correspondence depends 
  on true detector adjacency
"""

import os
import time
import pickle
import numpy as np
from scipy.signal import welch

from .config import MIN_STRIDE, N_WINDOWS, WIN_PRIMARY, CONTROL_BAND, SEARCH_BAND, DET, OUT
from .math_helpers import (
    fetch_window, local_fourier_seed, rank_selection, 
    fit_bopdmd, eig_to_f_gamma_continuous, energy_fraction
)
from .spatial_math import _real_mode_fit, choose_phase_reference_channel


def fit_offband_windows(ctx, band=CONTROL_BAND, win=WIN_PRIMARY, n_windows=N_WINDOWS, checkpoint_path=None):
    """
    Runs the exact same BOP-DMD pipeline as run_sliding_optdmd (in dmd_tracking.py), but anchored to a featureless 
    background band.

    Parameters:
    - ctx: context dictionary containing experimental parameters
    - band: tuple specifying the frequency band for analysis
    - win: float specifying the window size for analysis
    - n_windows: int specifying the number of windows to analyze
    - checkpoint_path: optional string specifying a path to save/load intermediate results

    Returns:
    - rows: list of dictionaries containing the results for each time window (t, f, gamma, energy_frac, phi)

    """
    d, fs = ctx['d'], ctx['fs']
    target = 0.5 * (band[0] + band[1])
    t0, t1 = ctx['t_on'] + 0.05, ctx['t_off'] - 0.05
    stride = max(MIN_STRIDE, (t1 - t0) / n_windows)
    centres = np.arange(t0 + win, t1 - win, stride)
    
    rows = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as fh:
            return pickle.load(fh)

    for tc in centres:
        try:
            X, t_local = fetch_window(d, tc, win, fs)
            seed = local_fourier_seed(X, fs, target, hw_search=(band[1] - band[0]) / 2)
            rank, _ = rank_selection(X)
            m = fit_bopdmd(X, t_local, rank, seed)
            
            f_all, g_all = eig_to_f_gamma_continuous(m.eigs)
            amps, modes = np.asarray(m.amplitudes), np.asarray(m.modes)
            
            efrac = energy_fraction(amps)
            cand = np.where((f_all > band[0]) & (f_all < band[1]))[0]
            if cand.size == 0: continue
            j = cand[np.argmax(efrac[cand])]
            
            rows.append(dict(t=tc, f=f_all[j], gamma=g_all[j], energy_frac=efrac[j], phi=modes[:, j]))
        except Exception:
            continue

    if checkpoint_path:
        with open(checkpoint_path, 'wb') as fh:
            pickle.dump(rows, fh)
            
    return rows


def analyse_offband_control(ctx, rows):
    """
    Extracts the spatial profile and continuity for the off-band control data.

    Parameters:
    - ctx: context dictionary containing experimental parameters
    - rows: list of dictionaries containing the results for each time window (t, f, gamma, energy_frac, phi)

    Returns:
    - dict containing:
        - t: array of time centers for each window
        - phi_mean_ref: mean spatial profile across stable windows
        - overlap_ref: array of overlaps between each window's spatial profile and the mean reference profile
        - r2_real_ref: R^2 value for the mean reference profile fit

    """
    t_arr = np.array([r['t'] for r in rows])
    phi_arr = np.array([r['phi'] for r in rows])
    
    t_lo, t_hi = np.percentile(t_arr, [33, 67])
    stable = (t_arr >= t_lo) & (t_arr <= t_hi)
    
    norms = np.linalg.norm(phi_arr, axis=1)
    phi_n = phi_arr / norms[:, None]
    
    phi_stack = phi_n[stable]
    ref_ch = phi_stack.shape[1] // 2
    phi_aligned = phi_stack * np.exp(-1j * np.angle(phi_stack[:, ref_ch]))[:, None]
    phi_mean_ref = phi_aligned.mean(axis=0)
    
    theta0_ref, r2_real_ref, _ = _real_mode_fit(phi_mean_ref)
    
    phi_ref_unit = phi_mean_ref / np.linalg.norm(phi_mean_ref) if np.linalg.norm(phi_mean_ref) > 0 else phi_mean_ref
    overlap_ref = np.abs(phi_n @ np.conj(phi_ref_unit))
    
    return dict(t=t_arr, phi_mean_ref=phi_mean_ref, overlap_ref=overlap_ref, r2_real_ref=r2_real_ref)


def compute_channel_envelope(ctx, phi_mean_real, t_window=(3.0, 3.5)):
    """
    Computes RAW signal RMS (no DMD) to see if the amplitude shape is just the instrument's sensitivity profile.

    Parameters:
    - ctx: context dictionary containing experimental parameters
    - phi_mean_real: mean spatial profile across stable windows (not used in this function)
    - t_window: tuple specifying the time window for analysis

    Returns:
    - dict containing:
        - ch: array of channel indices
        - broadband_rms: array of RMS values across all frequencies for each channel
        - search_rms: array of RMS values within the search band for each channel
        - control_rms: array of RMS values within the control band for each channel

    """
    d, fs = ctx['d'], ctx['fs']
    raw, _ = d.get_interval(det=DET, interval=list(t_window))
    n_ch = raw.shape[0]
    
    broadband_rms = raw.std(axis=1)
    search_rms = np.zeros(n_ch)
    control_rms = np.zeros(n_ch)
    
    for i in range(n_ch):
        f, Pxx = welch(raw[i], fs=fs, nperseg=4096)
        df = f[1] - f[0]
        search_rms[i] = np.sqrt(np.sum(Pxx[(f >= SEARCH_BAND[0]) & (f <= SEARCH_BAND[1])]) * df)
        control_rms[i] = np.sqrt(np.sum(Pxx[(f >= CONTROL_BAND[0]) & (f <= CONTROL_BAND[1])]) * df)
        
    return dict(ch=np.arange(1, n_ch + 1), broadband_rms=broadband_rms, 
                search_rms=search_rms, control_rms=control_rms)


def channel_permutation_control(phi_stack, n_perms=500, seed=0):
    """
    Randomly shuffles channel order. R^2 should stay the same (math invariant), 
    but the physical node/dip correspondence should collapse.

    Parameters:
    - phi_stack: 2D array of spatial profiles (windows x channels)
    - n_perms: int specifying the number of permutations to perform
    - seed: int specifying the random seed for reproducibility

    Returns:
    - dict containing:
        - r2_baseline: R^2 fit quality for the true channel order
        - r2_perm: R^2 fit quality for each of the random shuffles (should exactly match baseline)
        - frac_baseline: fraction of nodes that successfully coincide with an amplitude dip in the true order
        - frac_perm: fraction of nodes coinciding with dips for each random shuffle
    """
    rng = np.random.default_rng(seed)
    n_ch = phi_stack.shape[1]
    ref_ch = n_ch // 2
    
    phi_aligned = phi_stack * np.exp(-1j * np.angle(phi_stack[:, ref_ch]))[:, None]
    phi_mean = phi_aligned.mean(axis=0)
    
    _, r2_baseline, A = _real_mode_fit(phi_mean)
    mag = np.abs(phi_mean)
    sign_changes = np.where(np.diff(np.sign(A)) != 0)[0]
    
    near_dip = sum(1 for sc in sign_changes if mag[sc] <= np.percentile(mag, 40) or mag[sc + 1] <= np.percentile(mag, 40))
    frac_baseline = near_dip / len(sign_changes) if len(sign_changes) else np.nan
    
    r2_perm = np.full(n_perms, np.nan)
    frac_perm = np.full(n_perms, np.nan)
    
    for p in range(n_perms):
        perm = rng.permutation(n_ch)
        phi_p = phi_mean[perm]
        _, r2_p, A_p = _real_mode_fit(phi_p)
        r2_perm[p] = r2_p
        
        mag_p = np.abs(phi_p)
        sc_p = np.where(np.diff(np.sign(A_p)) != 0)[0]
        if len(sc_p):
            nd_p = sum(1 for sc in sc_p if mag_p[sc] <= np.percentile(mag_p, 40) or mag_p[sc + 1] <= np.percentile(mag_p, 40))
            frac_perm[p] = nd_p / len(sc_p)
            
    return dict(r2_baseline=r2_baseline, r2_perm=r2_perm, frac_baseline=frac_baseline, frac_perm=frac_perm)