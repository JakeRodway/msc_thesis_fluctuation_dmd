"""
Sliding-window DMD and mode tracking algorithms.

This module contains the core analytical pipeline, applying BOP-DMD (without bagging) across the plasma 
discharge to track the time-evolution of a coherent mode. It includes:
- A sliding-window fitting loop anchored to the STFT reference ridge (Viterbi)
- Extraction of the evolution of the tracked mode parameters (frequency, growth rate, energy fraction and 
spatial eigenvectors) with time
- Single-window fitting for candidate selection walkthrough (see notebook 2)
"""

import os
import time
import pickle
import numpy as np

from .config import MIN_STRIDE, MODE_BAND_MARGIN, N_WINDOWS, WIN_PRIMARY, REPRESENTATIVE_T
from .math_helpers import (
    fetch_window, local_fourier_seed, rank_selection, 
    fit_bopdmd, eig_to_f_gamma_continuous, energy_fraction
)

def run_sliding_optdmd(ctx, ref, win=WIN_PRIMARY, n_windows=N_WINDOWS, 
                       checkpoint_path=None, checkpoint_every=20, blend=5e3):
    """
    Slides a rank-truncated BOPDMD across the discharge, using the STFT ridge as an anchor.

    Parameters:
    - ctx: context dictionary containing experimental parameters
    - ref: dictionary containing the STFT ridge reference data
    - win: window length for the sliding DMD [s]
    - n_windows: number of windows to fit across the discharge
    - checkpoint_path: optional path to save intermediate results for resuming
    - checkpoint_every: number of windows to process before saving a checkpoint
    - blend: frequency blending range [Hz] for candidate selection

    Returns:
    - rows: list of dictionaries containing the fit results for each window

    """
    d, fs = ctx['d'], ctx['fs']
    t0, t1 = ctx['t_on'] + 0.05, ctx['t_off'] - 0.05
    stride = max(MIN_STRIDE, (t1 - t0) / n_windows)
    centres_all = np.arange(t0 + win, t1 - win, stride)
    centres = centres_all[centres_all >= ref['ridge_t_valid'][0]]

    rows = []
    done_t = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as fh:
            rows = pickle.load(fh)
        done_t = {r['t'] for r in rows}

    prev_f = rows[-1]['f'][rows[-1]['sel_idx']] if (rows and rows[-1]['sel_idx'] is not None) else None

    for i, tc in enumerate(centres):
        if tc in done_t:
            continue
        try:
            X, t_local = fetch_window(d, tc, win, fs)
            anchor = float(np.interp(tc, ref['ridge_t_valid'], ref['ridge_f_valid']))
            target = float(np.clip(prev_f, anchor - blend, anchor + blend)) if prev_f is not None else anchor
            seed = local_fourier_seed(X, fs, target)
            rank, _ = rank_selection(X)

            m = fit_bopdmd(X, t_local, rank, seed)
            f_all, g_all = eig_to_f_gamma_continuous(m.eigs)
            amps, modes = np.asarray(m.amplitudes), np.asarray(m.modes)
            
            efrac = energy_fraction(amps)
            cand = np.where((f_all > 0) & (np.abs(f_all - target) <= MODE_BAND_MARGIN))[0]
            j = cand[np.argmax(efrac[cand])] if cand.size else None
            ef_band = float(efrac[j] / efrac[cand].sum()) if (j is not None and efrac[cand].sum() > 0) else np.nan

            rows.append(dict(t=tc, seed=seed, rank=rank, f=f_all, gamma=g_all,
                             amplitudes=amps, modes=modes, sel_idx=j, ef_band=ef_band))
            prev_f = f_all[j] if j is not None else None

        except Exception:
            continue
            
        if checkpoint_path and (i + 1) % checkpoint_every == 0:
            with open(checkpoint_path, 'wb') as fh:
                pickle.dump(rows, fh)

    if checkpoint_path:
        with open(checkpoint_path, 'wb') as fh:
            pickle.dump(rows, fh)
            
    return rows


def track_mode(rows):
    """
    Extracts the selected candidate poles from the sliding window fit into clean arrays.

    Parameters:
    - rows: list of dictionaries containing the fit results for each window

    Returns:
    - track: dictionary containing the tracked mode data (time, frequency, growth rate, energy fraction, and spatial modes)

    """
    t_arr, f_arr, g_arr, ef_arr, phi_list = [], [], [], [], []
    
    for r in rows:
        j = r['sel_idx']
        if j is None: continue
        t_arr.append(r['t'])
        f_arr.append(r['f'][j])
        g_arr.append(r['gamma'][j])
        ef_arr.append(energy_fraction(r['amplitudes'])[j])
        phi_list.append(r['modes'][:, j])

    return dict(
        t=np.array(t_arr), 
        f=np.array(f_arr), 
        gamma=np.array(g_arr), 
        energy_frac=np.array(ef_arr), 
        phi=np.array(phi_list)
    )


def fit_one_window(ctx, ref, tc, win=WIN_PRIMARY, mode_band_margin=MODE_BAND_MARGIN):
    """
    Fits a single window at time `tc` and selects its candidate. 

    Parameters:
    - ctx: context dictionary containing experimental parameters
    - ref: dictionary containing the STFT ridge reference data
    - tc: center time for the window [s]
    - win: window length for the DMD fit [s]
    - mode_band_margin: frequency margin for candidate selection [Hz]

    Returns:
    - result: dictionary containing the fit results for the window (time, seed, rank, frequency, growth rate, amplitudes, modes, and selected index)

    """
    d, fs = ctx['d'], ctx['fs']
    X, t_local = fetch_window(d, tc, win, fs)
    seed_stft = float(np.interp(tc, ref['ridge_t_valid'], ref['ridge_f_valid']))
    seed = local_fourier_seed(X, fs, seed_stft)
    rank, _ = rank_selection(X)
    
    m = fit_bopdmd(X, t_local, rank, seed)
    f_all, g_all = eig_to_f_gamma_continuous(m.eigs)
    amps = np.asarray(m.amplitudes)
    modes = np.asarray(m.modes)
    
    efrac = energy_fraction(amps)
    cand = np.where((f_all > 0) & (np.abs(f_all - seed_stft) <= mode_band_margin))[0]
    j = cand[np.argmax(efrac[cand])] if cand.size else None
    
    return dict(t=tc, seed=seed, rank=rank, f=f_all, gamma=g_all,
                amplitudes=amps, modes=modes, sel_idx=j)