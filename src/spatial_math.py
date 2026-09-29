"""
Spatial analysis and structural characterisation of extracted DMD modes.

This module processes the complex eigenvectors returned by the DMD pipeline to 
extract, quantify and verify the physical wave properties. It includes:
- Phase alignment, dewrapping and stable reference channel selection
- Principal Component Analysis (PCA) to fit complex vectors to physical standing-wave profiles
- Temporal averaging of spatial profiles
- Spatial continuity tracking across the discharge to check structure persistence
"""

import numpy as np

def _dewrap_local(raw):
    """
    Removes artificial +/- pi jumps in an array of wrapped phase values.

    Unlike np.unwrap, the function evaluates each point against its immediate 
    neighbours, fixing the local visual flickering for noisy standing waves.

    Parameters:
    - raw: array-like, the wrapped phase values to be dewrapped

    Returns:
    - out: array-like, the dewrapped phase values

    """
    out = raw.copy()
    for i in range(1, len(raw) - 1):
        a, b, c = raw[i - 1], raw[i], raw[i + 1] # grab the neighbours 
        candidates = [b, b + 2 * np.pi, b - 2 * np.pi] # three possible unwrapped values for the middle point
        costs = [abs(cand - a) + abs(c - cand) for cand in candidates] # compute sum of jumps from left and right neighbours
        out[i] = candidates[int(np.argmin(costs))] # choose the candidate with the smallest total jump
    return out


def _real_mode_fit(phi):
    """
    Fits a complex spatial mode to a real standing wave profile using PCA.

    Analyses the complex vectors as a 2D point cloud to find the dominant global phase (line of best fit) 
    and projects the data onto it to reveal the physical amplitudes and node crossings (where signs flip).

    R^2 measures how well the mode is represented by a single standing wave, with 1.0 being perfect and 
    0.5 representing pure noise or a travelling wave.

    Parameters:
    - phi: complex spatial mode to be fitted

    Returns:
    - theta0: global phase of the standing wave
    - r2: fraction of variance explained by the principal component (0 to 1)
    - A: real-valued standing wave amplitude profile

    """
    re, im = phi.real, phi.imag
    C = np.array([[np.sum(re * re), np.sum(re * im)], # covariance matrix of real and imaginary parts
                  [np.sum(re * im), np.sum(im * im)]])
    eigvals, eigvecs = np.linalg.eigh(C) # computes evals/evecs, sorted in ascending order
    lam_max, v = eigvals[-1], eigvecs[:, -1] # chooses principal component, representing the length of the point cloud
    theta0 = np.arctan2(v[1], v[0]) # global phase of standing wave
    total = eigvals.sum() # total energy in the point cloud i.e. across all 32 channels
    r2 = lam_max / total if total > 0 else np.nan # fraction of variance explained by the principal component
    A = re * np.cos(theta0) + im * np.sin(theta0) # projection of the complex signal onto the principal component, i.e. the real-valued standing wave amplitude
    return theta0, r2, A


def choose_phase_reference_channel(phi_mean, flat_thresh=0.3):
    """
    Find the most "stable" channel to use as a global phase reference. Avoids channels 
    near mode crossings by selecting the middle of the longest "flat" run of phases.

    Parameters:
    - phi_mean: complex spatial mode to be analysed
    - flat_thresh: threshold for the absolute value of the wrapped phase difference between consecutive channels,
      below which the channels are considered part of a flat region (default 0.3 rad)

    Returns:
    - ref_idx: index of the chosen reference channel
    - run: list of indices of the channels in the longest flat run

    """
    ang = np.angle(phi_mean) # raw wrapped phase values
    d = np.abs(np.angle(np.exp(1j * np.diff(ang)))) # absolute value of the wrapped phase difference between consecutive channels
    flat = d < flat_thresh # boolean array indicating which channels are part of a flat region, so false values = crossing a node

    # Find the longest run of True values in flat, and return the index of the middle channel in that run
    best_len, best_start, cur_len, cur_start = 0, 0, 0, 0
    for i, v in enumerate(flat):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    run = list(range(best_start, best_start + best_len + 1))
    ref_idx = run[len(run) // 2]
    return ref_idx, run


def spatial_mode_analysis(ctx, track):
    """
    Computes the averaged spatial profile over the plateau region of the discharge.

    Parameters:
    - ctx: context dictionary containing experimental parameters
    - track: dictionary containing the tracked mode data

    Returns:
    - spatial: dictionary containing the averaged spatial mode and related quantities

    """
    dx = ctx['dx_m'] * 100   # cm
    n_ch = ctx['n_ch']
    ch = np.arange(1, n_ch + 1)
    x_cm = (ch - 1) * dx

    # Find stable middle third of tracked data
    t_lo, t_hi = np.percentile(track['t'], [33, 67])
    stable = (track['t'] >= t_lo) & (track['t'] <= t_hi)
    
    # Phase alignment
    phi_stack = track['phi'][stable]
    ref_ch = n_ch // 2
    phi_aligned = phi_stack * np.exp(-1j * np.angle(phi_stack[:, ref_ch]))[:, None]
    phi_mean = phi_aligned.mean(axis=0)
    phi_std = np.abs(phi_aligned).std(axis=0)

    # Apply PCA real mode fit
    theta0_avg, r2_real_avg, A_avg = _real_mode_fit(phi_mean)
    
    # Identify nodes (sign flips in real amplitude)
    sign_changes = np.where(np.diff(np.sign(A_avg)) != 0)[0]
    phase_display = _dewrap_local(np.angle(phi_mean))

    return dict(ch=ch, dx=dx, x_cm=x_cm, phi_mean=phi_mean, phi_std=phi_std, 
                theta0_avg=theta0_avg, r2_real_avg=r2_real_avg, A_avg=A_avg, 
                sign_changes=sign_changes, phase_display=phase_display)


def spatial_mode_evolution(ctx, track, spatial):
    """
    Tracks spatial continuity and overlap against the averaged reference over time.

    Parameters:
    - ctx: context dictionary containing experimental parameters
    - track: dictionary containing the tracked mode data
    - spatial: dictionary containing the averaged spatial mode and related quantities

    Returns:
    - evolution: dictionary containing time-resolved spatial mode analysis

    """
    n_ch = ctx['n_ch']
    ch = np.arange(1, n_ch + 1)
    phi = track['phi']
    t = track['t']

    norms = np.linalg.norm(phi, axis=1)
    phi_n = phi / norms[:, None]

    ref_idx, _ = choose_phase_reference_channel(spatial['phi_mean'])
    phi_aligned = phi_n * np.exp(-1j * np.angle(phi_n[:, ref_idx]))[:, None]

    # Calculate overlaps
    overlap_adj = np.abs(np.sum(np.conj(phi_n[:-1]) * phi_n[1:], axis=1)) # overlap between adjacent time windows
    phi_ref_unit = spatial['phi_mean'] / np.linalg.norm(spatial['phi_mean'])
    overlap_ref = np.abs(phi_n @ np.conj(phi_ref_unit)) # overlap with the averaged spatial mode

    # Calculate participation and per-window R2
    n_eff = (np.sum(np.abs(phi_n) ** 2, axis=1)) ** 2 / np.sum(np.abs(phi_n) ** 4, axis=1)
    r2_real = np.array([_real_mode_fit(phi_n[i])[1] for i in range(len(t))])

    # Regimes
    plateau = (track['f'] < 230e3) & (t >= 2.5) & (t <= 6.0)
    excursion = track['f'] >= 235e3
    regimes = {'plateau': (plateau, 'tab:blue'), 'excursion': (excursion, 'tab:red')}

    return dict(t=t, ch=ch, phi_n=phi_n, phi_aligned=phi_aligned, 
                overlap_adj=overlap_adj, overlap_ref=overlap_ref, 
                n_eff=n_eff, r2_real=r2_real, regimes=regimes, ref_idx=ref_idx)
