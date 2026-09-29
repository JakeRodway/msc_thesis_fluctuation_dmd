"""Settings for the fluctuation-diagnostic DMD analysis, with the values below chosen for Discharge 1, detector 1."""

import numpy as np
import os

# Discharge IDs are kept out of the repository. Provide them in src/local_settings.py
# (not tracked by git) as PID = '<Discharge 1 ID>' and PID_B = '<Discharge 2 ID>'.
try:
    from .local_settings import PID, PID_B
except ImportError:
    PID, PID_B = None, None
    
DET = 1 # Detector number, can be 1 or 2
CH_REF = 25 # Channel chosen for raw STFT calculations
EVAL_TREE_CH = 25 # The only channel the database holds pre-computed spectrogram data for

SEARCH_BAND = (190e3, 300e3) # Hz, band used for both the ridge tracker and candidate selection

T_MIN_BUFFER = 0.4 # s after t_on, adapt for time taken for mode to become clear 
T_MAX_BUFFER = 0.0 # s after t_off, adapt for time_taken for mode to disappear

STFT_MAX_DF = 40e3 # Hz/s, allowed frequency drift for Viterbi ridge

MODE_BAND_MARGIN = 10e3 # Hz, +/- around the seeded target for per-window candidate

WIN_PRIMARY = 0.5e-3 # s, primary DMD window length for sliding window analysis

WIN_SENSITIVITY = [0.25e-3, 0.5e-3, 0.75e-3, 1.0e-3, 
                   1.5e-3, 2.0e-3, 3.0e-3, 5.0e-3, 
                   7.0e-3, 10.0e-3, 15.0e-3, 20.0e-3] # s, DMD window lengths for sensitivity check

MARGIN = 0.15e-3 # s, extra data fetched each side of a window (cut later) for artificial edge effects

N_WINDOWS = 175 # number of candidate windows evenly spaced across the discharge

MIN_STRIDE = 15e-3 # s, minimum time step between sliding windows

RANK_CAP = 16 # sets a maximum rank for the DMD analysis

REPRESENTATIVE_T = [1.0, 4.0, 7.0] # s, target times for eval spectrum/rank/robustness plots

CONTROL_BAND = (330e3, 430e3) # Hz, negative control band for comparison

SEED = 0 # random seed for reproducibility
RNG = np.random.default_rng(SEED) # random number generator

TOP_K = 5 # number of candidates kept per window

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
os.makedirs(OUT, exist_ok=True)

# --- Method validation (synthetic mode of known f and gamma injected into real turbulence) ---

VAL_INTERVAL = [5.0, 5.02] # s, stretch of the same shot used as host turbulence

VAL_BAND = (325e3, 375e3) # Hz, band-pass applied to the host data, chosen to be free of coherent modes

VAL_F_INJ = 350e3 # Hz, injected frequency, sits in the middle of VAL_BAND

VAL_GAMMA_INJ = 900.0 # 1/s injected decay rate, so the correct answer is gamma = -900

VAL_SNRS = np.linspace(0.3, 2.0, 6) # injected std / host std, swept over this range

VAL_LOBE_CENTRES = [8, 16, 24] # channel, antinode positions of the injected standing pattern
VAL_LOBE_WIDTHS = [3, 3, 3] # channels, Gaussian width of each lobe
VAL_LOBE_SIGNS = [1, -1, 1] # alternating signs, giving 3 antinodes and 2 nodes

VAL_HANKEL_DELAYS = 20 # number of time-delay copies stacked for the Hankel DMD comparison
