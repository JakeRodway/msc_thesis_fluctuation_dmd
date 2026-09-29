# msc_thesis_fluctuation_dmd

Analysis code for my MSc thesis on spatially resolved mode extraction from a multichannel fluctuation diagnostic on a magnetic confinement fusion device, using optimised Dynamic Mode Decomposition. The work was carried out at Imperial College London.

The code applies sliding-window optimised DMD (BOP-DMD, run without bagging) to the multichannel fluctuation diagnostic. It tracks a coherent high-frequency mode through a discharge, recovering its frequency, growth rate and spatial structure along the detector array. It also includes the tests used to check that the extracted structure is physical and not an artefact of the method.

## Contents

```
src/
  config.py             analysis settings for the main discharge
  math_helpers.py       data loading, STFT ridge tracking, seeded BOP-DMD fitting, rank selection
  dmd_tracking.py       sliding-window fit and mode tracking across the discharge
  spatial_math.py       phase alignment, standing-wave fitting, spatial persistence
  negative_controls.py  off-band, channel-envelope and channel-permutation controls
  validation.py         synthetic mode injection into real measured turbulence (BOP-DMD vs Hankel DMD vs Fourier)
  diag_lib.py 		     loads the internal analysis library (name set in local_settings.py)
notebooks/
  01_spectrogram_and_window_sizing.ipynb
  02_dmd_tracking_and_mode_structure.ipynb
  03_negative_controls.ipynb
  04_method_validation.ipynb
  05_contrasting_discharge.ipynb   a second discharge with travelling-wave behaviour
out/                    cached fit results, so the slow fits (up to ~40 min) need not be rerun
```

## Dependencies

Public Python packages:

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`, `pandas`
- [`pydmd`](https://github.com/PyDMD/PyDMD) (developed with 2025.8.1)
- [`optht`](https://github.com/erichson/optht) (Gavish–Donoho optimal hard threshold for rank selection)

### Internal diagnostic analysis library (not included)

This code was written to run alongside an internal analysis library maintained by the diagnostic group. That library handles access to the experimental data archive, detector calibration and geometry, and pre-computed spectrograms. It is not publicly available, so it cannot be distributed here.

The parts of this code that depend on it are:

| Used in | From the internal library | Purpose |
|---|---|---|
| `math_helpers.py` | `get.ProcessedData` (`get_interval`, `spectrogram`, `_channel_spacing`) | raw diagnostic signals, reference spectrogram, channel spacing |
| `math_helpers.py` | `query_programs.get_archive_byname` | heating power trace for the discharge |
| `validation.py` | `process.get_fcal`, `process.bandpass` | frequency-dependent gain calibration and band-pass filtering |

Without this library and access to the data archive, these calls will fail. The analysis itself (the DMD fitting, mode tracking, spatial analysis and controls) uses only the public packages above. It can be adapted to other multichannel data by replacing `load_data`, `fetch_window` and `get_reference_spectrogram` in `math_helpers.py` with your own loaders.

For users with access to the internal library: the notebooks add the parent directory to `sys.path`, so placing this folder inside a checkout of that library's repository is enough for the imports to resolve.

The library's package name and the discharge IDs are not included in this repository. To run the code with data access, create `src/local_settings.py` containing:

    LIB_NAME = "<library package name>"
    PID = "<Discharge 1 ID>"
    PID_B = "<Discharge 2 ID>"
