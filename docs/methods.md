# Methods

How the pipeline turns raw Hexoskin signals into the feature tables. All signal
processing is delegated to [NeuroKit2](https://neuropsychology.github.io/NeuroKit/)
0.2.x; the modules here are thin, reproducible wrappers around it.

---

## Pipeline overview

For each (participant, piece), `run_all.py`:

1. Loads raw `ECG_I.wav` → `process_ecg` → whole-piece HRV + QC + light continuous trajectories.
2. Loads the device `RR_interval.csv` → an HRV cross-check (raw ECG vs device RRI).
3. Loads thoracic + abdominal RSP WAVs → `process_rsp` per channel → whole-piece RSP features + continuous trajectories.
4. Resamples every continuous series onto a shared 0–100 % "% of piece" grid (101 points) via `np.interp`, so trajectories can be overlaid/averaged across participants with different piece lengths.

A `numpy_compat.apply_numpy_compat()` shim runs first; it restores `np.trapz`
(removed in NumPy 2.0) so the pinned NeuroKit2 0.2.x runs on NumPy 2.x.

---

## ECG → HRV

`process_ecg.process_ecg(signal, sampling_rate=256)` wraps `nk.ecg_process` and
returns the per-sample frame (clean signal, R-peaks, instantaneous rate, quality).
Signals shorter than 5 s are rejected.

`features_ecg.piece_hrv` then computes the whole-piece HRV from the detected
R-peaks, with guards so short pieces don't produce garbage:

| Domain | Function | Guard |
|--------|----------|-------|
| Time | `nk.hrv_time` | ≥ 2 peaks (`MIN_PEAKS_TIME`) |
| Frequency | `nk.hrv_frequency(psd_method="welch", normalize=False)` | ≥ 2 peaks **and** duration ≥ 30 s (`MIN_DURATION_FREQ_S`) |
| Nonlinear | `nk.hrv_nonlinear` | ≥ 10 peaks (`MIN_PEAKS_NONLINEAR`) |

**Quality metrics** (`_quality_metrics`): `n_rate_outliers` counts samples with
instantaneous HR < 30 or > 200 bpm (`HR_OUTLIER_LOW/HIGH`); `pct_low_sqi` is the
fraction of samples with quality < 0.8 (`SQI_THRESHOLD`); plus `ecg_quality_mean`
and `pct_missing`.

### HRV from device RR-intervals (`hrv_from_rri`)

The Hexoskin device reports its own beat detection in `RR_interval.csv`. To run
the same NeuroKit HRV suite on those intervals without resampling a full signal,
intervals (ms) are placed on a **virtual 1000 Hz timeline** so one sample == 1 ms:

```python
peaks = np.cumsum(np.round(rri_ms).astype(int))   # peak indices at sr = 1000 Hz
compute_hrv_full(peaks, sr=1000, duration_s=peaks[-1] / 1000)
```

Requires ≥ 30 intervals. Results populate the `rri_*` columns of
`hrv_raw_vs_rri.csv`; the `raw_*` columns are the matching raw-ECG values.

### Respiratory sinus arrhythmia (RSA)

`compute_rsa` resamples the **abdominal** respiration to the ECG sample grid,
reprocesses it at the ECG rate (so R-peaks and breaths share a timeline), and
calls `nk.hrv_rsa` → Porges–Bohrer, peak-to-trough (P2T), and Gates estimators.

### Continuous ECG trajectories (`continuous_ecg`)

- `hr_bpm`: instantaneous HR binned to 1 Hz (`_bin_to_1hz`).
- `hrv_rmssd`: RMSSD over a sliding **30 s window, 5 s step** (`ROLL_WINDOW_S`,
  `ROLL_STEP_S` — shorter than MOXIE's 300 s because pieces are short), requiring
  ≥ 3 beats per window.

Each series is returned as `(t_sec, value)` and then mapped to the 0–100 % grid.

---

## Respiration → RSP features

`process_rsp.process_rsp(signal, channel, sampling_rate=128)` wraps
`nk.rsp_process(method="khodadad2018")` and **prefixes every output column with
`respiration_<channel>_`** so thoracic and abdominal can be processed and stored
side by side. Signals shorter than 10 s are rejected.

`features_rsp.piece_rsp` computes, per channel:

- **Rate / amplitude / RVT** means & SDs.
- **Slope & phase ratios** — inhale/exhale/hold proportions from the gradient of
  the cleaned signal and from NeuroKit's inspiration/expiration phase labels.
- **Respiratory-rate variability** via `nk.rsp_rrv` (`RRV_*`).
- **Interval-related features** via `nk.rsp_intervalrelated` (`NK_*`).
- **QC**: `pct_missing`, `pct_low_sqi`, `n_outliers` (RMSD-based).

`continuous_rsp` yields 1 Hz `rsp_rate` and `rsp_amplitude` per channel, mapped
to the same 0–100 % grid.

---

## ⚠️ Raw-ECG vs device-RRI — read before trusting HRV

Hexoskin's textile single-lead ECG is noisy. `nk.ecg_process` peak detection
occasionally inserts or misses beats. This barely moves **mean HR** (median |Δ|
≈ 0.06 bpm between raw-ECG and device-RRI) but **explodes successive-difference
HRV metrics** (RMSSD, pNN50, SD1, HF). On ~1/3 of 2Inspire sessions the raw-ECG
RMSSD is 2–10× the device value (e.g. 400–1000 ms vs 100–500 ms).

Practical guidance:

- For beat-to-beat HRV, **prefer the device-RRI path** (`hrv_raw_vs_rri.csv`,
  `rri_*`).
- Trust raw-ECG HRV only where `delta_rmssd` is small **and** `ecg_quality_mean`
  is high.
- Frequency/time means and trends are robust; successive-difference metrics are
  the fragile ones.
- Much of the apparent piece-3 spread in the HRV box plots is these noisy
  sessions, not a real piece effect.

---

## Reproducing the bundled outputs

The committed `results/neurokit_features/` and `plots/neurokit/` were produced by:

```bash
python neurokit_pipeline/run_all.py --participant all --session all
python neurokit_pipeline/plots.py
```

against the full 2Inspire Hexoskin export (not distributed). NeuroKit may emit
benign warnings for short pieces (e.g. "DFA_alpha2 … will not be calculated");
those features are written as `NaN`.
