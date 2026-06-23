# Data formats

Input layout, the participant manifest schema, and the full output column
dictionaries.

---

## 1. Input — Hexoskin export

Default root: `source_data/Hexoskin/` (override with `--hexoskin-path`).
One directory per recorded session, named by the Hexoskin "range" ID:

```
source_data/Hexoskin/
└── range_<hexoskin_id>-datatype_4096-1006/
    ├── record_<n>/
    │   ├── ECG_I.wav                    # single-lead ECG, 256 Hz
    │   ├── respiration_thoracic.wav     # thoracic respiration, 128 Hz
    │   └── respiration_abdominal.wav    # abdominal respiration, 128 Hz
    └── RR_interval.csv                  # device RR-intervals; seconds in the last column
```

How `signal_loader.HexLoader` resolves and reads these:

| Step | Logic |
|------|-------|
| Folder | `participant_id` + `session_num` → manifest `hexoskin_sessions.session_{n}` → `range_<id>-datatype_4096-1006/` |
| Record dir | first `record_*` subfolder (sorted) |
| ECG | `record_*/ECG_I.wav` via `soundfile`; flattened to 1-D float, fs from WAV header (fallback 256) |
| RSP | `record_*/respiration_{thoracic,abdominal}.wav`; fallback fs 128 |
| RRI | session-level `RR_interval.csv`, last column (seconds) → ms, kept to the physiological 300–2000 ms band |

Sampling rates are verified from the WAV headers: **ECG 256 Hz**, **RSP 128 Hz**
(constants `ECG_FS`, `RSP_FS` in `signal_loader.py`).

---

## 2. Input — participant manifest

`data/participants_anonymized.json`:

```json
{
  "metadata": { "total_participants": 28, "anonymized": true, "...": "..." },
  "participants": [
    {
      "participant_id": 1,
      "name": "P1",
      "demographics": { "age": 23, "musical_experience_years": 19, "hexoskin_vest_size": "S" },
      "session_info": { "session_date": "2025-05-07" },
      "sensor_data": {
        "hexoskin_sessions": { "session_1": "3324350", "session_2": "3324351", "session_3": "3324354" },
        "movella_sessions":  { "session_1": null,      "session_2": "112417", "session_3": "112644" },
        "video_recordings":  { "session_1": "1194",    "session_2": "1194",   "session_3": "1194" }
      },
      "data_completeness": { "has_hexoskin_data": true, "complete_sessions": 2, "...": "..." }
    }
  ]
}
```

Only `participant_id` and `sensor_data.hexoskin_sessions.session_{1,2,3}` are
required by the pipeline; everything else is descriptive. A session value of
`null` / `"None"` / missing means "no Hexoskin recording for that piece" and is
skipped silently.

`data/Participants_DCS_anonymized.csv` is the flat roster the manifest was built
from (columns: `ID, Name, date, Movella_1..3, Hexoskin_1..3, Video_1..3, Age,
Exp, Size`). It is committed for reference; the pipeline reads the JSON.

> The Movella DOT motion sensors and video are **not** processed by this
> pipeline — only ECG and respiration. The IDs are retained in the manifest for
> provenance.

---

## 3. Output column dictionaries

All outputs go to `results/neurokit_features/` (override with `--out-dir`).
Row counts below are for the full 2Inspire cohort (29 participants, 3 pieces).

### `ecg_hrv_piece.csv` — one row / participant × piece (74 rows)

Keys + QC, then the full NeuroKit HRV suite and RSA.

| Group | Columns |
|-------|---------|
| Keys | `participant_id`, `session_idx`, `ecg_fs` |
| Coverage / QC | `n_peaks`, `duration_s`, `ECG_Rate_Mean`, `pct_missing`, `n_rate_outliers`, `pct_low_sqi`, `ecg_quality_mean` |
| HRV time | `HRV_MeanNN`, `HRV_SDNN`, `HRV_SDANN{1,2,5}`, `HRV_SDNNI{1,2,5}`, `HRV_RMSSD`, `HRV_SDSD`, `HRV_CVNN`, `HRV_CVSD`, `HRV_MedianNN`, `HRV_MadNN`, `HRV_MCVNN`, `HRV_IQRNN`, `HRV_Prc{20,80}NN`, `HRV_pNN{50,20}`, `HRV_MinNN`, `HRV_MaxNN`, `HRV_HTI`, `HRV_TINN` |
| HRV frequency | `HRV_ULF`, `HRV_VLF`, `HRV_LF`, `HRV_HF`, `HRV_VHF`, `HRV_TP`, `HRV_LFHF`, `HRV_LFn`, `HRV_HFn`, `HRV_LnHF` |
| HRV nonlinear | `HRV_SD1`, `HRV_SD2`, `HRV_SD1SD2`, `HRV_S`, `HRV_CSI`, `HRV_CVI`, `HRV_CSI_Modified`, Poincaré asymmetry (`HRV_GI/SI/AI/PI/C1d/...`), `HRV_DFA_alpha1/2`, multifractal `HRV_MFDFA_alpha{1,2}_*`, entropy (`HRV_ApEn`, `HRV_SampEn`, `HRV_ShanEn`, `HRV_FuzzyEn`, `HRV_MSEn`, `HRV_CMSEn`, `HRV_RCMSEn`), complexity (`HRV_CD`, `HRV_HFD`, `HRV_KFD`, `HRV_LZC`), `HRV_Symbolic_*` |
| RSA | `RSA_P2T_Mean`, `RSA_P2T_Mean_log`, `RSA_P2T_SD`, `RSA_P2T_NoRSA`, `RSA_PorgesBohrer`, `RSA_Gates_Mean`, `RSA_Gates_Mean_log`, `RSA_Gates_SD` |

(Columns that NeuroKit cannot compute for a given piece — e.g. `DFA_alpha2` for
short pieces — are emitted as `NaN`.)

### `rsp_features_piece.csv` — one row / participant × piece × channel (148 rows)

| Group | Columns |
|-------|---------|
| Keys | `participant_id`, `session_idx`, `rsp_fs`, `channel` (`thoracic`/`abdominal`) |
| Rate / amplitude / RVT | `RSP_Rate_Mean`, `RSP_Rate_SD`, `RSP_Amp_Mean`, `RSP_Amp_SD`, `RSP_RVT_Mean`, `RSP_RVT_SD` |
| Slope / phase ratios | `RSP_Slope_Inhale_Ratio`, `RSP_Slope_Exhale_Ratio`, `RSP_Slope_Hold_Ratio`, `RSP_Slope_Dominant`, `RSP_Inhale_Ratio`, `RSP_Exhale_Ratio` |
| RRV (resp-rate variability) | `RRV_MeanBB`, `RRV_SDBB`, `RRV_RMSSD`, `RRV_SDSD`, `RRV_LF`, `RRV_HF`, `RRV_LFHF`, `RRV_LFn`, `RRV_HFn`, `RRV_SD1`, `RRV_SD2`, `RRV_SD2SD1`, `RRV_DFA_alpha1/2`, `RRV_ApEn`, `RRV_SampEn`, `RRV_CVBB`, `RRV_CVSD`, `RRV_MedianBB`, `RRV_MadBB`, `RRV_MCVBB`, `RRV_VLF`, `RRV_MFDFA_alpha{1,2}_*` |
| Interval-related (`nk.rsp_intervalrelated`) | `NK_RAV_Mean`, `NK_RAV_SD`, `NK_RAV_RMSSD`, `NK_RAV_CVSD`, `NK_RSP_Phase_Duration_Inspiration`, `NK_RSP_Phase_Duration_Expiration`, `NK_RSP_Phase_Duration_Ratio` |
| QC | `pct_missing`, `pct_low_sqi`, `n_outliers` |

### `hrv_raw_vs_rri.csv` — one row / participant × piece (74 rows)

`participant_id`, `session_idx`, `n_rri`, then paired `rri_*` / `raw_*` columns
for `rmssd`, `sdnn`, `meannn`, `hf`, `lf`, `lfhf`, plus `delta_rmssd`,
`rri_mean_hr`, `raw_mean_hr`, `delta_mean_hr`.

### `ecg_continuous_1hz.csv` (long)

`participant_id`, `session_idx`, `feature` ∈ {`hr_bpm`, `hrv_rmssd`}, `t_pct`
(0–100, resampled to the % of-piece grid), `value`.

### `rsp_continuous_1hz.csv` (long)

`participant_id`, `session_idx`, `channel` (`thoracic`/`abdominal`), `feature` ∈
{`rsp_rate`, `rsp_amplitude`}, `t_pct`, `value`.

### `failures.md`

Markdown list of skipped (participant, piece) entries with reasons (missing WAV,
`ecg_process` failure, etc.).
