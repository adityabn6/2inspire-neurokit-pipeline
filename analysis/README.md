# `analysis/` — clean 2Inspire music↔physiology layer

Correct, reproducible per-piece music↔physiology analysis for 2Inspire. See
`reports/hypotheses_music_physiology.md` (what to ask, + preliminary results).

Each script writes one tidy table to `results/analysis/`, **one row per
participant × piece**, with piece order taken from the manifest (not folder rank).

| Script | Output | What it does |
|---|---|---|
| `build_features.py` | `features_by_piece.csv` | Device-RR HRV with NeuroKit2 Kubios artifact correction + QC flag (`hrv_qc_ok`); merges RSA + respiration from `results/neurokit_features/`. |
| `audio_features.py` | `audio_features_by_piece.csv` | Per-piece chroma (CQT) key/mode/entropy/stability/flux + tempo, RMS, spectral centroid, onset-rate, HPSS harmonic ratio. |
| `coupling.py` | `coupling_by_piece.csv` | Respiration↔cardiac coupling (HF coherence, xcorr lag) — sync-robust (same device). |
| `motion_features.py` | `motion_by_piece.csv` | Hand-motion energy from the Movella DOT sensors (acc-magnitude SD) — motor-confound covariate. |
| `music_segments.py` | `music_segments_{long,wide}.csv` | First-pass music start/stop per clip vs Hexoskin duration (template for the manual timestamp annotations). |
| `remap_audio.py` | `audio_features_mapped.csv`, `audio_piece_map.csv` | Duration-based clip→Hexoskin-piece correction (fixes P18; excludes P1's single-clip case). |

`audio_features.py --segments music_segments_long.csv` trims each clip to its
`[music_start_s, music_stop_s]` window before feature extraction (drop in the
manual-timestamp CSV here when it arrives).
| `stats_models.py` | `stats_results.csv` | Within-subject stats: paired Wilcoxon + permutation (condition contrast), Friedman (across pieces), MixedLM/OLS (music→physiology), BH-FDR, effect sizes. |
| `plots.py` | `plots/analysis/*.png` | Condition contrast, across-piece, coupling full-vs-matched, music-by-condition, key-structure figures. |

`build_features.py` and `coupling.py` take `--window-s N` to trim each piece to
its first N seconds for **duration-matched** contrasts (piece 3 is much longer
than the sight-reading pieces). See `reports/results_summary.md`.

## Run

```bash
HEX=/home/adityabn/GoogleDrive/Research/Projects/2Inspire/source_data/Hexoskin
# 1. physiology (device RR + RSA/resp from the committed neurokit_features)
python analysis/build_features.py --participants all --hexoskin-path "$HEX"
# 2. music features  (unzip 2inspire_audio.zip somewhere first)
python analysis/audio_features.py --audio-dir /path/to/audio
# 3. respiration-cardiac coupling
python analysis/coupling.py --participants all --hexoskin-path "$HEX"
# 4. stats (merges the three tables)
python analysis/stats_models.py
```

## Key methodological choices (and why)

- **Piece index from the manifest** — folder-rank labelling silently swaps P18's
  pieces 1↔3.
- **Device RR + Kubios artifact correction + QC gate** — raw device RR is
  corrupted on ~12% of sessions (e.g. P2, RMSSD≈400 ms); these are flagged out,
  not averaged in.
- **LF/HF flagged low-confidence** on <2 min pieces; primary readouts are mean HR,
  RMSSD, RSA.
- **Music↔physiology only at piece level** — sub-second audio↔physiology sync is
  unrecoverable, so no time-resolved correlation.
- **Music→physiology estimated on piece 3 (OLS)** — pieces 1–2 are the *same*
  standardized score, so between-person music variance only exists in the
  self-selected piece; a participant random effect there is degenerate (1 obs/
  group), so OLS, not MixedLM.

Requires the `neurokit_pipeline/` package (same repo) and the deps in
`requirements.txt` (adds `librosa`, `statsmodels`).
