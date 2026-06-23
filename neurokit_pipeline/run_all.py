"""
run_all.py
-----------
Drive the 2Inspire NeuroKit ECG + RSP pipeline over every participant x piece.

Small dataset (<= 84 sessions) -> a plain local loop, no cluster. For each
participant/session it:

  1. loads raw ECG_I.wav -> nk.ecg_process -> whole-piece HRV + QC, and
     light continuous trajectories (instantaneous HR, rolling RMSSD);
  2. loads the device RR_interval.csv -> HRV cross-check (raw vs RRI);
  3. loads thoracic + abdominal RSP wavs -> nk.rsp_process per channel ->
     whole-piece RSP features + continuous RSP_Rate / RSP_Amplitude;
  4. resamples every continuous series onto a common 0-100 % of-piece grid so
     trajectories can be overlaid and averaged across participants.

Outputs (to --out-dir, default results/neurokit_features):
    ecg_hrv_piece.csv        ecg_continuous_1hz.csv
    rsp_features_piece.csv   rsp_continuous_1hz.csv
    hrv_raw_vs_rri.csv       failures.md

Usage
-----
    python code/neurokit_pipeline/run_all.py \
        --participant all --session all
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Run-by-absolute-path puts this script's dir on sys.path[0]; import siblings.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from numpy_compat import apply_numpy_compat  # noqa: E402
from signal_loader import HexLoader, ECG_FS, RSP_FS  # noqa: E402
import process_ecg as pe  # noqa: E402
import process_rsp as pr  # noqa: E402
import features_ecg as fe  # noqa: E402
import features_rsp as fr  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("run_all")

PROJECT_ROOT = _HERE.parent  # repo root (modules live in neurokit_pipeline/)
DEFAULT_HEX = PROJECT_ROOT / "source_data" / "Hexoskin"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "participants_anonymized.json"
DEFAULT_OUT = PROJECT_ROOT / "results" / "neurokit_features"

PCT_GRID = np.arange(0, 101, dtype=float)  # 0..100 % of piece


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def parse_spec(spec: str, lo: int, hi: int) -> list[int]:
    """Expand 'all' | '1-5' | '1,3,5' into a sorted list of ints."""
    if spec is None or spec.strip().lower() == "all":
        return list(range(lo, hi + 1))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(i for i in out if lo <= i <= hi)


def normalize_traj(t_sec: np.ndarray, value: np.ndarray) -> np.ndarray | None:
    """Resample a (t_sec, value) series onto the 0-100 % grid via np.interp."""
    if t_sec is None or len(t_sec) < 2:
        return None
    mask = np.isfinite(t_sec) & np.isfinite(value)
    if mask.sum() < 2:
        return None
    t_sec, value = t_sec[mask], value[mask]
    t_max = t_sec.max()
    if t_max <= 0:
        return None
    t_pct = 100.0 * (t_sec - t_sec.min()) / (t_max - t_sec.min()) if t_max > t_sec.min() else None
    if t_pct is None:
        return None
    return np.interp(PCT_GRID, t_pct, value)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    apply_numpy_compat()

    ap = argparse.ArgumentParser(description="2Inspire NeuroKit ECG+RSP pipeline.")
    ap.add_argument("--hexoskin-path", default=str(DEFAULT_HEX))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--participant", default="all")
    ap.add_argument("--session", default="all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    loader = HexLoader(args.hexoskin_path, args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    participants = parse_spec(args.participant, 1, 29)
    sessions = parse_spec(args.session, 1, 3)

    ecg_rows, rsp_rows, xcheck_rows = [], [], []
    ecg_cont, rsp_cont = [], []
    failures: list[str] = []

    for pid in participants:
        for sidx in sessions:
            tag = f"P{pid} S{sidx}"
            folder = loader.session_folder(pid, sidx)
            if folder is None:
                continue  # not in manifest / no folder on disk (expected for some)

            if args.dry_run:
                log.info("[dry-run] %s -> %s", tag, folder.name)
                continue

            log.info("Processing %s (%s)", tag, folder.name)

            # ---- ECG (raw) ----
            ecg_sig, ecg_fs = loader.load_ecg(folder)
            ecg_df = pe.process_ecg(ecg_sig, ecg_fs) if ecg_sig is not None else None

            # ---- RSP processing (both channels) up front (RSA needs abdominal) ----
            rsp_proc: dict[str, pd.DataFrame] = {}
            for ch in ("thoracic", "abdominal"):
                sig, fs = loader.load_rsp(folder, ch)
                if sig is None:
                    failures.append(f"- {tag}: missing {ch} RSP wav")
                    continue
                pdf = pr.process_rsp(sig, ch, fs)
                if pdf is None:
                    failures.append(f"- {tag}: rsp_process failed ({ch})")
                    continue
                rsp_proc[ch] = pdf

            abd_df = rsp_proc.get("abdominal")

            # ---- ECG features ----
            if ecg_df is not None:
                row = {"participant_id": pid, "session_idx": sidx, "ecg_fs": ecg_fs}
                row.update(fe.piece_hrv(ecg_df, ecg_fs, rsp_df=abd_df))
                ecg_rows.append(row)

                for feat, (t, v) in fe.continuous_ecg(ecg_df, ecg_fs).items():
                    grid = normalize_traj(t, v)
                    if grid is not None:
                        for tp, val in zip(PCT_GRID, grid):
                            ecg_cont.append({"participant_id": pid, "session_idx": sidx,
                                             "feature": feat, "t_pct": tp, "value": val})
            else:
                failures.append(f"- {tag}: ECG missing or ecg_process failed")

            # ---- RRI cross-check ----
            rri_ms = loader.load_rri_ms(folder)
            if rri_ms is not None and len(rri_ms) >= 30:
                rri_hrv = fe.hrv_from_rri(rri_ms)
                xrow = {"participant_id": pid, "session_idx": sidx,
                        "n_rri": rri_hrv.get("n_rri")}
                # raw-ECG counterparts (if available)
                raw = ecg_rows[-1] if (ecg_df is not None and ecg_rows and
                                       ecg_rows[-1]["participant_id"] == pid and
                                       ecg_rows[-1]["session_idx"] == sidx) else {}
                for key, nk_col in [("rmssd", "HRV_RMSSD"), ("sdnn", "HRV_SDNN"),
                                    ("meannn", "HRV_MeanNN"), ("hf", "HRV_HF"),
                                    ("lf", "HRV_LF"), ("lfhf", "HRV_LFHF")]:
                    xrow[f"rri_{key}"] = rri_hrv.get(nk_col, np.nan)
                    xrow[f"raw_{key}"] = raw.get(nk_col, np.nan)
                # disagreement flags
                xrow["delta_rmssd"] = xrow["raw_rmssd"] - xrow["rri_rmssd"]
                rri_hr = (60000.0 / xrow["rri_meannn"]
                          if xrow.get("rri_meannn") and np.isfinite(xrow["rri_meannn"]) and xrow["rri_meannn"] > 0
                          else np.nan)
                xrow["rri_mean_hr"] = rri_hr
                xrow["raw_mean_hr"] = raw.get("ECG_Rate_Mean", np.nan)
                xrow["delta_mean_hr"] = xrow["raw_mean_hr"] - rri_hr
                xcheck_rows.append(xrow)

            # ---- RSP features ----
            for ch, pdf in rsp_proc.items():
                rfs = RSP_FS
                row = {"participant_id": pid, "session_idx": sidx, "rsp_fs": rfs}
                row.update(fr.piece_rsp(pdf, ch, rfs))
                rsp_rows.append(row)

                for feat, (t, v) in fr.continuous_rsp(pdf, ch, rfs).items():
                    grid = normalize_traj(t, v)
                    if grid is not None:
                        for tp, val in zip(PCT_GRID, grid):
                            rsp_cont.append({"participant_id": pid, "session_idx": sidx,
                                             "channel": ch, "feature": feat,
                                             "t_pct": tp, "value": val})

    if args.dry_run:
        log.info("Dry run complete (%d participants x %d sessions).",
                 len(participants), len(sessions))
        return

    # ---- write outputs ----
    def _save(rows, name):
        df = pd.DataFrame(rows)
        path = out_dir / name
        df.to_csv(path, index=False)
        log.info("Wrote %s (%d rows)", path, len(df))
        return df

    _save(ecg_rows, "ecg_hrv_piece.csv")
    _save(rsp_rows, "rsp_features_piece.csv")
    _save(xcheck_rows, "hrv_raw_vs_rri.csv")
    _save(ecg_cont, "ecg_continuous_1hz.csv")
    _save(rsp_cont, "rsp_continuous_1hz.csv")

    fail_path = out_dir / "failures.md"
    fail_path.write_text(
        "# NeuroKit pipeline failures / skips\n\n" +
        ("\n".join(failures) if failures else "_None._") + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %s (%d entries)", fail_path, len(failures))


if __name__ == "__main__":
    main()
