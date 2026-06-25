"""
motion_features.py
------------------
Hand-movement energy per participant × piece from the Movella DOT sensors, as a
**motor-confound covariate** for the cardiac metrics (Iñesta 2008: playing piano
is real physical effort, ~2.5 METs, so HR has a large movement component).

Mapping: the manifest's ``movella_sessions.session_k`` value is the HHMMSS
timestamp embedded in the Movella filenames
``Xsens DOT_<device>_<YYYYMMDD>_<HHMMSS>.csv``. 3–4 DOT sensors are worn, so each
piece has several files sharing that timestamp; we aggregate across them.

Movement energy per sensor = SD of the acceleration-magnitude
``sqrt(Acc_X^2+Acc_Y^2+Acc_Z^2)`` — this removes the constant gravity offset and
captures how much the sensor moves. We also report mean jerk (|d/dt| of the
magnitude). Per piece we take the mean across sensors. Sampling ≈ 60 Hz.

``--window-s N`` trims to the first N s so the covariate matches the
duration-matched HR contrast in stats_models.

Output: results/analysis/motion_by_piece.csv
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("motion")

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOVELLA = "/home/adityabn/GoogleDrive/Research/Projects/2Inspire/source_data/Movella Dots"
DEFAULT_MANIFEST = _ROOT / "data" / "participants_anonymized.json"
DEFAULT_OUT = _ROOT / "results" / "analysis" / "motion_by_piece.csv"
MOVELLA_FS = 60.0


def _sensor_energy(path: Path, window_s: float) -> dict | None:
    # When windowed, only read the rows we need (+warmup pad) — the Drive mount
    # makes reading whole multi-minute files slow.
    nrows = int((window_s + 15) * MOVELLA_FS) if window_s > 0 else None
    try:
        df = pd.read_csv(path, usecols=["Acc_X", "Acc_Y", "Acc_Z"], nrows=nrows)
    except Exception as exc:  # noqa: BLE001
        log.warning("read failed %s: %s", path.name, exc)
        return None
    acc = df[["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy(dtype=float)
    # drop warmup rows that are exactly zero on all axes
    acc = acc[~np.all(acc == 0, axis=1)]
    if window_s > 0:
        acc = acc[: int(window_s * MOVELLA_FS)]
    if len(acc) < int(5 * MOVELLA_FS):     # <5 s unusable
        return None
    mag = np.sqrt((acc ** 2).sum(axis=1))
    jerk = np.abs(np.diff(mag)) * MOVELLA_FS
    return {"acc_std": float(np.std(mag)),
            "jerk_mean": float(np.mean(jerk)),
            "dur_s": len(acc) / MOVELLA_FS}


def index_by_time(movella_dir: Path) -> dict[str, list[Path]]:
    """Map HHMMSS -> sensor files, listing the directory ONCE (FUSE is slow)."""
    idx: dict[str, list[Path]] = {}
    for f in movella_dir.glob("*.csv"):
        tt = f.stem.rsplit("_", 1)[-1]              # trailing HHMMSS
        idx.setdefault(tt, []).append(f)
    return idx


def piece_motion(index: dict, hhmmss: str, window_s: float) -> dict:
    out = {"n_sensors": 0, "motion_energy": np.nan, "jerk_mean": np.nan,
           "motion_dur_s": np.nan}
    if not hhmmss:
        return out
    tt = str(hhmmss).split(".")[0].zfill(6)         # zero-pad e.g. 83710 -> 083710
    files = sorted(index.get(tt, []))
    energies = [e for e in (_sensor_energy(f, window_s) for f in files) if e]
    if not energies:
        return out
    out["n_sensors"] = len(energies)
    out["motion_energy"] = float(np.mean([e["acc_std"] for e in energies]))
    out["jerk_mean"] = float(np.mean([e["jerk_mean"] for e in energies]))
    out["motion_dur_s"] = float(np.mean([e["dur_s"] for e in energies]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-piece Movella hand-motion energy.")
    ap.add_argument("--movella-dir", default=DEFAULT_MOVELLA)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--window-s", type=float, default=0.0)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    movella_dir = Path(args.movella_dir)
    index = index_by_time(movella_dir)
    log.info("indexed %d Movella files at %d timestamps", sum(len(v) for v in index.values()), len(index))
    rows = []
    for p in manifest["participants"]:
        pid = p["participant_id"]
        mv = p["sensor_data"].get("movella_sessions", {})
        for piece in (1, 2, 3):
            hh = mv.get(f"session_{piece}")
            m = piece_motion(index, hh, args.window_s)
            row = {"participant": f"P{pid}", "participant_id": pid, "piece": piece,
                   "condition": "self_selected" if piece == 3 else "sight_reading",
                   "window_s": args.window_s}
            row.update(m)
            rows.append(row)
            log.info("P%s piece %s: %d sensors  motion=%.3f", pid, piece,
                     m["n_sensors"], m["motion_energy"])

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Wrote %s (%d rows; %d with motion)",
             out, len(df), int(df["motion_energy"].notna().sum()))


if __name__ == "__main__":
    main()
