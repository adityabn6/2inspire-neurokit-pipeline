"""
music_segments.py
-----------------
First-pass detection of the actual music window (start/stop) inside each
performance audio clip, and a comparison of that music duration against the
Hexoskin recording duration for the same piece.

Purpose: the audio clips include setup/silence around the playing. Knowing the
music start/stop lets us (a) trim audio features to the performance and (b) test
whether the audio music window and the Hexoskin recording cover the SAME span —
the prerequisite for putting music and physiology on a common normalized
timeline (coarse, ~second-level, within-piece alignment).

Detection: librosa.effects.split (top_db) → non-silent intervals; music_start =
first interval start, music_stop = last interval end. This is a starting point /
cross-check for the manual timestamp annotations, not a replacement.

Outputs:
  results/analysis/music_segments_long.csv   participant, piece, start/stop/dur, hex_dur, diff
  results/analysis/music_segments_wide.csv    one row/participant, columns p1/p2/p3 start,stop,dur
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("music_segments")

_ROOT = Path(__file__).resolve().parent.parent
ANA = _ROOT / "results" / "analysis"
NK = _ROOT / "results" / "neurokit_features"
SR = 22050
_RE = re.compile(r"P(\d+)_video(\d+)", re.IGNORECASE)


def detect(path: Path, top_db: float) -> dict:
    import librosa
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    total = len(y) / sr
    iv = librosa.effects.split(y, top_db=top_db)   # (N,2) non-silent sample ranges
    if len(iv) == 0:
        return {"audio_total_s": total, "music_start_s": np.nan,
                "music_stop_s": np.nan, "music_dur_s": np.nan, "n_segments": 0}
    start = iv[0, 0] / sr
    stop = iv[-1, 1] / sr
    return {"audio_total_s": round(total, 2),
            "music_start_s": round(start, 2),
            "music_stop_s": round(stop, 2),
            "music_dur_s": round(stop - start, 2),
            "n_segments": int(len(iv))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect music window per audio clip.")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--top-db", type=float, default=30.0,
                    help="silence threshold below peak (higher = stricter silence)")
    ap.add_argument("--out-long", default=str(ANA / "music_segments_long.csv"))
    ap.add_argument("--out-wide", default=str(ANA / "music_segments_wide.csv"))
    args = ap.parse_args()

    # Hexoskin recording duration per (participant, piece)
    hex_dur = {}
    ep = NK / "ecg_hrv_piece.csv"
    if ep.exists():
        e = pd.read_csv(ep)
        for _, r in e.iterrows():
            hex_dur[(int(r["participant_id"]), int(r["session_idx"]))] = float(r["duration_s"])

    rows = []
    for f in sorted(Path(args.audio_dir).glob("*.mp3")):
        m = _RE.search(f.name)
        if not m:
            continue
        pid, piece = int(m.group(1)), int(m.group(2))
        d = detect(f, args.top_db)
        hd = hex_dur.get((pid, piece), np.nan)
        d.update({"participant": f"P{pid}", "participant_id": pid, "piece": piece,
                  "hex_dur_s": round(hd, 2) if np.isfinite(hd) else np.nan,
                  "music_minus_hex_s": round(d["music_dur_s"] - hd, 2)
                  if np.isfinite(hd) and np.isfinite(d["music_dur_s"]) else np.nan})
        rows.append(d)
        log.info("P%s p%s: music %.0f-%.0fs (%.0fs) vs hex %.0fs  d=%s",
                 pid, piece, d["music_start_s"], d["music_stop_s"],
                 d["music_dur_s"], hd, d["music_minus_hex_s"])

    long = pd.DataFrame(rows)[
        ["participant", "participant_id", "piece", "audio_total_s",
         "music_start_s", "music_stop_s", "music_dur_s", "n_segments",
         "hex_dur_s", "music_minus_hex_s"]]
    Path(args.out_long).parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.out_long, index=False)

    # wide: one row per participant, columns per piece
    wide = long.pivot_table(index=["participant", "participant_id"], columns="piece",
                            values=["music_start_s", "music_stop_s", "music_dur_s",
                                    "hex_dur_s"])
    wide.columns = [f"p{pc}_{name[:-2] if name.endswith('_s') else name}"
                    for name, pc in wide.columns]
    wide = wide.reset_index().sort_values("participant_id")
    wide.to_csv(args.out_wide, index=False)

    log.info("Wrote %s (%d rows) and %s", args.out_long, len(long), args.out_wide)
    # quick agreement summary
    valid = long.dropna(subset=["music_minus_hex_s"])
    if len(valid):
        log.info("music vs hex duration: median diff %.1fs, |diff|<=5s in %d/%d",
                 valid["music_minus_hex_s"].median(),
                 int((valid["music_minus_hex_s"].abs() <= 5).sum()), len(valid))


if __name__ == "__main__":
    main()
