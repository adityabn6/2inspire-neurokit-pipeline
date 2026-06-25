"""
audio_features.py
-----------------
Per-piece music features from the performance audio.

Chroma-based key / mode / harmonic complexity is the
approach the PI liked and is methodologically sound (Krumhansl-Schmuckler key
finding; per-frame chroma entropy as harmonic complexity; tonal stability).

What is CHANGED / ADDED:
  * One row per participant x piece. No audio<->physiology correlation lives here
    — those belong at the piece level in stats_models.py. Correlating audio and
    physiology time-series independently resampled to n=20/50 points has NO real
    time alignment (sub-second sync is unrecoverable for this dataset) and yields
    uncorrected, pseudoreplicated
    correlations. That entire step is removed.
  * Constant-Q chroma (``chroma_cqt``) instead of ``chroma_stft`` — pitch-aligned,
    standard for tonal/harmonic analysis.
  * Adds chroma flux (frame-to-frame harmonic change) and onset rate (note
    density) — note density / loudness / tempo are the music features most tied
    to autonomic response in the literature (Solinski 2024; Bernardi 2006), so we
    extract them as covariates for the harmonic features.
  * Drops MFCCs (computed but never used) and the arbitrary weighted "tension"
    score (0.4*rms + 0.3*centroid + 0.3*onset).

Files are mapped ``P{ID}_video{piece}_IMG*.mp3`` → (participant, piece); the
``video`` index equals the manifest piece index.

Output: results/analysis/audio_features_by_piece.csv
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("audio_features")

SR = 22050
HOP = 512

# Krumhansl-Kessler key profiles (major / minor), C-rooted.
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_FNAME_RE = re.compile(r"P(\d+)_video(\d+)", re.IGNORECASE)


def estimate_key(mean_chroma: np.ndarray) -> tuple[str, str, float]:
    """Krumhansl-Schmuckler key/mode from a 12-vector mean chroma.

    Returns (key_name, mode, strength) where strength is the best Pearson
    correlation between the mean chroma and the rotated KS profile.
    """
    best = (-2.0, "C", "major")
    for shift in range(12):
        maj = np.corrcoef(mean_chroma, np.roll(KS_MAJOR, shift))[0, 1]
        minr = np.corrcoef(mean_chroma, np.roll(KS_MINOR, shift))[0, 1]
        if maj > best[0]:
            best = (maj, NOTE_NAMES[shift], "major")
        if minr > best[0]:
            best = (minr, NOTE_NAMES[shift], "minor")
    return best[1], best[2], float(best[0])


def extract(path: Path, start_s: float = 0.0, stop_s: float = 0.0) -> dict:
    import librosa  # imported lazily so --help works without it

    y, sr = librosa.load(str(path), sr=SR, mono=True)
    # trim to the detected/annotated music window if given
    if stop_s and stop_s > start_s:
        y = y[int(start_s * sr):int(stop_s * sr)]
    dur = float(len(y) / sr)
    out = {"audio_duration_s": dur}

    # --- harmony / tonality (constant-Q chroma) ---
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)  # (12, T)
    mean_chroma = chroma.mean(axis=1)
    key, mode, strength = estimate_key(mean_chroma)
    out.update({"key": key, "mode": mode, "key_strength": strength,
                "is_minor": int(mode == "minor")})

    # per-frame normalized chroma -> Shannon entropy (harmonic complexity)
    cn = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-9)
    entropy = -np.sum(cn * np.log2(cn + 1e-9), axis=0)  # bits, 0..log2(12)
    out["chroma_entropy_mean"] = float(np.mean(entropy))
    out["chroma_entropy_sd"] = float(np.std(entropy))

    # tonal stability: mean correlation of each frame to the global mean chroma
    gm = mean_chroma
    stab = [np.corrcoef(chroma[:, i], gm)[0, 1] for i in range(chroma.shape[1])
            if chroma[:, i].std() > 1e-9]
    out["key_stability"] = float(np.nanmean(stab)) if stab else np.nan

    # chroma flux: mean L2 frame-to-frame change (harmonic movement)
    flux = np.sqrt(((np.diff(cn, axis=1)) ** 2).sum(axis=0))
    out["chroma_flux_mean"] = float(np.mean(flux)) if flux.size else np.nan

    # --- intensity / dynamics (literature-validated autonomic drivers) ---
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    out["tempo_bpm"] = float(np.atleast_1d(tempo)[0])

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    out["rms_mean"] = float(np.mean(rms))
    out["rms_sd"] = float(np.std(rms))

    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]
    out["spectral_centroid_mean"] = float(np.mean(cent))

    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, units="time")
    out["onset_rate_hz"] = float(len(onsets) / dur) if dur > 0 else np.nan  # note density

    # harmonic vs percussive energy (HPSS)
    y_h, y_p = librosa.effects.hpss(y)
    h, p = float(np.sqrt(np.mean(y_h ** 2))), float(np.sqrt(np.mean(y_p ** 2)))
    out["harmonic_ratio"] = h / (h + p + 1e-9)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-piece music features from audio.")
    ap.add_argument("--audio-dir", required=True,
                    help="dir of P{ID}_video{piece}_IMG*.mp3 files")
    ap.add_argument("--out",
                    default=str(Path(__file__).resolve().parent.parent
                               / "results" / "analysis" / "audio_features_by_piece.csv"))
    ap.add_argument("--segments", default="",
                    help="optional music_segments_long.csv to trim each clip to "
                         "[music_start_s, music_stop_s] before feature extraction")
    ap.add_argument("--limit", type=int, default=0, help="process only N files (debug)")
    args = ap.parse_args()

    seg = {}
    if args.segments and Path(args.segments).exists():
        s = pd.read_csv(args.segments)
        for _, r in s.iterrows():
            if pd.notna(r.get("music_start_s")) and pd.notna(r.get("music_stop_s")):
                seg[(int(r["participant_id"]), int(r["piece"]))] = \
                    (float(r["music_start_s"]), float(r["music_stop_s"]))

    files = sorted(Path(args.audio_dir).glob("*.mp3"))
    if args.limit:
        files = files[:args.limit]

    rows = []
    for f in files:
        m = _FNAME_RE.search(f.name)
        if not m:
            log.warning("skip unparseable name: %s", f.name)
            continue
        pid, piece = int(m.group(1)), int(m.group(2))
        start_s, stop_s = seg.get((pid, piece), (0.0, 0.0))
        try:
            feat = extract(f, start_s, stop_s)
        except Exception as exc:  # noqa: BLE001
            log.warning("extract failed for %s: %s", f.name, exc)
            continue
        row = {"participant": f"P{pid}", "participant_id": pid, "piece": piece,
               "video": piece,  # video/clip number from filename (pre-remap)
               "condition": "self_selected" if piece == 3 else "sight_reading",
               "audio_file": f.name,
               "trim_start_s": start_s, "trim_stop_s": stop_s}
        row.update(feat)
        rows.append(row)
        log.info("P%s piece %s: key=%s %s entropy=%.2f tempo=%.0f onsets/s=%.2f",
                 pid, piece, feat["key"], feat["mode"],
                 feat["chroma_entropy_mean"], feat["tempo_bpm"], feat["onset_rate_hz"])

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Wrote %s (%d rows, %d cols)", out, len(df), df.shape[1])


if __name__ == "__main__":
    main()
