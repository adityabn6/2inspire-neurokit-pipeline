"""
remap_audio.py
--------------
Fix the audio↔physiology piece mapping by DURATION matching.

The audio clips are named by video number (P{id}_video{k}); we had been assuming
video k == Hexoskin piece k. That is wrong for P18 (whose video order is
chronological while the Hexoskin pieces are in performance order — video 1 ≈ 709 s
matches Hexoskin piece 3 ≈ 714 s). For each participant we instead assign each
audio clip to the Hexoskin piece with the closest recording duration
(optimal assignment), so audio features merge onto the correct physiology.

Degenerate case: P1's three "pieces" are a single video clip (identical detected
window) — it cannot be split into pieces by duration, so it is excluded from the
per-piece audio (flagged `degenerate`) until manual timestamps arrive.

Inputs : results/analysis/audio_features_trimmed.csv (col audio_duration_s =
         trimmed clip length), results/neurokit_features/ecg_hrv_piece.csv.
Outputs: results/analysis/audio_piece_map.csv  (video -> hex_piece, durations, flags)
         results/analysis/audio_features_mapped.csv  (audio features keyed by hex piece)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("remap_audio")

_ROOT = Path(__file__).resolve().parent.parent
ANA = _ROOT / "results" / "analysis"
NK = _ROOT / "results" / "neurokit_features"
DEGEN_TOL_S = 3.0   # clip durations within this of each other => can't separate
GROSS_RESID_S = 60.0  # identity mapping is "obviously wrong" above this
CONF_RESID_S = 30.0   # only override if the reassignment fits this well


def main() -> None:
    audio = pd.read_csv(ANA / "audio_features_trimmed.csv")
    hex_df = pd.read_csv(NK / "ecg_hrv_piece.csv")[
        ["participant_id", "session_idx", "duration_s"]]

    maps = []
    for pid, g in audio.groupby("participant_id"):
        clips = g[["video", "audio_duration_s"]].dropna().sort_values("video")
        hexp = hex_df[hex_df.participant_id == pid][["session_idx", "duration_s"]]
        if clips.empty or hexp.empty:
            continue
        cv = clips["video"].to_numpy()
        cd = clips["audio_duration_s"].to_numpy(float)
        hp = hexp["session_idx"].to_numpy()
        hd = hexp["duration_s"].to_numpy(float)

        # degenerate: all clip durations ~equal but hex durations differ -> can't map
        degenerate = (len(cd) > 1 and np.ptp(cd) < DEGEN_TOL_S and np.ptp(hd) > 10)

        hd_by_piece = dict(zip(hp, hd))
        # identity mapping (video k -> hex piece k) and its worst residual
        id_resids = [abs(cd[i] - hd_by_piece[cv[i]])
                     for i in range(len(cv)) if cv[i] in hd_by_piece]
        identity_max = max(id_resids) if id_resids else np.inf

        # Default = identity. Only override when identity is grossly wrong AND an
        # optimal duration assignment fits confidently (avoids reshuffling pieces
        # that merely differ by a few seconds).
        assign = {int(v): int(v) for v in cv}            # video -> hex piece
        use_optimal = False
        if not degenerate and identity_max > GROSS_RESID_S:
            ri, ci = linear_sum_assignment(np.abs(cd[:, None] - hd[None, :]))
            opt = {int(cv[i]): int(hp[j]) for i, j in zip(ri, ci)}
            opt_max = max(abs(cd[i] - hd[list(hp).index(opt[int(cv[i])])])
                          for i in range(len(cv)))
            if opt_max < CONF_RESID_S:
                assign, use_optimal = opt, True

        for i in range(len(cv)):
            v = int(cv[i]); hpiece = assign[v]
            maps.append({"participant_id": pid, "video": v, "hex_piece": hpiece,
                         "clip_dur_s": round(cd[i], 1),
                         "hex_dur_s": round(hd_by_piece.get(hpiece, np.nan), 1),
                         "dur_resid_s": round(abs(cd[i] - hd_by_piece.get(hpiece, np.nan)), 1)
                         if hpiece in hd_by_piece else np.nan,
                         "remapped": bool(v != hpiece), "degenerate": degenerate})

    mp = pd.DataFrame(maps)
    mp.to_csv(ANA / "audio_piece_map.csv", index=False)
    n_remap = int((mp.remapped & ~mp.degenerate).sum())
    log.info("mapping: %d clips, %d remapped, %d degenerate participants",
             len(mp), n_remap, mp[mp.degenerate].participant_id.nunique())
    for _, r in mp[mp.remapped | mp.degenerate].iterrows():
        log.info("  P%s video%s -> hex piece %s (clip %.0fs vs hex %.0fs)%s",
                 r.participant_id, r.video, r.hex_piece, r.clip_dur_s, r.hex_dur_s,
                 "  [DEGENERATE]" if r.degenerate else "  [REMAPPED]")

    # build mapped audio features keyed by corrected hex piece
    key = mp[~mp.degenerate][["participant_id", "video", "hex_piece"]]
    mapped = audio.merge(key, on=["participant_id", "video"], how="inner")
    mapped["piece"] = mapped["hex_piece"]
    mapped["condition"] = np.where(mapped["piece"] == 3, "self_selected", "sight_reading")
    mapped = mapped.drop(columns=["hex_piece"])
    mapped.to_csv(ANA / "audio_features_mapped.csv", index=False)
    log.info("Wrote audio_features_mapped.csv (%d rows; %d participants excluded as degenerate)",
             len(mapped), mp[mp.degenerate].participant_id.nunique())


if __name__ == "__main__":
    main()
