"""
coupling.py
-----------
Respiration <-> cardiac coupling per participant x piece.

Why this is sync-robust: it relates two *physiological* signals recorded by the
SAME device (Hexoskin RR-intervals and respiration), so it does NOT depend on
audio<->physiology alignment (which is unrecoverable for this dataset). HR and
breathing share the participant's internal clock.

Per piece it computes:
  * ``hr_resp_xcorr_peak``  — peak |Pearson r| between z-scored instantaneous
    IBI and respiration on a common 4 Hz grid.
  * ``hr_resp_lag_s``       — lag (s) at that peak (sign = lead/lag).
  * ``hf_coherence``        — mean magnitude-squared coherence in the respiratory
    (HF, 0.15-0.4 Hz) band — coupling strength.
  * ``resp_rate_hz``        — dominant respiration frequency from its PSD.

This complements RSA (nk.hrv_rsa, in build_features.py): RSA is the *magnitude*
of respiration's effect on HR; coherence/lag describe the *coupling structure*.

Output: results/analysis/coupling_by_piece.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "neurokit_pipeline"))
from signal_loader import HexLoader, RSP_FS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("coupling")

DEFAULT_HEX = "/home/adityabn/GoogleDrive/Research/Projects/2Inspire/source_data/Hexoskin"
DEFAULT_MANIFEST = _ROOT / "data" / "participants_anonymized.json"
DEFAULT_OUT = _ROOT / "results" / "analysis" / "coupling_by_piece.csv"

GRID_FS = 4.0           # common resample rate (Hz) — ample for <0.5 Hz dynamics
HF_LO, HF_HI = 0.15, 0.40
MAX_LAG_S = 15.0        # cap xcorr lag search to a physiological HR-resp window
COH_NPERSEG_S = 16.0    # coherence window (s): >=2 cycles of 0.15 Hz; multi-segment


def _ibi_on_grid(rri_ms: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Instantaneous IBI (ms) sampled on t_grid from a device RR series."""
    rr_t = np.cumsum(rri_ms) / 1000.0          # beat times (s)
    rr_t = rr_t - rr_t[0]
    f = interp1d(rr_t, rri_ms, bounds_error=False,
                 fill_value=(rri_ms[0], rri_ms[-1]))
    return f(t_grid)


def coupling_metrics(rri_ms: np.ndarray, rsp: np.ndarray, rsp_fs: int,
                     window_s: float = 0.0) -> dict:
    out = {"hr_resp_xcorr_peak": np.nan, "hr_resp_lag_s": np.nan,
           "hf_coherence": np.nan, "resp_rate_hz": np.nan}
    if rri_ms is None or len(rri_ms) < 30 or rsp is None or len(rsp) < rsp_fs * 20:
        return out

    rsp_dur = len(rsp) / float(rsp_fs)
    rr_dur = float(np.cumsum(rri_ms)[-1] / 1000.0)
    dur = min(rsp_dur, rr_dur)
    if window_s > 0:
        dur = min(dur, window_s)   # duration-matched contrast
    t_grid = np.arange(0, dur, 1.0 / GRID_FS)
    if len(t_grid) < 40:
        return out

    ibi = _ibi_on_grid(rri_ms, t_grid)
    rsp_t = np.arange(len(rsp)) / float(rsp_fs)
    rsp_g = interp1d(rsp_t, rsp, bounds_error=False, fill_value=0.0)(t_grid)

    iz = (ibi - ibi.mean()) / (ibi.std() + 1e-9)
    rz = (rsp_g - rsp_g.mean()) / (rsp_g.std() + 1e-9)

    cc = np.correlate(iz, rz, mode="full") / len(iz)
    lags = np.arange(-len(iz) + 1, len(iz)) / GRID_FS
    win = np.abs(lags) <= MAX_LAG_S          # only physiological lags
    cc_w, lags_w = cc[win], lags[win]
    k = int(np.argmax(np.abs(cc_w)))
    out["hr_resp_xcorr_peak"] = float(np.abs(cc_w[k]))
    out["hr_resp_lag_s"] = float(lags_w[k])

    # coherence: window short enough to yield several averaged segments
    nper = int(min(COH_NPERSEG_S * GRID_FS, len(iz) // 4))
    if nper < 16:
        return out
    fco, coh = signal.coherence(iz, rz, fs=GRID_FS, nperseg=nper)
    band = (fco >= HF_LO) & (fco <= HF_HI)
    if band.any():
        out["hf_coherence"] = float(np.mean(coh[band]))

    fr, psd = signal.welch(rz, fs=GRID_FS, nperseg=nper)
    valid = (fr >= 0.08) & (fr <= 0.6)
    if valid.any():
        out["resp_rate_hz"] = float(fr[valid][np.argmax(psd[valid])])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-piece respiration-cardiac coupling.")
    ap.add_argument("--hexoskin-path", default=DEFAULT_HEX)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--participants", default="all")
    ap.add_argument("--window-s", type=float, default=0.0,
                    help="trim each piece to first N s (0=full) for matched contrast")
    args = ap.parse_args()

    loader = HexLoader(args.hexoskin_path, args.manifest)
    if args.participants.strip().lower() == "all":
        pids = [p["participant_id"] for p in loader.participants()]
    else:
        pids = [int(x) for x in args.participants.split(",")]

    rows = []
    for pid in sorted(set(pids)):
        for piece in (1, 2, 3):
            folder = loader.session_folder(pid, piece)
            if folder is None:
                continue
            rri = loader.load_rri_ms(folder)
            rsp, fs = loader.load_rsp(folder, "abdominal")
            m = coupling_metrics(rri, rsp, fs or RSP_FS, window_s=args.window_s)
            row = {"participant": f"P{pid}", "participant_id": pid, "piece": piece,
                   "condition": "self_selected" if piece == 3 else "sight_reading",
                   "window_s": args.window_s}
            row.update(m)
            rows.append(row)
            log.info("P%s piece %s: xcorr=%.2f lag=%.1fs coh=%.2f resp=%.2fHz",
                     pid, piece, m["hr_resp_xcorr_peak"], m["hr_resp_lag_s"],
                     m["hf_coherence"], m["resp_rate_hz"])

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Wrote %s (%d rows)", out, len(df))


if __name__ == "__main__":
    main()
