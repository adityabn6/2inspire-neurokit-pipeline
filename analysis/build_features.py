"""
build_features.py
-----------------
Assemble ONE tidy row per participant x piece for the 2Inspire music-physiology
analysis.

Design decisions (see reports/hypotheses_music_physiology.md):

  * Piece index comes from the participant manifest (``session_1/2/3``), NOT from
    the alphabetical rank of the on-disk folder name. The rank heuristic silently
    swaps pieces 1 and 3 for P18 (whose 3rd piece was recorded first).
  * HRV is computed from the device ``RR_interval.csv`` (beat-to-beat), preferred
    over raw textile-ECG peaks which inflate RMSSD on ~1/3 of sessions.
  * The device RR series is ALSO artifact-corrected with NeuroKit2
    (``nk.signal_fixpeaks``, Kubios) before HRV — raw device RR has ectopic/
    missed beats that blow up RMSSD/RSA on some participants (e.g. P2 raw
    rri_rmssd ~ 400 ms, physiologically impossible). ``pct_rr_corrected`` and
    ``hrv_qc_ok`` record this.
  * Short pieces (sight-reading ~1-2 min) make LF and LF/HF unreliable; those
    columns are emitted with an ``lfhf_lowconf`` flag and should not be used as
    primary outcomes. Mean RR/HR, RMSSD, pNN50, HF and RSA are the trustworthy
    autonomic readouts.
  * RSA (nk.hrv_rsa) and respiration features are taken from the verified
    neurokit_pipeline outputs (results/neurokit_features/), which are already
    manifest-piece-labelled.
  * ``duration_s`` is carried through because piece length varies 50 s - 16 min
    (piece 3 / self-selected is often far longer) and is a confound for any
    piece contrast — model it, or use duration-matched windows.

Output: results/analysis/features_by_piece.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "neurokit_pipeline"))

from numpy_compat import apply_numpy_compat  # noqa: E402
apply_numpy_compat()
import neurokit2 as nk  # noqa: E402
from signal_loader import HexLoader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("build_features")

DEFAULT_HEX = "/home/adityabn/GoogleDrive/Research/Projects/2Inspire/source_data/Hexoskin"
DEFAULT_MANIFEST = _ROOT / "data" / "participants_anonymized.json"
DEFAULT_NK_DIR = _ROOT / "results" / "neurokit_features"
DEFAULT_OUT = _ROOT / "results" / "analysis" / "features_by_piece.csv"

# Plausibility guard for whole-piece HRV (after artifact correction). RMSSD this
# large is a beat-detection failure, not physiology.
RMSSD_PLAUSIBLE_MAX = 150.0
MIN_RRI = 30


# ---------------------------------------------------------------------------
# Robust device-RR HRV (NeuroKit2, Kubios artifact correction)
# ---------------------------------------------------------------------------
def robust_hrv_from_rri(rri_ms: np.ndarray) -> dict:
    """HRV from device RR-intervals (ms) with Kubios artifact correction.

    Builds peaks on a virtual 1000 Hz timeline (1 sample == 1 ms), corrects
    ectopic/missed beats with nk.signal_fixpeaks, then runs nk.hrv_time /
    nk.hrv_frequency. Returns a curated, trustworthy subset plus QC.
    """
    out = {"n_rri": int(len(rri_ms)) if rri_ms is not None else 0,
           "pct_rr_corrected": np.nan, "hrv_qc_ok": False}
    if rri_ms is None or len(rri_ms) < MIN_RRI:
        return out

    peaks = np.cumsum(np.round(rri_ms).astype(int))
    try:
        _, peaks_clean = nk.signal_fixpeaks(
            peaks, sampling_rate=1000, method="Kubios", iterative=True)
        peaks_clean = np.asarray(peaks_clean, dtype=int)
    except Exception as exc:  # noqa: BLE001
        log.warning("signal_fixpeaks failed (%s); using raw peaks", exc)
        peaks_clean = peaks

    rri_clean = np.diff(peaks_clean)
    n_corr = abs(len(peaks_clean) - len(peaks))
    out["pct_rr_corrected"] = float(100.0 * n_corr / max(len(peaks), 1))
    duration_s = float(peaks_clean[-1] / 1000.0) if len(peaks_clean) else 0.0
    out["rr_duration_s"] = duration_s

    try:
        t = nk.hrv_time(peaks_clean, sampling_rate=1000, show=False)
        out["mean_nn_ms"] = float(t["HRV_MeanNN"].iloc[0])
        out["mean_hr_bpm"] = 60000.0 / out["mean_nn_ms"] if out["mean_nn_ms"] else np.nan
        out["rmssd_ms"] = float(t["HRV_RMSSD"].iloc[0])
        out["sdnn_ms"] = float(t["HRV_SDNN"].iloc[0])
        out["pnn50"] = float(t["HRV_pNN50"].iloc[0])
    except Exception as exc:  # noqa: BLE001
        log.warning("hrv_time failed: %s", exc)

    # Frequency domain — flagged low-confidence on short pieces.
    if duration_s >= 30.0:
        try:
            f = nk.hrv_frequency(peaks_clean, sampling_rate=1000,
                                 psd_method="welch", normalize=False, show=False)
            out["hf_power"] = float(f["HRV_HF"].iloc[0])
            out["lf_power"] = float(f["HRV_LF"].iloc[0])
            out["lfhf"] = float(f["HRV_LFHF"].iloc[0])
        except Exception as exc:  # noqa: BLE001
            log.warning("hrv_frequency failed: %s", exc)
    out["lfhf_lowconf"] = bool(duration_s < 120.0)  # <2 min: LF/HF unreliable

    rmssd = out.get("rmssd_ms", np.nan)
    out["hrv_qc_ok"] = bool(
        np.isfinite(rmssd) and rmssd <= RMSSD_PLAUSIBLE_MAX
        and out["n_rri"] >= MIN_RRI)
    return out


# ---------------------------------------------------------------------------
# RSA + respiration from the verified neurokit_pipeline outputs
# ---------------------------------------------------------------------------
def load_pipeline_extras(nk_dir: Path) -> pd.DataFrame:
    """Merge RSA + duration + QC (ecg_hrv_piece) and abdominal respiration
    (rsp_features_piece) keyed by (participant_id, session_idx)."""
    ecg = pd.read_csv(nk_dir / "ecg_hrv_piece.csv")
    keep_ecg = ["participant_id", "session_idx", "duration_s", "ecg_quality_mean",
                "RSA_P2T_Mean", "RSA_PorgesBohrer", "RSA_Gates_Mean"]
    ecg = ecg[[c for c in keep_ecg if c in ecg.columns]]

    rsp = pd.read_csv(nk_dir / "rsp_features_piece.csv")
    abd = rsp[rsp["channel"] == "abdominal"].copy()
    keep_rsp = {"RSP_Rate_Mean": "resp_rate", "RSP_Amp_Mean": "resp_amp",
                "RSP_RVT_Mean": "resp_rvt", "RRV_RMSSD": "rrv_rmssd"}
    abd = abd[["participant_id", "session_idx"] + list(keep_rsp)].rename(columns=keep_rsp)

    return ecg.merge(abd, on=["participant_id", "session_idx"], how="outer")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble per-piece analysis table.")
    ap.add_argument("--hexoskin-path", default=DEFAULT_HEX)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--nk-dir", default=str(DEFAULT_NK_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--participants", default="all",
                    help="'all' | '1-5' | '1,3,18'")
    ap.add_argument("--window-s", type=float, default=0.0,
                    help="trim each piece's RR series to the first N seconds "
                         "(0 = full piece). Use for duration-matched contrasts.")
    args = ap.parse_args()

    loader = HexLoader(args.hexoskin_path, args.manifest)
    if args.participants.strip().lower() == "all":
        pids = [p["participant_id"] for p in loader.participants()]
    else:
        pids = []
        for part in args.participants.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-")
                pids.extend(range(int(a), int(b) + 1))
            elif part:
                pids.append(int(part))

    rows = []
    for pid in sorted(set(pids)):
        for piece in (1, 2, 3):  # manifest piece order — NOT folder rank
            folder = loader.session_folder(pid, piece)
            if folder is None:
                continue
            rri = loader.load_rri_ms(folder)
            if args.window_s > 0 and rri is not None and len(rri):
                cum_s = np.cumsum(rri) / 1000.0
                rri = rri[cum_s <= args.window_s]
            row = {
                "participant": f"P{pid}",
                "participant_id": pid,
                "session_idx": piece,            # join key for pipeline extras
                "piece": piece,
                "condition": "self_selected" if piece == 3 else "sight_reading",
                "window_s": args.window_s,
            }
            row.update(robust_hrv_from_rri(rri))
            rows.append(row)
            log.info("P%s piece %s: n_rri=%s rmssd=%.1f qc_ok=%s",
                     pid, piece, row.get("n_rri"),
                     row.get("rmssd_ms", float("nan")), row.get("hrv_qc_ok"))

    df = pd.DataFrame(rows)
    extras = load_pipeline_extras(Path(args.nk_dir))
    df = df.merge(extras, on=["participant_id", "session_idx"], how="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Wrote %s (%d rows, %d cols). HRV-QC-ok: %d/%d",
             out, len(df), df.shape[1],
             int(df["hrv_qc_ok"].sum()), len(df))


if __name__ == "__main__":
    main()
