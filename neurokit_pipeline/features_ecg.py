"""
features_ecg.py
----------------
HRV feature extraction from processed Hexoskin ECG, for the 2Inspire pipeline.

The HRV computation is vendored from MOXIE Pipeline
``src/utils/hrv_feature_utils.py`` (the same NeuroKit2 calls and minimum-data
guards), adapted to the 2Inspire case where each recording is a single
performance *piece* rather than an event-segmented protocol:

  * ``piece_hrv``       — whole-piece HRV (time/frequency/nonlinear + optional
                          RSA) from the raw-ECG R-peaks, plus QC columns.
  * ``hrv_from_rri``    — the same time/frequency/nonlinear bundle computed from
                          the device ``RR_interval.csv`` (cross-check source),
                          using the eda_open_exploration.py virtual-1000-Hz
                          idiom.
  * ``continuous_ecg``  — light continuous trajectories (instantaneous HR at
                          1 Hz; rolling RMSSD) for the trajectory plots.
"""

import logging
from typing import Optional

import neurokit2 as nk
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- thresholds (from MOXIE hrv_feature_utils) ---
MIN_PEAKS_TIME = 2
MIN_PEAKS_NONLINEAR = 10
MIN_DURATION_FREQ_S = 30.0
MIN_DURATION_FULL_FREQ_S = 120.0
SQI_THRESHOLD = 0.8
HR_OUTLIER_LOW = 30.0
HR_OUTLIER_HIGH = 200.0

# continuous-trajectory windowing (shorter than MOXIE's 300 s — pieces are short)
ROLL_WINDOW_S = 30
ROLL_STEP_S = 5


# ---------------------------------------------------------------------------
# Core HRV suite (vendored from MOXIE compute_hrv_full)
# ---------------------------------------------------------------------------
def _df_to_dict(df: pd.DataFrame) -> dict:
    out = {}
    for col in df.columns:
        val = df[col].iloc[0]
        try:
            val = float(val)
            if not np.isfinite(val):
                val = np.nan
        except (TypeError, ValueError):
            val = np.nan
        out[col] = val
    return out


def compute_hrv_full(
    peak_indices: np.ndarray,
    sampling_rate: int,
    segment_duration_s: float,
) -> dict:
    """Time + frequency + nonlinear HRV with graceful degradation (MOXIE logic)."""
    result: dict = {}
    n_peaks = len(peak_indices)

    if n_peaks >= MIN_PEAKS_TIME:
        try:
            result.update(_df_to_dict(
                nk.hrv_time(peak_indices, sampling_rate=sampling_rate, show=False)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("nk.hrv_time failed: %s", exc)

    if n_peaks >= MIN_PEAKS_TIME and segment_duration_s >= MIN_DURATION_FREQ_S:
        try:
            result.update(_df_to_dict(
                nk.hrv_frequency(peak_indices, sampling_rate=sampling_rate,
                                 psd_method="welch", normalize=False, show=False)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("nk.hrv_frequency failed: %s", exc)

    if n_peaks >= MIN_PEAKS_NONLINEAR:
        try:
            result.update(_df_to_dict(
                nk.hrv_nonlinear(peak_indices, sampling_rate=sampling_rate, show=False)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("nk.hrv_nonlinear failed: %s", exc)

    return result


def _resample_to_len(values: np.ndarray, n: int) -> np.ndarray:
    """Linear-resample a continuous series to length ``n``."""
    xp = np.linspace(0.0, 1.0, len(values))
    x = np.linspace(0.0, 1.0, n)
    return np.interp(x, xp, np.asarray(values, dtype=float))


def compute_rsa(ecg_df: pd.DataFrame, sampling_rate: int,
                rsp_df: Optional[pd.DataFrame], channel: str = "abdominal") -> dict:
    """Optional RSA (respiratory sinus arrhythmia) via nk.hrv_rsa.

    To keep the respiratory phase/peak markers internally consistent on the ECG
    sample grid, the abdominal RSP_Clean is resampled to the ECG length and
    *re-processed* with nk.rsp_process at the ECG rate (rather than remapping the
    original markers, which de-syncs cycle onsets from peaks). Best-effort:
    returns ``{}`` if RSP is unavailable or cycles can't be resolved.
    """
    if rsp_df is None or "ECG_R_Peaks" not in ecg_df.columns:
        return {}
    n = len(ecg_df)
    clean_col = f"respiration_{channel}_RSP_Clean"
    if clean_col not in rsp_df.columns:
        return {}
    try:
        rsp_clean = rsp_df[clean_col].to_numpy(dtype=float)
        if len(rsp_clean) < 2:
            return {}
        rsp_on_ecg_grid = _resample_to_len(rsp_clean, n)
        rsp_signals, _ = nk.rsp_process(
            rsp_on_ecg_grid, sampling_rate=sampling_rate, method="khodadad2018")
        ecg_signals = pd.DataFrame({
            "ECG_Rate": ecg_df["ECG_Rate"].to_numpy() if "ECG_Rate" in ecg_df.columns
            else np.full(n, np.nan),
            "ECG_R_Peaks": ecg_df["ECG_R_Peaks"].to_numpy(),
        })
        rsa = nk.hrv_rsa(ecg_signals, rsp_signals, sampling_rate=sampling_rate)
        if isinstance(rsa, pd.DataFrame):
            return _df_to_dict(rsa)
        if isinstance(rsa, dict):
            return {k: (float(v) if np.isscalar(v) and np.isfinite(v) else np.nan)
                    for k, v in rsa.items()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("nk.hrv_rsa failed: %s", exc)
    return {}


def _quality_metrics(df: pd.DataFrame) -> dict:
    m: dict = {}
    if "ECG_Rate" in df.columns:
        m["pct_missing"] = float(100.0 * df["ECG_Rate"].isna().mean())
        rate = df["ECG_Rate"].dropna()
        m["n_rate_outliers"] = int(((rate < HR_OUTLIER_LOW) | (rate > HR_OUTLIER_HIGH)).sum())
    else:
        m["pct_missing"] = np.nan
        m["n_rate_outliers"] = 0
    if "ECG_Quality" in df.columns:
        q = df["ECG_Quality"].dropna()
        m["pct_low_sqi"] = float(100.0 * (q < SQI_THRESHOLD).mean()) if len(q) else np.nan
        m["ecg_quality_mean"] = float(q.mean()) if len(q) else np.nan
    else:
        m["pct_low_sqi"] = np.nan
        m["ecg_quality_mean"] = np.nan
    return m


# ---------------------------------------------------------------------------
# Whole-piece HRV from raw-ECG R-peaks
# ---------------------------------------------------------------------------
def piece_hrv(
    ecg_df: pd.DataFrame,
    sampling_rate: int,
    rsp_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Whole-piece HRV + QC from a processed ECG frame.

    ``rsp_df`` is the processed abdominal RSP frame (channel-prefixed columns);
    when present, an optional RSA estimate is added via ``compute_rsa``.
    """
    peak_indices = np.where(ecg_df["ECG_R_Peaks"].to_numpy() == 1)[0]
    duration_s = float(len(ecg_df) / sampling_rate)

    row = {
        "n_peaks": int(len(peak_indices)),
        "duration_s": duration_s,
        "ECG_Rate_Mean": float(ecg_df["ECG_Rate"].mean())
        if "ECG_Rate" in ecg_df.columns and not ecg_df["ECG_Rate"].isna().all()
        else np.nan,
    }
    row.update(_quality_metrics(ecg_df))
    row.update(compute_hrv_full(peak_indices, sampling_rate, duration_s))
    row.update(compute_rsa(ecg_df, sampling_rate, rsp_df, channel="abdominal"))
    return row


# ---------------------------------------------------------------------------
# Cross-check HRV from the device RR_interval.csv
# ---------------------------------------------------------------------------
def hrv_from_rri(rri_ms: np.ndarray) -> dict:
    """HRV (time/frequency/nonlinear) from device RR-intervals (ms).

    Uses the eda_open_exploration.py pattern: build cumulative peak indices on
    a virtual 1000 Hz timeline (1 sample == 1 ms).
    """
    out: dict = {"n_rri": int(len(rri_ms))}
    if rri_ms is None or len(rri_ms) < 30:
        return out
    sr = 1000
    peaks = np.cumsum(np.round(rri_ms).astype(int))
    duration_s = float(peaks[-1] / sr) if len(peaks) else 0.0
    out.update(compute_hrv_full(peaks, sr, duration_s))
    return out


# ---------------------------------------------------------------------------
# Continuous trajectories
# ---------------------------------------------------------------------------
def _bin_to_1hz(values: np.ndarray, sampling_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Average a per-sample series into 1-second bins. Returns (t_sec, mean)."""
    n = len(values)
    if n == 0:
        return np.array([]), np.array([])
    t = np.arange(n) / float(sampling_rate)
    sec = np.floor(t).astype(int)
    s = pd.Series(values).groupby(sec).mean()
    return (s.index.to_numpy() + 0.5).astype(float), s.to_numpy(dtype=float)


def continuous_ecg(ecg_df: pd.DataFrame, sampling_rate: int) -> dict:
    """Continuous ECG trajectories keyed by feature name -> (t_sec, value)."""
    traj: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # Instantaneous heart rate (bpm) at 1 Hz
    if "ECG_Rate" in ecg_df.columns:
        traj["hr_bpm"] = _bin_to_1hz(ecg_df["ECG_Rate"].to_numpy(dtype=float), sampling_rate)

    # Rolling RMSSD (ms) over 30 s / 5 s windows
    peak_idx = np.where(ecg_df["ECG_R_Peaks"].to_numpy() == 1)[0]
    peak_t = peak_idx / float(sampling_rate)
    total_s = len(ecg_df) / float(sampling_rate)
    t_centers, rmssd_vals = [], []
    if len(peak_t) >= 3 and total_s >= ROLL_WINDOW_S:
        w = 0.0
        while w + ROLL_WINDOW_S <= total_s:
            in_win = peak_t[(peak_t >= w) & (peak_t < w + ROLL_WINDOW_S)]
            if len(in_win) >= 3:
                rr_ms = np.diff(in_win) * 1000.0
                rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) >= 2 else np.nan
            else:
                rmssd = np.nan
            t_centers.append(w + ROLL_WINDOW_S / 2.0)
            rmssd_vals.append(rmssd)
            w += ROLL_STEP_S
    if t_centers:
        traj["hrv_rmssd"] = (np.asarray(t_centers), np.asarray(rmssd_vals))

    return traj
