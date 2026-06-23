"""
features_rsp.py
----------------
Respiration feature extraction from processed Hexoskin RSP, for 2Inspire.

Vendored from MOXIE Pipeline ``src/utils/rsp_feature_utils.py`` and adapted to
operate on a *single* channel's processed frame (columns prefixed
``respiration_<channel>_``, as produced by ``process_rsp.process_rsp``):

  * ``piece_rsp``       — whole-piece scalar features per channel: RSP_Rate /
                          amplitude / RVT means+SDs, slope & phase inhale/exhale
                          ratios, NeuroKit2 RRV (nk.rsp_rrv) and interval-related
                          (nk.rsp_intervalrelated, NK_ prefix), plus QC.
  * ``continuous_rsp``  — RSP_Rate and RSP_Amplitude trajectories at 1 Hz.
"""

import logging
from typing import Optional

import neurokit2 as nk
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTLIER_RMSD_THRESHOLD = 3.0
GRADIENT_THRESHOLD_PCT = 0.05


def _col(df: pd.DataFrame, channel: str, base: str) -> Optional[str]:
    name = f"respiration_{channel}_{base}"
    return name if name in df.columns else None


def _gradient_threshold(df: pd.DataFrame, channel: str) -> Optional[float]:
    col = _col(df, channel, "RSP_Clean")
    if col is None:
        return None
    sig = df[col].dropna().to_numpy()
    if len(sig) == 0:
        return None
    return float(GRADIENT_THRESHOLD_PCT * np.std(np.gradient(sig)))


def _standardize(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Rename a channel's prefixed columns to bare NeuroKit2 names."""
    nk_df = pd.DataFrame(index=df.index)
    for base in ("RSP_Raw", "RSP_Clean", "RSP_Amplitude", "RSP_Rate",
                 "RSP_Phase", "RSP_Phase_Completion", "RSP_Peaks", "RSP_Troughs"):
        src = _col(df, channel, base)
        if src is not None:
            nk_df[base] = df[src]
    return nk_df


def _rrv_features(df: pd.DataFrame, channel: str, sampling_rate: int) -> dict:
    keys = ["RRV_MeanBB", "RRV_SDBB", "RRV_RMSSD", "RRV_SDSD", "RRV_LF", "RRV_HF",
            "RRV_LFHF", "RRV_LFn", "RRV_HFn", "RRV_SD1", "RRV_SD2", "RRV_SD2SD1",
            "RRV_DFA_alpha1", "RRV_DFA_alpha2", "RRV_ApEn", "RRV_SampEn"]
    nan = {k: np.nan for k in keys}
    col_rate = _col(df, channel, "RSP_Rate")
    col_tr = _col(df, channel, "RSP_Troughs")
    if col_rate is None or col_tr is None:
        return nan
    trough_idx = np.flatnonzero(df[col_tr].to_numpy() == 1)
    if len(trough_idx) < 3:
        return nan
    try:
        rrv = nk.rsp_rrv(df[col_rate].to_numpy(),
                         troughs={"RSP_Troughs": trough_idx},
                         sampling_rate=sampling_rate, show=False, silent=True)
        out = nan.copy()
        out.update(rrv.iloc[0].to_dict())
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("nk.rsp_rrv failed (%s): %s", channel, exc)
        return nan


def _intervalrelated(df: pd.DataFrame, channel: str, sampling_rate: int) -> dict:
    nk_df = _standardize(df, channel)
    if nk_df.empty:
        return {}
    try:
        interval = nk.rsp_intervalrelated(nk_df, sampling_rate=sampling_rate)
        feats = {}
        for k, v in interval.iloc[0].to_dict().items():
            if k.startswith("RRV_") or k == "RSP_Rate_Mean":
                continue
            feats[f"NK_{k}"] = v
        return feats
    except Exception as exc:  # noqa: BLE001
        logger.debug("nk.rsp_intervalrelated skipped (%s): %s", channel, exc)
        return {}


def _quality(df: pd.DataFrame, channel: str) -> dict:
    col_clean = _col(df, channel, "RSP_Clean")
    if col_clean is None:
        return {"pct_missing": np.nan, "pct_low_sqi": np.nan, "n_outliers": 0}
    x = df[col_clean].dropna()
    pct_missing = float(100.0 * df[col_clean].isna().mean())
    if x.empty:
        return {"pct_missing": pct_missing, "pct_low_sqi": np.nan, "n_outliers": 0}
    mean = x.mean()
    rmsd = float(np.sqrt(np.mean((x - mean) ** 2)))
    n_out = int((np.abs(x - mean) > OUTLIER_RMSD_THRESHOLD * rmsd).sum()) if rmsd else 0
    return {"pct_missing": pct_missing, "pct_low_sqi": np.nan, "n_outliers": n_out}


def piece_rsp(df: pd.DataFrame, channel: str, sampling_rate: int) -> dict:
    """Whole-piece RSP scalar features + QC for one channel (MOXIE logic)."""
    result: dict = {"channel": channel}
    col_rate = _col(df, channel, "RSP_Rate")
    col_amp = _col(df, channel, "RSP_Amplitude")
    col_rvt = _col(df, channel, "RSP_RVT")
    col_phase = _col(df, channel, "RSP_Phase")
    col_clean = _col(df, channel, "RSP_Clean")

    result["RSP_Rate_Mean"] = float(df[col_rate].mean()) if col_rate else np.nan
    result["RSP_Rate_SD"] = float(df[col_rate].std()) if col_rate else np.nan
    result["RSP_Amp_Mean"] = float(df[col_amp].mean()) if col_amp else np.nan
    result["RSP_Amp_SD"] = float(df[col_amp].std()) if col_amp else np.nan
    result["RSP_RVT_Mean"] = float(df[col_rvt].mean()) if col_rvt else np.nan
    result["RSP_RVT_SD"] = float(df[col_rvt].std()) if col_rvt else np.nan

    # slope-based inhale/exhale/hold ratios
    gt = _gradient_threshold(df, channel)
    if col_clean is not None and gt is not None:
        grad = np.gradient(df[col_clean].to_numpy())
        n = len(grad)
        if n > 0:
            inhale = int(np.sum(grad > gt))
            exhale = int(np.sum(grad < -gt))
            hold = int(np.sum(np.abs(grad) <= gt))
            result["RSP_Slope_Inhale_Ratio"] = inhale / n
            result["RSP_Slope_Exhale_Ratio"] = exhale / n
            result["RSP_Slope_Hold_Ratio"] = hold / n
            result["RSP_Slope_Dominant"] = max(
                {"Inhale": inhale, "Exhale": exhale, "Hold": hold},
                key=lambda k: {"Inhale": inhale, "Exhale": exhale, "Hold": hold}[k])
    else:
        result["RSP_Slope_Inhale_Ratio"] = np.nan
        result["RSP_Slope_Exhale_Ratio"] = np.nan
        result["RSP_Slope_Hold_Ratio"] = np.nan
        result["RSP_Slope_Dominant"] = "Unknown"

    # nk RSP_Phase ratios (1=inhale, 0=exhale)
    if col_phase:
        norm = df[col_phase].value_counts(normalize=True)
        result["RSP_Inhale_Ratio"] = float(norm.get(1.0, 0.0))
        result["RSP_Exhale_Ratio"] = float(norm.get(0.0, 0.0))
    else:
        result["RSP_Inhale_Ratio"] = np.nan
        result["RSP_Exhale_Ratio"] = np.nan

    result.update(_rrv_features(df, channel, sampling_rate))
    result.update(_intervalrelated(df, channel, sampling_rate))
    result.update(_quality(df, channel))
    return result


def continuous_rsp(df: pd.DataFrame, channel: str, sampling_rate: int) -> dict:
    """Continuous RSP trajectories keyed by feature name -> (t_sec, value)."""
    from features_ecg import _bin_to_1hz  # reuse 1 Hz binner

    traj: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    col_rate = _col(df, channel, "RSP_Rate")
    col_amp = _col(df, channel, "RSP_Amplitude")
    if col_rate:
        traj["rsp_rate"] = _bin_to_1hz(df[col_rate].to_numpy(dtype=float), sampling_rate)
    if col_amp:
        traj["rsp_amplitude"] = _bin_to_1hz(df[col_amp].to_numpy(dtype=float), sampling_rate)
    return traj
