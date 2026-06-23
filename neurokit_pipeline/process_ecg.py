"""
process_ecg.py
---------------
Process a raw Hexoskin ECG waveform with NeuroKit2.

Adapted from MOXIE Pipeline ``src/processing/process_hex_ecg.py``. The 2Inspire
raw signal arrives as a WAV array (loaded by ``signal_loader.HexLoader``) rather
than a Hexoskin CSV, so the CSV column-discovery / time-column fs inference is
dropped in favour of the known 256 Hz rate. The NeuroKit2 call is identical:

    nk.ecg_process(signal, sampling_rate=256)
        -> ECG_Clean, ECG_Rate, ECG_R_Peaks, ECG_Quality, ...
"""

import logging

import neurokit2 as nk
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def process_ecg(signal: np.ndarray, sampling_rate: int = 256) -> pd.DataFrame | None:
    """Run ``nk.ecg_process`` on a raw ECG array.

    Parameters
    ----------
    signal:
        Raw single-lead ECG samples (textile electrode; amplitude scale is
        irrelevant — NeuroKit2 cleans/normalises internally).
    sampling_rate:
        ECG sampling rate in Hz (256 for Hexoskin).

    Returns
    -------
    pd.DataFrame or None
        Per-sample processed frame, or ``None`` if processing fails or the
        signal is too short.
    """
    if signal is None or len(signal) < sampling_rate * 5:
        logger.warning("ECG signal too short (%s samples) — skipping.",
                       0 if signal is None else len(signal))
        return None
    try:
        signals, _ = nk.ecg_process(signal, sampling_rate=int(sampling_rate))
    except Exception as exc:  # noqa: BLE001
        logger.error("nk.ecg_process failed: %s", exc)
        return None
    return signals
