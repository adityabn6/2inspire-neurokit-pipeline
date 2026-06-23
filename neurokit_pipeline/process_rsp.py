"""
process_rsp.py
---------------
Process a raw Hexoskin respiration waveform (one channel) with NeuroKit2.

Adapted from MOXIE Pipeline ``src/processing/process_hex_rsp.py``. In MOXIE the
two RSP channels live in one interleaved CSV and are split by column; here each
channel (thoracic / abdominal) is a separate WAV array, so this processes a
single channel and the caller invokes it once per channel. The NeuroKit2 call
is identical:

    nk.rsp_process(signal, sampling_rate=128, method="khodadad2018")
        -> RSP_Clean, RSP_Amplitude, RSP_Rate, RSP_Phase,
           RSP_Peaks, RSP_Troughs, RSP_RVT, ...

Output columns are prefixed with ``respiration_<channel>_`` to match the
Hexoskin naming the MOXIE feature utils expect (e.g.
``respiration_abdominal_RSP_Rate``), so ``features_rsp`` can reuse that logic.
"""

import logging

import neurokit2 as nk
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def process_rsp(
    signal: np.ndarray,
    channel: str,
    sampling_rate: int = 128,
) -> pd.DataFrame | None:
    """Run ``nk.rsp_process`` (khodadad2018) on one RSP channel.

    Parameters
    ----------
    signal:
        Raw respiration samples for the channel.
    channel:
        ``"thoracic"`` or ``"abdominal"`` — used to build the column prefix
        ``respiration_<channel>_``.
    sampling_rate:
        RSP sampling rate in Hz (128 for Hexoskin).

    Returns
    -------
    pd.DataFrame or None
        Per-sample processed frame with channel-prefixed columns, or ``None``
        if processing fails or the signal is too short.
    """
    if signal is None or len(signal) < sampling_rate * 10:
        logger.warning("RSP signal (%s) too short (%s samples) — skipping.",
                       channel, 0 if signal is None else len(signal))
        return None
    try:
        processed, _ = nk.rsp_process(
            signal, sampling_rate=int(sampling_rate), method="khodadad2018"
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("nk.rsp_process failed for %s: %s", channel, exc)
        return None

    prefix = f"respiration_{channel}"
    processed.columns = [f"{prefix}_{c}" for c in processed.columns]
    return processed
