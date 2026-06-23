"""
signal_loader.py
-----------------
Locate and load raw Hexoskin signals for the 2Inspire NeuroKit pipeline.

This mirrors the proven folder-resolution logic from
``code/feature_extraction/hexoskin_loader.py::HexoskinDataLoader`` (manifest
lookup -> ``range_{id}-datatype_4096-1006`` -> ``record_*`` glob -> WAV via
soundfile) but is self-contained so it can be run as a plain script without
pulling in the BreathingSession/data_models chain.

Raw signals (verified from WAV headers + 2Inspire CLAUDE.md):
    record_*/ECG_I.wav                 ECG, 256 Hz
    record_*/respiration_thoracic.wav  thoracic RSP, 128 Hz
    record_*/respiration_abdominal.wav abdominal RSP, 128 Hz
    <session>/RR_interval.csv          device RR-intervals, seconds (last col)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf

logger = logging.getLogger(__name__)

# Fallback sampling rates if the WAV header is missing/odd (verified values).
ECG_FS = 256
RSP_FS = 128

ECG_FILE = "ECG_I.wav"
RSP_FILES = {
    "thoracic": "respiration_thoracic.wav",
    "abdominal": "respiration_abdominal.wav",
}


class HexLoader:
    """Resolve participant/session -> Hexoskin folder and read raw signals."""

    def __init__(self, hexoskin_path: str, manifest_path: str):
        self.hexoskin_path = Path(hexoskin_path).expanduser().resolve()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.hexoskin_path.is_dir():
            raise FileNotFoundError(f"Hexoskin dir not found: {self.hexoskin_path}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

    # ------------------------------------------------------------------
    # Folder resolution (mirrors HexoskinDataLoader)
    # ------------------------------------------------------------------
    def participants(self) -> list[dict]:
        return self.manifest.get("participants", [])

    def hexoskin_id(self, participant_id: int, session_num: int) -> Optional[str]:
        for p in self.participants():
            if p["participant_id"] == participant_id:
                hid = p["sensor_data"]["hexoskin_sessions"].get(f"session_{session_num}")
                if hid and str(hid) != "None":
                    s = str(hid)
                    return str(int(float(s))) if "." in s else s
        return None

    def session_folder(self, participant_id: int, session_num: int) -> Optional[Path]:
        hid = self.hexoskin_id(participant_id, session_num)
        if not hid:
            return None
        folder = self.hexoskin_path / f"range_{hid}-datatype_4096-1006"
        return folder if folder.is_dir() else None

    @staticmethod
    def _record_dir(session_folder: Path) -> Optional[Path]:
        recs = sorted(session_folder.glob("record_*"))
        return recs[0] if recs else None

    # ------------------------------------------------------------------
    # Raw signal readers
    # ------------------------------------------------------------------
    def load_ecg(self, session_folder: Path) -> tuple[Optional[np.ndarray], int]:
        """Return (ecg_signal, fs). ``(None, ECG_FS)`` if the WAV is absent."""
        rec = self._record_dir(session_folder)
        if rec is None:
            return None, ECG_FS
        path = rec / ECG_FILE
        if not path.exists():
            logger.warning("No %s in %s", ECG_FILE, rec)
            return None, ECG_FS
        try:
            sig, fs = sf.read(str(path))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read %s: %s", path, exc)
            return None, ECG_FS
        sig = np.asarray(sig, dtype=float).ravel()
        return sig, int(fs) if fs else ECG_FS

    def load_rsp(self, session_folder: Path, channel: str) -> tuple[Optional[np.ndarray], int]:
        """Return (rsp_signal, fs) for ``channel`` in {'thoracic','abdominal'}."""
        rec = self._record_dir(session_folder)
        if rec is None:
            return None, RSP_FS
        path = rec / RSP_FILES[channel]
        if not path.exists():
            logger.warning("No %s in %s", RSP_FILES[channel], rec)
            return None, RSP_FS
        try:
            sig, fs = sf.read(str(path))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read %s: %s", path, exc)
            return None, RSP_FS
        sig = np.asarray(sig, dtype=float).ravel()
        return sig, int(fs) if fs else RSP_FS

    def load_rri_ms(self, session_folder: Path) -> Optional[np.ndarray]:
        """Return device RR-intervals in milliseconds (physiological 300-2000 ms).

        Reads the *parent-level* ``RR_interval.csv`` (seconds in the last
        column) — the same source used by
        ``eda_open_exploration.py::hrv_from_session``.
        """
        rri_csv = session_folder / "RR_interval.csv"
        if not rri_csv.exists():
            return None
        try:
            df = pd.read_csv(rri_csv)
            rri_s = df.iloc[:, -1].to_numpy(dtype=float)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RR_interval.csv read failed (%s): %s", rri_csv, exc)
            return None
        rri_ms = rri_s * 1000.0
        rri_ms = rri_ms[(rri_ms > 300) & (rri_ms < 2000)]
        return rri_ms
