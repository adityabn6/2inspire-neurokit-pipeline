"""
numpy_compat.py
----------------
NumPy >= 2.0 compatibility shim for neurokit2.

neurokit2 (0.2.x) still references ``np.trapz``, which was removed in NumPy 2.0.
Call ``apply_numpy_compat()`` once before any ``nk.*`` function is invoked.

Ported verbatim from the MOXIE Pipeline
(``src/processing/process_hex_ecg.py::_apply_numpy_compat``).
"""

import numpy as np


def apply_numpy_compat() -> None:
    """Patch ``np.trapz`` back in for neurokit2 on NumPy >= 2.0."""
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]
