"""Estimation accuracy metrics."""
from __future__ import annotations

import numpy as np


def bpm_errors(est_hz, true_hz) -> dict[str, float]:
    """MAE / RMSE / bias in beats(or breaths)-per-minute."""
    err = (np.atleast_1d(est_hz) - np.atleast_1d(true_hz)) * 60.0
    return {
        "mae_bpm": float(np.mean(np.abs(err))),
        "rmse_bpm": float(np.sqrt(np.mean(err**2))),
        "bias_bpm": float(np.mean(err)),
    }
