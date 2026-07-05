"""Time-resolved rate tracking with a sliding window."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.signal import medfilt

from .spectral import estimate_rate_spectral


def sliding_rate(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float],
    window_s: float,
    hop_s: float,
    estimator: Callable = estimate_rate_spectral,
    smooth: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Windowed rate estimates over time. Returns (window-center times [s], rates [Hz])."""
    n_win = min(int(round(window_s * fs)), x.size)
    n_hop = max(1, int(round(hop_s * fs)))
    times, rates = [], []
    for start in range(0, x.size - n_win + 1, n_hop):
        seg = x[start : start + n_win]
        times.append((start + n_win / 2.0) / fs)
        rates.append(estimator(seg, fs, band))
    rates_arr = np.asarray(rates, dtype=float)
    if smooth > 1 and rates_arr.size >= smooth:
        kernel = smooth if smooth % 2 == 1 else smooth + 1
        rates_arr = medfilt(rates_arr, kernel_size=kernel)
    return np.asarray(times), rates_arr
