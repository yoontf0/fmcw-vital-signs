"""Noise suppression and band selection filters."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt


def bandpass(
    x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def hampel(x: np.ndarray, window: int = 7, n_sigma: float = 3.0) -> np.ndarray:
    """Hampel filter: replace impulse-like outliers with the local median.

    A sample is an outlier when it deviates from the rolling median by more
    than `n_sigma` robust standard deviations (1.4826 * MAD).
    """
    med = median_filter(x, size=window, mode="nearest")
    dev = np.abs(x - med)
    mad = median_filter(dev, size=window, mode="nearest")
    threshold = n_sigma * 1.4826 * mad + 1e-12
    return np.where(dev > threshold, med, x)
