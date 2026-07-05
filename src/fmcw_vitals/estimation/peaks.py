"""Time-domain rate estimation from inter-peak intervals."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def estimate_rate_peaks(
    x: np.ndarray, fs: float, band: tuple[float, float]
) -> float:
    """Median inter-peak-interval rate [Hz]; expects an already band-passed signal.

    Complements the spectral estimate: robust to a broadened spectral peak
    (e.g. heart-rate variability) but more sensitive to missed/spurious peaks.
    Returns NaN when fewer than two peaks are found.
    """
    min_distance = max(1, int(round(fs / band[1] * 0.7)))
    peaks, _ = find_peaks(x, distance=min_distance)
    if peaks.size < 2:
        return float("nan")
    intervals_s = np.diff(peaks) / fs
    return float(1.0 / np.median(intervals_s))
