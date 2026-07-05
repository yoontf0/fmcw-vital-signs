"""Fast-time processing: range FFT (spatial isolation of targets)."""
from __future__ import annotations

import numpy as np

from ..config import C, RadarConfig


def range_fft(
    adc: np.ndarray, radar: RadarConfig, window: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """FFT along fast time -> complex range profiles and the range axis [m].

    Returns (profiles of shape (n_frames, n_bins), range_axis of shape (n_bins,)).
    Only positive-beat-frequency bins (first half of the spectrum) are kept.
    """
    n = adc.shape[1]
    win = np.hanning(n) if window else np.ones(n)
    profiles = np.fft.fft(adc * win[None, :], axis=1)[:, : n // 2]
    beat_freqs = np.arange(n // 2) * radar.fs_hz / n
    range_axis = beat_freqs * C / (2.0 * radar.slope_hz_per_s)
    return profiles, range_axis
