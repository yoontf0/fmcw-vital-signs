"""Frequency-domain rate estimation with fine in-band resolution."""
from __future__ import annotations

import numpy as np


def band_spectrum(
    x: np.ndarray, fs: float, band: tuple[float, float], n_points: int = 2001
) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude spectrum restricted to `band` [Hz] with fine frequency spacing.

    Uses the chirp-z based zoom FFT when available (scipy >= 1.8), otherwise a
    heavily zero-padded FFT. The zoom transform evaluates the DTFT only inside
    the band of interest, giving far finer spacing than 1/T without the cost
    of a huge FFT.
    """
    x = np.asarray(x, dtype=float)
    xw = (x - x.mean()) * np.hanning(x.size)
    try:
        from scipy.signal import zoom_fft
    except ImportError:
        n_fft = int(2 ** np.ceil(np.log2(max(x.size * 16, 1024))))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
        spec = np.abs(np.fft.rfft(xw, n_fft))
        sel = (freqs >= band[0]) & (freqs <= band[1])
        return freqs[sel], spec[sel]
    freqs = np.linspace(band[0], band[1], n_points)
    spec = np.abs(zoom_fft(xw, list(band), m=n_points, fs=fs, endpoint=True))
    return freqs, spec


def estimate_rate_spectral(
    x: np.ndarray, fs: float, band: tuple[float, float]
) -> float:
    """Dominant frequency [Hz] in `band`, refined by parabolic interpolation."""
    freqs, spec = band_spectrum(x, fs, band)
    k = int(np.argmax(spec))
    if 0 < k < spec.size - 1:
        denom = spec[k - 1] - 2.0 * spec[k] + spec[k + 1]
        if abs(denom) > 1e-12:
            delta = float(np.clip(0.5 * (spec[k - 1] - spec[k + 1]) / denom, -0.5, 0.5))
            return float(freqs[k] + delta * (freqs[1] - freqs[0]))
    return float(freqs[k])
