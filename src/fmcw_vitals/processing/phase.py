"""Slow-time phase processing: extraction, unwrapping, differencing."""
from __future__ import annotations

import numpy as np


def extract_phase(bin_signal: np.ndarray) -> np.ndarray:
    """Wrapped phase [rad] of the complex slow-time signal at one range bin."""
    return np.angle(bin_signal)


def unwrap_phase(phase: np.ndarray) -> np.ndarray:
    """Remove 2*pi discontinuities to recover continuous displacement phase."""
    return np.unwrap(phase)


def phase_to_displacement(phase: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Convert phase [rad] to radial displacement [m]: d = phi * lambda / (4*pi)."""
    return phase * wavelength_m / (4.0 * np.pi)


def phase_difference(phase: np.ndarray) -> np.ndarray:
    """Successive phase difference (same length as input).

    Acts as a discrete differentiator: attenuates the large low-frequency
    breathing component relative to the heartbeat, so the heartbeat peak is
    no longer buried under breathing energy in the spectrum.
    """
    return np.diff(phase, prepend=phase[0])
