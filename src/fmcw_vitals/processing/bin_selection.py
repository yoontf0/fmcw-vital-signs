"""Target range-bin selection based on slow-time motion energy."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def select_target_bins(
    profiles_clutter_removed: np.ndarray,
    range_axis: np.ndarray,
    min_range_m: float,
    max_range_m: float,
    n_targets: int = 1,
    min_separation_m: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the range bins with the strongest slow-time motion energy.

    Static clutter must already be removed so that the remaining energy per
    bin measures *motion* (breathing chest), not reflector strength.

    Returns (sorted bin indices, motion-energy profile over all bins).
    """
    energy = np.mean(np.abs(profiles_clutter_removed) ** 2, axis=0)
    mask = (range_axis >= min_range_m) & (range_axis <= max_range_m)
    masked = np.where(mask, energy, 0.0)

    dr = range_axis[1] - range_axis[0]
    distance = max(1, int(round(min_separation_m / dr)))
    peaks, _ = find_peaks(masked, distance=distance)
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(masked))])

    order = np.argsort(masked[peaks])[::-1]
    selected = peaks[order[:n_targets]]
    return np.sort(selected), energy
