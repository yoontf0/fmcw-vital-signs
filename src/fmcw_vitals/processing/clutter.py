"""Static clutter suppression along slow time."""
from __future__ import annotations

import numpy as np


def remove_static_clutter(profiles: np.ndarray) -> np.ndarray:
    """Subtract the slow-time mean of each range bin.

    Stationary reflectors (walls, furniture) contribute a constant complex
    value per bin; removing the mean leaves only time-varying (moving) energy.
    """
    return profiles - profiles.mean(axis=0, keepdims=True)
