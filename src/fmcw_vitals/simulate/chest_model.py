"""Chest-surface displacement model: breathing + heartbeat motion."""
from __future__ import annotations

import numpy as np

from ..config import PersonConfig


def breathing_waveform(t: np.ndarray, rate_hz: float, amp_m: float) -> np.ndarray:
    """Slightly asymmetric breathing motion (fundamental + weak 2nd harmonic)."""
    w = 2.0 * np.pi * rate_hz
    raw = np.sin(w * t) + 0.2 * np.sin(2.0 * w * t - np.pi / 4.0)
    return amp_m * raw / 1.2


def heartbeat_waveform(
    t: np.ndarray,
    rate_hz: float,
    amp_m: float,
    hrv_depth: float,
    hrv_freq_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Heartbeat motion with sinusoidal heart-rate variability (frequency modulation).

    Returns (displacement [m], instantaneous heart rate [Hz]).
    """
    inst_rate_hz = rate_hz * (1.0 + hrv_depth * np.sin(2.0 * np.pi * hrv_freq_hz * t))
    dt = float(t[1] - t[0]) if t.size > 1 else 0.0
    phase = 2.0 * np.pi * np.cumsum(inst_rate_hz) * dt
    raw = np.sin(phase) + 0.3 * np.sin(2.0 * phase)
    return amp_m * raw / 1.3, inst_rate_hz


def chest_displacement(t: np.ndarray, person: PersonConfig) -> tuple[np.ndarray, dict]:
    """Total chest displacement [m] over slow time, plus ground-truth rates."""
    breath = breathing_waveform(
        t, person.breathing_rate_bpm / 60.0, person.breathing_amp_mm * 1e-3
    )
    heart, inst_hr_hz = heartbeat_waveform(
        t,
        person.heart_rate_bpm / 60.0,
        person.heart_amp_mm * 1e-3,
        person.hrv_depth,
        person.hrv_freq_hz,
    )
    truth = {
        "rr_bpm": np.full_like(t, person.breathing_rate_bpm),
        "hr_bpm": inst_hr_hz * 60.0,
    }
    return breath + heart, truth
