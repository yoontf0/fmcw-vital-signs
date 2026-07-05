"""FMCW IF-signal simulator: chest displacement -> raw complex ADC frames.

For a point target at instantaneous range R(t), the dechirped (IF) signal of
one chirp is

    x(n) = a * exp(j * (2*pi * f_b * n/fs + phi)),
    f_b  = 2 * S * R / c          (beat frequency, S = chirp slope)
    phi  = 4 * pi * R / lambda    (range-proportional phase)

Breathing/heartbeat chest motion of mm~um scale barely moves f_b but is
amplified in phi — the core idea of phase-based vital sign sensing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import C, RadarConfig, ScenarioConfig
from .chest_model import chest_displacement


@dataclass
class SimulationResult:
    adc: np.ndarray          # (n_frames, n_samples) complex baseband IF signal
    slow_time_s: np.ndarray  # (n_frames,) frame timestamps
    truth: list[dict]        # per person: PersonConfig, displacement, true rates


def simulate(radar: RadarConfig, scenario: ScenarioConfig) -> SimulationResult:
    """Synthesize raw IF frames for every person in the scenario.

    `scenario.snr_db` is the per-sample SNR of a unit-RCS target at 1 m; the
    range FFT adds ~10*log10(n_samples) dB of coherent processing gain.
    """
    rng = np.random.default_rng(scenario.seed)
    t_slow = np.arange(radar.n_frames) / radar.frame_rate_hz
    t_fast = np.arange(radar.n_samples) / radar.fs_hz

    adc = np.zeros((radar.n_frames, radar.n_samples), dtype=np.complex128)
    truth: list[dict] = []
    for person in scenario.persons:
        disp, rates = chest_displacement(t_slow, person)
        rng_m = person.range_m + disp
        amp = person.rcs / rng_m**2
        f_beat = 2.0 * radar.slope_hz_per_s * rng_m / C
        phi = 4.0 * np.pi * rng_m / radar.wavelength_m
        adc += amp[:, None] * np.exp(
            1j * (2.0 * np.pi * f_beat[:, None] * t_fast[None, :] + phi[:, None])
        )
        truth.append({"person": person, "displacement_m": disp, **rates})

    noise_std = 10.0 ** (-scenario.snr_db / 20.0)
    noise = rng.standard_normal(adc.shape) + 1j * rng.standard_normal(adc.shape)
    adc += noise_std * noise / np.sqrt(2.0)
    return SimulationResult(adc=adc, slow_time_s=t_slow, truth=truth)
