"""End-to-end pipeline: raw IF frames -> per-person vital-sign estimates.

Mirrors the processing chain of the reference paper:
range FFT -> clutter removal -> bin selection -> phase extraction/unwrapping
-> successive phase difference + impulse (Hampel) filtering -> band-pass
separation -> spectral / time-domain rate estimation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AnalysisConfig, RadarConfig, ScenarioConfig
from .estimation.spectral import estimate_rate_spectral
from .estimation.tracker import sliding_rate
from .processing.bin_selection import select_target_bins
from .processing.clutter import remove_static_clutter
from .processing.filters import bandpass, hampel
from .processing.phase import (
    extract_phase,
    phase_difference,
    phase_to_displacement,
    unwrap_phase,
)
from .processing.range_fft import range_fft
from .simulate.radar import SimulationResult, simulate


@dataclass
class PersonEstimate:
    bin_index: int
    range_m: float
    phase_wrapped: np.ndarray     # raw wrapped phase [rad]
    displacement_m: np.ndarray    # unwrapped phase as displacement, demeaned
    breath_signal: np.ndarray     # displacement band-passed to breathing band
    phase_diff: np.ndarray        # Hampel-cleaned successive phase difference
    heart_signal: np.ndarray      # phase difference band-passed to heart band
    rr_hz: float
    hr_hz: float
    track_times_s: np.ndarray
    rr_track_hz: np.ndarray
    hr_track_hz: np.ndarray


@dataclass
class PipelineResult:
    sim: SimulationResult
    profiles: np.ndarray
    profiles_clutter_removed: np.ndarray
    range_axis_m: np.ndarray
    motion_energy: np.ndarray
    estimates: list[PersonEstimate]


def run_pipeline(
    radar: RadarConfig, analysis: AnalysisConfig, scenario: ScenarioConfig
) -> PipelineResult:
    """Simulate a scenario and process it (convenience wrapper)."""
    sim = simulate(radar, scenario)
    return process(sim, radar, analysis, n_targets=len(scenario.persons))


def process(
    sim: SimulationResult,
    radar: RadarConfig,
    analysis: AnalysisConfig,
    n_targets: int = 1,
) -> PipelineResult:
    fs = radar.frame_rate_hz
    profiles, range_axis = range_fft(sim.adc, radar)
    clean = remove_static_clutter(profiles)
    bins, energy = select_target_bins(
        clean,
        range_axis,
        analysis.min_range_m,
        analysis.max_range_m,
        n_targets=n_targets,
    )

    estimates: list[PersonEstimate] = []
    for b in bins:
        wrapped = extract_phase(profiles[:, b])
        unwrapped = unwrap_phase(wrapped)
        disp = phase_to_displacement(unwrapped, radar.wavelength_m)
        disp = disp - disp.mean()

        breath = bandpass(disp, fs, *analysis.breathing_band_hz)
        dphi = hampel(
            phase_difference(unwrapped),
            window=analysis.hampel_window,
            n_sigma=analysis.hampel_n_sigma,
        )
        heart = bandpass(dphi, fs, *analysis.heart_band_hz)

        rr_hz = estimate_rate_spectral(breath, fs, analysis.breathing_band_hz)
        hr_hz = estimate_rate_spectral(heart, fs, analysis.heart_band_hz)

        track_t, rr_track = sliding_rate(
            breath, fs, analysis.breathing_band_hz, analysis.window_s, analysis.hop_s
        )
        _, hr_track = sliding_rate(
            heart, fs, analysis.heart_band_hz, analysis.window_s, analysis.hop_s
        )

        estimates.append(
            PersonEstimate(
                bin_index=int(b),
                range_m=float(range_axis[b]),
                phase_wrapped=wrapped,
                displacement_m=disp,
                breath_signal=breath,
                phase_diff=dphi,
                heart_signal=heart,
                rr_hz=rr_hz,
                hr_hz=hr_hz,
                track_times_s=track_t,
                rr_track_hz=rr_track,
                hr_track_hz=hr_track,
            )
        )

    return PipelineResult(
        sim=sim,
        profiles=profiles,
        profiles_clutter_removed=clean,
        range_axis_m=range_axis,
        motion_energy=energy,
        estimates=estimates,
    )
