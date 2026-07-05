import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fmcw_vitals.config import (
    AnalysisConfig,
    PersonConfig,
    RadarConfig,
    ScenarioConfig,
)
from fmcw_vitals.estimation.spectral import estimate_rate_spectral
from fmcw_vitals.pipeline import run_pipeline
from fmcw_vitals.processing.filters import bandpass

RADAR = RadarConfig(
    fc_hz=77e9, bandwidth_hz=3.6e9, n_samples=256, fs_hz=5e6,
    frame_rate_hz=20.0, duration_s=40.0,
)
ANALYSIS = AnalysisConfig()
PERSON = PersonConfig(
    range_m=0.8, breathing_rate_bpm=15.0, breathing_amp_mm=4.0,
    heart_rate_bpm=72.0, heart_amp_mm=0.35,
)


def test_recovers_vital_rates():
    scenario = ScenarioConfig(name="test", snr_db=0.0, persons=[PERSON], seed=1)
    result = run_pipeline(RADAR, ANALYSIS, scenario)
    est = result.estimates[0]
    assert abs(est.range_m - PERSON.range_m) < 2 * RADAR.range_resolution_m
    assert abs(est.rr_hz * 60.0 - PERSON.breathing_rate_bpm) < 1.0
    assert abs(est.hr_hz * 60.0 - PERSON.heart_rate_bpm) < 2.0


def test_two_person_separation():
    other = PersonConfig(
        range_m=1.5, breathing_rate_bpm=20.0, breathing_amp_mm=3.0,
        heart_rate_bpm=90.0, heart_amp_mm=0.30,
    )
    scenario = ScenarioConfig(name="test2", snr_db=0.0, persons=[PERSON, other], seed=2)
    result = run_pipeline(RADAR, ANALYSIS, scenario)
    assert len(result.estimates) == 2
    ranges = sorted(e.range_m for e in result.estimates)
    assert abs(ranges[0] - 0.8) < 0.1
    assert abs(ranges[1] - 1.5) < 0.1
    rates = sorted(e.rr_hz * 60.0 for e in result.estimates)
    assert abs(rates[0] - 15.0) < 1.5
    assert abs(rates[1] - 20.0) < 1.5


def test_bandpass_isolates_heart_band():
    fs = 20.0
    t = np.arange(0.0, 60.0, 1.0 / fs)
    x = np.sin(2 * np.pi * 0.25 * t) + 0.3 * np.sin(2 * np.pi * 1.2 * t)
    heart = bandpass(x, fs, 0.8, 2.0)
    f = estimate_rate_spectral(heart, fs, (0.8, 2.0))
    assert abs(f - 1.2) < 0.02
