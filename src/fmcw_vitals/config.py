"""Configuration dataclasses and YAML loaders."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

C = 299_792_458.0  # speed of light [m/s]


@dataclass(frozen=True)
class RadarConfig:
    fc_hz: float
    bandwidth_hz: float
    n_samples: int
    fs_hz: float
    frame_rate_hz: float
    duration_s: float

    @property
    def wavelength_m(self) -> float:
        return C / self.fc_hz

    @property
    def chirp_duration_s(self) -> float:
        return self.n_samples / self.fs_hz

    @property
    def slope_hz_per_s(self) -> float:
        return self.bandwidth_hz / self.chirp_duration_s

    @property
    def range_resolution_m(self) -> float:
        return C / (2.0 * self.bandwidth_hz)

    @property
    def n_frames(self) -> int:
        return round(self.duration_s * self.frame_rate_hz)


@dataclass(frozen=True)
class AnalysisConfig:
    min_range_m: float = 0.3
    max_range_m: float = 3.0
    breathing_band_hz: tuple[float, float] = (0.1, 0.5)
    heart_band_hz: tuple[float, float] = (0.8, 2.0)
    hampel_window: int = 7
    hampel_n_sigma: float = 3.0
    window_s: float = 20.0
    hop_s: float = 1.0


@dataclass(frozen=True)
class PersonConfig:
    range_m: float
    breathing_rate_bpm: float
    breathing_amp_mm: float
    heart_rate_bpm: float
    heart_amp_mm: float
    rcs: float = 1.0
    hrv_depth: float = 0.02   # fractional heart-rate modulation depth
    hrv_freq_hz: float = 0.05


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    snr_db: float
    persons: list[PersonConfig] = field(default_factory=list)
    seed: int = 0


def load_radar_config(path: str | Path) -> tuple[RadarConfig, AnalysisConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    radar = RadarConfig(**raw["radar"])
    analysis = dict(raw.get("analysis", {}))
    for key in ("breathing_band_hz", "heart_band_hz"):
        if key in analysis:
            analysis[key] = tuple(analysis[key])
    return radar, AnalysisConfig(**analysis)


def load_scenario(path: str | Path) -> ScenarioConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    persons = [PersonConfig(**p) for p in raw.pop("persons")]
    return ScenarioConfig(persons=persons, **raw)
