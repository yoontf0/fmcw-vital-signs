"""Run the FMCW vital-signs pipeline on a simulated scenario and save all figures.

Usage:
    python scripts/run_pipeline.py                          # single_person scenario
    python scripts/run_pipeline.py --scenario multi_person
    python scripts/run_pipeline.py --scenario path/to/custom.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fmcw_vitals.config import load_radar_config, load_scenario
from fmcw_vitals.estimation.spectral import band_spectrum
from fmcw_vitals.pipeline import run_pipeline
from fmcw_vitals.viz import plots
from fmcw_vitals.viz.style import apply_style


def resolve_scenario(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    candidate = ROOT / "configs" / "scenarios" / f"{arg}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"scenario not found: {arg}")


def _save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def run_scenario(radar_path: Path, scenario_path: Path, outdir: Path):
    radar, analysis = load_radar_config(radar_path)
    scenario = load_scenario(scenario_path)
    result = run_pipeline(radar, analysis, scenario)

    figdir = outdir / scenario.name
    figdir.mkdir(parents=True, exist_ok=True)
    apply_style()

    t = result.sim.slow_time_s
    fs = radar.frame_rate_hz
    # ascending selected bins line up with persons sorted by range
    truths = sorted(result.sim.truth, key=lambda d: d["person"].range_m)

    print(f"\n=== scenario: {scenario.name} (SNR {scenario.snr_db:+.0f} dB, "
          f"{radar.duration_s:.0f} s @ {fs:.0f} Hz) ===")

    _save(
        plots.plot_range_time(
            result.profiles_clutter_removed, result.range_axis_m, t,
            selected_bins=[e.bin_index for e in result.estimates],
            max_range_m=analysis.max_range_m + 0.5,
        ),
        figdir / "01_range_time_map.png",
    )
    _save(
        plots.plot_motion_energy(
            result.range_axis_m, result.motion_energy,
            selected_bins=[e.bin_index for e in result.estimates],
            min_range_m=analysis.min_range_m, max_range_m=analysis.max_range_m,
        ),
        figdir / "02_motion_energy.png",
    )

    multi = len(result.estimates) > 1
    for i, (est, truth) in enumerate(zip(result.estimates, truths)):
        person = truth["person"]
        rr_true = float(person.breathing_rate_bpm)
        hr_true = float(np.mean(truth["hr_bpm"]))
        print(
            f"target @ {est.range_m:.2f} m (true {person.range_m:.2f} m) | "
            f"RR {est.rr_hz * 60.0:5.1f} BPM (true {rr_true:.1f}) | "
            f"HR {est.hr_hz * 60.0:5.1f} BPM (true {hr_true:.1f})"
        )

        prefix = f"person{i + 1}_" if multi else ""
        _save(
            plots.plot_phase_pipeline(
                t, est.phase_wrapped, est.displacement_m * 1e3,
                est.phase_diff, est.heart_signal,
                title=f"Slow-time phase processing chain — target @ {est.range_m:.2f} m",
            ),
            figdir / f"03_{prefix}phase_pipeline.png",
        )

        bf, bs = band_spectrum(est.breath_signal, fs, analysis.breathing_band_hz)
        hf, hs = band_spectrum(est.heart_signal, fs, analysis.heart_band_hz)
        _save(
            plots.plot_spectra(bf, bs, hf, hs, est.rr_hz, est.hr_hz,
                               rr_true_bpm=rr_true, hr_true_bpm=hr_true),
            figdir / f"04_{prefix}spectra.png",
        )
        _save(
            plots.plot_tracking(est.track_times_s, est.rr_track_hz, est.hr_track_hz,
                                t, truth["rr_bpm"], truth["hr_bpm"]),
            figdir / f"05_{prefix}rate_tracking.png",
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radar", default=str(ROOT / "configs" / "radar_77ghz.yaml"))
    parser.add_argument("--scenario", default="single_person",
                        help="scenario name in configs/scenarios/ or a YAML path")
    parser.add_argument("--outdir", default=str(ROOT / "results" / "figures"))
    args = parser.parse_args()
    run_scenario(Path(args.radar), resolve_scenario(args.scenario), Path(args.outdir))


if __name__ == "__main__":
    main()
