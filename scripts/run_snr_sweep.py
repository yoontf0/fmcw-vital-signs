"""Monte-Carlo SNR sweep: estimation RMSE vs. raw-signal SNR.

Compares the zoom-FFT spectral estimator against time-domain peak counting
for heart rate, and shows where phase-based sensing breaks down as noise grows.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fmcw_vitals.config import load_radar_config, load_scenario
from fmcw_vitals.estimation.peaks import estimate_rate_peaks
from fmcw_vitals.pipeline import run_pipeline
from fmcw_vitals.viz import plots
from fmcw_vitals.viz.style import apply_style

SNR_DBS = [-30, -25, -20, -15, -10, -5, 0, 5]
N_TRIALS = 5


def _rmse_bpm(errors_hz: list[float]) -> float:
    err = np.asarray(errors_hz, dtype=float) * 60.0
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(err**2))) if err.size else float("nan")


def run_sweep(outdir: Path) -> None:
    radar, analysis = load_radar_config(ROOT / "configs" / "radar_77ghz.yaml")
    base = load_scenario(ROOT / "configs" / "scenarios" / "single_person.yaml")
    person = base.persons[0]
    fs = radar.frame_rate_hz

    rr_spec_rmse, hr_spec_rmse, hr_peak_rmse = [], [], []
    print(f"SNR sweep: {len(SNR_DBS)} levels x {N_TRIALS} trials "
          f"({radar.duration_s:.0f} s each)")
    for snr_idx, snr in enumerate(SNR_DBS):
        rr_err, hr_err, hr_peak_err = [], [], []
        for trial in range(N_TRIALS):
            scenario = replace(base, snr_db=float(snr), seed=1000 * trial + snr_idx)
            result = run_pipeline(radar, analysis, scenario)
            est = result.estimates[0]
            truth_hr_hz = float(np.mean(result.sim.truth[0]["hr_bpm"])) / 60.0
            rr_err.append(est.rr_hz - person.breathing_rate_bpm / 60.0)
            hr_err.append(est.hr_hz - truth_hr_hz)
            hr_peak_err.append(
                estimate_rate_peaks(est.heart_signal, fs, analysis.heart_band_hz)
                - truth_hr_hz
            )
        rr_spec_rmse.append(_rmse_bpm(rr_err))
        hr_spec_rmse.append(_rmse_bpm(hr_err))
        hr_peak_rmse.append(_rmse_bpm(hr_peak_err))
        print(f"  SNR {snr:+3d} dB | RR RMSE {rr_spec_rmse[-1]:6.2f} BPM | "
              f"HR RMSE (zoom-FFT) {hr_spec_rmse[-1]:6.2f} | "
              f"HR RMSE (peaks) {hr_peak_rmse[-1]:6.2f}")

    apply_style()
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plots.plot_snr_sweep(
        SNR_DBS,
        rr_curves={"Zoom-FFT": np.asarray(rr_spec_rmse)},
        hr_curves={
            "Zoom-FFT": np.asarray(hr_spec_rmse),
            "Peak counting": np.asarray(hr_peak_rmse),
        },
    )
    path = outdir / "06_snr_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(ROOT / "results" / "figures"))
    args = parser.parse_args()
    run_sweep(Path(args.outdir))


if __name__ == "__main__":
    main()
