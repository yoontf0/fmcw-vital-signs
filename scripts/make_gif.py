"""Build the README demo GIF: live range profile, scrolling vitals, BPM readout.

Renders the single-person scenario as if the radar were running in real time:
the instantaneous range profile fluctuates with the chest motion while the
displacement / heartbeat waveforms scroll past and the rate readouts update.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fmcw_vitals.config import load_radar_config, load_scenario
from fmcw_vitals.pipeline import run_pipeline
from fmcw_vitals.viz.style import COLORS, apply_style

WINDOW_S = 15.0          # visible scroll window
T_START, T_END = 16.0, 60.0
N_GIF_FRAMES = 110
FPS = 12
DPI = 72


def main() -> None:
    radar, analysis = load_radar_config(ROOT / "configs" / "radar_77ghz.yaml")
    scenario = load_scenario(ROOT / "configs" / "scenarios" / "single_person.yaml")
    result = run_pipeline(radar, analysis, scenario)
    est = result.estimates[0]

    t = result.sim.slow_time_s
    fs = radar.frame_rate_hz
    range_axis = result.range_axis_m
    rmask = range_axis <= analysis.max_range_m
    profile_db = 20.0 * np.log10(np.abs(result.profiles_clutter_removed) + 1e-12)
    disp_mm = est.displacement_m * 1e3
    heart = est.heart_signal

    apply_style()
    fig = plt.figure(figsize=(9.5, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.3],
                          left=0.07, right=0.98, top=0.86, bottom=0.1,
                          hspace=0.55, wspace=0.28)
    ax_rng = fig.add_subplot(gs[:, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[1, 1])

    vmax = float(profile_db[:, rmask].max())
    (line_rng,) = ax_rng.plot(profile_db[0, rmask], range_axis[rmask],
                              color=COLORS["raw"], lw=1.2)
    ax_rng.axhline(est.range_m, color=COLORS["heart"], ls="--", lw=1.2)
    ax_rng.text(vmax - 48, est.range_m + 0.06, f"target {est.range_m:.2f} m",
                color=COLORS["heart"], fontsize=9)
    ax_rng.set(xlim=(vmax - 50, vmax + 3), ylim=(0, analysis.max_range_m),
               xlabel="Magnitude [dB]", ylabel="Range [m]", title="Live range profile")

    (line_d,) = ax_d.plot([], [], color=COLORS["breath"])
    pad_d = 0.1 * (disp_mm.max() - disp_mm.min())
    ax_d.set(ylim=(disp_mm.min() - pad_d, disp_mm.max() + pad_d),
             ylabel="Displacement [mm]", title="Chest displacement (breathing)")

    (line_h,) = ax_h.plot([], [], color=COLORS["heart"], lw=1.1)
    pad_h = 0.15 * (heart.max() - heart.min())
    ax_h.set(ylim=(heart.min() - pad_h, heart.max() + pad_h),
             ylabel="ΔPhase [rad]", xlabel="Slow time [s]",
             title="Heartbeat (phase difference, band-passed)")

    fig.text(0.07, 0.94, "77 GHz FMCW radar — non-contact vital signs",
             fontsize=13, fontweight="semibold")
    txt_rr = fig.text(0.60, 0.935, "", color=COLORS["breath"],
                      fontsize=14, fontweight="bold", ha="left")
    txt_hr = fig.text(0.79, 0.935, "", color=COLORS["heart"],
                      fontsize=14, fontweight="bold", ha="left")

    def update(i: int):
        tt = T_START + (T_END - T_START) * i / (N_GIF_FRAMES - 1)
        idx = min(int(tt * fs), t.size - 1)
        i0 = max(0, idx - int(WINDOW_S * fs))

        line_rng.set_xdata(profile_db[idx, rmask])
        line_d.set_data(t[i0:idx], disp_mm[i0:idx])
        line_h.set_data(t[i0:idx], heart[i0:idx])
        for ax in (ax_d, ax_h):
            ax.set_xlim(tt - WINDOW_S, tt)

        j = int(np.clip(np.searchsorted(est.track_times_s, tt) - 1,
                        0, est.track_times_s.size - 1))
        txt_rr.set_text(f"RR {est.rr_track_hz[j] * 60.0:5.1f} BPM")
        txt_hr.set_text(f"HR {est.hr_track_hz[j] * 60.0:5.1f} BPM")
        return line_rng, line_d, line_h, txt_rr, txt_hr

    anim = FuncAnimation(fig, update, frames=N_GIF_FRAMES)
    out = ROOT / "results" / "figures" / "demo.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"saved {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
