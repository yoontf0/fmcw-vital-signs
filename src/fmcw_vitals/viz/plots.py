"""Publication-style figures for every pipeline stage. Each function returns a Figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .style import CMAP, COLORS


def _db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(x) + 1e-12)


def plot_range_time(
    profiles: np.ndarray,
    range_axis: np.ndarray,
    times: np.ndarray,
    selected_bins=(),
    max_range_m: float | None = None,
    title: str = "Range–time map (static clutter removed)",
):
    data = _db(profiles).T
    if max_range_m is not None:
        n = int(np.searchsorted(range_axis, max_range_m))
        data, range_axis = data[:n], range_axis[:n]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    vmax = float(data.max())
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap=CMAP,
        extent=[float(times[0]), float(times[-1]), float(range_axis[0]), float(range_axis[-1])],
        vmin=vmax - 50.0,
        vmax=vmax,
        interpolation="nearest",
    )
    for b in selected_bins:
        r = float(range_axis[b]) if b < range_axis.size else None
        if r is None:
            continue
        ax.axhline(r, color="w", ls="--", lw=1.0, alpha=0.9)
        ax.text(
            float(times[-1]) * 0.99, r, f" target @ {r:.2f} m",
            color="w", va="bottom", ha="right", fontsize=9,
        )
    ax.set(xlabel="Slow time [s]", ylabel="Range [m]", title=title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Magnitude [dB]", pad=0.01)
    return fig


def plot_motion_energy(
    range_axis: np.ndarray,
    energy: np.ndarray,
    selected_bins=(),
    min_range_m: float | None = None,
    max_range_m: float | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(range_axis, 10.0 * np.log10(energy + 1e-18), color=COLORS["raw"])
    if min_range_m is not None and max_range_m is not None:
        ax.axvspan(min_range_m, max_range_m, color=COLORS["breath"], alpha=0.08,
                   label="analysis window")
    for i, b in enumerate(selected_bins):
        ax.plot(
            range_axis[b], 10.0 * np.log10(energy[b] + 1e-18), "v",
            color=COLORS["heart"], markersize=9,
            label="selected bin" if i == 0 else None,
        )
    ax.set(xlabel="Range [m]", ylabel="Motion energy [dB]",
           title="Slow-time motion energy vs. range (target bin selection)")
    ax.legend(loc="upper right")
    return fig


def plot_phase_pipeline(
    times: np.ndarray,
    wrapped: np.ndarray,
    displacement_mm: np.ndarray,
    phase_diff: np.ndarray,
    heart_signal: np.ndarray,
    title: str = "Slow-time phase processing chain",
):
    fig, axes = plt.subplots(4, 1, figsize=(10, 9.5), sharex=True)
    axes[0].plot(times, wrapped, color=COLORS["raw"], lw=0.9)
    axes[0].set_ylabel("Phase [rad]")
    axes[0].set_title("(a) Wrapped phase of the selected range bin")

    axes[1].plot(times, displacement_mm, color=COLORS["breath"])
    axes[1].set_ylabel("Displacement [mm]")
    axes[1].set_title("(b) Unwrapped phase → chest displacement (breathing dominates)")

    axes[2].plot(times, phase_diff, color=COLORS["accent"], lw=0.9)
    axes[2].set_ylabel("ΔPhase [rad]")
    axes[2].set_title("(c) Successive phase difference + Hampel filter (breathing suppressed)")

    axes[3].plot(times, heart_signal, color=COLORS["heart"], lw=0.9)
    axes[3].set_ylabel("ΔPhase [rad]")
    axes[3].set_title("(d) Band-pass 0.8–2.0 Hz → heartbeat signal")
    axes[3].set_xlabel("Slow time [s]")

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    return fig


def plot_spectra(
    breath_freqs: np.ndarray,
    breath_spec: np.ndarray,
    heart_freqs: np.ndarray,
    heart_spec: np.ndarray,
    rr_est_hz: float,
    hr_est_hz: float,
    rr_true_bpm: float | None = None,
    hr_true_bpm: float | None = None,
):
    fig, (ax_b, ax_h) = plt.subplots(1, 2, figsize=(11, 4.0))

    for ax, freqs, spec, est_hz, true_bpm, color, name in (
        (ax_b, breath_freqs, breath_spec, rr_est_hz, rr_true_bpm, COLORS["breath"], "Breathing"),
        (ax_h, heart_freqs, heart_spec, hr_est_hz, hr_true_bpm, COLORS["heart"], "Heartbeat"),
    ):
        bpm = freqs * 60.0
        norm = spec / (spec.max() + 1e-18)
        ax.fill_between(bpm, norm, color=color, alpha=0.15)
        ax.plot(bpm, norm, color=color)
        if true_bpm is not None:
            ax.axvline(true_bpm, color=COLORS["truth"], ls="--", lw=1.4,
                       label=f"truth {true_bpm:.1f} BPM")
        ax.axvline(est_hz * 60.0, color=color, ls=":", lw=1.6,
                   label=f"estimate {est_hz * 60.0:.1f} BPM")
        ax.set(xlabel=f"{name} rate [BPM]", ylabel="Normalized amplitude",
               title=f"{name} spectrum (zoom FFT)")
        ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig


def plot_tracking(
    track_times: np.ndarray,
    rr_track_hz: np.ndarray,
    hr_track_hz: np.ndarray,
    slow_time: np.ndarray,
    rr_true_bpm: np.ndarray,
    hr_true_bpm: np.ndarray,
):
    fig, (ax_r, ax_h) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)

    ax_r.plot(slow_time, rr_true_bpm, color=COLORS["truth"], lw=1.6, label="ground truth")
    ax_r.plot(track_times, rr_track_hz * 60.0, "o-", color=COLORS["breath"],
              markersize=3.5, lw=1.0, label="estimate")
    ax_r.set_ylabel("Respiration [BPM]")
    ax_r.set_title("Respiration-rate tracking (20 s sliding window)")
    ax_r.legend(loc="upper right", fontsize=9)

    ax_h.plot(slow_time, hr_true_bpm, color=COLORS["truth"], lw=1.6, label="ground truth")
    ax_h.plot(track_times, hr_track_hz * 60.0, "o-", color=COLORS["heart"],
              markersize=3.5, lw=1.0, label="estimate")
    ax_h.set_ylabel("Heart rate [BPM]")
    ax_h.set_xlabel("Slow time [s]")
    ax_h.set_title("Heart-rate tracking (20 s sliding window)")
    ax_h.legend(loc="upper right", fontsize=9)

    for ax, truth in ((ax_r, rr_true_bpm), (ax_h, hr_true_bpm)):
        center = float(np.mean(truth))
        ax.set_ylim(center - 10.0, center + 10.0)

    fig.tight_layout()
    return fig


def plot_snr_sweep(
    snr_dbs,
    rr_curves: dict[str, np.ndarray],
    hr_curves: dict[str, np.ndarray],
):
    fig, (ax_r, ax_h) = plt.subplots(1, 2, figsize=(11, 4.2))
    markers = ["o", "s", "^", "d"]

    for ax, curves, name in ((ax_r, rr_curves, "Respiration"), (ax_h, hr_curves, "Heart rate")):
        for (label, rmse), marker in zip(curves.items(), markers):
            ax.semilogy(snr_dbs, rmse, marker + "-", label=label)
        ax.set(xlabel="Raw-signal SNR [dB]", ylabel="RMSE [BPM]",
               title=f"{name} RMSE vs. SNR")
        ax.legend(fontsize=9)

    fig.tight_layout()
    return fig
