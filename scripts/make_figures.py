"""Regenerate every figure used in the README (both scenarios + SNR sweep)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_pipeline import resolve_scenario, run_scenario
from run_snr_sweep import run_sweep


def main() -> None:
    outdir = ROOT / "results" / "figures"
    radar = ROOT / "configs" / "radar_77ghz.yaml"
    for name in ("single_person", "multi_person"):
        run_scenario(radar, resolve_scenario(name), outdir)
    run_sweep(outdir)


if __name__ == "__main__":
    main()
