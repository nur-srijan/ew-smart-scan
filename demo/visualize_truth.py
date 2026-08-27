"""
demo/visualize_truth.py — Phase 1 sanity check with richer emitter params.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from ew_sim.emitters import (
    FixedFrequencyEmitter, FHSSEmitter, ScanningEmitter
)
from ew_sim.truth_engine import TruthEngine

OUT = Path(__file__).parent.parent / "notebooks"

def main():
    print("=" * 60)
    print("  EW Smart Scan — Phase 1 Sanity Check")
    print("=" * 60)

    # Use dwell_us=1000 so each slot = 1.05 ms — makes pulses visible on plot.
    # PRIs are set to multiples of this slot size.
    engine = TruthEngine(K=35, T=2000, dwell_us=1000, switch_us=50)

    engine.add_emitters([
        # E0: Fixed-frequency, 1-in-5 duty (PRF 200 Hz, PW 1 ms)
        FixedFrequencyEmitter(
            emitter_id=0, band_index=4,
            pri_sec=5.25e-3,     # ~5 slots PRI
            pulse_width=1.05e-3, # 1 slot pulse width
            rng=np.random.default_rng(1),
        ),
        # E1: Fixed-frequency, 1-in-10 duty, jittered
        FixedFrequencyEmitter(
            emitter_id=1, band_index=18,
            pri_sec=10.5e-3,
            pulse_width=1.05e-3,
            pri_jitter=0.05,
            rng=np.random.default_rng(2),
        ),
        # E2: FHSS across 5 bands, hops every 10 slots
        FHSSEmitter(
            emitter_id=2,
            hop_bands=[7, 10, 14, 20, 26],
            hop_interval=10.5e-3,
            pri_sec=3.15e-3,
            pulse_width=1.05e-3,
            burst_size=3,
            rng=np.random.default_rng(3),
        ),
        # E3: FHSS across 6 bands, faster hopping
        FHSSEmitter(
            emitter_id=3,
            hop_bands=[2, 5, 12, 19, 29, 33],
            hop_interval=6.3e-3,
            pri_sec=2.1e-3,
            pulse_width=1.05e-3,
            burst_size=2,
            rng=np.random.default_rng(4),
        ),
        # E4: Scanning radar — T_scan=2.1s = 2000 slots, beamwidth=10° → TOT=58 slots
        ScanningEmitter(
            emitter_id=4, band_index=11,
            T_scan_sec=2.1,
            beamwidth_deg=10.0,  # wider beam for visibility
            pri_sec=2.1e-3,
            pulse_width=1.05e-3,
            initial_angle=0.0,
            gain_threshold=0.3,
            rng=np.random.default_rng(5),
        ),
    ])

    engine.build(verbose=True)

    print("\nPer-band occupancy (active bands only):")
    occ = engine.occupancy_per_band()
    for k, o in enumerate(occ):
        if o > 0:
            bar = "█" * int(o * 50)
            print(f"  Band {k:2d} ({engine.band_centres[k]:.1f} GHz): {bar} {o*100:.1f}%")

    engine.plot_waterfall(
        t_start=0, t_end=600,
        title="RF Ground Truth — Spectrogram Waterfall (Phase 1 Demo)",
        save_path=OUT / "waterfall.png",
    )
    engine.plot_occupancy_bar(save_path=OUT / "occupancy.png")

    print(f"\nPlots saved → {OUT}/")
    print("Phase 1 ✅  complete")

if __name__ == "__main__":
    main()
