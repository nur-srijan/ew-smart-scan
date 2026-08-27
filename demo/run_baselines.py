"""
demo/run_baselines.py
=====================
Benchmarks classical Electronic Warfare receiver scan strategies:
    1. Sequential Sweep (Traditional Open-Loop)
    2. Pseudo-Random Sweep (Costas Permutation)
    3. Priority Queue Sweep (Static EDB with partial knowledge)
    4. Uniform Random Sweep

Runs a Monte Carlo evaluation across 20 independent tactical episodes and
generates comparative Figures of Merit (IR, TTI, Discovery Curves).

Usage:
    uv run demo/run_baselines.py
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from schedulers.baselines import (
    SequentialSweep,
    PseudoRandomSweep,
    PriorityQueueSweep,
    UniformRandomSweep,
)
from eval.runner import MonteCarloRunner

OUT = Path(__file__).parent.parent / "notebooks"


def main():
    print("=" * 70)
    print("  EW Smart Scan — Phase 2: Classical Baselines & FoM Benchmark")
    print("=" * 70)

    K = 35
    T = 2000
    n_episodes = 20

    # 1. Instantiate the classical schedulers
    # Give priority queue higher weight to known radar bands (e.g. bands 4, 11, 18)
    edb_priorities = np.ones(K) * 0.5
    edb_priorities[4] = 3.0   # Fixed radar
    edb_priorities[11] = 4.0  # Scanning radar
    edb_priorities[18] = 2.5  # Fixed radar

    schedulers = [
        SequentialSweep(K=K),
        PseudoRandomSweep(K=K, seed=42),
        PriorityQueueSweep(K=K, priority_weights=edb_priorities, seed=42),
        UniformRandomSweep(K=K, seed=42),
    ]

    # 2. Run Monte Carlo benchmark
    runner = MonteCarloRunner(
        K=K,
        T=T,
        dwell_us=1000.0,
        switch_us=50.0,
        Pd=0.95,
        Pfa=1e-4,
    )

    results = runner.evaluate_policies(
        schedulers=schedulers,
        n_episodes=n_episodes,
        base_seed=1000,
        verbose=True,
    )

    # 3. Save comparative plots
    plot_path = OUT / "baseline_comparison.png"
    runner.plot_comparison(results, save_path=plot_path)

    print(f"\n[Phase 2] Benchmark complete! Comparison figure saved → {plot_path}")


if __name__ == "__main__":
    main()
