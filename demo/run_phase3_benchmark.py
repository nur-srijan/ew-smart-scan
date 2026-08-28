"""
demo/run_phase3_benchmark.py
============================
Phase 3 Benchmark: Full Comparative Evaluation across Classical Baselines,
RMAB Whittle Index, Hybrid Periodicity Predictor, and Deep RL Schedulers.

Executes a 20-episode Monte Carlo evaluation across identical multi-emitter
battlefields and generates comparative Figures of Merit (IR, TTI, Discovery).

Usage:
    uv run demo/run_phase3_benchmark.py
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from schedulers.baselines import (
    SequentialSweep,
    PseudoRandomSweep,
    PriorityQueueSweep,
)
from schedulers.rmab import WhittleIndexScheduler
from schedulers.predictor import HybridPredictiveScheduler
from schedulers.drl_agent import DRLScheduler
from eval.runner import MonteCarloRunner

OUT = Path(__file__).parent.parent / "notebooks"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def main():
    print("=" * 80)
    print("  EW Smart Scan — Phase 3: AI Schedulers vs Classical Baselines Benchmark")
    print("=" * 80)

    K = 35
    T = 2000
    n_episodes = 20

    # 1. Instantiate Schedulers
    edb_priorities = np.ones(K) * 0.5
    edb_priorities[4] = 3.0   # Fixed radar
    edb_priorities[11] = 4.0  # Scanning radar
    edb_priorities[18] = 2.5  # Fixed radar

    sb3_ckpt = CHECKPOINT_DIR / "ppo_recurrent_ew.zip"
    pt_ckpt = CHECKPOINT_DIR / "drl_scheduler.pt"
    drl_model_path = sb3_ckpt if sb3_ckpt.exists() else (pt_ckpt if pt_ckpt.exists() else None)

    schedulers = [
        SequentialSweep(K=K),
        PseudoRandomSweep(K=K, seed=42),
        PriorityQueueSweep(K=K, priority_weights=edb_priorities, seed=42),
        WhittleIndexScheduler(K=K, seed=42),
        HybridPredictiveScheduler(K=K, seed=42),
        DRLScheduler(K=K, model_path=drl_model_path, seed=42),
    ]

    # 2. Run Monte Carlo Evaluation
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
        base_seed=2000,
        verbose=True,
    )

    # 3. Export Comparative Figure
    plot_path = OUT / "phase3_comparison.png"
    runner.plot_comparison(results, save_path=plot_path)

    print(f"\n[Phase 3] Benchmark complete! Comparison figure saved → {plot_path}")


if __name__ == "__main__":
    main()
