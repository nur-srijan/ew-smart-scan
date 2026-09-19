"""
eval/benchmark_kaggle_1m.py
===========================
Monte Carlo comparative benchmark script for evaluating the newly trained
Kaggle 1M-step DRL curriculum model against legacy sweeps, priority queues,
Whittle index RMAB, and the earlier 25k baseline model across 20 test episodes.
"""

from pathlib import Path
import numpy as np
import json
import torch

from ew_sim.truth_engine import build_default_truth_engine
from schedulers.baselines import (
    SequentialSweep,
    PseudoRandomSweep,
    PriorityQueueSweep,
)
from schedulers.rmab import WhittleIndexScheduler
from schedulers.drl_agent import DRLScheduler
from eval.runner import MonteCarloRunner

def run_benchmark():
    K = 35
    T = 2000
    n_episodes = 20
    base_seed = 300

    print("=" * 80)
    print("  EW SMART SCAN — 1M-STEP KAGGLE T4 CURRICULUM BENCHMARK")
    print(f"  Evaluating across {n_episodes} Monte Carlo episodes (T={T} slots, K={K} bands)")
    print("=" * 80)

    schedulers = [
        SequentialSweep(K=K),
        PseudoRandomSweep(K=K, seed=42),
        PriorityQueueSweep(K=K),
        WhittleIndexScheduler(K=K),
    ]

    # Baseline 25k model
    p25k = Path("checkpoints/ppo_recurrent_ew.zip")
    if p25k.exists():
        sched_25k = DRLScheduler(K=K, model_path=p25k, deterministic=True, seed=42)
        sched_25k.name = "DRL-Local-25k"
        schedulers.append(sched_25k)

    # Kaggle 1M model
    p1m = Path("checkpoints/ppo_recurrent_kaggle_1m.zip")
    if p1m.exists():
        sched_1m = DRLScheduler(K=K, model_path=p1m, deterministic=True, seed=42)
        sched_1m.name = "DRL-Kaggle-1M"
        schedulers.append(sched_1m)

    runner = MonteCarloRunner(K=K, T=T)
    results = runner.evaluate_policies(schedulers, n_episodes=n_episodes, base_seed=base_seed, verbose=True)

    summary = {}
    for name, reports in results.items():
        irs = [r.overall_interception_ratio * 100.0 for r in reports]
        ttis = [r.mean_time_to_intercept_sec for r in reports]
        max_ttis = [r.max_time_to_intercept_sec for r in reports]
        disc = [r.discovery_rate * 100.0 for r in reports]
        pulses = [r.total_intercepted_pulses for r in reports]
        rews = [r.total_reward for r in reports]

        summary[name] = {
            "ir_mean": float(np.mean(irs)),
            "ir_std": float(np.std(irs)),
            "tti_mean_sec": float(np.mean(ttis)),
            "tti_std_sec": float(np.std(ttis)),
            "max_tti_mean_sec": float(np.mean(max_ttis)),
            "disc_mean_pct": float(np.mean(disc)),
            "pulses_intercepted_mean": float(np.mean(pulses)),
            "reward_mean": float(np.mean(rews)),
        }

    out_json = Path("eval/kaggle_1m_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved benchmark metrics to {out_json}")

if __name__ == "__main__":
    run_benchmark()
