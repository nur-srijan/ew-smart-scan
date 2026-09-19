"""
eval/compare_4m.py
==================
Comparative evaluation of the 4M Dynamic DRL model vs 1M Static model and baselines
across 10 diverse, randomly generated test episodes.
"""

from pathlib import Path
import numpy as np
import collections

from eval.fom import FoMEvaluator
from ew_sim.env import DynamicSpectrumEnv
from schedulers.drl_agent import DRLScheduler
from schedulers.baselines import SequentialSweep, PseudoRandomSweep
from schedulers.rmab import WhittleIndexScheduler

def run_test():
    models = {
        "SequentialSweep": SequentialSweep(K=35),
        "PseudoRandomSweep": PseudoRandomSweep(K=35, seed=42),
        "WhittleIndexRMAB": WhittleIndexScheduler(K=35),
        "DRL-1M-Static": DRLScheduler(K=35, model_path="checkpoints/ppo_recurrent_kaggle_1m.zip", deterministic=True, seed=42),
        "DRL-4M-Dynamic": DRLScheduler(K=35, model_path="checkpoints/ppo_recurrent_kaggle_dynamic_4m.zip", deterministic=True, seed=42),
    }

    results = {name: {"hits": [], "ir": [], "disc": []} for name in models}
    n_episodes = 10

    for ep in range(n_episodes):
        seed = 700 + ep
        for name, sched in models.items():
            env = DynamicSpectrumEnv(K=35, T=1000, stage=3, seed=seed)
            obs, info = env.reset(seed=seed)
            sched.reset(seed=seed)
            hits, actions, rewards = [], [], []
            for t in range(1000):
                a = sched.select_band(obs, info)
                obs, r, term, trunc, info = env.step(a)
                is_hit = bool(env.truth.is_active(a, env.current_step - 1))
                sched.update_feedback(a, is_hit, info)
                actions.append(a)
                hits.append(is_hit)
                rewards.append(r)
            evaluator = FoMEvaluator(env.truth)
            rep = evaluator.evaluate_trajectory(name, actions, hits, rewards)
            results[name]["hits"].append(rep.total_hits)
            results[name]["ir"].append(rep.overall_interception_ratio * 100.0)
            results[name]["disc"].append(rep.discovery_rate * 100.0)

    print("\n" + "=" * 75)
    print("  COMPARATIVE EVALUATION ON 10 DYNAMIC UNSEEN BATTLEFIELDS")
    print("=" * 75)
    header = f"{'Policy':<20} | {'IR Mean ± Std':<18} | {'Mean Hits':<12} | {'Discovery Rate'}"
    print(header)
    print("-" * len(header))
    for name, data in results.items():
        ir_m = np.mean(data["ir"])
        ir_s = np.std(data["ir"])
        hits_m = np.mean(data["hits"])
        disc_m = np.mean(data["disc"])
        print(f"{name:<20} | {ir_m:5.2f}% ± {ir_s:4.2f}%   | {hits_m:5.1f}/1000   | {disc_m:5.1f}%")
    print("-" * len(header))

if __name__ == "__main__":
    run_test()
