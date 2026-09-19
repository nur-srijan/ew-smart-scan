"""
eval/benchmark_multi_receiver.py
================================
Comprehensive comparative benchmark of Multi-Receiver Cooperative Scheduling (M=4)
vs Single-Receiver Scanning (M=1) across 20 dynamic, unseen RF battlefields.

Evaluates:
    1. SequentialSweep (M=1)
    2. PseudoRandomSweep (M=1)
    3. WhittleIndexRMAB (M=1)
    4. DRL-4M (M=1)
    5. MultiSequentialSweep (M=4)
    6. MultiPseudoRandomSweep (M=4)
    7. MultiWhittleRMAB (M=4)
    8. CooperativeRoleScheduler (M=4)
    9. MultiDRLScheduler (M=4)

Produces:
    - Statistical FoM table (Global IR, Primary Target IR, Hits, Discovery Rate, TTI, Collisions)
    - multi_receiver_benchmark.png (4-panel FoM comparative chart)
    - multi_receiver_cumulative_ir.png (Cumulative pulse interception progression over time)
"""

from pathlib import Path
from typing import Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ew_sim.env import EWSpectrumEnv
from ew_sim.multi_env import MultiReceiverEWSpectrumEnv, DynamicMultiReceiverEnv
from schedulers.baselines import SequentialSweep, PseudoRandomSweep
from schedulers.rmab import WhittleIndexScheduler
from schedulers.drl_agent import DRLScheduler
from schedulers.multi_schedulers import (
    MultiSequentialSweep,
    MultiPseudoRandomSweep,
    MultiWhittleIndexScheduler,
    CooperativeRoleScheduler,
)
from schedulers.multi_drl import MultiDRLScheduler
from eval.fom import FoMEvaluator

# Paths
ROOT = Path(__file__).parent.parent
CHECKPOINTS = ROOT / "checkpoints"
ARTIFACT_DIR = Path("/Users/nursrijan/.gemini/antigravity/brain/59130a49-983c-4a01-a3dd-bf066ff9b3a7")
EVAL_DIR = ROOT / "eval"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
    "axes.edgecolor": "#D0D7DE",
    "axes.linewidth": 1.0,
})

POLICY_COLORS = {
    "Sequential (M=1)": "#8C959F",
    "PseudoRand (M=1)": "#57606A",
    "Whittle (M=1)": "#0969DA",
    "DRL-4M (M=1)": "#1F883D",
    "MultiSeq (M=4)": "#8250DF",
    "MultiPseudo (M=4)": "#FB8500",
    "MultiWhittle (M=4)": "#2EA44F",
    "CoopRole (M=4)": "#059669",
    "MultiDRL (M=4)": "#0284C7",
}


def run_benchmark(n_episodes: int = 20, K: int = 35, M: int = 4, T: int = 1000):
    print("=" * 80)
    print(f"  MULTI-RECEIVER COOPERATIVE SCHEDULING BENCHMARK (M={M} vs M=1)")
    print(f"  Evaluating across {n_episodes} Dynamic Unseen Battlefields (K={K}, T={T})")
    print("=" * 80)

    drl_path = CHECKPOINTS / "ppo_recurrent_kaggle_dynamic_4m.zip"
    has_drl = drl_path.exists()

    multi_drl_zip = CHECKPOINTS / "ppo_recurrent_multi_stage2.zip"
    multi_drl_pt = CHECKPOINTS / "ppo_recurrent_multi_actor.pt"
    multi_drl_model = multi_drl_zip if multi_drl_zip.exists() else (multi_drl_pt if multi_drl_pt.exists() else None)

    policy_names = [
        "Sequential (M=1)",
        "Whittle (M=1)",
    ]
    if has_drl:
        policy_names.append("DRL-4M (M=1)")
    policy_names.extend([
        "MultiSeq (M=4)",
        "MultiWhittle (M=4)",
        "CoopRole (M=4)",
        "MultiDRL (M=4)",
    ])

    results: dict[str, dict[str, list[Any]]] = {
        name: {
            "ir": [],
            "fixed_ir": [],
            "hits": [],
            "disc": [],
            "tti": [],
            "collisions": [],
            "cum_hits": [],
        }
        for name in policy_names
    }

    for ep in range(n_episodes):
        seed = 3000 + ep
        # Generate dynamic randomized ground truth RF battlefield
        base_dyn_env = DynamicMultiReceiverEnv(K=K, M=M, T=T, stage=3, seed=seed)
        base_dyn_env.reset(seed=seed)
        truth = base_dyn_env.truth

        # ── 1. Single-Receiver Policies (M=1) ──────────────────────────────
        single_policies = {
            "Sequential (M=1)": SequentialSweep(K=K),
            "Whittle (M=1)": WhittleIndexScheduler(K=K, seed=seed),
        }
        if has_drl:
            single_policies["DRL-4M (M=1)"] = DRLScheduler(
                K=K, model_path=str(drl_path), deterministic=True, seed=seed
            )

        for name, sched in single_policies.items():
            env_s = EWSpectrumEnv(truth_engine=truth, K=K, T=T, seed=seed)
            obs, info = env_s.reset(seed=seed)
            sched.reset(seed=seed)

            actions, hits, rewards = [], [], []
            for _ in range(T):
                a = sched.select_band(obs, info)
                obs, r, term, trunc, info = env_s.step(a)
                is_hit = bool(truth.is_active(a, env_s.current_step - 1))
                sched.update_feedback(a, is_hit, info)
                actions.append(a)
                hits.append(is_hit)
                rewards.append(r)

            evaluator = FoMEvaluator(truth)
            rep = evaluator.evaluate_trajectory(name, actions, hits, rewards)
            results[name]["ir"].append(rep.overall_interception_ratio * 100.0)
            results[name]["hits"].append(rep.total_hits)
            results[name]["disc"].append(rep.discovery_rate * 100.0)
            results[name]["tti"].append(rep.mean_time_to_intercept_sec)
            results[name]["collisions"].append(0.0)
            results[name]["cum_hits"].append(rep.cumulative_hit_curve)

            # Per-emitter fixed radar IR
            fixed_irs = [
                em.interception_ratio * 100.0
                for em in rep.emitter_reports.values()
                if "Fixed" in em.emitter_type
            ]
            results[name]["fixed_ir"].append(float(np.mean(fixed_irs)) if fixed_irs else 0.0)

        # ── 2. Multi-Receiver Policies (M=4) ───────────────────────────────
        multi_policies = {
            "MultiSeq (M=4)": MultiSequentialSweep(K=K, M=M, seed=seed),
            "MultiWhittle (M=4)": MultiWhittleIndexScheduler(K=K, M=M, seed=seed),
            "CoopRole (M=4)": CooperativeRoleScheduler(K=K, M=M, seed=seed),
            "MultiDRL (M=4)": MultiDRLScheduler(
                K=K, M=M,
                model_path=str(multi_drl_model) if multi_drl_model else None,
                deterministic=True,
                collision_masking=True,
                seed=seed,
            ),
        }

        for name, sched in multi_policies.items():
            env_m = MultiReceiverEWSpectrumEnv(truth_engine=truth, K=K, M=M, T=T, seed=seed)
            obs, info = env_m.reset(seed=seed)
            sched.reset(seed=seed)

            actions, hits, rewards = [], [], []
            for _ in range(T):
                bands = sched.select_bands(obs, info)
                obs, r, term, trunc, info = env_m.step(bands)
                step_hits = info.get("last_band_hits", {})
                sched.update_feedback(bands, step_hits, info)
                actions.append(bands)
                hits.append(step_hits)
                rewards.append(r)

            evaluator = FoMEvaluator(truth)
            rep = evaluator.evaluate_trajectory(name, actions, hits, rewards)
            results[name]["ir"].append(rep.overall_interception_ratio * 100.0)
            results[name]["hits"].append(rep.total_hits)
            results[name]["disc"].append(rep.discovery_rate * 100.0)
            results[name]["tti"].append(rep.mean_time_to_intercept_sec)
            results[name]["collisions"].append(rep.collision_rate * 100.0)
            results[name]["cum_hits"].append(rep.cumulative_hit_curve)

            # Per-emitter fixed radar IR
            fixed_irs = [
                em.interception_ratio * 100.0
                for em in rep.emitter_reports.values()
                if "Fixed" in em.emitter_type
            ]
            results[name]["fixed_ir"].append(float(np.mean(fixed_irs)) if fixed_irs else 0.0)

        print(f"  Completed episode {ep + 1}/{n_episodes} (Seed {seed})")

    # ── Display Summary Table ──────────────────────────────────────────────
    print("\n" + "=" * 105)
    print(f"{'Policy':<18} | {'Global IR':<14} | {'Target (Fixed) IR':<18} | {'Hits':<10} | {'Discovery':<10} | {'Mean TTI':<10} | {'Collisions'}")
    print("=" * 105)
    for name in policy_names:
        d = results[name]
        ir_m, ir_s = np.mean(d["ir"]), np.std(d["ir"])
        f_ir_m = np.mean(d["fixed_ir"])
        hits_m = np.mean(d["hits"])
        disc_m = np.mean(d["disc"])
        tti_m = np.mean(d["tti"])
        coll_m = np.mean(d["collisions"])
        print(f"{name:<18} | {ir_m:5.2f}% ± {ir_s:4.2f}% | {f_ir_m:5.2f}%            | {hits_m:5.1f}    | {disc_m:5.1f}%    | {tti_m:5.3f}s   | {coll_m:4.1f}%")
    print("=" * 105)

    # ── Generate Benchmark Plot: 4-Panel Comparison ────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel A: Overall Interception Ratio
    ax = axes[0, 0]
    names_display = [n.replace(" ", "\n") for n in policy_names]
    ir_means = [np.mean(results[n]["ir"]) for n in policy_names]
    ir_stds = [np.std(results[n]["ir"]) for n in policy_names]
    bar_colors = [POLICY_COLORS[n] for n in policy_names]

    bars = ax.bar(names_display, ir_means, yerr=ir_stds, capsize=4, color=bar_colors, alpha=0.9, edgecolor="#333", linewidth=0.8)
    ax.axhline(20.0, color="#CF222E", linestyle="--", linewidth=1.5, label="M=1 Physical Limit (~20%)")
    ax.set_ylabel("Overall Interception Ratio (%)")
    ax.set_title("(a) Global Interception Ratio Across All Emitters")
    ax.set_ylim(0, max(45, max(ir_means) + 10))
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper left", framealpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

    # Panel B: Fixed Radar Target Interception Ratio
    ax = axes[0, 1]
    fix_means = [np.mean(results[n]["fixed_ir"]) for n in policy_names]
    fix_stds = [np.std(results[n]["fixed_ir"]) for n in policy_names]
    bars = ax.bar(names_display, fix_means, yerr=fix_stds, capsize=4, color=bar_colors, alpha=0.9, edgecolor="#333", linewidth=0.8)
    ax.axhline(75.0, color="#1A7F37", linestyle=":", linewidth=1.5, label="Target Barrier (>75%)")
    ax.set_ylabel("Target (Fixed Radar) IR (%)")
    ax.set_title("(b) Primary Target Emitter Interception Ratio")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper left", framealpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

    # Panel C: Mean Intercepted Pulses
    ax = axes[1, 0]
    hit_means = [np.mean(results[n]["hits"]) for n in policy_names]
    hit_stds = [np.std(results[n]["hits"]) for n in policy_names]
    bars = ax.bar(names_display, hit_means, yerr=hit_stds, capsize=4, color=bar_colors, alpha=0.9, edgecolor="#333", linewidth=0.8)
    ax.set_ylabel("Mean Intercepted Pulses (per 1000 slots)")
    ax.set_title("(c) Absolute Pulse Interceptions")
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

    # Panel D: Mean Time-to-Intercept
    ax = axes[1, 1]
    tti_means = [np.mean(results[n]["tti"]) * 1000.0 for n in policy_names]
    tti_stds = [np.std(results[n]["tti"]) * 1000.0 for n in policy_names]
    bars = ax.bar(names_display, tti_means, yerr=tti_stds, capsize=4, color=bar_colors, alpha=0.9, edgecolor="#333", linewidth=0.8)
    ax.set_ylabel("Mean Time-to-Intercept (Milliseconds)")
    ax.set_title("(d) Threat Warning Latency (Lower is Better)")
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.0f} ms", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

    plt.suptitle("Cooperative Multi-Receiver (M=4) Scheduling vs Baselines (20 Dynamic Battlefields)", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()

    out_art = ARTIFACT_DIR / "multi_receiver_benchmark.png"
    out_eval = EVAL_DIR / "multi_receiver_benchmark.png"
    fig.savefig(out_art, dpi=200, bbox_inches="tight")
    fig.savefig(out_eval, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved benchmark plot to:\n  {out_art}\n  {out_eval}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Multi-Receiver Benchmark")
    parser.add_argument("--episodes", type=int, default=20, help="Number of Monte Carlo episodes")
    args = parser.parse_args()
    run_benchmark(n_episodes=args.episodes)
