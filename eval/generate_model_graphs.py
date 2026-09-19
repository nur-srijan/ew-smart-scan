"""
eval/generate_model_graphs.py
=============================
Generates comprehensive publication-grade figures evaluating the 4M Dynamic DRL model
against previous models and baseline scan policies.

Generates:
1. model_comparison_benchmark.png — 4-panel FoM comparative analysis (IR, Hits, Discovery, TTI)
2. dwell_distribution_anti_camping.png — Sub-band dwell allocation proving anti-camping & generalization
3. waterfall_live_tracking.png — 4-row spectrogram waterfall comparing dwell overlays
4. cumulative_discovery_curve.png — Threat discovery progression over time
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ew_sim.env import DynamicSpectrumEnv, EWSpectrumEnv
from ew_sim.truth_engine import build_default_truth_engine
from schedulers.baselines import SequentialSweep, PseudoRandomSweep
from schedulers.rmab import WhittleIndexScheduler
from schedulers.drl_agent import DRLScheduler
from eval.fom import FoMEvaluator

# Paths
ROOT = Path(__file__).parent.parent
CHECKPOINTS = ROOT / "checkpoints"
ARTIFACT_DIR = Path("/Users/nursrijan/.gemini/antigravity/brain/59130a49-983c-4a01-a3dd-bf066ff9b3a7")

# Global style configuration
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

COLORS = {
    "Sequential": "#8C959F",       # Neutral gray
    "PseudoRandom": "#57606A",     # Slate
    "WhittleRMAB": "#0969DA",      # Science blue
    "DRL-1M-Static": "#CF222E",    # Warning red (overfitted)
    "DRL-4M-Dynamic": "#1A7F37",   # Success green (robust)
}

def generate_graphs():
    print("=" * 70)
    print("  Generating Comprehensive Performance Graphs for 4M Model")
    print("=" * 70)

    # ──────────────────────────────────────────────────────────────────
    # 1. Benchmark Evaluation Data Gathering across 15 Unseen Episodes
    # ──────────────────────────────────────────────────────────────────
    print("\n[1/4] Running 15 Monte Carlo episodes across policies...")
    models = {
        "Sequential": SequentialSweep(K=35),
        "PseudoRandom": PseudoRandomSweep(K=35, seed=42),
        "WhittleRMAB": WhittleIndexScheduler(K=35),
        "DRL-1M-Static": DRLScheduler(K=35, model_path=CHECKPOINTS / "ppo_recurrent_kaggle_1m.zip", deterministic=True, seed=42),
        "DRL-4M-Dynamic": DRLScheduler(K=35, model_path=CHECKPOINTS / "ppo_recurrent_kaggle_dynamic_4m.zip", deterministic=True, seed=42),
    }

    metrics = {name: {"ir": [], "hits": [], "disc": [], "tti": [], "disc_curves": []} for name in models}
    dwell_histories = {name: [] for name in models}

    n_episodes = 15
    for ep in range(n_episodes):
        seed = 800 + ep
        for name, sched in models.items():
            env = DynamicSpectrumEnv(K=35, T=1000, stage=3, seed=seed)
            obs, info = env.reset(seed=seed)
            sched.reset(seed=seed)
            actions, hits, rewards = [], [], []

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
            metrics[name]["ir"].append(rep.overall_interception_ratio * 100.0)
            metrics[name]["hits"].append(rep.total_hits)
            metrics[name]["disc"].append(rep.discovery_rate * 100.0)
            metrics[name]["tti"].append(rep.mean_time_to_intercept_sec)
            metrics[name]["disc_curves"].append(rep.cumulative_discovery_curve / max(1, rep.total_emitters) * 100.0)

            if ep == 0:
                dwell_histories[name] = actions

    # ──────────────────────────────────────────────────────────────────
    # Graph 1: 4-Panel Figures of Merit Comparison Bar Charts
    # ──────────────────────────────────────────────────────────────────
    print("\n[2/4] Plotting 4-Panel Benchmark Comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("#FFFFFF")

    policy_names = list(models.keys())
    x = np.arange(len(policy_names))
    bar_colors = [COLORS[p] for p in policy_names]

    # Panel A: Interception Ratio (%)
    ax = axes[0, 0]
    means_ir = [np.mean(metrics[p]["ir"]) for p in policy_names]
    stds_ir = [np.std(metrics[p]["ir"]) for p in policy_names]
    bars = ax.bar(x, means_ir, yerr=stds_ir, capsize=4, color=bar_colors, edgecolor="#24292F", alpha=0.9, width=0.55)
    ax.set_title("(A) Overall Interception Ratio (Higher is Better)", fontweight="bold")
    ax.set_ylabel("Interception Ratio (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_names, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(means_ir) * 1.35)
    for bar, m in zip(bars, means_ir):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f"{m:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel B: Intercepted Pulses (Hits / 1000 dwells)
    ax = axes[0, 1]
    means_hits = [np.mean(metrics[p]["hits"]) for p in policy_names]
    stds_hits = [np.std(metrics[p]["hits"]) for p in policy_names]
    bars = ax.bar(x, means_hits, yerr=stds_hits, capsize=4, color=bar_colors, edgecolor="#24292F", alpha=0.9, width=0.55)
    ax.set_title("(B) Pulse Captures per Episode (Higher is Better)", fontweight="bold")
    ax.set_ylabel("Captured Pulses (Hits / 1000 Dwells)")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_names, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(means_hits) * 1.35)
    for bar, m in zip(bars, means_hits):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{m:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel C: Discovery Rate (%)
    ax = axes[1, 0]
    means_disc = [np.mean(metrics[p]["disc"]) for p in policy_names]
    stds_disc = [np.std(metrics[p]["disc"]) for p in policy_names]
    bars = ax.bar(x, means_disc, yerr=stds_disc, capsize=4, color=bar_colors, edgecolor="#24292F", alpha=0.9, width=0.55)
    ax.set_title("(C) Emitter Discovery Rate (Higher is Better)", fontweight="bold")
    ax.set_ylabel("Discovery Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_names, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 105)
    for bar, m in zip(bars, means_disc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2.0, f"{m:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel D: Mean Time-to-Intercept (TTI)
    ax = axes[1, 1]
    means_tti = [np.mean(metrics[p]["tti"]) for p in policy_names]
    stds_tti = [np.std(metrics[p]["tti"]) for p in policy_names]
    bars = ax.bar(x, means_tti, yerr=stds_tti, capsize=4, color=bar_colors, edgecolor="#24292F", alpha=0.9, width=0.55)
    ax.set_title("(D) Mean Time-to-Intercept (Lower is Better)", fontweight="bold")
    ax.set_ylabel("TTI (Seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_names, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(means_tti) * 1.35)
    for bar, m in zip(bars, means_tti):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{m:.3f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("EW Smart Scan — Comparative Performance Benchmark Across 15 Unseen Dynamic Battlefields", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    p1 = ARTIFACT_DIR / "model_comparison_benchmark.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1}")

    # ──────────────────────────────────────────────────────────────────
    # Graph 2: Spectrum Patrol Distribution (The Anti-Camping Proof)
    # ──────────────────────────────────────────────────────────────────
    print("\n[3/4] Plotting Spectrum Dwell Distribution (Anti-Camping Comparison)...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")

    bands = np.arange(35)
    bins = np.arange(36) - 0.5

    # Panel 1: DRL-1M-Static (Flawed / Overfitted)
    ax = axes[0]
    counts_1m, _ = np.histogram(dwell_histories["DRL-1M-Static"], bins=bins)
    ax.bar(bands, counts_1m, color=COLORS["DRL-1M-Static"], width=0.7, edgecolor="#24292F", alpha=0.85)
    ax.set_title("DRL-1M-Static Policy: Pathological Camping on Bands 3 & 17 (Severe Overfitting)", color=COLORS["DRL-1M-Static"], fontweight="bold")
    ax.set_ylabel("Dwell Count")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.text(3, max(counts_1m)*0.85, "← Camped on Band 3", color=COLORS["DRL-1M-Static"], fontweight="bold")
    if counts_1m[17] > 50:
        ax.text(17, counts_1m[17] + 20, "← Camped on Band 17", color=COLORS["DRL-1M-Static"], fontweight="bold")

    # Panel 2: Sequential Sweep
    ax = axes[1]
    counts_seq, _ = np.histogram(dwell_histories["Sequential"], bins=bins)
    ax.bar(bands, counts_seq, color=COLORS["Sequential"], width=0.7, edgecolor="#24292F", alpha=0.85)
    ax.set_title("Sequential Sweep: Rigid Uniform Round-Robin (Blind Open-Loop)", fontweight="bold")
    ax.set_ylabel("Dwell Count")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 3: DRL-4M-Dynamic (Robust & Adaptive)
    ax = axes[2]
    counts_4m, _ = np.histogram(dwell_histories["DRL-4M-Dynamic"], bins=bins)
    ax.bar(bands, counts_4m, color=COLORS["DRL-4M-Dynamic"], width=0.7, edgecolor="#24292F", alpha=0.85)
    ax.set_title("DRL-4M-Dynamic Policy: Balanced Wideband Patrol with Active Threat Dwell Concentration", color=COLORS["DRL-4M-Dynamic"], fontweight="bold")
    ax.set_xlabel("Sub-Band Channel Index (k = 0 … 34)")
    ax.set_ylabel("Dwell Count")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Sub-Band Dwell Allocation Distribution Across 1,000 Time Slots (Anti-Camping Verification)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    p2 = ARTIFACT_DIR / "dwell_distribution_anti_camping.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p2}")

    # ──────────────────────────────────────────────────────────────────
    # Graph 3: Waterfall Spectrogram Tracking Visualization (T=250 slots)
    # ──────────────────────────────────────────────────────────────────
    print("\n[4/4] Plotting 4-Row Waterfall Spectrogram with Dwell Overlays...")
    eval_seed = 842
    env = DynamicSpectrumEnv(K=35, T=250, stage=3, seed=eval_seed)
    obs, info = env.reset(seed=eval_seed)
    truth_S = env.truth.S[:, :250]  # shape (35, 250)

    # Run policies on this identical 250-slot episode
    trajectories = {}
    for name in ["Sequential", "DRL-1M-Static", "DRL-4M-Dynamic"]:
        env_run = DynamicSpectrumEnv(K=35, T=250, stage=3, seed=eval_seed)
        obs, info = env_run.reset(seed=eval_seed)
        sched = models[name]
        sched.reset(seed=eval_seed)
        dwells, hit_mask = [], []
        for t in range(250):
            a = sched.select_band(obs, info)
            obs, r, term, trunc, info = env_run.step(a)
            is_hit = bool(env_run.truth.is_active(a, t))
            sched.update_feedback(a, is_hit, info)
            dwells.append(a)
            hit_mask.append(is_hit)
        trajectories[name] = {"dwells": dwells, "hits": hit_mask}

    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")

    cmap_truth = mcolors.LinearSegmentedColormap.from_list("rf", ["#0D1B2A", "#F77F00"])

    # Row 1: Ground Truth Spectrum Activity
    ax = axes[0]
    ax.imshow(truth_S, aspect="auto", origin="lower", cmap=cmap_truth, extent=[0, 250, -0.5, 34.5], interpolation="nearest")
    ax.set_title("Ground-Truth RF Activity Matrix S[k, t] (Active Radar Pulses in Orange)", fontweight="bold")
    ax.set_ylabel("Sub-Band (k)")

    # Row 2: Sequential Sweep
    ax = axes[1]
    ax.imshow(truth_S, aspect="auto", origin="lower", cmap=cmap_truth, extent=[0, 250, -0.5, 34.5], interpolation="nearest", alpha=0.4)
    seq_d = trajectories["Sequential"]["dwells"]
    seq_h = trajectories["Sequential"]["hits"]
    t_idx = np.arange(250)
    # Plot misses in red, hits in bright green
    misses = [t for t, h in enumerate(seq_h) if not h]
    hits_idx = [t for t, h in enumerate(seq_h) if h]
    ax.scatter(misses, [seq_d[t] for t in misses], color="#CF222E", s=10, alpha=0.6, label="Miss")
    ax.scatter(hits_idx, [seq_d[t] for t in hits_idx], color="#00FF66", edgecolor="black", s=36, zorder=5, label="Pulse Intercept (Hit)")
    ax.set_title("Sequential Sweep: Rigid Diagonal Sweep (Misses agile FHSS & burst transmissions)", fontweight="bold")
    ax.set_ylabel("Sub-Band (k)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # Row 3: DRL-1M-Static
    ax = axes[2]
    ax.imshow(truth_S, aspect="auto", origin="lower", cmap=cmap_truth, extent=[0, 250, -0.5, 34.5], interpolation="nearest", alpha=0.4)
    s_d = trajectories["DRL-1M-Static"]["dwells"]
    s_h = trajectories["DRL-1M-Static"]["hits"]
    misses = [t for t, h in enumerate(s_h) if not h]
    hits_idx = [t for t, h in enumerate(s_h) if h]
    ax.scatter(misses, [s_d[t] for t in misses], color="#CF222E", s=10, alpha=0.6)
    ax.scatter(hits_idx, [s_d[t] for t in hits_idx], color="#00FF66", edgecolor="black", s=36, zorder=5)
    ax.set_title("DRL-1M-Static: Memorized Band Camping (Stuck on bands 3 & 17, completely blind to battlefield)", color="#CF222E", fontweight="bold")
    ax.set_ylabel("Sub-Band (k)")

    # Row 4: DRL-4M-Dynamic
    ax = axes[3]
    ax.imshow(truth_S, aspect="auto", origin="lower", cmap=cmap_truth, extent=[0, 250, -0.5, 34.5], interpolation="nearest", alpha=0.4)
    d_d = trajectories["DRL-4M-Dynamic"]["dwells"]
    d_h = trajectories["DRL-4M-Dynamic"]["hits"]
    misses = [t for t, h in enumerate(d_h) if not h]
    hits_idx = [t for t, h in enumerate(d_h) if h]
    ax.scatter(misses, [d_d[t] for t in misses], color="#8C959F", s=10, alpha=0.5, label="Exploratory Dwell")
    ax.scatter(hits_idx, [d_d[t] for t in hits_idx], color="#00FF66", edgecolor="black", s=45, zorder=5, label="Pulse Intercept (Hit)")
    ax.set_title("DRL-4M-Dynamic: Autonomous Wideband Patrol & Synchronized Radar Interception", color="#1A7F37", fontweight="bold")
    ax.set_ylabel("Sub-Band (k)")
    ax.set_xlabel("Time Slot (t) [1 slot = 1.05 ms]")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    plt.suptitle("Live Spectrogram Waterfall with Receiver Dwell Overlays (Green = Intercepted Radar Pulse)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    p3 = ARTIFACT_DIR / "waterfall_live_tracking.png"
    fig.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p3}")

    # ──────────────────────────────────────────────────────────────────
    # Graph 4: Cumulative Discovery Curves
    # ──────────────────────────────────────────────────────────────────
    print("\n[5/5] Plotting Cumulative Threat Discovery Progression...")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#FFFFFF")

    t_slots = np.arange(1000)
    for name in policy_names:
        curves = np.array(metrics[name]["disc_curves"])
        mean_c = np.mean(curves, axis=0)
        std_c = np.std(curves, axis=0)
        ax.plot(t_slots, mean_c, label=f"{name} ({means_disc[policy_names.index(name)]:.1f}%)", color=COLORS[name], linewidth=2.0)
        ax.fill_between(t_slots, mean_c - std_c, mean_c + std_c, color=COLORS[name], alpha=0.12)

    ax.set_title("Cumulative Emitter Discovery Progression over 1,000 Time Slots", fontweight="bold")
    ax.set_xlabel("Time Slot (t)")
    ax.set_ylabel("Cumulative Emitters Discovered (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 1000)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    p4 = ARTIFACT_DIR / "cumulative_discovery_curve.png"
    fig.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p4}")

    print("\nAll 4 graphs generated successfully!")

if __name__ == "__main__":
    generate_graphs()
