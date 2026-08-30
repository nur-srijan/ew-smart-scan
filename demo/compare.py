"""
demo/compare.py
===============
Side-by-Side Live Policy Comparison and Spectrogram Dwell Visualizer.

Executes:
    1. Sequential Sweep (Open-loop baseline)
    2. Whittle Index RMAB (Analytical Restless Bandit)
    3. DRL Scheduler (Recurrent PPO Policy)
on the exact same live RF battlefield episode.

Overlays receiver dwells directly onto the RF ground-truth spectrogram:
    - Green markers: Successful pulse intercept (Hit)
    - Red markers: Sensed dwell with no active pulse (Miss)
    - Orange bars: Underlying ground-truth pulse transmissions

Exports:
    1. Static 4-Panel High-Res Image: notebooks/live_comparison_waterfall.png
    2. Interactive Standalone HTML Report: notebooks/interactive_comparison.html

Usage:
    uv run demo/compare.py
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ew_sim.env import EWSpectrumEnv
from ew_sim.truth_engine import build_default_truth_engine
from schedulers.baselines import SequentialSweep
from schedulers.rmab import WhittleIndexScheduler
from schedulers.drl_agent import DRLScheduler
from eval.fom import FoMEvaluator

OUT = Path(__file__).parent.parent / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS = Path(__file__).parent.parent / "checkpoints"


def run_policy_episode(scheduler, truth, K=35, T=600, Pd=0.95, Pfa=1e-4, seed=42):
    """Runs a single episode and returns full trajectory history."""
    env = EWSpectrumEnv(truth_engine=truth, K=K, T=T, Pd=Pd, Pfa=Pfa, seed=seed)
    scheduler.reset(seed=seed)
    obs, info = env.reset(seed=seed)

    actions = []
    hits = []
    rewards = []
    dwell_types = []  # "hit", "miss", "false_alarm", "quiet"

    for t in range(T):
        action = scheduler.select_band(obs, info)
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        is_truly_active = truth.is_active(action, t)
        hit_observed = bool(reward > 0) or bool(is_truly_active and env._rng.random() < Pd)
        scheduler.update_feedback(action, hit_observed, next_info)

        if is_truly_active and hit_observed:
            dwell_types.append("hit")
        elif is_truly_active and not hit_observed:
            dwell_types.append("miss")
        elif not is_truly_active and hit_observed:
            dwell_types.append("false_alarm")
        else:
            dwell_types.append("quiet")

        actions.append(action)
        hits.append(hit_observed)
        rewards.append(reward)

        obs = next_obs
        info = next_info

    evaluator = FoMEvaluator(truth)
    report = evaluator.evaluate_trajectory(
        policy_name=scheduler.name,
        actions=actions,
        hits=hits,
        rewards=rewards,
    )
    return actions, hits, dwell_types, report


def generate_static_comparison_plot(truth, policy_trajectories, T_view=400, save_path=None):
    """
    Generates a 4-panel static Matplotlib figure comparing truth with 3 policy dwell overlays.
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True, sharey=True)

    S_window = truth.S[:, :T_view]
    cmap = mcolors.LinearSegmentedColormap.from_list("ew", ["#0D1B2A", "#E07B39"])

    # Panel 0: Ground Truth
    ax0 = axes[0]
    ax0.imshow(S_window, aspect="auto", origin="lower", cmap=cmap, extent=[0, T_view, -0.5, truth.K - 0.5])
    ax0.set_title("1. Ground Truth RF Battleground (Active Emitter Pulses in Orange)", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Sub-band (k)", fontsize=9)

    # Panels 1 to 3: Policies
    policy_keys = ["SequentialSweep", "WhittleIndexRMAB", "DRLScheduler-RecurrentPPO"]
    titles = [
        "2. Sequential Sweep Dwells (Stroboscopic Resonance: Misses Hopping & Scanning Radars)",
        "3. Whittle Index RMAB Dwells (Analytical Bandit: Rapidly Exploits Active Channels)",
        "4. Recurrent DRL Agent Dwells (AI Scheduler: Synchronized High-Rate Pulse Capture)",
    ]

    for idx, key in enumerate(policy_keys):
        ax = axes[idx + 1]
        ax.imshow(S_window, aspect="auto", origin="lower", cmap=cmap, extent=[0, T_view, -0.5, truth.K - 0.5], alpha=0.45)
        
        actions, hits, dwell_types, report = policy_trajectories[key]
        act_win = actions[:T_view]
        dwell_win = dwell_types[:T_view]

        t_axis = np.arange(T_view)

        # Plot hits (green dots) and quiet dwells (cyan lines / markers)
        hit_mask = np.array([d == "hit" for d in dwell_win])
        miss_mask = np.array([d == "miss" for d in dwell_win])
        quiet_mask = np.array([d == "quiet" for d in dwell_win])

        # Plot dwell trajectory line
        ax.plot(t_axis, act_win, color="#FFFFFF", linewidth=0.8, alpha=0.4, linestyle=":")

        # Dwell markers
        if np.any(quiet_mask):
            ax.scatter(t_axis[quiet_mask], np.array(act_win)[quiet_mask], color="#38BDF8", s=10, alpha=0.6, label="Quiet Dwell")
        if np.any(hit_mask):
            ax.scatter(t_axis[hit_mask], np.array(act_win)[hit_mask], color="#22C55E", s=35, edgecolor="black", linewidth=0.6, zorder=5, label=f"Intercepted Pulse ({report.total_hits} Hits)")
        if np.any(miss_mask):
            ax.scatter(t_axis[miss_mask], np.array(act_win)[miss_mask], color="#EF4444", s=25, edgecolor="black", linewidth=0.5, label="Missed Pulse")

        ir_pct = report.overall_interception_ratio * 100.0
        ax.set_title(f"{titles[idx]} — IR: {ir_pct:.1f}% | Discovered: {report.discovery_rate*100:.0f}%", fontsize=11, fontweight="bold")
        ax.set_ylabel("Sub-band (k)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.75)

    axes[-1].set_xlabel("Time Slot Index (t)", fontsize=10, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"[compare.py] Static comparison waterfall saved → {save_path}")

    return fig


def generate_interactive_html_report(truth, policy_trajectories, T_view=400, save_path=None):
    """
    Generates an interactive Plotly HTML report with zoomable spectrograms and performance telemetry.
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "Sequential Sweep vs Ground Truth",
            "Whittle Index RMAB vs Ground Truth",
            "Recurrent DRL Agent vs Ground Truth",
        ],
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    S_window = truth.S[:, :T_view]
    t_axis = np.arange(T_view)

    # Base spectrogram heatmap for each subplot
    policies = ["SequentialSweep", "WhittleIndexRMAB", "DRLScheduler-RecurrentPPO"]
    colors = {"SequentialSweep": "#3B82F6", "WhittleIndexRMAB": "#F59E0B", "DRLScheduler-RecurrentPPO": "#10B981"}

    for row_idx, pol_name in enumerate(policies, start=1):
        # Background truth heatmap
        fig.add_trace(
            go.Heatmap(
                z=S_window,
                x=t_axis,
                y=list(range(truth.K)),
                colorscale=[[0, "#0F172A"], [1, "#E07B39"]],
                showscale=False,
                hoverinfo="x+y+z",
                opacity=0.5,
                name=f"RF Truth ({pol_name})",
            ),
            row=row_idx, col=1,
        )

        actions, hits, dwell_types, report = policy_trajectories[pol_name]
        act_win = actions[:T_view]
        dwell_win = dwell_types[:T_view]

        # Sensed Hit Points (Green)
        hit_x = [t for t, d in enumerate(dwell_win) if d == "hit"]
        hit_y = [act_win[t] for t in hit_x]

        fig.add_trace(
            go.Scatter(
                x=hit_x, y=hit_y,
                mode="markers",
                marker=dict(size=7, color="#22C55E", symbol="circle", line=dict(width=1, color="#FFFFFF")),
                name=f"{pol_name} - Hit ({len(hit_x)} pulses)",
            ),
            row=row_idx, col=1,
        )

        # Receiver Dwell Trajectory Line
        fig.add_trace(
            go.Scatter(
                x=t_axis, y=act_win,
                mode="lines",
                line=dict(color=colors[pol_name], width=1, dash="dot"),
                opacity=0.6,
                name=f"{pol_name} Trajectory",
            ),
            row=row_idx, col=1,
        )

        fig.update_yaxes(title_text="Sub-band (k)", row=row_idx, col=1)

    fig.update_xaxes(title_text="Time Slot Index (t)", row=3, col=1)
    fig.update_layout(
        title="<b>EW Smart Scan — Live Dwell Comparison Against RF Battleground</b>",
        height=900,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if save_path:
        fig.write_html(str(save_path))
        print(f"[compare.py] Interactive HTML report saved → {save_path}")

    return fig


def main():
    print("=" * 75)
    print("  EW Smart Scan — Phase 4: Side-by-Side Live Policy Visualizer")
    print("=" * 75)

    K = 35
    T = 600
    seed = 42

    # 1. Build common RF truth battlefield
    truth = build_default_truth_engine(K=K, T=T, dwell_us=1000, switch_us=50, seed=seed, verbose=False)

    # 2. Instantiate policies
    sb3_ckpt = CHECKPOINTS / "ppo_recurrent_ew.zip"
    pt_ckpt = CHECKPOINTS / "drl_scheduler.pt"
    drl_path = sb3_ckpt if sb3_ckpt.exists() else (pt_ckpt if pt_ckpt.exists() else None)

    schedulers = {
        "SequentialSweep": SequentialSweep(K=K),
        "WhittleIndexRMAB": WhittleIndexScheduler(K=K, seed=seed),
        "DRLScheduler-RecurrentPPO": DRLScheduler(K=K, model_path=drl_path, seed=seed),
    }

    # 3. Run side-by-side evaluation
    trajectories = {}
    print("\nRunning live evaluation episodes...")
    for name, sched in schedulers.items():
        acts, hts, dwells, rep = run_policy_episode(sched, truth, K=K, T=T, seed=seed)
        trajectories[name] = (acts, hts, dwells, rep)
        print(f"  ✓ {name:<26}: Interception Ratio = {rep.overall_interception_ratio*100:5.1f}% | Hits = {rep.total_hits:3d} | Discovery = {rep.discovery_rate*100:3.0f}%")

    # 4. Generate Visualizations
    static_png = OUT / "live_comparison_waterfall.png"
    interactive_html = OUT / "interactive_comparison.html"

    generate_static_comparison_plot(truth, trajectories, T_view=400, save_path=static_png)
    generate_interactive_html_report(truth, trajectories, T_view=400, save_path=interactive_html)

    print(f"\n[Phase 4] Visualizations generated successfully in {OUT}/")


if __name__ == "__main__":
    main()
