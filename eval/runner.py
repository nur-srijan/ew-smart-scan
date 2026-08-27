"""
eval/runner.py
==============
Monte Carlo Evaluation Runner for EW Scan Schedulers.

Executes comparative benchmarks across multiple scan policies (Sequential,
Pseudo-Random, Priority, RMAB, DRL) over N episodes with diverse seeds,
emitter profiles, and noise levels.

Generates:
    - Statistical summaries (mean +/- std for IR, TTI, Pd, Pfa)
    - Formatted comparison tables
    - Publication-grade comparison plots (TTI CDF, Interception Ratio bar charts)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ew_sim.env import EWSpectrumEnv
from ew_sim.truth_engine import build_default_truth_engine, TruthEngine
from schedulers.baselines import BaseScheduler
from eval.fom import FoMEvaluator, EpisodeReport


class MonteCarloRunner:
    """
    Runs multi-episode evaluations of scan policies in the EW environment.
    """

    def __init__(
        self,
        K: int = 35,
        T: int = 2000,
        dwell_us: float = 1000.0,
        switch_us: float = 50.0,
        Pd: float = 0.95,
        Pfa: float = 1e-4,
    ):
        self.K = K
        self.T = T
        self.dwell_us = dwell_us
        self.switch_us = switch_us
        self.Pd = Pd
        self.Pfa = Pfa

    def run_single_episode(
        self,
        scheduler: BaseScheduler,
        seed: int = 42,
        custom_truth_engine: Optional[TruthEngine] = None,
    ) -> EpisodeReport:
        """
        Run a single full episode (T steps) for a given scheduler.
        """
        if custom_truth_engine is not None:
            truth = custom_truth_engine
        else:
            truth = build_default_truth_engine(
                K=self.K, T=self.T,
                dwell_us=self.dwell_us, switch_us=self.switch_us,
                seed=seed, verbose=False,
            )

        env = EWSpectrumEnv(
            truth_engine=truth,
            K=self.K, T=self.T,
            Pd=self.Pd, Pfa=self.Pfa,
            seed=seed,
        )

        scheduler.reset(seed=seed)
        obs, info = env.reset(seed=seed)

        actions: list[int] = []
        hits: list[bool] = []
        rewards: list[float] = []

        terminated = truncated = False
        while not (terminated or truncated):
            action = scheduler.select_band(obs, info)
            next_obs, reward, terminated, truncated, next_info = env.step(action)

            # Sensed hit feedback
            is_active = truth.is_active(action, env.current_step - 1)
            # Detector outcome
            hit_observed = bool(reward > 0) or bool(is_active and env._rng.random() < self.Pd)

            scheduler.update_feedback(action, hit_observed, next_info)

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
        return report

    def evaluate_policies(
        self,
        schedulers: Sequence[BaseScheduler],
        n_episodes: int = 10,
        base_seed: int = 100,
        verbose: bool = True,
    ) -> dict[str, list[EpisodeReport]]:
        """
        Benchmark multiple policies over n_episodes with identical seeds for fairness.
        """
        all_results: dict[str, list[EpisodeReport]] = {s.name: [] for s in schedulers}

        if verbose:
            print(f"[MonteCarloRunner] Running {n_episodes} episodes across {len(schedulers)} policies...")

        for ep in range(n_episodes):
            seed = base_seed + ep
            # Build common truth engine so all schedulers face the exact same RF world
            truth = build_default_truth_engine(
                K=self.K, T=self.T,
                dwell_us=self.dwell_us, switch_us=self.switch_us,
                seed=seed, verbose=False,
            )

            for scheduler in schedulers:
                report = self.run_single_episode(
                    scheduler=scheduler,
                    seed=seed,
                    custom_truth_engine=truth,
                )
                all_results[scheduler.name].append(report)

            if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
                print(f"  Completed episode {ep + 1}/{n_episodes}")

        if verbose:
            self.print_summary_table(all_results)

        return all_results

    def print_summary_table(self, all_results: dict[str, list[EpisodeReport]]) -> None:
        """Print a nicely formatted ASCII summary table of benchmark results."""
        header = f"{'Policy':<22} | {'IR (%)':<12} | {'Mean TTI (s)':<14} | {'Max TTI (s)':<14} | {'Discovery (%)':<14} | {'Reward':<10}"
        sep = "-" * len(header)
        print("\n" + sep)
        print("          BENCHMARK FIGURES OF MERIT SUMMARY")
        print(sep)
        print(header)
        print(sep)

        for policy_name, reports in all_results.items():
            irs = [r.overall_interception_ratio * 100.0 for r in reports]
            ttis = [r.mean_time_to_intercept_sec for r in reports]
            max_ttis = [r.max_time_to_intercept_sec for r in reports]
            disc = [r.discovery_rate * 100.0 for r in reports]
            rews = [r.total_reward for r in reports]

            ir_str = f"{np.mean(irs):.1f} ± {np.std(irs):.1f}"
            tti_str = f"{np.mean(ttis):.3f} ± {np.std(ttis):.3f}"
            max_tti_str = f"{np.mean(max_ttis):.3f} ± {np.std(max_ttis):.3f}"
            disc_str = f"{np.mean(disc):.1f}%"
            rew_str = f"{np.mean(rews):.1f}"

            print(f"{policy_name:<22} | {ir_str:<12} | {tti_str:<14} | {max_tti_str:<14} | {disc_str:<14} | {rew_str:<10}")
        print(sep + "\n")

    def plot_comparison(
        self,
        all_results: dict[str, list[EpisodeReport]],
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """
        Generate publication-grade comparative performance figures.
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        policies = list(all_results.keys())
        palette = ["#457B9D", "#E76F51", "#2A9D8F", "#E9C46A", "#6A0572"]

        # 1. Interception Ratio Boxplot / Bar
        ax1 = axes[0]
        ir_data = [[r.overall_interception_ratio * 100.0 for r in all_results[p]] for p in policies]
        means_ir = [np.mean(d) for d in ir_data]
        stds_ir = [np.std(d) for d in ir_data]
        x_pos = np.arange(len(policies))

        bars = ax1.bar(x_pos, means_ir, yerr=stds_ir, capsize=5, color=palette[:len(policies)], alpha=0.85, edgecolor="black")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(policies, rotation=20, ha="right", fontsize=9)
        ax1.set_ylabel("Overall Interception Ratio (%)", fontsize=10, fontweight="bold")
        ax1.set_title("Interception Ratio (Higher is Better)", fontsize=11, fontweight="bold")
        ax1.set_ylim(0, 100)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

        # 2. Time-to-Intercept (TTI) Comparison
        ax2 = axes[1]
        tti_data = [[r.mean_time_to_intercept_sec for r in all_results[p]] for p in policies]
        means_tti = [np.mean(d) for d in tti_data]
        stds_tti = [np.std(d) for d in tti_data]

        bars2 = ax2.bar(x_pos, means_tti, yerr=stds_tti, capsize=5, color=palette[:len(policies)], alpha=0.85, edgecolor="black")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(policies, rotation=20, ha="right", fontsize=9)
        ax2.set_ylabel("Mean Time-to-Intercept (seconds)", fontsize=10, fontweight="bold")
        ax2.set_title("Time-to-Intercept (Lower is Better)", fontsize=11, fontweight="bold")
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h + 0.05, f"{h:.2f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")

        # 3. Cumulative Emitter Discovery Curves over Episode
        ax3 = axes[2]
        for idx, (policy, reports) in enumerate(all_results.items()):
            all_curves = np.array([r.cumulative_discovery_curve for r in reports])
            mean_curve = np.mean(all_curves, axis=0)
            t_axis = np.arange(len(mean_curve)) * reports[0].T_slot_sec
            ax3.plot(t_axis, mean_curve, label=policy, color=palette[idx % len(palette)], linewidth=2)

        ax3.set_xlabel("Time (seconds)", fontsize=10)
        ax3.set_ylabel("Emitters Discovered", fontsize=10, fontweight="bold")
        ax3.set_title("Discovery Rate Convergence", fontsize=11, fontweight="bold")
        ax3.legend(loc="lower right", fontsize=8)
        ax3.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[MonteCarloRunner] Comparison figure saved → {save_path}")

        return fig
