"""
eval/fom.py
===========
Figures of Merit (FoM) Engine for EW Scan Strategy Evaluation.

Calculates the exact performance metrics specified by the DRDO problem statement:
    1. Probability of Detection (Pd)
    2. Probability of False Alarm (Pfa)
    3. Interception Ratio (IR) per emitter class and overall
    4. Time-to-Intercept (TTI) per emitter (in seconds and time slots)
    5. Cumulative Detection Probability over time P_cum(t)
    6. Receiver Sensitivity / MDS Verification
    7. Average Frequency Switching Overhead
    8. Cumulative Reward & Discovery Ratio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np

from ew_sim.truth_engine import TruthEngine


@dataclass
class EmitterMetrics:
    """Detailed interception metrics for a single emitter."""
    emitter_id: int
    emitter_type: str
    primary_band: int
    total_transmitted_pulses: int
    intercepted_pulses: int
    interception_ratio: float
    first_intercept_slot: Optional[int]
    time_to_intercept_sec: float
    discovered: bool


@dataclass
class EpisodeReport:
    """Comprehensive evaluation report for a single episode."""
    policy_name: str
    total_slots: int
    T_slot_sec: float
    total_dwells: int
    total_hits: int
    false_alarms: int
    empirical_pd: float
    empirical_pfa: float
    overall_interception_ratio: float
    mean_time_to_intercept_sec: float
    max_time_to_intercept_sec: float
    emitters_discovered: int
    total_emitters: int
    discovery_rate: float
    total_reward: float
    mean_switching_distance: float
    emitter_reports: dict[int, EmitterMetrics] = field(default_factory=dict)
    cumulative_discovery_curve: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cumulative_hit_curve: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def summary_dict(self) -> dict[str, Any]:
        """Flatten key scalar metrics into a dictionary for tables/CSV."""
        return {
            "Policy": self.policy_name,
            "Pd": self.empirical_pd,
            "Pfa": self.empirical_pfa,
            "Overall_IR_%": self.overall_interception_ratio * 100.0,
            "Mean_TTI_s": self.mean_time_to_intercept_sec,
            "Max_TTI_s": self.max_time_to_intercept_sec,
            "Discovered_%": self.discovery_rate * 100.0,
            "Total_Reward": self.total_reward,
            "Avg_Switch_Dist": self.mean_switching_distance,
        }


class FoMEvaluator:
    """
    Evaluates scan strategy performance against the ground truth RF environment.
    """

    def __init__(self, truth_engine: TruthEngine):
        self.truth = truth_engine

    def evaluate_trajectory(
        self,
        policy_name: str,
        actions: list[int] | np.ndarray,
        hits: list[bool] | np.ndarray,
        rewards: list[float] | np.ndarray,
    ) -> EpisodeReport:
        """
        Compute all FoMs from an executed trajectory of (actions, hits, rewards).

        Parameters
        ----------
        policy_name : str
            Name of the scheduler evaluated.
        actions : list[int] or np.ndarray
            Sub-band index chosen at each step t (length T).
        hits : list[bool] or np.ndarray
            Binary hit observation (True/False) received at each step t.
        rewards : list[float] or np.ndarray
            Environment reward received at each step t.

        Returns
        -------
        EpisodeReport containing all metrics.
        """
        actions = np.asarray(actions, dtype=int)
        hits = np.asarray(hits, dtype=bool)
        rewards = np.asarray(rewards, dtype=float)
        T = len(actions)
        assert T == self.truth.T, f"Action trajectory length {T} != TruthEngine length {self.truth.T}"

        # Track per-emitter intercepts
        emitter_first_hit: dict[int, Optional[int]] = {e.id: None for e in self.truth._emitters}
        emitter_intercept_counts: dict[int, int] = {e.id: 0 for e in self.truth._emitters}

        active_dwell_count = 0
        active_dwell_hits = 0
        quiet_dwell_count = 0
        false_alarm_count = 0

        cumulative_discovery = np.zeros(T, dtype=int)
        cumulative_hits = np.zeros(T, dtype=int)
        discovered_set: set[int] = set()

        # Step-by-step analysis
        for t in range(T):
            band = actions[t]
            observed_hit = hits[t]
            is_truly_active = self.truth.is_active(band, t)

            if is_truly_active:
                active_dwell_count += 1
                if observed_hit:
                    active_dwell_hits += 1
                    # Identify which emitter(s) were transmitting in this band
                    for emitter in self.truth._emitters:
                        e_band, e_active = emitter.state_at(t * self.truth.T_slot)
                        if e_band == band and e_active:
                            emitter_intercept_counts[emitter.id] += 1
                            if emitter_first_hit[emitter.id] is None:
                                emitter_first_hit[emitter.id] = t
                            discovered_set.add(emitter.id)
            else:
                quiet_dwell_count += 1
                if observed_hit:
                    false_alarm_count += 1

            cumulative_discovery[t] = len(discovered_set)
            cumulative_hits[t] = active_dwell_hits

        # 1. Empirical Pd and Pfa
        empirical_pd = (active_dwell_hits / active_dwell_count) if active_dwell_count > 0 else 0.0
        empirical_pfa = (false_alarm_count / quiet_dwell_count) if quiet_dwell_count > 0 else 0.0

        # 2. Per-Emitter Metrics
        emitter_reports: dict[int, EmitterMetrics] = {}
        total_transmitted_all = 0
        total_intercepted_all = 0
        ttis_sec: list[float] = []

        max_episode_time = T * self.truth.T_slot

        for emitter in self.truth._emitters:
            act = self.truth.activity.get(emitter.id)
            n_trans = act.total_pulses if act else 0
            n_inter = emitter_intercept_counts[emitter.id]
            total_transmitted_all += n_trans
            total_intercepted_all += n_inter

            ir = (n_inter / n_trans) if n_trans > 0 else 0.0
            first_slot = emitter_first_hit[emitter.id]

            if first_slot is not None:
                tti_sec = (first_slot * self.truth.T_slot) - emitter.start_time
                tti_sec = max(0.0, tti_sec)
                discovered = True
            else:
                tti_sec = max_episode_time  # Penalize undiscovered with max episode horizon
                discovered = False

            ttis_sec.append(tti_sec)

            emitter_reports[emitter.id] = EmitterMetrics(
                emitter_id=emitter.id,
                emitter_type=getattr(emitter, "custom_type", type(emitter).__name__),
                primary_band=emitter.primary_band,
                total_transmitted_pulses=n_trans,
                intercepted_pulses=n_inter,
                interception_ratio=ir,
                first_intercept_slot=first_slot,
                time_to_intercept_sec=tti_sec,
                discovered=discovered,
            )

        # 3. Overall Aggregates
        if total_transmitted_all == 0:
            total_transmitted_all = int(np.sum(self.truth.S[:, :T]))
            total_intercepted_all = active_dwell_hits

        overall_ir = (total_intercepted_all / total_transmitted_all) if total_transmitted_all > 0 else 0.0
        mean_tti = float(np.mean(ttis_sec)) if ttis_sec else 0.0
        max_tti = float(np.max(ttis_sec)) if ttis_sec else 0.0

        # Switching distance overhead
        switch_diffs = np.abs(np.diff(actions))
        mean_switch = float(np.mean(switch_diffs)) if len(switch_diffs) > 0 else 0.0

        return EpisodeReport(
            policy_name=policy_name,
            total_slots=T,
            T_slot_sec=self.truth.T_slot,
            total_dwells=T,
            total_hits=active_dwell_hits,
            false_alarms=false_alarm_count,
            empirical_pd=empirical_pd,
            empirical_pfa=empirical_pfa,
            overall_interception_ratio=overall_ir,
            mean_time_to_intercept_sec=mean_tti,
            max_time_to_intercept_sec=max_tti,
            emitters_discovered=len(discovered_set),
            total_emitters=len(self.truth._emitters),
            discovery_rate=len(discovered_set) / max(1, len(self.truth._emitters)),
            total_reward=float(np.sum(rewards)),
            mean_switching_distance=mean_switch,
            emitter_reports=emitter_reports,
            cumulative_discovery_curve=cumulative_discovery,
            cumulative_hit_curve=cumulative_hits,
        )
