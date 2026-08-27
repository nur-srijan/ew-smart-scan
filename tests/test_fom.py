"""
tests/test_fom.py
Unit tests for FoMEvaluator and MonteCarloRunner.
"""

import numpy as np
import pytest

from ew_sim.truth_engine import build_default_truth_engine
from eval.fom import FoMEvaluator
from eval.runner import MonteCarloRunner
from schedulers.baselines import SequentialSweep, PseudoRandomSweep


class TestFoMEvaluator:
    def test_perfect_trajectory_metrics(self):
        truth = build_default_truth_engine(K=10, T=50, verbose=False, seed=42)
        evaluator = FoMEvaluator(truth)

        # Construct actions that always match active bands whenever possible
        actions = []
        hits = []
        rewards = []

        for t in range(50):
            # Find an active band if any
            active_bands = [k for k in range(10) if truth.is_active(k, t)]
            chosen_band = active_bands[0] if active_bands else 0
            is_active = truth.is_active(chosen_band, t)

            actions.append(chosen_band)
            hits.append(is_active)
            rewards.append(5.0 if is_active else 0.0)

        report = evaluator.evaluate_trajectory(
            policy_name="OracleTest",
            actions=actions,
            hits=hits,
            rewards=rewards,
        )

        assert report.policy_name == "OracleTest"
        assert report.total_slots == 50
        assert report.empirical_pd == 1.0
        assert report.empirical_pfa == 0.0
        assert report.overall_interception_ratio > 0.0


class TestMonteCarloRunner:
    def test_mc_runner_single_and_multi_episode(self):
        runner = MonteCarloRunner(K=10, T=100)
        s1 = SequentialSweep(K=10)
        s2 = PseudoRandomSweep(K=10)

        results = runner.evaluate_policies(
            schedulers=[s1, s2],
            n_episodes=2,
            base_seed=50,
            verbose=False,
        )

        assert "SequentialSweep" in results
        assert "PseudoRandomSweep" in results
        assert len(results["SequentialSweep"]) == 2
        assert len(results["PseudoRandomSweep"]) == 2

        report = results["SequentialSweep"][0]
        assert 0.0 <= report.overall_interception_ratio <= 1.0
        assert report.mean_time_to_intercept_sec >= 0.0
