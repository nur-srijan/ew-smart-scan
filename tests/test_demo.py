"""
tests/test_demo.py
Unit tests for demo and dashboard components.
"""

import pytest
from ew_sim.truth_engine import build_default_truth_engine
from schedulers.baselines import SequentialSweep
from demo.compare import run_policy_episode
from demo.dashboard import create_scenario, instantiate_scheduler


class TestDemoComponents:
    def test_run_policy_episode_execution(self):
        truth = build_default_truth_engine(K=10, T=50, verbose=False, seed=42)
        sched = SequentialSweep(K=10)
        actions, hits, dwells, report = run_policy_episode(sched, truth, K=10, T=50, seed=42)

        assert len(actions) == 50
        assert len(hits) == 50
        assert len(dwells) == 50
        assert report.total_dwells == 50

    def test_dashboard_create_scenario_presets(self):
        for preset in ["standard_mixed", "dense_agile", "fast_scanning", "turing_synthetic"]:
            engine = create_scenario(preset, K=10, T=50, seed=42)
            assert engine.S is not None
            assert engine.S.shape == (10, 50)

    def test_dashboard_instantiate_scheduler(self):
        for pol in ["SequentialSweep", "PseudoRandomSweep", "WhittleIndexRMAB", "DRLScheduler-RecurrentPPO"]:
            s = instantiate_scheduler(pol, K=10, seed=42)
            assert s is not None
            assert s.K == 10
