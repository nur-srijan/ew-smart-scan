"""
tests/test_multi_receiver.py
============================
Unit tests for Multi-Receiver Cooperative EW Spectrum Environment and Schedulers.

Verifies:
    1. Gymnasium compliance (check_env) for multi-channel environment.
    2. Action and observation space shapes and boundary validation.
    3. Multi-channel step execution, reward calculation, and pulse deduplication.
    4. Tuner collision penalties on redundant band allocations.
    5. Multi-channel baseline schedulers (MultiSequentialSweep, MultiPseudoRandomSweep).
    6. MultiWhittleIndexScheduler top-M selection, Bayesian belief, and AoI tracking.
    7. FoMEvaluator 2D trajectory metrics with collision rate and overall IR.
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from ew_sim.multi_env import MultiReceiverEWSpectrumEnv, DynamicMultiReceiverEnv
from ew_sim.truth_engine import build_default_truth_engine
from schedulers.multi_schedulers import (
    MultiSequentialSweep,
    MultiPseudoRandomSweep,
    MultiWhittleIndexScheduler,
)
from eval.fom import FoMEvaluator


class TestMultiReceiverEnv:
    """Test suite for MultiReceiverEWSpectrumEnv and DynamicMultiReceiverEnv."""

    def test_gymnasium_compliance(self):
        """Verify Gymnasium env checker passes without errors."""
        env = MultiReceiverEWSpectrumEnv(K=12, M=3, T=50, seed=42)
        check_env(env.unwrapped)

        dyn_env = DynamicMultiReceiverEnv(K=12, M=3, T=50, seed=42)
        check_env(dyn_env.unwrapped)

    def test_action_and_observation_spaces(self):
        """Verify MultiDiscrete and Box observation shapes."""
        K, M = 35, 4
        env = MultiReceiverEWSpectrumEnv(K=K, M=M, T=100)
        assert env.action_space.shape == (M,)
        assert env.action_space.nvec.tolist() == [K] * M
        assert env.observation_space.shape == (3 * K,)

    def test_step_execution_and_info_keys(self):
        """Verify step returns valid observation, scalar reward, and diagnostic info."""
        env = MultiReceiverEWSpectrumEnv(K=20, M=4, T=50, seed=123)
        obs, info = env.reset(seed=123)

        assert obs.shape == (60,)
        assert np.all((obs >= 0.0) & (obs <= 1.0))
        assert info["num_tuners"] == 4
        assert info["total_collisions"] == 0

        action = np.array([0, 1, 2, 3])
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        assert next_obs.shape == (60,)
        assert isinstance(reward, float)
        assert not terminated
        assert next_info["t"] == 1
        assert next_info["total_dwells"] == 4

    def test_collision_detection_and_penalty(self):
        """Verify duplicate tuner assignments trigger collision penalties."""
        env = MultiReceiverEWSpectrumEnv(K=20, M=4, T=50, w_collision=5.0, seed=42)
        env.reset(seed=42)

        # Action with all 4 tuners on the same band (3 collisions)
        action_collide = np.array([5, 5, 5, 5])
        _, reward_collide, _, _, info_collide = env.step(action_collide)

        assert info_collide["total_collisions"] == 3

        # Action with 4 unique bands (0 collisions)
        env.reset(seed=42)
        action_unique = np.array([0, 1, 2, 3])
        _, reward_unique, _, _, info_unique = env.step(action_unique)

        assert info_unique["total_collisions"] == 0
        # Colliding action should have significant penalty compared to unique
        assert reward_collide < reward_unique

    def test_parallel_aoi_reset(self):
        """Verify AoI resets to 0 for all M visited bands and increases for unvisited."""
        K, M = 10, 3
        env = MultiReceiverEWSpectrumEnv(K=K, M=M, T=20, seed=42)
        env.reset(seed=42)

        # First step visit bands 0, 1, 2
        env.step(np.array([0, 1, 2]))
        assert np.all(env._aoi[[0, 1, 2]] == 0.0)
        assert np.all(env._aoi[3:] == 1.0)

        # Second step visit bands 3, 4, 5
        env.step(np.array([3, 4, 5]))
        assert np.all(env._aoi[[3, 4, 5]] == 0.0)
        assert np.all(env._aoi[[0, 1, 2]] == 1.0)
        assert np.all(env._aoi[6:] == 2.0)

    def test_dynamic_multi_env_randomization(self):
        """Verify dynamic multi-receiver env randomizes emitter configurations on reset."""
        env = DynamicMultiReceiverEnv(K=25, M=4, T=50, seed=1)
        env.reset(seed=10)
        bands_reset1 = [e.primary_band for e in env.truth._emitters]

        env.reset(seed=20)
        bands_reset2 = [e.primary_band for e in env.truth._emitters]

        assert bands_reset1 != bands_reset2


class TestMultiSchedulers:
    """Test suite for multi-receiver scheduling policies."""

    def test_multi_sequential_sweep_zero_collisions(self):
        """Verify comb sweep never assigns two tuners to the same band."""
        K, M = 35, 4
        scheduler = MultiSequentialSweep(K=K, M=M)
        scheduler.reset()

        all_actions = []
        for t in range(50):
            action = scheduler.select_bands(np.zeros(3 * K))
            assert len(action) == M
            assert len(np.unique(action)) == M, f"Collision detected at step {t}: {action}"
            assert np.all((action >= 0) & (action < K))
            all_actions.append(action)

        # In 9 steps (ceil(35/4)), all 35 bands should be visited
        covered_first_9_steps = set(np.concatenate(all_actions[:9]))
        assert len(covered_first_9_steps) == K

    def test_multi_pseudorandom_sweep_zero_collisions(self):
        """Verify pseudo-random permutation sweep has zero tuner collisions."""
        K, M = 35, 4
        scheduler = MultiPseudoRandomSweep(K=K, M=M, seed=42)
        scheduler.reset(seed=42)

        for _ in range(50):
            action = scheduler.select_bands(np.zeros(3 * K))
            assert len(action) == M
            assert len(np.unique(action)) == M
            assert np.all((action >= 0) & (action < K))

    def test_multi_whittle_scheduler_top_m_selection(self):
        """Verify MultiWhittleIndexScheduler picks top-M distinct arms and updates feedback."""
        K, M = 35, 4
        scheduler = MultiWhittleIndexScheduler(K=K, M=M, seed=100)
        scheduler.reset()

        # Artificially bias belief on bands 4, 9, 15, 22
        favorite_bands = [4, 9, 15, 22]
        for b in favorite_bands:
            scheduler.belief[b] = 0.95

        action = scheduler.select_bands(np.zeros(3 * K))
        assert len(action) == M
        assert len(np.unique(action)) == M
        # The top 4 should be the favored bands
        assert set(action) == set(favorite_bands)

        # Update feedback with hits
        scheduler.update_feedback(action, hits={4: True, 9: True, 15: False, 22: True})
        assert scheduler.belief[4] > 0.90
        assert scheduler.belief[15] < 0.95  # Dropped from initial 0.95 due to miss
        assert scheduler.aoi[4] == 0.0
        assert scheduler.aoi[0] == 1.0


class TestMultiFoMEvaluation:
    """Test suite for FoMEvaluator multi-channel trajectory processing."""

    def test_multi_receiver_trajectory_evaluation(self):
        """Verify evaluate_trajectory correctly tracks multi-channel metrics."""
        truth = build_default_truth_engine(K=15, T=40, verbose=False, seed=42)
        evaluator = FoMEvaluator(truth)

        # Create trajectory with M=3 tuners
        T, M = 40, 3
        actions = np.zeros((T, M), dtype=int)
        hits = np.zeros((T, M), dtype=bool)
        rewards = np.zeros(T, dtype=float)

        for t in range(T):
            active_bands = [k for k in range(15) if truth.is_active(k, t)]
            if active_bands:
                # Assign tuners to active bands where possible
                for m in range(min(M, len(active_bands))):
                    actions[t, m] = active_bands[m]
                    hits[t, m] = True
                    rewards[t] += 6.0
            else:
                actions[t] = [0, 1, 2]

        report = evaluator.evaluate_trajectory(
            policy_name="MultiOracleTest",
            actions=actions,
            hits=hits,
            rewards=rewards,
        )

        assert report.policy_name == "MultiOracleTest"
        assert report.total_slots == 40
        assert report.num_tuners == 3
        assert report.total_collisions == 0
        assert report.collision_rate == 0.0
        assert report.overall_interception_ratio > 0.0

        summary = report.summary_dict()
        assert summary["Tuners"] == 3
        assert summary["Collision_Rate_%"] == 0.0
        assert "Overall_IR_%" in summary
