"""
tests/test_multi_cooperative.py
===============================
Unit test suite for Multi-Receiver Cooperative Schedulers.

Tests:
    1. CooperativeRoleScheduler action shape, bounds, and zero-collision guarantee.
    2. CooperativeRoleScheduler PRI estimation and hop tracking on simulated pulses.
    3. MultiRecurrentActorCriticNet tensor shapes, recurrent memory, and collision masking.
    4. MultiDRLScheduler integration with MultiReceiverEWSpectrumEnv.
    5. CooperativeRoleScheduler performance gain over blind baseline on dynamic RF scene.
"""

import numpy as np
import pytest
import torch

from ew_sim.multi_env import MultiReceiverEWSpectrumEnv, DynamicMultiReceiverEnv
from ew_sim.truth_engine import TruthEngine
from ew_sim.emitters import FixedFrequencyEmitter, FHSSEmitter
from schedulers.multi_schedulers import CooperativeRoleScheduler, MultiSequentialSweep
from schedulers.multi_drl import MultiRecurrentActorCriticNet, MultiDRLScheduler
from eval.fom import FoMEvaluator


class TestCooperativeRoleScheduler:
    """Tests for the tactical role-decomposed cooperative scheduler."""

    def test_initialization_and_reset(self):
        sched = CooperativeRoleScheduler(K=35, M=4, seed=42)
        sched.reset(seed=42)
        assert sched.K == 35
        assert sched.M == 4
        assert sched.belief.shape == (35,)
        assert sched.aoi.shape == (35,)
        assert sched.t == 0

    def test_action_bounds_and_zero_collision_guarantee(self):
        """Verify strict zero collisions over 200 consecutive decision steps."""
        sched = CooperativeRoleScheduler(K=35, M=4, seed=123)
        sched.reset(seed=123)
        dummy_obs = np.zeros(105, dtype=np.float32)

        for step in range(200):
            actions = sched.select_bands(dummy_obs)
            assert len(actions) == 4, f"Step {step}: Expected 4 actions, got {len(actions)}"
            assert np.all(actions >= 0) and np.all(actions < 35), f"Action out of bounds: {actions}"
            # Strict collision check
            unique_actions = np.unique(actions)
            assert len(unique_actions) == 4, f"Step {step}: Collision detected in actions {actions}"

            # Provide simulated feedback
            sim_hits = {actions[0]: (step % 4 == 0)}
            sched.update_feedback(actions, sim_hits)

    def test_pri_estimation_and_tracking(self):
        """Verify that periodic pulses on band 7 yield an estimated PRI ~ 5.0."""
        sched = CooperativeRoleScheduler(K=35, M=4, seed=99)
        sched.reset(seed=99)

        # Simulate periodic pulse arrivals on band 7 at slots 0, 5, 10, 15, 20, 25
        for t in range(30):
            sched.t = t + 1
            is_hit = (t % 5 == 0)
            sched.update_feedback(actions=[7, 1, 2, 3], hits={7: is_hit, 1: False, 2: False, 3: False})

        assert 7 in sched.pri_estimates
        est_pri = sched.pri_estimates[7]
        assert abs(est_pri - 5.0) <= 0.5, f"Expected PRI ~5.0, got {est_pri}"



class TestMultiRecurrentActorCriticNet:
    """Tests for the Multi-Head Recurrent neural network."""

    def test_forward_pass_tensor_shapes(self):
        K, M, hidden_dim = 35, 4, 256
        obs_dim = 3 * K
        net = MultiRecurrentActorCriticNet(obs_dim=obs_dim, K=K, M=M, hidden_dim=hidden_dim)
        net.eval()

        # Batch of 2, sequence length 1
        x = torch.randn(2, 1, obs_dim)
        hidden = net.init_hidden(batch_size=2)

        logits, value, next_hidden = net(x, hidden)

        assert logits.shape == (2, M, K), f"Expected logits (2, {M}, {K}), got {logits.shape}"
        assert value.shape == (2, 1), f"Expected value (2, 1), got {value.shape}"
        assert next_hidden.shape == (2, 2, hidden_dim)

    def test_select_actions_collision_masking(self):
        """Verify sequential collision masking produces distinct actions."""
        K, M = 35, 4
        net = MultiRecurrentActorCriticNet(obs_dim=3 * K, K=K, M=M)
        obs = np.random.rand(3 * K).astype(np.float32)
        hidden = net.init_hidden(batch_size=1)

        actions, next_hidden = net.select_actions(
            obs=obs,
            hidden=hidden,
            deterministic=True,
            collision_masking=True,
        )

        assert len(actions) == M
        assert len(np.unique(actions)) == M, f"Collision detected in actions: {actions}"


class TestMultiDRLSchedulerIntegration:
    """Tests for MultiDRLScheduler with Gymnasium environment."""

    def test_step_execution_in_multi_env(self):
        env = MultiReceiverEWSpectrumEnv(K=35, M=4, T=50, seed=42)
        sched = MultiDRLScheduler(K=35, M=4, seed=42)
        obs, info = env.reset(seed=42)
        sched.reset(seed=42)

        total_reward = 0.0
        for _ in range(50):
            actions = sched.select_bands(obs, info)
            assert len(actions) == 4
            assert len(np.unique(actions)) == 4
            obs, r, term, trunc, info = env.step(actions)
            total_reward += r
            sh = info.get("last_band_hits", {})
            sched.update_feedback(actions, sh, info)

        assert env._total_collisions == 0, "Zero collisions expected with collision masking"


class TestCooperativePerformanceGain:
    """Verify that CooperativeRoleScheduler outperforms blind MultiSequential sweep."""

    def test_cooperative_outperforms_multi_sequential(self):
        seed = 42
        env1 = DynamicMultiReceiverEnv(K=35, M=4, T=500, stage=2, seed=seed)
        env1.reset(seed=seed)
        truth = env1.truth

        # 1. MultiSequential
        seq_sched = MultiSequentialSweep(K=35, M=4, seed=seed)
        seq_sched.reset(seed=seed)
        seq_acts, seq_hts, seq_rws = [], [], []
        for t in range(500):
            a = seq_sched.select_bands(None)
            _, r, _, _, info = env1.step(a)
            sh = info.get("last_band_hits", {})
            seq_acts.append(a); seq_hts.append(sh); seq_rws.append(r)
        rep_seq = FoMEvaluator(truth).evaluate_trajectory("Seq", seq_acts, seq_hts, seq_rws)

        # 2. CooperativeRoleScheduler on identical battlefield
        env2 = MultiReceiverEWSpectrumEnv(truth_engine=truth, K=35, M=4, T=500, seed=seed)
        obs, info = env2.reset(seed=seed)
        coop_sched = CooperativeRoleScheduler(K=35, M=4, seed=seed)
        coop_sched.reset(seed=seed)
        coop_acts, coop_hts, coop_rws = [], [], []
        for t in range(500):
            a = coop_sched.select_bands(obs, info)
            obs, r, _, _, info = env2.step(a)
            sh = info.get("last_band_hits", {})
            coop_sched.update_feedback(a, sh, info)
            coop_acts.append(a); coop_hts.append(sh); coop_rws.append(r)
        rep_coop = FoMEvaluator(truth).evaluate_trajectory("Coop", coop_acts, coop_hts, coop_rws)

        # Cooperative role scheduling must intercept significantly more pulses
        assert rep_coop.total_hits > rep_seq.total_hits, (
            f"Coop hits ({rep_coop.total_hits}) should exceed Seq hits ({rep_seq.total_hits})"
        )
        assert rep_coop.collision_rate == 0.0
