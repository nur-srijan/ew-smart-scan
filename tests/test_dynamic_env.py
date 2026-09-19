"""
tests/test_dynamic_env.py
=========================
Unit and integration tests for DynamicSpectrumEnv with reset-level domain randomization.
"""

import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env

from ew_sim.env import DynamicSpectrumEnv
from ew_sim.emitters import FixedFrequencyEmitter, FHSSEmitter, ScanningEmitter
from schedulers.baselines import SequentialSweep
from schedulers.rmab import WhittleIndexScheduler


class TestDynamicSpectrumEnv:
    """Tests for the reset-level domain randomized environment."""

    def test_gymnasium_compliance(self):
        env = DynamicSpectrumEnv(K=35, T=100, stage=3, seed=42)
        check_env(env.unwrapped)

    def test_bands_change_across_resets(self):
        env = DynamicSpectrumEnv(K=35, T=100, stage=3, seed=42)
        obs1, _ = env.reset(seed=101)
        bands1 = sorted([e.primary_band for e in env.truth._emitters])

        obs2, _ = env.reset(seed=102)
        bands2 = sorted([e.primary_band for e in env.truth._emitters])

        obs3, _ = env.reset(seed=103)
        bands3 = sorted([e.primary_band for e in env.truth._emitters])

        assert bands1 != bands2 or bands2 != bands3, "Emitter bands must vary across resets"

    def test_stage1_fixed_emitters_only(self):
        env = DynamicSpectrumEnv(K=35, T=50, stage=1, seed=42)
        env.reset(seed=100)
        for emitter in env.truth._emitters:
            assert isinstance(emitter, FixedFrequencyEmitter)

    def test_stage2_contains_fhss(self):
        env = DynamicSpectrumEnv(K=35, T=50, stage=2, seed=42)
        env.reset(seed=200)
        has_fhss = any(isinstance(e, FHSSEmitter) for e in env.truth._emitters)
        assert has_fhss, "Stage 2 should contain FHSS emitters"

    def test_stage3_contains_scanning(self):
        env = DynamicSpectrumEnv(K=35, T=50, stage=3, seed=42)
        env.reset(seed=300)
        has_scan = any(isinstance(e, ScanningEmitter) for e in env.truth._emitters)
        assert has_scan, "Stage 3 should contain scanning radar emitter"

    def test_step_execution_and_rewards(self):
        env = DynamicSpectrumEnv(K=35, T=50, stage=3, seed=42)
        obs, info = env.reset(seed=500)
        assert obs.shape == (3 * 35,)

        total_reward = 0.0
        for _ in range(50):
            action = int(np.random.randint(0, 35))
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            total_reward += reward
            assert next_obs.shape == (3 * 35,)
            assert isinstance(reward, (float, np.floating))

        assert truncated is True, "Episode should truncate at T=50"

    def test_whittle_scheduler_runs_on_dynamic_env(self):
        env = DynamicSpectrumEnv(K=35, T=100, stage=3, seed=42)
        sched = WhittleIndexScheduler(K=35)
        obs, info = env.reset(seed=42)
        sched.reset(seed=42)

        hits = 0
        for _ in range(100):
            action = sched.select_band(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            is_hit = bool(env.truth.is_active(action, env.current_step - 1))
            sched.update_feedback(action, is_hit, info)
            if is_hit:
                hits += 1

        assert env.current_step == 100
