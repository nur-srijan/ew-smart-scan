"""
tests/test_env.py
Tests for EWSpectrumEnv (Gymnasium API compliance + behaviour).
"""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from ew_sim.env import EWSpectrumEnv
from ew_sim.truth_engine import build_default_truth_engine


@pytest.fixture
def small_env():
    """Tiny environment (K=10, T=100) for fast tests."""
    truth = build_default_truth_engine(K=10, T=100, verbose=False, seed=0)
    # Patch emitter bands to fit K=10
    for e in truth._emitters:
        e.primary_band = e.primary_band % 10
    env = EWSpectrumEnv(truth_engine=truth, K=10, T=100, seed=0)
    return env


class TestEnvAPI:
    def test_check_env(self, small_env):
        """gymnasium env_checker must pass with no warnings."""
        check_env(small_env, warn=True, skip_render_check=True)

    def test_observation_space_shape(self, small_env):
        K = small_env.K
        assert small_env.observation_space.shape == (3 * K,)

    def test_action_space_size(self, small_env):
        assert small_env.action_space.n == small_env.K

    def test_reset_returns_valid_obs(self, small_env):
        obs, info = small_env.reset()
        assert obs.shape == small_env.observation_space.shape
        assert small_env.observation_space.contains(obs)

    def test_step_returns_correct_types(self, small_env):
        small_env.reset()
        obs, reward, terminated, truncated, info = small_env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_obs_always_in_space(self, small_env):
        small_env.reset(seed=42)
        for _ in range(50):
            action = small_env.action_space.sample()
            obs, *_ = small_env.step(action)
            assert small_env.observation_space.contains(obs), \
                f"Observation out of bounds: {obs.min():.4f} … {obs.max():.4f}"

    def test_episode_terminates_at_T(self, small_env):
        small_env.reset()
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = small_env.step(
                small_env.action_space.sample()
            )
            steps += 1
            assert steps <= small_env.T + 1
        assert truncated is True

    def test_belief_sums_to_nonzero(self, small_env):
        small_env.reset()
        for _ in range(10):
            small_env.step(small_env.action_space.sample())
        info = small_env._get_info()
        assert info["belief"].sum() > 0

    def test_aoi_increases_for_unvisited(self, small_env):
        small_env.reset()
        # Always dwell on band 0
        for _ in range(5):
            small_env.step(0)
        # Band 1's AoI should have increased
        assert small_env._aoi[1] > 0

    def test_hit_rate_property(self, small_env):
        small_env.reset(seed=0)
        for _ in range(100):
            small_env.step(small_env.action_space.sample())
        hr = small_env.hit_rate
        assert 0.0 <= hr <= 1.0


class TestReward:
    def test_positive_reward_on_hit(self):
        """Reward should be positive when a pulse is genuinely intercepted."""
        truth = build_default_truth_engine(K=10, T=1000, verbose=False, seed=5)
        for e in truth._emitters:
            e.primary_band = e.primary_band % 10
        env = EWSpectrumEnv(truth_engine=truth, K=10, T=1000,
                            Pd=1.0, Pfa=0.0, seed=0)   # perfect detector
        env.reset(seed=0)

        # Find an active slot
        active_band = active_slot = None
        for t in range(1000):
            for k in range(10):
                if truth.is_active(k, t):
                    active_band, active_slot = k, t
                    break
            if active_band is not None:
                break

        if active_band is None:
            pytest.skip("No active slot found in tiny environment")

        env._t = active_slot
        _, reward, *_ = env.step(active_band)
        assert reward > 0
