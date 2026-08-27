"""
tests/test_baselines.py
Unit tests for classical baseline schedulers.
"""

import numpy as np
import pytest

from schedulers.baselines import (
    SequentialSweep,
    PseudoRandomSweep,
    PriorityQueueSweep,
    UniformRandomSweep,
)


class TestBaselineSchedulers:
    def test_sequential_sweep_progression(self):
        K = 10
        scheduler = SequentialSweep(K=K)
        obs = np.zeros(3 * K)

        actions = [scheduler.select_band(obs) for _ in range(25)]
        # Should cycle 0, 1, 2, ..., 9, 0, 1, ...
        expected = [i % K for i in range(25)]
        assert actions == expected

    def test_reverse_sequential_sweep(self):
        K = 5
        scheduler = SequentialSweep(K=K, reverse=True)
        obs = np.zeros(3 * K)
        actions = [scheduler.select_band(obs) for _ in range(7)]
        assert actions == [4, 3, 2, 1, 0, 4, 3]

    def test_pseudorandom_sweep_covers_all_bands(self):
        K = 12
        scheduler = PseudoRandomSweep(K=K, seed=42)
        obs = np.zeros(3 * K)

        # First K actions should be a full permutation of 0..K-1
        cycle1 = [scheduler.select_band(obs) for _ in range(K)]
        assert sorted(cycle1) == list(range(K))

        # Next K actions should also be a full permutation
        cycle2 = [scheduler.select_band(obs) for _ in range(K)]
        assert sorted(cycle2) == list(range(K))

    def test_priority_queue_sweep_distribution(self):
        K = 4
        # Give band 0 70% weight, others 10% each
        weights = [0.7, 0.1, 0.1, 0.1]
        scheduler = PriorityQueueSweep(K=K, priority_weights=weights, seed=42)
        obs = np.zeros(3 * K)

        actions = [scheduler.select_band(obs) for _ in range(500)]
        counts = np.bincount(actions, minlength=K)

        # Band 0 should have substantially more dwells than band 1, 2, 3
        assert counts[0] > 2 * counts[1]
        assert counts[0] > 2 * counts[2]

    def test_uniform_random_sweep_bounds(self):
        K = 15
        scheduler = UniformRandomSweep(K=K, seed=123)
        obs = np.zeros(3 * K)
        actions = [scheduler.select_band(obs) for _ in range(100)]
        for a in actions:
            assert 0 <= a < K
