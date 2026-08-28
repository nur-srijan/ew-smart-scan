"""
tests/test_rmab.py
Unit tests for WhittleIndexScheduler (RMAB).
"""

import numpy as np
import pytest

from schedulers.rmab import WhittleIndexScheduler


class TestWhittleIndexScheduler:
    def setup_method(self):
        self.scheduler = WhittleIndexScheduler(K=10, seed=42)

    def test_whittle_index_bounds(self):
        # Whittle index should increase monotonically with belief p
        self.scheduler.P01[0] = 0.05
        self.scheduler.P11[0] = 0.80

        self.scheduler.belief[0] = 0.1
        idx_low = self.scheduler.compute_whittle_index(0)

        self.scheduler.belief[0] = 0.9
        idx_high = self.scheduler.compute_whittle_index(0)

        assert idx_high > idx_low

    def test_bayesian_belief_update_on_hit_and_miss(self):
        k = 3
        # Sensed hit -> belief increases above prior
        self.scheduler.belief[k] = 0.5
        self.scheduler.update_feedback(action=k, hit=True)
        assert self.scheduler.belief[k] > 0.5
        assert self.scheduler.aoi[k] == 0.0

        # Sensed miss from 0.5 prior -> belief decreases below prior
        self.scheduler.belief[k] = 0.5
        self.scheduler.update_feedback(action=k, hit=False)
        assert self.scheduler.belief[k] < 0.5

    def test_aoi_increases_for_unsensed_bands(self):
        self.scheduler.reset()
        for _ in range(5):
            self.scheduler.update_feedback(action=0, hit=False)

        assert self.scheduler.aoi[0] == 0.0
        assert self.scheduler.aoi[1] == 5.0

    def test_action_bounds(self):
        obs = np.zeros(30)
        for _ in range(50):
            action = self.scheduler.select_band(obs)
            assert 0 <= action < 10
