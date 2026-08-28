"""
tests/test_predictor.py
Unit tests for OnlinePeriodicityEstimator and HybridPredictiveScheduler.
"""

import numpy as np
import pytest

from schedulers.predictor import (
    OnlinePeriodicityEstimator,
    HybridPredictiveScheduler,
)


class TestOnlinePeriodicityEstimator:
    def test_pri_estimation_from_periodic_hits(self):
        estimator = OnlinePeriodicityEstimator(K=10)
        band = 2
        # Feed synthetic hits every 5 slots (PRI = 5)
        for slot in range(0, 50, 5):
            estimator.record_hit(band, slot)

        assert band in estimator.estimated_pri_slots
        assert estimator.estimated_pri_slots[band] == pytest.approx(5.0, abs=0.5)

    def test_scan_period_estimation(self):
        estimator = OnlinePeriodicityEstimator(K=10)
        band = 4
        # Burst 1: slots 100, 101, 102
        for s in [100, 101, 102]:
            estimator.record_hit(band, s)

        # Burst 2: slots 300, 301, 302 (period = 200)
        for s in [300, 301, 302]:
            estimator.record_hit(band, s)

        assert band in estimator.estimated_scan_period_slots
        assert estimator.estimated_scan_period_slots[band] == pytest.approx(200.0, abs=5.0)

    def test_is_predicted_active(self):
        estimator = OnlinePeriodicityEstimator(K=10)
        band = 1
        for s in [0, 10, 20, 30, 40]:
            estimator.record_hit(band, s)

        # Next expected pulse at slot 50
        assert estimator.is_predicted_active(band, current_slot=50, tolerance_slots=1)
        # Sidelobe / dead-time at slot 55 should be False
        assert not estimator.is_predicted_active(band, current_slot=55, tolerance_slots=1)


class TestHybridPredictiveScheduler:
    def test_hybrid_scheduler_runs(self):
        scheduler = HybridPredictiveScheduler(K=10, seed=42)
        obs = np.zeros(30)
        actions = [scheduler.select_band(obs) for _ in range(25)]
        for a in actions:
            assert 0 <= a < 10
