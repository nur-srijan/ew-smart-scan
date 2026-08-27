"""
tests/test_truth_engine.py
Tests for TruthEngine build, query, and persistence.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ew_sim.emitters import FixedFrequencyEmitter, ScanningEmitter
from ew_sim.truth_engine import TruthEngine, build_default_truth_engine


class TestTruthEngineBasics:
    def setup_method(self):
        self.engine = TruthEngine(K=10, T=200, dwell_us=50, switch_us=5)

    def test_s_shape_after_build(self):
        emitter = FixedFrequencyEmitter(0, 3, pri_sec=1e-3, pulse_width=2e-6)
        self.engine.add_emitter(emitter)
        S = self.engine.build(verbose=False)
        assert S.shape == (10, 200)

    def test_s_dtype_is_int8(self):
        self.engine.build(verbose=False)
        assert self.engine.S.dtype == np.int8

    def test_s_values_binary(self):
        emitter = FixedFrequencyEmitter(0, 3, pri_sec=1e-3, pulse_width=2e-6)
        self.engine.add_emitter(emitter)
        self.engine.build(verbose=False)
        unique = np.unique(self.engine.S)
        assert set(unique).issubset({0, 1})

    def test_fixed_emitter_band_has_activity(self):
        emitter = FixedFrequencyEmitter(0, 4, pri_sec=1e-3, pulse_width=2e-6)
        self.engine.add_emitter(emitter)
        self.engine.build(verbose=False)
        # Band 4 should have some active slots
        assert self.engine.S[4].sum() > 0

    def test_other_bands_quiet_without_emitters(self):
        # Only add emitter on band 7, other bands should be 0
        emitter = FixedFrequencyEmitter(0, 7, pri_sec=1e-3, pulse_width=2e-6)
        self.engine.add_emitter(emitter)
        self.engine.build(verbose=False)
        for k in range(10):
            if k != 7:
                assert self.engine.S[k].sum() == 0, f"Band {k} should be quiet"

    def test_is_active_raises_before_build(self):
        with pytest.raises(RuntimeError):
            self.engine.is_active(0, 0)

    def test_is_active_returns_bool(self):
        emitter = FixedFrequencyEmitter(0, 3, pri_sec=1e-3, pulse_width=2e-6)
        self.engine.add_emitter(emitter)
        self.engine.build(verbose=False)
        result = self.engine.is_active(3, 0)
        assert isinstance(result, bool)


class TestTruthEngineScanning:
    def test_scanning_emitter_has_periodic_gaps(self):
        """Scanning radar should have many silent slots (sidelobes) in its band."""
        engine = TruthEngine(K=20, T=1000, dwell_us=1000, switch_us=100)
        emitter = ScanningEmitter(
            emitter_id=0, band_index=5,
            T_scan_sec=3.0, beamwidth_deg=2.0,
            pri_sec=1e-3, pulse_width=2e-6,
            initial_angle=0.0,
        )
        engine.add_emitter(emitter)
        engine.build(verbose=False)

        # TOT / T_scan ≈ 2/360 ≈ 0.0056 → ~0.56% active
        occ = engine.occupancy_per_band()[5]
        assert occ < 0.05, f"Expected <5% occupancy for scanning radar, got {occ:.2%}"
        assert occ > 0.0,  "Expected some active slots from mainlobe illumination"


class TestTruthEnginePersistence:
    def test_save_and_load(self):
        engine = TruthEngine(K=8, T=50)
        emitter = FixedFrequencyEmitter(0, 2, pri_sec=1e-3, pulse_width=2e-6)
        engine.add_emitter(emitter)
        engine.build(verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_truth"
            engine.save(path)
            loaded = TruthEngine.load(str(path) + ".npz")
            assert np.array_equal(engine.S, loaded.S)
            assert loaded.K == engine.K
            assert loaded.T == engine.T


def test_build_default_truth_engine():
    engine = build_default_truth_engine(K=35, T=500, verbose=False)
    assert engine.S is not None
    assert engine.S.shape == (35, 500)
    assert engine.S.sum() > 0   # At least some active slots
