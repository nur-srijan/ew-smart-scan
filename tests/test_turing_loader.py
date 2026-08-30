"""
tests/test_turing_loader.py
Unit tests for TuringDatasetAdapter and SyntheticTuringGenerator.
"""

import pytest
import numpy as np

from ew_sim.turing_loader import (
    PulseDescriptorWord,
    TuringDatasetAdapter,
    SyntheticTuringGenerator,
)


class TestTuringLoader:
    def setup_method(self):
        self.adapter = TuringDatasetAdapter(K=35, T=100, f_min_ghz=0.5, f_max_ghz=18.0)

    def test_freq_to_band_mapping(self):
        # 0.5 GHz should be band 0
        assert self.adapter.freq_to_band(0.5) == 0
        # 17.9 GHz should be band 34
        assert self.adapter.freq_to_band(17.9) == 34
        # Out of bounds should return None
        assert self.adapter.freq_to_band(0.1) is None
        assert self.adapter.freq_to_band(20.0) is None

    def test_toa_to_slot_mapping(self):
        slot0 = self.adapter.toa_to_slot(0.0)
        assert slot0 == 0
        # Check slot advancing
        slot1 = self.adapter.toa_to_slot(self.adapter.T_slot * 5)
        assert slot1 == 5

    def test_pdws_to_truth_engine(self):
        pdws = [
            PulseDescriptorWord(toa_sec=0.0, freq_ghz=2.8, pulse_width_sec=2e-6, amplitude_dbm=-40.0, emitter_id=1),
            PulseDescriptorWord(toa_sec=0.01, freq_ghz=9.8, pulse_width_sec=2e-6, amplitude_dbm=-40.0, emitter_id=2),
        ]
        engine = self.adapter.pdws_to_truth_engine(pdws)
        assert engine.S is not None
        assert engine.S.shape == (35, 100)
        assert engine.S.sum() >= 2

    def test_synthetic_turing_generator(self):
        gen = SyntheticTuringGenerator(seed=42)
        pdws = gen.generate_benchmark_pdws(duration_sec=0.5, num_emitters=4)
        assert len(pdws) > 10
        # Verify chronological ordering
        toas = [p.toa_sec for p in pdws]
        assert toas == sorted(toas)
