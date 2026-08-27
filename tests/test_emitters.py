"""
tests/test_emitters.py
Tests for all emitter classes.
"""

import math
import numpy as np
import pytest

from ew_sim.emitters import (
    FixedFrequencyEmitter,
    FHSSEmitter,
    ScanningEmitter,
    make_default_scenario,
)


# ── FixedFrequencyEmitter ───────────────────────────────────────────────────

class TestFixedFrequencyEmitter:
    def setup_method(self):
        self.emitter = FixedFrequencyEmitter(
            emitter_id=0, band_index=5,
            pri_sec=1e-3,       # 1 ms PRI
            pulse_width=2e-6,   # 2 µs pulse
        )

    def test_correct_band_returned(self):
        band, active = self.emitter.state_at(0.0)
        assert band == 5

    def test_active_at_pulse_start(self):
        # t=0 should be inside the first pulse
        band, active = self.emitter.state_at(0.0)
        assert active is True

    def test_inactive_between_pulses(self):
        # t = pulse_width + epsilon → after pulse, before next PRI
        t = 2e-6 + 1e-7
        band, active = self.emitter.state_at(t)
        assert active is False

    def test_active_at_second_pulse(self):
        t = 1e-3   # exactly PRI → start of 2nd pulse
        band, active = self.emitter.state_at(t)
        assert active is True

    def test_off_before_start(self):
        e = FixedFrequencyEmitter(0, 5, start_time=1.0)
        _, active = e.state_at(0.5)
        assert active is False

    def test_off_after_end(self):
        e = FixedFrequencyEmitter(0, 5, end_time=0.5)
        _, active = e.state_at(0.6)
        assert active is False


# ── FHSSEmitter ─────────────────────────────────────────────────────────────

class TestFHSSEmitter:
    def setup_method(self):
        self.emitter = FHSSEmitter(
            emitter_id=1,
            hop_bands=[3, 7, 12, 20],
            hop_interval=5e-3,
            pri_sec=1e-3,
            pulse_width=2e-6,
            burst_size=4,
            rng=np.random.default_rng(99),
        )

    def test_band_within_hop_set(self):
        for t_ms in [0, 5, 10, 15, 20]:
            band, _ = self.emitter.state_at(t_ms * 1e-3)
            assert band in [3, 7, 12, 20], f"Unexpected band {band} at t={t_ms}ms"

    def test_deterministic_replay(self):
        """Same emitter queried twice must give same band sequence."""
        seq1 = [self.emitter.state_at(i * 5e-3)[0] for i in range(10)]
        seq2 = [self.emitter.state_at(i * 5e-3)[0] for i in range(10)]
        assert seq1 == seq2

    def test_pulse_active_at_start_of_hop(self):
        band, active = self.emitter.state_at(0.0)
        assert active is True

    def test_inactive_after_burst(self):
        # After 4 pulses × 1ms PRI = 4ms into a 5ms hop window
        t = 4.1e-3   # just after 4th pulse ends, still within hop window
        band, active = self.emitter.state_at(t)
        assert active is False


# ── ScanningEmitter ─────────────────────────────────────────────────────────

class TestScanningEmitter:
    def setup_method(self):
        self.emitter = ScanningEmitter(
            emitter_id=2,
            band_index=11,
            T_scan_sec=3.0,
            beamwidth_deg=2.0,
            pri_sec=1e-3,
            pulse_width=2e-6,
            initial_angle=0.0,         # beam starts ON the receiver
            gain_threshold=0.5,
        )

    def test_tot_formula(self):
        expected_tot = (2.0 / 360.0) * 3.0   # ≈ 16.67 ms
        assert abs(self.emitter.TOT - expected_tot) < 1e-9

    def test_mainlobe_active_at_t0(self):
        """initial_angle=0 → beam on receiver at t=0."""
        band, active = self.emitter.state_at(0.0)
        assert band == 11
        assert active is True

    def test_sidelobe_inactive(self):
        """At t = T_scan/4 = 0.75s, beam has rotated 90° → sidelobe."""
        band, active = self.emitter.state_at(0.75)
        assert active is False

    def test_mainlobe_returns_after_one_rotation(self):
        """After one full rotation, mainlobe should illuminate again."""
        band, active = self.emitter.state_at(3.0)
        assert active is True

    def test_band_always_correct(self):
        for t in [0.0, 0.5, 1.0, 2.0, 3.0]:
            band, _ = self.emitter.state_at(t)
            assert band == 11


# ── make_default_scenario ───────────────────────────────────────────────────

def test_default_scenario_returns_five_emitters():
    emitters = make_default_scenario(K=35)
    assert len(emitters) == 5


def test_default_scenario_all_bands_in_range():
    emitters = make_default_scenario(K=35)
    for e in emitters:
        assert 0 <= e.primary_band < 35
