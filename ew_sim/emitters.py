"""
ew_sim/emitters.py
==================
Emitter models for the EW Smart Scan simulation.

Three tactical emitter classes are implemented:

    FixedFrequencyEmitter   – constant carrier, constant PRF
    FHSSEmitter             – Markov-chain frequency hopping across a discrete hop-set
    ScanningEmitter         – fixed carrier, rotating antenna beam (spatial scan)

Each emitter exposes:
    state_at(t_sec) -> (band_index: int | None, is_active: bool)

A return of (None, False) means the emitter is not detectable at time t_sec
(e.g. the mainlobe is not pointing toward the receiver, or the emitter is off).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseEmitter(ABC):
    """Abstract base for all emitter types."""

    def __init__(
        self,
        emitter_id: int,
        band_index: int,               # primary sub-band index (0-based)
        start_time: float = 0.0,       # seconds: emitter turns on
        end_time: float = float("inf"),
        rng: Optional[np.random.Generator] = None,
    ):
        self.id = emitter_id
        self.primary_band = band_index
        self.start_time = start_time
        self.end_time = end_time
        self.rng = rng if rng is not None else np.random.default_rng()

    @abstractmethod
    def state_at(self, t_sec: float) -> tuple[Optional[int], bool]:
        """
        Returns (band_index, is_transmitting).
        band_index is None if the emitter is not detectable (below sensitivity,
        mainlobe not pointing toward receiver, etc.).
        """
        ...

    def is_on(self, t_sec: float) -> bool:
        return self.start_time <= t_sec < self.end_time


# ---------------------------------------------------------------------------
# Emitter Class A: Fixed-Frequency Pulsed Radar
# ---------------------------------------------------------------------------

class FixedFrequencyEmitter(BaseEmitter):
    """
    A simple pulsed radar on a fixed carrier frequency.

    The emitter transmits a pulse every PRI seconds.
    The pulse is detectable for pulse_width seconds.

    Parameters
    ----------
    band_index   : sub-band index (0-based) of the carrier frequency
    pri_sec      : Pulse Repetition Interval in seconds (e.g. 1e-3 = 1 ms)
    pulse_width  : Pulse Width in seconds (e.g. 2e-6 = 2 µs)
    pri_jitter   : fractional jitter on PRI (0 = constant, 0.05 = 5% jitter)
    """

    def __init__(
        self,
        emitter_id: int,
        band_index: int,
        pri_sec: float = 1e-3,
        pulse_width: float = 2e-6,
        pri_jitter: float = 0.0,
        **kwargs,
    ):
        super().__init__(emitter_id, band_index, **kwargs)
        self.pri = pri_sec
        self.pulse_width = pulse_width
        self.pri_jitter = pri_jitter

        # Precompute random jitter table for deterministic replay
        self._jitter_seed = int(self.rng.integers(0, 2**31))
        self._jitter_rng = np.random.default_rng(self._jitter_seed)

    def _effective_pri(self, pulse_idx: int) -> float:
        """Return PRI for the n-th pulse (with optional jitter)."""
        if self.pri_jitter == 0.0:
            return self.pri
        jitter = self._jitter_rng.uniform(
            -self.pri_jitter * self.pri,
            +self.pri_jitter * self.pri,
        )
        return self.pri + jitter

    def state_at(self, t_sec: float) -> tuple[Optional[int], bool]:
        if not self.is_on(t_sec):
            return None, False

        t_rel = t_sec - self.start_time
        # Determine which pulse interval we're in
        pulse_idx = int(t_rel / self.pri)
        pulse_start = pulse_idx * self.pri  # approx; jitter makes this deterministic
        pulse_end = pulse_start + self.pulse_width

        # Is t_rel inside the pulse?
        in_pulse = pulse_start <= t_rel < pulse_end
        return (self.primary_band, True) if in_pulse else (self.primary_band, False)


# ---------------------------------------------------------------------------
# Emitter Class B: Frequency-Hopping Spread Spectrum (FHSS)
# ---------------------------------------------------------------------------

class FHSSEmitter(BaseEmitter):
    """
    A frequency-agile emitter that hops across a discrete set of sub-bands
    according to a Markov chain transition matrix.

    Parameters
    ----------
    hop_bands      : list of sub-band indices the emitter can hop to
    transition_mat : K×K Markov transition matrix (rows sum to 1).
                     If None, a uniform random matrix is generated.
    hop_interval   : dwell time on each frequency before hopping (seconds)
    pulse_width    : pulse duration within each hop interval (seconds)
    burst_size     : number of pulses per hop before frequency change
    """

    def __init__(
        self,
        emitter_id: int,
        hop_bands: list[int],
        transition_mat: Optional[np.ndarray] = None,
        hop_interval: float = 5e-3,
        pulse_width: float = 2e-6,
        pri_sec: float = 1e-3,
        burst_size: int = 5,
        **kwargs,
    ):
        # Use first band as nominal primary for base class
        super().__init__(emitter_id, hop_bands[0], **kwargs)
        self.hop_bands = list(hop_bands)
        self.hop_interval = hop_interval
        self.pulse_width = pulse_width
        self.pri = pri_sec
        self.burst_size = burst_size

        n = len(hop_bands)
        if transition_mat is not None:
            assert transition_mat.shape == (n, n), "Transition matrix size mismatch"
            self._P = transition_mat.copy()
        else:
            # Random Markov matrix: each row sums to 1
            raw = self.rng.random((n, n)) + 0.1
            self._P = raw / raw.sum(axis=1, keepdims=True)

        # Replay-deterministic hop sequence cache
        self._hop_cache: dict[int, int] = {}   # hop_slot → band_list_index
        self._hop_cache[0] = 0
        self._current_hop_idx = 0

    def _band_at_hop(self, hop_slot: int) -> int:
        """Return the band list index for hop slot `hop_slot` (Markov chain)."""
        if hop_slot in self._hop_cache:
            return self._hop_cache[hop_slot]
        # Build forward from last cached point
        last_cached = max(self._hop_cache.keys())
        state = self._hop_cache[last_cached]
        rng = np.random.default_rng(hash((self.id, last_cached)) & 0xFFFFFFFF)
        for slot in range(last_cached + 1, hop_slot + 1):
            state = rng.choice(len(self.hop_bands), p=self._P[state])
            self._hop_cache[slot] = int(state)
        return self._hop_cache[hop_slot]

    def state_at(self, t_sec: float) -> tuple[Optional[int], bool]:
        if not self.is_on(t_sec):
            return None, False

        t_rel = t_sec - self.start_time
        hop_slot = int(t_rel / self.hop_interval)
        band_idx = self._band_at_hop(hop_slot)
        current_band = self.hop_bands[band_idx]

        # Within the hop interval, determine if we're on a pulse
        t_in_hop = t_rel - hop_slot * self.hop_interval
        pulse_idx = int(t_in_hop / self.pri)
        if pulse_idx >= self.burst_size:
            return current_band, False  # inter-burst gap

        pulse_start = pulse_idx * self.pri
        in_pulse = pulse_start <= t_in_hop < pulse_start + self.pulse_width
        return (current_band, True) if in_pulse else (current_band, False)


# ---------------------------------------------------------------------------
# Emitter Class C: Spatially Scanning Radar
# ---------------------------------------------------------------------------

class ScanningEmitter(BaseEmitter):
    """
    A surveillance radar with a mechanically or electronically rotating antenna.

    The emitter transmits continuously on a fixed carrier, but the mainlobe
    only illuminates the receiver for `TOT` seconds every `T_scan` seconds.

    The received power (and thus detectability) is modelled as a sinc² spatial
    pattern. The emitter is marked 'active' only when the normalised gain
    exceeds `gain_threshold`.

    Parameters
    ----------
    band_index      : sub-band of the carrier frequency
    T_scan_sec      : antenna rotation period in seconds (e.g. 3.0 s for 20 RPM)
    beamwidth_deg   : 3-dB full beamwidth in degrees (e.g. 2.0°)
    pri_sec         : pulse repetition interval
    pulse_width     : pulse width in seconds
    initial_angle   : beam pointing angle at t=start_time (degrees, 0–360)
    gain_threshold  : normalised gain below which mainlobe is deemed undetectable
                      (default 0.5 ≈ −3 dB boundary)
    """

    def __init__(
        self,
        emitter_id: int,
        band_index: int,
        T_scan_sec: float = 3.0,
        beamwidth_deg: float = 2.0,
        pri_sec: float = 1e-3,
        pulse_width: float = 2e-6,
        initial_angle: float = 0.0,
        gain_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(emitter_id, band_index, **kwargs)
        self.T_scan = T_scan_sec
        self.beamwidth_deg = beamwidth_deg
        self.pri = pri_sec
        self.pulse_width = pulse_width
        self.initial_angle = initial_angle
        self.gain_threshold = gain_threshold

        # Pre-compute Time-on-Target
        self.TOT = (beamwidth_deg / 360.0) * T_scan_sec

    def _normalised_gain(self, t_sec: float) -> float:
        """
        Returns normalised antenna gain [0, 1] at receiver at time t_sec.
        Uses sinc² model; gain=1 when beam centre is on receiver (angle=0°).
        Receiver is assumed at boresight = 0°.
        """
        t_rel = t_sec - self.start_time
        # Current beam centre angle (degrees)
        beam_angle = (self.initial_angle + 360.0 * t_rel / self.T_scan) % 360.0
        # Angular offset to receiver (wrap to [-180, 180])
        delta = beam_angle
        if delta > 180.0:
            delta -= 360.0
        # sinc² pattern normalised to 3-dB at ±beamwidth/2
        # sinc²(x) = 0.5 at x = 0.443 → scale accordingly
        half_bw = self.beamwidth_deg / 2.0
        x = 0.443 * delta / half_bw if half_bw > 0 else 0.0
        return float(math.sin(math.pi * x + 1e-12) ** 2 / (math.pi * x + 1e-12) ** 2) \
            if abs(x) > 1e-9 else 1.0

    def state_at(self, t_sec: float) -> tuple[Optional[int], bool]:
        if not self.is_on(t_sec):
            return None, False

        gain = self._normalised_gain(t_sec)
        if gain < self.gain_threshold:
            # Sidelobe: signal below sensitivity → not detectable
            return self.primary_band, False

        # Mainlobe is pointing at receiver — check pulse timing
        t_rel = t_sec - self.start_time
        pulse_idx = int(t_rel / self.pri)
        pulse_start = pulse_idx * self.pri
        in_pulse = pulse_start <= t_rel < pulse_start + self.pulse_width
        return (self.primary_band, True) if in_pulse else (self.primary_band, False)

    @property
    def time_on_target(self) -> float:
        """Theoretical mainlobe illumination window (seconds)."""
        return self.TOT


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_default_scenario(
    K: int = 35,
    rng: Optional[np.random.Generator] = None,
) -> list[BaseEmitter]:
    """
    Returns a canonical mixed-emitter scenario for testing:
      - 2 fixed-frequency pulsed radars (different bands, different PRIs)
      - 2 FHSS emitters (different hop-sets, different rates)
      - 1 spatially scanning surveillance radar

    All emitters are active from t=0 to infinity.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    emitters: list[BaseEmitter] = [
        # ── Fixed-frequency emitters ─────────────────────────────────────
        FixedFrequencyEmitter(
            emitter_id=0, band_index=4,
            pri_sec=1e-3, pulse_width=2e-6,
            rng=np.random.default_rng(1),
        ),
        FixedFrequencyEmitter(
            emitter_id=1, band_index=18,
            pri_sec=3e-3, pulse_width=5e-6,
            pri_jitter=0.02,           # 2% PRI jitter
            rng=np.random.default_rng(2),
        ),
        # ── FHSS emitters ────────────────────────────────────────────────
        FHSSEmitter(
            emitter_id=2,
            hop_bands=[7, 10, 14, 20, 26],
            hop_interval=5e-3,
            pri_sec=1e-3,
            pulse_width=2e-6,
            burst_size=4,
            rng=np.random.default_rng(3),
        ),
        FHSSEmitter(
            emitter_id=3,
            hop_bands=[2, 5, 12, 19, 29, 33],
            hop_interval=2e-3,
            pri_sec=0.5e-3,
            pulse_width=1e-6,
            burst_size=3,
            rng=np.random.default_rng(4),
        ),
        # ── Spatially scanning radar ──────────────────────────────────────
        ScanningEmitter(
            emitter_id=4, band_index=11,
            T_scan_sec=3.0,
            beamwidth_deg=2.0,
            pri_sec=1e-3,
            pulse_width=2e-6,
            initial_angle=45.0,        # beam starts 45° away from receiver
            rng=np.random.default_rng(5),
        ),
    ]
    return emitters
