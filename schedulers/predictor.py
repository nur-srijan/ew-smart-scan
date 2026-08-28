"""
schedulers/predictor.py
=======================
Online Radar Periodicity, PRI Deinterleaver, and Scan Phase Estimator.

Maintains Time-of-Arrival (TOA) history per frequency sub-band and performs
online autocorrelation and delta-TOA clustering to estimate:
    1. Pulse Repetition Interval (PRI)
    2. Radar Antenna Rotation Period (T_scan)
    3. Mainlobe Time-on-Target (TOT)

Allows the scheduler to schedule targeted dwells precisely when an emitter's
mainlobe is rotating across the receiver, freeing the remaining 95%+ of time
to explore other sub-bands.
"""

from __future__ import annotations

from typing import Optional
from collections import defaultdict
import numpy as np

from schedulers.baselines import BaseScheduler
from schedulers.rmab import WhittleIndexScheduler


class OnlinePeriodicityEstimator:
    """
    Online tracker estimating pulse intervals (PRI) and antenna scan periods (T_scan).
    """

    def __init__(
        self,
        K: int,
        min_hits_for_pri: int = 4,
        min_bursts_for_scan: int = 2,
    ):
        self.K = K
        self.min_hits_for_pri = min_hits_for_pri
        self.min_bursts_for_scan = min_bursts_for_scan

        # Per-band TOA history (list of slot indices where hits occurred)
        self.toa_history: dict[int, list[int]] = defaultdict(list)
        # Per-band burst timestamps (list of burst start slots)
        self.burst_starts: dict[int, list[int]] = defaultdict(list)
        # Per-band burst durations (slots)
        self.burst_lengths: dict[int, list[int]] = defaultdict(list)

        # Estimated parameters
        self.estimated_pri_slots: dict[int, float] = {}
        self.estimated_scan_period_slots: dict[int, float] = {}
        self.estimated_tot_slots: dict[int, float] = {}

    def record_hit(self, band: int, slot: int) -> None:
        """Record an observed pulse detection at the given time slot."""
        history = self.toa_history[band]
        history.append(slot)

        # ── 1. Update Short-Term PRI Estimate ──────────────────────────────
        if len(history) >= self.min_hits_for_pri:
            recent = history[-15:]
            diffs = np.diff(recent)
            # Filter consecutive adjacent hits (same burst or 1-slot PRI)
            valid_diffs = diffs[diffs > 0]
            if len(valid_diffs) > 0:
                # Mode / Median of differences
                pri_est = float(np.median(valid_diffs))
                self.estimated_pri_slots[band] = pri_est

        # ── 2. Detect Multi-Pulse Burst Clusters & Rotation Period ─────────
        # If time since previous hit > 20 slots, treat this as start of a new burst
        if len(history) >= 2:
            gap = slot - history[-2]
            if gap > 20:
                self.burst_starts[band].append(slot)
            elif len(self.burst_starts[band]) == 0:
                self.burst_starts[band].append(history[0])

            # Update scan period estimate if we have multiple burst starts
            bursts = self.burst_starts[band]
            if len(bursts) >= self.min_bursts_for_scan:
                scan_diffs = np.diff(bursts)
                valid_scan_diffs = scan_diffs[scan_diffs > 50]  # Minimum scan rotation threshold
                if len(valid_scan_diffs) > 0:
                    self.estimated_scan_period_slots[band] = float(np.median(valid_scan_diffs))
                    # Estimate TOT as standard burst length (~10-40 slots)
                    self.estimated_tot_slots[band] = 25.0

    def is_predicted_active(self, band: int, current_slot: int, tolerance_slots: int = 5) -> bool:
        """
        Returns True if sub-band `band` is predicted to have an active pulse or
        mainlobe illumination at `current_slot`.
        """
        # Case A: Periodic Antenna Rotation
        if band in self.estimated_scan_period_slots and len(self.burst_starts[band]) > 0:
            T_scan = self.estimated_scan_period_slots[band]
            tot = self.estimated_tot_slots.get(band, 25.0)
            last_burst = self.burst_starts[band][-1]
            
            # Distance from expected periodic arrival: (current_slot - last_burst) mod T_scan
            dt = (current_slot - last_burst) % T_scan
            if dt <= (tot + tolerance_slots) or dt >= (T_scan - tolerance_slots):
                return True

        # Case B: Fast Periodic Pulse (PRI)
        if band in self.estimated_pri_slots and len(self.toa_history[band]) > 0:
            pri = self.estimated_pri_slots[band]
            last_hit = self.toa_history[band][-1]
            dt = (current_slot - last_hit) % pri
            if dt <= 1.0 or dt >= (pri - 1.0):
                return True

        return False


class HybridPredictiveScheduler(BaseScheduler):
    """
    Combines RMAB Whittle Index with the Online Periodicity Estimator.
    If a band is predicted to illuminate in the current time slot, it is given
    immediate high priority; otherwise, falls back to RMAB exploration/exploitation.
    """

    def __init__(
        self,
        K: int,
        Pd: float = 0.95,
        Pfa: float = 1e-4,
        seed: Optional[int] = None,
    ):
        super().__init__(K, name="HybridPredictiveRMAB", seed=seed)
        self.rmab = WhittleIndexScheduler(K=K, Pd=Pd, Pfa=Pfa, seed=seed)
        self.predictor = OnlinePeriodicityEstimator(K=K)

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self.rmab.reset(seed)
        self.predictor = OnlinePeriodicityEstimator(K=self.K)

    def _choose_band(self, obs: np.ndarray, info: Optional[dict] = None) -> int:
        # Check if any band has an active predicted window
        predicted_active_bands = []
        for k in range(self.K):
            if self.predictor.is_predicted_active(k, self.t):
                predicted_active_bands.append(k)

        if len(predicted_active_bands) > 0:
            # If multiple bands are predicted active, pick the one with highest Whittle score
            scores = [self.rmab.compute_whittle_index(k) for k in predicted_active_bands]
            best_k = predicted_active_bands[int(np.argmax(scores))]
            return best_k

        # Fallback to RMAB Whittle Index
        return self.rmab._choose_band(obs, info)

    def update_feedback(self, action: int, hit: bool, info: Optional[dict] = None) -> None:
        if hit:
            self.predictor.record_hit(action, self.t)
        self.rmab.update_feedback(action, hit, info)
