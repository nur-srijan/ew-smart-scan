"""
schedulers/baselines.py
=======================
Classical and Open-Loop EW Receiver Scan Schedulers.

Provides baseline strategies representing classical Electronic Support (ES) methods:
    1. SequentialSweep: Uniform, sequential sweep through bands (0 -> K-1 -> 0).
    2. PseudoRandomSweep: Randomized permutation sweep (Costas/Welch style).
    3. PriorityQueueSweep: Dwells on sub-bands proportional to static EDB weights.
    4. UniformRandomSweep: Independent random sampling across bands.

All schedulers conform to the BaseScheduler interface:
    select_band(obs: np.ndarray, info: Optional[dict]) -> int
    update_feedback(action: int, hit: bool, info: Optional[dict]) -> None
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseScheduler(ABC):
    """Abstract base class for all receiver scan schedulers."""

    def __init__(self, K: int, name: str = "BaseScheduler", seed: Optional[int] = None):
        self.K = K
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal state at the start of an evaluation episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0

    @abstractmethod
    def _choose_band(self, obs: np.ndarray, info: Optional[dict] = None) -> int:
        """Internal band selection logic."""
        ...

    def select_band(self, obs: np.ndarray, info: Optional[dict] = None) -> int:
        """
        Choose the next sub-band k in {0, ..., K-1} to tune the receiver.
        Advances internal time step counter self.t.
        """
        band = self._choose_band(obs, info)
        self.t += 1
        return band

    def update_feedback(self, action: int, hit: bool, info: Optional[dict] = None) -> None:
        """
        Receive feedback on whether a pulse was intercepted on the chosen sub-band.
        """
        pass


# ---------------------------------------------------------------------------
# 1. Sequential Sweep (Open-loop Uniform Baseline)
# ---------------------------------------------------------------------------

class SequentialSweep(BaseScheduler):
    """
    Standard sequential frequency sweeper: cycles 0 -> 1 -> ... -> K-1 -> 0.
    """

    def __init__(self, K: int, reverse: bool = False, seed: Optional[int] = None):
        super().__init__(K, name="SequentialSweep", seed=seed)
        self.reverse = reverse

    def _choose_band(self, obs: np.ndarray, info: Optional[dict] = None) -> int:
        if self.reverse:
            band = (self.K - 1 - (self.t % self.K))
        else:
            band = self.t % self.K
        return band


# ---------------------------------------------------------------------------
# 2. Pseudo-Random Sweep (Costas / Welch Permutation)
# ---------------------------------------------------------------------------

class PseudoRandomSweep(BaseScheduler):
    """
    Pseudo-random permutation sweeper. Cycles through permutations of K bands.
    """

    def __init__(self, K: int, reshuffle_every_cycle: bool = True, seed: Optional[int] = None):
        super().__init__(K, name="PseudoRandomSweep", seed=seed)
        self.reshuffle = reshuffle_every_cycle
        self._permutation = self.rng.permutation(self.K)

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self._permutation = self.rng.permutation(self.K)

    def _choose_band(self, obs: np.ndarray, info: Optional[dict] = None) -> int:
        idx_in_cycle = self.t % self.K
        if idx_in_cycle == 0 and self.t > 0 and self.reshuffle:
            self._permutation = self.rng.permutation(self.K)
        return int(self._permutation[idx_in_cycle])


# ---------------------------------------------------------------------------
# 3. Priority Queue Sweep (Pre-Mission Intelligence / Static EDB)
# ---------------------------------------------------------------------------

class PriorityQueueSweep(BaseScheduler):
    """
    Priority-based scheduler driven by static pre-mission data (EDB).
    """

    def __init__(
        self,
        K: int,
        priority_weights: Optional[np.ndarray | list[float]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(K, name="PriorityQueueSweep", seed=seed)
        if priority_weights is None:
            self.weights = np.ones(K) / K
        else:
            w = np.array(priority_weights, dtype=np.float64)
            assert len(w) == K, f"Priority weights length {len(w)} != K ({K})"
            self.weights = w / w.sum()

        self._schedule = self._build_schedule()

    def _build_schedule(self, cycle_length: int = 100) -> np.ndarray:
        counts = np.maximum(1, np.round(self.weights * cycle_length).astype(int))
        schedule = []
        for band, count in enumerate(counts):
            schedule.extend([band] * count)
        return self.rng.permutation(schedule)

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self._schedule = self._build_schedule()

    def _choose_band(self, obs: np.ndarray, info: Optional[dict] = None) -> int:
        band = int(self._schedule[self.t % len(self._schedule)])
        return band


# ---------------------------------------------------------------------------
# 4. Pure Random (Uniform Monte Carlo Search)
# ---------------------------------------------------------------------------

class UniformRandomSweep(BaseScheduler):
    """Randomly selects a band independently at every time slot."""

    def __init__(self, K: int, seed: Optional[int] = None):
        super().__init__(K, name="UniformRandomSweep", seed=seed)

    def _choose_band(self, obs: np.ndarray, info: Optional[dict] = None) -> int:
        return int(self.rng.integers(0, self.K))
