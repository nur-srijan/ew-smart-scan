"""
schedulers/multi_schedulers.py
==============================
Cooperative Multi-Receiver Scan Schedulers for EW Electronic Support.

Provides multi-channel scan strategies coordinating M independent tuners across K sub-bands:
    1. BaseMultiScheduler: Abstract base class for multi-receiver policies.
    2. MultiSequentialSweep: Comb partitioning / staggered sweep across K bands (zero collisions).
    3. MultiPseudoRandomSweep: Agile orthogonal permutation sweep across M tuners (zero collisions).
    4. MultiWhittleIndexScheduler: Multi-channel Restless Multi-Armed Bandit (RMAB) policy
       selecting the top-M distinct arms based on closed-form Whittle indices + AoI subsidies.

All schedulers output an action vector of shape (M,) containing integer band assignments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union
import numpy as np


class BaseMultiScheduler(ABC):
    """
    Abstract base class for multi-receiver EW scan schedulers.

    Parameters
    ----------
    K : int
        Total number of frequency sub-bands.
    M : int
        Number of independent receiver tuners/channels.
    name : str
        Human-readable name of the scheduler.
    seed : Optional[int]
        Random seed for stochastic operations.
    """

    def __init__(
        self,
        K: int,
        M: int = 4,
        name: str = "BaseMultiScheduler",
        seed: Optional[int] = None,
    ):
        self.K = K
        self.M = M
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.t: int = 0

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset scheduler state at the start of an evaluation episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0

    @abstractmethod
    def _choose_bands(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        """Internal multi-band selection logic returning an array of M band indices."""
        ...

    def select_bands(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        """
        Choose M sub-bands in {0, ..., K-1} to tune the M receiver channels.
        Advances internal time step counter self.t.
        """
        bands = self._choose_bands(obs, info)
        bands = np.asarray(bands, dtype=int)
        assert len(bands) == self.M, f"Expected {self.M} actions, got {len(bands)}"
        self.t += 1
        return bands

    def update_feedback(
        self,
        actions: np.ndarray | Sequence[int],
        hits: Union[dict[int, bool], np.ndarray, Sequence[bool]],
        info: Optional[dict] = None,
    ) -> None:
        """
        Receive multi-channel feedback on pulse interceptions.

        Parameters
        ----------
        actions : array-like of shape (M,)
            Bands assigned to the M tuners.
        hits : dict[int, bool] or sequence of bool
            Interception result per band or per tuner index.
        info : Optional[dict]
            Environment diagnostic info.
        """
        pass


# ---------------------------------------------------------------------------
# 1. Multi-Receiver Sequential Sweep (Comb Partitioning)
# ---------------------------------------------------------------------------

class MultiSequentialSweep(BaseMultiScheduler):
    """
    Cooperative sequential sweeper across M channels.
    
    Uses comb partitioning: tuner m at step t visits band (t * M + m) % K.
    Guarantees 100% collision-free coverage of the entire spectrum in ceil(K / M) steps.
    """

    def __init__(
        self,
        K: int = 35,
        M: int = 4,
        seed: Optional[int] = None,
    ):
        super().__init__(K=K, M=M, name="MultiSequentialSweep", seed=seed)

    def _choose_bands(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        base = self.t * self.M
        bands = np.array([(base + m) % self.K for m in range(self.M)], dtype=int)
        return bands


# ---------------------------------------------------------------------------
# 2. Multi-Receiver Pseudo-Random Sweep (Orthogonal Permutation)
# ---------------------------------------------------------------------------

class MultiPseudoRandomSweep(BaseMultiScheduler):
    """
    Cooperative pseudo-random permutation sweeper across M channels.
    
    Cycles through random permutations of the K bands M channels at a time.
    Guarantees zero tuner collisions at every step while providing frequency agility.
    """

    def __init__(
        self,
        K: int = 35,
        M: int = 4,
        reshuffle_every_cycle: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(K=K, M=M, name="MultiPseudoRandomSweep", seed=seed)
        self.reshuffle = reshuffle_every_cycle
        self._permutation = self.rng.permutation(self.K)

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self._permutation = self.rng.permutation(self.K)

    def _choose_bands(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        step_in_cycle = (self.t * self.M) % self.K
        if self.reshuffle and self.t > 0 and step_in_cycle < self.M:
            self._permutation = self.rng.permutation(self.K)

        base = self.t * self.M
        bands = np.array([self._permutation[(base + m) % self.K] for m in range(self.M)], dtype=int)
        return bands


# ---------------------------------------------------------------------------
# 3. Multi-Receiver Whittle Index RMAB Scheduler
# ---------------------------------------------------------------------------

class MultiWhittleIndexScheduler(BaseMultiScheduler):
    """
    Restless Multi-Armed Bandit (RMAB) Scheduler for M Cooperative Receivers.

    By the Whittle Index theorem, the optimal decoupled index policy for an
    M-channel bandit capacity constraint is to select the top-M distinct arms
    with the highest Whittle index + Age-of-Information exploration score.

    Features:
        - Parallel Bayesian belief tracking across all K bands.
        - Closed-form Whittle index priority computation per band.
        - Age-of-Information (AoI) exploration subsidy preventing arm starvation.
        - Anti-camping penalties preventing excessive dwell streaks on active bands.
        - Guaranteed zero tuner collisions (selects top-M unique bands).
    """

    def __init__(
        self,
        K: int = 35,
        M: int = 4,
        Pd: float = 0.95,
        Pfa: float = 1e-4,
        aoi_weight: float = 0.60,
        aoi_max: float = 50.0,
        lr_transition: float = 0.05,
        camping_penalty_weight: float = 0.40,
        seed: Optional[int] = None,
    ):
        super().__init__(K=K, M=M, name="MultiWhittleRMAB", seed=seed)
        self.Pd = Pd
        self.Pfa = Pfa
        self.aoi_weight = aoi_weight
        self.aoi_max = aoi_max
        self.lr_transition = lr_transition
        self.camping_penalty_weight = camping_penalty_weight

        # ── State Representation ───────────────────────────────────────────
        self.belief = np.ones(self.K, dtype=np.float64) * 0.1
        self.aoi = np.zeros(self.K, dtype=np.float64)
        self.P01 = np.ones(self.K, dtype=np.float64) * 0.03
        self.P11 = np.ones(self.K, dtype=np.float64) * 0.92

        self._last_state_at_visit = np.zeros(self.K, dtype=int)
        self._last_slot_at_visit = np.zeros(self.K, dtype=int)
        self._last_actions = np.arange(self.M, dtype=int) % self.K
        self._consecutive_dwells = np.zeros(self.K, dtype=int)

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self.belief = np.ones(self.K, dtype=np.float64) * 0.1
        self.aoi = np.zeros(self.K, dtype=np.float64)
        self.P01 = np.ones(self.K, dtype=np.float64) * 0.03
        self.P11 = np.ones(self.K, dtype=np.float64) * 0.92
        self._last_state_at_visit = np.zeros(self.K, dtype=int)
        self._last_slot_at_visit = np.zeros(self.K, dtype=int)
        self._last_actions = np.arange(self.M, dtype=int) % self.K
        self._consecutive_dwells = np.zeros(self.K, dtype=int)

    def compute_whittle_index(self, k: int) -> float:
        """
        Computes the closed-form Whittle Index for sub-band k given current belief b[k].

        Formula:
            delta = P11 - P01
            W(p) = [p * delta + P01] / [1 - delta + p * delta]
        """
        p = float(self.belief[k])
        p01 = float(self.P01[k])
        p11 = float(self.P11[k])

        delta = p11 - p01
        num = p * delta + p01
        denom = (1.0 - delta) + p * delta

        if denom <= 1e-9:
            return float(p)

        index = num / denom
        return float(index)

    def _choose_bands(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        """
        Calculates composite priority scores across all K sub-bands and greedily
        selects the top-M distinct bands.
        """
        scores = np.zeros(self.K, dtype=np.float64)

        for k in range(self.K):
            w_idx = self.compute_whittle_index(k)
            # AoI exploration subsidy
            aoi_bonus = self.aoi_weight * min(1.0, self.aoi[k] / self.aoi_max)
            # Anti-camping penalty
            camping_penalty = self.camping_penalty_weight * self._consecutive_dwells[k]
            tie_breaker = float(self.rng.uniform(0.0, 1e-5))

            scores[k] = w_idx + aoi_bonus - camping_penalty + tie_breaker

        # Greedily select top-M distinct arms
        top_m_bands = np.argsort(scores)[::-1][:self.M].astype(int)

        # Update consecutive dwell tracking
        for k in range(self.K):
            if k in top_m_bands:
                if k in self._last_actions:
                    self._consecutive_dwells[k] += 1
                else:
                    self._consecutive_dwells[k] = 1
            else:
                self._consecutive_dwells[k] = 0

        self._last_actions = top_m_bands.copy()
        return top_m_bands

    def update_feedback(
        self,
        actions: np.ndarray | Sequence[int],
        hits: Union[dict[int, bool], np.ndarray, Sequence[bool]],
        info: Optional[dict] = None,
    ) -> None:
        """
        Updates belief state b[k] and transition probabilities for all M sensed bands,
        and diffuses unobserved bands toward prior.
        """
        actions = np.asarray(actions, dtype=int)
        unique_actions = np.unique(actions)

        # Map band -> hit bool
        band_hit_map: dict[int, bool] = {}
        if isinstance(hits, dict):
            band_hit_map = hits
        else:
            hits_arr = np.asarray(hits)
            for idx, band in enumerate(actions):
                if idx < len(hits_arr):
                    band_hit_map[band] = bool(hits_arr[idx])

        # ── 1. Update Transition Matrix & Bayesian Belief for Sensed Bands ─
        duty = 0.20
        for k in unique_actions:
            hit = band_hit_map.get(k, False)
            observed_state = 1 if hit else 0

            # Transition probability update
            prev_slot = self._last_slot_at_visit[k]
            prev_state = self._last_state_at_visit[k]
            dt = self.t - prev_slot

            if dt <= 10 and self.t > 0:
                if prev_state == 0:
                    target_p01 = float(observed_state)
                    self.P01[k] = (1.0 - self.lr_transition) * self.P01[k] + self.lr_transition * target_p01
                else:
                    target_p11 = float(observed_state)
                    self.P11[k] = (1.0 - self.lr_transition) * self.P11[k] + self.lr_transition * target_p11

            self.P01[k] = np.clip(self.P01[k], 0.01, 0.50)
            self.P11[k] = np.clip(self.P11[k], 0.20, 0.99)

            self._last_state_at_visit[k] = observed_state
            self._last_slot_at_visit[k] = self.t

            # Bayesian update
            p = float(self.belief[k])
            if hit:
                p_hit_present = duty * self.Pd + (1.0 - duty) * self.Pfa
                p_hit_absent = self.Pfa
                posterior = (p * p_hit_present) / max(1e-9, (p * p_hit_present + (1.0 - p) * p_hit_absent))
            else:
                p_miss_present = duty * (1.0 - self.Pd) + (1.0 - duty) * (1.0 - self.Pfa)
                p_miss_absent = 1.0 - self.Pfa
                posterior = (p * p_miss_present) / max(1e-9, (p * p_miss_present + (1.0 - p) * p_miss_absent))

            self.belief[k] = float(np.clip(posterior, 0.001, 0.999))

        # ── 2. Markov State Diffusion for Unsensed Bands ───────────────────
        alpha = 0.02
        prior = 0.15
        unsensed_mask = np.ones(self.K, dtype=bool)
        unsensed_mask[unique_actions] = False

        self.belief[unsensed_mask] = (1.0 - alpha) * self.belief[unsensed_mask] + alpha * prior
        self.belief = np.clip(self.belief, 0.001, 0.999)

        # ── 3. Parallel Age-of-Information Updates ─────────────────────────
        self.aoi += 1.0
        self.aoi[unique_actions] = 0.0


# ---------------------------------------------------------------------------
# 4. Tactical Cooperative Role-Decomposition Scheduler
# ---------------------------------------------------------------------------

class CooperativeRoleScheduler(BaseMultiScheduler):
    """
    Tactical Cooperative Role-Decomposition Scheduler for Multi-Receiver EW Systems.

    Implements dynamic mission task allocation across M independent tuners:
        - Role 0 (Lock-on Tracker): Phase-locked tracking of periodic/fixed radar pulses
          using online Pulse Repetition Interval (PRI) estimation.
        - Role 1 & 2 (Agile FHSS Chaser Pair): Tracks dynamic frequency hoppers by learning
          the empirical Markov hop transition matrix P(f_{t+1} | f_t) and candidate hop sets.
        - Role 3 (Surveillance Sentry): Wideband patrol driven by Age-of-Information (AoI)
          to rapidly discover new threats and catch spatially scanning radar mainlobes.

    Guarantees 0.0% tuner collisions via strict disjoint role assignment.
    """

    def __init__(
        self,
        K: int = 35,
        M: int = 4,
        seed: Optional[int] = None,
    ):
        super().__init__(K=K, M=M, name="CooperativeRoleScheduler", seed=seed)
        self.belief = np.ones(K, dtype=np.float32) * 0.15
        self.aoi = np.zeros(K, dtype=np.float32)
        self.toa_history: dict[int, list[int]] = {k: [] for k in range(K)}
        self.pri_estimates: dict[int, float] = {}
        self.last_hit_slot: dict[int, int] = {}
        self.fixed_tracks: set[int] = set()
        self.fhss_tracks: set[int] = set()
        self.transition_counts = np.zeros((K, K), dtype=np.float32)
        self.last_active_bands: list[int] = []

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self.belief = np.ones(self.K, dtype=np.float32) * 0.15
        self.aoi = np.zeros(self.K, dtype=np.float32)
        self.toa_history = {k: [] for k in range(self.K)}
        self.pri_estimates = {}
        self.last_hit_slot = {}
        self.fixed_tracks = set()
        self.fhss_tracks = set()
        self.transition_counts = np.zeros((self.K, self.K), dtype=np.float32)
        self.last_active_bands = []

    def _choose_bands(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        available_bands = set(range(self.K))
        assigned: list[int] = []

        # ── Role 0: Phase-Locked Pulse Tracker (Fixed / Periodic Emitters) ──
        best_t0 = None
        best_t0_score = -1.0

        for k in self.fixed_tracks:
            if k in available_bands:
                pri = self.pri_estimates.get(k, 5.0)
                last_t = self.last_hit_slot.get(k, self.t)
                phase = (self.t - last_t) % max(1, round(pri))
                score = 12.0 if phase in (0, round(pri) - 1, 1) else 4.0
                if score > best_t0_score:
                    best_t0_score = score
                    best_t0 = k

        if best_t0 is None:
            for k in available_bands:
                pri = self.pri_estimates.get(k)
                if pri is not None and pri >= 2.0:
                    last_t = self.last_hit_slot.get(k, self.t)
                    phase = (self.t - last_t) % max(1, round(pri))
                    score = 10.0 if phase in (0, round(pri) - 1, 1) else 3.0
                    if score > best_t0_score:
                        best_t0_score = score
                        best_t0 = k

        if best_t0 is None:
            # Pick band with highest sustained belief not yet marked as hopping
            for k in available_bands:
                if self.belief[k] > 0.35 and k not in self.fhss_tracks:
                    score = float(self.belief[k]) * 5.0
                    if score > best_t0_score:
                        best_t0_score = score
                        best_t0 = k

        if best_t0 is not None:
            assigned.append(best_t0)
            available_bands.remove(best_t0)

        # ── Role 1 & 2: Agile FHSS Hop Chasers (Markov Transition Bracketing) ──
        fhss_candidates: list[tuple[int, float]] = []

        # 1. Markov transition destinations from recently active bands
        if self.last_active_bands:
            for prev_b in self.last_active_bands:
                row = self.transition_counts[prev_b]
                row_sum = np.sum(row)
                if row_sum > 0:
                    probs = row / row_sum
                    for dest_k in range(self.K):
                        if dest_k in available_bands and probs[dest_k] > 0.05:
                            fhss_candidates.append((dest_k, float(probs[dest_k]) * 10.0))

        # 2. Known FHSS member bands and hot channels
        for k in available_bands:
            if k in self.fhss_tracks:
                fhss_candidates.append((k, float(self.belief[k]) * 8.0 + 3.0))
            elif self.belief[k] > 0.25:
                fhss_candidates.append((k, float(self.belief[k]) * 6.0))

        fhss_candidates.sort(key=lambda x: x[1], reverse=True)
        for cand_band, _ in fhss_candidates:
            if len(assigned) < (self.M - 1) and cand_band in available_bands:
                assigned.append(cand_band)
                available_bands.remove(cand_band)

        # ── Role 3: Age-of-Information Surveillance Sentry ──────────────────
        # Patrol max-AoI bands with anti-resonance micro-dither to catch rotating radars
        aoi_ranked = sorted(
            list(available_bands),
            key=lambda k: float(self.aoi[k]) + float(self.rng.uniform(0.0, 0.4)),
            reverse=True,
        )
        for cand_band in aoi_ranked:
            if len(assigned) < self.M:
                assigned.append(cand_band)
                available_bands.remove(cand_band)
            else:
                break

        # Anti-resonance harmonic dither fallback if any tuner remains unassigned
        dither_base = (self.t * 7 + (self.t % 3) * 11)
        for m in range(self.M):
            if len(assigned) < self.M:
                b = (dither_base + m * 5) % self.K
                if b in available_bands:
                    assigned.append(b)
                    available_bands.remove(b)

        while len(assigned) < self.M:
            b = available_bands.pop()
            assigned.append(b)

        return np.array(assigned, dtype=int)

    def update_feedback(
        self,
        actions: np.ndarray | Sequence[int],
        hits: Union[dict[int, bool], np.ndarray, Sequence[bool]],
        info: Optional[dict] = None,
    ) -> None:
        actions = np.asarray(actions, dtype=int)
        unique_actions = np.unique(actions)

        if isinstance(hits, dict):
            band_hit_map = hits
        elif isinstance(hits, (list, tuple, np.ndarray)):
            band_hit_map = {actions[i]: bool(hits[i]) for i in range(min(len(actions), len(hits)))}
        else:
            band_hit_map = {}

        current_active_bands: list[int] = []

        # 1. Update TOA, PRI estimates, and Markov transition matrix
        for k in unique_actions:
            hit = band_hit_map.get(k, False)
            if hit:
                current_active_bands.append(k)
                self.belief[k] = min(0.98, self.belief[k] * 1.3 + 0.35)
                self.toa_history[k].append(self.t - 1)
                self.last_hit_slot[k] = self.t - 1

                # Update PRI estimate if >= 2 pulses observed
                hist = self.toa_history[k]
                if len(hist) >= 2:
                    deltas = np.diff(hist[-6:])
                    deltas = deltas[deltas >= 2]
                    if len(deltas) >= 1:
                        med = float(np.median(deltas))
                        if med >= 2.0:
                            self.pri_estimates[k] = med
                            # Classify periodic fixed tracks
                            if len(deltas) >= 2 and np.std(deltas) < 1.2:
                                self.fixed_tracks.add(k)

                # Update Markov transition from previous active bands
                if self.last_active_bands:
                    for prev_b in self.last_active_bands:
                        if prev_b != k:
                            self.transition_counts[prev_b, k] += 1.0
                            self.fhss_tracks.add(prev_b)
                            self.fhss_tracks.add(k)
            else:
                # Gentle decay on miss (radar duty cycle is 10-20%)
                self.belief[k] = max(0.05, self.belief[k] * 0.90)

        if current_active_bands:
            self.last_active_bands = current_active_bands

        # 2. Update Age of Information
        self.aoi += 1.0
        self.aoi[unique_actions] = 0.0


