"""
ew_sim/multi_env.py
===================
Multi-Receiver Cooperative EW Spectrum Scanning Environment.

Models a modern Electronic Warfare digital receiver system with M independent
tuners/channels monitoring K sub-bands simultaneously.

Features:
    1. MultiDiscrete([K] * M) action space representing M independent tuner assignments.
    2. Multi-channel parallel detection with pulse deduplication.
    3. Tuner collision / redundancy penalty to incentivize cooperative spectral diversity.
    4. Parallel Bayesian belief and Age-of-Information (AoI) updates across all active tuners.
    5. Dynamic domain randomization across episodes (DynamicMultiReceiverEnv).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ew_sim.truth_engine import TruthEngine, build_default_truth_engine
from ew_sim.emitters import FixedFrequencyEmitter, FHSSEmitter, ScanningEmitter


class MultiReceiverEWSpectrumEnv(gym.Env):
    """
    Gymnasium Environment for Multi-Receiver Cooperative EW Spectrum Scanning.

    Parameters
    ----------
    truth_engine    : Pre-built TruthEngine instance. If None, builds default scenario.
    K               : Number of frequency sub-bands (default: 35).
    M               : Number of independent receiver tuners/channels (default: 4).
    T               : Episode length in time slots (default: 1000).
    Pd              : Probability of detection (true positive rate).
    Pfa             : Probability of false alarm (false positive rate).
    w_hit           : Reward weight per intercepted radar pulse.
    w_new           : Reward bonus for discovering a previously undetected emitter.
    w_aoi           : Exploration bonus weight based on Age of Information.
    w_switch        : Frequency switching penalty per tuner normalized by (K-1).
    w_collision     : Penalty per redundant tuner pair occupying the same band.
    aoi_max         : Maximum AoI cap for normalization.
    seed            : Random seed for detector noise.
    render_mode     : "human" or None.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        truth_engine: Optional[TruthEngine] = None,
        K: int = 35,
        M: int = 4,
        T: int = 1000,
        Pd: float = 0.95,
        Pfa: float = 1e-4,
        w_hit: float = 6.0,
        w_new: float = 40.0,
        w_aoi: float = 2.5,
        w_switch: float = 0.2,
        w_collision: float = 2.0,
        aoi_max: int = 200,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.K = K
        self.M = M
        self.T = T
        self.Pd = Pd
        self.Pfa = Pfa
        self.w_hit = w_hit
        self.w_new = w_new
        self.w_aoi = w_aoi
        self.w_switch = w_switch
        self.w_collision = w_collision
        self.aoi_max = aoi_max

        if truth_engine is not None:
            self.truth = truth_engine
            self.K = truth_engine.K
        else:
            self.truth = build_default_truth_engine(
                K=self.K, T=self.T, verbose=False,
                seed=seed if seed is not None else 42,
            )
            self.K = self.truth.K

        # ── Spaces ─────────────────────────────────────────────────────────
        # Action: M independent tuner band assignments in {0, ..., K-1}
        self.action_space = spaces.MultiDiscrete([self.K] * self.M)

        # Observation: [belief(K) | aoi_norm(K) | last_action_multihot(K)]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3 * self.K,),
            dtype=np.float32,
        )

        # ── Internal State ─────────────────────────────────────────────────
        self._rng = np.random.default_rng(seed)
        self._t: int = 0
        self._belief: np.ndarray = np.ones(self.K, dtype=np.float32) * 0.5
        self._aoi: np.ndarray = np.zeros(self.K, dtype=np.float32)
        self._last_actions: np.ndarray = np.arange(self.M, dtype=int) % self.K
        self._consecutive_dwells: np.ndarray = np.zeros(self.K, dtype=int)
        self._discovered: set[int] = set()

        # Episode statistics
        self._total_hits: int = 0
        self._total_dwells: int = 0
        self._total_collisions: int = 0
        self._false_alarms: int = 0
        self._last_band_hits: dict[int, bool] = {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 0
        self._belief = np.ones(self.K, dtype=np.float32) * 0.5
        self._aoi = np.zeros(self.K, dtype=np.float32)
        self._last_actions = np.arange(self.M, dtype=int) % self.K
        self._consecutive_dwells = np.zeros(self.K, dtype=int)
        self._discovered = set()
        self._total_hits = 0
        self._total_dwells = 0
        self._total_collisions = 0
        self._false_alarms = 0
        self._last_band_hits = {}

        return self._get_obs(), self._get_info()


    def step(self, action: np.ndarray | Sequence[int]):
        action = np.asarray(action, dtype=int)
        assert self.action_space.contains(action), f"Invalid multi-receiver action {action}"

        # 1. Collision Detection (Duplicate Tuners on same band)
        unique_bands, counts = np.unique(action, return_counts=True)
        num_redundant_tuners = int(np.sum(counts - 1))
        self._total_collisions += num_redundant_tuners

        # 2. Parallel Signal Interception & Pulse Deduplication
        band_hits: dict[int, bool] = {}
        band_true_active: dict[int, bool] = {}
        new_discoveries = 0

        for band in unique_bands:
            is_active = self.truth.is_active(band, self._t)
            band_true_active[band] = is_active
            if is_active:
                hit = bool(self._rng.random() < self.Pd)
            else:
                hit = bool(self._rng.random() < self.Pfa)
                if hit:
                    self._false_alarms += 1
            band_hits[band] = hit

            # Discovery check
            if hit and is_active:
                for emitter in self.truth._emitters:
                    if emitter.primary_band == band and emitter.id not in self._discovered:
                        self._discovered.add(emitter.id)
                        new_discoveries += 1

        # 3. Update consecutive dwell counters per band
        for k in range(self.K):
            if k in unique_bands:
                self._consecutive_dwells[k] += 1
            else:
                self._consecutive_dwells[k] = 0

        # 4. Compute Joint Multi-Receiver Reward
        reward = self._compute_reward(
            action=action,
            unique_bands=unique_bands,
            band_hits=band_hits,
            band_true_active=band_true_active,
            new_discoveries=new_discoveries,
            num_redundant_tuners=num_redundant_tuners,
        )

        # 5. Parallel Bayesian Belief Updates for all observed bands
        for band in unique_bands:
            self._update_belief(band, band_hits[band])

        # Decay unvisited bands toward prior
        unvisited_mask = np.ones(self.K, dtype=bool)
        unvisited_mask[unique_bands] = False
        alpha = 0.02
        prior = 0.15
        self._belief[unvisited_mask] = (1.0 - alpha) * self._belief[unvisited_mask] + alpha * prior

        # 6. Parallel Age-of-Information (AoI) Updates
        self._aoi += 1.0
        self._aoi[unique_bands] = 0.0
        self._aoi = np.clip(self._aoi, 0.0, self.aoi_max)

        # 7. Update metrics
        self._last_band_hits = band_hits.copy()
        hits_this_step = sum(1 for b, hit in band_hits.items() if hit and band_true_active[b])
        self._total_hits += hits_this_step
        self._total_dwells += len(unique_bands)
        self._last_actions = action.copy()
        self._t += 1
        truncated = self._t >= self.T

        return self._get_obs(), reward, False, truncated, self._get_info()


    def _compute_reward(
        self,
        action: np.ndarray,
        unique_bands: np.ndarray,
        band_hits: dict[int, bool],
        band_true_active: dict[int, bool],
        new_discoveries: int,
        num_redundant_tuners: int,
    ) -> float:
        r = 0.0

        # Interception rewards
        for band in unique_bands:
            if band_hits[band] and band_true_active[band]:
                consec = self._consecutive_dwells[band]
                decay = 1.0 / (1.0 + 0.25 * max(0, consec - 1))
                r += self.w_hit * decay

        # Emitter discovery bonus
        r += self.w_new * new_discoveries

        # AoI exploration reward across observed channels
        for band in unique_bands:
            r += (self.w_aoi / self.M) * (self._aoi[band] / self.aoi_max)

        # Anti-camping penalty per band if camped > 3 steps
        for band in unique_bands:
            if self._consecutive_dwells[band] >= 3:
                r -= 0.8 * (self._consecutive_dwells[band] - 2)

        # Switching cost averaged across M tuners
        switch_dist = np.abs(action - self._last_actions) / max(self.K - 1, 1)
        r -= self.w_switch * float(np.mean(switch_dist))

        # Redundant tuner collision penalty
        r -= self.w_collision * num_redundant_tuners

        return float(r)

    def _update_belief(self, band: int, hit: bool) -> None:
        p = float(self._belief[band])
        duty = 0.20
        if hit:
            p_hit_present = duty * self.Pd + (1.0 - duty) * self.Pfa
            p_hit_absent = self.Pfa
            posterior = (p * p_hit_present) / max(1e-9, (p * p_hit_present + (1.0 - p) * p_hit_absent))
        else:
            p_miss_present = duty * (1.0 - self.Pd) + (1.0 - duty) * (1.0 - self.Pfa)
            p_miss_absent = 1.0 - self.Pfa
            posterior = (p * p_miss_present) / max(1e-9, (p * p_miss_present + (1.0 - p) * p_miss_absent))
        self._belief[band] = float(np.clip(posterior, 0.01, 0.99))

    def _get_obs(self) -> np.ndarray:
        aoi_norm = self._aoi / self.aoi_max
        last_actions_multihot = np.zeros(self.K, dtype=np.float32)
        last_actions_multihot[self._last_actions] = 1.0
        return np.concatenate([
            self._belief.astype(np.float32),
            aoi_norm.astype(np.float32),
            last_actions_multihot,
        ])

    def _get_info(self) -> dict[str, Any]:
        return {
            "t": self._t,
            "total_hits": self._total_hits,
            "total_dwells": self._total_dwells,
            "total_collisions": self._total_collisions,
            "false_alarms": self._false_alarms,
            "emitters_discovered": len(self._discovered),
            "num_emitters": len(self.truth._emitters),
            "num_tuners": self.M,
            "last_band_hits": self._last_band_hits.copy(),
        }


    @property
    def current_step(self) -> int:
        return self._t

    @property
    def hit_rate(self) -> float:
        return self._total_hits / max(self._total_dwells, 1)

    @property
    def discovery_ratio(self) -> float:
        return len(self._discovered) / max(len(self.truth._emitters), 1)

    @property
    def collision_rate(self) -> float:
        return self._total_collisions / max(self._t * self.M, 1)


# ---------------------------------------------------------------------------
# Dynamic Multi-Receiver Environment with Reset-Level Domain Randomization
# ---------------------------------------------------------------------------

class DynamicMultiReceiverEnv(MultiReceiverEWSpectrumEnv):
    """
    Multi-Receiver EW Environment with Dynamic Domain Randomization on every reset.
    """

    def __init__(
        self,
        K: int = 35,
        M: int = 4,
        T: int = 1000,
        stage: Optional[int] = 3,
        num_emitters_range: tuple[int, int] = (3, 6),
        dwell_us: float = 1000.0,
        switch_us: float = 50.0,
        Pd: float = 0.95,
        Pfa: float = 1e-4,
        w_hit: float = 6.0,
        w_new: float = 40.0,
        w_aoi: float = 2.5,
        w_switch: float = 0.2,
        w_collision: float = 2.0,
        aoi_max: int = 200,
        noise_floor: float = 0.0,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        self.stage = stage
        self.num_emitters_range = num_emitters_range
        self.dwell_us = dwell_us
        self.switch_us = switch_us
        self.noise_floor = noise_floor

        rng = np.random.default_rng(seed)
        initial_truth = self._build_randomized_truth(rng, K, T)

        super().__init__(
            truth_engine=initial_truth,
            K=K,
            M=M,
            T=T,
            Pd=Pd,
            Pfa=Pfa,
            w_hit=w_hit,
            w_new=w_new,
            w_aoi=w_aoi,
            w_switch=w_switch,
            w_collision=w_collision,
            aoi_max=aoi_max,
            seed=seed,
            render_mode=render_mode,
        )

    def _build_randomized_truth(self, rng: np.random.Generator, K: int, T: int) -> TruthEngine:
        engine = TruthEngine(
            K=K, T=T,
            dwell_us=self.dwell_us, switch_us=self.switch_us,
            noise_floor=self.noise_floor, rng=rng,
        )
        t_slot = engine.T_slot
        emitters = []
        eid = 0

        all_bands = list(range(K))
        rng.shuffle(all_bands)

        n_emitters = int(rng.integers(self.num_emitters_range[0], self.num_emitters_range[1] + 1))
        # 1-2 Fixed radars
        n_fixed = min(int(rng.integers(1, 3)), n_emitters)
        for i in range(n_fixed):
            band = all_bands[i]
            pri_slots = float(rng.uniform(3.0, 10.0))
            emitters.append(
                FixedFrequencyEmitter(
                    emitter_id=eid,
                    band_index=band,
                    pri_sec=pri_slots * t_slot,
                    pulse_width=t_slot,
                    pri_jitter=float(rng.uniform(0.01, 0.08)),
                    rng=rng,
                )
            )
            eid += 1

        # 1-3 FHSS hoppers
        n_fhss = min(int(rng.integers(1, 4)), max(1, n_emitters - n_fixed - 1))
        for _ in range(n_fhss):
            hop_len = int(rng.integers(4, 8))
            hop_bands = list(rng.choice(K, size=min(hop_len, K), replace=False))
            hop_interval_slots = float(rng.uniform(5.0, 12.0))
            emitters.append(
                FHSSEmitter(
                    emitter_id=eid,
                    hop_bands=hop_bands,
                    hop_interval=hop_interval_slots * t_slot,
                    pri_sec=2.0 * t_slot,
                    pulse_width=t_slot,
                    burst_size=int(rng.integers(2, 4)),
                    rng=rng,
                )
            )
            eid += 1

        # 1-2 Scanning radars
        scan_band = int(rng.choice(all_bands[n_fixed:])) if len(all_bands) > n_fixed else int(rng.integers(0, K))
        t_scan = float(rng.uniform(1.5, 3.5))
        beamwidth = float(rng.uniform(8.0, 15.0))
        emitters.append(
            ScanningEmitter(
                emitter_id=eid,
                band_index=scan_band,
                T_scan_sec=t_scan,
                beamwidth_deg=beamwidth,
                pri_sec=2.0 * t_slot,
                pulse_width=t_slot,
                initial_angle=float(rng.uniform(0.0, 360.0)),
                rng=rng,
            )
        )

        engine.add_emitters(emitters)
        engine.build(verbose=False)
        return engine

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.truth = self._build_randomized_truth(self._rng, self.K, self.T)
        self._t = 0
        self._belief = np.ones(self.K, dtype=np.float32) * 0.5
        self._aoi = np.zeros(self.K, dtype=np.float32)
        self._last_actions = np.arange(self.M, dtype=int) % self.K
        self._consecutive_dwells = np.zeros(self.K, dtype=int)
        self._discovered = set()
        self._total_hits = 0
        self._total_dwells = 0
        self._total_collisions = 0
        self._false_alarms = 0
        self._last_band_hits = {}

        return self._get_obs(), self._get_info()

