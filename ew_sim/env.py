"""
ew_sim/env.py
=============
OpenAI Gymnasium-compatible Electronic Warfare Spectrum Environment.

The receiver scheduler interacts with this environment through the standard
Gymnasium API:

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Observation space
-----------------
A flat vector of shape (3 * K,) containing:
    [0   : K  ]  belief_vector  b[k]   ∈ [0, 1]   Bayesian occupancy belief
    [K   : 2K ]  aoi_vector     AoI[k] ∈ [0, 1]   Normalised age-of-information
    [2K  : 3K ]  last_action_onehot    ∈ {0, 1}    One-hot of previous action

Action space
------------
    Discrete(K) — choose which sub-band to dwell on next

Reward function
---------------
    R = w_hit * hit
      + w_new  * new_emitter_discovered
      + w_aoi  * AoI[action] / AoI_max      (exploration bonus)
      - w_sw   * |action - prev_action| / K  (LO switching penalty)

Physics model
-------------
The environment models two imperfect detectors:
    P_d  = probability of detecting a pulse given the band IS active
    P_fa = probability of a false alarm given the band is QUIET
These are configurable (default: P_d=0.95, P_fa=1e-4).
"""

from __future__ import annotations

from typing import Any, Optional, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ew_sim.truth_engine import TruthEngine, build_default_truth_engine
from ew_sim.emitters import FixedFrequencyEmitter, FHSSEmitter, ScanningEmitter


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class EWSpectrumEnv(gym.Env):
    """
    EW Spectrum Scanning Environment.

    Parameters
    ----------
    truth_engine    : a pre-built TruthEngine.  If None, build_default_truth_engine
                      is called with `env_kwargs`.
    K               : number of sub-bands (used only when building default engine)
    T               : episode length in time slots
    Pd              : probability of detection  (true positive rate)
    Pfa             : probability of false alarm (false positive rate)
    w_hit           : reward weight for a successful interception
    w_new           : bonus reward for discovering a previously unseen emitter
    w_aoi           : exploration bonus weight (proportional to age-of-info)
    w_switch        : switching penalty weight per unit normalised band distance
    aoi_max         : cap on AoI used for normalisation (slots)
    seed            : random seed for reproducibility
    render_mode     : "human" (text summary) or None
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        truth_engine: Optional[TruthEngine] = None,
        K: int = 35,
        T: int = 2000,
        Pd: float = 0.95,
        Pfa: float = 1e-4,
        w_hit: float = 6.0,
        w_new: float = 40.0,
        w_aoi: float = 2.5,
        w_switch: float = 0.2,
        aoi_max: int = 200,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.K = K
        self.T = T
        self.Pd = Pd
        self.Pfa = Pfa
        self.w_hit = w_hit
        self.w_new = w_new
        self.w_aoi = w_aoi
        self.w_switch = w_switch
        self.aoi_max = aoi_max

        # Build or accept truth engine
        if truth_engine is not None:
            self.truth = truth_engine
            self.K = truth_engine.K
        else:
            self.truth = build_default_truth_engine(
                K=K, T=T, verbose=False,
                seed=seed if seed is not None else 42,
            )
            self.K = K

        # ── Spaces ─────────────────────────────────────────────────────────
        # obs = [belief(K) | aoi_norm(K) | last_action_onehot(K)]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3 * self.K,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.K)

        # ── Runtime state (initialised in reset()) ──────────────────────────
        self._rng = np.random.default_rng(seed)
        self._t: int = 0
        self._belief: np.ndarray = np.ones(self.K, dtype=np.float32) * 0.5
        self._aoi: np.ndarray = np.zeros(self.K, dtype=np.float32)
        self._last_action: int = 0
        self._consecutive_dwells: int = 0
        self._discovered: set[int] = set()   # emitter IDs seen so far

        # Episode statistics
        self._total_hits: int = 0
        self._total_dwells: int = 0
        self._false_alarms: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 0
        self._belief = np.ones(self.K, dtype=np.float32) * 0.5
        self._aoi = np.zeros(self.K, dtype=np.float32)
        self._last_action = 0
        self._consecutive_dwells = 0
        self._discovered = set()
        self._total_hits = 0
        self._total_dwells = 0
        self._false_alarms = 0

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """
        Execute one scheduling step: tune receiver to sub-band `action`,
        observe hit/miss, update belief and AoI, compute reward.
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        # ── 1. Ground truth & noisy detection ──────────────────────────────
        true_active = self.truth.is_active(action, self._t)
        if true_active:
            hit = bool(self._rng.random() < self.Pd)
        else:
            hit = bool(self._rng.random() < self.Pfa)
            if hit:
                self._false_alarms += 1

        # ── 2. Check if this is a new emitter discovery ─────────────────────
        new_discovery = False
        if hit and true_active:
            for emitter in self.truth._emitters:
                if (emitter.primary_band == action
                        and emitter.id not in self._discovered):
                    self._discovered.add(emitter.id)
                    new_discovery = True

        # ── 3. Consecutive dwell tracking & anti-camping ───────────────────
        if action == self._last_action:
            self._consecutive_dwells += 1
        else:
            self._consecutive_dwells = 0

        # ── 4. Compute reward ───────────────────────────────────────────────
        reward = self._compute_reward(action, hit, new_discovery)

        # ── 5. Belief update (Bayes with duty-cycle) ────────────────────────
        self._update_belief(action, hit)

        # ── 6. AoI update ───────────────────────────────────────────────────
        self._aoi += 1.0
        self._aoi[action] = 0.0
        self._aoi = np.clip(self._aoi, 0, self.aoi_max)

        # ── 7. Statistics ───────────────────────────────────────────────────
        self._total_dwells += 1
        if hit and true_active:
            self._total_hits += 1

        self._last_action = action
        self._t += 1

        terminated = False                       # no terminal state mid-episode
        truncated = self._t >= self.T

        if self.render_mode == "human":
            self._render_text(action, hit, true_active, reward)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> Optional[str]:
        if self.render_mode == "human":
            return (
                f"t={self._t:4d}  "
                f"hits={self._total_hits}/{self._total_dwells}  "
                f"discovered={len(self._discovered)}/{len(self.truth._emitters)}"
            )
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        action: int,
        hit: bool,
        new_discovery: bool,
    ) -> float:
        r = 0.0
        # 1. Pulse intercept reward with burst decay to discourage camping
        if hit:
            burst_decay = 1.0 / (1.0 + 0.25 * self._consecutive_dwells)
            r += self.w_hit * burst_decay

        # 2. Large discovery reward for finding previously unobserved emitters
        if new_discovery:
            r += self.w_new

        # 3. Age-of-Information exploration reward (high for unvisited bands)
        r += self.w_aoi * (self._aoi[action] / self.aoi_max)

        # 4. Anti-Camping Penalty: Penalize staying on the same band > 3 consecutive slots
        if self._consecutive_dwells >= 3:
            r -= 1.2 * (self._consecutive_dwells - 2)

        # 5. Mild switching cost
        r -= self.w_switch * abs(action - self._last_action) / max(self.K - 1, 1)
        return float(r)

    def _update_belief(self, action: int, hit: bool) -> None:
        """
        Bayesian belief update accounting for radar duty cycle (d ~ 0.20),
        followed by Markov diffusion for unsensed bands.
        """
        p = float(self._belief[action])
        duty = 0.20

        if hit:
            p_hit_present = duty * self.Pd + (1.0 - duty) * self.Pfa
            p_hit_absent = self.Pfa
            posterior = (p * p_hit_present) / max(1e-9, (p * p_hit_present + (1.0 - p) * p_hit_absent))
        else:
            p_miss_present = duty * (1.0 - self.Pd) + (1.0 - duty) * (1.0 - self.Pfa)
            p_miss_absent = 1.0 - self.Pfa
            posterior = (p * p_miss_present) / max(1e-9, (p * p_miss_present + (1.0 - p) * p_miss_absent))

        self._belief[action] = float(np.clip(posterior, 0.01, 0.99))

        # Diffuse unsensed bands toward prior
        alpha = 0.02
        prior = 0.15
        mask = np.ones(self.K, dtype=bool)
        mask[action] = False
        self._belief[mask] = (
            (1.0 - alpha) * self._belief[mask] + alpha * prior
        )

    def _get_obs(self) -> np.ndarray:
        """Concatenate [belief | aoi_normalised | last_action_onehot]."""
        aoi_norm = self._aoi / self.aoi_max
        last_action_oh = np.zeros(self.K, dtype=np.float32)
        last_action_oh[self._last_action] = 1.0
        return np.concatenate([
            self._belief.astype(np.float32),
            aoi_norm.astype(np.float32),
            last_action_oh,
        ])

    def _get_info(self) -> dict[str, Any]:
        return {
            "t": self._t,
            "total_hits": self._total_hits,
            "total_dwells": self._total_dwells,
            "false_alarms": self._false_alarms,
            "emitters_discovered": len(self._discovered),
            "num_emitters": len(self.truth._emitters),
            "belief": self._belief.copy(),
            "aoi": self._aoi.copy(),
        }

    def _render_text(
        self,
        action: int,
        hit: bool,
        true_active: bool,
        reward: float,
    ) -> None:
        status = (
            "HIT  ✓" if (hit and true_active) else
            "MISS  " if (not hit and true_active) else
            "FA   !" if (hit and not true_active) else
            "QUIET "
        )
        print(
            f"t={self._t - 1:4d}  band={action:2d}  {status}  "
            f"R={reward:+.2f}  "
            f"hits={self._total_hits}  "
            f"disc={len(self._discovered)}"
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def current_step(self) -> int:
        return self._t

    @property
    def hit_rate(self) -> float:
        return self._total_hits / max(self._total_dwells, 1)

    @property
    def discovery_ratio(self) -> float:
        return len(self._discovered) / max(len(self.truth._emitters), 1)


# ---------------------------------------------------------------------------
# Dynamic Domain-Randomized Environment
# ---------------------------------------------------------------------------

class DynamicSpectrumEnv(EWSpectrumEnv):
    """
    Dynamic Electronic Warfare Spectrum Scanning Environment.

    Regenerates a randomized RF tactical battlefield on EVERY episode reset.
    Guarantees that policies learn generalized tracking and patrolling behaviors
    driven by the observation state (belief & AoI) rather than memorizing fixed band indices.

    Parameters
    ----------
    K                   : Number of sub-bands (default: 35)
    T                   : Episode duration in time slots (default: 1000)
    stage               : Curriculum stage (1: Fixed only, 2: Fixed+FHSS, 3: Full battlefield, None: Fully random)
    num_emitters_range  : (min_emitters, max_emitters) active per episode
    dwell_us            : Receiver dwell time in microseconds (default: 1000.0)
    switch_us           : LO tuning time in microseconds (default: 50.0)
    Pd                  : Probability of detection
    Pfa                 : Probability of false alarm
    noise_floor         : Ground-truth spurious occupancy rate
    seed                : Random seed
    """

    def __init__(
        self,
        K: int = 35,
        T: int = 1000,
        stage: Optional[int] = 3,
        num_emitters_range: tuple[int, int] = (2, 5),
        dwell_us: float = 1000.0,
        switch_us: float = 50.0,
        Pd: float = 0.95,
        Pfa: float = 1e-4,
        w_hit: float = 6.0,
        w_new: float = 40.0,
        w_aoi: float = 2.5,
        w_switch: float = 0.2,
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
        self._seed = seed

        rng = np.random.default_rng(seed)
        initial_truth = self._build_randomized_truth(rng, K, T)

        super().__init__(
            truth_engine=initial_truth,
            K=K,
            T=T,
            Pd=Pd,
            Pfa=Pfa,
            w_hit=w_hit,
            w_new=w_new,
            w_aoi=w_aoi,
            w_switch=w_switch,
            aoi_max=aoi_max,
            seed=seed,
            render_mode=render_mode,
        )

    def _build_randomized_truth(self, rng: np.random.Generator, K: int, T: int) -> TruthEngine:
        """Constructs a randomized TruthEngine based on current stage."""
        engine = TruthEngine(
            K=K,
            T=T,
            dwell_us=self.dwell_us,
            switch_us=self.switch_us,
            noise_floor=self.noise_floor,
            rng=rng,
        )
        t_slot = engine.T_slot
        emitters = []
        eid = 0

        # Available bands for allocation without collision
        all_bands = list(range(K))
        rng.shuffle(all_bands)

        if self.stage == 1:
            # Stage 1: Fixed frequency radars on random bands
            n_emitters = int(rng.integers(self.num_emitters_range[0], self.num_emitters_range[1] + 1))
            for i in range(min(n_emitters, len(all_bands))):
                band = all_bands[i]
                pri_slots = float(rng.uniform(3.0, 12.0))
                pri_sec = pri_slots * t_slot
                jitter = float(rng.uniform(0.0, 0.08))
                emitters.append(
                    FixedFrequencyEmitter(
                        emitter_id=eid,
                        band_index=band,
                        pri_sec=pri_sec,
                        pulse_width=t_slot,
                        pri_jitter=jitter,
                        rng=rng,
                    )
                )
                eid += 1

        elif self.stage == 2:
            # Stage 2: 1-2 Fixed + 1-2 FHSS agile emitters
            n_fixed = int(rng.integers(1, 3))
            for i in range(n_fixed):
                band = all_bands[i]
                pri_slots = float(rng.uniform(4.0, 10.0))
                emitters.append(
                    FixedFrequencyEmitter(
                        emitter_id=eid,
                        band_index=band,
                        pri_sec=pri_slots * t_slot,
                        pulse_width=t_slot,
                        pri_jitter=float(rng.uniform(0.0, 0.05)),
                        rng=rng,
                    )
                )
                eid += 1

            n_fhss = int(rng.integers(1, 3))
            for _ in range(n_fhss):
                hop_len = int(rng.integers(4, 8))
                hop_bands = list(rng.choice(K, size=min(hop_len, K), replace=False))
                hop_interval_slots = float(rng.uniform(6.0, 12.0))
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

        else:
            # Stage 3 or None: Full tactical battlefield (Fixed + FHSS + Scanning)
            n_fixed = int(rng.integers(1, 3))
            for i in range(n_fixed):
                band = all_bands[i]
                pri_slots = float(rng.uniform(4.0, 12.0))
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

            n_fhss = int(rng.integers(1, 3))
            for _ in range(n_fhss):
                hop_len = int(rng.integers(4, 7))
                hop_bands = list(rng.choice(K, size=min(hop_len, K), replace=False))
                hop_interval_slots = float(rng.uniform(6.0, 12.0))
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

            # Scanning Radar
            scan_band = int(rng.choice(all_bands[n_fixed:])) if len(all_bands) > n_fixed else int(rng.integers(0, K))
            t_scan = float(rng.uniform(1.5, 3.5))
            beamwidth = float(rng.uniform(8.0, 15.0))
            init_angle = float(rng.uniform(0.0, 360.0))
            emitters.append(
                ScanningEmitter(
                    emitter_id=eid,
                    band_index=scan_band,
                    T_scan_sec=t_scan,
                    beamwidth_deg=beamwidth,
                    pri_sec=2.0 * t_slot,
                    pulse_width=t_slot,
                    initial_angle=init_angle,
                    rng=rng,
                )
            )
            eid += 1

        engine.add_emitters(emitters)
        engine.build(verbose=False)
        return engine

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment and generate a brand-new randomized tactical scenario."""
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Regenerate truth engine with dynamic domain randomization
        self.truth = self._build_randomized_truth(self._rng, self.K, self.T)

        # Reset episode counters and tracking
        self._t = 0
        self._belief = np.ones(self.K, dtype=np.float32) * 0.5
        self._aoi = np.zeros(self.K, dtype=np.float32)
        self._last_action = 0
        self._consecutive_dwells = 0
        self._discovered = set()
        self._total_hits = 0
        self._total_dwells = 0
        self._false_alarms = 0

        return self._get_obs(), self._get_info()
