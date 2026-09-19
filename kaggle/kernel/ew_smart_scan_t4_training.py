# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🛡️ DRDO EW Smart Scan — 4M-Step Dynamic Domain-Randomized DRL Curriculum Training (T4 GPU)
#
# **SIH 2026 · Smart Scan EW — AI-Driven Spectrum Scheduling for Electronic Support**
#
# This notebook trains a Recurrent PPO (LSTM) deep reinforcement learning agent
# for Electronic Warfare spectrum surveillance using **reset-level domain randomization**.
# On every episode reset across all 16 parallel workers, a brand-new tactical battlefield
# is dynamically generated with randomized carrier bands, PRIs, hop sets, and spatial rotations.
#
# This prevents band memorization and trains a truly generalized policy that actively tracks
# signals based on Bayesian belief state $b[k]$ and Age of Information $AoI[k]$.
#
# ### Curriculum Schedule (4,000,000 Total Steps)
# 1. **Stage 1** (1,000,000 steps): Fixed-frequency pulsed radars on random bands
# 2. **Stage 2** (1,500,000 steps): Agile Frequency-Hopping (FHSS) + Fixed radars
# 3. **Stage 3** (1,500,000 steps): Full contested tactical battlefield (Fixed + FHSS + Rotating Scanning Radars)
#
# ### Hardware
# - **GPU**: NVIDIA Tesla T4 (16 GB VRAM)
# - **Parallelism**: 16 SubprocVecEnv workers (250,000 steps per worker)

# %% [markdown]
# ## 1. Install Dependencies

# %%
# !pip install -q gymnasium>=0.29 stable-baselines3>=2.3 sb3-contrib>=2.3 torch numpy scipy matplotlib

# %% [markdown]
# ## 2. Imports & Hardware Verification

# %%
import os
import math
import time
import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, SupportsFloat

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 PyTorch device: {device}")
if device == "cuda":
    print(f"   GPU:  {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.2f} GB")
else:
    print("   ⚠️ No GPU detected — execution will be slow!")

# %% [markdown]
# ## 3. Electronic Warfare Simulation Core (Self-Contained)

# %%
# ═══════════════════════════════════════════════════════════════════════
# Tactical Emitter Models
# ═══════════════════════════════════════════════════════════════════════

class BaseEmitter(ABC):
    """Abstract base for all tactical emitter models."""

    def __init__(
        self,
        emitter_id: int,
        band_index: int,
        start_time: float = 0.0,
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
        ...

    def is_on(self, t_sec: float) -> bool:
        return self.start_time <= t_sec < self.end_time


class FixedFrequencyEmitter(BaseEmitter):
    """Pulsed radar on a fixed carrier frequency."""

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
        self._jitter_seed = int(self.rng.integers(0, 2**31))
        self._jitter_rng = np.random.default_rng(self._jitter_seed)

    def state_at(self, t_sec: float) -> tuple[Optional[int], bool]:
        if not self.is_on(t_sec):
            return None, False
        t_rel = t_sec - self.start_time
        pulse_idx = int(t_rel / self.pri)
        pulse_start = pulse_idx * self.pri
        in_pulse = pulse_start <= t_rel < pulse_start + self.pulse_width
        return (self.primary_band, True) if in_pulse else (self.primary_band, False)


class FHSSEmitter(BaseEmitter):
    """Frequency-Hopping Spread Spectrum radar with Markov transitions."""

    def __init__(
        self,
        emitter_id: int,
        hop_bands: list[int],
        transition_mat: Optional[np.ndarray] = None,
        hop_interval: float = 5e-3,
        pulse_width: float = 2e-6,
        pri_sec: float = 1e-3,
        burst_size: int = 4,
        **kwargs,
    ):
        super().__init__(emitter_id, hop_bands[0], **kwargs)
        self.hop_bands = list(hop_bands)
        self.hop_interval = hop_interval
        self.pulse_width = pulse_width
        self.pri = pri_sec
        self.burst_size = burst_size

        n = len(hop_bands)
        if transition_mat is not None:
            self._P = transition_mat.copy()
        else:
            raw = self.rng.random((n, n)) + 0.1
            self._P = raw / raw.sum(axis=1, keepdims=True)

        self._hop_cache: dict[int, int] = {0: 0}

    def _band_at_hop(self, hop_slot: int) -> int:
        if hop_slot in self._hop_cache:
            return self._hop_cache[hop_slot]
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
        t_in_hop = t_rel - hop_slot * self.hop_interval
        pulse_idx = int(t_in_hop / self.pri)
        if pulse_idx >= self.burst_size:
            return current_band, False
        pulse_start = pulse_idx * self.pri
        in_pulse = pulse_start <= t_in_hop < pulse_start + self.pulse_width
        return (current_band, True) if in_pulse else (current_band, False)


class ScanningEmitter(BaseEmitter):
    """Spatially scanning surveillance radar with sinc² antenna pattern."""

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
        self.TOT = (beamwidth_deg / 360.0) * T_scan_sec

    def _normalised_gain(self, t_sec: float) -> float:
        t_rel = t_sec - self.start_time
        beam_angle = (self.initial_angle + 360.0 * t_rel / self.T_scan) % 360.0
        delta = beam_angle
        if delta > 180.0:
            delta -= 360.0
        half_bw = self.beamwidth_deg / 2.0
        x = 0.443 * delta / half_bw if half_bw > 0 else 0.0
        return float(math.sin(math.pi * x + 1e-12) ** 2 / (math.pi * x + 1e-12) ** 2) \
            if abs(x) > 1e-9 else 1.0

    def state_at(self, t_sec: float) -> tuple[Optional[int], bool]:
        if not self.is_on(t_sec):
            return None, False
        gain = self._normalised_gain(t_sec)
        if gain < self.gain_threshold:
            return self.primary_band, False
        t_rel = t_sec - self.start_time
        pulse_idx = int(t_rel / self.pri)
        pulse_start = pulse_idx * self.pri
        in_pulse = pulse_start <= t_rel < pulse_start + self.pulse_width
        return (self.primary_band, True) if in_pulse else (self.primary_band, False)


# ═══════════════════════════════════════════════════════════════════════
# Ground-Truth Engine
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EmitterActivity:
    emitter_id: int
    band_index: int
    active_slots: list[int] = field(default_factory=list)

    @property
    def total_pulses(self) -> int:
        return len(self.active_slots)


class TruthEngine:
    """Stores ground truth S[K, T] binary spectrum occupancy matrix."""

    def __init__(
        self,
        K: int = 35,
        T: int = 1000,
        dwell_us: float = 1000.0,
        switch_us: float = 50.0,
        f_min_ghz: float = 0.5,
        f_max_ghz: float = 18.0,
        noise_floor: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.K = K
        self.T = T
        self.T_slot = (dwell_us + switch_us) * 1e-6
        self.f_min = f_min_ghz
        self.f_max = f_max_ghz
        self.noise_floor = noise_floor
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.ibw_ghz = (f_max_ghz - f_min_ghz) / K
        self.band_centres = np.array([f_min_ghz + (k + 0.5) * self.ibw_ghz for k in range(K)])
        self._emitters: list[BaseEmitter] = []
        self.S: Optional[np.ndarray] = None
        self.activity: dict[int, EmitterActivity] = {}

    def add_emitter(self, emitter: BaseEmitter) -> None:
        self._emitters.append(emitter)
        self.activity[emitter.id] = EmitterActivity(emitter_id=emitter.id, band_index=emitter.primary_band)

    def add_emitters(self, emitters: list[BaseEmitter]) -> None:
        for e in emitters:
            self.add_emitter(e)

    def build(self, verbose: bool = False) -> np.ndarray:
        self.S = np.zeros((self.K, self.T), dtype=np.int8)
        for t in range(self.T):
            t_sec = t * self.T_slot
            for emitter in self._emitters:
                band, active = emitter.state_at(t_sec)
                if band is not None and 0 <= band < self.K and active:
                    self.S[band, t] = 1
                    self.activity[emitter.id].active_slots.append(t)

        if self.noise_floor > 0:
            noise_mask = self.rng.random((self.K, self.T)) < self.noise_floor
            self.S = np.clip(self.S + noise_mask.astype(np.int8), 0, 1)
        return self.S

    def is_active(self, band: int, slot: int) -> bool:
        if self.S is None:
            raise RuntimeError("Call build() first.")
        return bool(self.S[band, slot])


# ═══════════════════════════════════════════════════════════════════════
# Dynamic Gymnasium Environment with Reset-Level Domain Randomization
# ═══════════════════════════════════════════════════════════════════════

class DynamicSpectrumEnv(gym.Env):
    """
    Electronic Warfare Spectrum Scanning Environment with Dynamic Domain Randomization.
    Regenerates a fresh, non-memorizable RF battlefield on every episode reset.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

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
        super().__init__()
        self.render_mode = render_mode
        self.K = K
        self.T = T
        self.stage = stage
        self.num_emitters_range = num_emitters_range
        self.dwell_us = dwell_us
        self.switch_us = switch_us
        self.Pd = Pd
        self.Pfa = Pfa
        self.w_hit = w_hit
        self.w_new = w_new
        self.w_aoi = w_aoi
        self.w_switch = w_switch
        self.aoi_max = aoi_max
        self.noise_floor = noise_floor

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3 * self.K,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.K)

        self._rng = np.random.default_rng(seed)
        self.truth = self._build_randomized_truth(self._rng, self.K, self.T)

        self._t = 0
        self._belief = np.ones(self.K, dtype=np.float32) * 0.5
        self._aoi = np.zeros(self.K, dtype=np.float32)
        self._last_action = 0
        self._consecutive_dwells = 0
        self._discovered: set[int] = set()
        self._total_hits = 0
        self._total_dwells = 0
        self._false_alarms = 0

    def _build_randomized_truth(self, rng: np.random.Generator, K: int, T: int) -> TruthEngine:
        """Dynamically generates emitters on randomized carrier bands."""
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

        if self.stage == 1:
            # Stage 1: Fixed-frequency radars on random bands
            n_emitters = int(rng.integers(self.num_emitters_range[0], self.num_emitters_range[1] + 1))
            for i in range(min(n_emitters, len(all_bands))):
                band = all_bands[i]
                pri_slots = float(rng.uniform(3.0, 12.0))
                jitter = float(rng.uniform(0.0, 0.08))
                emitters.append(
                    FixedFrequencyEmitter(
                        emitter_id=eid,
                        band_index=band,
                        pri_sec=pri_slots * t_slot,
                        pulse_width=t_slot,
                        pri_jitter=jitter,
                        rng=rng,
                    )
                )
                eid += 1

        elif self.stage == 2:
            # Stage 2: Fixed + FHSS agile frequency hoppers
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
            # Stage 3: Full complex tactical battlefield
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

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Build fresh randomized RF world on EVERY reset
        self.truth = self._build_randomized_truth(self._rng, self.K, self.T)

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

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        true_active = self.truth.is_active(action, self._t)
        if true_active:
            hit = bool(self._rng.random() < self.Pd)
        else:
            hit = bool(self._rng.random() < self.Pfa)
            if hit:
                self._false_alarms += 1

        new_discovery = False
        if hit and true_active:
            for emitter in self.truth._emitters:
                if emitter.primary_band == action and emitter.id not in self._discovered:
                    self._discovered.add(emitter.id)
                    new_discovery = True

        if action == self._last_action:
            self._consecutive_dwells += 1
        else:
            self._consecutive_dwells = 0

        reward = self._compute_reward(action, hit, new_discovery)
        self._update_belief(action, hit)

        self._aoi += 1.0
        self._aoi[action] = 0.0
        self._aoi = np.clip(self._aoi, 0, self.aoi_max)

        self._total_dwells += 1
        if hit and true_active:
            self._total_hits += 1

        self._last_action = action
        self._t += 1
        truncated = self._t >= self.T

        return self._get_obs(), reward, False, truncated, self._get_info()

    def _compute_reward(self, action: int, hit: bool, new_discovery: bool) -> float:
        r = 0.0
        if hit:
            burst_decay = 1.0 / (1.0 + 0.25 * self._consecutive_dwells)
            r += self.w_hit * burst_decay
        if new_discovery:
            r += self.w_new
        r += self.w_aoi * (self._aoi[action] / self.aoi_max)
        if self._consecutive_dwells >= 3:
            r -= 1.2 * (self._consecutive_dwells - 2)
        r -= self.w_switch * abs(action - self._last_action) / max(self.K - 1, 1)
        return float(r)

    def _update_belief(self, action: int, hit: bool) -> None:
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
        alpha = 0.02
        prior = 0.15
        mask = np.ones(self.K, dtype=bool)
        mask[action] = False
        self._belief[mask] = (1.0 - alpha) * self._belief[mask] + alpha * prior

    def _get_obs(self) -> np.ndarray:
        aoi_norm = self._aoi / self.aoi_max
        last_action_oh = np.zeros(self.K, dtype=np.float32)
        last_action_oh[self._last_action] = 1.0
        return np.concatenate([self._belief.astype(np.float32), aoi_norm.astype(np.float32), last_action_oh])

    def _get_info(self) -> dict[str, Any]:
        return {
            "t": self._t,
            "total_hits": self._total_hits,
            "total_dwells": self._total_dwells,
            "false_alarms": self._false_alarms,
            "emitters_discovered": len(self._discovered),
            "num_emitters": len(self.truth._emitters),
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


# %% [markdown]
# ## 4. Training Hyperparameters & Configuration

# %%
# ═══════════════════════════════════════════════════════════════════════
# Multi-Million Step Training Configuration
# ═══════════════════════════════════════════════════════════════════════

TOTAL_STEPS = 4_000_000     # 4 Million Total Timesteps
NUM_ENVS = 16               # Parallel SubprocVecEnv Workers (250,000 steps/worker)
K = 35                      # Sub-bands
T = 1000                    # Episode length

# Curriculum Steps Partition
STAGE1_STEPS = 1_000_000    # Stage 1: Fixed radars
STAGE2_STEPS = 1_500_000    # Stage 2: FHSS + Fixed
STAGE3_STEPS = 1_500_000    # Stage 3: Full complex battlefield

CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 70)
print(f"  🛡️ DRDO EW Smart Scan — 4M-Step Dynamic Domain-Randomized Training")
print(f"  Target Timesteps: {TOTAL_STEPS:,} on {device.upper()}")
print(f"  Parallel Workers: {NUM_ENVS} SubprocVecEnvs (250k steps/worker)")
print(f"  Stage 1: {STAGE1_STEPS:,} steps (Fixed Radars)")
print(f"  Stage 2: {STAGE2_STEPS:,} steps (FHSS Hoppers)")
print(f"  Stage 3: {STAGE3_STEPS:,} steps (Full Battlefield)")
print("=" * 70)

# %% [markdown]
# ## 5. Parallel Environment Factory

# %%
def make_env_worker(stage: int, worker_seed: int):
    """Factory returning an environment constructor for SubprocVecEnv."""
    def _init():
        return DynamicSpectrumEnv(
            K=K,
            T=T,
            stage=stage,
            num_emitters_range=(2, 5),
            dwell_us=1000.0,
            switch_us=50.0,
            seed=worker_seed,
        )
    return _init

# %% [markdown]
# ## 6. Telemetry Logger

# %%
class TacticalTelemetryLogger(BaseCallback):
    """Logs curriculum progression and reward metrics."""

    def __init__(self, stage_name: str = "", check_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.stage_name = stage_name
        self.check_freq = check_freq
        self._start_time = None

    def _on_training_start(self):
        self._start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            mean_r = np.mean(rewards) if rewards else 0.0
            elapsed = time.time() - self._start_time if self._start_time else 0
            fps = self.n_calls / max(elapsed, 1)
            print(f"  [{self.stage_name} Step {self.n_calls:7d}] "
                  f"Mean Reward: {mean_r:+.2f}  |  FPS: {fps:.0f}  |  "
                  f"Elapsed: {elapsed:.0f}s")
        return True

# %% [markdown]
# ## 7. 🚀 Curriculum Execution Loop

# %%
training_log = {}
overall_start = time.time()

# ──────────────────────────────────────────────────────────────────────
# Stage 1: Fixed-Frequency Radars (Dynamic Carrier Bands & PRIs)
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  [Stage 1/3] Fixed Radars — Dynamic Bands & PRIs ({STAGE1_STEPS:,} steps)")
print(f"{'='*65}")

env_fns1 = [make_env_worker(stage=1, worker_seed=100 + i) for i in range(NUM_ENVS)]
vec_env1 = SubprocVecEnv(env_fns1)
vec_env1 = VecMonitor(vec_env1)

model = RecurrentPPO(
    policy="MlpLstmPolicy",
    env=vec_env1,
    learning_rate=4e-4,
    n_steps=512,
    batch_size=128,
    n_epochs=5,
    gamma=0.98,
    gae_lambda=0.95,
    ent_coef=0.08,  # high exploration entropy prevents band camping
    device=device,
    verbose=0,
)

stage1_start = time.time()
cb1 = TacticalTelemetryLogger(stage_name="Stage 1", check_freq=25000)
model.learn(total_timesteps=STAGE1_STEPS, callback=cb1)
stage1_time = time.time() - stage1_start
vec_env1.close()

model.save(CHECKPOINT_DIR / "stage1_fixed_dynamic.zip")
training_log["stage1"] = {"steps": STAGE1_STEPS, "time_sec": stage1_time}
print(f"\n  ✅ Stage 1 complete in {stage1_time:.0f}s ({STAGE1_STEPS/stage1_time:.0f} steps/sec)")

# ──────────────────────────────────────────────────────────────────────
# Stage 2: Agile Frequency Hopping (FHSS) + Fixed Radars
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  [Stage 2/3] Agile Hoppers — Dynamic Hop Sets ({STAGE2_STEPS:,} steps)")
print(f"{'='*65}")

env_fns2 = [make_env_worker(stage=2, worker_seed=200 + i) for i in range(NUM_ENVS)]
vec_env2 = SubprocVecEnv(env_fns2)
vec_env2 = VecMonitor(vec_env2)
model.set_env(vec_env2)

stage2_start = time.time()
cb2 = TacticalTelemetryLogger(stage_name="Stage 2", check_freq=25000)
model.learn(total_timesteps=STAGE2_STEPS, callback=cb2)
stage2_time = time.time() - stage2_start
vec_env2.close()

model.save(CHECKPOINT_DIR / "stage2_fhss_dynamic.zip")
training_log["stage2"] = {"steps": STAGE2_STEPS, "time_sec": stage2_time}
print(f"\n  ✅ Stage 2 complete in {stage2_time:.0f}s ({STAGE2_STEPS/stage2_time:.0f} steps/sec)")

# ──────────────────────────────────────────────────────────────────────
# Stage 3: Full Complex Battlefield (Fixed + FHSS + Rotating Antenna)
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  [Stage 3/3] Full Battlefield — Dynamic Radars ({STAGE3_STEPS:,} steps)")
print(f"{'='*65}")

env_fns3 = [make_env_worker(stage=3, worker_seed=300 + i) for i in range(NUM_ENVS)]
vec_env3 = SubprocVecEnv(env_fns3)
vec_env3 = VecMonitor(vec_env3)
model.set_env(vec_env3)

stage3_start = time.time()
cb3 = TacticalTelemetryLogger(stage_name="Stage 3", check_freq=25000)
model.learn(total_timesteps=STAGE3_STEPS, callback=cb3)
stage3_time = time.time() - stage3_start
vec_env3.close()

model.save(CHECKPOINT_DIR / "stage3_full_dynamic.zip")
training_log["stage3"] = {"steps": STAGE3_STEPS, "time_sec": stage3_time}
print(f"\n  ✅ Stage 3 complete in {stage3_time:.0f}s ({STAGE3_STEPS/stage3_time:.0f} steps/sec)")

# %% [markdown]
# ## 8. Save Final Model Artifacts

# %%
final_path = CHECKPOINT_DIR / "ppo_recurrent_kaggle_dynamic_4m.zip"
model.save(final_path)

total_time = time.time() - overall_start
training_log["total_time_sec"] = total_time
training_log["total_steps"] = TOTAL_STEPS
training_log["device"] = device
training_log["num_envs"] = NUM_ENVS
if device == "cuda":
    training_log["gpu"] = torch.cuda.get_device_name(0)

report_path = CHECKPOINT_DIR / "training_report.json"
with open(report_path, "w") as f:
    json.dump(training_log, f, indent=2)

print(f"\n{'='*70}")
print(f"  🎯 4M-STEP DYNAMIC CURRICULUM TRAINING COMPLETE")
print(f"{'='*70}")
print(f"  Total Duration: {total_time/60:.1f} minutes")
print(f"  Total Steps:    {TOTAL_STEPS:,}")
print(f"  Avg Throughput: {TOTAL_STEPS/total_time:.0f} steps/sec")
print(f"  Final Model:    {final_path}")
print(f"{'='*70}")

for p in sorted(CHECKPOINT_DIR.iterdir()):
    size_mb = p.stat().st_size / 1e6
    print(f"   {p.name:42s} {size_mb:.1f} MB")

# %% [markdown]
# ## 9. Out-of-Distribution (OOD) Monte Carlo Benchmark
#
# Evaluates the trained model against legacy baselines across 20 unseen, randomly generated battlefields.

# %%
print("\n" + "=" * 70)
print("  🧪 OUT-OF-DISTRIBUTION (OOD) MONTE CARLO BENCHMARK (20 Unseen Seeds)")
print("=" * 70)

N_EVAL_EPISODES = 20
eval_model = RecurrentPPO.load(final_path)

results = {
    "SequentialSweep": {"hits": [], "ir": [], "disc": [], "ttis": []},
    "PseudoRandomSweep": {"hits": [], "ir": [], "disc": [], "ttis": []},
    "DRL-Dynamic-4M": {"hits": [], "ir": [], "disc": [], "ttis": []},
}

for ep in range(N_EVAL_EPISODES):
    eval_seed = 9000 + ep
    eval_env = DynamicSpectrumEnv(K=35, T=1000, stage=3, seed=eval_seed)

    # 1. Sequential Sweep
    obs, info = eval_env.reset(seed=eval_seed)
    seq_hits, seq_disc = 0, set()
    first_hits = {}
    for t in range(1000):
        action = t % 35
        obs, r, term, trunc, info = eval_env.step(action)
        if eval_env.truth.is_active(action, t):
            seq_hits += 1
            for e in eval_env.truth._emitters:
                if e.primary_band == action and e.id not in first_hits:
                    first_hits[e.id] = t * eval_env.truth.T_slot
                    seq_disc.add(e.id)
    total_transmitted = max(1, int(eval_env.truth.S.sum()))
    results["SequentialSweep"]["hits"].append(seq_hits)
    results["SequentialSweep"]["ir"].append(seq_hits / total_transmitted * 100.0)
    results["SequentialSweep"]["disc"].append(len(seq_disc) / len(eval_env.truth._emitters) * 100.0)
    results["SequentialSweep"]["ttis"].append(np.mean(list(first_hits.values())) if first_hits else 1.05)

    # 2. Pseudo-Random Sweep
    obs, info = eval_env.reset(seed=eval_seed)
    rnd_hits, rnd_disc = 0, set()
    first_hits = {}
    rng_sweep = np.random.default_rng(eval_seed)
    perm = list(range(35))
    rng_sweep.shuffle(perm)
    for t in range(1000):
        action = perm[t % 35]
        if (t + 1) % 35 == 0:
            rng_sweep.shuffle(perm)
        obs, r, term, trunc, info = eval_env.step(action)
        if eval_env.truth.is_active(action, t):
            rnd_hits += 1
            for e in eval_env.truth._emitters:
                if e.primary_band == action and e.id not in first_hits:
                    first_hits[e.id] = t * eval_env.truth.T_slot
                    rnd_disc.add(e.id)
    results["PseudoRandomSweep"]["hits"].append(rnd_hits)
    results["PseudoRandomSweep"]["ir"].append(rnd_hits / total_transmitted * 100.0)
    results["PseudoRandomSweep"]["disc"].append(len(rnd_disc) / len(eval_env.truth._emitters) * 100.0)
    results["PseudoRandomSweep"]["ttis"].append(np.mean(list(first_hits.values())) if first_hits else 1.05)

    # 3. DRL Dynamic 4M Agent
    obs, info = eval_env.reset(seed=eval_seed)
    drl_hits, drl_disc = 0, set()
    first_hits = {}
    lstm_state = None
    for t in range(1000):
        action, lstm_state = eval_model.predict(obs, state=lstm_state, deterministic=True)
        obs, r, term, trunc, info = eval_env.step(int(action))
        if eval_env.truth.is_active(int(action), t):
            drl_hits += 1
            for e in eval_env.truth._emitters:
                if e.primary_band == int(action) and e.id not in first_hits:
                    first_hits[e.id] = t * eval_env.truth.T_slot
                    drl_disc.add(e.id)
    results["DRL-Dynamic-4M"]["hits"].append(drl_hits)
    results["DRL-Dynamic-4M"]["ir"].append(drl_hits / total_transmitted * 100.0)
    results["DRL-Dynamic-4M"]["disc"].append(len(drl_disc) / len(eval_env.truth._emitters) * 100.0)
    results["DRL-Dynamic-4M"]["ttis"].append(np.mean(list(first_hits.values())) if first_hits else 1.05)

# Print Summary Table
print(f"\n{'Policy':<22} | {'Interception Ratio (IR)':<24} | {'Mean Hits':<12} | {'Discovery Rate':<16} | {'Mean TTI'}")
print("-" * 88)
for policy, data in results.items():
    ir_mean, ir_std = np.mean(data["ir"]), np.std(data["ir"])
    hits_mean = np.mean(data["hits"])
    disc_mean = np.mean(data["disc"])
    tti_mean = np.mean(data["ttis"])
    print(f"{policy:<22} | {ir_mean:5.1f}% ± {ir_std:4.1f}%          | {hits_mean:5.1f}/1000   | {disc_mean:5.1f}%          | {tti_mean:.3f} s")
print("-" * 88)

benchmark_report = {
    p: {
        "ir_mean": float(np.mean(d["ir"])),
        "ir_std": float(np.std(d["ir"])),
        "hits_mean": float(np.mean(d["hits"])),
        "discovery_mean": float(np.mean(d["disc"])),
        "tti_mean": float(np.mean(d["ttis"])),
    }
    for p, d in results.items()
}
with open(CHECKPOINT_DIR / "evaluation_benchmark.json", "w") as f:
    json.dump(benchmark_report, f, indent=2)

print("\nBenchmark report saved to /kaggle/working/checkpoints/evaluation_benchmark.json")

# %% [markdown]
# ---
# *SIH 2026 — EW Smart Scan · Multi-Million Step Dynamic DRL Curriculum Training on T4 GPU*
