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
# # 🛡️ DRDO EW Smart Scan — 1M-Step DRL Curriculum Training (T4 GPU)
#
# **SIH 2026 · Smart Scan EW — AI-Driven Spectrum Scheduling for Electronic Support**
#
# This notebook trains a Recurrent PPO (LSTM) deep reinforcement learning agent
# for intelligent Electronic Warfare spectrum scanning. The agent learns to
# autonomously schedule a single-channel receiver across 35 sub-bands to maximize
# pulse interceptions against a complex, mixed-emitter battlefield.
#
# ### Training Pipeline
# 1. **Stage 1** (333K steps): Fixed-frequency radars — learn basic interception
# 2. **Stage 2** (333K steps): FHSS agile emitters — learn to track frequency hoppers
# 3. **Stage 3** (334K steps): Full tactical battlefield with scanning radars
#
# ### Hardware
# - **GPU**: NVIDIA Tesla T4 (16 GB VRAM)
# - **Parallelism**: 16 SubprocVecEnv workers

# %% [markdown]
# ## 1. Install Dependencies

# %%
# !pip install -q gymnasium>=0.29 stable-baselines3>=2.3 sb3-contrib>=2.3 torch numpy scipy matplotlib

# %% [markdown]
# ## 2. Imports & GPU Check

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
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("   ⚠️  No GPU detected — training will be slow!")

# %% [markdown]
# ## 3. EW Simulation Core (Inlined)
#
# All simulation modules are bundled here so the notebook is fully self-contained.

# %% [markdown]
# ### 3a. Emitter Models

# %%
# ═══════════════════════════════════════════════════════════════════════
# ew_sim/emitters.py — Emitter models for the EW Smart Scan simulation
# ═══════════════════════════════════════════════════════════════════════

class BaseEmitter(ABC):
    """Abstract base for all emitter types."""

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
    """Fixed carrier pulsed radar."""

    def __init__(
        self, emitter_id: int, band_index: int,
        pri_sec: float = 1e-3, pulse_width: float = 2e-6,
        pri_jitter: float = 0.0, **kwargs,
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
    """Frequency-Hopping Spread Spectrum emitter with Markov transitions."""

    def __init__(
        self, emitter_id: int, hop_bands: list[int],
        transition_mat: Optional[np.ndarray] = None,
        hop_interval: float = 5e-3, pulse_width: float = 2e-6,
        pri_sec: float = 1e-3, burst_size: int = 5, **kwargs,
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
        self._current_hop_idx = 0

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
    """Spatially scanning surveillance radar with sinc² beam pattern."""

    def __init__(
        self, emitter_id: int, band_index: int,
        T_scan_sec: float = 3.0, beamwidth_deg: float = 2.0,
        pri_sec: float = 1e-3, pulse_width: float = 2e-6,
        initial_angle: float = 0.0, gain_threshold: float = 0.5,
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


def make_default_scenario(K: int = 35, rng: Optional[np.random.Generator] = None) -> list[BaseEmitter]:
    """Canonical 5-emitter mixed scenario."""
    if rng is None:
        rng = np.random.default_rng(42)
    return [
        FixedFrequencyEmitter(0, band_index=4, pri_sec=1e-3, pulse_width=2e-6, rng=np.random.default_rng(1)),
        FixedFrequencyEmitter(1, band_index=18, pri_sec=3e-3, pulse_width=5e-6, pri_jitter=0.02, rng=np.random.default_rng(2)),
        FHSSEmitter(2, hop_bands=[7, 10, 14, 20, 26], hop_interval=5e-3, pri_sec=1e-3, pulse_width=2e-6, burst_size=4, rng=np.random.default_rng(3)),
        FHSSEmitter(3, hop_bands=[2, 5, 12, 19, 29, 33], hop_interval=2e-3, pri_sec=0.5e-3, pulse_width=1e-6, burst_size=3, rng=np.random.default_rng(4)),
        ScanningEmitter(4, band_index=11, T_scan_sec=3.0, beamwidth_deg=2.0, pri_sec=1e-3, pulse_width=2e-6, initial_angle=45.0, rng=np.random.default_rng(5)),
    ]

# %% [markdown]
# ### 3b. Truth Engine

# %%
# ═══════════════════════════════════════════════════════════════════════
# ew_sim/truth_engine.py — Ground truth spectrum occupancy matrix
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
    """Generates S[K, T] ground-truth binary spectrum occupancy matrix."""

    def __init__(
        self, K: int = 35, T: int = 5000,
        dwell_us: float = 50.0, switch_us: float = 5.0,
        f_min_ghz: float = 0.5, f_max_ghz: float = 18.0,
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

    def build(self, verbose: bool = True) -> np.ndarray:
        self.S = np.zeros((self.K, self.T), dtype=np.int8)
        if verbose:
            print(f"[TruthEngine] Building S[{self.K}, {self.T}] with {len(self._emitters)} emitters …")

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

        if verbose:
            occupancy = self.S.mean() * 100
            print(f"[TruthEngine] Done. Global occupancy = {occupancy:.2f}%")
            for eid, act in self.activity.items():
                print(f"  Emitter {eid}: {act.total_pulses} active slots "
                      f"({act.total_pulses / self.T * 100:.1f}% duty)")
        return self.S

    def is_active(self, band: int, slot: int) -> bool:
        if self.S is None:
            raise RuntimeError("Call build() before querying the truth engine.")
        return bool(self.S[band, slot])

    def occupancy_per_band(self) -> np.ndarray:
        if self.S is None:
            raise RuntimeError("Call build() first.")
        return self.S.mean(axis=1)


def build_default_truth_engine(K=35, T=5000, dwell_us=50.0, switch_us=5.0, seed=42, verbose=True):
    engine = TruthEngine(K=K, T=T, dwell_us=dwell_us, switch_us=switch_us, rng=np.random.default_rng(seed))
    emitters = make_default_scenario(K=K, rng=np.random.default_rng(seed + 1))
    engine.add_emitters(emitters)
    engine.build(verbose=verbose)
    return engine

# %% [markdown]
# ### 3c. Gymnasium Environment

# %%
# ═══════════════════════════════════════════════════════════════════════
# ew_sim/env.py — EW Spectrum Scanning Gymnasium Environment
# ═══════════════════════════════════════════════════════════════════════

class EWSpectrumEnv(gym.Env):
    """EW Spectrum Scanning Environment with anti-camping and Bayesian belief."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        truth_engine: Optional[TruthEngine] = None,
        K: int = 35, T: int = 2000,
        Pd: float = 0.95, Pfa: float = 1e-4,
        w_hit: float = 6.0, w_new: float = 40.0,
        w_aoi: float = 2.5, w_switch: float = 0.2,
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

        if truth_engine is not None:
            self.truth = truth_engine
            self.K = truth_engine.K
        else:
            self.truth = build_default_truth_engine(K=K, T=T, verbose=False, seed=seed if seed is not None else 42)
            self.K = K

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3 * self.K,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.K)
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._belief = np.ones(self.K, dtype=np.float32) * 0.5
        self._aoi = np.zeros(self.K, dtype=np.float32)
        self._last_action = 0
        self._consecutive_dwells = 0
        self._discovered: set[int] = set()
        self._total_hits = 0
        self._total_dwells = 0
        self._false_alarms = 0

    def reset(self, *, seed=None, options=None):
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

    def _compute_reward(self, action, hit, new_discovery):
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

    def _update_belief(self, action, hit):
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

    def _get_obs(self):
        aoi_norm = self._aoi / self.aoi_max
        last_action_oh = np.zeros(self.K, dtype=np.float32)
        last_action_oh[self._last_action] = 1.0
        return np.concatenate([self._belief.astype(np.float32), aoi_norm.astype(np.float32), last_action_oh])

    def _get_info(self):
        return {
            "t": self._t,
            "total_hits": self._total_hits,
            "total_dwells": self._total_dwells,
            "false_alarms": self._false_alarms,
            "emitters_discovered": len(self._discovered),
            "num_emitters": len(self.truth._emitters),
        }

    @property
    def current_step(self): return self._t

    @property
    def hit_rate(self): return self._total_hits / max(self._total_dwells, 1)

    @property
    def discovery_ratio(self): return len(self._discovered) / max(len(self.truth._emitters), 1)

# %% [markdown]
# ## 4. Training Configuration

# %%
# ═══════════════════════════════════════════════════════════════════════
# Training hyperparameters
# ═══════════════════════════════════════════════════════════════════════

TOTAL_STEPS = 1_000_000    # 1M total timesteps across curriculum
NUM_ENVS = 16              # Parallel SubprocVecEnv workers
K = 35                     # Sub-bands
T = 1000                   # Episode length (time slots)
CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

STEPS_PER_STAGE = TOTAL_STEPS // 3
FINAL_STAGE_STEPS = TOTAL_STEPS - 2 * STEPS_PER_STAGE  # Handle remainder

print(f"📊 Training Configuration:")
print(f"   Total Steps: {TOTAL_STEPS:,}")
print(f"   Parallel Envs: {NUM_ENVS}")
print(f"   Steps per Stage: {STEPS_PER_STAGE:,} / {STEPS_PER_STAGE:,} / {FINAL_STAGE_STEPS:,}")
print(f"   Sub-bands (K): {K}")
print(f"   Episode Length (T): {T}")

# %% [markdown]
# ## 5. Environment Factory with Domain Randomization

# %%
def make_randomized_env_fn(stage: int, K: int = 35, T: int = 1000, seed: int = 42):
    """Factory returning an environment constructor with domain randomization."""
    def _init():
        rng = np.random.default_rng(seed)
        engine = TruthEngine(K=K, T=T, dwell_us=1000, switch_us=50, rng=rng)

        if stage == 1:
            # Stage 1: Fixed frequency radars on random bands
            b1 = int(rng.integers(1, 10))
            b2 = int(rng.integers(15, 25))
            engine.add_emitters([
                FixedFrequencyEmitter(0, band_index=b1, pri_sec=5.25e-3, pulse_width=1.05e-3),
                FixedFrequencyEmitter(1, band_index=b2, pri_sec=8.4e-3, pulse_width=1.05e-3, pri_jitter=0.08),
            ])
        elif stage == 2:
            # Stage 2: Frequency hopping radars
            hop_set1 = list(rng.choice(range(K), size=5, replace=False))
            hop_set2 = list(rng.choice(range(K), size=6, replace=False))
            engine.add_emitters([
                FixedFrequencyEmitter(0, band_index=4, pri_sec=5.25e-3, pulse_width=1.05e-3),
                FHSSEmitter(1, hop_bands=hop_set1, hop_interval=8.4e-3, pri_sec=2.1e-3, pulse_width=1.05e-3, burst_size=3),
                FHSSEmitter(2, hop_bands=hop_set2, hop_interval=6.3e-3, pri_sec=2.1e-3, pulse_width=1.05e-3, burst_size=2),
            ])
        else:
            # Stage 3: Full tactical battlefield
            hop_set = list(rng.choice(range(K), size=6, replace=False))
            scan_band = int(rng.integers(8, 28))
            engine.add_emitters([
                FixedFrequencyEmitter(0, band_index=3, pri_sec=5.25e-3, pulse_width=1.05e-3),
                FixedFrequencyEmitter(1, band_index=17, pri_sec=10.5e-3, pulse_width=1.05e-3, pri_jitter=0.05),
                FHSSEmitter(2, hop_bands=hop_set, hop_interval=8.4e-3, pri_sec=2.1e-3, pulse_width=1.05e-3, burst_size=3),
                ScanningEmitter(3, band_index=scan_band, T_scan_sec=2.1, beamwidth_deg=10.0, pri_sec=2.1e-3, pulse_width=1.05e-3),
            ])

        engine.build(verbose=False)
        return EWSpectrumEnv(truth_engine=engine, K=K, T=T, seed=seed)
    return _init

# %% [markdown]
# ## 6. Telemetry Logger

# %%
class TacticalTelemetryLogger(BaseCallback):
    """Logs curriculum progression, reward metrics, and training speed."""

    def __init__(self, stage_name: str = "", check_freq: int = 5000, verbose: int = 1):
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
# ## 7. 🚀 Curriculum Training Loop

# %%
print("=" * 70)
print(f"  🛡️ DRDO EW Smart Scan — Large-Scale Training on {device.upper()}")
print(f"  Target Steps: {TOTAL_STEPS:,} across {NUM_ENVS} Parallel Vectorized Envs")
print("=" * 70)

training_log = {}
overall_start = time.time()

# ──────────────────────────────────────────────────────────────────────
# Stage 1: Fixed Emitters
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  [Stage 1/3] Fixed-Frequency Radars ({STEPS_PER_STAGE:,} steps)")
print(f"{'='*60}")

env_fns = [make_randomized_env_fn(stage=1, seed=100 + i) for i in range(NUM_ENVS)]
vec_env1 = SubprocVecEnv(env_fns)
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
    ent_coef=0.08,
    device=device,
    verbose=0,
)

stage1_start = time.time()
cb1 = TacticalTelemetryLogger(stage_name="Stage 1", check_freq=10000)
model.learn(total_timesteps=STEPS_PER_STAGE, callback=cb1)
stage1_time = time.time() - stage1_start
vec_env1.close()

# Save Stage 1 checkpoint
model.save(CHECKPOINT_DIR / "stage1_fixed_emitters.zip")
training_log["stage1"] = {"steps": STEPS_PER_STAGE, "time_sec": stage1_time}
print(f"\n  ✅ Stage 1 complete in {stage1_time:.0f}s ({STEPS_PER_STAGE/stage1_time:.0f} steps/sec)")

# ──────────────────────────────────────────────────────────────────────
# Stage 2: Agile Frequency Hopping
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  [Stage 2/3] FHSS Agile Emitters ({STEPS_PER_STAGE:,} steps)")
print(f"{'='*60}")

env_fns2 = [make_randomized_env_fn(stage=2, seed=200 + i) for i in range(NUM_ENVS)]
vec_env2 = SubprocVecEnv(env_fns2)
vec_env2 = VecMonitor(vec_env2)
model.set_env(vec_env2)

stage2_start = time.time()
cb2 = TacticalTelemetryLogger(stage_name="Stage 2", check_freq=10000)
model.learn(total_timesteps=STEPS_PER_STAGE, callback=cb2)
stage2_time = time.time() - stage2_start
vec_env2.close()

model.save(CHECKPOINT_DIR / "stage2_fhss_emitters.zip")
training_log["stage2"] = {"steps": STEPS_PER_STAGE, "time_sec": stage2_time}
print(f"\n  ✅ Stage 2 complete in {stage2_time:.0f}s ({STEPS_PER_STAGE/stage2_time:.0f} steps/sec)")

# ──────────────────────────────────────────────────────────────────────
# Stage 3: Full Tactical Battlefield
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  [Stage 3/3] Full Battlefield & Scanning Radars ({FINAL_STAGE_STEPS:,} steps)")
print(f"{'='*60}")

env_fns3 = [make_randomized_env_fn(stage=3, seed=300 + i) for i in range(NUM_ENVS)]
vec_env3 = SubprocVecEnv(env_fns3)
vec_env3 = VecMonitor(vec_env3)
model.set_env(vec_env3)

stage3_start = time.time()
cb3 = TacticalTelemetryLogger(stage_name="Stage 3", check_freq=10000)
model.learn(total_timesteps=FINAL_STAGE_STEPS, callback=cb3)
stage3_time = time.time() - stage3_start
vec_env3.close()

model.save(CHECKPOINT_DIR / "stage3_full_battlefield.zip")
training_log["stage3"] = {"steps": FINAL_STAGE_STEPS, "time_sec": stage3_time}
print(f"\n  ✅ Stage 3 complete in {stage3_time:.0f}s ({FINAL_STAGE_STEPS/stage3_time:.0f} steps/sec)")

# %% [markdown]
# ## 8. Save Final Model & Training Report

# %%
# Save final checkpoint (the one we'll download)
final_path = CHECKPOINT_DIR / "ppo_recurrent_kaggle_1m.zip"
model.save(final_path)

total_time = time.time() - overall_start
training_log["total_time_sec"] = total_time
training_log["total_steps"] = TOTAL_STEPS
training_log["device"] = device
training_log["num_envs"] = NUM_ENVS
if device == "cuda":
    training_log["gpu"] = torch.cuda.get_device_name(0)

# Save training report
report_path = CHECKPOINT_DIR / "training_report.json"
with open(report_path, "w") as f:
    json.dump(training_log, f, indent=2)

print(f"\n{'='*70}")
print(f"  🎯 TRAINING COMPLETE")
print(f"{'='*70}")
print(f"  Total Time:    {total_time/60:.1f} minutes")
print(f"  Total Steps:   {TOTAL_STEPS:,}")
print(f"  Avg Throughput: {TOTAL_STEPS/total_time:.0f} steps/sec")
print(f"  Final Model:   {final_path}")
print(f"  Training Log:  {report_path}")
print(f"{'='*70}")

# List output files
print("\n📁 Output files in /kaggle/working/checkpoints/:")
for p in sorted(CHECKPOINT_DIR.iterdir()):
    size_mb = p.stat().st_size / 1e6
    print(f"   {p.name:40s} {size_mb:.1f} MB")

# %% [markdown]
# ## 9. Quick Validation (Post-Training Sanity Check)

# %%
# Run a quick 1-episode evaluation on Stage 3 scenario
print("\n🧪 Post-Training Sanity Check (1 episode, Stage 3 scenario)...")

eval_engine = TruthEngine(K=35, T=1000, dwell_us=1000, switch_us=50, rng=np.random.default_rng(999))
eval_engine.add_emitters([
    FixedFrequencyEmitter(0, band_index=3, pri_sec=5.25e-3, pulse_width=1.05e-3),
    FixedFrequencyEmitter(1, band_index=17, pri_sec=10.5e-3, pulse_width=1.05e-3, pri_jitter=0.05),
    FHSSEmitter(2, hop_bands=[5, 10, 15, 20, 25, 30], hop_interval=8.4e-3, pri_sec=2.1e-3, pulse_width=1.05e-3, burst_size=3),
    ScanningEmitter(3, band_index=22, T_scan_sec=2.1, beamwidth_deg=10.0, pri_sec=2.1e-3, pulse_width=1.05e-3),
])
eval_engine.build(verbose=False)
eval_env = EWSpectrumEnv(truth_engine=eval_engine, K=35, T=1000, seed=999)

# Load and evaluate
eval_model = RecurrentPPO.load(final_path)
obs, info = eval_env.reset()
lstm_states = None
episode_done = False
total_reward = 0.0
n_steps = 0

while not episode_done:
    action, lstm_states = eval_model.predict(obs, state=lstm_states, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(int(action))
    total_reward += reward
    n_steps += 1
    episode_done = terminated or truncated

print(f"\n  📊 Evaluation Results:")
print(f"     Steps:          {n_steps}")
print(f"     Total Reward:   {total_reward:+.1f}")
print(f"     Hit Rate:       {eval_env.hit_rate*100:.1f}%")
print(f"     Discovery Rate: {eval_env.discovery_ratio*100:.0f}%")
print(f"     Hits/Dwells:    {info['total_hits']}/{info['total_dwells']}")
print(f"     False Alarms:   {info['false_alarms']}")

# %% [markdown]
# ---
# *SIH 2026 — EW Smart Scan · AI-Driven Spectrum Scheduling for DRDO Electronic Support*
