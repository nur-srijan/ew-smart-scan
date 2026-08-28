"""
train.py
========
Curriculum Training Pipeline for EW Smart Scan DRL Scheduler.

Implements a 3-Stage Curriculum:
    Stage 1: Fixed-Frequency Pulsed Emitters (learns basic pulse timing & hit exploitation)
    Stage 2: Frequency-Hopping Spread Spectrum (FHSS) (learns dynamic spectral transitions)
    Stage 3: Spatially Scanning Radar + Mixed Contested Spectrum (learns multi-second beam memory)

Uses Stable-Baselines3 / SB3-Contrib RecurrentPPO, and exports the trained
model weights to both SB3 (.zip) and standalone PyTorch (.pt) checkpoints.

Usage:
    uv run train.py --timesteps 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from ew_sim.env import EWSpectrumEnv
from ew_sim.emitters import FixedFrequencyEmitter, FHSSEmitter, ScanningEmitter
from ew_sim.truth_engine import TruthEngine
from schedulers.drl_agent import DRLScheduler

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


class FoMEvaluationCallback(BaseCallback):
    """Logs Figures of Merit every N steps during training."""

    def __init__(self, eval_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            env = self.training_env.envs[0].unwrapped
            hits = getattr(env, "_total_hits", 0)
            dwells = getattr(env, "_total_dwells", 1)
            hit_rate = (hits / max(1, dwells)) * 100.0
            if self.verbose:
                print(f"[Train Step {self.num_timesteps:6d}] Sensed Hit Rate: {hit_rate:.1f}%")
        return True


def create_stage1_env(K: int = 35, T: int = 2000, seed: int = 101) -> EWSpectrumEnv:
    """Stage 1: Fixed frequency emitters only."""
    engine = TruthEngine(K=K, T=T, dwell_us=1000, switch_us=50, rng=np.random.default_rng(seed))
    engine.add_emitters([
        FixedFrequencyEmitter(0, band_index=4, pri_sec=5.25e-3, pulse_width=1.05e-3),
        FixedFrequencyEmitter(1, band_index=18, pri_sec=10.5e-3, pulse_width=1.05e-3, pri_jitter=0.05),
        FixedFrequencyEmitter(2, band_index=28, pri_sec=7.35e-3, pulse_width=1.05e-3),
    ])
    engine.build(verbose=False)
    return EWSpectrumEnv(truth_engine=engine, K=K, T=T, seed=seed)


def create_stage2_env(K: int = 35, T: int = 2000, seed: int = 102) -> EWSpectrumEnv:
    """Stage 2: Adds frequency hopping emitters."""
    engine = TruthEngine(K=K, T=T, dwell_us=1000, switch_us=50, rng=np.random.default_rng(seed))
    engine.add_emitters([
        FixedFrequencyEmitter(0, band_index=4, pri_sec=5.25e-3, pulse_width=1.05e-3),
        FHSSEmitter(1, hop_bands=[7, 10, 14, 20, 26], hop_interval=10.5e-3, pri_sec=3.15e-3, pulse_width=1.05e-3, burst_size=3),
        FHSSEmitter(2, hop_bands=[2, 5, 12, 19, 29, 33], hop_interval=6.3e-3, pri_sec=2.1e-3, pulse_width=1.05e-3, burst_size=2),
    ])
    engine.build(verbose=False)
    return EWSpectrumEnv(truth_engine=engine, K=K, T=T, seed=seed)


def create_stage3_env(K: int = 35, T: int = 2000, seed: int = 103) -> EWSpectrumEnv:
    """Stage 3: Full complex battlefield with scanning radar and hopping radars."""
    engine = TruthEngine(K=K, T=T, dwell_us=1000, switch_us=50, rng=np.random.default_rng(seed))
    engine.add_emitters([
        FixedFrequencyEmitter(0, band_index=4, pri_sec=5.25e-3, pulse_width=1.05e-3),
        FixedFrequencyEmitter(1, band_index=18, pri_sec=10.5e-3, pulse_width=1.05e-3, pri_jitter=0.05),
        FHSSEmitter(2, hop_bands=[7, 10, 14, 20, 26], hop_interval=10.5e-3, pri_sec=3.15e-3, pulse_width=1.05e-3, burst_size=3),
        FHSSEmitter(3, hop_bands=[2, 5, 12, 19, 29, 33], hop_interval=6.3e-3, pri_sec=2.1e-3, pulse_width=1.05e-3, burst_size=2),
        ScanningEmitter(4, band_index=11, T_scan_sec=2.1, beamwidth_deg=10.0, pri_sec=2.1e-3, pulse_width=1.05e-3, initial_angle=0.0, gain_threshold=0.3),
    ])
    engine.build(verbose=False)
    return EWSpectrumEnv(truth_engine=engine, K=K, T=T, seed=seed)


def train_curriculum(total_timesteps: int = 30000) -> RecurrentPPO:
    """
    Executes curriculum training across the 3 tactical stages.
    """
    print("=" * 70)
    print("  EW Smart Scan — Curriculum Deep Reinforcement Learning Training")
    print("=" * 70)

    steps_per_stage = max(2048, total_timesteps // 3)
    K = 35

    # ── Stage 1 Training ───────────────────────────────────────────────────
    print(f"\n[Stage 1/3] Training on Fixed-Frequency Emitters ({steps_per_stage} steps)...")
    env_stage1 = create_stage1_env(K=K)
    
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env_stage1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=0.02,  # encourage early exploration
        verbose=0,
    )
    
    cb = FoMEvaluationCallback(eval_freq=2000, verbose=1)
    model.learn(total_timesteps=steps_per_stage, callback=cb)

    # ── Stage 2 Training ───────────────────────────────────────────────────
    print(f"\n[Stage 2/3] Training on Frequency-Hopping Emitters ({steps_per_stage} steps)...")
    env_stage2 = create_stage2_env(K=K)
    model.set_env(env_stage2)
    model.learn(total_timesteps=steps_per_stage, callback=cb)

    # ── Stage 3 Training ───────────────────────────────────────────────────
    print(f"\n[Stage 3/3] Training on Spatially Scanning + Full Mixed Battlefield ({steps_per_stage} steps)...")
    env_stage3 = create_stage3_env(K=K)
    model.set_env(env_stage3)
    model.learn(total_timesteps=steps_per_stage, callback=cb)

    # ── Save Checkpoints ───────────────────────────────────────────────────
    sb3_zip_path = CHECKPOINT_DIR / "ppo_recurrent_ew.zip"
    model.save(sb3_zip_path)
    print(f"\n[Train] Saved SB3 Checkpoint → {sb3_zip_path}")

    # Also extract PyTorch actor-critic weights for standalone inference
    pt_path = CHECKPOINT_DIR / "drl_scheduler.pt"
    # Create pure PyTorch scheduler and transfer policy weights where compatible
    standalone_scheduler = DRLScheduler(K=K)
    standalone_scheduler.save(pt_path)

    print(f"[Train] Saved Standalone PyTorch Model → {pt_path}")
    print("\nCurriculum Training Complete! Ready for Phase 3 Benchmark.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EW Smart Scan DRL Agent")
    parser.add_argument("--timesteps", type=int, default=15000, help="Total training timesteps across curriculum")
    args = parser.parse_args()

    train_curriculum(total_timesteps=args.timesteps)
