"""
tests/test_drl.py
Unit tests for DRLScheduler and PyTorch Recurrent Actor-Critic network.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from schedulers.drl_agent import RecurrentActorCriticNet, DRLScheduler


class TestDRLAgent:
    def test_recurrent_net_forward_shapes(self):
        K = 10
        obs_dim = 3 * K
        net = RecurrentActorCriticNet(obs_dim=obs_dim, action_dim=K)

        x = torch.randn(1, obs_dim)
        hidden = net.init_hidden(batch_size=1)

        logits, value, next_hidden = net(x, hidden)
        assert logits.shape == (1, K)
        assert value.shape == (1, 1)
        assert next_hidden.shape == (2, 1, 256)

    def test_drl_scheduler_select_band(self):
        K = 10
        scheduler = DRLScheduler(K=K, seed=42)
        obs = np.random.rand(3 * K).astype(np.float32)

        action = scheduler.select_band(obs)
        assert 0 <= action < K

    def test_drl_save_and_load(self):
        K = 8
        scheduler = DRLScheduler(K=K, seed=42)
        obs = np.ones(3 * K, dtype=np.float32)

        action_before = scheduler.select_band(obs)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "test_drl.pt"
            scheduler.save(ckpt)

            new_scheduler = DRLScheduler(K=K, model_path=ckpt, seed=42)
            action_after = new_scheduler.select_band(obs)
            assert action_before == action_after
