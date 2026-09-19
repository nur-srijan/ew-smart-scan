"""
schedulers/multi_drl.py
=======================
Multi-Receiver Cooperative Deep Reinforcement Learning (DRL) Scheduler.

Implements a Multi-Head Recurrent Actor-Critic network (GRU + Multi-Categorical heads)
coordinating M independent receiver tuners across K sub-bands.

Features:
    1. Multi-Head Actor: M parallel action heads outputting (M, K) band logits.
    2. Sequential Greedy Collision Masking: Dynamically suppresses previously assigned
       bands for subsequent tuner heads, mathematically guaranteeing 0.0% collisions.
    3. Zero-SB3 Standalone PyTorch Execution: Lightweight CPU/MPS inference for edge runtime.
    4. ONNX Export Ready: Seamless compilation to TensorRT INT8 for NVIDIA Jetson Orin Nano.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from schedulers.multi_schedulers import BaseMultiScheduler


# ---------------------------------------------------------------------------
# Pure PyTorch Multi-Head Recurrent Actor-Critic Network
# ---------------------------------------------------------------------------

class MultiRecurrentActorCriticNet(nn.Module):
    """
    Multi-Head 2-Layer GRU Actor-Critic Architecture for M-Tuner EW Scanning.

    Parameters
    ----------
    obs_dim : int
        Observation dimension (3 * K = 105 for K=35).
    K : int
        Number of frequency sub-bands (default: 35).
    M : int
        Number of concurrent receiver tuners (default: 4).
    hidden_dim : int
        Recurrent memory size (default: 256).
    """

    def __init__(
        self,
        obs_dim: int = 105,
        K: int = 35,
        M: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.K = K
        self.M = M
        self.hidden_dim = hidden_dim

        # Input feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Recurrent temporal memory (remembers multi-second radar scan periods)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # M parallel actor heads (one per tuner)
        self.actor_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, K),
            )
            for _ in range(M)
        ])

        # State value critic head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, obs_dim) or (batch, seq_len, obs_dim)
        hidden : Optional recurrent hidden state of shape (2, batch, hidden_dim)

        Returns
        -------
        logits : Tensor of shape (batch, M, K)
        value : Tensor of shape (batch, 1)
        next_hidden : Tensor of shape (2, batch, hidden_dim)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)
        elif x.dim() == 2:
            x = x.unsqueeze(1)               # (batch, 1, obs_dim)

        batch_size = x.shape[0]
        feat = self.encoder(x)
        gru_out, next_hidden = self.gru(feat, hidden)

        last_feat = gru_out[:, -1, :]  # (batch, hidden_dim)

        # Compute logits for each tuner head
        head_logits = [head(last_feat).unsqueeze(1) for head in self.actor_heads]
        logits = torch.cat(head_logits, dim=1)  # (batch, M, K)

        value = self.critic_head(last_feat)     # (batch, 1)

        return logits, value, next_hidden

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Create zero-initialized GRU hidden states."""
        return torch.zeros(2, batch_size, self.hidden_dim, dtype=torch.float32)

    def select_actions(
        self,
        obs: np.ndarray,
        hidden: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        collision_masking: bool = True,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """
        Select M band actions with collision masking guarantee.

        Returns
        -------
        actions : np.ndarray of shape (M,)
        next_hidden : torch.Tensor
        """
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, _, next_hidden = self.forward(x, hidden)
            # logits: (1, M, K)
            logits_m = logits[0]  # (M, K)

            actions = np.zeros(self.M, dtype=int)
            mask = torch.zeros(self.K, dtype=torch.bool, device=logits.device)

            for m in range(self.M):
                lm = logits_m[m].clone()
                if collision_masking:
                    lm[mask] = -1e9

                if deterministic:
                    a = int(torch.argmax(lm).item())
                else:
                    probs = F.softmax(lm, dim=-1)
                    a = int(torch.multinomial(probs, num_samples=1).item())

                actions[m] = a
                mask[a] = True

        return actions, next_hidden


# ---------------------------------------------------------------------------
# Multi-Receiver DRL Scheduler Policy Wrapper
# ---------------------------------------------------------------------------

class MultiDRLScheduler(BaseMultiScheduler):
    """
    Cooperative Multi-Receiver Electronic Warfare Scheduler powered by DRL.

    Parameters
    ----------
    K : int
        Number of frequency sub-bands (default: 35).
    M : int
        Number of concurrent receiver tuners (default: 4).
    model_path : Optional[str or Path]
        Path to PyTorch weights (.pt) or SB3 checkpoint (.zip).
    deterministic : bool
        Whether to select actions deterministically (greedy argmax).
    collision_masking : bool
        Whether to enforce sequential masking to guarantee 0% collisions.
    device : str
        Compute device ('cpu', 'mps', 'cuda').
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        K: int = 35,
        M: int = 4,
        model_path: Optional[Union[str, Path]] = None,
        deterministic: bool = True,
        collision_masking: bool = True,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        super().__init__(K=K, M=M, name="MultiDRLScheduler", seed=seed)
        self.deterministic = deterministic
        self.collision_masking = collision_masking
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        obs_dim = 3 * K
        self.net = MultiRecurrentActorCriticNet(obs_dim=obs_dim, K=K, M=M).to(self.device)
        self.hidden: Optional[torch.Tensor] = None

        if model_path is not None:
            self.load_model(model_path)

        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        if getattr(self, "sb3_model", None) is not None:
            self.hidden = None
        else:
            self.hidden = self.net.init_hidden(batch_size=1).to(self.device)

    def load_model(self, model_path: Union[str, Path]) -> None:
        """Loads weights from either a standalone PyTorch state_dict (.pt) or an SB3 zip."""
        p = Path(model_path)
        if not p.exists():
            print(f"[MultiDRLScheduler] Warning: Checkpoint {p} not found, using initialized weights.")
            return

        if p.suffix == ".pt":
            state_dict = torch.load(p, map_location=self.device, weights_only=True)
            self.net.load_state_dict(state_dict, strict=False)
            self.sb3_model = None
            print(f"[MultiDRLScheduler] Loaded PyTorch state_dict from {p}")
        elif p.suffix == ".zip":
            try:
                from sb3_contrib import RecurrentPPO
                self.sb3_model = RecurrentPPO.load(str(p), device=str(self.device))
                print(f"[MultiDRLScheduler] Loaded SB3 RecurrentPPO checkpoint from {p}")
            except Exception as e:
                self.sb3_model = None
                print(f"[MultiDRLScheduler] SB3 load exception: {e}. Using initialized PyTorch weights.")

    def _choose_bands(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        if getattr(self, "sb3_model", None) is not None:
            ep_start = np.array([self.t == 0])
            actions_raw, self.hidden = self.sb3_model.predict(
                obs,
                state=self.hidden,
                episode_start=ep_start,
                deterministic=self.deterministic,
            )
            actions = np.asarray(actions_raw, dtype=int).flatten()

            if self.collision_masking and len(np.unique(actions)) < self.M:
                seen = set()
                cleaned = []
                for a in actions:
                    if a not in seen and 0 <= a < self.K:
                        seen.add(a)
                        cleaned.append(a)
                for cand in range(self.K):
                    if len(cleaned) == self.M:
                        break
                    if cand not in seen:
                        seen.add(cand)
                        cleaned.append(cand)
                actions = np.array(cleaned[:self.M], dtype=int)

            return actions

        actions, self.hidden = self.net.select_actions(
            obs=obs,
            hidden=self.hidden,
            deterministic=self.deterministic,
            collision_masking=self.collision_masking,
        )
        return actions

    def save_pytorch_weights(self, output_path: Union[str, Path]) -> None:
        """Saves PyTorch state_dict for zero-dependency edge deployment."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), out)
        print(f"[MultiDRLScheduler] Saved PyTorch weights to {out}")
