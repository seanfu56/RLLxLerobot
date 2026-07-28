"""Normalisation and the diffusion timestep embedding."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

NORM_MODES = ("meanstd", "minmax", "identity")


class Normalizer(nn.Module):
    """Per-dimension affine normalisation stored as buffers so checkpoints are self-contained.

    Getting this right matters more than it looks: joint angles live in degrees
    (range ~1) while the gripper spans 0-100, so an un-normalised state vector is
    dominated by one channel and the network learns to ignore the rest.
    """

    def __init__(self, mode: str, stats: dict[str, list[float]] | None, dim: int, eps: float = 1e-3):
        super().__init__()
        if mode not in NORM_MODES:
            raise ValueError(f"mode must be one of {NORM_MODES}, got {mode!r}")
        self.mode = mode
        if mode == "identity" or stats is None:
            loc = torch.zeros(dim)
            scale = torch.ones(dim)
        elif mode == "meanstd":
            loc = torch.tensor(stats["mean"], dtype=torch.float32)
            scale = torch.tensor(stats["std"], dtype=torch.float32).clamp_min(eps)
        else:
            minimum = torch.tensor(stats["min"], dtype=torch.float32)
            maximum = torch.tensor(stats["max"], dtype=torch.float32)
            loc = (maximum + minimum) / 2
            scale = ((maximum - minimum) / 2).clamp_min(eps)
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def normalize(self, values: Tensor) -> Tensor:
        return (values - self.loc) / self.scale

    def denormalize(self, values: Tensor) -> Tensor:
        return values * self.scale + self.loc

    forward = normalize


def sinusoidal_embedding(timesteps: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    """Standard diffusion timestep embedding, [B] -> [B, dim]."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
