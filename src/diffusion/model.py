"""A memory-conscious 3D U-Net for video denoising.

Only the spatial axes are downsampled.  Time remains aligned at every level,
and every residual block uses a 3x3x3 convolution so adjacent frames influence
one another instead of being generated as unrelated images.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def spatial_resize(video: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize video frames independently, without interpolating across time."""
    batch, channels, frames, height, width = video.shape
    if (height, width) == size:
        return video
    flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    flat = F.interpolate(flat, size=size, mode="bilinear", align_corners=False)
    return flat.reshape(batch, frames, channels, *size).permute(0, 2, 1, 3, 4)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10_000) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        )
        angles = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[-1]))
        return embedding


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.skip(inputs)


class SpatialDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class SpatialUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, scale_factor=(1, 2, 2), mode="nearest")
        return self.conv(inputs)


@dataclass(frozen=True)
class VideoUNetConfig:
    channels: int = 3
    condition_channels: int = 0
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    blocks_per_level: int = 2
    time_embedding_dim: int = 256
    dropout: float = 0.0
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "channel_multipliers", tuple(int(value) for value in self.channel_multipliers)
        )
        if self.channels < 1 or self.condition_channels < 0:
            raise ValueError("channels must be positive and condition_channels non-negative")
        if self.base_channels < 1 or not self.channel_multipliers:
            raise ValueError("base_channels and channel_multipliers must define a non-empty U-Net")
        if self.blocks_per_level < 1:
            raise ValueError("blocks_per_level must be positive")
        if self.time_embedding_dim < 4:
            raise ValueError("time_embedding_dim must be at least 4")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["channel_multipliers"] = list(self.channel_multipliers)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "VideoUNetConfig":
        return cls(**payload)


class VideoUNet(nn.Module):
    """Predict noise for an unconditional or low-resolution-conditioned video."""

    def __init__(self, config: VideoUNetConfig):
        super().__init__()
        self.config = config
        widths = [config.base_channels * multiplier for multiplier in config.channel_multipliers]
        input_channels = config.channels + config.condition_channels

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(config.time_embedding_dim),
            nn.Linear(config.time_embedding_dim, config.time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(config.time_embedding_dim * 4, config.time_embedding_dim),
        )
        self.input_conv = nn.Conv3d(input_channels, widths[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = widths[0]
        for level, width in enumerate(widths):
            blocks = nn.ModuleList()
            for _ in range(config.blocks_per_level):
                blocks.append(
                    ResidualBlock3D(current, width, config.time_embedding_dim, config.dropout)
                )
                current = width
            self.down_blocks.append(blocks)
            if level < len(widths) - 1:
                self.downsamples.append(SpatialDownsample(current, widths[level + 1]))
                current = widths[level + 1]

        self.middle = nn.ModuleList(
            [
                ResidualBlock3D(current, current, config.time_embedding_dim, config.dropout),
                ResidualBlock3D(current, current, config.time_embedding_dim, config.dropout),
            ]
        )

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(widths))):
            width = widths[level]
            blocks = nn.ModuleList(
                [
                    ResidualBlock3D(
                        current + width, width, config.time_embedding_dim, config.dropout
                    )
                ]
            )
            current = width
            for _ in range(config.blocks_per_level - 1):
                blocks.append(
                    ResidualBlock3D(current, width, config.time_embedding_dim, config.dropout)
                )
            self.up_blocks.append(blocks)
            if level > 0:
                self.upsamples.append(SpatialUpsample(current, widths[level - 1]))
                current = widths[level - 1]

        self.output_norm = nn.GroupNorm(_groups(current), current)
        self.output_conv = nn.Conv3d(current, config.channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def _block(
        self, block: ResidualBlock3D, inputs: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training:
            return checkpoint(block, inputs, time_embedding, use_reentrant=False)
        return block(inputs, time_embedding)

    def forward(
        self,
        noisy_video: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_video.ndim != 5:
            raise ValueError(f"Expected BxCxTxHxW video, got {tuple(noisy_video.shape)}")
        if noisy_video.shape[1] != self.config.channels:
            raise ValueError(
                f"Expected {self.config.channels} noisy-video channels, got {noisy_video.shape[1]}"
            )
        if timesteps.ndim != 1 or timesteps.shape[0] != noisy_video.shape[0]:
            raise ValueError(
                f"Expected one timestep per video, got {tuple(timesteps.shape)} for "
                f"batch size {noisy_video.shape[0]}"
            )
        factor = 2 ** (len(self.config.channel_multipliers) - 1)
        if noisy_video.shape[-2] % factor or noisy_video.shape[-1] % factor:
            raise ValueError(
                f"Spatial dimensions {tuple(noisy_video.shape[-2:])} must be divisible by {factor}"
            )
        if self.config.condition_channels:
            if condition is None:
                raise ValueError("This U-Net requires a low-resolution conditioning video")
            if condition.ndim != 5:
                raise ValueError(f"Expected BxCxTxHxW condition, got {tuple(condition.shape)}")
            if (
                condition.shape[0] != noisy_video.shape[0]
                or condition.shape[2] != noisy_video.shape[2]
            ):
                raise ValueError(
                    "Condition batch and temporal dimensions must match the noisy video: "
                    f"{tuple(condition.shape)} vs {tuple(noisy_video.shape)}"
                )
            if condition.shape[1] != self.config.condition_channels:
                raise ValueError(
                    f"Expected {self.config.condition_channels} condition channels, "
                    f"got {condition.shape[1]}"
                )
            condition = spatial_resize(condition, tuple(noisy_video.shape[-2:]))
            inputs = torch.cat((noisy_video, condition), dim=1)
        else:
            if condition is not None:
                raise ValueError("An unconditional U-Net does not accept a condition")
            inputs = noisy_video

        time_embedding = self.time_embedding(timesteps)
        hidden = self.input_conv(inputs)
        skips: list[torch.Tensor] = []
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                hidden = self._block(block, hidden, time_embedding)
            skips.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)

        for block in self.middle:
            hidden = self._block(block, hidden, time_embedding)

        for up_index, blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            if hidden.shape[2:] != skip.shape[2:]:
                raise RuntimeError(
                    f"U-Net skip shape mismatch: {tuple(hidden.shape)} and {tuple(skip.shape)}"
                )
            hidden = torch.cat((hidden, skip), dim=1)
            for block in blocks:
                hidden = self._block(block, hidden, time_embedding)
            if up_index < len(self.upsamples):
                hidden = self.upsamples[up_index](hidden)

        return self.output_conv(F.silu(self.output_norm(hidden)))
