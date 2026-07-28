"""Conv2d U-Net used only for 56x56 to 224x224 image super-resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .model import SinusoidalTimeEmbedding


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.skip(inputs)


class Downsample2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class Upsample2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(inputs, scale_factor=2, mode="nearest"))


@dataclass(frozen=True)
class ImageUNetConfig:
    channels: int = 3
    condition_channels: int = 3
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    blocks_per_level: int = 2
    time_embedding_dim: int = 256
    dropout: float = 0.0
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "channel_multipliers", tuple(int(value) for value in self.channel_multipliers)
        )
        if self.channels < 1 or self.condition_channels < 1:
            raise ValueError("Image super-resolution requires positive target and condition channels")
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
    def from_dict(cls, payload: dict) -> "ImageUNetConfig":
        return cls(**payload)


class ImageUNet(nn.Module):
    """Predict noise for one high-resolution image conditioned on its low-res copy."""

    def __init__(self, config: ImageUNetConfig):
        super().__init__()
        self.config = config
        widths = [config.base_channels * multiplier for multiplier in config.channel_multipliers]
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(config.time_embedding_dim),
            nn.Linear(config.time_embedding_dim, config.time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(config.time_embedding_dim * 4, config.time_embedding_dim),
        )
        self.input_conv = nn.Conv2d(
            config.channels + config.condition_channels,
            widths[0],
            kernel_size=3,
            padding=1,
        )

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = widths[0]
        for level, width in enumerate(widths):
            blocks = nn.ModuleList()
            for _ in range(config.blocks_per_level):
                blocks.append(
                    ResidualBlock2D(current, width, config.time_embedding_dim, config.dropout)
                )
                current = width
            self.down_blocks.append(blocks)
            if level < len(widths) - 1:
                self.downsamples.append(Downsample2D(current, widths[level + 1]))
                current = widths[level + 1]

        self.middle = nn.ModuleList(
            [
                ResidualBlock2D(current, current, config.time_embedding_dim, config.dropout),
                ResidualBlock2D(current, current, config.time_embedding_dim, config.dropout),
            ]
        )

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(widths))):
            width = widths[level]
            blocks = nn.ModuleList(
                [
                    ResidualBlock2D(
                        current + width, width, config.time_embedding_dim, config.dropout
                    )
                ]
            )
            current = width
            for _ in range(config.blocks_per_level - 1):
                blocks.append(
                    ResidualBlock2D(current, width, config.time_embedding_dim, config.dropout)
                )
            self.up_blocks.append(blocks)
            if level > 0:
                self.upsamples.append(Upsample2D(current, widths[level - 1]))
                current = widths[level - 1]

        self.output_norm = nn.GroupNorm(_groups(current), current)
        self.output_conv = nn.Conv2d(current, config.channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def _block(
        self, block: ResidualBlock2D, inputs: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training:
            return checkpoint(block, inputs, time_embedding, use_reentrant=False)
        return block(inputs, time_embedding)

    def forward(
        self,
        noisy_image: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_image.ndim != 4:
            raise ValueError(f"Expected BxCxHxW image batch, got {tuple(noisy_image.shape)}")
        if noisy_image.shape[1] != self.config.channels:
            raise ValueError(
                f"Expected {self.config.channels} image channels, got {noisy_image.shape[1]}"
            )
        if timesteps.ndim != 1 or timesteps.shape[0] != noisy_image.shape[0]:
            raise ValueError(
                f"Expected one timestep per image, got {tuple(timesteps.shape)} for "
                f"batch size {noisy_image.shape[0]}"
            )
        if condition is None or condition.ndim != 4:
            raise ValueError("Image super-resolution requires a BxCxHxW low-res condition")
        if (
            condition.shape[0] != noisy_image.shape[0]
            or condition.shape[1] != self.config.condition_channels
        ):
            raise ValueError(
                f"Condition shape {tuple(condition.shape)} is incompatible with "
                f"image shape {tuple(noisy_image.shape)}"
            )
        factor = 2 ** (len(self.config.channel_multipliers) - 1)
        if noisy_image.shape[-2] % factor or noisy_image.shape[-1] % factor:
            raise ValueError(
                f"Spatial dimensions {tuple(noisy_image.shape[-2:])} must be divisible by {factor}"
            )
        condition = F.interpolate(
            condition,
            size=tuple(noisy_image.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )

        time_embedding = self.time_embedding(timesteps)
        hidden = self.input_conv(torch.cat((noisy_image, condition), dim=1))
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
