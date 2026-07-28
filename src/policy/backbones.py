"""Image normalisation and the spatial-softmax bottleneck.

The vision trunk itself (ImageNet ResNet-18 with its BatchNorms swapped for
GroupNorms) is assembled in ``model.RgbEncoder``; what lives here are the two
pieces around it that are worth keeping separate and testable.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNormalizer(nn.Module):
    """uint8 [B, 3, H, W] -> ImageNet-normalised float."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, images: Tensor) -> Tensor:
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        return (images - self.mean) / self.std


def replace_batchnorm_with_groupnorm(module: nn.Module, groups: int = 16) -> nn.Module:
    """Swap every BatchNorm2d for a GroupNorm, in place.

    BatchNorm's running statistics are unreliable at these batch sizes and are
    not tracked by the EMA, so a checkpoint's averaged weights would be paired
    with un-averaged statistics.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features
            setattr(module, name, nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels))
        else:
            replace_batchnorm_with_groupnorm(child, groups)
    return module


class SpatialSoftmax(nn.Module):
    """Expected 2D keypoint locations over a feature map.

    Unlike average pooling this keeps *where* a feature fired, which is exactly
    the signal a "move down until the gripper touches the table" policy needs.
    Returns [B, num_keypoints * 2] normalised coordinates in [-1, 1].
    """

    def __init__(self, in_channels: int, num_keypoints: int = 32, temperature: float = 1.0):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.temperature = temperature
        self.project = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
        self.output_dim = num_keypoints * 2

    def forward(self, features: Tensor) -> Tensor:
        batch, _, height, width = features.shape
        heatmap = self.project(features).flatten(2)  # (B, K, H*W)
        attention = torch.softmax(heatmap / self.temperature, dim=-1)

        device, dtype = features.device, attention.dtype
        rows = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        cols = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (H*W, 2)

        keypoints = attention @ coords  # (B, K, 2)
        return keypoints.reshape(batch, self.num_keypoints * 2)
