"""MP4 output shared by training previews and the sampling CLI."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def write_mp4(path: Path, video: torch.Tensor, fps: float) -> None:
    """Write one normalized ``C x T x H x W`` RGB video."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"Expected CxTxHxW RGB video, got {tuple(video.shape)}")
    frames = (
        video.detach()
        .float()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .round()
        .byte()
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )
    height, width = frames.shape[1:3]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {path}")
    try:
        for frame_rgb in frames:
            writer.write(cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
