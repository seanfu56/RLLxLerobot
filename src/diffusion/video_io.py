"""MP4 output shared by training previews and the sampling CLI."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def normalized_rgb(image: torch.Tensor) -> np.ndarray:
    """Convert normalized ``3 x H x W`` RGB to uint8 ``H x W x 3``."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected 3xHxW RGB image, got {tuple(image.shape)}")
    return (
        image.detach()
        .float()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def write_png(path: Path, image: torch.Tensor) -> None:
    """Write one normalized RGB tensor as PNG."""
    rgb = normalized_rgb(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(path), cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError(f"Could not write PNG: {path}")


def write_frame_strip(path: Path, video: torch.Tensor) -> None:
    """Save ``C x T x H x W`` frames as one left-to-right RGB PNG."""
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"Expected CxTxHxW RGB video, got {tuple(video.shape)}")
    frames = [normalized_rgb(video[:, index]) for index in range(video.shape[1])]
    strip_rgb = np.concatenate(frames, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(path),
        cv2.cvtColor(np.ascontiguousarray(strip_rgb), cv2.COLOR_RGB2BGR),
    ):
        raise RuntimeError(f"Could not write frame strip: {path}")


def verify_mp4(path: Path, expected_video: torch.Tensor) -> None:
    """Decode an MP4 and verify shape plus its first frame and RGB channel order."""
    expected_frames = expected_video.shape[1]
    expected_height, expected_width = expected_video.shape[-2:]
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen generated MP4: {path}")
    decoded: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            decoded.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(decoded) != expected_frames:
        raise RuntimeError(
            f"{path} contains {len(decoded)} decoded frames; expected {expected_frames}"
        )
    if decoded[0].shape[:2] != (expected_height, expected_width):
        raise RuntimeError(
            f"{path} decoded at {decoded[0].shape[:2]}; "
            f"expected {(expected_height, expected_width)}"
        )
    expected_first = normalized_rgb(expected_video[:, 0])
    first_frame_mae = float(
        np.abs(decoded[0].astype(np.float32) - expected_first.astype(np.float32)).mean()
    )
    # mp4v is lossy, but normal camera frames are well below 5. A swapped RGB/BGR
    # frame or a missing/noisy first frame is typically tens of levels away.
    if first_frame_mae > 15.0:
        raise RuntimeError(
            f"{path} first frame differs from its condition by MAE {first_frame_mae:.2f}; "
            "the condition may be missing or the RGB channel order may be wrong"
        )


def write_mp4(path: Path, video: torch.Tensor, fps: float) -> None:
    """Write and verify one normalized ``C x T x H x W`` RGB video."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"Expected CxTxHxW RGB video, got {tuple(video.shape)}")
    frames = np.stack([normalized_rgb(video[:, index]) for index in range(video.shape[1])])
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
    verify_mp4(path, video)
