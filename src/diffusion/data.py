"""Raw-video loading for the two-stage video diffusion models.

The policy bundle contains 224x224 decoded arrays, but diffusion training goes
back to each episode's original MP4.  Every source frame is center-cropped to a
square *before* resizing.  Thus a 480x640 frame loses 80 pixels from both its
left and right edges and becomes 480x480 before the 56 or 224 resize.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

LOW_RESOLUTION = 56
HIGH_RESOLUTION = 224
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frames: int
    width: int
    height: int
    fps: float


def center_square_crop(frame: np.ndarray) -> np.ndarray:
    """Return the largest centered square from an ``H x W x C`` frame."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 frame, got shape {frame.shape}")
    height, width = frame.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return frame[top : top + side, left : left + side]


def _resolve_raw_root(bundle_dir: Path, shard_meta: dict, source: str) -> Path:
    """Resolve a shard's raw directory even if its stored absolute path moved."""
    recorded = Path(str(shard_meta.get("raw_root", ""))).expanduser()
    candidates = [
        recorded,
        bundle_dir.parent.parent / "raw" / source,
        bundle_dir.parent / "raw" / source,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    choices = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Raw video source {source!r} was not found; checked {choices}")


def _paths_from_bundle(bundle_dir: Path) -> list[Path]:
    manifest_path = bundle_dir / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    videos: list[Path] = []
    for relative_shard in manifest.get("shards", []):
        shard_dir = (bundle_dir.parent / relative_shard).resolve()
        shard_path = shard_dir / "shard.json"
        if not shard_path.is_file():
            raise FileNotFoundError(f"Bundle references missing shard metadata: {shard_path}")
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        source = str(shard["source"])
        raw_root = _resolve_raw_root(bundle_dir, shard, source)
        for episode in shard.get("episodes", []):
            path = raw_root / str(episode["name"]) / "video.mp4"
            if not path.is_file():
                raise FileNotFoundError(f"Episode video referenced by the bundle is missing: {path}")
            videos.append(path.resolve())
    if not videos:
        raise ValueError(f"Bundle has no episode videos: {manifest_path}")
    return videos


def resolve_video_paths(location: str | Path) -> list[Path]:
    """Resolve an MP4, raw episode tree, or policy bundle to ordered videos."""
    location = Path(location).expanduser()
    if location.is_file():
        if location.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"Not a supported video file: {location}")
        return [location.resolve()]
    if not location.is_dir():
        raise FileNotFoundError(f"Dataset location does not exist: {location}")
    if (location / "bundle.json").is_file():
        return _paths_from_bundle(location.resolve())
    videos = sorted(
        path.resolve()
        for path in location.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise FileNotFoundError(f"No videos found below {location}")
    return videos


def probe_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        info = VideoInfo(
            path=path,
            frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        capture.release()
    if info.frames < 1 or info.width < 1 or info.height < 1:
        raise ValueError(f"Invalid video metadata for {path}: {info}")
    return info


def split_video_paths(
    paths: Sequence[Path], val_videos: int, seed: int
) -> tuple[list[Path], list[Path]]:
    """Hold out whole episodes, preventing overlapping clips across the split."""
    if val_videos < 0:
        raise ValueError(f"val_videos must be non-negative, got {val_videos}")
    if val_videos == 0:
        return list(paths), []
    if val_videos >= len(paths):
        raise ValueError(
            f"val_videos={val_videos} leaves no training videos out of {len(paths)}"
        )
    rng = np.random.default_rng(seed)
    held_out = set(int(index) for index in rng.permutation(len(paths))[:val_videos])
    train = [path for index, path in enumerate(paths) if index not in held_out]
    val = [path for index, path in enumerate(paths) if index in held_out]
    return train, val


class VideoClipDataset(Dataset):
    """Decode fixed-length clips and return normalized ``C x T x H x W`` tensors.

    ``resolutions`` controls which pyramid levels are returned.  Keys are named
    ``video_56`` and ``video_224``.  The original MP4 is decoded once per item;
    all requested sizes come from the same center-cropped source frames.
    """

    def __init__(
        self,
        videos: Iterable[str | Path],
        *,
        clip_length: int = 16,
        frame_stride: int = 1,
        clip_step: int = 4,
        resolutions: Sequence[int] = (LOW_RESOLUTION,),
    ):
        if clip_length < 1:
            raise ValueError(f"clip_length must be positive, got {clip_length}")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be positive, got {frame_stride}")
        if clip_step < 1:
            raise ValueError(f"clip_step must be positive, got {clip_step}")
        resolutions = tuple(dict.fromkeys(int(size) for size in resolutions))
        if not resolutions or any(size < 1 for size in resolutions):
            raise ValueError(f"resolutions must contain positive sizes, got {resolutions}")

        self.clip_length = int(clip_length)
        self.frame_stride = int(frame_stride)
        self.clip_step = int(clip_step)
        self.resolutions = resolutions
        self.span = (self.clip_length - 1) * self.frame_stride + 1
        self.videos = [probe_video(Path(path).expanduser().resolve()) for path in videos]
        fps_values = {round(info.fps, 3) for info in self.videos}
        if len(fps_values) != 1:
            raise ValueError(f"Videos mix frame rates: {sorted(fps_values)}")
        self.fps = self.videos[0].fps
        self.index: list[tuple[int, int]] = []
        for video_index, info in enumerate(self.videos):
            if info.frames < self.span:
                continue
            final_start = info.frames - self.span
            starts = list(range(0, final_start + 1, self.clip_step))
            if starts[-1] != final_start:
                starts.append(final_start)
            self.index.extend((video_index, start) for start in starts)
        if not self.index:
            raise ValueError(
                f"No video is long enough for clip_length={clip_length}, "
                f"frame_stride={frame_stride} (span {self.span} source frames)"
            )

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _resize_rgb(frame_bgr: np.ndarray, size: int) -> np.ndarray:
        square = center_square_crop(frame_bgr)
        resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        video_index, start = self.index[item]
        info = self.videos[video_index]
        capture = cv2.VideoCapture(str(info.path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {info.path}")
        frames: dict[int, list[np.ndarray]] = {size: [] for size in self.resolutions}
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            last = start + self.span
            for source_index in range(start, last):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to decode frame {source_index} of {info.path}")
                if (source_index - start) % self.frame_stride:
                    continue
                for size in self.resolutions:
                    frames[size].append(self._resize_rgb(frame, size))
        finally:
            capture.release()

        result: dict[str, torch.Tensor] = {}
        for size, values in frames.items():
            if len(values) != self.clip_length:
                raise RuntimeError(
                    f"Decoded {len(values)} rather than {self.clip_length} frames from {info.path}"
                )
            array = np.stack(values).copy()
            tensor = torch.from_numpy(array).permute(3, 0, 1, 2).float()
            result[f"video_{size}"] = tensor.div_(127.5).sub_(1.0)
        return result


def video_worker_init(_: int) -> None:
    """Keep each DataLoader worker from starting its own OpenCV thread pool."""
    cv2.setNumThreads(0)
