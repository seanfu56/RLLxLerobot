"""Raw-video loading for the two-stage video diffusion models.

The policy bundle contains 224x224 decoded arrays, but diffusion training goes
back to each episode's original MP4.  Every source frame is center-cropped to a
square *before* resizing.  Thus a 480x640 frame loses 80 pixels from both its
left and right edges and becomes 480x480 before the 56 or 224 resize.

Each episode supplies exactly four trajectory frames.  Frame zero is always the
condition.  The final frame is drawn from the last ten frames of the episode,
and the two remaining frames are placed as evenly as possible between them.
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
SAMPLED_FRAMES = 4
GENERATED_FRAMES = SAMPLED_FRAMES - 1
TAIL_FRAMES = 10
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


def four_frame_indices(frame_count: int, last_index: int) -> tuple[int, int, int, int]:
    """Return frame zero, two evenly spaced interior frames, and ``last_index``."""
    if frame_count < 4:
        raise ValueError(f"At least four source frames are required, got {frame_count}")
    if not 3 <= last_index < frame_count:
        raise ValueError(
            f"last_index must be in [3, {frame_count}), got {last_index}"
        )
    # Integer rounding to the nearest one-third and two-thirds positions.
    first_middle = (last_index + 1) // 3
    second_middle = (2 * last_index + 1) // 3
    return 0, first_middle, second_middle, last_index


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
    """Hold out whole episodes so trajectory samples cannot cross the split."""
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
    """Sample four trajectory frames and return normalized ``C x 4 x H x W`` tensors.

    ``resolutions`` controls which pyramid levels are returned.  Keys are named
    ``video_56`` and ``video_224``.  The original MP4 is decoded once per item;
    all requested sizes come from the same center-cropped source frames.

    Training redraws the final index from the last ``tail_frames`` every time an
    episode is requested. Validation should set ``random_tail=False`` so the
    episode's actual final frame is used every time.
    """

    def __init__(
        self,
        videos: Iterable[str | Path],
        *,
        resolutions: Sequence[int] = (LOW_RESOLUTION,),
        tail_frames: int = TAIL_FRAMES,
        random_tail: bool = True,
        seed: int = 0,
    ):
        if tail_frames < 1:
            raise ValueError(f"tail_frames must be positive, got {tail_frames}")
        resolutions = tuple(dict.fromkeys(int(size) for size in resolutions))
        if not resolutions or any(size < 1 for size in resolutions):
            raise ValueError(f"resolutions must contain positive sizes, got {resolutions}")

        self.resolutions = resolutions
        self.tail_frames = int(tail_frames)
        self.random_tail = bool(random_tail)
        self.seed = int(seed)
        self._draws = 0
        self.videos = [probe_video(Path(path).expanduser().resolve()) for path in videos]
        fps_values = {round(info.fps, 3) for info in self.videos}
        if len(fps_values) != 1:
            raise ValueError(f"Videos mix frame rates: {sorted(fps_values)}")
        self.fps = self.videos[0].fps
        too_short = [str(info.path) for info in self.videos if info.frames < 4]
        if too_short:
            raise ValueError(f"Videos with fewer than four frames: {too_short}")

    def __len__(self) -> int:
        return len(self.videos)

    def _rng(self, item: int) -> np.random.Generator:
        self._draws += 1
        worker_seed = torch.initial_seed() % (2**31)
        entropy = self.seed * 1_000_003 + worker_seed + item * 7919 + self._draws * 104_729
        return np.random.default_rng(entropy % (2**32))

    def sample_indices(self, item: int) -> tuple[int, int, int, int]:
        info = self.videos[item]
        if self.random_tail:
            first_tail_index = max(3, info.frames - self.tail_frames)
            last_index = int(self._rng(item).integers(first_tail_index, info.frames))
        else:
            last_index = info.frames - 1
        return four_frame_indices(info.frames, last_index)

    @staticmethod
    def _resize_rgb(frame_bgr: np.ndarray, size: int) -> np.ndarray:
        return preprocess_frame(frame_bgr, size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        info = self.videos[item]
        indices = self.sample_indices(item)
        capture = cv2.VideoCapture(str(info.path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {info.path}")
        frames: dict[int, list[np.ndarray]] = {size: [] for size in self.resolutions}
        try:
            for source_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to decode frame {source_index} of {info.path}")
                for size in self.resolutions:
                    frames[size].append(self._resize_rgb(frame, size))
        finally:
            capture.release()

        result: dict[str, torch.Tensor] = {}
        for size, values in frames.items():
            if len(values) != SAMPLED_FRAMES:
                raise RuntimeError(
                    f"Decoded {len(values)} rather than {SAMPLED_FRAMES} frames from {info.path}"
                )
            array = np.stack(values).copy()
            tensor = torch.from_numpy(array).permute(3, 0, 1, 2).float()
            result[f"video_{size}"] = tensor.div_(127.5).sub_(1.0)
        result["frame_indices"] = torch.tensor(indices, dtype=torch.long)
        return result


def preprocess_frame(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """Center-crop a BGR source frame, resize it, and return RGB uint8."""
    square = center_square_crop(frame_bgr)
    resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def video_worker_init(_: int) -> None:
    """Keep each DataLoader worker from starting its own OpenCV thread pool."""
    cv2.setNumThreads(0)
