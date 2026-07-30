"""Clip loading for Cosmos3-Nano post-training on the Piper demonstrations.

The policy bundle stores 224x224 arrays, but Cosmos3 is fine-tuned at 256x256,
so this module goes back to each episode's original 480x640 MP4 rather than
upsampling the bundle arrays.  Every source frame is center-cropped to a square
*before* resizing, matching ``src/diffusion/data.py``:

    640 pixels wide = 80 removed + 480 retained + 80 removed
    480 pixels high =                   480 retained

The crop/resize helpers are duplicated here rather than imported from
``src/diffusion`` on purpose: Cosmos3 needs a much newer ``diffusers`` and
``transformers`` than the ``piper`` environment carries, so the two packages are
expected to run under different interpreters (see ``README.md``).

Two clip modes are supported:

``window``
    A contiguous ``num_frames`` window at the recording's native 20 FPS.  The
    window start is redrawn every time an episode is requested during training,
    which is the only data augmentation used here.  Validation uses the
    centered window so the loss is comparable across steps.
``episode``
    The whole episode uniformly resampled to ``num_frames``.  Playback is
    retimed, so the reported FPS is scaled to keep the clip's real duration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# AutoencoderKLWan compresses 16x spatially and 4x temporally, and the
# transformer patchifies latents 2x2, so the pixel resolution must be a
# multiple of 16 and num_frames must satisfy (num_frames - 1) % 4 == 0.
SPATIAL_MULTIPLE = 16
TEMPORAL_MULTIPLE = 4

RESOLUTION = 256
NUM_FRAMES = 93
CLIP_MODES = ("window", "episode")


@dataclass(frozen=True)
class Episode:
    """One recorded demonstration, resolved back to its original MP4."""

    path: Path
    source: str
    name: str
    task: str
    frames: int
    fps: float

    @property
    def key(self) -> str:
        return f"{self.source}_{self.name}"


def center_square_crop(frame: np.ndarray) -> np.ndarray:
    """Return the largest centered square from an ``H x W x C`` frame."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 frame, got shape {frame.shape}")
    height, width = frame.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return frame[top : top + side, left : left + side]


def preprocess_frame(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """Center-crop a BGR source frame, resize it, and return RGB uint8."""
    square = center_square_crop(frame_bgr)
    resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def check_geometry(resolution: int, num_frames: int) -> None:
    """Reject shapes the Wan VAE and the Cosmos3 patchifier cannot represent."""
    if resolution % SPATIAL_MULTIPLE != 0:
        raise ValueError(
            f"resolution must be a multiple of {SPATIAL_MULTIPLE}, got {resolution}"
        )
    if (num_frames - 1) % TEMPORAL_MULTIPLE != 0:
        raise ValueError(
            f"num_frames must satisfy (num_frames - 1) % {TEMPORAL_MULTIPLE} == 0, "
            f"got {num_frames}"
        )
    if num_frames < 1 + TEMPORAL_MULTIPLE:
        raise ValueError(f"num_frames must be at least 5, got {num_frames}")


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


def probe_video(path: Path) -> tuple[int, float]:
    """Return ``(frame_count, fps)`` for one MP4."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frames < 1 or fps <= 0:
        raise ValueError(f"Invalid video metadata for {path}: frames={frames} fps={fps}")
    return frames, fps


def load_episodes(bundle_dir: str | Path) -> list[Episode]:
    """Expand a policy bundle into its episodes, resolved to raw MP4 paths."""
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    manifest_path = bundle_dir / "bundle.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Not a policy bundle (no bundle.json): {bundle_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    episodes: list[Episode] = []
    for relative_shard in manifest.get("shards", []):
        shard_dir = (bundle_dir.parent / relative_shard).resolve()
        shard_path = shard_dir / "shard.json"
        if not shard_path.is_file():
            raise FileNotFoundError(f"Bundle references missing shard metadata: {shard_path}")
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        source = str(shard["source"])
        raw_root = _resolve_raw_root(bundle_dir, shard, source)
        for entry in shard.get("episodes", []):
            name = str(entry["name"])
            path = raw_root / name / "video.mp4"
            if not path.is_file():
                raise FileNotFoundError(f"Episode video referenced by the bundle is missing: {path}")
            frames, fps = probe_video(path)
            episodes.append(
                Episode(
                    path=path.resolve(),
                    source=source,
                    name=name,
                    task=str(entry["task"]),
                    frames=frames,
                    fps=fps,
                )
            )
    if not episodes:
        raise ValueError(f"Bundle has no episodes: {manifest_path}")
    return episodes


def split_episodes(
    episodes: Sequence[Episode], val_episodes: int, seed: int
) -> tuple[list[Episode], list[Episode]]:
    """Hold out whole episodes so no clip can straddle the split."""
    if val_episodes < 0:
        raise ValueError(f"val_episodes must be non-negative, got {val_episodes}")
    if val_episodes == 0:
        return list(episodes), []
    if val_episodes >= len(episodes):
        raise ValueError(
            f"val_episodes={val_episodes} leaves no training episodes out of {len(episodes)}"
        )
    rng = np.random.default_rng(seed)
    held_out = {int(index) for index in rng.permutation(len(episodes))[:val_episodes]}
    train = [episode for index, episode in enumerate(episodes) if index not in held_out]
    val = [episode for index, episode in enumerate(episodes) if index in held_out]
    return train, val


def load_captions(path: str | Path) -> dict[str, str]:
    """Load the task-name to prompt mapping used for text conditioning."""
    path = Path(path).expanduser()
    captions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(captions, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in captions.items()
    ):
        raise ValueError(f"{path} must be a flat JSON object of task -> caption strings")
    return captions


def resolve_captions(
    episodes: Iterable[Episode], captions: dict[str, str], override: str | None = None
) -> dict[str, str]:
    """Map every task present in ``episodes`` to a caption, or fail loudly."""
    tasks = sorted({episode.task for episode in episodes})
    if override is not None:
        return {task: override for task in tasks}
    missing = [task for task in tasks if task not in captions]
    if missing:
        raise KeyError(
            f"No caption for task(s) {missing}; known tasks are {sorted(captions)}. "
            f"Add them to the captions file or pass --caption to override every task."
        )
    return {task: captions[task] for task in tasks}


def read_clip(path: Path, indices: Sequence[int], resolution: int) -> list[Image.Image]:
    """Decode the requested source frames as 256x256-style RGB PIL images.

    Frames are read in order and monotonically, so a contiguous window decodes
    sequentially instead of paying for one seek per frame.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames: list[Image.Image] = []
    try:
        position = -1
        cached: Image.Image | None = None
        for wanted in indices:
            if wanted == position and cached is not None:
                # Uniform resampling can land on the same source frame twice.
                frames.append(cached)
                continue
            if wanted != position + 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, wanted)
                position = wanted - 1
            while position < wanted:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to decode frame {wanted} of {path}")
                position += 1
            cached = Image.fromarray(preprocess_frame(frame, resolution))
            frames.append(cached)
    finally:
        capture.release()
    return frames


class PiperClipDataset(Dataset):
    """One item = one clip from one episode, as PIL frames plus its caption.

    Returns a plain dict (not tensors): ``Cosmos3OmniPipeline.video_processor``
    does the normalization, and the caption has to reach ``tokenize_prompt`` as
    a string.  Batch size is fixed at one clip, so ``collate_clip`` below just
    unwraps the single-element list.
    """

    def __init__(
        self,
        episodes: Sequence[Episode],
        captions: dict[str, str],
        *,
        num_frames: int = NUM_FRAMES,
        resolution: int = RESOLUTION,
        clip_mode: str = "window",
        random_window: bool = True,
        seed: int = 0,
    ):
        check_geometry(resolution, num_frames)
        if clip_mode not in CLIP_MODES:
            raise ValueError(f"clip_mode must be one of {CLIP_MODES}, got {clip_mode!r}")
        if not episodes:
            raise ValueError("PiperClipDataset needs at least one episode")

        self.episodes = list(episodes)
        self.captions = dict(captions)
        self.num_frames = int(num_frames)
        self.resolution = int(resolution)
        self.clip_mode = clip_mode
        self.random_window = bool(random_window)
        self.seed = int(seed)
        self._draws = 0

        missing = sorted({e.task for e in self.episodes} - set(self.captions))
        if missing:
            raise KeyError(f"No caption for task(s) {missing}")

        if clip_mode == "window":
            too_short = [
                f"{e.key} ({e.frames} frames)" for e in self.episodes if e.frames < self.num_frames
            ]
            if too_short:
                raise ValueError(
                    f"clip_mode='window' needs at least {self.num_frames} frames per episode; "
                    f"too short: {too_short}. Lower --num-frames or use --clip-mode episode."
                )

    def __len__(self) -> int:
        return len(self.episodes)

    def _rng(self, item: int) -> np.random.Generator:
        self._draws += 1
        worker_seed = torch.initial_seed() % (2**31)
        entropy = self.seed * 1_000_003 + worker_seed + item * 7919 + self._draws * 104_729
        return np.random.default_rng(entropy % (2**32))

    def clip_plan(self, item: int) -> tuple[list[int], float]:
        """Return the source frame indices for this clip and its playback FPS."""
        episode = self.episodes[item]
        if self.clip_mode == "window":
            span = episode.frames - self.num_frames
            if self.random_window and span > 0:
                start = int(self._rng(item).integers(0, span + 1))
            else:
                start = span // 2
            return list(range(start, start + self.num_frames)), episode.fps

        indices = np.linspace(0, episode.frames - 1, self.num_frames).round().astype(int)
        # The episode is retimed onto num_frames, so report the FPS that keeps
        # the clip's wall-clock duration equal to the original recording's.
        fps = self.num_frames * episode.fps / episode.frames
        return indices.tolist(), float(fps)

    def __getitem__(self, item: int) -> dict:
        episode = self.episodes[item]
        indices, fps = self.clip_plan(item)
        frames = read_clip(episode.path, indices, self.resolution)
        return {
            "frames": frames,
            "caption": self.captions[episode.task],
            "fps": fps,
            "episode": episode.key,
            "task": episode.task,
        }


def collate_clip(batch: list[dict]) -> dict:
    """Unwrap the single-clip batch (Cosmos3OmniTransformer has no batch axis)."""
    if len(batch) != 1:
        raise ValueError(f"Expected batch_size=1, got {len(batch)}")
    return batch[0]


def video_worker_init(_: int) -> None:
    """Keep each DataLoader worker from starting its own OpenCV thread pool."""
    cv2.setNumThreads(0)
