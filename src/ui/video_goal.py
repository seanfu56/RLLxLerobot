"""Generating the goal a goal-conditioned policy runs against.

A behaviour-cloning rollout is one step: look, act. A goal-conditioned rollout
is two: predict where this episode should go, then act towards it. This module
is the first step. Given the frame in front of the camera right now, it asks a
video model what should happen next and returns the answer as a short strip of
frames, one of which becomes the goal.

Two sources, because the repository has two video models:

``cosmos``
    Cosmos3-Nano over HTTP (``src/cosmos/serve.py``). The model cannot share an
    environment with the arm - it needs a diffusers stack ``lerobot`` declares
    itself incompatible with - so it lives behind a socket and this only needs
    ``src/cosmos/client.py``, which imports nothing but the standard library,
    numpy and OpenCV.
``diffusion``
    The self-trained cascade in ``src/diffusion``, loaded in process. It is
    small enough to sit alongside the policy on one GPU, and it produces exactly
    the four frames the policy was trained against.

**Which frame is the goal.** The third of four sampled evenly across the
generated video, which is the same rule ``src/policy/goal.py`` trains under -
the two share ``uniform_frame_offsets`` so they cannot drift apart. For the
four-frame cascade that is simply frame 2. For a 93-frame Cosmos clip it is
frame 61, two thirds of the way through. The operator can override it in the UI;
this is the default, not a rule.

Frames cross every boundary here as PNG bytes. The runtime lives in a separate
process from Streamlit, so whatever comes back has to pickle cheaply, and PNG is
also what an ``st.image`` and an ``<img>`` tag both accept without conversion.

**Keeping the video.** ``GoalVideo.save`` writes one to disk. A generated video
is the only record of what the policy was told to do - a rollout that fails
because the goal was wrong and one that fails because the policy is wrong look
identical in the episode's video - and it is not reproducible from the
checkpoint, since the sampler runs unseeded unless the operator fixes it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

VIDEO_SOURCES = ("cosmos", "diffusion")
DEFAULT_COSMOS_URL = "http://127.0.0.1:8501"
DEFAULT_BASE_CHECKPOINT = Path("models/video_diffusion/pick-can-all/base/latest.pt")
DEFAULT_SUPERRES_CHECKPOINT = Path("models/video_diffusion/pick-can-all-sr/latest.pt")

__all__ = [
    "DEFAULT_BASE_CHECKPOINT",
    "DEFAULT_COSMOS_URL",
    "DEFAULT_SUPERRES_CHECKPOINT",
    "VIDEO_SOURCES",
    "GoalVideo",
    "VideoSourceConfig",
    "default_goal_index",
    "encode_png",
    "generate_goal_video",
]


@dataclass(frozen=True)
class VideoSourceConfig:
    """Which video model to ask, and how. Picklable: it crosses a process pipe."""

    kind: str = "cosmos"

    # --- cosmos, over HTTP ---
    url: str = DEFAULT_COSMOS_URL
    prompt: str | None = None
    num_frames: int | None = None
    steps: int | None = None
    guidance_scale: float | None = None
    timeout_s: float = 300.0

    # --- the self-trained cascade, in process ---
    base_checkpoint: Path = DEFAULT_BASE_CHECKPOINT
    superres_checkpoint: Path = DEFAULT_SUPERRES_CHECKPOINT
    inference_steps: int = 50
    device: str = "cuda"

    # --- shared ---
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in VIDEO_SOURCES:
            raise ValueError(f"kind must be one of {VIDEO_SOURCES}, got {self.kind!r}")
        object.__setattr__(self, "base_checkpoint", Path(self.base_checkpoint).expanduser())
        object.__setattr__(self, "superres_checkpoint", Path(self.superres_checkpoint).expanduser())
        if self.kind == "diffusion":
            for label, path in (
                ("base", self.base_checkpoint),
                ("super-resolution", self.superres_checkpoint),
            ):
                if not path.is_file():
                    raise ValueError(f"The {label} checkpoint was not found: {path}")
            if self.inference_steps < 1:
                raise ValueError("inference_steps must be at least 1")
        elif not self.url:
            raise ValueError("A Cosmos3 server URL is required")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

    def describe(self) -> str:
        if self.kind == "cosmos":
            return f"Cosmos3-Nano at {self.url}"
        return f"src/diffusion cascade ({self.base_checkpoint.parent.parent.name})"


@dataclass(frozen=True)
class GoalVideo:
    """What the video model predicted, plus which frame of it is the goal."""

    frames_png: tuple[bytes, ...]
    goal_index: int
    source: str
    seconds: float
    fps: float

    def __post_init__(self) -> None:
        if not self.frames_png:
            raise ValueError("A goal video needs at least one frame")
        if not 0 <= self.goal_index < len(self.frames_png):
            raise ValueError(
                f"goal_index must be in [0, {len(self.frames_png)}), got {self.goal_index}"
            )

    @property
    def frame_count(self) -> int:
        return len(self.frames_png)

    def frame(self, index: int) -> np.ndarray:
        """One frame as the BGR array ``GoalPolicyRunner.set_goal`` expects."""
        if not 0 <= index < self.frame_count:
            raise IndexError(f"Frame {index} is outside [0, {self.frame_count})")
        raw = np.frombuffer(self.frames_png[index], dtype=np.uint8)
        decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"Goal video frame {index} could not be decoded")
        return decoded

    def save(
        self,
        directory: str | Path,
        *,
        chosen_index: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write the video to ``directory``: frames, a playable clip, and meta.json.

        The frames are written as the PNG bytes that were generated, not
        re-encoded, so what lands on disk is bit-for-bit the picture the policy
        was aimed at. The clip beside them is for watching, and is allowed to be
        missing: ``meta.json`` names it, or records null when this OpenCV build
        could not write one.

        ``chosen_index`` is the frame the operator actually aimed at, which is
        not always the default one, and is copied out to ``goal.png`` so the
        goal of an episode can be seen without knowing the sampling rule.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for index, payload in enumerate(self.frames_png):
            (directory / f"frame_{index:03d}.png").write_bytes(payload)
        if chosen_index is not None:
            if not 0 <= chosen_index < self.frame_count:
                raise IndexError(
                    f"chosen_index {chosen_index} is outside [0, {self.frame_count})"
                )
            (directory / "goal.png").write_bytes(self.frames_png[chosen_index])

        meta = {
            "source": self.source,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "generation_seconds": round(self.seconds, 3),
            "default_goal_index": self.goal_index,
            "goal_index": self.goal_index if chosen_index is None else int(chosen_index),
            "clip": _write_clip(directory, self, self.fps),
            "frames": "frame_NNN.png, in generated order; frame 0 is the camera frame "
            "the model was conditioned on",
            **(extra or {}),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return directory


def default_goal_index(frame_count: int) -> int:
    """The third of four frames sampled evenly - the rule training used.

    Imported from ``policy.goal`` rather than restated, because a policy aiming
    at one frame of the episode while the video model hands it another is the
    one failure mode of this pipeline that produces no error at all.
    """
    try:
        from policy.goal import DEFAULT_GOAL_FRAME_INDEX, DEFAULT_GOAL_FRAMES, uniform_frame_offsets
    except ModuleNotFoundError:  # running with src/ui itself on sys.path
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from policy.goal import DEFAULT_GOAL_FRAME_INDEX, DEFAULT_GOAL_FRAMES, uniform_frame_offsets

    if frame_count < 1:
        raise ValueError(f"frame_count must be >= 1, got {frame_count}")
    if frame_count < DEFAULT_GOAL_FRAMES:
        return frame_count - 1
    return uniform_frame_offsets(frame_count - 1, DEFAULT_GOAL_FRAMES)[DEFAULT_GOAL_FRAME_INDEX]


def encode_png(frame_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed to encode a PNG")
    return buffer.tobytes()


def _write_clip(directory: Path, video: GoalVideo, fps: float) -> str | None:
    """A watchable version of the frames next to them; None if none could be written.

    Best effort on purpose. The PNGs are the archive - they are lossless and
    already written by the time this runs - and an OpenCV build without the mp4
    encoder is not a reason to fail a save that has otherwise succeeded. MJPG in
    an AVI container is the fallback because it is the encoder the recording
    workers in ``teleop_runtime`` already rely on.
    """
    frames = [video.frame(index) for index in range(video.frame_count)]
    height, width = frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        return None  # a mixed-size clip would be silently truncated by the writer
    for name, fourcc in (("video.mp4", "mp4v"), ("video.avi", "MJPG")):
        path = directory / name
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), max(float(fps), 1.0), (width, height)
        )
        if not writer.isOpened():
            writer.release()
            path.unlink(missing_ok=True)
            continue
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        return name
    return None


# ---------------------------------------------------------------------------
# the self-trained cascade
# ---------------------------------------------------------------------------


class _DiffusionCascade:
    """Both stages of ``src/diffusion``, loaded once and kept.

    Generating a goal takes a couple of seconds; loading the two checkpoints
    takes longer than that, and an operator generating a goal per episode would
    otherwise pay it every time.
    """

    def __init__(self, config: VideoSourceConfig):
        import torch

        from diffusion.data import GENERATED_FRAMES, HIGH_RESOLUTION, LOW_RESOLUTION
        from diffusion.sample import condition_tensor, load_stage

        self.key = (config.base_checkpoint, config.superres_checkpoint, config.device)
        self.device = torch.device(
            config.device if torch.cuda.is_available() or not config.device.startswith("cuda") else "cpu"
        )
        self.base, self.base_diffusion, _ = load_stage(
            config.base_checkpoint, "base", self.device, True
        )
        self.superres, self.superres_diffusion, _ = load_stage(
            config.superres_checkpoint, "superres", self.device, True
        )
        self._torch = torch
        self._condition_tensor = condition_tensor
        self._low = LOW_RESOLUTION
        self._high = HIGH_RESOLUTION
        self._generated = GENERATED_FRAMES

    def generate(self, frame_bgr: np.ndarray, config: VideoSourceConfig) -> list[np.ndarray]:
        """Condition frame plus three generated frames, as 224px BGR arrays."""
        torch = self._torch
        from diffusion.video_io import normalized_rgb

        generator = (
            torch.Generator(device=self.device).manual_seed(int(config.seed))
            if config.seed is not None
            else None
        )
        low_condition = self._condition_tensor(frame_bgr, self._low, self.device)
        high_condition = self._condition_tensor(frame_bgr, self._high, self.device)
        with torch.no_grad():
            low = self.base_diffusion.sample(
                self.base,
                (1, 3, self._generated, self._low, self._low),
                condition=low_condition,
                inference_steps=config.inference_steps,
                generator=generator,
            )
            low_images = (
                low.permute(0, 2, 1, 3, 4)
                .reshape(self._generated, 3, self._low, self._low)
                .contiguous()
            )
            high_images = self.superres_diffusion.sample(
                self.superres,
                (self._generated, 3, self._high, self._high),
                condition=low_images,
                inference_steps=config.inference_steps,
                generator=generator,
            )
            video = torch.cat(
                (high_condition, high_images.permute(1, 0, 2, 3).unsqueeze(0)), dim=2
            )[0].float().cpu()
        return [
            cv2.cvtColor(normalized_rgb(video[:, index]), cv2.COLOR_RGB2BGR)
            for index in range(video.shape[1])
        ]


_CASCADE: _DiffusionCascade | None = None


def _cascade(config: VideoSourceConfig) -> _DiffusionCascade:
    global _CASCADE
    key = (config.base_checkpoint, config.superres_checkpoint, config.device)
    if _CASCADE is None or _CASCADE.key != key:
        _CASCADE = _DiffusionCascade(config)
    return _CASCADE


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def generate_goal_video(config: VideoSourceConfig, frame_bgr: Any) -> GoalVideo:
    """Ask the configured video model what should happen from this frame."""
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 BGR camera frame, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    started = time.perf_counter()
    if config.kind == "cosmos":
        # src/cosmos/client.py imports nothing but the standard library, numpy
        # and OpenCV, so it is safe to reach for from the robot environment.
        try:
            from cosmos.client import CosmosClient
        except ModuleNotFoundError:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from cosmos.client import CosmosClient

        client = CosmosClient(config.url, config.timeout_s)
        options = {
            "prompt": config.prompt,
            "num_frames": config.num_frames,
            "steps": config.steps,
            "guidance_scale": config.guidance_scale,
            "seed": config.seed,
        }
        frames = client.generate(frame, **{k: v for k, v in options.items() if v is not None})
    else:
        frames = _cascade(config).generate(frame, config)

    elapsed = time.perf_counter() - started
    if not frames:
        raise RuntimeError(f"{config.describe()} returned no frames")
    return GoalVideo(
        frames_png=tuple(encode_png(item) for item in frames),
        goal_index=default_goal_index(len(frames)),
        source=config.describe(),
        seconds=elapsed,
        # Cosmos generates a real-time clip at the recording rate; the cascade
        # generates four sparse trajectory states, which src/diffusion plays at
        # one a second so each is actually visible.
        fps=20.0 if config.kind == "cosmos" else 1.0,
    )
