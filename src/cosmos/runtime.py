"""Load a fine-tuned Cosmos3-Nano once and turn camera frames into video.

This is the direct, in-process inference API:

    from cosmos.runtime import CosmosRunner

    runner = CosmosRunner(adapter="models/cosmos3_nano_lora/pick-can-all/adapter")
    frames = runner.generate(frame_bgr)          # 93 RGB PIL frames at 256x256
    goal = runner.goal_frame(frame_bgr)          # just the last one

``sample.py`` (offline, dataset-driven) and ``serve.py`` (HTTP, robot-driven)
are both thin wrappers over this class.  Construction is the expensive part --
the 8B transformer takes tens of seconds to load and ~35 GB of VRAM -- so build
one runner and keep it alive rather than calling this per request.

Frames follow the repository's camera convention: a ``np.ndarray`` argument is
**BGR**, the layout OpenCV and ``policy.inference.PolicyRunner`` use.  It is
center-cropped square and resized exactly the way training preprocessed the
recordings.  A ``PIL.Image`` argument is taken as RGB and only resized.
Generated frames come back as RGB PIL images.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cosmos.data import NUM_FRAMES, RESOLUTION, check_geometry, preprocess_frame
else:
    from .data import NUM_FRAMES, RESOLUTION, check_geometry, preprocess_frame

LOGGER = logging.getLogger("cosmos3-runtime")

DEFAULT_MODEL = "nvidia/Cosmos3-Nano"
DEFAULT_ADAPTER = "models/cosmos3_nano_lora/pick-can-all/adapter"
DEFAULT_CAPTIONS = Path(__file__).resolve().parent / "captions.json"
DEFAULT_NEGATIVE = "blurry, distorted, low quality, static, jittery, warped geometry"
DEFAULT_FPS = 20.0
DEFAULT_STEPS = 35
DEFAULT_GUIDANCE = 6.0
DEFAULT_FLOW_SHIFT = 10.0
DEFAULT_TASK = "pick up can"


def default_prompt(captions_path: str | Path = DEFAULT_CAPTIONS, task=DEFAULT_TASK) -> str:
    """The caption to use when a caller does not supply one.

    Single-task bundles have exactly one caption, so there is an unambiguous
    default.  Anything else has to be named explicitly -- silently picking one
    of several tasks would be a subtle way to condition on the wrong thing.
    """
    captions = json.loads(Path(captions_path).expanduser().read_text(encoding="utf-8"))
    if len(captions) < 1:
        raise ValueError(
            f"{captions_path} holds {len(captions)} captions, so there is no unambiguous "
            f"default; pass prompt=... explicitly. Tasks: {sorted(captions)}"
        )
    elif len(captions) > 1 and task not in captions:
        raise ValueError(
            f"{captions_path} holds {len(captions)} captions, so there is no unambiguous "
            f"default; pass prompt=... explicitly. Tasks: {sorted(captions)}"
        )
    else:
        return captions[task] if task in captions else next(iter(captions.values()))


def prepare_condition_frame(frame: np.ndarray | Image.Image, resolution: int) -> Image.Image:
    """Normalize a conditioning frame to an RGB PIL image at ``resolution``."""
    if isinstance(frame, Image.Image):
        image = frame.convert("RGB")
        if image.size != (resolution, resolution):
            image = image.resize((resolution, resolution), Image.BICUBIC)
        return image
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 BGR frame, got shape {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 BGR frame, got dtype {array.dtype}")
    return Image.fromarray(preprocess_frame(array, resolution))


class CosmosRunner:
    """A loaded Cosmos3-Nano pipeline with the LoRA adapter merged in."""

    def __init__(
        self,
        *,
        pretrained: str = DEFAULT_MODEL,
        adapter: str | Path | None = DEFAULT_ADAPTER,
        device: str = "cuda",
        resolution: int = RESOLUTION,
        num_frames: int = NUM_FRAMES,
        fps: float = DEFAULT_FPS,
        steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        flow_shift: float = DEFAULT_FLOW_SHIFT,
        negative_prompt: str = DEFAULT_NEGATIVE,
        prompt: str | None = None,
        task: str = DEFAULT_TASK,
        dtype: torch.dtype = torch.bfloat16,
    ):
        check_geometry(resolution, num_frames)
        self.resolution = int(resolution)
        self.num_frames = int(num_frames)
        self.fps = float(fps)
        self.steps = int(steps)
        self.guidance_scale = float(guidance_scale)
        self.negative_prompt = negative_prompt
        self.prompt = prompt if prompt is not None else default_prompt()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pretrained = pretrained

        from diffusers import Cosmos3OmniPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        started = time.perf_counter()
        pipe = Cosmos3OmniPipeline.from_pretrained(
            pretrained, torch_dtype=dtype, enable_safety_checker=False
        )

        self.adapter = str(adapter) if adapter else ""
        if self.adapter:
            from peft import PeftModel

            adapter_dir = Path(self.adapter).expanduser()
            if not adapter_dir.is_dir():
                raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
            # Merging folds LoRA into the base weights, so every later request
            # costs the same as the base model instead of paying for the
            # adapter in every layer.
            pipe.transformer = PeftModel.from_pretrained(
                pipe.transformer, str(adapter_dir)
            ).merge_and_unload()
            LOGGER.info("Merged LoRA adapter from %s", adapter_dir)
        else:
            LOGGER.info("No adapter: running the base %s", pretrained)

        pipe.transformer.to(self.device).eval()
        pipe.vae.to(self.device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=flow_shift
        )
        self.pipe = pipe
        self.load_seconds = time.perf_counter() - started
        LOGGER.info("Loaded %s on %s in %.1fs", pretrained, self.device, self.load_seconds)

    def describe(self) -> dict:
        """Everything a client needs to know about what it is talking to."""
        return {
            "model": self.pretrained,
            "adapter": self.adapter or None,
            "device": str(self.device),
            "resolution": self.resolution,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "load_seconds": round(self.load_seconds, 2),
        }

    def generate(
        self,
        frame: np.ndarray | Image.Image,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        num_frames: int | None = None,
        fps: float | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """Image-to-video from one conditioning frame. Returns RGB PIL frames."""
        num_frames = self.num_frames if num_frames is None else int(num_frames)
        check_geometry(self.resolution, num_frames)
        image = prepare_condition_frame(frame, self.resolution)
        generator = (
            torch.Generator(device=self.device).manual_seed(int(seed)) if seed is not None else None
        )
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt if prompt is not None else self.prompt,
                negative_prompt=(
                    negative_prompt if negative_prompt is not None else self.negative_prompt
                ),
                image=image,
                num_frames=num_frames,
                height=self.resolution,
                width=self.resolution,
                fps=self.fps if fps is None else float(fps),
                num_inference_steps=self.steps if steps is None else int(steps),
                guidance_scale=(
                    self.guidance_scale if guidance_scale is None else float(guidance_scale)
                ),
                generator=generator,
            )
        return result.video

    def goal_frame(self, frame: np.ndarray | Image.Image, **kwargs) -> Image.Image:
        """The predicted end state, for a goal-conditioned policy.

        See ``policy/goal.py``: ``GoalPolicyRunner.set_goal`` wants one picture
        of how the episode should end, and that is the last generated frame.
        """
        return self.generate(frame, **kwargs)[-1]

    def warmup(self, steps: int = 2) -> float:
        """One throw-away generation so the first real request is not the slow one.

        Uses a blank frame and a short denoising schedule: the point is to
        trigger CUDA kernel autotuning and allocator growth, not to produce
        anything usable.
        """
        blank = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        started = time.perf_counter()
        self.generate(blank, steps=steps, seed=0)
        elapsed = time.perf_counter() - started
        LOGGER.info("Warmup (%d steps) took %.1fs", steps, elapsed)
        return elapsed
