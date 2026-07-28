#!/usr/bin/env python3
"""Generate three future frames from a fixed first frame, then super-resolve them."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from diffusion.data import (
        GENERATED_FRAMES,
        HIGH_RESOLUTION,
        LOW_RESOLUTION,
        SAMPLED_FRAMES,
        TAIL_FRAMES,
        VIDEO_SUFFIXES,
        preprocess_frame,
    )
    from diffusion.diffusion import GaussianDiffusion
    from diffusion.image_model import ImageUNet, ImageUNetConfig
    from diffusion.model import VideoUNet, VideoUNetConfig
    from diffusion.video_io import write_mp4
else:
    from .data import (
        GENERATED_FRAMES,
        HIGH_RESOLUTION,
        LOW_RESOLUTION,
        SAMPLED_FRAMES,
        TAIL_FRAMES,
        VIDEO_SUFFIXES,
        preprocess_frame,
    )
    from .diffusion import GaussianDiffusion
    from .image_model import ImageUNet, ImageUNetConfig
    from .model import VideoUNet, VideoUNetConfig
    from .video_io import write_mp4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--superres-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--condition",
        type=Path,
        required=True,
        help="Image, MP4, or episode directory whose first frame is fixed",
    )
    parser.add_argument("--output", type=Path, default=Path("samples/generated.mp4"))
    parser.add_argument("--num-videos", type=int, default=1)
    parser.add_argument("--base-inference-steps", type=int, default=50)
    parser.add_argument("--superres-inference-steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    parser.add_argument("--raw-weights", action="store_true", help="Do not use EMA weights")
    parser.add_argument("--save-low-resolution", action="store_true")
    return parser.parse_args(argv)


def load_stage(
    checkpoint_path: Path,
    expected_stage: str,
    device: torch.device,
    use_ema: bool,
) -> tuple[nn.Module, GaussianDiffusion, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("stage") != expected_stage:
        raise ValueError(
            f"{checkpoint_path} is stage {checkpoint.get('stage')!r}, "
            f"expected {expected_stage!r}"
        )
    expected_model_type = (
        "video_unet_3d" if expected_stage == "base" else "image_unet_2d"
    )
    if checkpoint.get("model_type") != expected_model_type:
        raise ValueError(
            f"{checkpoint_path} uses model type {checkpoint.get('model_type')!r}; "
            f"expected {expected_model_type!r}"
        )
    if expected_stage == "base":
        model = VideoUNet(VideoUNetConfig.from_dict(checkpoint["model_config"]))
    else:
        model = ImageUNet(ImageUNetConfig.from_dict(checkpoint["model_config"]))
    model = model.to(device)
    weights = checkpoint.get("ema") if use_ema else checkpoint.get("model")
    if weights is None:
        raise ValueError(f"{checkpoint_path} does not contain the requested model weights")
    model.load_state_dict(weights)
    model.eval()
    diffusion = GaussianDiffusion(**checkpoint["diffusion_config"]).to(device)
    return model, diffusion, checkpoint


def output_paths(path: Path, count: int, suffix: str = "") -> list[Path]:
    if path.suffix.lower() != ".mp4":
        path = path / "generated.mp4"
    if count == 1:
        return [path.with_name(path.stem + suffix + path.suffix)]
    return [
        path.with_name(f"{path.stem}_{index:03d}{suffix}{path.suffix}")
        for index in range(count)
    ]


def read_condition_frame(path: Path) -> np.ndarray:
    """Read a BGR image or the first BGR frame of a video/episode directory."""
    path = path.expanduser()
    if path.is_dir():
        path = path / "video.mp4"
    if not path.is_file():
        raise FileNotFoundError(f"Condition image or video does not exist: {path}")
    if path.suffix.lower() in VIDEO_SUFFIXES:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open condition video: {path}")
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise RuntimeError(f"Could not decode the first frame of {path}")
        return frame
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not decode condition image: {path}")
    return frame


def condition_tensor(frame_bgr: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    rgb = preprocess_frame(frame_bgr, size)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
    return tensor.div_(127.5).sub_(1.0).to(device)[None, :, None]


def amp_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_videos < 1:
        raise SystemExit("--num-videos must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable; use --device cpu for a smoke test")

    base_model, base_diffusion, base_checkpoint = load_stage(
        args.base_checkpoint, "base", device, not args.raw_weights
    )
    superres_model, superres_diffusion, superres_checkpoint = load_stage(
        args.superres_checkpoint, "superres", device, not args.raw_weights
    )
    base_data = base_checkpoint["data_config"]
    superres_data = superres_checkpoint["data_config"]
    if int(base_data["low_resolution"]) != LOW_RESOLUTION:
        raise ValueError(f"Base checkpoint is not a {LOW_RESOLUTION}px model")
    if (
        int(superres_data["low_resolution"]) != LOW_RESOLUTION
        or int(superres_data["high_resolution"]) != HIGH_RESOLUTION
    ):
        raise ValueError(
            f"Super-resolution checkpoint is not {LOW_RESOLUTION}->{HIGH_RESOLUTION}"
        )
    expected_temporal_config = {
        "sampled_frames": SAMPLED_FRAMES,
        "generated_frames": GENERATED_FRAMES,
        "tail_frames": TAIL_FRAMES,
    }
    for label, data in (("base", base_data), ("super-resolution", superres_data)):
        for key, expected in expected_temporal_config.items():
            if int(data.get(key, -1)) != expected:
                raise ValueError(
                    f"{label} checkpoint has {key}={data.get(key)!r}; expected {expected}"
                )
    if base_model.config.condition_channels != 3:
        raise ValueError("Base checkpoint must condition on the fixed RGB first frame")
    if superres_model.config.condition_channels != 3:
        raise ValueError(
            "Image super-resolution checkpoint must condition on one low-resolution RGB image"
        )

    condition_frame = read_condition_frame(args.condition)
    first_low = condition_tensor(condition_frame, LOW_RESOLUTION, device).expand(
        args.num_videos, -1, -1, -1, -1
    )
    first_high = condition_tensor(condition_frame, HIGH_RESOLUTION, device).expand(
        args.num_videos, -1, -1, -1, -1
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)

    def progress(label: str):
        def report(done: int, total: int, _: torch.Tensor) -> None:
            if done == 1 or done == total or done % max(1, total // 10) == 0:
                print(f"{label}: {done}/{total}", flush=True)

        return report

    with amp_context(device, args.amp):
        low = base_diffusion.sample(
            base_model,
            (
                args.num_videos,
                3,
                GENERATED_FRAMES,
                LOW_RESOLUTION,
                LOW_RESOLUTION,
            ),
            condition=first_low,
            inference_steps=args.base_inference_steps,
            eta=args.eta,
            generator=generator,
            callback=progress("base 56x56"),
        )
        low_images = (
            low.permute(0, 2, 1, 3, 4)
            .reshape(
                args.num_videos * GENERATED_FRAMES,
                3,
                LOW_RESOLUTION,
                LOW_RESOLUTION,
            )
            .contiguous()
        )
        high_images = superres_diffusion.sample(
            superres_model,
            (
                args.num_videos * GENERATED_FRAMES,
                3,
                HIGH_RESOLUTION,
                HIGH_RESOLUTION,
            ),
            condition=low_images,
            inference_steps=args.superres_inference_steps,
            eta=args.eta,
            generator=generator,
            callback=progress("superres 224x224"),
        )
        high = (
            high_images.reshape(
                args.num_videos,
                GENERATED_FRAMES,
                3,
                HIGH_RESOLUTION,
                HIGH_RESOLUTION,
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        full_low = torch.cat((first_low, low), dim=2)
        full_high = torch.cat((first_high, high), dim=2)

    for path, video in zip(
        output_paths(args.output, args.num_videos), full_high, strict=True
    ):
        write_mp4(path, video, args.fps)
        print(f"Wrote {path}", flush=True)
    if args.save_low_resolution:
        for path, video in zip(
            output_paths(args.output, args.num_videos, "_56"), full_low, strict=True
        ):
            write_mp4(path, video, args.fps)
            print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
