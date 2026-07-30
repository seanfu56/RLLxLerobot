#!/usr/bin/env python3
"""Generate three future frames from a fixed first frame, then super-resolve them.

``--condition`` accepts one image, one video, an episode directory, or a whole
dataset: a policy bundle or any directory tree of videos.  Given a dataset, the
first frame of every episode conditions its own four-frame video.  Both
checkpoints are loaded once and episodes are sampled in batches.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
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
        VideoClipDataset,
        preprocess_frame,
        resolve_video_paths,
    )
    from diffusion.diffusion import GaussianDiffusion
    from diffusion.image_model import ImageUNet, ImageUNetConfig
    from diffusion.model import VideoUNet, VideoUNetConfig
    from diffusion.video_io import write_frame_strip, write_mp4
else:
    from .data import (
        GENERATED_FRAMES,
        HIGH_RESOLUTION,
        LOW_RESOLUTION,
        SAMPLED_FRAMES,
        TAIL_FRAMES,
        VIDEO_SUFFIXES,
        VideoClipDataset,
        preprocess_frame,
        resolve_video_paths,
    )
    from .diffusion import GaussianDiffusion
    from .image_model import ImageUNet, ImageUNetConfig
    from .model import VideoUNet, VideoUNetConfig
    from .video_io import write_frame_strip, write_mp4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--superres-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--condition",
        type=Path,
        required=True,
        help="Image, MP4, episode directory, policy bundle, or directory of videos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("samples"),
        help="Output directory, or one .mp4 path when a single video is generated",
    )
    parser.add_argument(
        "--num-videos", type=int, default=1, help="Videos generated per condition frame"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Videos sampled together; each video's starting noise is unaffected by it",
    )
    parser.add_argument("--base-inference-steps", type=int, default=50)
    parser.add_argument("--superres-inference-steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Playback rate for the four sparse trajectory frames",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    parser.add_argument("--raw-weights", action="store_true", help="Do not use EMA weights")
    parser.add_argument("--save-low-resolution", action="store_true")
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Also write the four frames side by side as one PNG strip",
    )
    parser.add_argument(
        "--save-reference",
        action="store_true",
        help="Also write each episode's real four frames; requires video conditions",
    )
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


def resolve_condition_paths(location: Path) -> list[Path]:
    """Resolve one condition frame source, or every episode of a dataset."""
    location = location.expanduser()
    if not location.exists():
        raise FileNotFoundError(f"Condition path does not exist: {location}")
    if location.is_file():
        return [location.resolve()]
    episode_video = location / "video.mp4"
    if episode_video.is_file():
        return [episode_video.resolve()]
    return resolve_video_paths(location)


def condition_label(path: Path) -> str:
    """Name a condition after its episode, since every episode file is video.mp4."""
    if path.stem == "video" and path.parent.name:
        shard = path.parent.parent.name
        return f"{shard}_{path.parent.name}" if shard else path.parent.name
    return path.stem


def unique_labels(paths: Sequence[Path]) -> list[str]:
    labels: list[str] = []
    counts: dict[str, int] = {}
    for path in paths:
        label = condition_label(path) or "condition"
        seen = counts.get(label, 0)
        counts[label] = seen + 1
        labels.append(label if seen == 0 else f"{label}_{seen:02d}")
    return labels


@dataclass(frozen=True)
class SampleJob:
    condition_path: Path
    label: str
    seed: int


def build_jobs(paths: Sequence[Path], num_videos: int, seed: int) -> list[SampleJob]:
    """One job per generated video, each with its own seed for reproducibility."""
    jobs: list[SampleJob] = []
    for path, label in zip(paths, unique_labels(paths), strict=True):
        for sample_index in range(num_videos):
            jobs.append(
                SampleJob(
                    condition_path=path,
                    label=label if num_videos == 1 else f"{label}_s{sample_index:02d}",
                    seed=seed + len(jobs),
                )
            )
    return jobs


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


def read_reference_video(path: Path) -> tuple[torch.Tensor, list[int]]:
    """Return the episode's real four 224px frames and their source indices."""
    dataset = VideoClipDataset(
        [path],
        resolutions=(HIGH_RESOLUTION,),
        tail_frames=TAIL_FRAMES,
        random_tail=False,
    )
    item = dataset[0]
    return item[f"video_{HIGH_RESOLUTION}"], item["frame_indices"].tolist()


def condition_tensor(frame_bgr: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    rgb = preprocess_frame(frame_bgr, size)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
    return tensor.div_(127.5).sub_(1.0).to(device)[None, :, None]


def job_noise(job: SampleJob, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw both stages' starting noise from the job's own seed.

    Seeding per job rather than per batch keeps a video's starting noise the
    same however ``--batch-size`` groups the episodes.  The finished pixels
    still shift a little, because cuDNN picks convolution algorithms by batch
    size; that drift is far smaller than the difference between two seeds.
    """
    generator = torch.Generator(device=device).manual_seed(job.seed)
    base = torch.randn(
        (1, 3, GENERATED_FRAMES, LOW_RESOLUTION, LOW_RESOLUTION),
        device=device,
        generator=generator,
    )
    superres = torch.randn(
        (GENERATED_FRAMES, 3, HIGH_RESOLUTION, HIGH_RESOLUTION),
        device=device,
        generator=generator,
    )
    return base, superres


def amp_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_videos < 1:
        raise SystemExit("--num-videos must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
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

    condition_paths = resolve_condition_paths(args.condition)
    if args.save_reference:
        not_videos = [
            str(path)
            for path in condition_paths
            if path.suffix.lower() not in VIDEO_SUFFIXES
        ]
        if not_videos:
            raise SystemExit(
                f"--save-reference needs video conditions; these are not videos: {not_videos}"
            )
    jobs = build_jobs(condition_paths, args.num_videos, args.seed)

    single_file = args.output.suffix.lower() == ".mp4"
    if single_file and len(jobs) != 1:
        raise SystemExit(
            f"--output must be a directory when generating {len(jobs)} videos, "
            f"got the file {args.output}"
        )
    output_dir = args.output.parent if single_file else args.output
    outputs = (
        [args.output]
        if single_file
        else [output_dir / f"{job.label}.mp4" for job in jobs]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"{len(condition_paths)} condition frame(s) x {args.num_videos} video(s) = "
        f"{len(jobs)} generated video(s) | batch={args.batch_size} | "
        f"base {args.base_inference_steps} + superres {args.superres_inference_steps} "
        f"DDIM steps | weights={'raw' if args.raw_weights else 'EMA'} | device={device}",
        flush=True,
    )

    # eta=0 is deterministic, so this generator only matters for stochastic DDIM.
    generator = torch.Generator(device=device).manual_seed(args.seed)
    records: list[dict] = []

    for start in range(0, len(jobs), args.batch_size):
        chunk = jobs[start : start + args.batch_size]
        chunk_outputs = outputs[start : start + args.batch_size]
        count = len(chunk)
        chunk_label = f"[{start + 1}-{start + count}/{len(jobs)}]"

        def progress(stage: str):
            def report(done: int, total: int, _: torch.Tensor) -> None:
                if done == total or done % max(1, total // 4) == 0:
                    print(f"{chunk_label} {stage}: {done}/{total}", flush=True)

            return report

        frames = [read_condition_frame(job.condition_path) for job in chunk]
        first_low = torch.cat(
            [condition_tensor(frame, LOW_RESOLUTION, device) for frame in frames]
        )
        first_high = torch.cat(
            [condition_tensor(frame, HIGH_RESOLUTION, device) for frame in frames]
        )
        noise = [job_noise(job, device) for job in chunk]
        base_noise = torch.cat([item[0] for item in noise])
        # Rows are ordered episode-major to match the base output's flattening.
        superres_noise = torch.cat([item[1] for item in noise])

        with amp_context(device, args.amp):
            low = base_diffusion.sample(
                base_model,
                (count, 3, GENERATED_FRAMES, LOW_RESOLUTION, LOW_RESOLUTION),
                condition=first_low,
                inference_steps=args.base_inference_steps,
                eta=args.eta,
                generator=generator,
                initial_noise=base_noise,
                callback=progress("base 56x56"),
            )
            low_images = (
                low.permute(0, 2, 1, 3, 4)
                .reshape(count * GENERATED_FRAMES, 3, LOW_RESOLUTION, LOW_RESOLUTION)
                .contiguous()
            )
            high_images = superres_diffusion.sample(
                superres_model,
                (count * GENERATED_FRAMES, 3, HIGH_RESOLUTION, HIGH_RESOLUTION),
                condition=low_images,
                inference_steps=args.superres_inference_steps,
                eta=args.eta,
                generator=generator,
                initial_noise=superres_noise,
                callback=progress("superres 224x224"),
            )
            high = (
                high_images.reshape(
                    count, GENERATED_FRAMES, 3, HIGH_RESOLUTION, HIGH_RESOLUTION
                )
                .permute(0, 2, 1, 3, 4)
                .contiguous()
            )
            full_low = torch.cat((first_low, low), dim=2).float().cpu()
            full_high = torch.cat((first_high, high), dim=2).float().cpu()

        for job, path, video_high, video_low in zip(
            chunk, chunk_outputs, full_high, full_low, strict=True
        ):
            write_mp4(path, video_high, args.fps)
            record = {
                "condition": str(job.condition_path),
                "seed": job.seed,
                "video": str(path),
            }
            if args.save_frames:
                strip_path = path.with_name(f"{path.stem}_frames.png")
                write_frame_strip(strip_path, video_high)
                record["frame_strip"] = str(strip_path)
            if args.save_low_resolution:
                low_path = path.with_name(f"{path.stem}_56{path.suffix}")
                write_mp4(low_path, video_low, args.fps)
                record["low_resolution_video"] = str(low_path)
            if args.save_reference:
                reference, frame_indices = read_reference_video(job.condition_path)
                reference_path = path.with_name(f"{path.stem}_reference{path.suffix}")
                write_mp4(reference_path, reference, args.fps)
                record["reference"] = str(reference_path)
                record["frame_indices"] = frame_indices
                if args.save_frames:
                    reference_strip = path.with_name(f"{path.stem}_reference_frames.png")
                    write_frame_strip(reference_strip, reference)
                    record["reference_frame_strip"] = str(reference_strip)
            records.append(record)
            print(f"Wrote {path}", flush=True)

    if not single_file:
        manifest = output_dir / "samples.json"
        manifest.write_text(
            json.dumps(
                {
                    "base_checkpoint": str(args.base_checkpoint),
                    "superres_checkpoint": str(args.superres_checkpoint),
                    "condition": str(args.condition),
                    "weights": "raw" if args.raw_weights else "ema",
                    "base_inference_steps": args.base_inference_steps,
                    "superres_inference_steps": args.superres_inference_steps,
                    "eta": args.eta,
                    "fps": args.fps,
                    "seed": args.seed,
                    "samples": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
