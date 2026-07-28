#!/usr/bin/env python3
"""Train one stage of the Piper video diffusion cascade.

Examples:
    python src/diffusion/train.py --stage base
    python src/diffusion/train.py --stage superres

Each episode supplies four trajectory frames. The first is fixed conditioning;
the base stage generates the other three at 56x56. The super-resolution stage
learns their 56x56 to 224x224 mapping while also seeing the first 224x224 frame.
It does not require the base checkpoint during training; the two checkpoints
are chained by ``sample.py`` after both stages have been trained.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from diffusion.data import (
        GENERATED_FRAMES,
        HIGH_RESOLUTION,
        LOW_RESOLUTION,
        SAMPLED_FRAMES,
        TAIL_FRAMES,
        VideoClipDataset,
        resolve_video_paths,
        split_video_paths,
        video_worker_init,
    )
    from diffusion.diffusion import GaussianDiffusion
    from diffusion.model import VideoUNet, VideoUNetConfig, spatial_resize
else:
    from .data import (
        GENERATED_FRAMES,
        HIGH_RESOLUTION,
        LOW_RESOLUTION,
        SAMPLED_FRAMES,
        TAIL_FRAMES,
        VideoClipDataset,
        resolve_video_paths,
        split_video_paths,
        video_worker_init,
    )
    from .diffusion import GaussianDiffusion
    from .model import VideoUNet, VideoUNetConfig, spatial_resize


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        decay = min(self.decay, (1 + step) / (10 + step))
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1 - decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if state.keys() != self.shadow.keys():
            missing = self.shadow.keys() - state.keys()
            unexpected = state.keys() - self.shadow.keys()
            raise ValueError(
                f"EMA state mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        for name, value in state.items():
            self.shadow[name].copy_(value)

    @torch.no_grad()
    def swap_in(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup: dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def swap_out(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, parameter in model.named_parameters():
            if name in backup:
                parameter.copy_(backup[name])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    data = parser.add_argument_group("data")
    data.add_argument(
        "--stage", choices=("base", "superres"), required=True, help="Cascade stage to train"
    )
    data.add_argument(
        "--bundle",
        type=Path,
        default=Path("piper-data/dataset/pick-can-all"),
        help="Policy bundle, raw-video directory, or one video",
    )
    data.add_argument(
        "--val-videos", type=int, default=6, help="Whole episodes held out for validation"
    )
    data.add_argument("--num-workers", type=int, default=4)

    architecture = parser.add_argument_group("architecture")
    architecture.add_argument(
        "--base-channels",
        type=int,
        default=None,
        help="Default: 64 for base and 32 for super-resolution",
    )
    architecture.add_argument(
        "--channel-multipliers",
        type=int,
        nargs="+",
        default=None,
        help="Default: 1 2 4 4 (three spatial downsamplings)",
    )
    architecture.add_argument("--blocks-per-level", type=int, default=2)
    architecture.add_argument("--time-embedding-dim", type=int, default=256)
    architecture.add_argument("--dropout", type=float, default=0.0)
    architecture.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Recompute residual blocks during backward to reduce activation memory",
    )
    architecture.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    architecture.set_defaults(gradient_checkpointing=None)

    diffusion = parser.add_argument_group("diffusion")
    diffusion.add_argument("--diffusion-steps", type=int, default=1000)
    diffusion.add_argument("--beta-schedule", choices=("cosine", "linear"), default="cosine")
    diffusion.add_argument(
        "--snr-gamma",
        type=float,
        default=5.0,
        help="Min-SNR loss cap; 0 disables weighting",
    )

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--steps", type=int, default=100_000)
    optimization.add_argument(
        "--batch-size", type=int, default=None, help="Default: 4 for base, 1 for superres"
    )
    optimization.add_argument(
        "--grad-accumulation",
        type=int,
        default=None,
        help="Default: 1 for base, 4 for superres",
    )
    optimization.add_argument("--lr", type=float, default=2e-4)
    optimization.add_argument("--min-lr", type=float, default=2e-6)
    optimization.add_argument("--warmup-steps", type=int, default=1000)
    optimization.add_argument("--weight-decay", type=float, default=1e-4)
    optimization.add_argument("--grad-clip", type=float, default=1.0)
    optimization.add_argument("--ema-decay", type=float, default=0.9999)
    optimization.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    optimization.add_argument("--device", default="cuda")
    optimization.add_argument("--seed", type=int, default=1000)

    output = parser.add_argument_group("output")
    output.add_argument("--output-root", type=Path, default=Path("models/video_diffusion"))
    output.add_argument(
        "--run-name",
        default=None,
        help="Directory relative to output-root; default: <bundle-name>/<stage>",
    )
    output.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume")
    output.add_argument("--log-freq", type=int, default=50)
    output.add_argument("--eval-freq", type=int, default=1000)
    output.add_argument("--eval-batches", type=int, default=8)
    output.add_argument("--save-freq", type=int, default=5000)
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable; use --device cpu for a smoke test")
    return device


def bundle_name(path: Path) -> str:
    manifest = path / "bundle.json"
    if manifest.is_file():
        return str(json.loads(manifest.read_text(encoding="utf-8")).get("name", path.name))
    return path.stem


def model_config_from_args(args: argparse.Namespace) -> VideoUNetConfig:
    is_superres = args.stage == "superres"
    return VideoUNetConfig(
        channels=3,
        condition_channels=6 if is_superres else 3,
        base_channels=(
            args.base_channels
            if args.base_channels is not None
            else (32 if is_superres else 64)
        ),
        channel_multipliers=tuple(args.channel_multipliers or (1, 2, 4, 4)),
        blocks_per_level=args.blocks_per_level,
        time_embedding_dim=args.time_embedding_dim,
        dropout=args.dropout,
        gradient_checkpointing=(
            is_superres
            if args.gradient_checkpointing is None
            else args.gradient_checkpointing
        ),
    )


def lr_at(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return args.min_lr + (args.lr - args.min_lr) * cosine


def stage_tensors(
    batch: dict[str, torch.Tensor], stage: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None]:
    low = batch[f"video_{LOW_RESOLUTION}"].to(device, non_blocking=True)
    if low.shape[2] != SAMPLED_FRAMES:
        raise ValueError(
            f"Expected {SAMPLED_FRAMES} sampled frames, got tensor shape {tuple(low.shape)}"
        )
    first_low = low[:, :, :1]
    future_low = low[:, :, 1:]
    if stage == "base":
        return future_low, first_low
    high = batch[f"video_{HIGH_RESOLUTION}"].to(device, non_blocking=True)
    first_high = high[:, :, :1]
    future_high = high[:, :, 1:]
    low_at_high_resolution = spatial_resize(
        future_low, (HIGH_RESOLUTION, HIGH_RESOLUTION)
    )
    fixed_first_frame = first_high.expand(
        -1, -1, GENERATED_FRAMES, -1, -1
    )
    condition = torch.cat((low_at_high_resolution, fixed_first_frame), dim=1)
    return future_high, condition


def autocast_context(device: torch.device, amp: str):
    if amp == "off" or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate(
    model: VideoUNet,
    ema: ExponentialMovingAverage,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    *,
    stage: str,
    device: torch.device,
    amp: str,
    batches: int,
    snr_gamma: float | None,
) -> float:
    if not len(loader):
        return float("nan")
    backup = ema.swap_in(model)
    was_training = model.training
    model.eval()
    losses: list[float] = []
    try:
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            target, condition = stage_tensors(batch, stage, device)
            generator = torch.Generator(device=device).manual_seed(91_337 + index)
            timesteps = torch.randint(
                0,
                diffusion.timesteps,
                (target.shape[0],),
                device=device,
                generator=generator,
            )
            noise = torch.randn(
                target.shape,
                device=device,
                dtype=target.dtype,
                generator=generator,
            )
            with autocast_context(device, amp):
                loss = diffusion.training_loss(
                    model,
                    target,
                    condition,
                    snr_gamma=snr_gamma,
                    timesteps=timesteps,
                    noise=noise,
                )
            losses.append(float(loss))
    finally:
        model.train(was_training)
        ema.swap_out(model, backup)
    return float(np.mean(losses)) if losses else float("nan")


def jsonable_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def best_logged_validation(path: Path) -> float:
    if not path.is_file():
        return float("inf")
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line).get("val_loss")
        except json.JSONDecodeError:
            continue
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return min(values, default=float("inf"))


def save_checkpoint(
    path: Path,
    *,
    step: int,
    stage: str,
    model: VideoUNet,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    diffusion: GaussianDiffusion,
    args: argparse.Namespace,
) -> None:
    payload = {
        "format_version": 1,
        "stage": stage,
        "step": step,
        "model_config": model.config.to_dict(),
        "diffusion_config": diffusion.config_dict(),
        "data_config": {
            "low_resolution": LOW_RESOLUTION,
            "high_resolution": HIGH_RESOLUTION,
            "sampled_frames": SAMPLED_FRAMES,
            "generated_frames": GENERATED_FRAMES,
            "tail_frames": TAIL_FRAMES,
            "fps": args.source_fps,
        },
        "train_config": jsonable_args(args),
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_resume(
    path: Path,
    *,
    model: VideoUNet,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    diffusion: GaussianDiffusion,
    stage: str,
    device: torch.device,
) -> int:
    checkpoint_data = torch.load(path, map_location=device, weights_only=False)
    if checkpoint_data.get("stage") != stage:
        raise ValueError(
            f"Cannot resume {stage!r} from a {checkpoint_data.get('stage')!r} checkpoint"
        )
    if checkpoint_data.get("model_config") != model.config.to_dict():
        raise ValueError("Resume checkpoint model configuration differs from the CLI configuration")
    if checkpoint_data.get("diffusion_config") != diffusion.config_dict():
        raise ValueError("Resume checkpoint diffusion configuration differs from the CLI configuration")
    model.load_state_dict(checkpoint_data["model"])
    ema.load_state_dict(checkpoint_data["ema"])
    optimizer.load_state_dict(checkpoint_data["optimizer"])
    if checkpoint_data.get("scaler"):
        scaler.load_state_dict(checkpoint_data["scaler"])
    return int(checkpoint_data["step"])


def make_loader(
    dataset: VideoClipDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    kwargs = {}
    if workers:
        kwargs.update(prefetch_factor=2, persistent_workers=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=shuffle and len(dataset) >= batch_size,
        worker_init_fn=video_worker_init,
        **kwargs,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.steps < 1 or args.eval_batches < 1:
        raise SystemExit("--steps and --eval-batches must be positive")
    if args.log_freq < 1 or args.eval_freq < 1 or args.save_freq < 1:
        raise SystemExit("--log-freq, --eval-freq, and --save-freq must be positive")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.base_channels is not None and args.base_channels < 1:
        raise SystemExit("--base-channels must be positive")
    if args.grad_accumulation is not None and args.grad_accumulation < 1:
        raise SystemExit("--grad-accumulation must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise SystemExit("--ema-decay must be in [0, 1)")

    set_seed(args.seed)
    device = resolve_device(args.device)
    paths = resolve_video_paths(args.bundle)
    train_paths, val_paths = split_video_paths(paths, args.val_videos, args.seed)
    resolutions = (
        (LOW_RESOLUTION,)
        if args.stage == "base"
        else (LOW_RESOLUTION, HIGH_RESOLUTION)
    )
    train_dataset = VideoClipDataset(
        train_paths,
        resolutions=resolutions,
        tail_frames=TAIL_FRAMES,
        random_tail=True,
        seed=args.seed,
    )
    val_dataset = (
        VideoClipDataset(
            val_paths,
            resolutions=resolutions,
            tail_frames=TAIL_FRAMES,
            random_tail=False,
            seed=args.seed,
        )
        if val_paths
        else None
    )
    args.source_fps = train_dataset.fps
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else (1 if args.stage == "superres" else 4)
    )
    grad_accumulation = args.grad_accumulation or (
        4 if args.stage == "superres" else 1
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        workers=args.num_workers,
        shuffle=True,
        device=device,
    )
    val_loader = (
        make_loader(
            val_dataset,
            batch_size=batch_size,
            workers=args.num_workers,
            shuffle=False,
            device=device,
        )
        if val_dataset is not None
        else None
    )

    model = VideoUNet(model_config_from_args(args)).to(device)
    diffusion = GaussianDiffusion(args.diffusion_steps, args.beta_schedule).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=(device.type == "cuda" and args.amp == "fp16")
    )
    ema = ExponentialMovingAverage(model, args.ema_decay)
    snr_gamma = args.snr_gamma if args.snr_gamma > 0 else None

    name = args.run_name or f"{bundle_name(args.bundle)}/{args.stage}"
    run_dir = args.output_root / name
    if args.resume is None and run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(
            f"Run directory is not empty: {run_dir}. Use --resume or choose --run-name."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "stage": args.stage,
                "model_config": model.config.to_dict(),
                "diffusion_config": diffusion.config_dict(),
                "data_config": {
                    "low_resolution": LOW_RESOLUTION,
                    "high_resolution": HIGH_RESOLUTION,
                    "sampled_frames": SAMPLED_FRAMES,
                    "generated_frames": GENERATED_FRAMES,
                    "tail_frames": TAIL_FRAMES,
                    "fps": args.source_fps,
                },
                "train_config": jsonable_args(args),
                "train_videos": [str(path) for path in train_paths],
                "val_videos": [str(path) for path in val_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    start_step = 0
    if args.resume is not None:
        start_step = load_resume(
            args.resume,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scaler=scaler,
            diffusion=diffusion,
            stage=args.stage,
            device=device,
        )
        if start_step > args.steps:
            raise SystemExit(
                f"Checkpoint is at step {start_step}, beyond requested --steps={args.steps}"
            )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"{args.stage}: {parameters / 1e6:.1f}M parameters | "
        f"{len(train_paths)} train/{len(val_paths)} val videos | "
        f"{len(train_dataset)} trajectory samples/epoch | device={device}",
        flush=True,
    )
    print(
        f"condition=frame 1 | generated=frames 2-{SAMPLED_FRAMES} at "
        f"{LOW_RESOLUTION if args.stage == 'base' else HIGH_RESOLUTION}px | "
        f"batch={batch_size} x accumulation={grad_accumulation} | run={run_dir}",
        flush=True,
    )
    source_shapes = sorted({(info.height, info.width) for info in train_dataset.videos})
    crop_shapes = sorted({(min(height, width),) * 2 for height, width in source_shapes})
    print(
        f"source frame shapes={source_shapes} | symmetric square crops={crop_shapes}",
        flush=True,
    )

    iterator = iter(train_loader)
    model.train()
    log_path = run_dir / "log.jsonl"
    best_loss = best_logged_validation(log_path)
    recent_losses: list[float] = []
    last_log_time = time.monotonic()

    for zero_based_step in range(start_step, args.steps):
        completed_step = zero_based_step + 1
        learning_rate = lr_at(zero_based_step, args)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(grad_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            target, condition = stage_tensors(batch, args.stage, device)
            with autocast_context(device, args.amp):
                loss = diffusion.training_loss(
                    model, target, condition, snr_gamma=snr_gamma
                )
                scaled_loss = loss / grad_accumulation
            scaler.scale(scaled_loss).backward()
            accumulated += float(loss.detach())

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model, completed_step)
        train_loss = accumulated / grad_accumulation
        recent_losses.append(train_loss)

        if completed_step % args.log_freq == 0 or completed_step == 1:
            elapsed = time.monotonic() - last_log_time
            steps_in_window = 1 if completed_step == 1 else args.log_freq
            record = {
                "step": completed_step,
                "train_loss": float(np.mean(recent_losses)),
                "lr": learning_rate,
                "grad_norm": float(gradient_norm),
                "steps_per_second": steps_in_window / max(elapsed, 1e-8),
            }
            print(
                f"step {completed_step:7d} | loss {record['train_loss']:.5f} | "
                f"lr {learning_rate:.2e} | grad {float(gradient_norm):.2f} | "
                f"{record['steps_per_second']:.2f} step/s",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            recent_losses.clear()
            last_log_time = time.monotonic()

        should_evaluate = (
            val_loader is not None
            and (completed_step % args.eval_freq == 0 or completed_step == args.steps)
        )
        if should_evaluate:
            validation_loss = evaluate(
                model,
                ema,
                diffusion,
                val_loader,
                stage=args.stage,
                device=device,
                amp=args.amp,
                batches=args.eval_batches,
                snr_gamma=snr_gamma,
            )
            print(f"eval {completed_step:7d} | val_loss {validation_loss:.5f}", flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"step": completed_step, "val_loss": validation_loss}) + "\n"
                )
            if validation_loss < best_loss:
                best_loss = validation_loss
                save_checkpoint(
                    run_dir / "best.pt",
                    step=completed_step,
                    stage=args.stage,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scaler=scaler,
                    diffusion=diffusion,
                    args=args,
                )

        if completed_step % args.save_freq == 0:
            save_checkpoint(
                run_dir / "latest.pt",
                step=completed_step,
                stage=args.stage,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scaler=scaler,
                diffusion=diffusion,
                args=args,
            )

    save_checkpoint(
        run_dir / "final.pt",
        step=args.steps,
        stage=args.stage,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scaler=scaler,
        diffusion=diffusion,
        args=args,
    )
    print(f"Training complete: {run_dir / 'final.pt'}", flush=True)


if __name__ == "__main__":
    main()
