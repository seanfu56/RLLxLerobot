#!/usr/bin/env python3
"""LoRA post-training (SFT) of Cosmos3-Nano on the Piper pick demonstrations.

Example:
    python src/cosmos/train.py --bundle piper-data/dataset/pick-can-all

``Cosmos3OmniTransformer`` consumes one packed joint text+vision token sequence
per forward call and has no batch dimension, so this trainer processes a single
clip at a time and relies on ``--grad-accum`` for an effective batch size.

To avoid re-deriving the joint-sequence packing (mRoPE ids, sequence indexes,
per-modality timestep splicing) or the VAE latent normalization, the training
step calls the pipeline's own helpers (``_encode_video``, ``tokenize_prompt``,
``_prepare_text_segment``, ``_prepare_vision_segment``) and only replaces the
iterative denoising loop with a single random-timestep rectified-flow step.

Only the generation pathway's attention projections are adapted; the
understanding (text) pathway stays frozen, matching the official Cosmos3 LoRA
SFT recipe.

This script needs a much newer diffusers/transformers stack than the ``piper``
conda environment provides -- see ``src/cosmos/README.md``.  Everything up to
and including ``--dry-run`` works with the ``piper`` dependencies alone.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cosmos.data import (
        NUM_FRAMES,
        RESOLUTION,
        PiperClipDataset,
        check_geometry,
        collate_clip,
        load_captions,
        load_episodes,
        resolve_captions,
        split_episodes,
        video_worker_init,
    )
    from cosmos.flow import (
        WAVER_MODE_SCALE,
        build_condition_mask,
        predict_velocity,
        sample_train_sigma,
    )
else:
    from .data import (
        NUM_FRAMES,
        RESOLUTION,
        PiperClipDataset,
        check_geometry,
        collate_clip,
        load_captions,
        load_episodes,
        resolve_captions,
        split_episodes,
        video_worker_init,
    )
    from .flow import (
        WAVER_MODE_SCALE,
        build_condition_mask,
        predict_velocity,
        sample_train_sigma,
    )

LOGGER = logging.getLogger("cosmos3-sft")

# Generation-pathway attention projections in Cosmos3OmniTransformer's joint
# attention block. The understanding-pathway projections (q_proj/k_proj/...)
# are deliberately excluded so the text tower is untouched.
LORA_TARGET_MODULES = ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]

DEFAULT_CAPTIONS = Path(__file__).resolve().parent / "captions.json"
# Sigmas at which validation loss is measured. Fixed so the number is
# comparable across steps rather than being dominated by the sampler's noise.
VALIDATION_SIGMAS = (0.25, 0.5, 0.75)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA supervised fine-tuning of Cosmos3-Nano on a Piper policy bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--bundle",
        type=Path,
        default=Path("piper-data/dataset/pick-can-all"),
        help="Policy bundle directory containing bundle.json",
    )
    data.add_argument(
        "--captions",
        type=Path,
        default=DEFAULT_CAPTIONS,
        help="JSON object mapping each episode task string to a prompt",
    )
    data.add_argument(
        "--caption",
        default=None,
        help="Override the caption for every task, ignoring --captions",
    )
    data.add_argument("--resolution", type=int, default=RESOLUTION, help="Square pixel resolution")
    data.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    data.add_argument("--clip-mode", choices=("window", "episode"), default="window")
    data.add_argument("--val-episodes", type=int, default=6, help="Episodes held out for validation")
    data.add_argument("--split-seed", type=int, default=42)
    data.add_argument("--num-workers", type=int, default=2)

    model = parser.add_argument_group("model")
    model.add_argument("--pretrained", default="nvidia/Cosmos3-Nano")
    model.add_argument("--lora-rank", type=int, default=32)
    model.add_argument("--lora-alpha", type=int, default=32)
    model.add_argument("--use-dora", action="store_true")
    model.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for memory in the transformer blocks",
    )

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--steps", type=int, default=1000, help="Optimizer steps")
    optimization.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Clips per optimizer step; this is the effective batch size",
    )
    optimization.add_argument("--lr", type=float, default=1e-4)
    optimization.add_argument("--weight-decay", type=float, default=1e-3)
    optimization.add_argument("--warmup-steps", type=int, default=20)
    optimization.add_argument("--lr-min-scale", type=float, default=0.1, help="Final LR multiplier")
    optimization.add_argument("--grad-clip", type=float, default=1.0)
    optimization.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    optimization.add_argument("--allow-tf32", action="store_true")
    optimization.add_argument("--seed", type=int, default=1000)
    optimization.add_argument(
        "--sigma-shift",
        type=float,
        default=5.0,
        help="Flow shift applied to the Waver video-time sampler",
    )
    optimization.add_argument(
        "--cfg-dropout",
        type=float,
        default=0.1,
        help="Probability of replacing the caption with an empty string",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--output-root", type=Path, default=Path("models/cosmos3_nano_lora"))
    output.add_argument("--run-name", default=None, help="Defaults to the bundle directory name")
    output.add_argument("--report-to", default="tensorboard", choices=("tensorboard", "wandb", "all"))
    output.add_argument("--log-freq", type=int, default=10)
    output.add_argument("--eval-freq", type=int, default=50, help="0 disables validation")
    output.add_argument("--save-freq", type=int, default=200, help="0 saves only at the end")
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve data, captions and geometry, then exit without loading the model",
    )

    args = parser.parse_args(argv)
    check_geometry(args.resolution, args.num_frames)
    if not 0.0 <= args.cfg_dropout <= 1.0:
        raise ValueError(f"--cfg-dropout must be in [0, 1], got {args.cfg_dropout}")
    if args.grad_accum < 1:
        raise ValueError(f"--grad-accum must be positive, got {args.grad_accum}")
    if args.run_name is None:
        args.run_name = args.bundle.expanduser().resolve().name
    return args


def compute_flow_loss(
    pipe,
    transformer,
    clean_latent: torch.Tensor,
    caption: str,
    sigma: torch.Tensor,
    *,
    device: torch.device,
    dit_dtype: torch.dtype,
    fps: float,
    num_frames: int,
    resolution: int,
    noise: torch.Tensor | None = None,
    drop_text_condition: bool = False,
) -> torch.Tensor:
    """One rectified-flow step: build xt and its target, pack the joint sequence
    with the pipeline's own helpers, and return the MSE over the noised frames."""
    if noise is None:
        noise = torch.randn_like(clean_latent)
    cond_mask = build_condition_mask(clean_latent.shape[2], device=device, dtype=clean_latent.dtype)

    target = noise - clean_latent
    xt = (1 - sigma) * clean_latent + sigma * noise
    xt = clean_latent * cond_mask + xt * (1 - cond_mask)

    prediction = predict_velocity(
        pipe,
        transformer,
        xt,
        "" if drop_text_condition else caption,
        sigma,
        device=device,
        dit_dtype=dit_dtype,
        fps=fps,
        num_frames=num_frames,
        resolution=resolution,
    )
    # Zero the conditioning frame's contribution instead of slicing, so the
    # loss stays a single fused op over the whole latent.
    masked_prediction = target * cond_mask + prediction * (1 - cond_mask)
    return F.mse_loss(masked_prediction, target)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_scale: float
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by linear decay to ``min_scale`` of the peak LR."""
    warmup_steps = min(max(warmup_steps, 0), total_steps)
    decay_steps = max(total_steps - warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = min(max(float(step - warmup_steps) / float(decay_steps), 0.0), 1.0)
        return 1.0 + (min_scale - 1.0) * progress

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def encode_validation_clips(pipe, dataset: PiperClipDataset, device: torch.device) -> list[dict]:
    """Encode the held-out clips once; the VAE is frozen so latents never change."""
    cached: list[dict] = []
    for index in range(len(dataset)):
        item = dataset[index]
        video = pipe.video_processor.preprocess_video(
            item["frames"], height=dataset.resolution, width=dataset.resolution
        )
        video = video.to(device=device, dtype=pipe.vae.dtype)
        cached.append(
            {
                "latent": pipe._encode_video(video).float().cpu(),
                "caption": item["caption"],
                "fps": item["fps"],
                "episode": item["episode"],
            }
        )
    return cached


@torch.no_grad()
def run_validation(pipe, transformer, cached: list[dict], args, device, dit_dtype) -> float:
    """Mean flow-matching loss over the held-out clips at fixed noise levels."""
    was_training = transformer.training
    transformer.eval()
    losses: list[float] = []
    for entry in cached:
        clean_latent = entry["latent"].to(device)
        # One noise draw shared across sigmas keeps the estimate low-variance.
        noise = torch.randn(clean_latent.shape, device=device, dtype=clean_latent.dtype)
        for value in VALIDATION_SIGMAS:
            sigma = torch.full((1, 1, 1, 1, 1), value, device=device, dtype=torch.float32)
            loss = compute_flow_loss(
                pipe,
                transformer,
                clean_latent,
                entry["caption"],
                sigma,
                device=device,
                dit_dtype=dit_dtype,
                fps=entry["fps"],
                num_frames=args.num_frames,
                resolution=args.resolution,
                noise=noise,
            )
            losses.append(loss.item())
    if was_training:
        transformer.train()
    return float(np.mean(losses))


def describe_plan(args, train_episodes, val_episodes, captions) -> dict:
    """Summary of everything resolved before the model is touched."""
    latent_frames = (args.num_frames - 1) // 4 + 1
    patches = (args.resolution // 16 // 2) ** 2
    return {
        "bundle": str(args.bundle),
        "train_episodes": len(train_episodes),
        "val_episodes": len(val_episodes),
        "clip": {
            "resolution": f"{args.resolution}x{args.resolution}",
            "num_frames": args.num_frames,
            "clip_mode": args.clip_mode,
            "source_fps": sorted({round(e.fps, 3) for e in train_episodes}),
            "vision_tokens": latent_frames * patches,
        },
        "captions": captions,
        "effective_batch_size": args.grad_accum,
        "steps": args.steps,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    episodes = load_episodes(args.bundle)
    caption_table = {} if args.caption is not None else load_captions(args.captions)
    captions = resolve_captions(episodes, caption_table, override=args.caption)
    train_episodes, val_episodes = split_episodes(episodes, args.val_episodes, args.split_seed)

    if args.dry_run:
        print(json.dumps(describe_plan(args, train_episodes, val_episodes, captions), indent=2))
        return

    # Imported late so --help and --dry-run stay usable without the Cosmos3
    # model stack installed.
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
    from diffusers import Cosmos3OmniPipeline
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig, get_peft_model
    from tqdm.auto import tqdm

    run_dir = (args.output_root / args.run_name).expanduser()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=str(run_dir), logging_dir=str(run_dir / "logs")),
    )
    set_seed(args.seed + accelerator.process_index)
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "train_args.json").write_text(
            json.dumps({key: str(value) for key, value in vars(args).items()}, indent=2),
            encoding="utf-8",
        )
    device = accelerator.device
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    train_dataset = PiperClipDataset(
        train_episodes,
        captions,
        num_frames=args.num_frames,
        resolution=args.resolution,
        clip_mode=args.clip_mode,
        random_window=True,
        seed=args.seed,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_clip,
        worker_init_fn=video_worker_init,
        persistent_workers=args.num_workers > 0,
    )

    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.pretrained, torch_dtype=torch.bfloat16, enable_safety_checker=False
    )
    pipe.vae.to(device).requires_grad_(False)
    pipe.transformer.to(device).requires_grad_(False)
    dit_dtype = pipe.transformer.dtype
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()

    # get_peft_model injects the LoRA layers into pipe.transformer in place, so
    # pipe._encode_video / tokenize_prompt / _prepare_*_segment keep working.
    transformer = get_peft_model(
        pipe.transformer,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=LORA_TARGET_MODULES,
            use_dora=args.use_dora,
        ),
    )
    if accelerator.mixed_precision in ("fp16", "bf16"):
        cast_training_params(transformer, dtype=torch.float32)
    lora_parameters = [p for p in transformer.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(lora_parameters, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = build_lr_scheduler(optimizer, args.warmup_steps, args.steps, args.lr_min_scale)
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    def save_adapter(tag: str) -> None:
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        target = run_dir / tag
        accelerator.unwrap_model(transformer).save_pretrained(str(target), safe_serialization=True)
        LOGGER.info("Saved LoRA adapter to %s", target)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            "cosmos3-nano-lora",
            config={
                key: (value if isinstance(value, (int, float, str, bool)) else str(value))
                for key, value in vars(args).items()
            },
        )
        plan = describe_plan(args, train_episodes, val_episodes, captions)
        plan["trainable_parameters_m"] = round(
            sum(p.numel() for p in lora_parameters) / 1e6, 2
        )
        LOGGER.info("Training plan:\n%s", json.dumps(plan, indent=2))

    validation_cache: list[dict] = []
    if val_episodes and args.eval_freq:
        val_dataset = PiperClipDataset(
            val_episodes,
            captions,
            num_frames=args.num_frames,
            resolution=args.resolution,
            clip_mode=args.clip_mode,
            random_window=False,
            seed=args.split_seed,
        )
        LOGGER.info("Encoding %d validation clips...", len(val_dataset))
        validation_cache = encode_validation_clips(pipe, val_dataset, device)

    # Cosmos3OmniPipeline construction leaves autograd disabled globally.
    torch.set_grad_enabled(True)
    transformer.train()

    progress = tqdm(
        range(args.steps), desc="steps", disable=not accelerator.is_local_main_process
    )
    global_step = 0
    accumulated_loss = 0.0
    data_iterator = iter(train_dataloader)

    if validation_cache:
        # LoRA is zero-initialized, so step 0 is the base model's loss.
        baseline = run_validation(pipe, transformer, validation_cache, args, device, dit_dtype)
        baseline = accelerator.reduce(torch.tensor(baseline, device=device), reduction="mean").item()
        if accelerator.is_main_process:
            accelerator.log({"val_loss": baseline}, step=0)
            LOGGER.info("step 0 (base model): val_loss=%.4f", baseline)

    while global_step < args.steps:
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_dataloader)
            batch = next(data_iterator)

        with accelerator.accumulate(transformer):
            with torch.no_grad():
                video = pipe.video_processor.preprocess_video(
                    batch["frames"], height=args.resolution, width=args.resolution
                )
                video = video.to(device=device, dtype=pipe.vae.dtype)
                clean_latent = pipe._encode_video(video).float()

            sigma = sample_train_sigma(device, args.sigma_shift)
            drop_text = torch.rand((), device=device).item() < args.cfg_dropout
            loss = compute_flow_loss(
                pipe,
                transformer,
                clean_latent,
                batch["caption"],
                sigma,
                device=device,
                dit_dtype=dit_dtype,
                fps=batch["fps"],
                num_frames=args.num_frames,
                resolution=args.resolution,
                drop_text_condition=drop_text,
            )
            accumulated_loss += (
                accelerator.gather(loss.detach().repeat(1)).mean().item() / args.grad_accum
            )

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(lora_parameters, args.grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{accumulated_loss:.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")
            if accelerator.is_main_process:
                accelerator.log(
                    {"train_loss": accumulated_loss, "lr": lr_scheduler.get_last_lr()[0]},
                    step=global_step,
                )
                if global_step % max(args.log_freq, 1) == 0:
                    LOGGER.info("step %d: train_loss=%.4f", global_step, accumulated_loss)
            accumulated_loss = 0.0

            if validation_cache and args.eval_freq and (
                global_step % args.eval_freq == 0 or global_step == args.steps
            ):
                val_loss = run_validation(
                    pipe, transformer, validation_cache, args, device, dit_dtype
                )
                val_loss = accelerator.reduce(
                    torch.tensor(val_loss, device=device), reduction="mean"
                ).item()
                if accelerator.is_main_process:
                    accelerator.log({"val_loss": val_loss}, step=global_step)
                    LOGGER.info("step %d: val_loss=%.4f", global_step, val_loss)

            if args.save_freq and global_step % args.save_freq == 0 and global_step < args.steps:
                save_adapter(f"checkpoint-{global_step:06d}")

    save_adapter("adapter")
    accelerator.end_training()


if __name__ == "__main__":
    main()
