#!/usr/bin/env python3
"""Generate video from a fine-tuned Cosmos3-Nano LoRA adapter.

Example:
    python src/cosmos/sample.py \
      --adapter models/cosmos3_nano_lora/pick-can-all/adapter

By default this reproduces the training script's held-out split and runs four
independent image-to-video trajectories on each validation episode's
conditioning frame, writing the generated clips next to the ground-truth clip
they should match.  Flow Matching sampling is deterministic after its initial
Gaussian latent has been drawn, so diversity is obtained by sampling
independent latents (seeds), not by adding noise between ODE solver steps. Pass
``--adapter ""`` to sample the base model instead, which is the comparison
worth making before and after any fine-tuning run.
"""

from __future__ import annotations

import argparse
import logging
import secrets
from pathlib import Path

from PIL import Image, ImageOps

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cosmos.data import (
        NUM_FRAMES,
        RESOLUTION,
        PiperClipDataset,
        check_geometry,
        load_captions,
        load_episodes,
        resolve_captions,
        split_episodes,
    )
    from cosmos.runtime import (
        DEFAULT_ADAPTER,
        DEFAULT_CAPTIONS,
        DEFAULT_FLOW_SHIFT,
        DEFAULT_FPS,
        DEFAULT_GUIDANCE,
        DEFAULT_MODEL,
        DEFAULT_NEGATIVE,
        DEFAULT_STEPS,
        CosmosRunner,
    )
else:
    from .data import (
        NUM_FRAMES,
        RESOLUTION,
        PiperClipDataset,
        check_geometry,
        load_captions,
        load_episodes,
        resolve_captions,
        split_episodes,
    )
    from .runtime import (
        DEFAULT_ADAPTER,
        DEFAULT_CAPTIONS,
        DEFAULT_FLOW_SHIFT,
        DEFAULT_FPS,
        DEFAULT_GUIDANCE,
        DEFAULT_MODEL,
        DEFAULT_NEGATIVE,
        DEFAULT_STEPS,
        CosmosRunner,
    )

LOGGER = logging.getLogger("cosmos3-sample")
DEFAULT_SAMPLES_PER_CONDITION = 4
MAX_SEED = 2**63 - 1


def resolve_sample_seeds(seed: int | None, count: int) -> list[int]:
    """Return distinct Flow-Matching initial-noise seeds for one condition.

    A supplied seed is the reproducible start of a consecutive seed sequence.
    Without one, draw the base seed from the OS so separate invocations explore
    different trajectories.  The same resulting seed set is reused across
    conditions for paired episode comparisons. Reusing an explicit base seed
    in separate base and LoRA runs pairs those comparisons while still giving
    each condition ``count`` independent samples.
    """
    if count < 1:
        raise ValueError(f"--samples-per-condition must be positive, got {count}")
    if seed is None:
        seed = secrets.randbelow(MAX_SEED - count + 2)
    if seed < 0 or seed > MAX_SEED:
        raise ValueError(f"--seed must be between 0 and {MAX_SEED}, got {seed}")
    if seed + count - 1 > MAX_SEED:
        raise ValueError(
            f"--seed {seed} leaves room for only {MAX_SEED - seed + 1} samples, "
            f"not --samples-per-condition {count}"
        )
    return list(range(seed, seed + count))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image-to-video sampling with a fine-tuned Cosmos3-Nano adapter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_argument_group("source")
    source.add_argument("--pretrained", default=DEFAULT_MODEL)
    source.add_argument(
        "--adapter",
        type=str,
        default=DEFAULT_ADAPTER,
        help="PEFT adapter directory; pass an empty string to sample the base model",
    )
    source.add_argument("--bundle", type=Path, default=Path("piper-data/dataset/pick-can-all"))
    source.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    source.add_argument("--caption", default=None, help="Override the caption for every task")
    source.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Condition on this single first-frame image instead of the held-out episodes",
    )
    source.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Condition on every image in this directory, sorted by filename",
    )
    source.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="How many held-out episodes to sample (ignored with image input)",
    )
    source.add_argument("--val-episodes", type=int, default=6, help="Must match training")
    source.add_argument("--split-seed", type=int, default=42, help="Must match training")

    generation = parser.add_argument_group("generation")
    generation.add_argument("--resolution", type=int, default=RESOLUTION)
    generation.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    generation.add_argument("--clip-mode", choices=("window", "episode"), default="window")
    generation.add_argument("--fps", type=float, default=None, help="Defaults to the clip's own FPS")
    generation.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Denoising steps")
    generation.add_argument(
        "--sampler",
        choices=("unipc", "diverse-flow"),
        default="unipc",
        help="Standard independent UniPC trajectories or coupled DiverseFlow Euler inference",
    )
    generation.add_argument(
        "--guidance-scale",
        type=float,
        default=DEFAULT_GUIDANCE,
        help=(
            "Classifier-free guidance; lower values (toward 1) preserve more mode coverage, "
            "while higher values trade diversity for stronger prompt adherence"
        ),
    )
    generation.add_argument("--flow-shift", type=float, default=DEFAULT_FLOW_SHIFT)
    generation.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    generation.add_argument(
        "--samples-per-condition",
        type=int,
        default=DEFAULT_SAMPLES_PER_CONDITION,
        help="Independent Gaussian trajectories to generate for each conditioning frame",
    )
    generation.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base seed for reproducible sampling; variants use consecutive seeds. "
            "Omit to choose a fresh random base seed on every invocation"
        ),
    )

    diverse = parser.add_argument_group("DiverseFlow (with --sampler diverse-flow)")
    diverse.add_argument(
        "--diversity-scale",
        type=float,
        default=1.0,
        help=(
            "DPP-gradient strength W in gamma(sigma)=W*sigma/||gradient||; "
            "the paper uses 20 for images, so increase cautiously for video"
        ),
    )
    diverse.add_argument(
        "--diversity-feature",
        choices=("latent", "dino"),
        default="latent",
        help="Low-memory pooled Cosmos latents or paper-aligned decoded DINOv2 features",
    )
    diverse.add_argument(
        "--diversity-feature-frames",
        type=int,
        default=4,
        help="Ordered temporal segments/frames represented by the diversity feature",
    )
    diverse.add_argument(
        "--dino-variant",
        choices=("small", "base", "large", "giant"),
        default="large",
        help="DINOv2 backbone used by --diversity-feature dino",
    )
    diverse.add_argument(
        "--dpp-bandwidth",
        type=float,
        default=1.0,
        help="RBF spread h in the DPP kernel",
    )
    diverse.add_argument(
        "--dpp-jitter",
        type=float,
        default=1e-4,
        help="Diagonal stabilization for near-duplicate particle sets",
    )
    diverse.add_argument(
        "--diversity-every",
        type=int,
        default=1,
        help="Evaluate the DPP gradient every N Euler steps",
    )
    diverse.add_argument(
        "--quality-percentile",
        type=float,
        default=0.995,
        help="Gaussian-source percentile for the DPP quality constraint; 0 disables it",
    )
    diverse.add_argument(
        "--quality-floor",
        type=float,
        default=1e-4,
        help="Minimum DPP quality assigned to an off-prior particle",
    )
    diverse.add_argument(
        "--quality-strength",
        type=float,
        default=1.0,
        help="Penalty per unit squared source-norm excess",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=Path, default=Path("samples/cosmos3-nano"))
    output.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip writing the ground-truth clip alongside each generation",
    )

    args = parser.parse_args(argv)
    check_geometry(args.resolution, args.num_frames)
    if args.image is not None and args.images_dir is not None:
        raise ValueError("Pass either --image or --images-dir, not both")
    if args.image is None and args.images_dir is None and args.episodes < 1:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")
    # Resolve once so an omitted seed draws only one random sequence per run.
    args.sample_seeds = resolve_sample_seeds(args.seed, args.samples_per_condition)
    if (
        args.sampler == "diverse-flow"
        and args.diversity_scale > 0
        and len(args.sample_seeds) < 2
    ):
        raise ValueError(
            "--sampler diverse-flow with positive --diversity-scale needs "
            "--samples-per-condition >= 2"
        )
    return args


def build_conditions(args) -> list[dict]:
    """Resolve the (conditioning frame, caption, fps) triples to generate from."""
    if args.image is not None or args.images_dir is not None:
        if args.image is not None:
            paths = [args.image.expanduser()]
        else:
            image_dir = args.images_dir.expanduser()
            if not image_dir.is_dir():
                raise FileNotFoundError(f"Image directory not found: {image_dir}")
            suffixes = {".jpg", ".jpeg", ".png", ".webp"}
            paths = sorted(
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in suffixes
            )
            if not paths:
                raise ValueError(f"No supported images found in {image_dir}")

        if args.caption is not None:
            caption = args.caption
        else:
            caption_table = load_captions(args.captions)
            if len(caption_table) != 1:
                raise ValueError(
                    f"{args.captions} has {len(caption_table)} captions; "
                    "pass --caption when sampling external images"
                )
            caption = next(iter(caption_table.values()))
        fps = args.fps if args.fps is not None else DEFAULT_FPS
        conditions = []
        for path in paths:
            with Image.open(path) as source:
                image = ImageOps.fit(
                    source.convert("RGB"),
                    (args.resolution, args.resolution),
                    method=Image.Resampling.LANCZOS,
                )
            conditions.append(
                {
                    "name": path.stem,
                    "image": image,
                    "caption": caption,
                    "fps": fps,
                    "reference": None,
                }
            )
        return conditions

    episodes = load_episodes(args.bundle)
    caption_table = {} if args.caption is not None else load_captions(args.captions)
    captions = resolve_captions(episodes, caption_table, override=args.caption)
    _, val_episodes = split_episodes(episodes, args.val_episodes, args.split_seed)
    if not val_episodes:
        raise ValueError("No held-out episodes; raise --val-episodes or pass --image")
    selected = val_episodes[: args.episodes]
    dataset = PiperClipDataset(
        selected,
        captions,
        num_frames=args.num_frames,
        resolution=args.resolution,
        clip_mode=args.clip_mode,
        random_window=False,
        seed=args.split_seed,
    )
    conditions = []
    for index in range(len(dataset)):
        item = dataset[index]
        conditions.append(
            {
                "name": item["episode"],
                "image": item["frames"][0],
                "caption": item["caption"],
                "fps": args.fps if args.fps is not None else item["fps"],
                "reference": item["frames"],
            }
        )
    return conditions


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO
    )

    conditions = build_conditions(args)

    from diffusers.utils import export_to_video

    adapter = args.adapter.strip() if args.adapter else ""
    runner = CosmosRunner(
        pretrained=args.pretrained,
        adapter=adapter or None,
        resolution=args.resolution,
        num_frames=args.num_frames,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        flow_shift=args.flow_shift,
        negative_prompt=args.negative_prompt,
        # Captions here are per-episode, so they are passed per generate() call.
        prompt=conditions[0]["caption"],
    )

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "lora" if adapter else "base"
    seeds = args.sample_seeds
    if args.sampler == "diverse-flow":
        if __package__ in (None, ""):
            from cosmos.diverse_flow import (
                DinoTrajectoryFeatures,
                DiverseFlowConfig,
                generate_diverse,
            )
        else:
            from .diverse_flow import (
                DinoTrajectoryFeatures,
                DiverseFlowConfig,
                generate_diverse,
            )

        diverse_config = DiverseFlowConfig(
            steps=args.steps,
            cfg_scale=args.guidance_scale,
            flow_shift=args.flow_shift,
            diversity_scale=args.diversity_scale,
            feature=args.diversity_feature,
            feature_frames=args.diversity_feature_frames,
            dino_variant=args.dino_variant,
            dpp_bandwidth=args.dpp_bandwidth,
            dpp_jitter=args.dpp_jitter,
            diversity_every=args.diversity_every,
            quality_percentile=args.quality_percentile,
            quality_floor=args.quality_floor,
            quality_strength=args.quality_strength,
        )
        dino = (
            DinoTrajectoryFeatures(
                args.dino_variant,
                frames=args.diversity_feature_frames,
                device=runner.device,
            )
            if args.diversity_feature == "dino" and args.diversity_scale > 0
            else None
        )
        LOGGER.info(
            "Generating %d coupled DiverseFlow trajectories per condition with seeds %s "
            "(feature %s, diversity %.3g, CFG %.2f, flow shift %.2f)",
            len(seeds),
            seeds,
            args.diversity_feature,
            args.diversity_scale,
            args.guidance_scale,
            args.flow_shift,
        )
    else:
        diverse_config = None
        dino = None
        LOGGER.info(
            "Generating %d independent UniPC trajectories per condition with seeds %s "
            "(CFG %.2f, flow shift %.2f)",
            len(seeds),
            seeds,
            args.guidance_scale,
            args.flow_shift,
        )

    for condition in conditions:
        if args.sampler == "diverse-flow":
            assert diverse_config is not None
            videos = generate_diverse(
                runner.pipe,
                condition["image"],
                condition["caption"],
                args.negative_prompt,
                seeds,
                diverse_config,
                device=runner.device,
                fps=condition["fps"],
                num_frames=args.num_frames,
                resolution=args.resolution,
                dino=dino,
            )
            generated = zip(seeds, videos)
        else:
            generated = (
                (
                    seed,
                    runner.generate(
                        condition["image"],
                        prompt=condition["caption"],
                        fps=condition["fps"],
                        seed=seed,
                    ),
                )
                for seed in seeds
            )

        for seed, frames in generated:
            sampler_suffix = "_diverse" if args.sampler == "diverse-flow" else ""
            target = output_dir / (
                f"{condition['name']}_{suffix}{sampler_suffix}_seed{seed}.mp4"
            )
            export_to_video(frames, str(target), fps=round(condition["fps"]))
            LOGGER.info("Wrote %s", target)

        if condition["reference"] is not None and not args.no_reference:
            reference = output_dir / f"{condition['name']}_reference.mp4"
            # Pass PIL frames, not arrays: export_to_video rescales ndarray
            # input by 255, which would overflow already-uint8 frames.
            export_to_video(condition["reference"], str(reference), fps=round(condition["fps"]))
            LOGGER.info("Wrote %s", reference)


if __name__ == "__main__":
    main()
