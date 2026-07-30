#!/usr/bin/env python3
"""Generate video from a fine-tuned Cosmos3-Nano LoRA adapter.

Example:
    python src/cosmos/sample.py \
      --adapter models/cosmos3_nano_lora/pick-can-all/adapter

By default this reproduces the training script's held-out split and runs
image-to-video on each validation episode's conditioning frame, writing the
generated clip next to the ground-truth clip it should match.  Pass
``--adapter ""`` to sample the base model instead, which is the comparison
worth making before and after any fine-tuning run.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from PIL import Image

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
        DEFAULT_GUIDANCE,
        DEFAULT_MODEL,
        DEFAULT_NEGATIVE,
        DEFAULT_STEPS,
        CosmosRunner,
    )

LOGGER = logging.getLogger("cosmos3-sample")


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
        help="Condition on this image instead of the held-out episodes",
    )
    source.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="How many held-out episodes to sample (ignored with --image)",
    )
    source.add_argument("--val-episodes", type=int, default=6, help="Must match training")
    source.add_argument("--split-seed", type=int, default=42, help="Must match training")

    generation = parser.add_argument_group("generation")
    generation.add_argument("--resolution", type=int, default=RESOLUTION)
    generation.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    generation.add_argument("--clip-mode", choices=("window", "episode"), default="window")
    generation.add_argument("--fps", type=float, default=None, help="Defaults to the clip's own FPS")
    generation.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Denoising steps")
    generation.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE)
    generation.add_argument("--flow-shift", type=float, default=DEFAULT_FLOW_SHIFT)
    generation.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    generation.add_argument("--seed", type=int, default=1000)

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=Path, default=Path("samples/cosmos3-nano"))
    output.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip writing the ground-truth clip alongside each generation",
    )

    args = parser.parse_args(argv)
    check_geometry(args.resolution, args.num_frames)
    if args.image is None and args.episodes < 1:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")
    return args


def build_conditions(args) -> list[dict]:
    """Resolve the (conditioning frame, caption, fps) triples to generate from."""
    episodes = load_episodes(args.bundle)
    caption_table = {} if args.caption is not None else load_captions(args.captions)
    captions = resolve_captions(episodes, caption_table, override=args.caption)

    if args.image is not None:
        image = Image.open(args.image).convert("RGB").resize(
            (args.resolution, args.resolution), Image.BICUBIC
        )
        caption = args.caption or captions[sorted(captions)[0]]
        return [
            {
                "name": args.image.stem,
                "image": image,
                "caption": caption,
                "fps": args.fps if args.fps is not None else episodes[0].fps,
                "reference": None,
            }
        ]

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

    for condition in conditions:
        frames = runner.generate(
            condition["image"],
            prompt=condition["caption"],
            fps=condition["fps"],
            seed=args.seed,
        )
        target = output_dir / f"{condition['name']}_{suffix}.mp4"
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
