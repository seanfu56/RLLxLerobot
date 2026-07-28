#!/usr/bin/env python3
"""Prepare Piper teleop recordings for policy training.

Raw recordings live under ``piper-data/<source>/episode_XXX/`` with
``actions.csv``, ``observations.csv``, ``frame_timestamps.csv``, ``meta.json``
and ``video.mp4``.  Decoding MP4 at every training step is slow and makes random
access to goal frames expensive, so every source is decoded exactly once into a
shard:

    <output-root>/_shards/<source>/
        frames.npy      uint8  [N, S, S, 3]  RGB, center cropped then resized
        state.npy       float32 [N, 7]       measured joint state
        action.npy      float32 [N, 7]       commanded joint target
        shard.json                           episode boundaries + metadata

A *bundle* is what training consumes.  It is a manifest that points at one or
more shards, so ``can-all`` costs a few kilobytes instead of duplicating the
pixels of ``can`` and ``can-2``:

    <output-root>/<bundle>/bundle.json

Usage:
    python src/policy/data_prep.py                       # every bundle
    python src/policy/data_prep.py --bundles pick-bar pick-can
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

JOINT_NAMES = tuple([f"joint_{index}.pos" for index in range(1, 7)] + ["gripper.pos"])

# bundle name -> raw source directories under --raw-root
BUNDLE_SPECS: dict[str, tuple[str, ...]] = {
    "pick-bar": ("pick-bar",),
    "pick-can": ("pick-can",),
    "pick-can-2": ("pick-can-2",),
    "pick-scissors": ("pick-scissors",),
    "pick-scissors-2": ("pick-scissors-2",),
    "pick-can-all": ("pick-can", "pick-can-2"),
    "pick-scissors-all": ("pick-scissors", "pick-scissors-2"),
}

DEFAULT_IMAGE_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, default=Path("piper-data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/policy_lab/data"))
    parser.add_argument(
        "--bundles",
        nargs="+",
        default=sorted(BUNDLE_SPECS),
        choices=sorted(BUNDLE_SPECS),
        help="Which bundles to build (default: all six).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Square resolution written to disk. Must be a multiple of 32 (the ResNet stride).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-decode shards that already exist instead of reusing them.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# raw episode loading
# ---------------------------------------------------------------------------


def discover_episodes(source_root: Path) -> list[Path]:
    episodes = sorted(path for path in source_root.glob("episode_*") if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"No episode directories found under {source_root}")
    return episodes


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_episode(episode_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (states, actions, meta) for one episode, validating alignment.

    ``observations.csv`` holds the *measured* Piper state captured just before
    the paired action was sent; commanded actions are not a substitute, so
    legacy episodes without it are rejected.
    """
    required = ("actions.csv", "observations.csv", "frame_timestamps.csv", "meta.json", "video.mp4")
    missing = [name for name in required if not (episode_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{episode_path} is missing: {', '.join(missing)}")

    actions = _read_csv(episode_path / "actions.csv")
    observations = _read_csv(episode_path / "observations.csv")
    timestamps = _read_csv(episode_path / "frame_timestamps.csv")
    meta = json.loads((episode_path / "meta.json").read_text(encoding="utf-8"))

    expected = int(meta["video_frames"])
    if not actions or not (len(actions) == len(observations) == len(timestamps) == expected):
        raise ValueError(
            f"{episode_path} is not aligned: actions={len(actions)}, "
            f"observations={len(observations)}, timestamps={len(timestamps)}, meta={expected}"
        )

    for index, (action_row, observation_row) in enumerate(zip(actions, observations, strict=True)):
        if int(action_row["frame_index"]) != index or int(observation_row["frame_index"]) != index:
            raise ValueError(f"{episode_path} has inconsistent frame indices at row {index}")
        if action_row.get("source_control_sequence") != observation_row.get("source_control_sequence"):
            raise ValueError(f"{episode_path} action/observation control sequences differ at row {index}")

    state = np.asarray(
        [[float(row[name]) for name in JOINT_NAMES] for row in observations], dtype=np.float32
    )
    action = np.asarray(
        [[float(row[name]) for name in JOINT_NAMES] for row in actions], dtype=np.float32
    )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{episode_path} contains non-finite state/action values")

    return state, action, meta


def preprocess_frame(frame_bgr: np.ndarray, image_size: int) -> np.ndarray:
    """Center crop to a square, resize to ``image_size`` and convert to RGB.

    The overhead camera is 640x480 and the horizontal margins carry no task
    information, so the 480x480 center crop is lossless for our purposes.
    """
    height, width = frame_bgr.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    cropped = frame_bgr[top : top + side, left : left + side]
    resized = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# shard building
# ---------------------------------------------------------------------------


def build_shard(source_root: Path, shard_dir: Path, image_size: int, force: bool) -> Path:
    marker = shard_dir / "shard.json"
    if marker.is_file() and not force:
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("image_size") != image_size:
            raise ValueError(
                f"{shard_dir} was built at image_size={existing.get('image_size')} but "
                f"{image_size} was requested. Pass --force to rebuild."
            )
        print(f"  reuse {shard_dir} ({existing['num_frames']} frames)")
        return shard_dir

    episode_paths = discover_episodes(source_root)
    loaded = [(path, *load_episode(path)) for path in episode_paths]
    total_frames = sum(len(state) for _, state, _, _ in loaded)

    fps_values = {round(float(meta["video_fps_target"]), 3) for _, _, _, meta in loaded}
    if len(fps_values) != 1:
        raise ValueError(f"{source_root} mixes frame rates: {sorted(fps_values)}")
    fps = fps_values.pop()

    shard_dir.mkdir(parents=True, exist_ok=True)
    frames = np.lib.format.open_memmap(
        shard_dir / "frames.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(total_frames, image_size, image_size, 3),
    )
    states = np.zeros((total_frames, len(JOINT_NAMES)), dtype=np.float32)
    actions = np.zeros((total_frames, len(JOINT_NAMES)), dtype=np.float32)

    episodes: list[dict] = []
    cursor = 0
    for episode_path, state, action, meta in loaded:
        capture = cv2.VideoCapture(str(episode_path / "video.mp4"))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {episode_path / 'video.mp4'}")
        try:
            for offset in range(len(state)):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to decode {episode_path/'video.mp4'} frame {offset}")
                frames[cursor + offset] = preprocess_frame(frame, image_size)
            if capture.read()[0]:
                raise ValueError(f"{episode_path} video has more frames than action rows")
        finally:
            capture.release()

        states[cursor : cursor + len(state)] = state
        actions[cursor : cursor + len(action)] = action
        episodes.append(
            {
                "name": episode_path.name,
                "source": source_root.name,
                "start": cursor,
                "end": cursor + len(state),
                "task": str(meta.get("task", "")),
            }
        )
        cursor += len(state)
        print(f"  {episode_path.name}: {len(state)} frames (total {cursor}/{total_frames})", flush=True)

    frames.flush()
    del frames
    np.save(shard_dir / "state.npy", states)
    np.save(shard_dir / "action.npy", actions)
    marker.write_text(
        json.dumps(
            {
                "source": source_root.name,
                "raw_root": str(source_root),
                "image_size": image_size,
                "fps": fps,
                "joint_names": list(JOINT_NAMES),
                "num_frames": total_frames,
                "num_episodes": len(episodes),
                "episodes": episodes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return shard_dir


def summarize(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }


def write_bundle(bundle_dir: Path, name: str, shard_dirs: list[Path]) -> None:
    shard_metas = [json.loads((path / "shard.json").read_text(encoding="utf-8")) for path in shard_dirs]
    image_sizes = {meta["image_size"] for meta in shard_metas}
    fps_values = {meta["fps"] for meta in shard_metas}
    if len(image_sizes) != 1 or len(fps_values) != 1:
        raise ValueError(f"Bundle {name} mixes image sizes {image_sizes} or frame rates {fps_values}")

    states = np.concatenate([np.load(path / "state.npy") for path in shard_dirs], axis=0)
    actions = np.concatenate([np.load(path / "action.npy") for path in shard_dirs], axis=0)

    bundle_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "shards": [str(Path(*path.parts[-2:])) for path in shard_dirs],
        "shard_paths_relative_to": "bundle parent directory",
        "image_size": image_sizes.pop(),
        "fps": fps_values.pop(),
        "joint_names": list(JOINT_NAMES),
        "num_frames": int(sum(meta["num_frames"] for meta in shard_metas)),
        "num_episodes": int(sum(meta["num_episodes"] for meta in shard_metas)),
        "tasks": sorted({episode["task"] for meta in shard_metas for episode in meta["episodes"]}),
        "stats": {"state": summarize(states), "action": summarize(actions)},
    }
    (bundle_dir / "bundle.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"bundle {name}: {payload['num_episodes']} episodes, {payload['num_frames']} frames -> {bundle_dir}"
    )


def main() -> None:
    args = parse_args()
    if args.image_size % 32 != 0:
        raise SystemExit(
            f"--image-size must be a multiple of 32 (the ResNet-18 stride), got {args.image_size}"
        )

    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    shard_root = output_root / "_shards"

    needed_sources: list[str] = []
    for bundle in args.bundles:
        for source in BUNDLE_SPECS[bundle]:
            if source not in needed_sources:
                needed_sources.append(source)

    for source in needed_sources:
        source_root = raw_root / source
        if not source_root.is_dir():
            raise SystemExit(f"Raw source not found: {source_root}")
        print(f"shard {source}")
        build_shard(source_root, shard_root / source, args.image_size, args.force)

    for bundle in args.bundles:
        shard_dirs = [shard_root / source for source in BUNDLE_SPECS[bundle]]
        write_bundle(output_root / bundle, bundle, shard_dirs)


if __name__ == "__main__":
    main()
