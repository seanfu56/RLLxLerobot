"""Synthetic bundles built straight from numpy, without encoding video."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from policy.data_prep import JOINT_NAMES


def write_shard(shard_dir: Path, episode_lengths: list[int], image_size: int = 28,
                source: str | None = None, seed: int = 0) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    source = source or shard_dir.name
    rng = np.random.default_rng(seed)
    total = sum(episode_lengths)
    dim = len(JOINT_NAMES)

    frames = rng.integers(0, 256, size=(total, image_size, image_size, 3), dtype=np.uint8)
    states = np.zeros((total, dim), dtype=np.float32)
    actions = np.zeros((total, dim), dtype=np.float32)

    episodes = []
    cursor = 0
    for index, length in enumerate(episode_lengths):
        # state counts up inside the episode; action leads the state by one step
        ramp = np.arange(length, dtype=np.float32)[:, None] + np.arange(dim, dtype=np.float32)[None, :]
        states[cursor : cursor + length] = ramp
        actions[cursor : cursor + length] = ramp + 1.0
        episodes.append(
            {
                "name": f"episode_{index:03d}",
                "source": source,
                "start": cursor,
                "end": cursor + length,
                "task": "unit test",
            }
        )
        cursor += length

    np.save(shard_dir / "frames.npy", frames)
    np.save(shard_dir / "state.npy", states)
    np.save(shard_dir / "action.npy", actions)
    (shard_dir / "shard.json").write_text(
        json.dumps(
            {
                "source": source,
                "raw_root": str(shard_dir),
                "image_size": image_size,
                "fps": 20.0,
                "joint_names": list(JOINT_NAMES),
                "num_frames": total,
                "num_episodes": len(episodes),
                "episodes": episodes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return shard_dir


def write_bundle(root: Path, name: str, shards: dict[str, list[int]], image_size: int = 28) -> Path:
    """Create ``root/<name>/bundle.json`` pointing at ``root/_shards/<source>``."""
    shard_paths = []
    for index, (source, lengths) in enumerate(shards.items()):
        shard_paths.append(write_shard(root / "_shards" / source, lengths, image_size, source, seed=index))

    bundle_dir = root / name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "bundle.json").write_text(
        json.dumps(
            {
                "name": name,
                "shards": [f"_shards/{source}" for source in shards],
                "image_size": image_size,
                "fps": 20.0,
                "joint_names": list(JOINT_NAMES),
                "num_frames": int(sum(sum(lengths) for lengths in shards.values())),
                "num_episodes": int(sum(len(lengths) for lengths in shards.values())),
                "tasks": ["unit test"],
                "stats": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return bundle_dir
