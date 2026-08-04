"""Bundle loading and action chunking with an observation history.

A sample is the last ``n_obs_steps`` frames and states, plus the ``horizon``
actions that follow. The history matters because a single frame cannot express
velocity: the same picture of a gripper above an object occurs on the way down
and on the way back up, and the correct action differs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .angles import wrap_numpy

ACTION_REPRS = ("absolute", "delta")


@dataclass(frozen=True)
class EpisodeRef:
    """One episode inside a bundle, in shard-local frame coordinates."""

    shard: int
    start: int
    end: int
    name: str
    source: str
    task: str

    def __len__(self) -> int:
        return self.end - self.start


class Bundle:
    """Frames/state/action arrays for one prepared bundle.

    Frames stay memory mapped: ``can-all`` is roughly 1 GB of pixels and every
    worker process would otherwise pay for its own copy.
    """

    def __init__(self, bundle_dir: Path):
        bundle_dir = Path(bundle_dir).expanduser().resolve()
        manifest_path = bundle_dir / "bundle.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{manifest_path} not found. Run src/policy/data_prep.py first."
            )
        self.dir = bundle_dir
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.name: str = self.manifest["name"]
        self.image_size: int = int(self.manifest["image_size"])
        self.fps: float = float(self.manifest["fps"])
        self.joint_names: list[str] = list(self.manifest["joint_names"])

        self.frames: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.episodes: list[EpisodeRef] = []
        for shard_index, relative in enumerate(self.manifest["shards"]):
            shard_dir = (bundle_dir.parent / relative).resolve()
            shard_meta = json.loads((shard_dir / "shard.json").read_text(encoding="utf-8"))
            self.frames.append(np.load(shard_dir / "frames.npy", mmap_mode="r"))
            self.states.append(np.load(shard_dir / "state.npy"))
            self.actions.append(np.load(shard_dir / "action.npy"))
            for episode in shard_meta["episodes"]:
                self.episodes.append(
                    EpisodeRef(
                        shard=shard_index,
                        start=int(episode["start"]),
                        end=int(episode["end"]),
                        name=episode["name"],
                        source=episode["source"],
                        task=episode["task"],
                    )
                )

    @property
    def state_dim(self) -> int:
        return int(self.states[0].shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions[0].shape[1])

    def split_episodes(self, val_episodes: int, seed: int) -> tuple[list[EpisodeRef], list[EpisodeRef]]:
        """Hold out ``val_episodes`` episodes per source, chosen deterministically."""
        if val_episodes <= 0:
            return list(self.episodes), []
        rng = np.random.default_rng(seed)
        train: list[EpisodeRef] = []
        val: list[EpisodeRef] = []
        for source in sorted({episode.source for episode in self.episodes}):
            group = [episode for episode in self.episodes if episode.source == source]
            if val_episodes >= len(group):
                raise ValueError(
                    f"--val-episodes={val_episodes} leaves no training episodes for source {source}"
                )
            order = rng.permutation(len(group))
            held = {int(index) for index in order[:val_episodes]}
            for index, episode in enumerate(group):
                (val if index in held else train).append(episode)
        return train, val


def augment_frame(frame: np.ndarray, rng: np.random.Generator, crop_scale: float, color_jitter: float,
                  crop_box: tuple[int, int, int] | None = None) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Random square crop + photometric jitter. Returns the frame and its crop box.

    Passing a ``crop_box`` back in reuses that crop instead of drawing a new one,
    which is how every frame of one sample - the observation stack and, under
    ``policy.goal``, the goal frame - ends up sharing a single crop.
    """
    size = frame.shape[0]
    if crop_box is None:
        side = int(round(size * float(rng.uniform(crop_scale, 1.0))))
        side = max(1, min(size, side))
        top = int(rng.integers(0, size - side + 1))
        left = int(rng.integers(0, size - side + 1))
        crop_box = (top, left, side)
    top, left, side = crop_box
    out = frame[top : top + side, left : left + side]
    if side != size:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_LINEAR)
    if color_jitter > 0.0:
        gain = 1.0 + float(rng.uniform(-color_jitter, color_jitter))
        bias = 255.0 * float(rng.uniform(-color_jitter, color_jitter)) * 0.5
        out = np.clip(out.astype(np.float32) * gain + bias, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(out), crop_box


class ChunkDataset(Dataset):
    """Yields ``n_obs_steps`` frames + states and the ``horizon`` actions that follow.

    Observation steps are the frames at ``[t - n_obs_steps + 1, ..., t]``, clamped
    at the episode start. The action chunk starts at ``t``, so the newest
    observation and the first action describe the same instant.

        dataset = ChunkDataset(bundle, episodes, horizon=16, n_obs_steps=2)
        sample["image"]    # (2, 3, H, W) uint8
        sample["action"]   # (16, action_dim)
    """

    def __init__(
        self,
        bundle: Bundle,
        episodes: list[EpisodeRef],
        *,
        horizon: int,
        n_obs_steps: int = 2,
        drop_n_last_frames: int = 0,
        augment: bool = False,
        crop_scale: float = 0.9,
        color_jitter: float = 0.1,
        seed: int = 0,
    ):
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1, got {n_obs_steps}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        self.bundle = bundle
        self.episodes = episodes
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.drop_n_last_frames = int(drop_n_last_frames)
        self.augment = bool(augment)
        self.crop_scale = float(crop_scale)
        self.color_jitter = float(color_jitter)
        self.seed = int(seed)
        self._draws = 0

        # Every frame is a valid start except the last few, whose chunks would be
        # mostly copy-padding. At least one sample per episode is always kept.
        self.index: list[tuple[int, int]] = [
            (episode_index, offset)
            for episode_index, episode in enumerate(episodes)
            for offset in range(max(1, len(episode) - self.drop_n_last_frames))
        ]

    def __len__(self) -> int:
        return len(self.index)

    def _rng(self, item: int) -> np.random.Generator:
        # Seeded from the worker seed plus a per-worker call counter: reproducible
        # for a given worker layout, and still a fresh crop every epoch even with
        # persistent workers, where torch's base seed never changes.
        self._draws += 1
        base = torch.initial_seed() % (2**31)
        entropy = self.seed * 1_000_003 + base + item * 7919 + self._draws * 104_729
        return np.random.default_rng(entropy % (2**32))

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_index, offset = self.index[item]
        episode = self.episodes[episode_index]
        frames = self.bundle.frames[episode.shard]
        states = self.bundle.states[episode.shard]
        actions = self.bundle.actions[episode.shard]
        rng = self._rng(item)
        last_offset = len(episode) - 1

        history = np.clip(offset - np.arange(self.n_obs_steps - 1, -1, -1), 0, last_offset)
        crop_box = None
        images = []
        for step_offset in history:
            # np.array copies: the shard is a read-only memmap shared by every worker.
            frame = np.array(frames[episode.start + int(step_offset)])
            if self.augment:
                # One crop box for the whole stack. The camera is static, so a
                # per-frame crop would inject apparent motion that is not real and
                # teach the policy to read velocity out of the augmentation.
                frame, crop_box = augment_frame(frame, rng, self.crop_scale, self.color_jitter, crop_box)
            images.append(torch.from_numpy(frame).permute(2, 0, 1).contiguous())

        chunk_offsets = np.minimum(offset + np.arange(self.horizon), last_offset)
        is_pad = (offset + np.arange(self.horizon)) > last_offset

        return {
            "image": torch.stack(images),  # (T, 3, H, W) uint8
            "state": torch.from_numpy(states[episode.start + history].copy()),  # (T, D)
            "action": torch.from_numpy(actions[episode.start + chunk_offsets].copy()),
            "action_is_pad": torch.from_numpy(is_pad),
            **self.extra_items(episode, offset, rng, crop_box),
        }

    def extra_items(
        self,
        episode: EpisodeRef,
        offset: int,
        rng: np.random.Generator,
        crop_box: tuple[int, int, int] | None,
    ) -> dict[str, torch.Tensor]:
        """Extra fields a subclass adds to the sample. Empty here.

        ``crop_box`` is the crop the observation stack used, so an extra frame
        from the same (static) camera can be cropped identically instead of
        appearing shifted relative to what the policy is looking at. It is None
        when augmentation is off. ``policy.goal.GoalChunkDataset`` uses this to
        attach the goal frame.
        """
        return {}


def default_drop_n_last_frames(horizon: int, n_action_steps: int, n_obs_steps: int) -> int:
    """The reference implementation's rule, clamped to something non-negative.

    Dropping the tail avoids training mostly on copy-padded chunks, which
    otherwise teach the policy to stop moving near the end of every episode.
    """
    return max(0, horizon - n_action_steps - n_obs_steps + 1)


def compute_stats(
    bundle: Bundle,
    episodes: list[EpisodeRef],
    chunk_size: int,
    angular_dims: tuple[int, ...] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Normalisation statistics over the training episodes only.

    Both delta conventions are described, because they have very different scales:

    * ``delta``     - ``action[t+k] - state[t]``, one shared anchor, grows with k
    * ``increment`` - ``action[t+k] - action[t+k-1]`` (``state[t]`` for k=0), a
      per-step rate whose distribution barely changes with k
    """
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    increments: list[np.ndarray] = []
    if angular_dims is None:
        angular_dims = tuple(
            index for index, name in enumerate(bundle.joint_names)
            if name in {"eef.rx", "eef.ry", "eef.rz"}
        )
    for episode in episodes:
        state = bundle.states[episode.shard][episode.start : episode.end]
        action = bundle.actions[episode.shard][episode.start : episode.end]
        states.append(state)
        actions.append(action)
        length = len(state)
        offsets = np.minimum(np.arange(length)[:, None] + np.arange(chunk_size)[None, :], length - 1)
        chunk_actions = action[offsets]  # (length, chunk_size, dim)
        deltas.append(
            wrap_numpy(chunk_actions - state[:, None, :], angular_dims).reshape(-1, state.shape[1])
        )
        previous = np.concatenate([state[:, None, :], chunk_actions[:, :-1, :]], axis=1)
        increments.append(
            wrap_numpy(chunk_actions - previous, angular_dims).reshape(-1, state.shape[1])
        )

    def describe(values: np.ndarray) -> dict[str, list[float]]:
        return {
            "mean": values.mean(axis=0).astype(np.float64).tolist(),
            "std": values.std(axis=0).astype(np.float64).tolist(),
            "min": values.min(axis=0).astype(np.float64).tolist(),
            "max": values.max(axis=0).astype(np.float64).tolist(),
        }

    return {
        "state": describe(np.concatenate(states, axis=0)),
        "action": describe(np.concatenate(actions, axis=0)),
        "delta": describe(np.concatenate(deltas, axis=0)),
        "increment": describe(np.concatenate(increments, axis=0)),
    }
