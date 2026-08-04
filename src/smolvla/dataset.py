"""Dataset adapter for the local ``piper-data/dataset`` shard format.

SmolVLA consumes one current image, one goal image, the current state and an
action chunk.  The goal is sampled from the strict future interval
``(current_frame, episode_last_frame]`` on every training access.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from policy.datasets import Bundle, EpisodeRef


class SmolVLADataset(Dataset):
    def __init__(
        self,
        bundle: Bundle,
        episodes: list[EpisodeRef],
        *,
        horizon: int,
        random_goal: bool = True,
        task_prefix: str = "",
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.bundle = bundle
        self.episodes = episodes
        self.horizon = int(horizon)
        self.random_goal = bool(random_goal)
        self.task_prefix = task_prefix
        # A strict future goal is undefined at the final frame, so do not make
        # a synthetic self-goal there. Starts immediately before the end still
        # receive the normal tail-padded action chunk.
        self.index = [
            (episode_index, offset)
            for episode_index, episode in enumerate(episodes)
            for offset in range(max(0, len(episode) - 1))
        ]

    def __len__(self) -> int:
        return len(self.index)

    def _goal_offset(self, offset: int, last: int, item: int) -> int:
        if not self.random_goal:
            return last
        rng = np.random.default_rng((torch.initial_seed() + item * 1_000_003 + offset * 9_973) % 2**32)
        return int(rng.integers(offset + 1, last + 1))

    def __getitem__(self, item: int) -> dict[str, Any]:
        episode_index, offset = self.index[item]
        episode = self.episodes[episode_index]
        frames = self.bundle.frames[episode.shard]
        states = self.bundle.states[episode.shard]
        actions = self.bundle.actions[episode.shard]
        last = len(episode) - 1
        goal = self._goal_offset(offset, last, item)
        action_offsets = np.minimum(offset + np.arange(self.horizon), last)
        is_pad = (offset + np.arange(self.horizon)) > last

        def image(frame_offset: int) -> torch.Tensor:
            frame = np.asarray(frames[episode.start + frame_offset]).copy()
            return torch.from_numpy(frame).permute(2, 0, 1).contiguous().float() / 255.0

        task = f"{self.task_prefix}{episode.task}".strip()
        return {
            "observation.images.current": image(offset),
            "observation.images.goal": image(goal),
            "observation.state": torch.from_numpy(states[episode.start + offset].copy()),
            "action": torch.from_numpy(actions[episode.start + action_offsets].copy()),
            "actions_id_pad": torch.from_numpy(is_pad),
            "task": task,
        }
