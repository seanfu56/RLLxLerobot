"""Dataset adapter for the local ``piper-data/dataset`` shard format."""

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
        goal_selection: str = "future_uniform",
        goal_window: int = 10,
        goal_frames: int = 4,
        goal_frame_index: int = 2,
        task_prefix: str = "",
        include_goal: bool = True,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.bundle = bundle
        self.episodes = episodes
        self.horizon = int(horizon)
        self.random_goal = bool(random_goal)
        if goal_selection not in {"future_uniform", "uniform4", "tail"}:
            raise ValueError("goal_selection must be future_uniform, uniform4, or tail")
        if goal_window < 1:
            raise ValueError("goal_window must be >= 1")
        if goal_frames < 2 or not 0 <= goal_frame_index < goal_frames:
            raise ValueError("goal_frame_index must be within goal_frames >= 2")
        self.goal_selection = goal_selection
        self.goal_window = int(goal_window)
        self.goal_frames = int(goal_frames)
        self.goal_frame_index = int(goal_frame_index)
        self.task_prefix = task_prefix
        self.include_goal = bool(include_goal)
        self.index = [
            (episode_index, offset)
            for episode_index, episode in enumerate(episodes)
            for offset in range(
                max(
                    0,
                    len(episode) - self.horizon
                    if self.goal_selection == "future_uniform"
                    else len(episode) - 1,
                )
            )
        ]

    def __len__(self) -> int:
        return len(self.index)

    def _goal_offset(self, offset: int, last: int, item: int) -> int:
        rng = np.random.default_rng((torch.initial_seed() + item * 1_000_003 + offset * 9_973) % 2**32)
        if self.goal_selection == "future_uniform":
            first = offset + self.horizon
            return last if not self.random_goal else int(rng.integers(first, last + 1))

        first_end = max(0, last - self.goal_window + 1)
        endpoint = last if not self.random_goal else int(rng.integers(first_end, last + 1))
        if self.goal_selection == "tail":
            return endpoint
        divisor = self.goal_frames - 1
        return (self.goal_frame_index * endpoint + divisor // 2) // divisor

    def __getitem__(self, item: int) -> dict[str, Any]:
        episode_index, offset = self.index[item]
        episode = self.episodes[episode_index]
        frames = self.bundle.frames[episode.shard]
        states = self.bundle.states[episode.shard]
        actions = self.bundle.actions[episode.shard]
        last = len(episode) - 1
        goal = self._goal_offset(offset, last, item) if self.include_goal else None
        action_offsets = np.minimum(offset + np.arange(self.horizon), last)
        is_pad = (offset + np.arange(self.horizon)) > last

        def image(frame_offset: int) -> torch.Tensor:
            frame = np.asarray(frames[episode.start + frame_offset]).copy()
            return torch.from_numpy(frame).permute(2, 0, 1).contiguous().float() / 255.0

        task = f"{self.task_prefix}{episode.task}".strip()
        sample = {
            "observation.images.current": image(offset),
            "observation.state": torch.from_numpy(states[episode.start + offset].copy()),
            "action": torch.from_numpy(actions[episode.start + action_offsets].copy()),
            "actions_id_pad": torch.from_numpy(is_pad),
            "task": task,
        }
        if goal is not None:
            sample["observation.images.goal"] = image(goal)
        return sample
