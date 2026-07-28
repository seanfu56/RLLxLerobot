"""Goal-conditioned variant of the CNN action-chunk policy.

The plain policy answers "given what I see now, what comes next?". A
goal-conditioned one answers "given what I see now *and* a picture of how this
should end, what comes next?". The goal is one more image, encoded by the same
ResNet and appended to the conditioning vector - nothing about the U-Net, the
objective, the action representation or the sampler changes.

    python src/policy/train.py --bundle pick-can-m1 --run-name G_pick_can \
        --goal-conditioned --goal-dropout 0.1

**Which frame is the goal.** One drawn uniformly from the last ``--goal-window``
frames of the episode the sample came from - the last 10 by default. Not the
final frame alone: the arm is still settling over the last half second, so the
tail is a set of pictures that all mean "done", and training against a fresh
draw each epoch stops the policy keying on one exact pixel arrangement. Nothing
in the data has to be labelled for this; the goal is *hindsight*, read off the
demonstration that was already recorded.

**Why bother on a single-task bundle.** Mostly you should not expect much. Every
episode of ``pick-can-m1`` ends in nearly the same picture, so the goal carries
almost no information the policy could not infer from the task itself, and it
will learn to ignore it. Goal conditioning starts paying when one bundle holds
several end states - two cans, two placements, a bundle that merges sessions -
because then the goal is what disambiguates which one is wanted. Train with
``--goal-dropout`` and compare the same checkpoint with and without a goal
(``policy/infer.py --no-goal``) to find out which case you are in; if the two
scores match, the goal is decoration.

**What it costs.** One extra ResNet pass per prediction - the goal is encoded
alongside the observation frames, so it is roughly a third more vision work at
``n_obs_steps=2``, and nothing at all in the sampler loop, which is where the
latency actually is. The goal is encoded once per prediction regardless of the
sampler step count. It is *not* cached across predictions, even though it is
constant for a whole rollout: a cached feature is one more thing that can go
stale, and the failure mode - the arm driving confidently towards a goal the
operator has already replaced - is worse than the few milliseconds it saves.

Layout: the dataset draws the goal frame, the policy encodes it, the runner
holds one for the whole rollout.

    GoalChunkDataset    adds ``goal_image`` to each sample
    GoalChunkPolicy     appends its feature to the conditioning vector
    GoalPolicyRunner    ``set_goal(frame)``, then the usual ``select_action``
    load_runner         picks the runner class a checkpoint needs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from policy.config import PolicyConfig
    from policy.datasets import ChunkDataset, EpisodeRef, augment_frame
    from policy.inference import PolicyRunner
    from policy.model import ChunkPolicy
else:
    from .config import PolicyConfig
    from .datasets import ChunkDataset, EpisodeRef, augment_frame
    from .inference import PolicyRunner
    from .model import ChunkPolicy

DEFAULT_GOAL_WINDOW = 10

__all__ = [
    "DEFAULT_GOAL_WINDOW", "GoalChunkDataset", "GoalChunkPolicy", "GoalPolicyRunner",
    "goal_offset_range", "is_goal_conditioned", "load_runner",
]


def goal_offset_range(episode_length: int, window: int) -> tuple[int, int]:
    """Half-open range of episode offsets a goal may be drawn from.

    The last ``window`` frames, clamped for episodes shorter than the window -
    a 6-frame episode with ``window=10`` offers all six rather than an empty or
    negative range.
    """
    if episode_length < 1:
        raise ValueError(f"episode_length must be >= 1, got {episode_length}")
    if window < 1:
        raise ValueError(f"goal window must be >= 1, got {window}")
    return episode_length - min(window, episode_length), episode_length


class GoalChunkDataset(ChunkDataset):
    """``ChunkDataset`` plus a ``goal_image`` (3, H, W) drawn from the episode tail.

    ``random_goal`` is the train/validation difference. Training redraws the goal
    every time a sample is visited, so the policy sees the whole tail window;
    validation pins it to the final frame, because a goal that moves between
    evaluations makes ``val_loss`` noisy and ``val_loss`` is what selects
    ``best.pt``.

    Under augmentation the goal is cropped with the observation stack's crop box
    rather than its own. The camera is static, so a crop is a global shift of the
    scene: cropping the goal differently would move the target relative to what
    the policy is looking at and teach it an offset that is not there.
    """

    def __init__(
        self,
        *args: Any,
        goal_window: int = DEFAULT_GOAL_WINDOW,
        random_goal: bool = True,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if goal_window < 1:
            raise ValueError(f"goal_window must be >= 1, got {goal_window}")
        self.goal_window = int(goal_window)
        self.random_goal = bool(random_goal)

    def goal_offset(self, episode: EpisodeRef, rng: np.random.Generator) -> int:
        start, end = goal_offset_range(len(episode), self.goal_window)
        if not self.random_goal:
            return end - 1
        return int(rng.integers(start, end))

    def extra_items(
        self,
        episode: EpisodeRef,
        offset: int,
        rng: np.random.Generator,
        crop_box: tuple[int, int, int] | None,
    ) -> dict[str, torch.Tensor]:
        frames = self.bundle.frames[episode.shard]
        # np.array copies: the shard is a read-only memmap shared by every worker.
        frame = np.array(frames[episode.start + self.goal_offset(episode, rng)])
        if self.augment:
            frame, _ = augment_frame(frame, rng, self.crop_scale, self.color_jitter, crop_box)
        return {"goal_image": torch.from_numpy(frame).permute(2, 0, 1).contiguous()}


class GoalChunkPolicy(ChunkPolicy):
    """``ChunkPolicy`` conditioned on a goal frame as well as the observation.

    The goal shares the observation encoder. A second ResNet would double the
    vision parameters to learn the same notion of "where things are" twice, from
    the same few thousand frames and from images drawn from the same
    distribution - the goal frame is simply a frame of the episode.

    ``goal_dropout`` swaps the goal feature for a learned null embedding on that
    fraction of training samples. Two things follow: the checkpoint can be run
    with no goal at all, which is both a fallback and the honest way to measure
    what the goal is contributing, and the goal is never so load-bearing that a
    slightly wrong one derails the chunk. It is orthogonal to ``cond_dropout``,
    which drops the *whole* conditioning vector, goal included, and is what
    classifier-free guidance needs.
    """

    def __init__(self, config: PolicyConfig):
        if not config.goal_conditioned:
            raise ValueError(
                "GoalChunkPolicy needs goal_conditioned=True; use ChunkPolicy for a plain run."
            )
        super().__init__(config)
        # Stands in for "no goal given". Sized like one encoder feature, and
        # always allocated so the state dict does not depend on goal_dropout.
        self.null_goal = nn.Parameter(torch.randn(1, self.rgb_encoder.feature_dim) * 0.02)

    def _global_cond_dim(self) -> int:
        return super()._global_cond_dim() + self.rgb_encoder.feature_dim

    def encode_goal(self, batch: dict[str, Tensor], batch_size: int, dtype: torch.dtype) -> Tensor:
        """Goal feature [B, feature_dim], or the null embedding when there is no goal."""
        goal = batch.get("goal_image")
        if goal is None:
            if self.config.goal_dropout <= 0.0:
                raise ValueError(
                    "This policy was trained with goal_dropout=0, so its null-goal embedding is "
                    "untrained and sampling without a goal would condition on noise. Pass "
                    "goal_image, or retrain with --goal-dropout 0.1."
                )
            return self.null_goal.to(dtype).expand(batch_size, -1)

        features = self.rgb_encoder(goal)
        if self.training and self.config.goal_dropout > 0.0:
            keep = (torch.rand(batch_size, 1, device=features.device) >= self.config.goal_dropout)
            features = torch.where(keep, features, self.null_goal.to(features.dtype))
        return features.to(dtype)

    def encode_context(self, batch: dict[str, Tensor]) -> Tensor:
        observation = self.observation_features(batch)
        goal = self.encode_goal(batch, observation.shape[0], observation.dtype)
        # Goal dropout first, then cond_dropout over the concatenation: dropping
        # the observation has to drop the goal with it, or the "unconditional"
        # branch would still know where the episode ends.
        return self.apply_cond_dropout(torch.cat([observation, goal], dim=-1))


class GoalPolicyRunner(PolicyRunner):
    """``PolicyRunner`` that carries a goal frame for the length of a rollout.

        runner = GoalPolicyRunner("models/simple/G_pick_can/best.pt")
        runner.set_goal(goal_frame_bgr)              # once, before starting
        action = runner.select_action(frame, state)  # as usual

    The goal survives ``reset``: it describes the task, not the episode in
    progress. Changing it mid-rollout is allowed and drops the queued chunk, so
    actions planned for the old goal cannot reach the arm afterwards.
    """

    policy_class = GoalChunkPolicy

    def __init__(self, checkpoint_path: str | Path, *, goal: np.ndarray | None = None, **kwargs: Any):
        # Checked before the weights are touched, so the error names the fix
        # rather than reporting a state-dict mismatch.
        if not is_goal_conditioned(checkpoint_path):
            raise ValueError(
                f"{Path(checkpoint_path).parent.name} was not trained with --goal-conditioned; "
                "use PolicyRunner."
            )
        self.goal: torch.Tensor | None = None
        super().__init__(checkpoint_path, **kwargs)
        if goal is not None:
            self.set_goal(goal)

    @property
    def has_goal(self) -> bool:
        return self.goal is not None

    def set_goal(self, frame_bgr: np.ndarray) -> None:
        """Set the goal from a raw camera frame, cropped the way training was."""
        self.goal = self._to_tensor(frame_bgr)
        self.queue.clear()

    def clear_goal(self) -> None:
        self.goal = None
        self.queue.clear()

    def observation_batch(self) -> dict[str, torch.Tensor]:
        batch = super().observation_batch()
        if self.goal is None and self.config.goal_dropout <= 0.0:
            raise RuntimeError(
                f"{self.run_name} needs a goal frame: call set_goal() before the first step. "
                "(Only a checkpoint trained with --goal-dropout can run without one.)"
            )
        if self.goal is not None:
            batch["goal_image"] = self.goal
        return batch

    def warmup(self, frame_shape: tuple[int, int] = (480, 640)) -> float:
        """One throw-away inference, with a blank goal if none has been set yet.

        Warming up should not force the operator to choose a goal first, and the
        observation frame is blank here too, so the pair is consistent. The
        temporary goal is removed afterwards - it must not be the one a rollout
        silently runs against.
        """
        if self.has_goal:
            return super().warmup(frame_shape)
        self.set_goal(np.zeros((*frame_shape, 3), dtype=np.uint8))
        try:
            return super().warmup(frame_shape)
        finally:
            self.clear_goal()

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "has_goal": self.has_goal}


def is_goal_conditioned(checkpoint_path: str | Path) -> bool:
    """Whether a checkpoint needs a goal frame, without building the model.

    ``run.json`` sits next to every checkpoint ``train.py`` writes and answers
    this in a few kilobytes; reading the ``.pt`` is the fallback for a checkpoint
    that has been copied away from its run directory.
    """
    path = Path(checkpoint_path)
    run_json = path.parent / "run.json"
    if run_json.is_file():
        config = json.loads(run_json.read_text(encoding="utf-8")).get("config", {})
        return bool(config.get("goal_conditioned", False))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return bool(payload["config"].get("goal_conditioned", False))


def load_runner(checkpoint_path: str | Path, **kwargs: Any) -> PolicyRunner:
    """Build the runner a checkpoint needs, goal-conditioned or not.

    ``goal=`` is accepted only by the goal-conditioned branch, so passing one to
    a plain checkpoint is an error rather than a silently ignored argument.
    """
    if is_goal_conditioned(checkpoint_path):
        return GoalPolicyRunner(checkpoint_path, **kwargs)
    if kwargs.pop("goal", None) is not None:
        raise ValueError(
            f"{Path(checkpoint_path).parent.name} was not trained with --goal-conditioned, "
            "so it has nowhere to put a goal frame."
        )
    return PolicyRunner(checkpoint_path, **kwargs)
