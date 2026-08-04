"""Load a trained checkpoint and turn camera frames into joint commands.

    runner = PolicyRunner("models/simple/S_pick_bar/best.pt")
    action = runner.select_action(frame_bgr, state)   # (7,) joint targets

A chunk is predicted only when the queue runs dry, so ``select_action`` costs a
sampler pass once every ``action_steps`` calls and a dictionary pop otherwise.

The one subtlety is the observation history: the policy conditions on the last
``n_obs_steps`` frames, so ``select_action`` records every frame it is given even
on the steps where it only pops a queued action. Skipping that would make the
history span a whole execution horizon instead of consecutive control steps, and
the velocity the second frame is there to provide would be wrong.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from policy.config import GUIDANCE_MODES, PolicyConfig
    from policy.data_prep import preprocess_frame
    from policy.model import ChunkPolicy
else:
    from .config import GUIDANCE_MODES, PolicyConfig
    from .data_prep import preprocess_frame
    from .model import ChunkPolicy


class PolicyRunner:
    # The module the checkpoint's weights are loaded into. Subclasses that
    # extend the architecture override this; ``policy.goal.load_runner`` picks
    # the right subclass from the config so callers never have to.
    policy_class = ChunkPolicy

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        use_ema: bool = True,
        action_steps: int | None = None,
        num_inference_steps: int | None = None,
        guidance_weight: float | None = None,
        guidance_mode: str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.config = PolicyConfig.from_dict(payload["config"])
        self.joint_names: list[str] = list(payload.get("joint_names", []))
        self.fps: float = float(payload.get("fps", 20.0))
        self.step: int | None = payload.get("step")
        self.use_ema = bool(use_ema and payload.get("ema"))

        # A goal-conditioned checkpoint has a wider conditioning vector and an
        # extra encoder input, so loading it here would fail on a state-dict key
        # mismatch. Say what it actually needs instead.
        if self.config.goal_conditioned and self.policy_class is ChunkPolicy:
            raise ValueError(
                f"{self.checkpoint_path.parent.name} is goal-conditioned and needs a goal frame "
                "at every step. Load it with policy.goal.load_runner() or GoalPolicyRunner, "
                "which takes one."
            )

        self.device = device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu"
        self.policy = self.policy_class(self.config)
        # ``null_cond`` arrived with classifier-free guidance; checkpoints written
        # before it are otherwise complete and still perfectly usable at weight
        # 1.0, so tolerate exactly that key and keep every other mismatch loud.
        loaded = self.policy.load_state_dict(payload["model"], strict=False)
        missing = [key for key in loaded.missing_keys if key != "null_cond"]
        if missing or loaded.unexpected_keys:
            raise ValueError(
                f"Checkpoint does not match the model: missing {missing[:5]}, "
                f"unexpected {list(loaded.unexpected_keys)[:5]}"
            )
        if self.use_ema:
            # EMA only shadows trainable parameters, so buffers stay as loaded.
            result = self.policy.load_state_dict(payload["ema"], strict=False)
            if result.unexpected_keys:
                raise ValueError(f"EMA weights do not match the model: {result.unexpected_keys[:5]}")
        self.policy.to(self.device).eval()

        if num_inference_steps is not None:
            if num_inference_steps < 1:
                raise ValueError("num_inference_steps must be at least 1")
            self.policy.objective.num_inference_steps = int(num_inference_steps)
        self.num_inference_steps: int = int(self.policy.objective.num_inference_steps)

        self.action_steps = int(action_steps or self.config.n_action_steps)
        if not 1 <= self.action_steps <= self.config.horizon:
            raise ValueError(f"action_steps must be in [1, {self.config.horizon}]")

        # Refuse guidance the checkpoint cannot support. Sampling against an
        # untrained null embedding does not fail loudly - it produces confident,
        # wrong actions - so this has to be caught before the arm is enabled.
        self.guidance_mode = str(
            self.config.guidance_mode if guidance_mode is None else guidance_mode
        )
        if self.guidance_mode not in GUIDANCE_MODES:
            raise ValueError(
                f"guidance_mode must be one of {GUIDANCE_MODES}, got {self.guidance_mode!r}"
            )
        self.guidance_weight = float(
            self.config.guidance_weight if guidance_weight is None else guidance_weight
        )
        if self.guidance_weight < 0.0:
            raise ValueError(f"guidance_weight must be >= 0, got {self.guidance_weight}")
        if self.guidance_weight != 1.0:
            reason = self.config.guidance_unavailable_reason(self.guidance_mode)
            if reason is not None:
                raise ValueError(
                    f"{self.run_name}: guidance_weight={self.guidance_weight} in mode "
                    f"{self.guidance_mode!r} {reason}"
                )

        self.queue: deque[np.ndarray] = deque()
        self.frames: deque[torch.Tensor] = deque(maxlen=self.config.n_obs_steps)
        self.states: deque[torch.Tensor] = deque(maxlen=self.config.n_obs_steps)

    # ------------------------------------------------------------------
    # properties the control loop and the UI read
    # ------------------------------------------------------------------

    @property
    def run_name(self) -> str:
        return self.checkpoint_path.parent.name

    @property
    def supports_guidance(self) -> bool:
        """Whether a weight other than 1.0 is usable in the mode now selected."""
        return self.config.guidance_available(self.guidance_mode)

    @property
    def guidance_modes(self) -> tuple[str, ...]:
        """Every mode this checkpoint could be guided in, for the UI to offer."""
        return self.config.guidance_modes

    @property
    def queued_actions(self) -> int:
        return len(self.queue)

    def _to_tensor(self, frame_bgr: np.ndarray) -> torch.Tensor:
        processed = preprocess_frame(frame_bgr, self.config.image_size)
        return torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _observe(self, frame_bgr: np.ndarray, state: np.ndarray) -> None:
        """Record one observation, filling the history on the first call.

        Repeating the first frame matches how training clamps the history at the
        episode start, so the policy sees a familiar "no motion yet" input rather
        than an arbitrary one.
        """
        frame = self._to_tensor(frame_bgr)
        measured = torch.as_tensor(state, dtype=torch.float32).view(1, -1).to(self.device)
        if not self.frames:
            for _ in range(self.config.n_obs_steps):
                self.frames.append(frame)
                self.states.append(measured)
            return
        self.frames.append(frame)
        self.states.append(measured)

    def set_guidance_weight(self, weight: float) -> None:
        """Change the guidance weight between rollouts, with the same checks as loading."""
        weight = float(weight)
        if weight < 0.0:
            raise ValueError(f"guidance_weight must be >= 0, got {weight}")
        if weight != 1.0:
            reason = self.config.guidance_unavailable_reason(self.guidance_mode)
            if reason is not None:
                raise ValueError(f"{self.run_name}: mode {self.guidance_mode!r} {reason}")
        self.guidance_weight = weight
        self.queue.clear()

    def set_guidance_mode(self, mode: str) -> None:
        """Change what the guided branch drops, between rollouts.

        Refused when the current weight could not be applied in the new mode:
        switching to a mode whose null embedding was never trained has to fail
        here rather than at the first step of a rollout.
        """
        mode = str(mode)
        if self.guidance_weight != 1.0:
            reason = self.config.guidance_unavailable_reason(mode)
            if reason is not None:
                raise ValueError(f"{self.run_name}: mode {mode!r} {reason}")
        elif mode not in GUIDANCE_MODES:
            raise ValueError(f"guidance_mode must be one of {GUIDANCE_MODES}, got {mode!r}")
        self.guidance_mode = mode
        self.queue.clear()

    def reset(self) -> None:
        """Drop the queued chunk and the history so the next call starts fresh."""
        self.queue.clear()
        self.frames.clear()
        self.states.clear()

    def observation_batch(self) -> dict[str, torch.Tensor]:
        """The recorded history as a batch of one. Subclasses add their inputs here."""
        if not self.frames:
            raise RuntimeError("No observation recorded yet; call select_action or predict_chunk first")
        return {
            "image": torch.stack(list(self.frames), dim=1),  # (1, T, 3, H, W)
            "state": torch.stack(list(self.states), dim=1),  # (1, T, D)
        }

    @torch.no_grad()
    def _predict_from_history(self) -> np.ndarray:
        batch = self.observation_batch()
        actions = self.policy.predict(
            batch, guidance_weight=self.guidance_weight, guidance_mode=self.guidance_mode
        )
        return actions[0].float().cpu().numpy()

    def predict_chunk(self, frame_bgr: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Record the observation and predict a full chunk from it."""
        self._observe(frame_bgr, state)
        return self._predict_from_history()

    def select_action(self, frame_bgr: np.ndarray, state: np.ndarray) -> np.ndarray:
        self._observe(frame_bgr, state)
        if not self.queue:
            self.queue.extend(self._predict_from_history()[: self.action_steps])
        return self.queue.popleft()

    def action_dict(self, action: np.ndarray) -> dict[str, float]:
        """Name the joint targets so they can be sent to a LeRobot robot."""
        names = self.joint_names or [f"joint_{index}.pos" for index in range(1, len(action) + 1)]
        if len(names) != len(action):
            raise ValueError(f"Checkpoint names {len(names)} joints but predicted {len(action)}")
        return {name: float(value) for name, value in zip(names, action, strict=True)}

    def warmup(self, frame_shape: tuple[int, int] = (480, 640)) -> float:
        """One throw-away inference so the first live step is not the slowest.

        CUDA context creation and cuDNN autotuning happen here rather than while
        the arm is already moving. The history and queue are cleared afterwards
        so no dummy observation or action can reach the robot.
        """
        height, width = frame_shape
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        state = np.zeros(self.config.state_dim, dtype=np.float32)
        started = time.perf_counter()
        try:
            self.predict_chunk(frame, state)
        finally:
            self.reset()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return time.perf_counter() - started

    def describe(self) -> dict[str, Any]:
        """Plain-data summary for logs and the supervisory UI.

        These keys are what ``src/ui/policy_runtime.py`` turns into a
        ``PolicyInfo``; dropping one breaks loading only at runtime.
        """
        absolute = [
            (self.joint_names[index] if index < len(self.joint_names) else str(index))
            for index in self.config.absolute_dims
        ]
        return {
            "checkpoint": str(self.checkpoint_path),
            "run_name": self.run_name,
            "step": self.step,
            "policy": "simple",
            "objective": self.config.objective,
            "chunk_size": self.config.horizon,
            "action_steps": self.action_steps,
            "num_inference_steps": self.num_inference_steps,
            "image_size": self.config.image_size,
            "vision": "resnet18",
            "pool": "spatial_softmax",
            "action_repr": self.config.action_repr,
            "delta_mode": self.config.delta_mode,
            "absolute_dims": absolute,
            "state_dim": self.config.state_dim,
            "action_dim": self.config.action_dim,
            "joint_names": list(self.joint_names),
            "fps": self.fps,
            "device": self.device,
            "use_ema": self.use_ema,
            # Shown by the supervisory UI.
            "n_obs_steps": self.config.n_obs_steps,
            "down_dims": list(self.config.down_dims),
            "guidance_weight": self.guidance_weight,
            "guidance_mode": self.guidance_mode,
            "guidance_modes": list(self.guidance_modes),
            "supports_guidance": self.supports_guidance,
            "cond_dropout": self.config.cond_dropout,
            "goal_conditioned": self.config.goal_conditioned,
            "goal_selection": self.config.goal_selection,
            "goal_frames": self.config.goal_frames,
            "goal_frame_index": self.config.goal_frame_index,
            "goal_window": self.config.goal_window,
            "goal_dropout": self.config.goal_dropout,
        }
