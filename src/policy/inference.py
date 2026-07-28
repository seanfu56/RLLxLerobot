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
    from policy.config import PolicyConfig
    from policy.data_prep import preprocess_frame
    from policy.model import ChunkPolicy
else:
    from .config import PolicyConfig
    from .data_prep import preprocess_frame
    from .model import ChunkPolicy


class PolicyRunner:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        use_ema: bool = True,
        action_steps: int | None = None,
        num_inference_steps: int | None = None,
        guidance_weight: float | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.config = PolicyConfig.from_dict(payload["config"])
        self.joint_names: list[str] = list(payload.get("joint_names", []))
        self.fps: float = float(payload.get("fps", 20.0))
        self.step: int | None = payload.get("step")
        self.use_ema = bool(use_ema and payload.get("ema"))

        self.device = device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu"
        self.policy = ChunkPolicy(self.config)
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
        self.guidance_weight = float(
            self.config.guidance_weight if guidance_weight is None else guidance_weight
        )
        if self.guidance_weight < 0.0:
            raise ValueError(f"guidance_weight must be >= 0, got {self.guidance_weight}")
        if self.guidance_weight != 1.0 and self.config.cond_dropout <= 0.0:
            raise ValueError(
                f"{self.run_name} was trained with cond_dropout=0, so it has no unconditional "
                f"branch and guidance_weight={self.guidance_weight} would extrapolate from an "
                "untrained embedding. Retrain with --cond-dropout 0.1, or use guidance_weight=1.0."
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
        return self.config.cond_dropout > 0.0

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
        if weight != 1.0 and not self.supports_guidance:
            raise ValueError(
                f"{self.run_name} was trained with cond_dropout=0 and has no unconditional branch."
            )
        self.guidance_weight = weight
        self.queue.clear()

    def reset(self) -> None:
        """Drop the queued chunk and the history so the next call starts fresh."""
        self.queue.clear()
        self.frames.clear()
        self.states.clear()

    @torch.no_grad()
    def _predict_from_history(self) -> np.ndarray:
        if not self.frames:
            raise RuntimeError("No observation recorded yet; call select_action or predict_chunk first")
        batch = {
            "image": torch.stack(list(self.frames), dim=1),  # (1, T, 3, H, W)
            "state": torch.stack(list(self.states), dim=1),  # (1, T, D)
        }
        return self.policy.predict(batch, guidance_weight=self.guidance_weight)[0].float().cpu().numpy()

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
            "cond_dropout": self.config.cond_dropout,
        }
