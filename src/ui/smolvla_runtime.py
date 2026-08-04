"""Inference adapter for the local SmolVLA fine-tuning script.

The local trainer deliberately keeps normalization outside LeRobot's policy
processor.  This adapter mirrors that contract and exposes the small runner
interface used by ``policy_runtime``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

EEF_NAMES = ("eef.x", "eef.y", "eef.z", "eef.rx", "eef.ry", "eef.rz", "gripper.pos")
JOINT_NAMES = tuple(f"joint_{index}.pos" for index in range(1, 7)) + ("gripper.pos",)


class SmolVLARunner:
    """Run a locally fine-tuned SmolVLA checkpoint on Piper EEF data."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        action_steps: int | None = None,
        task: str = "",
    ) -> None:
        import torch

        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        except ImportError as exc:
            raise RuntimeError(
                "SmolVLA inference requires LeRobot with the smolvla extra installed."
            ) from exc

        self._torch = torch
        self._language_tokens_key = OBS_LANGUAGE_TOKENS
        self._language_mask_key = OBS_LANGUAGE_ATTENTION_MASK
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.local_config = self._read_local_config()
        self.action_names = self._read_action_names()
        requested_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        self.task = str(task)
        self.policy = SmolVLAPolicy.from_pretrained(self.checkpoint_path)
        self.policy.to(self.device).eval()
        if action_steps is not None:
            if not 1 <= action_steps <= int(self.policy.config.chunk_size):
                raise ValueError("SmolVLA action_steps must be within the trained horizon.")
            self.policy.config.n_action_steps = int(action_steps)
        self._state_mean = self._tensor("state_mean")
        self._state_std = self._tensor("state_std")
        self._action_mean = self._tensor("action_mean")
        self._action_std = self._tensor("action_std")
        if any(value.numel() != len(EEF_NAMES) for value in (
            self._state_mean, self._state_std, self._action_mean, self._action_std
        )):
            raise ValueError("SmolVLA local statistics must contain seven EEF values.")
        self._token_cache: tuple[Any, Any] | None = None
        self._goal: Any | None = None

    def _read_local_config(self) -> dict[str, Any]:
        path = self.checkpoint_path / "smolvla_local_config.json"
        if not path.is_file():
            raise ValueError(f"SmolVLA checkpoint is missing {path.name}: {self.checkpoint_path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object in {path}")
        return payload

    def _read_action_names(self) -> tuple[str, ...]:
        configured = self.local_config.get("action_names")
        if isinstance(configured, list) and len(configured) == 7:
            return tuple(str(name) for name in configured)
        bundle = self.local_config.get("bundle")
        if bundle:
            manifest_path = Path(str(bundle)) / "bundle.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                names = manifest.get("joint_names") or manifest.get("action_names")
                if isinstance(names, list) and len(names) == 7:
                    return tuple(str(name) for name in names)
            except (OSError, json.JSONDecodeError):
                pass
        # Older local configs did not persist feature names. Their EEF runs
        # are identifiable by the current training convention; joint-trained
        # runs can still be selected as joint targets in the UI.
        return EEF_NAMES

    def _tensor(self, key: str) -> Any:
        values = self.local_config.get(key)
        if not isinstance(values, list):
            raise ValueError(f"SmolVLA local config is missing {key!r}")
        return self._torch.tensor(values, dtype=self._torch.float32, device=self.device)

    def _image(self, frame_bgr: Any) -> Any:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("SmolVLA expects a three-channel camera frame.")
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        return self._torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)

    def _language(self) -> tuple[Any, Any]:
        if self._token_cache is not None:
            return self._token_cache
        tokenizer = self.policy.model.vlm_with_expert.processor.tokenizer
        encoded = tokenizer(
            [self.task],
            padding=True,
            truncation=True,
            max_length=self.policy.config.tokenizer_max_length,
            return_tensors="pt",
        )
        self._token_cache = (
            encoded.input_ids.to(self.device),
            encoded.attention_mask.bool().to(self.device),
        )
        return self._token_cache

    def set_goal(self, frame_bgr: Any | None) -> None:
        self._goal = None if frame_bgr is None else self._image(frame_bgr)
        self.reset()

    @property
    def has_goal(self) -> bool:
        return self._goal is not None

    @property
    def queued_actions(self) -> int:
        queues = getattr(self.policy, "_queues", {})
        queue = queues.get("action")
        return len(queue) if queue is not None else 0

    def reset(self) -> None:
        self.policy.reset()

    def warmup(self, frame_shape: tuple[int, int] = (480, 640)) -> float:
        # Loading occurs before the browser selects a generated goal. Use a
        # placeholder only to initialize CUDA/model kernels, then discard it.
        previous_goal = self._goal
        height, width = frame_shape
        self._goal = self._torch.zeros((3, height, width), dtype=self._torch.float32, device=self.device)
        state = np.zeros(len(EEF_NAMES), dtype=np.float32)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        started = time.perf_counter()
        try:
            self.select_action(frame, state)
        finally:
            self._goal = previous_goal
            self.reset()
        return time.perf_counter() - started

    def select_action(self, frame_bgr: Any, state: Any) -> np.ndarray:
        if self._goal is None:
            raise RuntimeError("SmolVLA requires a goal frame. Generate and aim at a goal first.")
        current = self._image(frame_bgr).unsqueeze(0)
        goal = self._goal.unsqueeze(0)
        measured = self._torch.as_tensor(state, dtype=self._torch.float32, device=self.device).view(1, -1)
        measured = (measured - self._state_mean) / self._state_std
        tokens, mask = self._language()
        batch = {
            "observation.images.current": current,
            "observation.images.goal": goal,
            "observation.state": measured,
            self._language_tokens_key: tokens,
            self._language_mask_key: mask,
        }
        action = self.policy.select_action(batch)
        action = action.reshape(-1).to(self.device)
        action = action * self._action_std + self._action_mean
        return action.detach().cpu().numpy().astype(np.float32, copy=False)

    def action_dict(self, action: np.ndarray) -> dict[str, float]:
        if len(action) != len(self.action_names):
            raise ValueError(
                f"SmolVLA returned {len(action)} values; expected {len(self.action_names)} values."
            )
        return {name: float(value) for name, value in zip(self.action_names, action, strict=True)}

    def describe(self) -> dict[str, Any]:
        config = self.policy.config
        return {
            "checkpoint": str(self.checkpoint_path),
            "run_name": self.checkpoint_path.name,
            "step": None,
            "policy": "smolvla",
            "objective": "flow",
            "chunk_size": int(config.chunk_size),
            "action_steps": int(config.n_action_steps),
            "num_inference_steps": None,
            "image_size": 224,
            "vision": "SmolVLM",
            "pool": "action-expert",
            "action_repr": "absolute",
            "delta_mode": "absolute",
            "absolute_dims": (),
            "joint_names": self.action_names,
            "fps": 20.0,
            "device": str(self.device),
            "use_ema": False,
            "n_obs_steps": 1,
            "goal_conditioned": True,
            "has_goal": self.has_goal,
            "supports_guidance": False,
            "guidance_modes": (),
            "guidance_weight": 1.0,
            "guidance_mode": "full",
        }
