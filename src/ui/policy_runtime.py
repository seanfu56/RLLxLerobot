"""UI-independent runtime for closed-loop policy inference on the Piper.

This is the inference twin of ``teleop_runtime.py`` and deliberately reuses its
camera system, preview publisher, video writer and episode format. The only
structural difference is where an action comes from: teleoperation reads a
ROBOTIS leader arm, this module runs a trained ``src/policy`` checkpoint over
the overhead camera frame and the measured joint state.

Two properties matter more here than in teleoperation:

* **The arm is only driven while a rollout is running.** The control worker
  always reads observations, so the preview, telemetry and watchdog stay live
  while idle, but ``robot.send_action`` is called only between Start and Stop.
* **The checkpoint is loaded in the hardware process.** ``load_policy`` is a
  separate command from ``connect`` so an operator can pay the CUDA and weight
  startup cost, and switch checkpoints, without a robot attached.

Torch and LeRobot imports are lazy so the state machine can be tested on a
machine with neither installed.
"""

from __future__ import annotations

import atexit
import csv
import importlib
import json
import math
import multiprocessing
import shutil
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

try:
    from ui.teleop_runtime import (
        ACTION_KEYS,
        CAMERA_KEY,
        DEFAULT_STATIC_PATH,
        OBSERVATION_COLUMNS,
        REST_STATE_DEG,
        CameraBackend,
        CameraSettings,
        TimingSummary,
        VideoResult,
        _default_video_converter,
        _FrameReference,
        _next_episode_index,
        _normalise_camera_source,
        _ObservationCameraSystem,
        _plain_scalars,
        _timing_summary,
    )
except ModuleNotFoundError as exc:  # running with src/ui itself on sys.path
    if exc.name != "ui":
        raise
    from teleop_runtime import (
        ACTION_KEYS,
        CAMERA_KEY,
        DEFAULT_STATIC_PATH,
        OBSERVATION_COLUMNS,
        REST_STATE_DEG,
        CameraBackend,
        CameraSettings,
        TimingSummary,
        VideoResult,
        _default_video_converter,
        _FrameReference,
        _next_episode_index,
        _normalise_camera_source,
        _ObservationCameraSystem,
        _plain_scalars,
        _timing_summary,
    )

DEFAULT_PREVIEW_PATH = DEFAULT_STATIC_PATH.with_name("policy_live.jpg")
DEFAULT_MODEL_ROOT = Path("models/sweeps")
DEFAULT_OUTPUT_DIR = Path("outputs/policy_eval")

# What an operator can say about a finished rollout. The label is the whole
# point of running evaluation episodes: it is the success rate, and it is the
# preference signal src/finetune turns into DPO pairs.
OUTCOMES = ("success", "failure")


class PolicyState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class PolicySettings:
    """Everything needed to build a ``PolicyRunner`` in the hardware process."""

    checkpoint: Path
    action_steps: int = 4
    num_inference_steps: int | None = None
    device: str = "cuda"
    use_ema: bool = True
    # Classifier-free guidance. None keeps whatever the checkpoint was trained
    # with; 1.0 is plain conditional sampling.
    guidance_weight: float | None = None
    # Goal frame for a --goal-conditioned checkpoint. Read in the hardware
    # process, so this is a path rather than an array.
    goal_image: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", Path(self.checkpoint).expanduser())
        if not self.checkpoint.is_file():
            raise ValueError(f"Checkpoint not found: {self.checkpoint}")
        if self.goal_image is not None:
            object.__setattr__(self, "goal_image", Path(self.goal_image).expanduser())
            if not self.goal_image.is_file():
                raise ValueError(f"Goal image not found: {self.goal_image}")
        if self.action_steps < 1:
            raise ValueError("action_steps must be at least 1.")
        if self.num_inference_steps is not None and self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1.")
        if self.guidance_weight is not None and self.guidance_weight < 0:
            raise ValueError("guidance_weight must not be negative.")
        if not self.device:
            raise ValueError("Device must not be empty.")


@dataclass(frozen=True)
class PolicyInfo:
    """Picklable description of the checkpoint the hardware process loaded."""

    checkpoint: str
    run_name: str
    step: int | None
    policy: str
    objective: str
    chunk_size: int
    action_steps: int
    num_inference_steps: int | None
    image_size: int
    vision: str
    pool: str
    action_repr: str
    delta_mode: str
    absolute_dims: tuple[str, ...]
    joint_names: tuple[str, ...]
    fps: float
    device: str
    use_ema: bool
    warmup_s: float | None = None
    # Observation frames the policy conditions on, and the U-Net stage widths.
    # Only the simple policy reports more than one frame or any stage widths.
    n_obs_steps: int = 1
    down_dims: tuple[int, ...] = ()
    guidance_weight: float = 1.0
    # True only when the checkpoint was trained with conditioning dropout, i.e.
    # when it actually has an unconditional branch to guide away from.
    supports_guidance: bool = False
    # Goal conditioning (see src/policy/goal.py). ``has_goal`` is False when a
    # goal-conditioned checkpoint is running against its learned null embedding,
    # which only a --goal-dropout run can do.
    goal_conditioned: bool = False
    has_goal: bool = False

    def headline(self) -> str:
        # Both objectives sample iteratively: DDIM steps for diffusion, Euler
        # steps for flow matching.
        sampler = f"{self.num_inference_steps} sampler steps"
        observing = f", observing {self.n_obs_steps}" if self.n_obs_steps > 1 else ""
        guided = f" · guidance {self.guidance_weight:g}" if self.guidance_weight != 1.0 else ""
        goal = ""
        if self.goal_conditioned:
            goal = " · goal set" if self.has_goal else " · no goal (null embedding)"
        return (
            f"{self.run_name} · {self.policy}/{self.objective} ({sampler}) · "
            f"chunk {self.chunk_size}, executing {self.action_steps}{observing}{guided}{goal}"
        )


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    can_port: str = "piper_left"
    speed_rate: int = 30
    gripper_max_mm: float = 100.0
    # The plugin clamps every command against the previous one. This is the
    # main defence against a bad prediction, so it defaults to on and small.
    max_relative_target: float | None = 2.0
    control_hz: float = 20.0
    watchdog_timeout_s: float = 2.0
    watchdog_startup_timeout_s: float = 5.0
    # Pose the arm homes to on connect and rests in on disconnect. Evaluation
    # is only comparable from the pose the demonstrations started in.
    start_pose: Mapping[str, float] | None = None
    home_on_connect: bool = True
    camera: CameraSettings = field(default_factory=CameraSettings)
    preview_path: Path = field(default_factory=lambda: DEFAULT_PREVIEW_PATH)

    def __post_init__(self) -> None:
        object.__setattr__(self, "preview_path", Path(self.preview_path))
        if self.start_pose is not None:
            pose = {str(key): float(value) for key, value in self.start_pose.items()}
            missing = [key for key in ACTION_KEYS if key not in pose]
            if missing:
                raise ValueError(f"Start pose is missing {', '.join(missing)}.")
            object.__setattr__(self, "start_pose", pose)
        elif self.home_on_connect:
            raise ValueError("Homing on connect needs a start pose.")
        if not self.can_port:
            raise ValueError("CAN port must not be empty.")
        if not 1 <= self.speed_rate <= 100:
            raise ValueError("Piper speed rate must be between 1 and 100.")
        if self.gripper_max_mm not in (70.0, 100.0):
            raise ValueError("Piper gripper maximum must be 70 or 100 mm.")
        if self.control_hz <= 0:
            raise ValueError("Control rate must be positive.")
        if self.max_relative_target is not None and self.max_relative_target <= 0:
            raise ValueError("Maximum relative target must be positive or unset.")
        if self.watchdog_timeout_s <= 0 or self.watchdog_startup_timeout_s <= 0:
            raise ValueError("Watchdog timeouts must be positive.")


@dataclass(frozen=True)
class RolloutConfig:
    output_dir: Path = DEFAULT_OUTPUT_DIR
    task: str = "pick up can"
    duration_s: float | None = 20.0
    record: bool = True
    # Where a policy leaves the arm is wherever the episode ended, which is not
    # a pose the next rollout can start from. Returning closes the loop so
    # consecutive trials are comparable and the per-step clamp starts fresh.
    return_to_start: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError("Rollout duration must be positive or unset.")


@dataclass(frozen=True)
class RolloutSnapshot:
    episode_index: int
    output_dir: str
    task: str
    samples: int
    video_frames: int
    video_dropped_frames: int
    duration_s: float
    running: bool
    recorded: bool
    # "success", "failure", or None while the operator has not judged it yet.
    outcome: str | None = None


@dataclass(frozen=True)
class PolicySnapshot:
    state: PolicyState
    policy: PolicyInfo | None
    driving: bool
    returning_to_start: bool
    target_control_hz: float
    effective_control_hz: float
    minimum_effective_hz: float
    missed_control_deadlines: int
    last_command_age_s: float | None
    control_stage: str
    control_stage_age_s: float | None
    control_sequence: int
    latest_action: Mapping[str, float]
    latest_observation: Mapping[str, float]
    control_period: TimingSummary
    observation_latency: TimingSummary
    inference_latency: TimingSummary
    robot_send_latency: TimingSummary
    plans: int
    queued_actions: int
    camera_frame_age_s: float | None
    camera_capture_hz: float
    camera_frames: int
    preview_encode_ms: float
    video_queue_depth: int
    video_dropped_frames: int
    rollout_elapsed_s: float
    rollout_duration_s: float | None
    rollout_samples: int
    missed_rollout_frames: int
    draft: RolloutSnapshot | None
    last_error: str | None
    error_source: str | None


@dataclass
class PolicyHardwareBundle:
    robot: Any
    emergency_disable: Callable[[], None] | None = None
    # Smooth, rate-limited move to a complete pose. Only ever called from the
    # control thread, so it never competes with the policy for the CAN bus.
    move_to_pose: Callable[[Mapping[str, float]], None] | None = None


@dataclass
class _ControlSample:
    sequence: int
    timestamp_s: float
    observation_timestamp_s: float
    action: Mapping[str, float]
    observation: Mapping[str, float]
    frame: _FrameReference | None = None


@dataclass
class _RolloutDraft:
    config: RolloutConfig
    episode_index: int
    pending_dir: Path | None
    avi_path: Path | None
    started_s: float
    stopped_s: float | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    missed_frames: int = 0
    video: VideoResult | None = None
    running: bool = True
    outcome: str | None = None


def _default_hardware_factory(config: PolicyRuntimeConfig) -> PolicyHardwareBundle:
    """Build the Piper follower exactly as the LeRobot evaluation script does."""
    try:
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots import make_robot_from_config
        from lerobot.utils.import_utils import register_third_party_plugins
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is unavailable. Activate the repository environment and install "
            "the Piper plugin before connecting."
        ) from exc

    register_third_party_plugins()
    try:
        piper_module = importlib.import_module("lerobot_robot_piper.config_piper_follower")
    except ImportError as exc:
        raise RuntimeError(
            "The Piper follower plugin must be installed in the active Python environment."
        ) from exc

    camera_source = _normalise_camera_source(config.camera.device)
    if isinstance(camera_source, str):
        camera_source = Path(camera_source)
    camera_config = OpenCVCameraConfig(
        index_or_path=camera_source,
        fps=int(round(config.camera.fps)),
        width=config.camera.width,
        height=config.camera.height,
        color_mode="bgr",
        fourcc=config.camera.fourcc,
    )
    start_pose = dict(config.start_pose) if config.start_pose else None
    robot_config = piper_module.PiperFollowerConfig(
        can_port=config.can_port,
        unit="deg",
        speed_rate=config.speed_rate,
        gripper_max_mm=config.gripper_max_mm,
        max_relative_target=config.max_relative_target,
        go_home_on_connect=bool(config.home_on_connect and start_pose),
        home_position_deg=start_pose or dict(REST_STATE_DEG),
        rest_position_deg=start_pose or dict(REST_STATE_DEG),
        cameras={CAMERA_KEY: camera_config},
    )
    robot = make_robot_from_config(robot_config)

    def emergency_disable() -> None:
        piper = getattr(robot, "piper", None)
        if piper is None:
            return
        piper.DisableArm()
        try:
            piper.GripperCtrl(0, 0, 0x00, 0)
        finally:
            for camera in getattr(robot, "cameras", {}).values():
                try:
                    camera.disconnect()
                except BaseException:
                    pass
            if hasattr(robot, "_is_connected"):
                robot._is_connected = False

    def move_to_pose(pose: Mapping[str, float]) -> None:
        # The plugin's own homing and rest trajectories: a 100 Hz smoothstep at
        # 30 deg/s that verifies the measured endpoint. Reimplementing it here
        # would duplicate the joint clamping and the settle check.
        mover = getattr(robot, "_smooth_move_to_pose", None)
        if mover is None:
            raise RuntimeError(
                "The installed Piper plugin has no _smooth_move_to_pose; it cannot "
                "return the arm to the start pose."
            )
        mover(dict(pose), pose_name="start")

    return PolicyHardwareBundle(
        robot=robot,
        emergency_disable=emergency_disable,
        move_to_pose=move_to_pose,
    )


class _PolicyDriver:
    """Adapts ``PolicyRunner`` to the control loop and counts its work."""

    def __init__(self, runner: Any, info: PolicyInfo) -> None:
        self.runner = runner
        self.info = info
        self._plans = 0

    @property
    def plans(self) -> int:
        return self._plans

    @property
    def queued_actions(self) -> int:
        return int(getattr(self.runner, "queued_actions", 0))

    def reset(self) -> None:
        self.runner.reset()

    def set_guidance_weight(self, weight: float) -> None:
        """Retune guidance in place, without reloading the weights.

        The runner drops its queued chunk, so the actions produced under the old
        weight cannot reach the arm after the change.
        """
        setter = getattr(self.runner, "set_guidance_weight", None)
        if setter is None or not self.info.supports_guidance:
            raise RuntimeError(
                f"{self.info.run_name} has no unconditional branch; it was trained without "
                "conditioning dropout, so classifier-free guidance is not available."
            )
        setter(weight)
        self.info = replace(self.info, guidance_weight=float(weight))

    def set_goal(self, frame_bgr: Any | None) -> None:
        """Point the policy at a goal frame, or take its goal away.

        ``GoalPolicyRunner.set_goal`` drops the queued chunk, so actions planned
        for the previous goal cannot reach the arm after the change.
        """
        if not self.info.goal_conditioned:
            raise RuntimeError(
                f"{self.info.run_name} was not trained with --goal-conditioned, so it has "
                "nowhere to put a goal frame."
            )
        if frame_bgr is None:
            self.runner.clear_goal()
        else:
            self.runner.set_goal(frame_bgr)
        self.info = replace(self.info, has_goal=bool(getattr(self.runner, "has_goal", False)))

    def act(
        self, frame: Any, state: Any
    ) -> tuple[dict[str, float], float | None]:
        """Next joint command, and the sampler latency when it re-planned."""
        if self.queued_actions:
            return self.runner.action_dict(self.runner.select_action(frame, state)), None
        started = time.perf_counter()
        action = self.runner.select_action(frame, state)
        elapsed = time.perf_counter() - started
        self._plans += 1
        return self.runner.action_dict(action), elapsed


def _read_goal_image(path: Path) -> Any:
    """Load a goal frame as the raw BGR the runner's own crop expects."""
    import cv2

    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not read goal image: {path}")
    return frame


def _build_runner(settings: PolicySettings) -> Any:
    """Import torch and build the runner; only ever called in the child process.

    ``load_runner`` picks the goal-conditioned runner for a goal-conditioned
    checkpoint and the plain one otherwise, and refuses a goal frame that has
    nowhere to go.
    """
    try:
        from policy.goal import load_runner
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from policy.goal import load_runner

    extra = {}
    if settings.goal_image is not None:
        extra["goal"] = _read_goal_image(settings.goal_image)

    return load_runner(
        settings.checkpoint,
        device=settings.device,
        use_ema=settings.use_ema,
        action_steps=settings.action_steps,
        num_inference_steps=settings.num_inference_steps,
        guidance_weight=settings.guidance_weight,
        **extra,
    )


def _default_policy_factory(settings: PolicySettings) -> _PolicyDriver:
    """Import torch and build the runner; only ever called in the child process."""
    runner = _build_runner(settings)
    warmup_s = runner.warmup()
    description = runner.describe()
    info = PolicyInfo(
        checkpoint=description["checkpoint"],
        run_name=description["run_name"],
        step=description["step"],
        policy=description["policy"],
        objective=description["objective"],
        chunk_size=description["chunk_size"],
        action_steps=description["action_steps"],
        num_inference_steps=description["num_inference_steps"],
        image_size=description["image_size"],
        vision=description["vision"],
        pool=description["pool"],
        action_repr=description["action_repr"],
        delta_mode=description["delta_mode"],
        absolute_dims=tuple(description["absolute_dims"]),
        joint_names=tuple(description["joint_names"]),
        fps=description["fps"],
        device=description["device"],
        use_ema=description["use_ema"],
        warmup_s=warmup_s,
        n_obs_steps=int(description.get("n_obs_steps", 1)),
        down_dims=tuple(description.get("down_dims") or ()),
        guidance_weight=float(description.get("guidance_weight", 1.0)),
        supports_guidance=float(description.get("cond_dropout") or 0.0) > 0.0,
        goal_conditioned=bool(description.get("goal_conditioned", False)),
        has_goal=bool(description.get("has_goal", False)),
    )
    return _PolicyDriver(runner, info)


@dataclass
class _PendingMove:
    """A pose move queued for the control thread, with its completion result."""

    pose: Mapping[str, float]
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class _PolicyControlWorker:
    """One loop: read the robot, optionally act, send, publish telemetry.

    While ``driving`` is false the loop still reads observations so the preview
    and watchdog stay alive, but nothing is sent to the arm.

    A queued pose move runs on this same thread between iterations, which is
    what keeps the arm to a single writer: the policy and the return-to-start
    trajectory can never be on the CAN bus at the same time.
    """

    def __init__(
        self,
        hardware: PolicyHardwareBundle,
        target_hz: float,
        get_driver: Callable[[], _PolicyDriver | None],
        on_observation: Callable[[Mapping[str, Any], float], _FrameReference | None],
        on_sample: Callable[..., None],
        on_progress: Callable[[str], None],
        on_error: Callable[[str, BaseException], None],
        on_long_operation: Callable[[bool], None] = lambda active: None,
    ) -> None:
        self._hardware = hardware
        self._period_s = 1.0 / target_hz
        self._get_driver = get_driver
        self._on_observation = on_observation
        self._on_sample = on_sample
        self._on_progress = on_progress
        self._on_error = on_error
        self._on_long_operation = on_long_operation
        self._driving = threading.Event()
        self._stop = threading.Event()
        self._pending_lock = threading.Lock()
        self._pending_move: _PendingMove | None = None
        self._thread = threading.Thread(target=self._run, name="policy-control", daemon=True)

    def start(self) -> None:
        self._thread.start()

    @property
    def driving(self) -> bool:
        return self._driving.is_set()

    def set_driving(self, enabled: bool) -> None:
        if enabled:
            self._driving.set()
        else:
            self._driving.clear()

    def request_stop(self) -> None:
        self._driving.clear()
        self._stop.set()
        with self._pending_lock:
            pending, self._pending_move = self._pending_move, None
        if pending is not None:
            pending.error = RuntimeError("Control worker stopped before the move ran.")
            pending.done.set()

    def stop(self, timeout_s: float = 3.0) -> bool:
        self.request_stop()
        if threading.current_thread() is self._thread:
            return True
        self._thread.join(timeout=timeout_s)
        return not self._thread.is_alive()

    def request_move(self, pose: Mapping[str, float]) -> _PendingMove:
        """Queue a pose move for the control thread. Never call while driving."""
        if self._hardware.move_to_pose is None:
            raise RuntimeError("This hardware bundle cannot move the arm to a pose.")
        move = _PendingMove(pose=dict(pose))
        with self._pending_lock:
            if self._pending_move is not None:
                raise RuntimeError("A pose move is already queued.")
            if self._stop.is_set():
                raise RuntimeError("Control worker is stopping.")
            self._pending_move = move
        return move

    def _run_pending_move(self) -> None:
        with self._pending_lock:
            move, self._pending_move = self._pending_move, None
        if move is None:
            return
        # The move commands the arm for seconds at a time and publishes no
        # control samples, so the staleness watchdog is suspended around it.
        self._on_long_operation(True)
        self._on_progress("returning to start pose")
        try:
            if self._driving.is_set():
                raise RuntimeError("Refusing to move the arm while the policy is driving it.")
            assert self._hardware.move_to_pose is not None
            self._hardware.move_to_pose(move.pose)
        except BaseException as exc:
            move.error = exc
        finally:
            self._on_long_operation(False)
            move.done.set()

    def _run(self) -> None:
        previous_start: float | None = None
        sequence = 0
        self._on_progress("starting")
        while not self._stop.is_set():
            self._run_pending_move()
            if self._stop.is_set():
                break
            loop_start = time.perf_counter()
            period_s = loop_start - previous_start if previous_start is not None else None
            previous_start = loop_start

            try:
                self._on_progress("robot.get_observation")
                observation_started = time.perf_counter()
                observation = self._hardware.robot.get_observation()
                observation_latency_s = time.perf_counter() - observation_started
                observation_timestamp_s = time.perf_counter()
                frame_reference = self._on_observation(observation, observation_timestamp_s)

                driver = self._get_driver()
                action: dict[str, float] = {}
                inference_s: float | None = None
                send_latency_s: float | None = None
                driving = self._driving.is_set() and driver is not None
                if driving:
                    assert driver is not None
                    if self._stop.is_set():
                        break
                    self._on_progress("policy.select_action")
                    frame = observation.get(CAMERA_KEY)
                    if frame is None:
                        raise RuntimeError(
                            f"The robot observation carries no {CAMERA_KEY!r} camera frame."
                        )
                    state = [float(observation[key]) for key in ACTION_KEYS]
                    action, inference_s = driver.act(frame, state)

                    self._on_progress("robot.send_action")
                    send_started = time.perf_counter()
                    sent = self._hardware.robot.send_action(action)
                    send_latency_s = time.perf_counter() - send_started
                    if sent is not None:
                        action = _plain_scalars(sent)

                sequence += 1
                self._on_sample(
                    sequence=sequence,
                    timestamp_s=time.perf_counter(),
                    observation_timestamp_s=observation_timestamp_s,
                    action=action,
                    observation=_plain_scalars(observation),
                    frame=frame_reference,
                    driving=driving,
                    period_s=period_s,
                    observation_latency_s=observation_latency_s,
                    inference_latency_s=inference_s,
                    send_latency_s=send_latency_s,
                    queued_actions=driver.queued_actions if driver is not None else 0,
                    plans=driver.plans if driver is not None else 0,
                )
                self._on_progress("waiting for next deadline")
            except BaseException as exc:
                self._driving.clear()
                self._on_error("control", exc)
                return

            # As in teleoperation, a long iteration is never replayed: the loop
            # sleeps only for the unused remainder and reports the miss.
            work_s = time.perf_counter() - loop_start
            if work_s < self._period_s:
                self._stop.wait(self._period_s - work_s)
            else:
                self._on_sample(missed_deadlines=max(math.floor(work_s / self._period_s), 1))


def _default_camera_factory(
    config: PolicyRuntimeConfig,
    on_error: Callable[[str, BaseException], None],
) -> CameraBackend:
    return _ObservationCameraSystem(config.camera, config.preview_path, on_error)


class PolicyRuntime:
    """Process-wide owner of the policy, the Piper, and rollout recordings."""

    def __init__(
        self,
        hardware_factory: Callable[
            [PolicyRuntimeConfig], PolicyHardwareBundle
        ] = _default_hardware_factory,
        camera_factory: Callable[
            [PolicyRuntimeConfig, Callable[[str, BaseException], None]], CameraBackend
        ] = _default_camera_factory,
        policy_factory: Callable[[PolicySettings], _PolicyDriver] = _default_policy_factory,
        video_converter: Callable[[Path, Path], None] = _default_video_converter,
        telemetry_window: int = 2000,
    ) -> None:
        self._hardware_factory = hardware_factory
        self._camera_factory = camera_factory
        self._policy_factory = policy_factory
        self._video_converter = video_converter
        self._lock = threading.RLock()
        self._command_lock = threading.RLock()
        self._state = PolicyState.DISCONNECTED
        self._config: PolicyRuntimeConfig | None = None
        self._hardware: PolicyHardwareBundle | None = None
        self._camera: CameraBackend | None = None
        self._control: _PolicyControlWorker | None = None
        self._driver: _PolicyDriver | None = None
        self._draft: _RolloutDraft | None = None
        self._latest_control: _ControlSample | None = None
        self._last_error: str | None = None
        self._error_source: str | None = None
        self._connected_at_s: float | None = None
        self._control_stage = "not started"
        self._control_stage_started_s: float | None = None
        self._control_periods: deque[float] = deque(maxlen=telemetry_window)
        self._observation_latencies: deque[float] = deque(maxlen=telemetry_window)
        self._inference_latencies: deque[float] = deque(maxlen=telemetry_window)
        self._send_latencies: deque[float] = deque(maxlen=telemetry_window)
        self._missed_control_deadlines = 0
        self._queued_actions = 0
        self._plans = 0
        self._long_operation = False
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # policy
    # ------------------------------------------------------------------

    @property
    def state(self) -> PolicyState:
        with self._lock:
            return self._state

    def load_policy(self, settings: PolicySettings) -> PolicySnapshot:
        """Load or replace the checkpoint. Refused while the arm is driven."""
        with self._command_lock:
            with self._lock:
                if self._state is PolicyState.RUNNING:
                    raise RuntimeError("Stop the rollout before changing the checkpoint.")
                if self._state in {PolicyState.CONNECTING, PolicyState.STOPPING}:
                    raise RuntimeError(f"Cannot load a policy while runtime is {self._state.value}.")
                if self._draft is not None:
                    raise RuntimeError("Save or discard the recorded rollout first.")
            driver = self._policy_factory(settings)
            with self._lock:
                self._driver = driver
                self._plans = 0
                self._queued_actions = 0
                self._inference_latencies.clear()
                if self._error_source == "policy":
                    self._last_error = None
                    self._error_source = None
            return self.get_snapshot()

    def unload_policy(self) -> PolicySnapshot:
        with self._command_lock:
            with self._lock:
                if self._state is PolicyState.RUNNING:
                    raise RuntimeError("Stop the rollout before unloading the checkpoint.")
                self._driver = None
            return self.get_snapshot()

    # ------------------------------------------------------------------
    # hardware lifecycle
    # ------------------------------------------------------------------

    def connect(self, config: PolicyRuntimeConfig) -> PolicySnapshot:
        with self._command_lock:
            with self._lock:
                if self._state in {PolicyState.IDLE, PolicyState.RUNNING}:
                    return self.get_snapshot()
                if self._state is not PolicyState.DISCONNECTED:
                    raise RuntimeError(f"Cannot connect while runtime is {self._state.value}.")
                self._state = PolicyState.CONNECTING
                self._config = config
                self._last_error = None
                self._error_source = None
                self._reset_telemetry_locked()

            hardware: PolicyHardwareBundle | None = None
            camera: CameraBackend | None = None
            robot_connected = False
            try:
                hardware = self._hardware_factory(config)
                camera = self._camera_factory(config, self._worker_failed)
                camera.start()
                hardware.robot.connect()
                robot_connected = True

                control = _PolicyControlWorker(
                    hardware,
                    config.control_hz,
                    self._get_driver,
                    camera.publish_observation,
                    self._publish_control,
                    self._publish_control_progress,
                    self._worker_failed,
                    self._set_long_operation,
                )
                with self._lock:
                    if self._state is PolicyState.ERROR:
                        raise RuntimeError(self._last_error or "A worker failed during connection.")
                    self._hardware = hardware
                    self._camera = camera
                    self._control = control
                    self._connected_at_s = time.perf_counter()
                    self._state = PolicyState.IDLE
                control.start()
                self._start_watchdog()
                return self.get_snapshot()
            except BaseException as exc:
                if camera is not None:
                    try:
                        camera.stop()
                    except BaseException:
                        pass
                if hardware is not None and robot_connected:
                    try:
                        hardware.robot.disconnect()
                    except BaseException:
                        pass
                with self._lock:
                    self._hardware = None
                    self._camera = None
                    self._control = None
                    self._state = PolicyState.DISCONNECTED
                    self._last_error = str(exc)
                    self._error_source = "connect"
                raise

    def disconnect(self) -> PolicySnapshot:
        """Stop driving and control first, then let the plugin run its rest move."""
        with self._command_lock:
            if self.state is PolicyState.RUNNING:
                # The plugin's own rest move to the same pose follows below.
                self.stop_rollout(return_to_start=False)
            with self._lock:
                if self._state is PolicyState.DISCONNECTED:
                    return self.get_snapshot()
                if self._state in {PolicyState.CONNECTING, PolicyState.STOPPING}:
                    raise RuntimeError(f"Cannot disconnect while runtime is {self._state.value}.")
                self._state = PolicyState.STOPPING
                control = self._control
                camera = self._camera
                hardware = self._hardware

            self._stop_watchdog()
            control_stopped = control is None or control.stop()

            warnings: list[str] = []
            if camera is not None:
                try:
                    camera.stop()
                except BaseException as exc:
                    warnings.append(f"camera: {exc}")

            if not control_stopped:
                with self._lock:
                    self._state = PolicyState.ERROR
                    self._last_error = (
                        "Control worker did not stop; safe rest was not started to avoid "
                        "concurrent robot commands. Use the physical emergency stop."
                    )
                    self._error_source = "disconnect"
                raise RuntimeError(self._last_error)

            if hardware is not None:
                try:
                    hardware.robot.disconnect()
                except BaseException as exc:
                    with self._lock:
                        self._state = PolicyState.ERROR
                        self._last_error = f"Robot safe disconnect failed: {exc}"
                        self._error_source = "disconnect"
                    raise RuntimeError(self._last_error) from exc

            with self._lock:
                self._hardware = None
                self._camera = None
                self._control = None
                self._connected_at_s = None
                self._latest_control = None
                self._state = PolicyState.DISCONNECTED
                if warnings:
                    self._last_error = "; ".join(warnings)
                    self._error_source = "disconnect_warning"
            return self.get_snapshot()

    def emergency_stop(self) -> PolicySnapshot:
        """Stop scheduling and disable the Piper immediately, without a rest move."""
        with self._command_lock:
            with self._lock:
                if self._state is PolicyState.DISCONNECTED:
                    return self.get_snapshot()
                self._state = PolicyState.STOPPING
                control = self._control
                camera = self._camera
                hardware = self._hardware

            self._stop_watchdog()
            if control is not None:
                control.request_stop()
                control.stop(timeout_s=1.0)
            if camera is not None:
                try:
                    self._finish_draft(camera.stop_recording())
                except BaseException:
                    pass
                try:
                    camera.stop()
                except BaseException:
                    pass
            if hardware is not None and hardware.emergency_disable is not None:
                try:
                    hardware.emergency_disable()
                except BaseException:
                    pass

            with self._lock:
                self._hardware = None
                self._camera = None
                self._control = None
                self._connected_at_s = None
                self._latest_control = None
                self._state = PolicyState.DISCONNECTED
                self._last_error = (
                    "Software emergency stop requested; verify the physical robot state."
                )
                self._error_source = "emergency_stop"
            return self.get_snapshot()

    # ------------------------------------------------------------------
    # rollouts
    # ------------------------------------------------------------------

    def start_rollout(self, config: RolloutConfig) -> PolicySnapshot:
        """Begin driving the arm from the policy, optionally recording it."""
        with self._command_lock:
            with self._lock:
                if self._state is not PolicyState.IDLE:
                    raise RuntimeError("A rollout can only start while the runtime is IDLE.")
                if self._draft is not None:
                    raise RuntimeError("Save or discard the previous rollout before starting another.")
                camera = self._camera
                control = self._control
                driver = self._driver
                runtime_config = self._config
            if driver is None:
                raise RuntimeError("Load a checkpoint before starting a rollout.")
            if camera is None or control is None or runtime_config is None:
                raise RuntimeError("Hardware is not connected.")

            pending_dir: Path | None = None
            avi_path: Path | None = None
            episode_index = 0
            if config.record:
                config.output_dir.mkdir(parents=True, exist_ok=True)
                episode_index = _next_episode_index(config.output_dir)
                pending_dir = config.output_dir / f".episode_{episode_index:03d}.pending"
                pending_dir.mkdir(parents=False, exist_ok=False)
                avi_path = pending_dir / "video.avi"

            started_s = time.perf_counter()
            draft = _RolloutDraft(
                config=config,
                episode_index=episode_index,
                pending_dir=pending_dir,
                avi_path=avi_path,
                started_s=started_s,
            )
            try:
                # The chunk queued from an earlier rollout was planned for a
                # different scene; the first command of a rollout must come
                # from the frame in front of the camera now.
                driver.reset()
                if avi_path is not None:
                    # One control step produces one action row and one frame, so
                    # the video plays back at the control rate.
                    camera.start_recording(avi_path, started_s, runtime_config.control_hz)
                with self._lock:
                    self._draft = draft
                    self._state = PolicyState.RUNNING
                control.set_driving(True)
            except BaseException:
                control.set_driving(False)
                try:
                    camera.stop_recording()
                except BaseException:
                    pass
                if pending_dir is not None:
                    shutil.rmtree(pending_dir, ignore_errors=True)
                with self._lock:
                    self._draft = None
                    if self._state is PolicyState.RUNNING:
                        self._state = PolicyState.IDLE
                raise
            return self.get_snapshot()

    def stop_rollout(self, return_to_start: bool | None = None) -> PolicySnapshot:
        """Stop driving, close the recording, and park the arm at the start pose.

        ``return_to_start`` defaults to what the rollout was started with;
        ``disconnect`` passes False because its own rest move follows.
        """
        with self._command_lock:
            with self._lock:
                if self._state is not PolicyState.RUNNING:
                    return self.get_snapshot()
                self._state = PolicyState.STOPPING
                camera = self._camera
                control = self._control
                draft = self._draft

            # Stop commanding before anything else; the arm holds its last target.
            if control is not None:
                control.set_driving(False)
            video = VideoResult(0, 0, (), 0.0)
            error: BaseException | None = None
            if camera is not None:
                try:
                    video = camera.stop_recording()
                except BaseException as exc:
                    error = exc
            self._finish_draft(video)

            if return_to_start is None:
                return_to_start = draft.config.return_to_start if draft is not None else True
            return_error = self._move_to_start_pose() if return_to_start else None

            with self._lock:
                if self._state is PolicyState.STOPPING:
                    self._state = PolicyState.IDLE
                if error is not None:
                    self._last_error = f"Could not finish the recording: {error}"
                    self._error_source = "stop_rollout"
                elif return_error is not None:
                    self._last_error = (
                        f"The arm did not return to the start pose: {return_error} "
                        "Check where it stopped before starting another rollout."
                    )
                    self._error_source = "return_to_start"
            return self.get_snapshot()

    def return_to_start(self) -> PolicySnapshot:
        """Park the arm at the configured start pose. Only while idle."""
        with self._command_lock:
            with self._lock:
                if self._state is not PolicyState.IDLE:
                    raise RuntimeError(
                        f"The arm can only be parked while the runtime is IDLE, not "
                        f"{self._state.value}."
                    )
            error = self._move_to_start_pose()
            with self._lock:
                if error is not None:
                    self._last_error = f"The arm did not return to the start pose: {error}"
                    self._error_source = "return_to_start"
                elif self._error_source == "return_to_start":
                    self._last_error = None
                    self._error_source = None
            return self.get_snapshot()

    def _move_to_start_pose(self, timeout_s: float = 45.0) -> BaseException | None:
        """Run the parking move on the control thread; returns its failure, if any.

        Failing to park is reported rather than raised: the arm is still held by
        the controller, and the operator decides whether to retry or disconnect.
        """
        with self._lock:
            control = self._control
            config = self._config
        if control is None or config is None:
            return RuntimeError("Hardware is not connected.")
        if not config.start_pose:
            return RuntimeError("No start pose is configured for this connection.")
        try:
            move = control.request_move(config.start_pose)
        except BaseException as exc:
            return exc
        if not move.done.wait(timeout_s):
            return TimeoutError(f"The parking move did not finish within {timeout_s:.0f}s.")
        return move.error

    def save_rollout(self) -> Path:
        """Commit the finished rollout in the recorded-episode layout."""
        with self._command_lock:
            with self._lock:
                if self._state is PolicyState.RUNNING:
                    raise RuntimeError("Stop the rollout before saving it.")
                draft = self._draft
                runtime_config = self._config
                policy = self._driver.info if self._driver is not None else None
            if draft is None or draft.running or draft.video is None:
                raise RuntimeError("There is no completed rollout to save.")
            if draft.pending_dir is None:
                raise RuntimeError("This rollout was run without recording.")

            final_dir = draft.config.output_dir / f"episode_{draft.episode_index:03d}"
            if final_dir.exists():
                raise FileExistsError(f"Episode directory already exists: {final_dir}")
            assert draft.avi_path is not None
            self._video_converter(draft.avi_path, draft.pending_dir / "video.mp4")

            frame_timestamps = draft.video.frame_timestamps
            if len(draft.rows) != len(frame_timestamps):
                raise RuntimeError(
                    "Cannot save an unaligned rollout: "
                    f"{len(draft.rows)} actions and {len(frame_timestamps)} frames."
                )

            action_fields = [
                "sample_index",
                "frame_index",
                "sample_timestamp_s",
                "source_control_sequence",
                "source_control_timestamp_s",
                "capture_sequence",
                "frame_timestamp_s",
                "action_frame_delta_ms",
                "inference_latency_ms",
                *ACTION_KEYS,
            ]
            observation_fields = [
                "sample_index",
                "frame_index",
                "source_control_sequence",
                "observation_timestamp_s",
                "source_control_timestamp_s",
                "observation_action_delta_ms",
                *ACTION_KEYS,
            ]
            with (draft.pending_dir / "actions.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=action_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(draft.rows)
            with (draft.pending_dir / "observations.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=observation_fields)
                writer.writeheader()
                writer.writerows(
                    {
                        "sample_index": row["sample_index"],
                        "frame_index": row["frame_index"],
                        "source_control_sequence": row["source_control_sequence"],
                        "observation_timestamp_s": row["observation_timestamp_s"],
                        "source_control_timestamp_s": row["source_control_timestamp_s"],
                        "observation_action_delta_ms": row["observation_action_delta_ms"],
                        **{
                            key: row[column]
                            for key, column in zip(ACTION_KEYS, OBSERVATION_COLUMNS, strict=True)
                        },
                    }
                    for row in draft.rows
                )
            with (draft.pending_dir / "frame_timestamps.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["frame_index", "capture_sequence", "timestamp_s"]
                )
                writer.writeheader()
                writer.writerows(frame_timestamps)

            stopped_s = draft.stopped_s or time.perf_counter()
            control_hz = runtime_config.control_hz if runtime_config else None
            meta = {
                "task": draft.config.task,
                "episode_index": draft.episode_index,
                "source": "policy_rollout",
                # The operator's verdict, or null when the episode was saved
                # unjudged. src/finetune reads this key to build DPO pairs and
                # src/guidance to label its classifier, so it is written even
                # when it is null rather than being left out.
                "outcome": draft.outcome,
                "duration_s": round(stopped_s - draft.started_s, 6),
                "action_sample_hz_target": control_hz,
                "action_samples": len(draft.rows),
                "action_missed_deadlines": draft.missed_frames,
                "camera_capture_fps_target": (
                    runtime_config.camera.fps if runtime_config else None
                ),
                "video_fps_target": control_hz,
                "video_frames": draft.video.frame_count,
                "video_dropped_frames": draft.video.dropped_frames,
                "observation_samples": len(draft.rows),
                "observation_source": (
                    "measured Piper state captured before the paired action was sent"
                ),
                "control_hz_target": control_hz,
                "gripper_max_mm": runtime_config.gripper_max_mm if runtime_config else None,
                "max_relative_target": (
                    runtime_config.max_relative_target if runtime_config else None
                ),
                "start_pose": dict(runtime_config.start_pose)
                if runtime_config and runtime_config.start_pose
                else None,
                "timestamp_clock": (
                    "time.perf_counter monotonic seconds relative to rollout start"
                ),
                "alignment": (
                    "one action and measured observation row per video frame; "
                    "actions.csv and observations.csv frame_index map directly to "
                    "frame_timestamps.csv frame_index"
                ),
                "policy": _policy_meta(policy),
            }
            (draft.pending_dir / "meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            draft.avi_path.unlink(missing_ok=True)
            draft.pending_dir.rename(final_dir)
            with self._lock:
                self._draft = None
            return final_dir

    def set_rollout_outcome(self, outcome: str | None) -> PolicySnapshot:
        """Record the operator's judgement of the finished rollout.

        Only a finished rollout can be judged: while the arm is still driving
        nobody knows yet whether the episode worked, and a label applied mid
        -rollout would be attached to an episode that then went on to fail.
        The label is written into the episode's ``meta.json`` by ``save_rollout``
        and is what turns a directory of trials into a success rate.
        """
        if outcome is not None and outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES} or None, got {outcome!r}")
        with self._command_lock:
            with self._lock:
                if self._state is PolicyState.RUNNING:
                    raise RuntimeError("Stop the rollout before judging it.")
                draft = self._draft
                if draft is None or draft.running:
                    raise RuntimeError("There is no finished rollout to judge.")
                draft.outcome = outcome
            return self.get_snapshot()

    def discard_rollout(self) -> PolicySnapshot:
        with self._command_lock:
            if self.state is PolicyState.RUNNING:
                self.stop_rollout()
            with self._lock:
                draft = self._draft
            if draft is not None and draft.pending_dir is not None:
                shutil.rmtree(draft.pending_dir, ignore_errors=True)
            with self._lock:
                self._draft = None
            return self.get_snapshot()

    def set_preview_enabled(self, enabled: bool) -> PolicySnapshot:
        with self._command_lock:
            with self._lock:
                camera = self._camera
            if camera is not None:
                camera.set_preview_enabled(enabled)
            return self.get_snapshot()

    def set_guidance_weight(self, weight: float) -> PolicySnapshot:
        """Change the guidance weight on the loaded policy between rollouts.

        Refused while the arm is driven: the weight changes what the policy
        commands, and swapping it mid-trajectory makes the rollout impossible to
        interpret and the motion discontinuous.
        """
        with self._command_lock:
            with self._lock:
                if self._state is PolicyState.RUNNING:
                    raise RuntimeError("Stop the rollout before changing the guidance weight.")
                driver = self._driver
            if driver is None:
                raise RuntimeError("Load a checkpoint before setting a guidance weight.")
            driver.set_guidance_weight(float(weight))
            return self.get_snapshot()

    # ------------------------------------------------------------------
    # goal conditioning
    # ------------------------------------------------------------------

    def generate_goal_video(self, config: Any) -> Any:
        """Ask the configured video model what should happen from the current frame.

        The camera frame is read here rather than in the browser: it has to be
        the scene the rollout is about to start from, and a frame the operator
        screenshotted a minute ago is a different scene. Refused while the arm
        is driving - the goal is fixed for a rollout, and generating a new one
        mid-episode would be planning against a scene that has already moved.
        """
        try:
            from ui.video_goal import generate_goal_video
        except ModuleNotFoundError:
            from video_goal import generate_goal_video

        with self._command_lock:
            with self._lock:
                if self._state is PolicyState.RUNNING:
                    raise RuntimeError("Stop the rollout before generating a new goal.")
                if self._state is not PolicyState.IDLE:
                    raise RuntimeError(
                        f"The camera is not running; the runtime is {self._state.value}."
                    )
                camera = self._camera
            if camera is None:
                raise RuntimeError("Hardware is not connected.")
            frame = camera.latest_frame()
            if frame is None:
                raise RuntimeError(
                    "No camera frame has been captured yet. Wait for the live preview."
                )
            return generate_goal_video(config, frame)

    def set_goal_frame(self, frame_png: bytes | None) -> PolicySnapshot:
        """Give the loaded policy its goal, as one PNG-encoded frame.

        A PNG rather than an array because this crosses a process pipe, and
        rather than an index into the generated video because the runtime does
        not keep that video: it is handed to the browser once and the browser
        decides which frame it wants.
        """
        with self._command_lock:
            with self._lock:
                if self._state is PolicyState.RUNNING:
                    raise RuntimeError("Stop the rollout before changing the goal.")
                driver = self._driver
            if driver is None:
                raise RuntimeError("Load a checkpoint before setting a goal.")
            if frame_png is None:
                driver.set_goal(None)
                return self.get_snapshot()

            import cv2
            import numpy as np

            decoded = cv2.imdecode(np.frombuffer(frame_png, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("The goal frame could not be decoded; expected a PNG.")
            driver.set_goal(decoded)
            return self.get_snapshot()

    # ------------------------------------------------------------------
    # telemetry
    # ------------------------------------------------------------------

    def get_snapshot(self) -> PolicySnapshot:
        now = time.perf_counter()
        with self._lock:
            config = self._config
            latest = self._latest_control
            state = self._state
            camera = self._camera
            control = self._control
            draft = self._draft
            driver = self._driver
            periods = list(self._control_periods)
            observation_latencies = list(self._observation_latencies)
            inference_latencies = list(self._inference_latencies)
            send_latencies = list(self._send_latencies)
            missed_control = self._missed_control_deadlines
            queued_actions = self._queued_actions
            plans = self._plans
            long_operation = self._long_operation
            last_error = self._last_error
            error_source = self._error_source
            control_stage = self._control_stage
            control_stage_started_s = self._control_stage_started_s

        camera_snapshot = camera.snapshot() if camera is not None else {}
        draft_snapshot: RolloutSnapshot | None = None
        rollout_elapsed_s = 0.0
        rollout_duration_s: float | None = None
        rollout_samples = 0
        missed_frames = 0
        if draft is not None:
            end_s = now if draft.running else (draft.stopped_s or now)
            rollout_elapsed_s = max(end_s - draft.started_s, 0.0)
            rollout_duration_s = draft.config.duration_s
            rollout_samples = len(draft.rows)
            missed_frames = draft.missed_frames
            video = draft.video
            draft_snapshot = RolloutSnapshot(
                episode_index=draft.episode_index,
                output_dir=str(draft.config.output_dir),
                task=draft.config.task,
                samples=rollout_samples,
                video_frames=(
                    int(camera_snapshot.get("written_frames", 0) or 0)
                    if draft.running
                    else (video.frame_count if video else 0)
                ),
                video_dropped_frames=(
                    int(camera_snapshot.get("dropped_frames", 0) or 0)
                    if draft.running
                    else (video.dropped_frames if video else 0)
                ),
                duration_s=rollout_elapsed_s,
                running=draft.running,
                recorded=draft.pending_dir is not None,
                outcome=draft.outcome,
            )

        average_period = sum(periods) / len(periods) if periods else 0.0
        return PolicySnapshot(
            state=state,
            policy=driver.info if driver is not None else None,
            driving=bool(control is not None and control.driving),
            returning_to_start=long_operation,
            target_control_hz=config.control_hz if config else 0.0,
            effective_control_hz=1.0 / average_period if average_period > 0 else 0.0,
            minimum_effective_hz=1.0 / max(periods) if periods and max(periods) > 0 else 0.0,
            missed_control_deadlines=missed_control,
            last_command_age_s=(now - latest.timestamp_s) if latest is not None else None,
            control_stage=control_stage,
            control_stage_age_s=(
                now - control_stage_started_s if control_stage_started_s is not None else None
            ),
            control_sequence=latest.sequence if latest is not None else 0,
            latest_action=MappingProxyType(dict(latest.action) if latest is not None else {}),
            latest_observation=MappingProxyType(
                dict(latest.observation) if latest is not None else {}
            ),
            control_period=_timing_summary(periods),
            observation_latency=_timing_summary(observation_latencies),
            inference_latency=_timing_summary(inference_latencies),
            robot_send_latency=_timing_summary(send_latencies),
            plans=plans,
            queued_actions=queued_actions,
            camera_frame_age_s=(
                float(camera_snapshot["frame_age_s"])
                if camera_snapshot.get("frame_age_s") is not None
                else None
            ),
            camera_capture_hz=float(camera_snapshot.get("capture_hz", 0.0) or 0.0),
            camera_frames=int(camera_snapshot.get("frame_count", 0) or 0),
            preview_encode_ms=float(camera_snapshot.get("encode_ms", 0.0) or 0.0),
            video_queue_depth=int(camera_snapshot.get("queue_depth", 0) or 0),
            video_dropped_frames=int(camera_snapshot.get("dropped_frames", 0) or 0),
            rollout_elapsed_s=rollout_elapsed_s,
            rollout_duration_s=rollout_duration_s,
            rollout_samples=rollout_samples,
            missed_rollout_frames=missed_frames,
            draft=draft_snapshot,
            last_error=last_error,
            error_source=error_source,
        )

    # ------------------------------------------------------------------
    # worker callbacks
    # ------------------------------------------------------------------

    def _get_driver(self) -> _PolicyDriver | None:
        with self._lock:
            return self._driver

    def _publish_control(self, **values: Any) -> None:
        missed = int(values.get("missed_deadlines", 0))
        if missed:
            with self._lock:
                self._missed_control_deadlines += missed
            return

        sample = _ControlSample(
            sequence=int(values["sequence"]),
            timestamp_s=float(values["timestamp_s"]),
            observation_timestamp_s=float(values["observation_timestamp_s"]),
            action=dict(values["action"]),
            observation=dict(values["observation"]),
            frame=values.get("frame"),
        )
        inference_latency_s = values.get("inference_latency_s")
        with self._lock:
            self._latest_control = sample
            self._queued_actions = int(values.get("queued_actions", 0))
            self._plans = int(values.get("plans", 0))
            period_s = values.get("period_s")
            if period_s is not None:
                self._control_periods.append(float(period_s))
            observation_latency_s = values.get("observation_latency_s")
            if observation_latency_s is not None:
                self._observation_latencies.append(float(observation_latency_s))
            if inference_latency_s is not None:
                self._inference_latencies.append(float(inference_latency_s))
            send_latency_s = values.get("send_latency_s")
            if send_latency_s is not None:
                self._send_latencies.append(float(send_latency_s))
            draft = self._draft
            camera = self._camera

        # Recording happens on the control thread: enqueueing a frame is a
        # queue put, and writing the row here is what keeps action row N and
        # video frame N the same control iteration by construction.
        if values.get("driving") and draft is not None and draft.running:
            self._record_sample(draft, sample, camera, inference_latency_s)

    def _record_sample(
        self,
        draft: _RolloutDraft,
        sample: _ControlSample,
        camera: CameraBackend | None,
        inference_latency_s: float | None,
    ) -> None:
        """One control step becomes one action row and one video frame."""
        if draft.pending_dir is None or camera is None:
            return
        frame = sample.frame
        usable = (
            frame is not None
            and sample.timestamp_s >= draft.started_s
            and frame.timestamp_s >= draft.started_s
            and all(key in sample.action and key in sample.observation for key in ACTION_KEYS)
            and all(
                math.isfinite(sample.action[key]) and math.isfinite(sample.observation[key])
                for key in ACTION_KEYS
                if key in sample.action and key in sample.observation
            )
        )
        index = len(draft.rows)
        if not usable or not camera.record_frame(index, frame):
            draft.missed_frames += 1
            return
        assert frame is not None
        row: dict[str, Any] = {
            "sample_index": index,
            "frame_index": index,
            "sample_timestamp_s": round(sample.timestamp_s - draft.started_s, 6),
            "source_control_sequence": sample.sequence,
            "source_control_timestamp_s": round(sample.timestamp_s - draft.started_s, 6),
            "observation_timestamp_s": round(
                sample.observation_timestamp_s - draft.started_s, 6
            ),
            "observation_action_delta_ms": round(
                (sample.timestamp_s - sample.observation_timestamp_s) * 1000, 3
            ),
            "capture_sequence": frame.capture_sequence,
            "frame_timestamp_s": round(frame.timestamp_s - draft.started_s, 6),
            "action_frame_delta_ms": round(
                (sample.timestamp_s - frame.timestamp_s) * 1000, 3
            ),
            "inference_latency_ms": (
                round(inference_latency_s * 1000, 3) if inference_latency_s is not None else ""
            ),
        }
        row.update({key: sample.action[key] for key in ACTION_KEYS})
        row.update(
            {
                column: sample.observation[key]
                for key, column in zip(ACTION_KEYS, OBSERVATION_COLUMNS, strict=True)
            }
        )
        draft.rows.append(row)

    def _publish_control_progress(self, stage: str) -> None:
        with self._lock:
            self._control_stage = stage
            self._control_stage_started_s = time.perf_counter()

    def _set_long_operation(self, active: bool) -> None:
        """Suspend the staleness watchdog around a deliberate multi-second move."""
        with self._lock:
            self._long_operation = active
            # The move published no control samples. Rebasing the last sample on
            # the way out stops the watchdog firing on the gap it just created.
            if not active and self._latest_control is not None:
                self._latest_control = replace(
                    self._latest_control, timestamp_s=time.perf_counter()
                )

    def _worker_failed(self, source: str, exc: BaseException) -> None:
        with self._lock:
            if self._state in {PolicyState.DISCONNECTED, PolicyState.STOPPING}:
                return
            self._state = PolicyState.ERROR
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._error_source = source
            control = self._control
            camera = self._camera
        # A failing policy or camera must not leave the arm under command.
        if control is not None:
            control.set_driving(False)
            control.request_stop()
        if camera is not None:
            try:
                self._finish_draft(camera.stop_recording())
            except BaseException:
                pass

    def _finish_draft(self, video: VideoResult) -> None:
        with self._lock:
            draft = self._draft
            if draft is not None and draft.running:
                draft.video = video
                draft.stopped_s = time.perf_counter()
                draft.running = False

    def _reset_telemetry_locked(self) -> None:
        self._latest_control = None
        self._control_periods.clear()
        self._observation_latencies.clear()
        self._inference_latencies.clear()
        self._send_latencies.clear()
        self._missed_control_deadlines = 0
        self._queued_actions = 0
        self._control_stage = "not started"
        self._control_stage_started_s = None

    # ------------------------------------------------------------------
    # watchdog
    # ------------------------------------------------------------------

    def _start_watchdog(self) -> None:
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="policy-watchdog", daemon=True
        )
        self._watchdog_thread.start()

    def _stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        thread = self._watchdog_thread
        if thread is not None and threading.current_thread() is not thread:
            thread.join(timeout=1.0)
        self._watchdog_thread = None

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(0.05):
            now = time.perf_counter()
            should_stop_rollout = False
            control_to_stop: _PolicyControlWorker | None = None
            with self._lock:
                state = self._state
                config = self._config
                latest = self._latest_control
                connected_at = self._connected_at_s
                draft = self._draft
                if (
                    state is PolicyState.RUNNING
                    and draft is not None
                    and draft.config.duration_s is not None
                    and now - draft.started_s >= draft.config.duration_s
                ):
                    should_stop_rollout = True
                # A parking move owns the control thread for seconds by design,
                # so staleness is not evidence of a fault while it runs.
                active = state in {PolicyState.IDLE, PolicyState.RUNNING} and not self._long_operation
                first_update_pending = latest is None
                reference = connected_at if first_update_pending else latest.timestamp_s
                steady_timeout_s = max(
                    config.watchdog_timeout_s if config is not None else 0.0,
                    3.0 / config.control_hz if config is not None else 0.0,
                )
                timeout_s = (
                    config.watchdog_startup_timeout_s
                    if first_update_pending and config is not None
                    else steady_timeout_s
                )
                update_age_s = now - reference if reference is not None else 0.0
                stale = (
                    active
                    and config is not None
                    and reference is not None
                    and update_age_s > timeout_s
                )
                if stale:
                    stage = self._control_stage
                    stage_started_s = self._control_stage_started_s
                    stage_age_s = now - stage_started_s if stage_started_s is not None else 0.0
                    update_kind = (
                        "initial control update" if first_update_pending else "control update"
                    )
                    self._state = PolicyState.ERROR
                    self._last_error = (
                        "RuntimeError: "
                        f"No successful {update_kind} for {update_age_s:.3f}s "
                        f"(limit {timeout_s:.3f}s). Control worker was in "
                        f"{stage} for {stage_age_s:.3f}s."
                    )
                    self._error_source = "watchdog"
                    control_to_stop = self._control
            if control_to_stop is not None:
                control_to_stop.set_driving(False)
                control_to_stop.request_stop()
            if should_stop_rollout:
                try:
                    self.stop_rollout()
                except BaseException as exc:
                    self._worker_failed("rollout_timeout", exc)


def count_outcomes(output_dir: Path | str) -> dict[str, int]:
    """Tally the judged episodes already saved under ``output_dir``.

    Reads only ``meta.json``, so a directory of a hundred episodes costs a
    hundred small reads and no video decoding. ``unjudged`` counts episodes
    saved before an outcome was picked - they are still valid recordings, they
    just cannot contribute to a success rate or to a preference pair.
    """
    counts = {"success": 0, "failure": 0, "unjudged": 0}
    directory = Path(output_dir).expanduser()
    if not directory.is_dir():
        return counts
    for meta_path in sorted(directory.glob("episode_*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        outcome = meta.get("outcome")
        counts[outcome if outcome in OUTCOMES else "unjudged"] += 1
    return counts


def _policy_meta(policy: PolicyInfo | None) -> dict[str, Any] | None:
    if policy is None:
        return None
    return {
        "checkpoint": policy.checkpoint,
        "run_name": policy.run_name,
        "step": policy.step,
        "policy": policy.policy,
        "objective": policy.objective,
        "chunk_size": policy.chunk_size,
        "action_steps": policy.action_steps,
        "num_inference_steps": policy.num_inference_steps,
        "action_repr": policy.action_repr,
        "delta_mode": policy.delta_mode,
        "vision": policy.vision,
        "pool": policy.pool,
        "image_size": policy.image_size,
        "use_ema": policy.use_ema,
        # Recorded per rollout: guidance is retunable between trials, so the
        # episode has to say which weight actually produced it.
        "n_obs_steps": policy.n_obs_steps,
        "guidance_weight": policy.guidance_weight,
    }


# ---------------------------------------------------------------------------
# hardware process
# ---------------------------------------------------------------------------

_PROCESS_OPERATIONS = frozenset(
    {
        "load_policy",
        "unload_policy",
        "connect",
        "disconnect",
        "emergency_stop",
        "start_rollout",
        "stop_rollout",
        "save_rollout",
        "set_rollout_outcome",
        "discard_rollout",
        "return_to_start",
        "set_preview_enabled",
        "set_guidance_weight",
        "generate_goal_video",
        "set_goal_frame",
        "get_snapshot",
    }
)


def _snapshot_for_ipc(snapshot: PolicySnapshot) -> PolicySnapshot:
    """Replace read-only mapping proxies with picklable plain dictionaries."""
    return replace(
        snapshot,
        latest_action=dict(snapshot.latest_action),
        latest_observation=dict(snapshot.latest_observation),
    )


def _runtime_process_main(connection: Any) -> None:
    """Own the policy and the hardware in a process that never imports Streamlit."""
    runtime = PolicyRuntime()
    try:
        while True:
            request_id, operation, args, kwargs = connection.recv()
            if operation == "shutdown":
                try:
                    if runtime.state is not PolicyState.DISCONNECTED:
                        runtime.disconnect()
                    connection.send(
                        (request_id, "ok", _snapshot_for_ipc(runtime.get_snapshot()))
                    )
                except BaseException:
                    connection.send((request_id, "error", traceback.format_exc()))
                return
            if operation not in _PROCESS_OPERATIONS:
                connection.send(
                    (request_id, "error", f"Unsupported runtime operation: {operation}")
                )
                continue
            try:
                result: Any = getattr(runtime, operation)(*args, **kwargs)
                if isinstance(result, PolicySnapshot):
                    result = _snapshot_for_ipc(result)
                connection.send((request_id, "ok", result))
            except BaseException:
                connection.send((request_id, "error", traceback.format_exc()))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        try:
            if runtime.state is not PolicyState.DISCONNECTED:
                runtime.disconnect()
        except BaseException:
            try:
                runtime.emergency_stop()
            except BaseException:
                pass
        connection.close()


class ProcessPolicyRuntime:
    """Streamlit-side proxy for the policy runtime in a spawned process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._context = multiprocessing.get_context("spawn")
        self._connection: Any = None
        self._process: multiprocessing.Process | None = None
        self._request_id = 0
        self._last_snapshot = PolicyRuntime().get_snapshot()
        self._last_snapshot_received_s = 0.0

    @property
    def state(self) -> PolicyState:
        with self._lock:
            return self._last_snapshot.state

    @property
    def process_pid(self) -> int | None:
        with self._lock:
            process = self._process
            return process.pid if process is not None and process.is_alive() else None

    # Loading a checkpoint has to build a CUDA context and run one
    # warm-up inference, which is minutes on a cold machine, not seconds.
    def load_policy(self, settings: PolicySettings) -> PolicySnapshot:
        return self._snapshot_rpc("load_policy", settings, timeout_s=600.0)

    def unload_policy(self) -> PolicySnapshot:
        if not self._has_process():
            return self.get_snapshot()
        return self._snapshot_rpc("unload_policy", timeout_s=10.0)

    def connect(self, config: PolicyRuntimeConfig) -> PolicySnapshot:
        return self._snapshot_rpc("connect", config, timeout_s=90.0)

    def disconnect(self) -> PolicySnapshot:
        if not self._has_process():
            return self.get_snapshot()
        return self._snapshot_rpc("disconnect", timeout_s=60.0)

    def emergency_stop(self) -> PolicySnapshot:
        if not self._has_process():
            return self.get_snapshot()
        return self._snapshot_rpc("emergency_stop", timeout_s=10.0)

    def start_rollout(self, config: RolloutConfig) -> PolicySnapshot:
        return self._snapshot_rpc("start_rollout", config, timeout_s=15.0)

    # Stopping now includes the parking trajectory, which is rate limited at
    # 30 deg/s and verifies its endpoint before returning.
    def stop_rollout(self) -> PolicySnapshot:
        return self._snapshot_rpc("stop_rollout", timeout_s=90.0)

    def return_to_start(self) -> PolicySnapshot:
        return self._snapshot_rpc("return_to_start", timeout_s=90.0)

    def save_rollout(self) -> Path:
        result = self._rpc("save_rollout", timeout_s=120.0)
        if not isinstance(result, Path):
            raise RuntimeError("Runtime process returned an invalid saved episode path.")
        return result

    def set_rollout_outcome(self, outcome: str | None) -> PolicySnapshot:
        return self._snapshot_rpc("set_rollout_outcome", outcome, timeout_s=10.0)

    def discard_rollout(self) -> PolicySnapshot:
        if not self._has_process():
            return self.get_snapshot()
        return self._snapshot_rpc("discard_rollout", timeout_s=90.0)

    def set_preview_enabled(self, enabled: bool) -> PolicySnapshot:
        if not self._has_process():
            return self.get_snapshot()
        return self._snapshot_rpc("set_preview_enabled", enabled, timeout_s=5.0)

    def set_guidance_weight(self, weight: float) -> PolicySnapshot:
        return self._snapshot_rpc("set_guidance_weight", weight, timeout_s=10.0)

    # Generation is the slow half of a goal-conditioned rollout: a 93-frame
    # Cosmos clip is seconds of transformer, and the first call also pays for
    # loading the model behind whichever source is configured.
    def generate_goal_video(self, config: Any) -> Any:
        return self._rpc("generate_goal_video", config, timeout_s=600.0)

    def set_goal_frame(self, frame_png: bytes | None) -> PolicySnapshot:
        return self._snapshot_rpc("set_goal_frame", frame_png, timeout_s=30.0)

    def get_snapshot(self) -> PolicySnapshot:
        if not self._has_process():
            with self._lock:
                return self._last_snapshot
        with self._lock:
            if time.perf_counter() - self._last_snapshot_received_s < 0.1:
                return self._last_snapshot
        try:
            return self._snapshot_rpc("get_snapshot", timeout_s=5.0)
        except BaseException as exc:
            return self._process_error_snapshot(exc)

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
        if process is None:
            return
        try:
            if process.is_alive():
                self._snapshot_rpc("shutdown", timeout_s=60.0)
        except BaseException:
            pass
        process.join(timeout=5.0)
        with self._lock:
            connection = self._connection
            self._connection = None
            self._process = None
        if connection is not None:
            connection.close()

    def _snapshot_rpc(self, operation: str, *args: Any, timeout_s: float) -> PolicySnapshot:
        result = self._rpc(operation, *args, timeout_s=timeout_s)
        if not isinstance(result, PolicySnapshot):
            raise RuntimeError(f"Runtime process returned an invalid response for {operation}.")
        with self._lock:
            self._last_snapshot = result
            self._last_snapshot_received_s = time.perf_counter()
        return result

    def _rpc(self, operation: str, *args: Any, timeout_s: float) -> Any:
        with self._lock:
            self._ensure_process_locked()
            connection = self._connection
            process = self._process
            if connection is None or process is None:
                raise RuntimeError("Runtime process is unavailable.")
            self._request_id += 1
            request_id = self._request_id
            try:
                connection.send((request_id, operation, args, {}))
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RuntimeError("Could not contact the runtime process.") from exc

            deadline = time.perf_counter() + timeout_s
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Runtime process did not finish {operation} within {timeout_s:.1f}s."
                    )
                if not connection.poll(min(remaining, 0.1)):
                    if not process.is_alive():
                        raise RuntimeError(
                            f"Runtime process exited during {operation} "
                            f"(exit code {process.exitcode})."
                        )
                    continue
                response_id, status, payload = connection.recv()
                if response_id != request_id:
                    continue
                if status == "error":
                    raise RuntimeError(str(payload))
                return payload

    def _ensure_process_locked(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            return
        if self._connection is not None:
            self._connection.close()
        parent_connection, child_connection = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_runtime_process_main,
            args=(child_connection,),
            name="streamlit-policy-runtime",
            daemon=True,
        )
        process.start()
        child_connection.close()
        self._connection = parent_connection
        self._process = process

    def _has_process(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.is_alive()

    def _process_error_snapshot(self, exc: BaseException) -> PolicySnapshot:
        with self._lock:
            snapshot = replace(
                self._last_snapshot,
                state=PolicyState.ERROR,
                last_error=f"{type(exc).__name__}: {exc}",
                error_source="runtime_process",
            )
            self._last_snapshot = snapshot
            self._last_snapshot_received_s = time.perf_counter()
            return snapshot


_PROCESS_RUNTIME: ProcessPolicyRuntime | None = None
_PROCESS_RUNTIME_LOCK = threading.Lock()


def get_runtime() -> ProcessPolicyRuntime:
    """Return the shared proxy whose child process exclusively owns hardware."""
    global _PROCESS_RUNTIME
    with _PROCESS_RUNTIME_LOCK:
        if _PROCESS_RUNTIME is None:
            _PROCESS_RUNTIME = ProcessPolicyRuntime()
        return _PROCESS_RUNTIME


def _shutdown_process_runtime() -> None:
    runtime = _PROCESS_RUNTIME
    if runtime is not None:
        runtime.shutdown()


atexit.register(_shutdown_process_runtime)
