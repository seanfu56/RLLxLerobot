"""UI-independent runtime for smooth single-camera teleoperation.

The Streamlit page is deliberately only a supervisory layer. This module owns
the hardware lifecycle and all timing-sensitive work:

* one CLI-equivalent leader-to-follower control worker;
* one lower-rate action sampler used only while recording;
* LeRobot-owned camera capture plus independent preview and video writer workers;
* a process-wide lifecycle state machine and immutable telemetry snapshots.

Hardware-specific imports are lazy so the state machine and timing logic can
be tested on machines without LeRobot, OpenCV, CAN, or Dynamixel installed.
"""

from __future__ import annotations

import atexit
import csv
import importlib
import json
import math
import multiprocessing
import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

ACTION_KEYS = tuple([f"joint_{index}.pos" for index in range(1, 7)] + ["gripper.pos"])
OBSERVATION_PREFIX = "observation."
OBSERVATION_COLUMNS = tuple(f"{OBSERVATION_PREFIX}{key}" for key in ACTION_KEYS)
CAMERA_KEY = "overhead"
DEFAULT_STATIC_PATH = Path(__file__).resolve().parent / "static" / "live.jpg"

# Pose the Piper is driven to on Safe disconnect, before torque is released.
# PiperFollower.disconnect() runs its smooth trajectory to config.rest_position_deg
# and only then disables the arm and gripper, so this is the pose the arm is left
# resting in. It is set here rather than in the plugin default so the UI's parked
# pose is explicit and does not change the pose used by the CLI scripts.
REST_STATE_DEG = {
    "joint_1.pos": 0.00,
    "joint_2.pos": 0.00,
    "joint_3.pos": 0.00,
    "joint_4.pos": 0.00,
    "joint_5.pos": 29.00,
    "joint_6.pos": 0.00,
    "gripper.pos": 0.60,
}


class RuntimeState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class CameraSettings:
    device: str = "/dev/video3"
    width: int = 640
    height: int = 480
    fps: float = 30.0
    fourcc: str | None = None
    preview_hz: float = 20.0
    jpeg_quality: int = 80
    video_queue_size: int = 120
    preview_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("Camera device must not be empty.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive.")
        if self.fps <= 0 or self.preview_hz <= 0:
            raise ValueError("Camera and preview rates must be positive.")
        if self.fourcc is not None and len(self.fourcc) != 4:
            raise ValueError("Camera FourCC must contain exactly four characters.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100.")
        if self.video_queue_size <= 0:
            raise ValueError("Video queue size must be positive.")


@dataclass(frozen=True)
class RuntimeConfig:
    can_port: str = "piper_left"
    leader_port: str = "/dev/robotis_left"
    speed_rate: int = 50
    gripper_max_mm: float = 100.0
    max_relative_target: float | None = None
    control_hz: float = 200.0
    watchdog_timeout_s: float = 2.0
    watchdog_startup_timeout_s: float = 3.0
    leader_smoothing_factor: float = 0.0
    gripper_spring_enabled: bool = False
    camera: CameraSettings = field(default_factory=CameraSettings)
    preview_path: Path = field(default_factory=lambda: DEFAULT_STATIC_PATH)

    def __post_init__(self) -> None:
        object.__setattr__(self, "preview_path", Path(self.preview_path))
        if not self.can_port or not self.leader_port:
            raise ValueError("CAN and leader ports must not be empty.")
        if not 1 <= self.speed_rate <= 100:
            raise ValueError("Piper speed rate must be between 1 and 100.")
        if self.gripper_max_mm not in (70.0, 100.0):
            raise ValueError("Piper gripper maximum must be 70 or 100 mm.")
        if self.control_hz <= 0:
            raise ValueError("Control rate must be positive.")
        if self.watchdog_timeout_s <= 0:
            raise ValueError("Watchdog timeout must be positive.")
        if self.watchdog_startup_timeout_s <= 0:
            raise ValueError("Watchdog startup timeout must be positive.")
        if not 0 <= self.leader_smoothing_factor < 1:
            raise ValueError("Leader smoothing factor must be in [0, 1).")


@dataclass(frozen=True)
class RecordingConfig:
    output_dir: Path
    task: str
    action_sample_hz: float = 20.0
    duration_s: float | None = 15.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())
        if self.action_sample_hz <= 0:
            raise ValueError("Action sample rate must be positive.")
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError("Episode duration must be positive or unset.")


@dataclass(frozen=True)
class TimingSummary:
    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0


@dataclass(frozen=True)
class DraftSnapshot:
    episode_index: int
    output_dir: str
    task: str
    action_samples: int
    video_frames: int
    video_dropped_frames: int
    duration_s: float
    recording: bool


@dataclass(frozen=True)
class RuntimeSnapshot:
    state: RuntimeState
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
    leader_latency: TimingSummary
    robot_send_latency: TimingSummary
    camera_frame_age_s: float | None
    camera_capture_hz: float
    camera_frames: int
    preview_encode_ms: float
    video_queue_depth: int
    video_dropped_frames: int
    action_samples: int
    action_sample_hz: float
    missed_action_deadlines: int
    recording_elapsed_s: float
    recording_duration_s: float | None
    draft: DraftSnapshot | None
    last_error: str | None
    error_source: str | None


@dataclass
class HardwareBundle:
    robot: Any
    teleop: Any
    teleop_action_processor: Callable[[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]
    robot_action_processor: Callable[[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]
    emergency_disable: Callable[[], None] | None = None


@dataclass(frozen=True)
class VideoResult:
    frame_count: int
    dropped_frames: int
    frame_timestamps: tuple[dict[str, float | int], ...]
    elapsed_s: float


@dataclass(frozen=True)
class _FrameReference:
    capture_sequence: int
    timestamp_s: float
    frame: Any


class CameraBackend(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def set_preview_enabled(self, enabled: bool) -> None: ...

    def publish_observation(
        self, observation: Mapping[str, Any], timestamp_s: float
    ) -> _FrameReference | None: ...

    def start_recording(
        self, path: Path, episode_start_s: float, output_fps: float
    ) -> None: ...

    def record_frame(
        self, frame_index: int, reference: _FrameReference
    ) -> bool: ...

    def stop_recording(self) -> VideoResult: ...

    def snapshot(self) -> Mapping[str, float | int | None]: ...

    # The most recent captured frame, or None before the first observation.
    # Only the goal-conditioned policy page uses it, to hand the video model
    # the scene the rollout is about to start from.
    def latest_frame(self) -> Any | None: ...


@dataclass(frozen=True)
class _ControlSample:
    sequence: int
    timestamp_s: float
    observation_timestamp_s: float
    action: Mapping[str, float]
    observation: Mapping[str, float]
    frame: _FrameReference | None = None


@dataclass
class _EpisodeDraft:
    config: RecordingConfig
    episode_index: int
    pending_dir: Path
    avi_path: Path
    started_s: float
    stopped_s: float | None = None
    actions: list[dict[str, Any]] = field(default_factory=list)
    action_missed_deadlines: int = 0
    video: VideoResult | None = None
    recording: bool = True


def _plain_scalars(values: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            continue
        try:
            result[key] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _timing_summary(values_s: list[float]) -> TimingSummary:
    if not values_s:
        return TimingSummary()
    ordered = sorted(values_s)
    return TimingSummary(
        count=len(ordered),
        mean_ms=sum(ordered) / len(ordered) * 1000,
        p50_ms=_percentile(ordered, 0.50) * 1000,
        p95_ms=_percentile(ordered, 0.95) * 1000,
        p99_ms=_percentile(ordered, 0.99) * 1000,
        max_ms=ordered[-1] * 1000,
    )


def _normalise_camera_source(device: str) -> str | int:
    stripped = device.strip()
    return int(stripped) if stripped.isdigit() else stripped


def _next_episode_index(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    indices: list[int] = []
    for path in output_dir.iterdir():
        name = path.name
        if name.startswith(".episode_") and name.endswith(".pending"):
            name = name[1:-8]
        if not name.startswith("episode_"):
            continue
        try:
            indices.append(int(name.split("_", maxsplit=1)[1]))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def _default_hardware_factory(config: RuntimeConfig) -> HardwareBundle:
    try:
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.processor import make_default_processors
        from lerobot.robots import make_robot_from_config
        from lerobot.teleoperators import make_teleoperator_from_config
        from lerobot.utils.import_utils import register_third_party_plugins
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is unavailable. Activate the repository environment and install "
            "the Piper and ROBOTIS plugins before connecting."
        ) from exc

    register_third_party_plugins()
    try:
        piper_module = importlib.import_module("lerobot_robot_piper.config_piper_follower")
        robotis_module = importlib.import_module("lerobot_teleoperator_robotis.config_robotis_leader")
    except ImportError as exc:
        raise RuntimeError(
            "The Piper follower and ROBOTIS leader plugins must both be installed "
            "in the active Python environment."
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
    robot_config = piper_module.PiperFollowerConfig(
        can_port=config.can_port,
        speed_rate=config.speed_rate,
        gripper_max_mm=config.gripper_max_mm,
        max_relative_target=config.max_relative_target,
        rest_position_deg=dict(REST_STATE_DEG),
        cameras={CAMERA_KEY: camera_config},
    )
    teleop_config = robotis_module.RobotisLeaderConfig(
        port=config.leader_port,
        smoothing_factor=config.leader_smoothing_factor,
        gripper_spring_enabled=config.gripper_spring_enabled,
        gripper_piper_open_mm=config.gripper_max_mm,
    )
    robot = make_robot_from_config(robot_config)
    teleop = make_teleoperator_from_config(teleop_config)
    teleop_processor, robot_processor, _ = make_default_processors()

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

    return HardwareBundle(
        robot=robot,
        teleop=teleop,
        teleop_action_processor=teleop_processor,
        robot_action_processor=robot_processor,
        emergency_disable=emergency_disable,
    )


class _ControlWorker:
    def __init__(
        self,
        hardware: HardwareBundle,
        target_hz: float,
        on_observation: Callable[
            [Mapping[str, Any], float], _FrameReference | None
        ],
        on_sample: Callable[..., None],
        on_progress: Callable[[str], None],
        on_error: Callable[[str, BaseException], None],
    ) -> None:
        self._hardware = hardware
        self._period_s = 1.0 / target_hz
        self._on_observation = on_observation
        self._on_sample = on_sample
        self._on_progress = on_progress
        self._on_error = on_error
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="teleop-control", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def request_stop(self) -> None:
        self._stop.set()

    def stop(self, timeout_s: float = 3.0) -> bool:
        self.request_stop()
        if threading.current_thread() is self._thread:
            return True
        self._thread.join(timeout=timeout_s)
        return not self._thread.is_alive()

    def _run(self) -> None:
        previous_start: float | None = None
        sequence = 0
        self._on_progress("starting")
        while not self._stop.is_set():
            loop_start = time.perf_counter()
            period_s = loop_start - previous_start if previous_start is not None else None
            previous_start = loop_start

            try:
                # Keep this path in the same order as
                # lerobot.scripts.lerobot_teleoperate.teleop_loop.
                self._on_progress("robot.get_observation")
                observation_started = time.perf_counter()
                observation = self._hardware.robot.get_observation()
                observation_latency_s = time.perf_counter() - observation_started
                observation_timestamp_s = time.perf_counter()

                if self._stop.is_set():
                    break
                self._on_progress("teleop.get_action")
                leader_started = time.perf_counter()
                raw_action = self._hardware.teleop.get_action()
                leader_latency_s = time.perf_counter() - leader_started

                self._on_progress("action processors")
                teleop_action = self._hardware.teleop_action_processor((raw_action, observation))
                final_action = self._hardware.robot_action_processor((teleop_action, observation))
                if self._stop.is_set():
                    break

                self._on_progress("robot.send_action")
                send_started = time.perf_counter()
                sent_action = self._hardware.robot.send_action(final_action)
                send_latency_s = time.perf_counter() - send_started
                command_timestamp_s = time.perf_counter()
                frame_reference = self._on_observation(
                    observation, observation_timestamp_s
                )
                sequence += 1
                self._on_sample(
                    sequence=sequence,
                    timestamp_s=command_timestamp_s,
                    observation_timestamp_s=observation_timestamp_s,
                    action=_plain_scalars(sent_action if sent_action is not None else final_action),
                    observation=_plain_scalars(observation),
                    frame=frame_reference,
                    period_s=period_s,
                    observation_latency_s=observation_latency_s,
                    leader_latency_s=leader_latency_s,
                    send_latency_s=send_latency_s,
                )
                self._on_progress("waiting for next deadline")
            except BaseException as exc:
                self._on_error("control", exc)
                return

            # The CLI sleeps only for the unused part of this iteration. It does
            # not enqueue or replay commands when an iteration runs long.
            work_s = time.perf_counter() - loop_start
            if work_s < self._period_s:
                self._stop.wait(self._period_s - work_s)
            else:
                deadlines_missed = max(math.floor(work_s / self._period_s), 1)
                self._on_sample(missed_deadlines=deadlines_missed)


class _ActionSampler:
    def __init__(
        self,
        sample_hz: float,
        episode_start_s: float,
        get_latest: Callable[[], _ControlSample | None],
        record_frame: Callable[[int, _FrameReference], bool],
        on_error: Callable[[str, BaseException], None],
    ) -> None:
        self._period_s = 1.0 / sample_hz
        self._episode_start_s = episode_start_s
        self._get_latest = get_latest
        self._record_frame = record_frame
        self._on_error = on_error
        self._stop = threading.Event()
        self._rows: list[dict[str, Any]] = []
        self._missed_deadlines = 0
        self._thread = threading.Thread(target=self._run, name="action-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> tuple[list[dict[str, Any]], int]:
        self._stop.set()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise RuntimeError("Action sampler did not stop within the timeout.")
        return list(self._rows), self._missed_deadlines

    def snapshot(self) -> tuple[int, int]:
        return len(self._rows), self._missed_deadlines

    def _run(self) -> None:
        next_deadline = self._episode_start_s + self._period_s
        sample_index = 0
        try:
            while not self._stop.is_set():
                before_deadline = time.perf_counter()
                if before_deadline < next_deadline:
                    if self._stop.wait(next_deadline - before_deadline):
                        return
                now = time.perf_counter()
                sample = self._get_latest()
                frame = sample.frame if sample is not None else None
                valid_pair = (
                    sample is not None
                    and frame is not None
                    and sample.timestamp_s >= self._episode_start_s
                    and frame.timestamp_s >= self._episode_start_s
                )
                if valid_pair:
                    assert sample is not None
                    missing_action = [key for key in ACTION_KEYS if key not in sample.action]
                    missing_observation = [
                        key for key in ACTION_KEYS if key not in sample.observation
                    ]
                    if missing_action or missing_observation:
                        raise RuntimeError(
                            "Cannot record an incomplete control sample: "
                            f"missing actions={missing_action}, "
                            f"missing observations={missing_observation}"
                        )
                    non_finite = [
                        key
                        for key in ACTION_KEYS
                        if not math.isfinite(sample.action[key])
                        or not math.isfinite(sample.observation[key])
                    ]
                    if non_finite:
                        raise RuntimeError(
                            "Cannot record non-finite action/observation values for "
                            f"{non_finite}"
                        )

                if valid_pair and self._record_frame(sample_index, frame):
                    assert sample is not None
                    row: dict[str, Any] = {
                        "sample_index": sample_index,
                        "frame_index": sample_index,
                        "sample_timestamp_s": round(now - self._episode_start_s, 6),
                        "source_control_sequence": sample.sequence,
                        "source_control_timestamp_s": round(
                            sample.timestamp_s - self._episode_start_s, 6
                        ),
                        "observation_timestamp_s": round(
                            sample.observation_timestamp_s - self._episode_start_s, 6
                        ),
                        "observation_action_delta_ms": round(
                            (sample.timestamp_s - sample.observation_timestamp_s) * 1000,
                            3,
                        ),
                        "capture_sequence": frame.capture_sequence,
                        "frame_timestamp_s": round(
                            frame.timestamp_s - self._episode_start_s, 6
                        ),
                        "action_frame_delta_ms": round(
                            (sample.timestamp_s - frame.timestamp_s) * 1000,
                            3,
                        ),
                    }
                    row.update({key: sample.action[key] for key in ACTION_KEYS})
                    row.update(
                        {
                            column: sample.observation[key]
                            for key, column in zip(
                                ACTION_KEYS, OBSERVATION_COLUMNS, strict=True
                            )
                        }
                    )
                    self._rows.append(row)
                    sample_index += 1
                else:
                    self._missed_deadlines += 1

                next_deadline += self._period_s
                after_sample = time.perf_counter()
                if after_sample < next_deadline:
                    self._stop.wait(next_deadline - after_sample)
                    continue
                skipped = math.floor((after_sample - next_deadline) / self._period_s) + 1
                self._missed_deadlines += skipped
                next_deadline += skipped * self._period_s
        except BaseException as exc:
            self._on_error("action_sampler", exc)


class _PreviewPublisher:
    def __init__(
        self,
        settings: CameraSettings,
        preview_path: Path,
        cv2_module: Any,
        get_latest: Callable[[], tuple[int, float, Any] | None],
        on_error: Callable[[str, BaseException], None],
    ) -> None:
        self._settings = settings
        self._path = preview_path
        self._tmp_path = preview_path.with_name(f".{preview_path.name}.tmp")
        self._cv2 = cv2_module
        self._get_latest = get_latest
        self._on_error = on_error
        self._enabled = settings.preview_enabled
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_encode_ms = 0.0
        self._thread = threading.Thread(target=self._run, name="preview-publisher", daemon=True)

    def start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {"encode_ms": self._last_encode_ms}

    def stop(self) -> None:
        self._stop.set()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=3.0)
        self._tmp_path.unlink(missing_ok=True)

    def _run(self) -> None:
        period_s = 1.0 / self._settings.preview_hz
        next_deadline = time.perf_counter()
        last_sequence = -1
        try:
            while not self._stop.is_set():
                with self._lock:
                    enabled = self._enabled
                latest = self._get_latest()
                if enabled and latest is not None and latest[0] != last_sequence:
                    started = time.perf_counter()
                    ok, buffer = self._cv2.imencode(
                        ".jpg",
                        latest[2],
                        [self._cv2.IMWRITE_JPEG_QUALITY, self._settings.jpeg_quality],
                    )
                    if ok:
                        self._tmp_path.write_bytes(buffer.tobytes())
                        os.replace(self._tmp_path, self._path)
                        last_sequence = latest[0]
                    with self._lock:
                        self._last_encode_ms = (time.perf_counter() - started) * 1000

                next_deadline += period_s
                now = time.perf_counter()
                if now >= next_deadline:
                    skipped = math.floor((now - next_deadline) / period_s) + 1
                    next_deadline += skipped * period_s
                self._stop.wait(max(next_deadline - time.perf_counter(), 0.0))
        except BaseException as exc:
            self._on_error("preview", exc)


class _VideoWriterWorker:
    def __init__(
        self,
        path: Path,
        settings: CameraSettings,
        frame_size: tuple[int, int],
        episode_start_s: float,
        output_fps: float,
        cv2_module: Any,
        on_error: Callable[[str, BaseException], None],
    ) -> None:
        self._path = path
        self._episode_start_s = episode_start_s
        self._on_error = on_error
        self._queue: queue.Queue[tuple[int, int, float, Any]] = queue.Queue(
            maxsize=settings.video_queue_size
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._accepting = True
        self._dropped = 0
        self._written = 0
        self._timestamps: list[dict[str, float | int]] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2_module.VideoWriter(
            str(path),
            cv2_module.VideoWriter_fourcc(*"MJPG"),
            output_fps,
            frame_size,
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Cannot create video writer at {path}.")
        self._writer = writer
        self._thread = threading.Thread(target=self._run, name="video-writer", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def enqueue(
        self,
        frame_index: int,
        capture_sequence: int,
        timestamp_s: float,
        frame: Any,
    ) -> bool:
        with self._lock:
            accepting = self._accepting
        if not accepting:
            return False
        try:
            self._queue.put_nowait(
                (frame_index, capture_sequence, timestamp_s, frame)
            )
            return True
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "queue_depth": self._queue.qsize(),
                "dropped_frames": self._dropped,
                "written_frames": self._written,
            }

    def stop(self, timeout_s: float = 10.0) -> VideoResult:
        with self._lock:
            self._accepting = False
        self._stop.set()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise RuntimeError("Video writer did not flush within the timeout.")
        with self._lock:
            timestamps = tuple(self._timestamps)
            written = self._written
            dropped = self._dropped
        elapsed = timestamps[-1]["timestamp_s"] if timestamps else 0.0
        return VideoResult(written, dropped, timestamps, float(elapsed))

    def _run(self) -> None:
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    frame_index, capture_sequence, timestamp_s, frame = self._queue.get(
                        timeout=0.05
                    )
                except queue.Empty:
                    continue
                self._writer.write(frame)
                with self._lock:
                    self._written += 1
                    self._timestamps.append(
                        {
                            "frame_index": frame_index,
                            "capture_sequence": capture_sequence,
                            "timestamp_s": round(timestamp_s - self._episode_start_s, 6),
                        }
                    )
                self._queue.task_done()
        except BaseException as exc:
            with self._lock:
                unwritten = self._queue.qsize()
                self._dropped += unwritten
            while True:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    break
            self._on_error("video_writer", exc)
        finally:
            self._writer.release()


class _ObservationCameraSystem:
    """Preview and record frames captured by PiperFollower's LeRobot camera."""

    def __init__(
        self,
        settings: CameraSettings,
        preview_path: Path,
        on_error: Callable[[str, BaseException], None],
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is unavailable. Install opencv-python-headless in the active environment."
            ) from exc
        cv2.setNumThreads(1)
        self._cv2 = cv2
        self._settings = settings
        self._on_error = on_error
        self._lock = threading.Lock()
        self._latest: tuple[int, float, Any] | None = None
        self._last_frame: Any = None
        self._frame_periods: deque[float] = deque(maxlen=300)
        self._frame_count = 0
        self._writer: _VideoWriterWorker | None = None
        self._preview = _PreviewPublisher(
            settings,
            preview_path,
            cv2,
            self._get_latest,
            on_error,
        )

    def start(self) -> None:
        self._preview.start()

    def set_preview_enabled(self, enabled: bool) -> None:
        self._preview.set_enabled(enabled)

    def publish_observation(
        self, observation: Mapping[str, Any], timestamp_s: float
    ) -> _FrameReference | None:
        frame = observation.get(CAMERA_KEY)
        if frame is None:
            return None
        with self._lock:
            if frame is self._last_frame:
                latest = self._latest
                if latest is None:
                    return None
                return _FrameReference(latest[0], latest[1], latest[2])
            previous = self._latest
            self._last_frame = frame
            self._frame_count += 1
            sequence = self._frame_count
            self._latest = (sequence, timestamp_s, frame)
            if previous is not None:
                self._frame_periods.append(timestamp_s - previous[1])
            return _FrameReference(sequence, timestamp_s, frame)

    def start_recording(
        self, path: Path, episode_start_s: float, output_fps: float
    ) -> None:
        writer = _VideoWriterWorker(
            path,
            self._settings,
            (self._settings.width, self._settings.height),
            episode_start_s,
            output_fps,
            self._cv2,
            self._on_error,
        )
        writer.start()
        with self._lock:
            if self._writer is not None:
                writer.stop()
                raise RuntimeError("Camera is already recording.")
            self._writer = writer

    def record_frame(
        self, frame_index: int, reference: _FrameReference
    ) -> bool:
        with self._lock:
            writer = self._writer
        if writer is None:
            return False
        return writer.enqueue(
            frame_index,
            reference.capture_sequence,
            reference.timestamp_s,
            reference.frame,
        )

    def stop_recording(self) -> VideoResult:
        with self._lock:
            writer = self._writer
            self._writer = None
        if writer is None:
            return VideoResult(0, 0, (), 0.0)
        return writer.stop()

    def snapshot(self) -> Mapping[str, float | int | None]:
        with self._lock:
            latest = self._latest
            periods = list(self._frame_periods)
            count = self._frame_count
            writer = self._writer
        frame_age = time.perf_counter() - latest[1] if latest is not None else None
        capture_hz = 1.0 / (sum(periods) / len(periods)) if periods else 0.0
        result: dict[str, float | int | None] = {
            "frame_age_s": frame_age,
            "capture_hz": capture_hz,
            "frame_count": count,
        }
        result.update(self._preview.snapshot())
        if writer is None:
            result.update(queue_depth=0, dropped_frames=0, written_frames=0)
        else:
            result.update(writer.snapshot())
        return result

    def stop(self) -> None:
        errors: list[BaseException] = []
        try:
            self.stop_recording()
        except BaseException as exc:
            errors.append(exc)
        try:
            self._preview.stop()
        except BaseException as exc:
            errors.append(exc)
        with self._lock:
            self._latest = None
            self._last_frame = None
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))

    def _get_latest(self) -> tuple[int, float, Any] | None:
        with self._lock:
            return self._latest

    def latest_frame(self) -> Any | None:
        """The most recent captured frame, or None before the first observation."""
        latest = self._get_latest()
        return None if latest is None else latest[2]


def _default_camera_factory(
    config: RuntimeConfig,
    on_error: Callable[[str, BaseException], None],
) -> CameraBackend:
    return _ObservationCameraSystem(config.camera, config.preview_path, on_error)


def _default_video_converter(source: Path, destination: Path) -> None:
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "1",
                str(destination),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to save recordings as MP4.") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")


class TeleopRuntime:
    """One process-wide owner for teleoperation hardware and recordings."""

    def __init__(
        self,
        hardware_factory: Callable[[RuntimeConfig], HardwareBundle] = _default_hardware_factory,
        camera_factory: Callable[
            [RuntimeConfig, Callable[[str, BaseException], None]], CameraBackend
        ] = _default_camera_factory,
        video_converter: Callable[[Path, Path], None] = _default_video_converter,
        telemetry_window: int = 2000,
    ) -> None:
        self._hardware_factory = hardware_factory
        self._camera_factory = camera_factory
        self._video_converter = video_converter
        self._lock = threading.RLock()
        self._command_lock = threading.RLock()
        self._state = RuntimeState.DISCONNECTED
        self._config: RuntimeConfig | None = None
        self._hardware: HardwareBundle | None = None
        self._camera: CameraBackend | None = None
        self._control: _ControlWorker | None = None
        self._sampler: _ActionSampler | None = None
        self._draft: _EpisodeDraft | None = None
        self._latest_control: _ControlSample | None = None
        self._last_error: str | None = None
        self._error_source: str | None = None
        self._connected_at_s: float | None = None
        self._control_stage = "not started"
        self._control_stage_started_s: float | None = None
        self._control_periods: deque[float] = deque(maxlen=telemetry_window)
        self._observation_latencies: deque[float] = deque(maxlen=telemetry_window)
        self._leader_latencies: deque[float] = deque(maxlen=telemetry_window)
        self._send_latencies: deque[float] = deque(maxlen=telemetry_window)
        self._missed_control_deadlines = 0
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    @property
    def state(self) -> RuntimeState:
        with self._lock:
            return self._state

    def connect(self, config: RuntimeConfig) -> RuntimeSnapshot:
        """Connect once; repeated requests while connected return the current owner."""
        with self._command_lock:
            with self._lock:
                if self._state in {RuntimeState.IDLE, RuntimeState.RECORDING}:
                    return self.get_snapshot()
                if self._state is not RuntimeState.DISCONNECTED:
                    raise RuntimeError(f"Cannot connect while runtime is {self._state.value}.")
                self._state = RuntimeState.CONNECTING
                self._config = config
                self._last_error = None
                self._error_source = None
                self._reset_telemetry_locked()

            hardware: HardwareBundle | None = None
            camera: CameraBackend | None = None
            teleop_connected = False
            robot_connected = False
            try:
                hardware = self._hardware_factory(config)
                camera = self._camera_factory(config, self._worker_failed)
                camera.start()
                hardware.teleop.connect()
                teleop_connected = True
                hardware.robot.connect()
                robot_connected = True

                control = _ControlWorker(
                    hardware,
                    config.control_hz,
                    camera.publish_observation,
                    self._publish_control,
                    self._publish_control_progress,
                    self._worker_failed,
                )
                with self._lock:
                    if self._state is RuntimeState.ERROR:
                        raise RuntimeError(self._last_error or "A worker failed during connection.")
                    self._hardware = hardware
                    self._camera = camera
                    self._control = control
                    self._connected_at_s = time.perf_counter()
                    self._state = RuntimeState.IDLE
                control.start()
                self._start_watchdog()
                return self.get_snapshot()
            except BaseException as exc:
                if camera is not None:
                    try:
                        camera.stop()
                    except BaseException:
                        pass
                if hardware is not None:
                    if robot_connected:
                        try:
                            hardware.robot.disconnect()
                        except BaseException:
                            pass
                    if teleop_connected:
                        try:
                            hardware.teleop.disconnect()
                        except BaseException:
                            pass
                with self._lock:
                    self._hardware = None
                    self._camera = None
                    self._control = None
                    self._state = RuntimeState.DISCONNECTED
                    self._last_error = str(exc)
                    self._error_source = "connect"
                raise

    def disconnect(self) -> RuntimeSnapshot:
        """Stop control first, then let the Piper plugin execute its rest trajectory."""
        with self._command_lock:
            if self.state is RuntimeState.RECORDING:
                self.stop_episode()
            with self._lock:
                if self._state is RuntimeState.DISCONNECTED:
                    return self.get_snapshot()
                if self._state in {RuntimeState.CONNECTING, RuntimeState.STOPPING}:
                    raise RuntimeError(f"Cannot disconnect while runtime is {self._state.value}.")
                self._state = RuntimeState.STOPPING
                control = self._control
                camera = self._camera
                hardware = self._hardware
                sampler = self._sampler

            self._stop_watchdog()
            control_stopped = control is None or control.stop()
            if sampler is not None and camera is not None:
                try:
                    actions, missed = sampler.stop()
                    video = camera.stop_recording()
                    with self._lock:
                        draft = self._draft
                        if draft is not None and draft.recording:
                            draft.actions = actions
                            draft.action_missed_deadlines = missed
                            draft.video = video
                            draft.stopped_s = time.perf_counter()
                            draft.recording = False
                        self._sampler = None
                except BaseException as exc:
                    with self._lock:
                        self._last_error = f"Could not finish active recording: {exc}"
                        self._error_source = "disconnect"

            warnings: list[str] = []
            if camera is not None:
                try:
                    camera.stop()
                except BaseException as exc:
                    warnings.append(f"camera: {exc}")

            if not control_stopped:
                with self._lock:
                    self._state = RuntimeState.ERROR
                    self._last_error = (
                        "Control worker did not stop; safe rest was not started to avoid "
                        "concurrent robot commands. Use the physical emergency stop."
                    )
                    self._error_source = "disconnect"
                raise RuntimeError(self._last_error)

            if hardware is not None:
                try:
                    hardware.teleop.disconnect()
                except BaseException as exc:
                    warnings.append(f"leader: {exc}")
                try:
                    hardware.robot.disconnect()
                except BaseException as exc:
                    with self._lock:
                        self._state = RuntimeState.ERROR
                        self._last_error = f"Robot safe disconnect failed: {exc}"
                        self._error_source = "disconnect"
                    raise RuntimeError(self._last_error) from exc

            with self._lock:
                self._hardware = None
                self._camera = None
                self._control = None
                self._sampler = None
                self._connected_at_s = None
                self._latest_control = None
                self._state = RuntimeState.DISCONNECTED
                if warnings:
                    self._last_error = "; ".join(warnings)
                    self._error_source = "disconnect_warning"
            return self.get_snapshot()

    def emergency_stop(self) -> RuntimeSnapshot:
        """Stop scheduling and disable the Piper immediately, without a rest move."""
        with self._command_lock:
            with self._lock:
                if self._state is RuntimeState.DISCONNECTED:
                    return self.get_snapshot()
                self._state = RuntimeState.STOPPING
                control = self._control
                sampler = self._sampler
                camera = self._camera
                hardware = self._hardware

            self._stop_watchdog()
            if control is not None:
                control.request_stop()
                control.stop(timeout_s=1.0)
            if sampler is not None:
                try:
                    sampler.stop(timeout_s=1.0)
                except BaseException:
                    pass
            if camera is not None:
                try:
                    video = camera.stop_recording()
                    self._finish_draft_after_stop(video, sampler)
                except BaseException:
                    pass
                try:
                    camera.stop()
                except BaseException:
                    pass
            if hardware is not None:
                if hardware.emergency_disable is not None:
                    try:
                        hardware.emergency_disable()
                    except BaseException:
                        pass
                try:
                    hardware.teleop.disconnect()
                except BaseException:
                    pass

            with self._lock:
                self._hardware = None
                self._camera = None
                self._control = None
                self._sampler = None
                self._connected_at_s = None
                self._latest_control = None
                self._state = RuntimeState.DISCONNECTED
                self._last_error = "Software emergency stop requested; verify the physical robot state."
                self._error_source = "emergency_stop"
            return self.get_snapshot()

    def start_episode(self, config: RecordingConfig) -> RuntimeSnapshot:
        with self._command_lock:
            with self._lock:
                if self._state is not RuntimeState.IDLE:
                    raise RuntimeError("An episode can only start while the runtime is IDLE.")
                if self._draft is not None:
                    raise RuntimeError("Save or discard the current episode before starting another.")
                camera = self._camera
            if camera is None:
                raise RuntimeError("Camera is not connected.")

            config.output_dir.mkdir(parents=True, exist_ok=True)
            episode_index = _next_episode_index(config.output_dir)
            pending_dir = config.output_dir / f".episode_{episode_index:03d}.pending"
            pending_dir.mkdir(parents=False, exist_ok=False)
            avi_path = pending_dir / "video.avi"
            started_s = time.perf_counter()
            draft = _EpisodeDraft(
                config=config,
                episode_index=episode_index,
                pending_dir=pending_dir,
                avi_path=avi_path,
                started_s=started_s,
            )
            sampler = _ActionSampler(
                config.action_sample_hz,
                started_s,
                self._get_latest_control,
                camera.record_frame,
                self._worker_failed,
            )
            try:
                camera.start_recording(
                    avi_path,
                    started_s,
                    config.action_sample_hz,
                )
                with self._lock:
                    self._draft = draft
                    self._sampler = sampler
                    self._state = RuntimeState.RECORDING
                sampler.start()
            except BaseException:
                try:
                    camera.stop_recording()
                except BaseException:
                    pass
                shutil.rmtree(pending_dir, ignore_errors=True)
                with self._lock:
                    self._draft = None
                    self._sampler = None
                    self._state = RuntimeState.IDLE
                raise
            return self.get_snapshot()

    def stop_episode(self) -> RuntimeSnapshot:
        with self._command_lock:
            with self._lock:
                if self._state is not RuntimeState.RECORDING:
                    return self.get_snapshot()
                sampler = self._sampler
                camera = self._camera
            if sampler is None or camera is None:
                raise RuntimeError("Recording workers are unavailable.")

            try:
                actions, missed = sampler.stop()
                video = camera.stop_recording()
                if len(actions) != video.frame_count:
                    raise RuntimeError(
                        "Aligned recording failed: "
                        f"{len(actions)} actions were paired but {video.frame_count} "
                        "video frames were written."
                    )
                with self._lock:
                    draft = self._draft
                    if draft is not None:
                        draft.actions = actions
                        draft.action_missed_deadlines = missed
                        draft.video = video
                        draft.stopped_s = time.perf_counter()
                        draft.recording = False
                    self._sampler = None
                    self._state = RuntimeState.IDLE
            except BaseException as exc:
                self._worker_failed("recording_stop", exc)
                raise
            return self.get_snapshot()

    def save_episode(self) -> Path:
        with self._command_lock:
            with self._lock:
                if self._state is RuntimeState.RECORDING:
                    raise RuntimeError("Stop the episode before saving it.")
                draft = self._draft
                runtime_config = self._config
            if draft is None or draft.recording or draft.video is None:
                raise RuntimeError("There is no completed episode to save.")

            final_dir = draft.config.output_dir / f"episode_{draft.episode_index:03d}"
            if final_dir.exists():
                raise FileExistsError(f"Episode directory already exists: {final_dir}")
            mp4_path = draft.pending_dir / "video.mp4"
            self._video_converter(draft.avi_path, mp4_path)

            action_fields = [
                "sample_index",
                "frame_index",
                "sample_timestamp_s",
                "source_control_sequence",
                "source_control_timestamp_s",
                "capture_sequence",
                "frame_timestamp_s",
                "action_frame_delta_ms",
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
            frame_timestamps = draft.video.frame_timestamps
            if len(draft.actions) != len(frame_timestamps):
                raise RuntimeError(
                    "Cannot save an unaligned episode: "
                    f"{len(draft.actions)} actions and {len(frame_timestamps)} frames."
                )
            for index, (action, frame) in enumerate(
                zip(draft.actions, frame_timestamps, strict=True)
            ):
                if (
                    action.get("sample_index") != index
                    or action.get("frame_index") != index
                    or frame.get("frame_index") != index
                ):
                    raise RuntimeError(
                        f"Cannot save episode: alignment index {index} is inconsistent."
                    )
            with (draft.pending_dir / "actions.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=action_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(draft.actions)
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
                        "observation_action_delta_ms": row[
                            "observation_action_delta_ms"
                        ],
                        **{
                            key: row[column]
                            for key, column in zip(
                                ACTION_KEYS, OBSERVATION_COLUMNS, strict=True
                            )
                        },
                    }
                    for row in draft.actions
                )
            with (draft.pending_dir / "frame_timestamps.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["frame_index", "capture_sequence", "timestamp_s"],
                )
                writer.writeheader()
                writer.writerows(frame_timestamps)

            stopped_s = draft.stopped_s or time.perf_counter()
            meta = {
                "task": draft.config.task,
                "episode_index": draft.episode_index,
                "duration_s": round(stopped_s - draft.started_s, 6),
                "action_sample_hz_target": draft.config.action_sample_hz,
                "action_samples": len(draft.actions),
                "action_missed_deadlines": draft.action_missed_deadlines,
                "camera_capture_fps_target": (
                    runtime_config.camera.fps if runtime_config else None
                ),
                "video_fps_target": draft.config.action_sample_hz,
                "video_frames": draft.video.frame_count,
                "video_dropped_frames": draft.video.dropped_frames,
                "observation_samples": len(draft.actions),
                "observation_source": (
                    "measured Piper state captured before the paired action was sent"
                ),
                "control_hz_target": runtime_config.control_hz if runtime_config else None,
                "gripper_max_mm": (
                    runtime_config.gripper_max_mm if runtime_config else None
                ),
                "timestamp_clock": "time.perf_counter monotonic seconds relative to episode start",
                "alignment": (
                    "one action and measured observation row per video frame; "
                    "actions.csv and observations.csv frame_index map directly to "
                    "frame_timestamps.csv frame_index"
                ),
            }
            (draft.pending_dir / "meta.json").write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )
            draft.avi_path.unlink(missing_ok=True)
            draft.pending_dir.rename(final_dir)
            with self._lock:
                self._draft = None
            return final_dir

    def discard_episode(self) -> RuntimeSnapshot:
        with self._command_lock:
            if self.state is RuntimeState.RECORDING:
                self.stop_episode()
            with self._lock:
                draft = self._draft
            if draft is not None:
                shutil.rmtree(draft.pending_dir, ignore_errors=True)
                with self._lock:
                    self._draft = None
            return self.get_snapshot()

    def set_preview_enabled(self, enabled: bool) -> RuntimeSnapshot:
        with self._command_lock:
            with self._lock:
                camera = self._camera
            if camera is not None:
                camera.set_preview_enabled(enabled)
            return self.get_snapshot()

    def get_snapshot(self) -> RuntimeSnapshot:
        now = time.perf_counter()
        with self._lock:
            config = self._config
            latest = self._latest_control
            state = self._state
            camera = self._camera
            sampler = self._sampler
            draft = self._draft
            periods = list(self._control_periods)
            observation_latencies = list(self._observation_latencies)
            leader_latencies = list(self._leader_latencies)
            send_latencies = list(self._send_latencies)
            missed_control = self._missed_control_deadlines
            last_error = self._last_error
            error_source = self._error_source
            control_stage = self._control_stage
            control_stage_started_s = self._control_stage_started_s

        camera_snapshot = camera.snapshot() if camera is not None else {}
        action_samples, action_misses = sampler.snapshot() if sampler is not None else (0, 0)
        draft_snapshot: DraftSnapshot | None = None
        recording_elapsed_s = 0.0
        recording_duration_s: float | None = None
        if draft is not None:
            end_s = now if draft.recording else (draft.stopped_s or now)
            recording_elapsed_s = max(end_s - draft.started_s, 0.0)
            recording_duration_s = draft.config.duration_s
            video = draft.video
            draft_snapshot = DraftSnapshot(
                episode_index=draft.episode_index,
                output_dir=str(draft.config.output_dir),
                task=draft.config.task,
                action_samples=action_samples if draft.recording else len(draft.actions),
                video_frames=(
                    int(camera_snapshot.get("written_frames", 0) or 0)
                    if draft.recording
                    else (video.frame_count if video else 0)
                ),
                video_dropped_frames=(
                    int(camera_snapshot.get("dropped_frames", 0) or 0)
                    if draft.recording
                    else (video.dropped_frames if video else 0)
                ),
                duration_s=recording_elapsed_s,
                recording=draft.recording,
            )
            if not draft.recording:
                action_samples = len(draft.actions)
                action_misses = draft.action_missed_deadlines

        average_period = sum(periods) / len(periods) if periods else 0.0
        effective_hz = 1.0 / average_period if average_period > 0 else 0.0
        minimum_hz = 1.0 / max(periods) if periods and max(periods) > 0 else 0.0
        return RuntimeSnapshot(
            state=state,
            target_control_hz=config.control_hz if config else 0.0,
            effective_control_hz=effective_hz,
            minimum_effective_hz=minimum_hz,
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
            leader_latency=_timing_summary(leader_latencies),
            robot_send_latency=_timing_summary(send_latencies),
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
            action_samples=action_samples,
            action_sample_hz=(
                action_samples / recording_elapsed_s if recording_elapsed_s > 0 else 0.0
            ),
            missed_action_deadlines=action_misses,
            recording_elapsed_s=recording_elapsed_s,
            recording_duration_s=recording_duration_s,
            draft=draft_snapshot,
            last_error=last_error,
            error_source=error_source,
        )

    def _publish_control(self, **values: Any) -> None:
        with self._lock:
            missed = int(values.get("missed_deadlines", 0))
            if missed:
                self._missed_control_deadlines += missed
                return
            sequence = int(values["sequence"])
            timestamp_s = float(values["timestamp_s"])
            self._latest_control = _ControlSample(
                sequence=sequence,
                timestamp_s=timestamp_s,
                observation_timestamp_s=float(values["observation_timestamp_s"]),
                action=dict(values["action"]),
                observation=dict(values["observation"]),
                frame=values.get("frame"),
            )
            period_s = values.get("period_s")
            if period_s is not None:
                self._control_periods.append(float(period_s))
            observation_latency_s = values["observation_latency_s"]
            if observation_latency_s is not None:
                self._observation_latencies.append(float(observation_latency_s))
            self._leader_latencies.append(float(values["leader_latency_s"]))
            self._send_latencies.append(float(values["send_latency_s"]))

    def _publish_control_progress(self, stage: str) -> None:
        with self._lock:
            self._control_stage = stage
            self._control_stage_started_s = time.perf_counter()

    def _get_latest_control(self) -> _ControlSample | None:
        with self._lock:
            sample = self._latest_control
            if sample is None:
                return None
            return _ControlSample(
                sample.sequence,
                sample.timestamp_s,
                sample.observation_timestamp_s,
                dict(sample.action),
                dict(sample.observation),
                sample.frame,
            )

    def _worker_failed(self, source: str, exc: BaseException) -> None:
        with self._lock:
            if self._state in {RuntimeState.DISCONNECTED, RuntimeState.STOPPING}:
                return
            self._state = RuntimeState.ERROR
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._error_source = source
            control = self._control
        if control is not None:
            control.request_stop()

    def _start_watchdog(self) -> None:
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="teleop-watchdog",
            daemon=True,
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
            should_stop_episode = False
            control_to_stop: _ControlWorker | None = None
            with self._lock:
                state = self._state
                config = self._config
                latest = self._latest_control
                connected_at = self._connected_at_s
                draft = self._draft
                if (
                    state is RuntimeState.RECORDING
                    and draft is not None
                    and draft.config.duration_s is not None
                    and now - draft.started_s >= draft.config.duration_s
                ):
                    should_stop_episode = True
                active = state in {RuntimeState.IDLE, RuntimeState.RECORDING}
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
                stage = self._control_stage
                stage_started_s = self._control_stage_started_s
                if stale:
                    stage_age_s = now - stage_started_s if stage_started_s is not None else 0.0
                    update_kind = (
                        "initial control update" if first_update_pending else "control update"
                    )
                    self._state = RuntimeState.ERROR
                    self._last_error = (
                        "RuntimeError: "
                        f"No successful {update_kind} for {update_age_s:.3f}s "
                        f"(limit {timeout_s:.3f}s). Control worker was in "
                        f"{stage} for {stage_age_s:.3f}s."
                    )
                    self._error_source = "watchdog"
                    control_to_stop = self._control
            if control_to_stop is not None:
                control_to_stop.request_stop()
            if should_stop_episode:
                try:
                    self.stop_episode()
                except BaseException as exc:
                    self._worker_failed("episode_timeout", exc)

    def _finish_draft_after_stop(
        self,
        video: VideoResult,
        sampler: _ActionSampler | None,
    ) -> None:
        actions: list[dict[str, Any]] = []
        missed = 0
        if sampler is not None:
            try:
                actions, missed = sampler.stop(timeout_s=0.1)
            except BaseException:
                pass
        with self._lock:
            draft = self._draft
            if draft is not None and draft.recording:
                draft.actions = actions
                draft.action_missed_deadlines = missed
                draft.video = video
                draft.stopped_s = time.perf_counter()
                draft.recording = False

    def _reset_telemetry_locked(self) -> None:
        self._latest_control = None
        self._control_periods.clear()
        self._observation_latencies.clear()
        self._leader_latencies.clear()
        self._send_latencies.clear()
        self._missed_control_deadlines = 0
        self._control_stage = "not started"
        self._control_stage_started_s = None


_RUNTIME_PROCESS_OPERATIONS = frozenset(
    {
        "connect",
        "disconnect",
        "emergency_stop",
        "start_episode",
        "stop_episode",
        "save_episode",
        "discard_episode",
        "set_preview_enabled",
        "get_snapshot",
    }
)


def _snapshot_for_ipc(snapshot: RuntimeSnapshot) -> RuntimeSnapshot:
    """Replace read-only mapping proxies with picklable plain dictionaries."""
    return replace(
        snapshot,
        latest_action=dict(snapshot.latest_action),
        latest_observation=dict(snapshot.latest_observation),
    )


def _runtime_process_main(connection: Any) -> None:
    """Own all hardware in a Python process that never imports Streamlit."""
    runtime = TeleopRuntime()
    try:
        while True:
            request_id, operation, args, kwargs = connection.recv()
            if operation == "shutdown":
                try:
                    if runtime.state is not RuntimeState.DISCONNECTED:
                        runtime.disconnect()
                    result: Any = runtime.get_snapshot()
                    connection.send(
                        (request_id, "ok", _snapshot_for_ipc(result))
                    )
                except BaseException as exc:
                    connection.send(
                        (
                            request_id,
                            "error",
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
                return
            if operation not in _RUNTIME_PROCESS_OPERATIONS:
                connection.send(
                    (request_id, "error", f"Unsupported runtime operation: {operation}")
                )
                continue
            try:
                result = getattr(runtime, operation)(*args, **kwargs)
                if isinstance(result, RuntimeSnapshot):
                    result = _snapshot_for_ipc(result)
                connection.send((request_id, "ok", result))
            except BaseException as exc:
                connection.send(
                    (
                        request_id,
                        "error",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
    except (EOFError, BrokenPipeError):
        pass
    finally:
        try:
            if runtime.state is not RuntimeState.DISCONNECTED:
                runtime.disconnect()
        except BaseException:
            try:
                runtime.emergency_stop()
            except BaseException:
                pass
        connection.close()


class ProcessTeleopRuntime:
    """Streamlit-side proxy for a CLI-equivalent runtime in a spawned process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._context = multiprocessing.get_context("spawn")
        self._connection: Any = None
        self._process: multiprocessing.Process | None = None
        self._request_id = 0
        self._last_snapshot = TeleopRuntime().get_snapshot()
        self._last_snapshot_received_s = 0.0

    @property
    def state(self) -> RuntimeState:
        with self._lock:
            return self._last_snapshot.state

    @property
    def process_pid(self) -> int | None:
        with self._lock:
            process = self._process
            return process.pid if process is not None and process.is_alive() else None

    def connect(self, config: RuntimeConfig) -> RuntimeSnapshot:
        return self._snapshot_rpc("connect", config, timeout_s=45.0)

    def disconnect(self) -> RuntimeSnapshot:
        if not self._has_process():
            return self.get_snapshot()
        return self._snapshot_rpc("disconnect", timeout_s=60.0)

    def emergency_stop(self) -> RuntimeSnapshot:
        if not self._has_process():
            return self.get_snapshot()
        return self._snapshot_rpc("emergency_stop", timeout_s=10.0)

    def start_episode(self, config: RecordingConfig) -> RuntimeSnapshot:
        return self._snapshot_rpc("start_episode", config, timeout_s=10.0)

    def stop_episode(self) -> RuntimeSnapshot:
        return self._snapshot_rpc("stop_episode", timeout_s=20.0)

    def save_episode(self) -> Path:
        result = self._rpc("save_episode", timeout_s=60.0)
        if not isinstance(result, Path):
            raise RuntimeError("Runtime process returned an invalid saved episode path.")
        return result

    def discard_episode(self) -> RuntimeSnapshot:
        if not self._has_process():
            return self.get_snapshot()
        return self._snapshot_rpc("discard_episode", timeout_s=20.0)

    def set_preview_enabled(self, enabled: bool) -> RuntimeSnapshot:
        if not self._has_process():
            return self.get_snapshot()
        return self._snapshot_rpc("set_preview_enabled", enabled, timeout_s=5.0)

    def get_snapshot(self) -> RuntimeSnapshot:
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

    def _snapshot_rpc(
        self,
        operation: str,
        *args: Any,
        timeout_s: float,
    ) -> RuntimeSnapshot:
        result = self._rpc(operation, *args, timeout_s=timeout_s)
        if not isinstance(result, RuntimeSnapshot):
            raise RuntimeError(
                f"Runtime process returned an invalid response for {operation}."
            )
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
            name="streamlit-teleop-runtime",
            daemon=True,
        )
        process.start()
        child_connection.close()
        self._connection = parent_connection
        self._process = process

    def _has_process(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.is_alive()

    def _process_error_snapshot(self, exc: BaseException) -> RuntimeSnapshot:
        with self._lock:
            snapshot = replace(
                self._last_snapshot,
                state=RuntimeState.ERROR,
                last_error=f"{type(exc).__name__}: {exc}",
                error_source="runtime_process",
            )
            self._last_snapshot = snapshot
            self._last_snapshot_received_s = time.perf_counter()
            return snapshot


_PROCESS_RUNTIME: ProcessTeleopRuntime | None = None
_PROCESS_RUNTIME_LOCK = threading.Lock()


def get_runtime() -> ProcessTeleopRuntime:
    """Return the shared proxy whose child process exclusively owns hardware."""
    global _PROCESS_RUNTIME
    with _PROCESS_RUNTIME_LOCK:
        if _PROCESS_RUNTIME is None:
            _PROCESS_RUNTIME = ProcessTeleopRuntime()
        return _PROCESS_RUNTIME


def _shutdown_process_runtime() -> None:
    runtime = _PROCESS_RUNTIME
    if runtime is not None:
        runtime.shutdown()


atexit.register(_shutdown_process_runtime)
