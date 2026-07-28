from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from ui.teleop_runtime import (
    ACTION_KEYS,
    CameraSettings,
    HardwareBundle,
    ProcessTeleopRuntime,
    RecordingConfig,
    RuntimeConfig,
    RuntimeState,
    TeleopRuntime,
    VideoResult,
    _FrameReference,
)


def wait_until(predicate, timeout_s: float = 2.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Condition was not reached before timeout.")


class FakeRobot:
    def __init__(
        self,
        send_delay_s: float = 0.0,
        fail_after: int | None = None,
        first_observation_delay_s: float = 0.0,
        observation_stall_after: int | None = None,
        observation_stall_s: float = 0.0,
        events: list[str] | None = None,
    ) -> None:
        self.send_delay_s = send_delay_s
        self.fail_after = fail_after
        self.first_observation_delay_s = first_observation_delay_s
        self.observation_stall_after = observation_stall_after
        self.observation_stall_s = observation_stall_s
        self.events = events
        self.connected = False
        self.connect_count = 0
        self.disconnect_count = 0
        self.observation_count = 0
        self.sent_actions: list[dict[str, float]] = []
        self.send_times: list[float] = []
        self._lock = threading.Lock()

    def connect(self) -> None:
        self.connected = True
        self.connect_count += 1

    def disconnect(self) -> None:
        self.connected = False
        self.disconnect_count += 1

    def get_observation(self) -> dict[str, float]:
        if self.events is not None:
            self.events.append("robot.get_observation")
        with self._lock:
            self.observation_count += 1
            observation_count = self.observation_count
        if observation_count == 1 and self.first_observation_delay_s:
            time.sleep(self.first_observation_delay_s)
        if (
            self.observation_stall_after is not None
            and observation_count >= self.observation_stall_after
        ):
            time.sleep(self.observation_stall_s)
        return {key: 0.0 for key in ACTION_KEYS}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if self.events is not None:
            self.events.append("robot.send_action")
        if self.send_delay_s:
            time.sleep(self.send_delay_s)
        with self._lock:
            if self.fail_after is not None and len(self.sent_actions) >= self.fail_after:
                raise RuntimeError("synthetic CAN failure")
            self.sent_actions.append(dict(action))
            self.send_times.append(time.perf_counter())
        return dict(action)

    def send_count(self) -> int:
        with self._lock:
            return len(self.sent_actions)


class FakeTeleop:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events
        self.connected = False
        self.connect_count = 0
        self.disconnect_count = 0
        self.read_count = 0

    def connect(self) -> None:
        self.connected = True
        self.connect_count += 1

    def disconnect(self) -> None:
        self.connected = False
        self.disconnect_count += 1

    def get_action(self) -> dict[str, float]:
        if self.events is not None:
            self.events.append("teleop.get_action")
        self.read_count += 1
        return {key: float(self.read_count) for key in ACTION_KEYS}


class FakeCamera:
    def __init__(self, config: RuntimeConfig, on_error) -> None:
        self.config = config
        self.on_error = on_error
        self.started = False
        self.recording = False
        self.recording_path: Path | None = None
        self.recording_started_s = 0.0
        self.preview_enabled = config.camera.preview_enabled
        self.published_observations = 0
        self.recorded_frames: list[dict[str, float | int]] = []
        self.output_fps = 0.0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.recording = False
        self.started = False

    def set_preview_enabled(self, enabled: bool) -> None:
        self.preview_enabled = enabled

    def publish_observation(
        self, observation: dict[str, Any], timestamp_s: float
    ) -> _FrameReference:
        del observation
        self.published_observations += 1
        return _FrameReference(
            self.published_observations,
            timestamp_s,
            object(),
        )

    def start_recording(
        self, path: Path, episode_start_s: float, output_fps: float
    ) -> None:
        self.recording = True
        self.recording_path = path
        self.recording_started_s = episode_start_s
        self.recorded_frames = []
        self.output_fps = output_fps
        path.write_bytes(b"fake avi")

    def record_frame(
        self, frame_index: int, reference: _FrameReference
    ) -> bool:
        if not self.recording:
            return False
        self.recorded_frames.append(
            {
                "frame_index": frame_index,
                "capture_sequence": reference.capture_sequence,
                "timestamp_s": round(
                    reference.timestamp_s - self.recording_started_s,
                    6,
                ),
            }
        )
        return True

    def stop_recording(self) -> VideoResult:
        if not self.recording:
            return VideoResult(0, 0, (), 0.0)
        self.recording = False
        elapsed = max(time.perf_counter() - self.recording_started_s, 0.001)
        timestamps = tuple(self.recorded_frames)
        return VideoResult(len(timestamps), 0, timestamps, elapsed)

    def snapshot(self) -> dict[str, float | int | None]:
        return {
            "frame_age_s": 0.001 if self.started else None,
            "capture_hz": self.config.camera.fps if self.started else 0.0,
            "frame_count": 1 if self.started else 0,
            "encode_ms": 0.2,
            "queue_depth": 0,
            "dropped_frames": 0,
            "written_frames": len(self.recorded_frames),
        }


class RuntimeHarness:
    def __init__(
        self,
        send_delay_s: float = 0.0,
        fail_after: int | None = None,
        transform: bool = False,
        first_observation_delay_s: float = 0.0,
        observation_stall_after: int | None = None,
        observation_stall_s: float = 0.0,
    ) -> None:
        self.events: list[str] = []
        self.robot = FakeRobot(
            send_delay_s,
            fail_after,
            first_observation_delay_s,
            observation_stall_after,
            observation_stall_s,
            self.events,
        )
        self.teleop = FakeTeleop(self.events)
        self.hardware_calls = 0
        self.emergency_calls = 0
        self.transform = transform

    def hardware_factory(self, _config: RuntimeConfig) -> HardwareBundle:
        self.hardware_calls += 1

        def teleop_processor(values: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
            self.events.append("teleop_processor")
            action, _observation = values
            result = dict(action)
            if self.transform:
                result["joint_1.pos"] += 1
            return result

        def robot_processor(values: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
            self.events.append("robot_processor")
            action, _observation = values
            result = dict(action)
            if self.transform:
                result["joint_1.pos"] += 1
            return result

        def emergency_disable() -> None:
            self.emergency_calls += 1
            self.robot.connected = False

        return HardwareBundle(
            self.robot,
            self.teleop,
            teleop_processor,
            robot_processor,
            emergency_disable,
        )

    @staticmethod
    def camera_factory(config: RuntimeConfig, on_error) -> FakeCamera:
        return FakeCamera(config, on_error)

    @staticmethod
    def video_converter(source: Path, destination: Path) -> None:
        destination.write_bytes(b"fake mp4:" + source.read_bytes())

    def make_runtime(self) -> TeleopRuntime:
        return TeleopRuntime(
            hardware_factory=self.hardware_factory,
            camera_factory=self.camera_factory,
            video_converter=self.video_converter,
            telemetry_window=500,
        )


class TeleopRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temporary_directory.name)
        self.runtimes: list[TeleopRuntime] = []

    def tearDown(self) -> None:
        for runtime in self.runtimes:
            try:
                runtime.discard_episode()
            except Exception:
                pass
            try:
                runtime.emergency_stop()
            except Exception:
                pass
        self.temporary_directory.cleanup()

    def make_runtime(self, harness: RuntimeHarness) -> TeleopRuntime:
        runtime = harness.make_runtime()
        self.runtimes.append(runtime)
        return runtime

    def config(self, control_hz: float = 100.0) -> RuntimeConfig:
        return RuntimeConfig(
            control_hz=control_hz,
            watchdog_timeout_s=1.0,
            camera=CameraSettings(fps=30.0),
            preview_path=self.output_dir / "static" / "live.jpg",
        )

    def test_runtime_defaults_to_large_jaw_and_rejects_unknown_travel(self) -> None:
        self.assertEqual(RuntimeConfig().gripper_max_mm, 100.0)
        self.assertEqual(RuntimeConfig(gripper_max_mm=70.0).gripper_max_mm, 70.0)
        with self.assertRaisesRegex(ValueError, "70 or 100"):
            RuntimeConfig(gripper_max_mm=80.0)

    def test_factory_processors_and_single_hardware_owner_are_used(self) -> None:
        harness = RuntimeHarness(transform=True)
        runtime = self.make_runtime(harness)
        config = self.config()

        runtime.connect(config)
        runtime.connect(config)
        wait_until(lambda: harness.robot.send_count() >= 3)

        self.assertEqual(harness.hardware_calls, 1)
        self.assertEqual(harness.robot.connect_count, 1)
        self.assertEqual(harness.teleop.connect_count, 1)
        self.assertGreaterEqual(harness.robot.observation_count, 1)
        first_action = harness.robot.sent_actions[0]
        self.assertEqual(first_action["joint_1.pos"], 3.0)

        runtime.disconnect()
        runtime.disconnect()
        self.assertEqual(harness.robot.disconnect_count, 1)
        self.assertEqual(harness.teleop.disconnect_count, 1)

    def test_recording_samples_without_reducing_control_rate_and_saves_outputs(self) -> None:
        harness = RuntimeHarness()
        runtime = self.make_runtime(harness)
        runtime.connect(self.config(control_hz=100.0))
        wait_until(lambda: harness.robot.send_count() >= 8)
        commands_before = harness.robot.send_count()

        runtime.start_episode(
            RecordingConfig(
                output_dir=self.output_dir,
                task="test task",
                action_sample_hz=20.0,
                duration_s=0.35,
            )
        )
        wait_until(lambda: runtime.state is RuntimeState.IDLE, timeout_s=1.5)
        commands_during = harness.robot.send_count() - commands_before
        snapshot = runtime.get_snapshot()

        self.assertEqual(snapshot.target_control_hz, 100.0)
        self.assertGreaterEqual(commands_during, 25)
        self.assertIsNotNone(snapshot.draft)
        self.assertGreaterEqual(snapshot.draft.action_samples, 5)
        self.assertLessEqual(snapshot.draft.action_samples, 10)
        self.assertEqual(snapshot.state, RuntimeState.IDLE)

        final_dir = runtime.save_episode()
        self.assertEqual(final_dir, self.output_dir / "episode_000")
        self.assertTrue((final_dir / "video.mp4").is_file())
        self.assertTrue((final_dir / "actions.csv").is_file())
        self.assertTrue((final_dir / "observations.csv").is_file())
        self.assertTrue((final_dir / "frame_timestamps.csv").is_file())
        self.assertTrue((final_dir / "meta.json").is_file())
        self.assertFalse((final_dir / "video.avi").exists())

        with (final_dir / "actions.csv").open(newline="") as handle:
            action_rows = list(csv.DictReader(handle))
        with (final_dir / "observations.csv").open(newline="") as handle:
            observation_rows = list(csv.DictReader(handle))
        with (final_dir / "frame_timestamps.csv").open(newline="") as handle:
            frame_rows = list(csv.DictReader(handle))
        meta = json.loads((final_dir / "meta.json").read_text())

        self.assertEqual(len(action_rows), len(observation_rows))
        self.assertEqual(len(observation_rows), len(frame_rows))
        self.assertGreaterEqual(len(action_rows), 5)
        for index, (action, observation, frame) in enumerate(
            zip(action_rows, observation_rows, frame_rows, strict=True)
        ):
            self.assertEqual(int(action["sample_index"]), index)
            self.assertEqual(int(action["frame_index"]), index)
            self.assertEqual(int(observation["sample_index"]), index)
            self.assertEqual(int(observation["frame_index"]), index)
            self.assertEqual(int(frame["frame_index"]), index)
            self.assertEqual(action["capture_sequence"], frame["capture_sequence"])
            self.assertEqual(
                action["source_control_sequence"],
                observation["source_control_sequence"],
            )
            self.assertGreaterEqual(float(action["source_control_timestamp_s"]), 0.0)
            self.assertGreaterEqual(float(observation["observation_timestamp_s"]), 0.0)
            self.assertGreaterEqual(float(action["frame_timestamp_s"]), 0.0)
            for key in ACTION_KEYS:
                self.assertGreater(float(action[key]), 0.0)
                self.assertEqual(float(observation[key]), 0.0)
        self.assertEqual(meta["action_samples"], meta["video_frames"])
        self.assertEqual(meta["observation_samples"], meta["video_frames"])
        self.assertIn("measured Piper state", meta["observation_source"])
        self.assertEqual(meta["video_fps_target"], 20.0)
        self.assertEqual(meta["camera_capture_fps_target"], 30.0)
        self.assertEqual(meta["gripper_max_mm"], 100.0)

    def test_slow_robot_send_counts_deadline_misses_without_replaying_commands(self) -> None:
        harness = RuntimeHarness(send_delay_s=0.012)
        runtime = self.make_runtime(harness)
        runtime.connect(self.config(control_hz=200.0))
        wait_until(lambda: harness.robot.send_count() >= 8)
        snapshot = runtime.get_snapshot()

        self.assertGreater(snapshot.missed_control_deadlines, 0)
        sequences = list(range(1, snapshot.control_sequence + 1))
        self.assertEqual(len(sequences), harness.robot.send_count())
        self.assertEqual(sequences[-1], snapshot.control_sequence)

    def test_control_loop_matches_cli_observation_action_send_order(self) -> None:
        harness = RuntimeHarness()
        runtime = self.make_runtime(harness)
        runtime.connect(self.config(control_hz=100.0))

        wait_until(lambda: harness.robot.send_count() >= 30)
        self.assertEqual(
            harness.events[:5],
            [
                "robot.get_observation",
                "teleop.get_action",
                "teleop_processor",
                "robot_processor",
                "robot.send_action",
            ],
        )
        self.assertGreaterEqual(harness.robot.observation_count, 30)
        self.assertLessEqual(
            abs(harness.robot.observation_count - harness.robot.send_count()),
            1,
        )
        camera = runtime._camera
        self.assertIsInstance(camera, FakeCamera)
        self.assertGreaterEqual(camera.published_observations, 30)

    def test_startup_watchdog_grace_allows_a_slow_first_observation(self) -> None:
        harness = RuntimeHarness(first_observation_delay_s=0.15)
        runtime = self.make_runtime(harness)
        runtime.connect(
            RuntimeConfig(
                control_hz=100.0,
                watchdog_timeout_s=0.05,
                watchdog_startup_timeout_s=0.4,
                camera=CameraSettings(fps=30.0),
                preview_path=self.output_dir / "static" / "live.jpg",
            )
        )

        time.sleep(0.08)
        starting_snapshot = runtime.get_snapshot()
        self.assertEqual(starting_snapshot.state, RuntimeState.IDLE)
        self.assertEqual(starting_snapshot.control_sequence, 0)
        self.assertEqual(starting_snapshot.control_stage, "robot.get_observation")

        wait_until(lambda: runtime.get_snapshot().control_sequence >= 2)
        self.assertEqual(runtime.state, RuntimeState.IDLE)

    def test_steady_state_watchdog_reports_the_stalled_hardware_stage(self) -> None:
        harness = RuntimeHarness(
            observation_stall_after=5,
            observation_stall_s=0.25,
        )
        runtime = self.make_runtime(harness)
        runtime.connect(
            RuntimeConfig(
                control_hz=100.0,
                watchdog_timeout_s=0.05,
                watchdog_startup_timeout_s=0.4,
                camera=CameraSettings(fps=30.0),
                preview_path=self.output_dir / "static" / "live.jpg",
            )
        )

        wait_until(lambda: runtime.state is RuntimeState.ERROR)
        snapshot = runtime.get_snapshot()
        self.assertEqual(snapshot.error_source, "watchdog")
        self.assertIn("robot.get_observation", snapshot.last_error or "")
        self.assertIn("limit 0.050s", snapshot.last_error or "")

    def test_control_failure_enters_error_and_safe_disconnect_still_works(self) -> None:
        harness = RuntimeHarness(fail_after=3)
        runtime = self.make_runtime(harness)
        runtime.connect(self.config())
        wait_until(lambda: runtime.state is RuntimeState.ERROR)
        snapshot = runtime.get_snapshot()

        self.assertEqual(snapshot.error_source, "control")
        self.assertIn("synthetic CAN failure", snapshot.last_error or "")
        runtime.disconnect()
        self.assertEqual(runtime.state, RuntimeState.DISCONNECTED)
        self.assertEqual(harness.robot.disconnect_count, 1)

    def test_disconnect_finalizes_recording_after_control_error(self) -> None:
        harness = RuntimeHarness(fail_after=8)
        runtime = self.make_runtime(harness)
        runtime.connect(self.config())
        runtime.start_episode(
            RecordingConfig(
                output_dir=self.output_dir,
                task="failure during recording",
                action_sample_hz=20.0,
                duration_s=2.0,
            )
        )
        wait_until(lambda: runtime.state is RuntimeState.ERROR)

        runtime.disconnect()
        snapshot = runtime.get_snapshot()

        self.assertEqual(snapshot.state, RuntimeState.DISCONNECTED)
        self.assertIsNotNone(snapshot.draft)
        self.assertFalse(snapshot.draft.recording)
        self.assertGreater(snapshot.draft.video_frames, 0)

    def test_emergency_stop_uses_immediate_disable_not_rest_disconnect(self) -> None:
        harness = RuntimeHarness()
        runtime = self.make_runtime(harness)
        runtime.connect(self.config())
        wait_until(lambda: harness.robot.send_count() >= 2)

        runtime.emergency_stop()

        self.assertEqual(runtime.state, RuntimeState.DISCONNECTED)
        self.assertEqual(harness.emergency_calls, 1)
        self.assertEqual(harness.robot.disconnect_count, 0)
        self.assertEqual(harness.teleop.disconnect_count, 1)

    def test_process_proxy_runs_runtime_outside_the_ui_process(self) -> None:
        proxy = ProcessTeleopRuntime()
        try:
            snapshot = proxy._snapshot_rpc("get_snapshot", timeout_s=10.0)
            process = proxy._process
            self.assertEqual(snapshot.state, RuntimeState.DISCONNECTED)
            self.assertIsNotNone(process)
            self.assertNotEqual(process.pid, os.getpid())
            self.assertEqual(proxy.process_pid, process.pid)
            self.assertTrue(process.is_alive())
        finally:
            proxy.shutdown()


if __name__ == "__main__":
    unittest.main()
