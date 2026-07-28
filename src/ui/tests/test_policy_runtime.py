from __future__ import annotations

import csv
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from ui.policy_runtime import (
    PolicyHardwareBundle,
    PolicyInfo,
    PolicyRuntime,
    PolicyRuntimeConfig,
    PolicySettings,
    PolicyState,
    RolloutConfig,
    _PolicyDriver,
    _PROCESS_OPERATIONS,
    _policy_meta,
)
from ui.teleop_runtime import (
    ACTION_KEYS,
    CameraSettings,
    VideoResult,
    _FrameReference,
)

START_POSE = {key: 1.0 for key in ACTION_KEYS}


def wait_until(predicate, timeout_s: float = 3.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Condition was not reached before timeout.")


class FakeRobot:
    def __init__(self, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.connected = False
        self.disconnect_count = 0
        self.observation_count = 0
        self.sent_actions: list[dict[str, float]] = []
        self._lock = threading.Lock()

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.disconnect_count += 1

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            self.observation_count += 1
            index = self.observation_count
        observation: dict[str, Any] = {key: float(index) for key in ACTION_KEYS}
        observation["overhead"] = object()
        return observation

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        with self._lock:
            if self.fail_after is not None and len(self.sent_actions) >= self.fail_after:
                raise RuntimeError("synthetic CAN failure")
            self.sent_actions.append(dict(action))
        return dict(action)

    def send_count(self) -> int:
        with self._lock:
            return len(self.sent_actions)


class FakeRunner:
    """Stands in for policy.inference.PolicyRunner without torch."""

    def __init__(self, chunk_size: int = 4, action_steps: int = 2) -> None:
        self.chunk_size = chunk_size
        self.action_steps = action_steps
        self.queue: list[list[float]] = []
        self.chunk_calls = 0
        self.reset_calls = 0
        self.states: list[list[float]] = []
        self.guidance_weight = 1.0

    def set_guidance_weight(self, weight: float) -> None:
        # Mirrors PolicyRunner: the queued chunk came from the old weight.
        self.guidance_weight = float(weight)
        self.queue.clear()

    @property
    def queued_actions(self) -> int:
        return len(self.queue)

    def reset(self) -> None:
        self.reset_calls += 1
        self.queue.clear()

    def select_action(self, frame: Any, state: Any) -> list[float]:
        del frame
        if not self.queue:
            self.chunk_calls += 1
            self.states.append(list(state))
            self.queue = [
                [float(self.chunk_calls) + offset] * len(ACTION_KEYS)
                for offset in range(self.action_steps)
            ]
        return self.queue.pop(0)

    def action_dict(self, action: list[float]) -> dict[str, float]:
        return {key: float(value) for key, value in zip(ACTION_KEYS, action, strict=True)}


class FakeCamera:
    def __init__(self, config: PolicyRuntimeConfig, on_error) -> None:
        self.config = config
        self.on_error = on_error
        self.started = False
        self.recording = False
        self.recording_started_s = 0.0
        self.output_fps = 0.0
        self.preview_enabled = config.camera.preview_enabled
        self.published = 0
        self.recorded_frames: list[dict[str, float | int]] = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.recording = False
        self.started = False

    def set_preview_enabled(self, enabled: bool) -> None:
        self.preview_enabled = enabled

    def publish_observation(self, observation: dict[str, Any], timestamp_s: float):
        del observation
        self.published += 1
        return _FrameReference(self.published, timestamp_s, object())

    def start_recording(self, path: Path, episode_start_s: float, output_fps: float) -> None:
        self.recording = True
        self.recording_started_s = episode_start_s
        self.output_fps = output_fps
        self.recorded_frames = []
        path.write_bytes(b"fake avi")

    def record_frame(self, frame_index: int, reference: _FrameReference) -> bool:
        if not self.recording:
            return False
        self.recorded_frames.append(
            {
                "frame_index": frame_index,
                "capture_sequence": reference.capture_sequence,
                "timestamp_s": round(reference.timestamp_s - self.recording_started_s, 6),
            }
        )
        return True

    def stop_recording(self) -> VideoResult:
        if not self.recording:
            return VideoResult(0, 0, (), 0.0)
        self.recording = False
        frames = tuple(self.recorded_frames)
        return VideoResult(len(frames), 0, frames, 0.1)

    def snapshot(self) -> dict[str, float | int | None]:
        return {
            "frame_age_s": 0.001 if self.started else None,
            "capture_hz": self.config.camera.fps if self.started else 0.0,
            "frame_count": self.published,
            "encode_ms": 0.2,
            "queue_depth": 0,
            "dropped_frames": 0,
            "written_frames": len(self.recorded_frames),
        }


def fake_policy_info(**overrides: Any) -> PolicyInfo:
    defaults = dict(
        checkpoint="/tmp/fake/best.pt",
        run_name="fake_run",
        step=1000,
        policy="simple",
        objective="diffusion",
        chunk_size=4,
        action_steps=2,
        num_inference_steps=10,
        image_size=224,
        vision="resnet18",
        pool="spatial_softmax",
        action_repr="delta",
        delta_mode="incremental",
        absolute_dims=(),
        joint_names=tuple(ACTION_KEYS),
        fps=20.0,
        device="cpu",
        use_ema=True,
        warmup_s=0.01,
    )
    defaults.update(overrides)
    return PolicyInfo(**defaults)


class Harness:
    def __init__(self, fail_after: int | None = None, move_delay_s: float = 0.0,
                 supports_guidance: bool = False) -> None:
        self.robot = FakeRobot(fail_after)
        self.runner = FakeRunner()
        self.supports_guidance = supports_guidance
        self.camera: FakeCamera | None = None
        self.emergency_calls = 0
        self.policy_calls = 0
        self.converted: list[tuple[Path, Path]] = []
        self.move_delay_s = move_delay_s
        self.moves: list[dict[str, float]] = []
        self.move_failure: BaseException | None = None
        self.sends_during_move: list[int] = []

    def hardware_factory(self, _config: PolicyRuntimeConfig) -> PolicyHardwareBundle:
        def emergency_disable() -> None:
            self.emergency_calls += 1
            self.robot.connected = False

        def move_to_pose(pose) -> None:
            self.sends_during_move.append(self.robot.send_count())
            if self.move_delay_s:
                time.sleep(self.move_delay_s)
            if self.move_failure is not None:
                raise self.move_failure
            self.moves.append(dict(pose))

        return PolicyHardwareBundle(self.robot, emergency_disable, move_to_pose)

    def camera_factory(self, config: PolicyRuntimeConfig, on_error) -> FakeCamera:
        self.camera = FakeCamera(config, on_error)
        return self.camera

    def policy_factory(self, settings: PolicySettings) -> _PolicyDriver:
        self.policy_calls += 1
        return _PolicyDriver(
            self.runner,
            fake_policy_info(
                checkpoint=str(settings.checkpoint),
                supports_guidance=self.supports_guidance,
                guidance_weight=settings.guidance_weight or 1.0,
            ),
        )

    def video_converter(self, source: Path, destination: Path) -> None:
        self.converted.append((source, destination))
        destination.write_bytes(b"fake mp4")

    def runtime(self) -> PolicyRuntime:
        return PolicyRuntime(
            hardware_factory=self.hardware_factory,
            camera_factory=self.camera_factory,
            policy_factory=self.policy_factory,
            video_converter=self.video_converter,
        )


def runtime_config(**overrides: Any) -> PolicyRuntimeConfig:
    defaults: dict[str, Any] = dict(
        can_port="fake_can",
        control_hz=100.0,
        start_pose=dict(START_POSE),
        watchdog_timeout_s=5.0,
        watchdog_startup_timeout_s=5.0,
        camera=CameraSettings(device="/dev/null", preview_enabled=False),
        preview_path=Path(tempfile.gettempdir()) / "policy_test_live.jpg",
    )
    defaults.update(overrides)
    return PolicyRuntimeConfig(**defaults)


class PolicySettingsTest(unittest.TestCase):
    def test_rejects_missing_checkpoint(self) -> None:
        with self.assertRaises(ValueError):
            PolicySettings(checkpoint=Path("/nonexistent/best.pt"))

    def test_start_pose_must_cover_every_joint(self) -> None:
        with self.assertRaises(ValueError):
            runtime_config(start_pose={"joint_1.pos": 0.0})


class GuidanceWeightTest(unittest.TestCase):
    """Retuning classifier-free guidance on an already loaded policy."""

    def setUp(self) -> None:
        self.harness = Harness(supports_guidance=True)
        self.runtime = self.harness.runtime()
        self.temporary = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temporary.name) / "rollouts"
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self._safe_disconnect)

    def _safe_disconnect(self) -> None:
        if self.runtime.state is not PolicyState.DISCONNECTED:
            try:
                self.runtime.disconnect()
            except BaseException:
                pass

    def _load_policy(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            self.runtime.load_policy(PolicySettings(checkpoint=Path(handle.name), device="cpu"))

    def test_retuning_updates_the_policy_without_reloading_it(self) -> None:
        self._load_policy()
        self.assertEqual(self.harness.policy_calls, 1)

        snapshot = self.runtime.set_guidance_weight(2.5)
        self.assertAlmostEqual(snapshot.policy.guidance_weight, 2.5)
        self.assertAlmostEqual(self.harness.runner.guidance_weight, 2.5)
        # The weights were never reloaded, which is the whole point.
        self.assertEqual(self.harness.policy_calls, 1)
        self.assertIn("guidance 2.5", snapshot.policy.headline())

    def test_retuning_drops_the_queued_chunk(self) -> None:
        # Actions planned under the previous weight must not reach the arm.
        self._load_policy()
        self.harness.runner.queue = [[1.0] * len(ACTION_KEYS)]
        self.runtime.set_guidance_weight(2.0)
        self.assertEqual(self.harness.runner.queued_actions, 0)

    def test_retuning_needs_a_loaded_policy(self) -> None:
        with self.assertRaises(RuntimeError):
            self.runtime.set_guidance_weight(2.0)

    def test_retuning_is_refused_while_the_arm_is_driven(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=None))
        wait_until(lambda: self.harness.robot.send_count() >= 2)
        with self.assertRaises(RuntimeError):
            self.runtime.set_guidance_weight(2.0)
        self.runtime.stop_rollout()
        # Allowed again once the rollout has stopped.
        self.runtime.set_guidance_weight(2.0)
        self.assertAlmostEqual(self.harness.runner.guidance_weight, 2.0)

    def test_a_checkpoint_without_an_unconditional_branch_is_refused(self) -> None:
        harness = Harness(supports_guidance=False)
        runtime = harness.runtime()
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            runtime.load_policy(PolicySettings(checkpoint=Path(handle.name), device="cpu"))
        with self.assertRaises(RuntimeError) as error:
            runtime.set_guidance_weight(2.0)
        self.assertIn("conditioning dropout", str(error.exception))

    def test_the_recorded_rollout_states_which_weight_produced_it(self) -> None:
        self._load_policy()
        self.runtime.set_guidance_weight(1.5)
        meta = _policy_meta(self.runtime.get_snapshot().policy)
        self.assertAlmostEqual(meta["guidance_weight"], 1.5)

    def test_the_operation_is_reachable_across_the_process_boundary(self) -> None:
        self.assertIn("set_guidance_weight", _PROCESS_OPERATIONS)


class PolicyRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = Harness()
        self.runtime = self.harness.runtime()
        self.temporary = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temporary.name) / "rollouts"
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self._safe_disconnect)

    def _safe_disconnect(self) -> None:
        if self.runtime.state is not PolicyState.DISCONNECTED:
            try:
                self.runtime.disconnect()
            except BaseException:
                pass

    def _load_policy(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            self.runtime.load_policy(PolicySettings(checkpoint=Path(handle.name), device="cpu"))

    def test_idle_reads_observations_without_commanding_the_arm(self) -> None:
        self.runtime.connect(runtime_config())
        wait_until(lambda: self.harness.robot.observation_count > 5)
        self.assertEqual(self.harness.robot.send_count(), 0)
        self.assertIs(self.runtime.state, PolicyState.IDLE)
        snapshot = self.runtime.get_snapshot()
        self.assertFalse(snapshot.driving)
        self.assertGreater(snapshot.camera_frames, 0)

    def test_rollout_requires_a_loaded_policy(self) -> None:
        self.runtime.connect(runtime_config())
        with self.assertRaises(RuntimeError):
            self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir))
        self.assertEqual(self.harness.robot.send_count(), 0)

    def test_rollout_drives_the_arm_and_replans_per_action_steps(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(
            RolloutConfig(output_dir=self.output_dir, duration_s=None, task="pick up can")
        )
        wait_until(lambda: self.harness.robot.send_count() >= 8)
        self.runtime.stop_rollout()

        sends = self.harness.robot.send_count()
        self.assertIs(self.runtime.state, PolicyState.IDLE)
        # action_steps=2 in FakeRunner: one prediction feeds two commands.
        self.assertAlmostEqual(self.harness.runner.chunk_calls, sends / 2, delta=1)
        # The measured state must reach the policy, not a commanded value.
        self.assertTrue(all(state[0] > 0 for state in self.harness.runner.states))

    def test_stop_rollout_stops_commanding_immediately(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=None))
        wait_until(lambda: self.harness.robot.send_count() >= 4)
        self.runtime.stop_rollout()
        after_stop = self.harness.robot.send_count()
        time.sleep(0.2)
        self.assertEqual(self.harness.robot.send_count(), after_stop)
        self.assertFalse(self.runtime.get_snapshot().driving)

    def test_stopped_rollout_parks_the_arm_at_the_start_pose(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=None))
        wait_until(lambda: self.harness.robot.send_count() >= 4)
        self.runtime.stop_rollout()

        self.assertEqual(self.harness.moves, [dict(START_POSE)])
        self.assertIs(self.runtime.state, PolicyState.IDLE)
        self.assertIsNone(self.runtime.get_snapshot().last_error)
        # The policy must have stopped commanding before the move began.
        self.assertEqual(self.harness.sends_during_move, [self.harness.robot.send_count()])

    def test_timed_rollout_parks_the_arm_too(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=0.2))
        wait_until(lambda: self.runtime.state is PolicyState.IDLE, timeout_s=3.0)
        self.assertEqual(self.harness.moves, [dict(START_POSE)])

    def test_parking_can_be_switched_off(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(
            RolloutConfig(output_dir=self.output_dir, duration_s=None, return_to_start=False)
        )
        wait_until(lambda: self.harness.robot.send_count() >= 4)
        self.runtime.stop_rollout()
        self.assertEqual(self.harness.moves, [])

    def test_park_button_works_while_idle_and_is_refused_while_driving(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.return_to_start()
        self.assertEqual(self.harness.moves, [dict(START_POSE)])

        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=None))
        wait_until(lambda: self.harness.robot.send_count() >= 2)
        with self.assertRaises(RuntimeError):
            self.runtime.return_to_start()
        self.runtime.stop_rollout()
        self.assertEqual(len(self.harness.moves), 2)

    def test_failed_parking_is_reported_without_losing_control(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.harness.move_failure = RuntimeError("start pose was not reached before timeout")
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=None))
        wait_until(lambda: self.harness.robot.send_count() >= 2)
        self.runtime.stop_rollout()

        snapshot = self.runtime.get_snapshot()
        self.assertIs(snapshot.state, PolicyState.IDLE)
        self.assertEqual(snapshot.error_source, "return_to_start")
        self.assertIn("was not reached", snapshot.last_error or "")
        # Control is still running, so the arm is still held and a retry works.
        self.harness.move_failure = None
        self.runtime.return_to_start()
        self.assertIsNone(self.runtime.get_snapshot().last_error)

    def test_watchdog_tolerates_a_slow_parking_move(self) -> None:
        self.harness = Harness(move_delay_s=1.0)
        self.runtime = self.harness.runtime()
        self._load_policy()
        # The move takes far longer than the stale-control timeout.
        self.runtime.connect(runtime_config(watchdog_timeout_s=0.3))
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=None))
        wait_until(lambda: self.harness.robot.send_count() >= 2)
        self.runtime.stop_rollout()
        self.assertIs(self.runtime.state, PolicyState.IDLE)
        self.assertEqual(self.harness.moves, [dict(START_POSE)])
        time.sleep(0.3)
        self.assertIs(self.runtime.state, PolicyState.IDLE)

    def test_disconnect_does_not_park_before_its_own_rest_move(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=None))
        wait_until(lambda: self.harness.robot.send_count() >= 2)
        self.runtime.disconnect()
        self.assertEqual(self.harness.moves, [])
        self.assertEqual(self.harness.robot.disconnect_count, 1)

    def test_timed_rollout_stops_itself(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=0.2))
        wait_until(lambda: self.runtime.state is PolicyState.IDLE, timeout_s=3.0)
        snapshot = self.runtime.get_snapshot()
        self.assertIsNotNone(snapshot.draft)
        self.assertFalse(snapshot.draft.running)
        self.assertFalse(snapshot.driving)

    def test_saved_rollout_matches_the_recorded_episode_layout(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(
            RolloutConfig(output_dir=self.output_dir, duration_s=None, task="pick up can")
        )
        wait_until(lambda: self.runtime.get_snapshot().rollout_samples >= 5)
        self.runtime.stop_rollout()
        saved = self.runtime.save_rollout()

        self.assertEqual(saved.name, "episode_000")
        for name in ("video.mp4", "actions.csv", "observations.csv", "frame_timestamps.csv", "meta.json"):
            self.assertTrue((saved / name).is_file(), name)

        with (saved / "actions.csv").open(newline="") as handle:
            actions = list(csv.DictReader(handle))
        with (saved / "observations.csv").open(newline="") as handle:
            observations = list(csv.DictReader(handle))
        with (saved / "frame_timestamps.csv").open(newline="") as handle:
            frames = list(csv.DictReader(handle))
        meta = json.loads((saved / "meta.json").read_text(encoding="utf-8"))

        self.assertEqual(len(actions), len(observations))
        self.assertEqual(len(actions), len(frames))
        self.assertEqual(meta["video_frames"], len(frames))
        self.assertEqual(meta["task"], "pick up can")
        self.assertEqual(meta["policy"]["run_name"], "fake_run")
        for index, (action, observation, frame) in enumerate(
            zip(actions, observations, frames, strict=True)
        ):
            self.assertEqual(int(action["frame_index"]), index)
            self.assertEqual(int(observation["frame_index"]), index)
            self.assertEqual(int(frame["frame_index"]), index)
            self.assertEqual(
                action["source_control_sequence"], observation["source_control_sequence"]
            )
            for key in ACTION_KEYS:
                self.assertIn(key, action)
                self.assertIn(key, observation)
        self.assertFalse((saved / "video.avi").exists())

    def test_discard_removes_the_pending_directory(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=None))
        wait_until(lambda: self.runtime.get_snapshot().rollout_samples >= 2)
        self.runtime.stop_rollout()
        self.runtime.discard_rollout()
        self.assertEqual(list(self.output_dir.iterdir()), [])
        self.assertIsNone(self.runtime.get_snapshot().draft)

    def test_unrecorded_rollout_writes_nothing(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(
            RolloutConfig(output_dir=self.output_dir, duration_s=None, record=False)
        )
        wait_until(lambda: self.harness.robot.send_count() >= 4)
        self.runtime.stop_rollout()
        self.assertFalse(self.output_dir.exists())
        with self.assertRaises(RuntimeError):
            self.runtime.save_rollout()
        self.runtime.discard_rollout()

    def test_checkpoint_cannot_change_while_driving(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=None))
        wait_until(lambda: self.harness.robot.send_count() >= 2)
        with self.assertRaises(RuntimeError):
            self._load_policy()
        self.runtime.stop_rollout()
        self.runtime.discard_rollout()
        self._load_policy()
        self.assertEqual(self.harness.policy_calls, 2)

    def test_new_rollout_replans_from_the_live_frame(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=None))
        wait_until(lambda: self.harness.robot.send_count() >= 3)
        self.runtime.stop_rollout()
        self.runtime.discard_rollout()
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=None))
        wait_until(lambda: self.harness.robot.send_count() >= 5)
        self.runtime.stop_rollout()
        self.assertGreaterEqual(self.harness.runner.reset_calls, 2)

    def test_robot_failure_stops_commands_and_reports_error(self) -> None:
        self.harness = Harness(fail_after=3)
        self.runtime = self.harness.runtime()
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=None))
        wait_until(lambda: self.runtime.state is PolicyState.ERROR)
        snapshot = self.runtime.get_snapshot()
        self.assertEqual(snapshot.error_source, "control")
        self.assertIn("synthetic CAN failure", snapshot.last_error or "")
        sends = self.harness.robot.send_count()
        time.sleep(0.2)
        self.assertEqual(self.harness.robot.send_count(), sends)

    def test_emergency_stop_disables_without_rest_move(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=None))
        wait_until(lambda: self.harness.robot.send_count() >= 2)
        self.runtime.emergency_stop()
        self.assertIs(self.runtime.state, PolicyState.DISCONNECTED)
        self.assertEqual(self.harness.emergency_calls, 1)
        self.assertEqual(self.harness.robot.disconnect_count, 0)

    def test_disconnect_stops_control_then_rests(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        wait_until(lambda: self.harness.robot.observation_count > 3)
        self.runtime.disconnect()
        self.assertIs(self.runtime.state, PolicyState.DISCONNECTED)
        self.assertEqual(self.harness.robot.disconnect_count, 1)
        self.assertFalse(self.harness.camera.started)

    def test_watchdog_reports_a_stalled_control_loop(self) -> None:
        config = runtime_config(watchdog_startup_timeout_s=0.5)
        original = self.harness.robot.get_observation

        def stalling_observation() -> dict[str, Any]:
            time.sleep(2.0)
            return original()

        self.harness.robot.get_observation = stalling_observation  # type: ignore[method-assign]
        self.runtime.connect(config)
        wait_until(lambda: self.runtime.state is PolicyState.ERROR, timeout_s=4.0)
        self.assertEqual(self.runtime.get_snapshot().error_source, "watchdog")


if __name__ == "__main__":
    unittest.main()
