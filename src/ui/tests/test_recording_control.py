from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ui.recording_control import RecordingCommandController
from ui.teleop_runtime import RecordingConfig, RuntimeState


class _FakeRuntime:
    def __init__(
        self,
        state: RuntimeState = RuntimeState.IDLE,
        *,
        draft: object | None = None,
    ) -> None:
        self.state = state
        self.draft = draft
        self.started: list[RecordingConfig] = []
        self.stop_calls = 0
        self.discard_calls = 0

    def get_snapshot(self):
        return SimpleNamespace(state=self.state, draft=self.draft)

    def start_episode(self, config: RecordingConfig):
        self.started.append(config)
        self.state = RuntimeState.RECORDING

    def stop_episode(self):
        self.stop_calls += 1
        self.state = RuntimeState.IDLE
        self.draft = object()

    def discard_episode(self):
        self.discard_calls += 1
        self.draft = None


class RecordingCommandControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.config = RecordingConfig(
            output_dir=Path(self.temporary_directory.name),
            task="test",
            action_sample_hz=20.0,
            duration_s=15.0,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_start_waits_for_deadline_and_runs_once(self) -> None:
        runtime = _FakeRuntime()
        controller = RecordingCommandController()

        outcome = controller.request_start(
            runtime,
            self.config,
            5.0,
            source="voice",
            now_s=10.0,
        )

        self.assertTrue(outcome.accepted)
        self.assertEqual(controller.remaining_seconds(12.0), 3.0)
        self.assertIsNone(controller.advance(runtime, now_s=14.999))
        started = controller.advance(runtime, now_s=15.0)
        self.assertEqual(started.action, "recording_started")
        self.assertEqual(runtime.started, [self.config])
        self.assertIsNone(controller.advance(runtime, now_s=20.0))

    def test_stop_cancels_a_pending_countdown(self) -> None:
        runtime = _FakeRuntime()
        controller = RecordingCommandController()
        controller.request_start(
            runtime,
            self.config,
            5.0,
            source="voice",
            now_s=10.0,
        )

        outcome = controller.request_stop(runtime, source="voice", now_s=11.0)

        self.assertEqual(outcome.action, "countdown_cancelled")
        self.assertIsNone(controller.pending)
        self.assertEqual(runtime.stop_calls, 0)

    def test_stop_immediately_forwards_to_recording_runtime(self) -> None:
        runtime = _FakeRuntime(RuntimeState.RECORDING)
        controller = RecordingCommandController()

        outcome = controller.request_stop(runtime, source="voice", now_s=10.0)

        self.assertTrue(outcome.accepted)
        self.assertEqual(outcome.action, "recording_stopped")
        self.assertEqual(runtime.stop_calls, 1)

    def test_start_is_rejected_when_a_draft_exists(self) -> None:
        runtime = _FakeRuntime(draft=object())
        controller = RecordingCommandController()

        outcome = controller.request_start(
            runtime,
            self.config,
            5.0,
            source="voice",
            now_s=10.0,
        )

        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.action, "start_ignored")
        self.assertIsNone(controller.pending)

    def test_rerecord_discards_draft_only_after_countdown(self) -> None:
        runtime = _FakeRuntime(draft=object())
        controller = RecordingCommandController()
        controller.request_start(
            runtime,
            self.config,
            5.0,
            source="button",
            discard_draft=True,
            now_s=10.0,
        )

        controller.advance(runtime, now_s=15.0)

        self.assertEqual(runtime.discard_calls, 1)
        self.assertEqual(runtime.started, [self.config])


if __name__ == "__main__":
    unittest.main()
