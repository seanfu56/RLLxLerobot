from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ui.policy_runtime import (
    PolicyRuntime,
    PolicyState,
    RolloutConfig,
)
from ui.tests.test_policy_runtime import Harness, runtime_config, wait_until
from ui.video_goal import (
    GoalVideo,
    VideoSourceConfig,
    default_goal_index,
    encode_png,
)


def fake_video(frames: int = 4, goal_index: int | None = None) -> GoalVideo:
    images = [
        np.full((16, 16, 3), index * 40, dtype=np.uint8) for index in range(frames)
    ]
    return GoalVideo(
        frames_png=tuple(encode_png(image) for image in images),
        goal_index=default_goal_index(frames) if goal_index is None else goal_index,
        source="unit test",
        seconds=0.01,
        fps=1.0,
    )


class GoalIndexTests(unittest.TestCase):
    def test_the_four_frame_cascade_aims_at_its_third_frame(self) -> None:
        self.assertEqual(default_goal_index(4), 2)

    def test_a_long_clip_aims_two_thirds_of_the_way_through(self) -> None:
        # 93 Cosmos frames: (2 * 92 + 1) // 3 = 61, the same expression the
        # policy dataset uses to pick a goal out of a real episode.
        self.assertEqual(default_goal_index(93), 61)

    def test_it_matches_what_the_policy_was_trained_on(self) -> None:
        from policy.goal import DEFAULT_GOAL_FRAME_INDEX, uniform_frame_offsets

        for count in (4, 8, 17, 93, 200):
            self.assertEqual(
                default_goal_index(count),
                uniform_frame_offsets(count - 1, 4)[DEFAULT_GOAL_FRAME_INDEX],
            )

    def test_a_clip_shorter_than_the_rule_falls_back_to_its_last_frame(self) -> None:
        self.assertEqual(default_goal_index(3), 2)
        self.assertEqual(default_goal_index(1), 0)

    def test_an_empty_clip_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            default_goal_index(0)


class GoalVideoTests(unittest.TestCase):
    def test_frames_survive_the_png_round_trip(self) -> None:
        video = fake_video(4)
        self.assertEqual(video.frame_count, 4)
        np.testing.assert_array_equal(video.frame(1), np.full((16, 16, 3), 40, np.uint8))

    def test_the_goal_index_has_to_point_at_a_frame(self) -> None:
        with self.assertRaises(ValueError):
            fake_video(4, goal_index=4)

    def test_a_video_with_no_frames_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            GoalVideo(frames_png=(), goal_index=0, source="x", seconds=0.0, fps=1.0)

    def test_asking_for_a_frame_that_is_not_there_is_an_error(self) -> None:
        with self.assertRaises(IndexError):
            fake_video(4).frame(9)


class GoalVideoSaveTests(unittest.TestCase):
    """Keeping the video: without it nothing records what the policy aimed at."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "goal_video"

    def test_every_frame_is_written_with_a_readable_meta(self) -> None:
        video = fake_video(4)
        written = video.save(self.directory)
        self.assertEqual(written, self.directory)
        self.assertEqual(len(sorted(written.glob("frame_*.png"))), 4)
        meta = json.loads((written / "meta.json").read_text())
        self.assertEqual(meta["frame_count"], 4)
        self.assertEqual(meta["goal_index"], 2)
        self.assertEqual(meta["source"], "unit test")

    def test_the_frames_are_the_generated_bytes_not_a_re_encode(self) -> None:
        video = fake_video(4)
        written = video.save(self.directory)
        self.assertEqual((written / "frame_002.png").read_bytes(), video.frames_png[2])

    def test_the_aimed_frame_is_copied_out_as_the_goal(self) -> None:
        """The operator can override the default, so goal.png is not frame 2."""
        written = fake_video(4).save(self.directory, chosen_index=1)
        goal = cv2.imread(str(written / "goal.png"))
        np.testing.assert_array_equal(goal, fake_video(4).frame(1))
        meta = json.loads((written / "meta.json").read_text())
        self.assertEqual(meta["goal_index"], 1)
        self.assertEqual(meta["default_goal_index"], 2)

    def test_a_goal_outside_the_video_is_an_error(self) -> None:
        with self.assertRaises(IndexError):
            fake_video(4).save(self.directory, chosen_index=9)

    def test_extra_fields_land_in_the_meta(self) -> None:
        written = fake_video(4).save(self.directory, extra={"episode": "episode_007"})
        self.assertEqual(json.loads((written / "meta.json").read_text())["episode"], "episode_007")

    def test_a_playable_clip_is_written_beside_the_frames(self) -> None:
        written = fake_video(6).save(self.directory)
        name = json.loads((written / "meta.json").read_text())["clip"]
        self.assertIsNotNone(name)
        capture = cv2.VideoCapture(str(written / name))
        try:
            frames = 0
            while capture.read()[0]:
                frames += 1
        finally:
            capture.release()
        self.assertEqual(frames, 6)

    def test_a_clip_that_cannot_be_written_does_not_lose_the_frames(self) -> None:
        """The PNGs are the archive; the clip is only there to be watched."""
        video = GoalVideo(
            frames_png=(
                encode_png(np.zeros((8, 8, 3), np.uint8)),
                encode_png(np.zeros((16, 16, 3), np.uint8)),
            ),
            goal_index=0,
            source="mixed sizes",
            seconds=0.01,
            fps=1.0,
        )
        written = video.save(self.directory)
        self.assertEqual(len(sorted(written.glob("frame_*.png"))), 2)
        self.assertIsNone(json.loads((written / "meta.json").read_text())["clip"])


class VideoSourceConfigTests(unittest.TestCase):
    def test_the_source_has_to_be_one_of_the_two_models(self) -> None:
        with self.assertRaises(ValueError):
            VideoSourceConfig(kind="sora")

    def test_the_cascade_checks_its_checkpoints_exist_before_a_rollout_starts(self) -> None:
        with self.assertRaises(ValueError) as error:
            VideoSourceConfig(kind="diffusion", base_checkpoint=Path("/nope/base.pt"))
        self.assertIn("base checkpoint", str(error.exception))

    def test_the_cosmos_source_needs_a_url(self) -> None:
        with self.assertRaises(ValueError):
            VideoSourceConfig(kind="cosmos", url="")

    def test_each_source_says_what_it_is(self) -> None:
        self.assertIn("Cosmos3-Nano", VideoSourceConfig(kind="cosmos").describe())

    def test_the_cascade_accepts_checkpoints_that_are_there(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.pt"
            superres = Path(directory) / "sr.pt"
            base.write_bytes(b"")
            superres.write_bytes(b"")
            config = VideoSourceConfig(
                kind="diffusion", base_checkpoint=base, superres_checkpoint=superres
            )
            self.assertIn("src/diffusion", config.describe())


class GoalRuntimeTests(unittest.TestCase):
    """The runtime half: generating a goal video and aiming the policy at it."""

    def setUp(self) -> None:
        self.harness = Harness(goal_conditioned=True)
        self.runtime = self.harness.runtime()
        self.temporary = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temporary.name) / "rollouts"
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self._safe_disconnect)
        self.generated: list[Any] = []

        def fake_generate(config, frame):
            self.generated.append((config, frame))
            return fake_video(4)

        # Patch the entry point the runtime imports lazily, so no video model
        # and no GPU is needed to exercise the plumbing around it.
        import ui.video_goal as video_goal

        self._original = video_goal.generate_goal_video
        video_goal.generate_goal_video = fake_generate
        self.addCleanup(setattr, video_goal, "generate_goal_video", self._original)

    def _safe_disconnect(self) -> None:
        if self.runtime.state is not PolicyState.DISCONNECTED:
            try:
                self.runtime.disconnect()
            except BaseException:
                pass

    def _load_policy(self) -> None:
        from ui.policy_runtime import PolicySettings

        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            self.runtime.load_policy(PolicySettings(checkpoint=Path(handle.name), device="cpu"))

    def test_generation_conditions_on_the_live_camera_frame(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        wait_until(lambda: self.harness.camera.latest_frame() is not None)
        video = self.runtime.generate_goal_video(VideoSourceConfig(kind="cosmos"))
        self.assertEqual(video.frame_count, 4)
        self.assertEqual(video.goal_index, 2)
        # The frame handed to the video model is the camera's, not a stale one.
        self.assertIs(self.generated[0][1], self.harness.camera.latest_frame())

    def test_generation_needs_connected_hardware(self) -> None:
        self._load_policy()
        with self.assertRaises(RuntimeError):
            self.runtime.generate_goal_video(VideoSourceConfig(kind="cosmos"))

    def test_generation_is_refused_while_the_arm_is_driven(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=None))
        wait_until(lambda: self.harness.robot.send_count() >= 2)
        with self.assertRaises(RuntimeError):
            self.runtime.generate_goal_video(VideoSourceConfig(kind="cosmos"))
        self.runtime.stop_rollout()
        self.runtime.discard_rollout()

    def test_setting_the_goal_reaches_the_runner_and_shows_in_the_snapshot(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        video = fake_video(4)
        self.assertFalse(self.runtime.get_snapshot().policy.has_goal)
        self.runtime.set_goal_frame(video.frames_png[video.goal_index])
        self.assertTrue(self.runtime.get_snapshot().policy.has_goal)
        np.testing.assert_array_equal(
            self.harness.runner.goal, video.frame(video.goal_index)
        )

    def test_the_goal_can_be_taken_away_again(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.set_goal_frame(fake_video(4).frames_png[2])
        self.runtime.set_goal_frame(None)
        self.assertFalse(self.runtime.get_snapshot().policy.has_goal)
        self.assertIsNone(self.harness.runner.goal)

    def test_a_goal_cannot_be_swapped_mid_rollout(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        self.runtime.set_goal_frame(fake_video(4).frames_png[2])
        self.runtime.start_rollout(RolloutConfig(output_dir=self.output_dir, duration_s=None))
        wait_until(lambda: self.harness.robot.send_count() >= 2)
        with self.assertRaises(RuntimeError):
            self.runtime.set_goal_frame(fake_video(4).frames_png[0])
        self.runtime.stop_rollout()
        self.runtime.discard_rollout()

    def test_something_that_is_not_a_png_is_refused(self) -> None:
        self._load_policy()
        self.runtime.connect(runtime_config())
        with self.assertRaises(ValueError):
            self.runtime.set_goal_frame(b"not an image")

    def test_a_goal_needs_a_loaded_policy(self) -> None:
        self.runtime.connect(runtime_config())
        with self.assertRaises(RuntimeError):
            self.runtime.set_goal_frame(fake_video(4).frames_png[2])

    def test_a_plain_checkpoint_has_nowhere_to_put_a_goal(self) -> None:
        """Loading the wrong kind of run here fails loudly, not at the first step."""
        harness = Harness(goal_conditioned=False)
        runtime = harness.runtime()
        self.addCleanup(
            lambda: runtime.disconnect()
            if runtime.state is not PolicyState.DISCONNECTED
            else None
        )
        from ui.policy_runtime import PolicySettings

        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            runtime.load_policy(PolicySettings(checkpoint=Path(handle.name), device="cpu"))
        runtime.connect(runtime_config())
        with self.assertRaises(RuntimeError) as error:
            runtime.set_goal_frame(fake_video(4).frames_png[2])
        self.assertIn("--goal-conditioned", str(error.exception))

    def test_both_operations_are_reachable_across_the_process_boundary(self) -> None:
        from ui.policy_runtime import _PROCESS_OPERATIONS

        self.assertIn("generate_goal_video", _PROCESS_OPERATIONS)
        self.assertIn("set_goal_frame", _PROCESS_OPERATIONS)


if __name__ == "__main__":
    unittest.main()
