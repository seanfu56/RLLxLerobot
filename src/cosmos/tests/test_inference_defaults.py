"""Regression tests for keeping inference conditioned the way training was.

One caption covers both recording modes of a bundle, so nothing in the prompt
or the conditioning frame says which motion to produce and the sampler settings
decide it instead.  Two of those settings used to drift away from training: the
conditioning FPS and classifier-free guidance.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from cosmos.runtime import training_clip_fps
from cosmos.sample import parse_args
from cosmos.tests.test_training_data import make_episode


def write_run(directory: Path, **overrides) -> Path:
    """A minimal train_args.json of the shape train.py writes (all strings)."""
    recorded = {
        "bundle": "piper-data/dataset/pick-can-all",
        "clip_mode": "episode",
        "num_frames": "93",
    }
    recorded.update({key: str(value) for key, value in overrides.items()})
    (directory / "train_args.json").write_text(json.dumps(recorded), encoding="utf-8")
    adapter = directory / "adapter"
    adapter.mkdir(exist_ok=True)
    return adapter


class TrainingClipFpsTests(unittest.TestCase):
    def test_episode_mode_recovers_the_retimed_rate_not_the_camera_rate(self) -> None:
        # 121 source frames at 20 FPS retimed onto 93 frames is 15.37 FPS, and
        # conditioning at the camera's 20 would misstate the clip's duration.
        episodes = [make_episode("m1", index, frames=121) for index in range(3)]
        with TemporaryDirectory() as temporary:
            adapter = write_run(Path(temporary))
            with mock.patch("cosmos.runtime.load_episodes", return_value=episodes):
                self.assertAlmostEqual(training_clip_fps(adapter), 93 * 20.0 / 121)

    def test_median_is_taken_over_unequal_episode_lengths(self) -> None:
        episodes = [
            make_episode("m1", 0, frames=100),
            make_episode("m1", 1, frames=121),
            make_episode("m2", 2, frames=200),
        ]
        with TemporaryDirectory() as temporary:
            adapter = write_run(Path(temporary))
            with mock.patch("cosmos.runtime.load_episodes", return_value=episodes):
                self.assertAlmostEqual(training_clip_fps(adapter), 93 * 20.0 / 121)

    def test_window_mode_runs_keep_the_native_rate(self) -> None:
        with TemporaryDirectory() as temporary:
            adapter = write_run(Path(temporary), clip_mode="window")
            self.assertIsNone(training_clip_fps(adapter))

    def test_an_uninspectable_run_falls_back_instead_of_raising(self) -> None:
        with TemporaryDirectory() as temporary:
            self.assertIsNone(training_clip_fps(Path(temporary) / "adapter"))
            adapter = write_run(Path(temporary))
            with mock.patch(
                "cosmos.runtime.load_episodes", side_effect=FileNotFoundError("bundle moved")
            ):
                self.assertIsNone(training_clip_fps(adapter))


class SamplingDefaultsTests(unittest.TestCase):
    def test_sampling_defaults_to_episode_mode_like_training(self) -> None:
        # Under 'window' the held-out condition is the centre frame of the
        # episode, which is mid-motion rather than the true first frame.
        self.assertEqual(parse_args([]).clip_mode, "episode")

    def test_sampling_defaults_to_the_derived_rate_rather_than_a_constant(self) -> None:
        # None means "ask the adapter's run"; a number here would reintroduce
        # the drift between the camera rate and the retimed training rate.
        self.assertIsNone(parse_args([]).fps)


if __name__ == "__main__":
    unittest.main()
