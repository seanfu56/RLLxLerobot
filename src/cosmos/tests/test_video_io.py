"""The writer's only real contract: what comes back is what went in.

These are byte-equality tests rather than PSNR-threshold tests on purpose. A
codec or container swap that quietly reintroduces lossy encoding would still
score 40+ dB and pass anything softer, and 40 dB is precisely the regime that
made the preference signal unreliable in the first place.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cosmos.video_io import frames_to_array, rebuild_from_frames, save_video


def read_back(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return np.stack(frames) if frames else np.zeros((0, 0, 0, 3), np.uint8)


class SaveVideoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: [p.unlink() for p in self.tmp.rglob("*") if p.is_file()])

    def test_noise_survives_the_round_trip_exactly(self) -> None:
        """Incompressible input is the worst case; a lossy codec cannot fake it."""
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 256, (12, 64, 64, 3), dtype=np.uint8)
        path = save_video(frames, self.tmp / "noise.mp4", 20.0)
        np.testing.assert_array_equal(read_back(path), frames)

    def test_gradients_survive_the_round_trip_exactly(self) -> None:
        """Smooth content is where chroma subsampling does its damage."""
        ramp = np.linspace(0, 255, 64, dtype=np.float32)
        frames = np.stack([
            np.clip(ramp[None, :, None] + i, 0, 255).astype(np.uint8)
            * np.ones((64, 1, 3), np.uint8)
            for i in range(8)
        ])
        path = save_video(frames, self.tmp / "ramp.mp4", 20.0)
        np.testing.assert_array_equal(read_back(path), frames)

    def test_float_frames_are_scaled_and_uint8_frames_are_not(self) -> None:
        """``export_to_video`` multiplied every ndarray by 255; this must not."""
        frames = np.full((4, 8, 8, 3), 200, np.uint8)
        np.testing.assert_array_equal(frames_to_array(frames), frames)
        np.testing.assert_array_equal(frames_to_array(frames / 255.0), frames)

    def test_odd_dimensions_are_refused_rather_than_resized(self) -> None:
        frames = np.zeros((4, 65, 64, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, "even dimensions"):
            save_video(frames, self.tmp / "odd.mp4", 20.0)

    def test_lossy_preview_is_actually_lossy(self) -> None:
        """Guards the test itself: exactness above must mean something."""
        rng = np.random.default_rng(1)
        frames = rng.integers(0, 256, (12, 64, 64, 3), dtype=np.uint8)
        path = save_video(frames, self.tmp / "preview.mp4", 20.0, lossless=False)
        self.assertFalse(np.array_equal(read_back(path), frames))


class RebuildFromFramesTest(unittest.TestCase):
    def test_rebuilds_the_clip_from_the_png_frames(self) -> None:
        rng = np.random.default_rng(2)
        frames = rng.integers(0, 256, (6, 64, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "20260101_000000"
            archive.mkdir()
            for index, frame in enumerate(frames):
                cv2.imwrite(str(archive / f"frame_{index:03d}.png"),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            (archive / "meta.json").write_text('{"clip": "video.mp4", "fps": 20.0}')
            # A lossy clip standing in for what the archives held.
            save_video(frames, archive / "video.mp4", 20.0, lossless=False)

            record = rebuild_from_frames(archive)
            self.assertEqual(record["frames"], 6)
            np.testing.assert_array_equal(read_back(archive / "video.mp4"), frames)

    def test_returns_none_when_there_are_no_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(rebuild_from_frames(Path(tmp)))


if __name__ == "__main__":
    unittest.main()
