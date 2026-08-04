from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ui.snapshot_camera import (
    SnapshotCamera,
    SnapshotSettings,
    existing_snapshots,
)


def wait_until(predicate, timeout_s: float = 2.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Condition was not reached before timeout.")


class FakeCapture:
    """A ``cv2.VideoCapture`` that hands out frames of a known colour."""

    def __init__(self, opened: bool = True, fail_reads: bool = False) -> None:
        self.opened = opened
        self.fail_reads = fail_reads
        self.released = False
        self.properties: dict[int, float] = {}
        self.reads = 0
        self._lock = threading.Lock()

    def isOpened(self) -> bool:  # noqa: N802 - the OpenCV name
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.properties[prop] = value
        return True

    def read(self) -> tuple[bool, Any]:
        with self._lock:
            self.reads += 1
            index = self.reads
        if self.fail_reads:
            return False, None
        time.sleep(0.002)
        return True, np.full((8, 12, 3), index % 256, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class FakeCv2:
    """Only the handful of names ``SnapshotCamera`` touches; encoding is real
    OpenCV, because the point of several of these tests is the file on disk."""

    CAP_V4L2 = 200
    CAP_ANY = 0
    CAP_PROP_FOURCC = 6
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_BUFFERSIZE = 38
    IMWRITE_JPEG_QUALITY = cv2.IMWRITE_JPEG_QUALITY

    def __init__(self, captures: list[FakeCapture] | None = None) -> None:
        self.captures = captures or [FakeCapture()]
        self.opened_with: list[tuple[Any, int | None]] = []

    def VideoCapture(self, source: Any, backend: int | None = None) -> FakeCapture:  # noqa: N802
        self.opened_with.append((source, backend))
        return self.captures[min(len(self.opened_with) - 1, len(self.captures) - 1)]

    def VideoWriter_fourcc(self, *chars: str) -> int:  # noqa: N802
        return cv2.VideoWriter_fourcc(*chars)

    def imencode(self, extension: str, frame: Any, params: Any = None) -> tuple[bool, Any]:
        return cv2.imencode(extension, frame, params or [])


class SettingsTests(unittest.TestCase):
    def test_a_numeric_device_becomes_an_opencv_index(self) -> None:
        self.assertEqual(SnapshotSettings(device="0").source, 0)
        self.assertEqual(SnapshotSettings(device="/dev/video2").source, "/dev/video2")

    def test_invalid_settings_are_rejected_before_the_device_is_touched(self) -> None:
        for kwargs in (
            {"device": "  "},
            {"width": 0},
            {"fps": 0.0},
            {"fourcc": "MJP"},
            {"jpeg_quality": 0},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                SnapshotSettings(**kwargs)


class CameraTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def camera(self, fake: FakeCv2 | None = None, **kwargs: Any) -> SnapshotCamera:
        camera = SnapshotCamera(
            SnapshotSettings(device="/dev/video9", preview_hz=50.0, **kwargs),
            preview_path=self.root / "static" / "snapshot_live.jpg",
            cv2_module=fake or FakeCv2(),
        )
        self.addCleanup(camera.stop)
        return camera

    def test_start_configures_the_device_and_publishes_a_preview(self) -> None:
        fake = FakeCv2()
        camera = self.camera(fake, width=320, height=240, fourcc="MJPG")
        camera.start()

        capture = fake.captures[0]
        self.assertEqual(capture.properties[FakeCv2.CAP_PROP_FRAME_WIDTH], 320)
        self.assertEqual(capture.properties[FakeCv2.CAP_PROP_FRAME_HEIGHT], 240)
        self.assertIn(FakeCv2.CAP_PROP_FOURCC, capture.properties)
        self.assertEqual(fake.opened_with[0][1], FakeCv2.CAP_V4L2)

        preview = self.root / "static" / "snapshot_live.jpg"
        wait_until(lambda: preview.is_file())
        self.assertIsNotNone(cv2.imread(str(preview)))

    def test_a_device_that_will_not_open_fails_on_start(self) -> None:
        fake = FakeCv2([FakeCapture(opened=False), FakeCapture(opened=False)])
        camera = self.camera(fake)
        with self.assertRaises(RuntimeError) as error:
            camera.start()
        self.assertIn("/dev/video9", str(error.exception))

    def test_a_device_that_opens_but_never_yields_a_frame_fails_on_start(self) -> None:
        camera = self.camera(FakeCv2([FakeCapture(fail_reads=True)]))
        with self.assertRaises(RuntimeError):
            camera.start(first_frame_timeout_s=0.3)
        self.assertFalse(camera.running)

    def test_capture_writes_the_current_frame_and_reports_it(self) -> None:
        camera = self.camera()
        camera.start()

        snapshot = camera.capture(output_dir=self.root / "shots", prefix="scene")
        self.assertTrue(snapshot.path.is_file())
        self.assertTrue(snapshot.path.name.startswith("scene_"))
        self.assertEqual(snapshot.path.suffix, ".png")
        self.assertEqual((snapshot.width, snapshot.height), (12, 8))
        self.assertEqual(snapshot.bytes_written, snapshot.path.stat().st_size)

        written = cv2.imread(str(snapshot.path))
        self.assertEqual(written.shape, (8, 12, 3))

    def test_png_saves_the_frame_without_loss(self) -> None:
        camera = self.camera()
        camera.start()
        wait_until(lambda: camera.latest_frame() is not None)

        frame = camera.latest_frame()
        snapshot = camera.capture(output_dir=self.root / "shots")
        # The reader thread advances between the two calls, so compare against
        # the value the file itself claims rather than the frame read above.
        written = cv2.imread(str(snapshot.path))
        self.assertEqual(written.shape, frame.shape)
        self.assertEqual(len(set(written.reshape(-1, 3)[:, 0].tolist())), 1)

    def test_captures_in_the_same_millisecond_do_not_overwrite_each_other(self) -> None:
        camera = self.camera()
        camera.start()

        first = camera.capture(output_dir=self.root / "shots")
        second = camera.capture(output_dir=self.root / "shots")
        third = camera.capture(output_dir=self.root / "shots")
        self.assertEqual(len({first.path, second.path, third.path}), 3)
        self.assertEqual(len(existing_snapshots(self.root / "shots")), 3)

    def test_jpg_is_available_and_an_unknown_format_is_refused(self) -> None:
        camera = self.camera()
        camera.start()

        snapshot = camera.capture(output_dir=self.root / "shots", image_format="jpg")
        self.assertEqual(snapshot.path.suffix, ".jpg")
        with self.assertRaises(ValueError):
            camera.capture(output_dir=self.root / "shots", image_format="tiff")

    def test_capture_before_the_camera_runs_is_an_error_not_an_empty_file(self) -> None:
        camera = self.camera()
        with self.assertRaises(RuntimeError):
            camera.capture(output_dir=self.root / "shots")
        self.assertEqual(existing_snapshots(self.root / "shots"), [])

    def test_a_frame_handed_out_is_not_mutated_by_the_reader_thread(self) -> None:
        camera = self.camera()
        camera.start()
        wait_until(lambda: camera.latest_frame() is not None)

        held = camera.latest_frame()
        before = held.copy()
        wait_until(lambda: camera.status().frames > 5)
        np.testing.assert_array_equal(held, before)

    def test_status_reports_progress_and_stop_releases_the_device(self) -> None:
        fake = FakeCv2()
        camera = self.camera(fake)
        camera.start()
        wait_until(lambda: camera.status().frames > 2)

        status = camera.status()
        self.assertTrue(status.running)
        self.assertTrue(status.has_frame)
        self.assertEqual(status.frame_shape, (12, 8))
        self.assertIsNone(status.last_error)

        camera.stop()
        self.assertTrue(fake.captures[0].released)
        self.assertFalse(camera.running)
        self.assertFalse((self.root / "static" / "snapshot_live.jpg").exists())

    def test_a_camera_that_stops_yielding_frames_reports_it(self) -> None:
        capture = FakeCapture()
        camera = self.camera(FakeCv2([capture]))
        camera.start()
        capture.fail_reads = True
        wait_until(lambda: camera.status().last_error is not None, timeout_s=5.0)
        self.assertIn("unplugged", camera.status().last_error)


class GalleryTests(unittest.TestCase):
    def test_snapshots_are_listed_newest_first_and_other_files_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            for index, filename in enumerate(("a.png", "b.jpg", "notes.txt")):
                path = directory / filename
                path.write_bytes(b"x")
                import os

                os.utime(path, (index, index))
            listed = existing_snapshots(directory)
            self.assertEqual([path.name for path in listed], ["b.jpg", "a.png"])
            self.assertEqual([path.name for path in existing_snapshots(directory, limit=1)], ["b.jpg"])

    def test_a_directory_that_does_not_exist_yet_lists_nothing(self) -> None:
        self.assertEqual(existing_snapshots(Path("/nonexistent/snapshots")), [])


if __name__ == "__main__":
    unittest.main()
