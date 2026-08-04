"""Reading one USB camera and writing single frames to disk.

The teleoperation and policy pages own their camera through LeRobot, inside a
spawned hardware process, because a control loop is running next to it. Taking a
still photograph needs none of that: no arm, no CAN bus, no plugins. This module
therefore talks to OpenCV directly and imports nothing from ``lerobot``, so the
snapshot page runs in any environment that has OpenCV and Streamlit - including
a laptop with a webcam and no robot attached.

One background thread owns the device. It has to, for two reasons that both come
from V4L2 rather than from this code:

* A capture that is not drained keeps handing out the frame that was queued
  when it stalled. Grabbing continuously is what makes "capture" mean *now*
  rather than "whenever the page last reran".
* ``VideoCapture.read`` blocks. Calling it from a Streamlit rerun would put the
  camera's frame interval into every page interaction.

The thread also publishes the preview JPEG. Streamlit reruns never touch the
device; they read a small immutable status object and, when the operator presses
the button, one already-decoded frame.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_STATIC_PATH = Path(__file__).resolve().parent / "static" / "snapshot_live.jpg"
DEFAULT_OUTPUT_DIR = Path("outputs/snapshots")
IMAGE_FORMATS = ("png", "jpg")
KNOWN_CAMERA_GLOBS = ("/dev/cam_*", "/dev/video*")

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_STATIC_PATH",
    "IMAGE_FORMATS",
    "KNOWN_CAMERA_GLOBS",
    "CameraStatus",
    "Snapshot",
    "SnapshotCamera",
    "SnapshotSettings",
    "available_devices",
    "existing_snapshots",
]


@dataclass(frozen=True)
class SnapshotSettings:
    """Which camera to open, and how to serve it to the browser."""

    device: str = "/dev/video0"
    width: int = 640
    height: int = 480
    fps: float = 30.0
    fourcc: str | None = None
    preview_hz: float = 20.0
    jpeg_quality: int = 80

    def __post_init__(self) -> None:
        if not self.device.strip():
            raise ValueError("Camera device must not be empty.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive.")
        if self.fps <= 0 or self.preview_hz <= 0:
            raise ValueError("Camera and preview rates must be positive.")
        if self.fourcc is not None and len(self.fourcc) != 4:
            raise ValueError("Camera FourCC must contain exactly four characters.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100.")

    @property
    def source(self) -> str | int:
        """``/dev/video0`` stays a path; a bare ``0`` becomes an OpenCV index."""
        stripped = self.device.strip()
        return int(stripped) if stripped.isdigit() else stripped


@dataclass(frozen=True)
class Snapshot:
    """One saved still, as the page needs to describe it afterwards."""

    path: Path
    width: int
    height: int
    captured_at: float
    bytes_written: int

    @property
    def label(self) -> str:
        stamp = datetime.fromtimestamp(self.captured_at).strftime("%H:%M:%S")
        return f"{self.path.name} · {self.width}×{self.height} · {stamp}"


@dataclass(frozen=True)
class CameraStatus:
    """What the page shows about a running camera. Immutable, so a rerun that
    reads it cannot see a half-updated set of counters."""

    running: bool
    frames: int
    measured_fps: float
    frame_shape: tuple[int, int] | None
    last_frame_age_s: float | None
    last_error: str | None

    @property
    def has_frame(self) -> bool:
        return self.frame_shape is not None


def available_devices() -> list[str]:
    """The camera nodes present right now, udev aliases first.

    ``config/`` installs ``/dev/cam_*`` symlinks, which survive a replug in a way
    ``/dev/videoN`` numbering does not, so they are the better default.
    """
    import glob

    devices: set[str] = set()
    for pattern in KNOWN_CAMERA_GLOBS:
        devices.update(glob.glob(pattern))
    return sorted(devices, key=lambda name: (not name.startswith("/dev/cam_"), name))


def existing_snapshots(output_dir: Path, limit: int | None = None) -> list[Path]:
    """Saved stills, newest first. Missing directory is not an error - it only
    means nothing has been captured into it yet."""
    directory = Path(output_dir)
    if not directory.is_dir():
        return []
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower().lstrip(".") in IMAGE_FORMATS
    ]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files[:limit] if limit is not None else files


class SnapshotCamera:
    """One OpenCV capture, drained by one thread, readable from any other."""

    def __init__(
        self,
        settings: SnapshotSettings,
        preview_path: Path = DEFAULT_STATIC_PATH,
        cv2_module: Any | None = None,
    ) -> None:
        if cv2_module is None:
            import cv2 as cv2_module  # noqa: PLC0415 - kept out of import time for tests

        self.settings = settings
        self._cv2 = cv2_module
        self._preview_path = Path(preview_path)
        self._preview_tmp = self._preview_path.with_name(f".{self._preview_path.name}.tmp")
        self._capture: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._first_frame = threading.Event()
        self._lock = threading.Lock()
        self._latest: Any | None = None
        self._latest_at = 0.0
        self._frames = 0
        self._measured_fps = 0.0
        self._last_error: str | None = None

    # -- lifecycle ----------------------------------------------------------

    def start(self, first_frame_timeout_s: float = 5.0) -> None:
        """Open the device and block until it has actually produced a frame.

        A ``VideoCapture`` that opened is not yet a camera that works: an
        unsupported resolution or a device held by another process shows up on
        the first ``read``, not on construction. Waiting here means the page can
        report the failure next to the button the operator just pressed.
        """
        if self._thread is not None:
            raise RuntimeError("This camera is already running.")

        capture = self._open()
        self._capture = capture
        self._stop.clear()
        self._first_frame.clear()
        self._preview_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="snapshot-camera", daemon=True)
        self._thread.start()

        if not self._first_frame.wait(first_frame_timeout_s):
            error = self._last_error
            self.stop()
            raise RuntimeError(
                error
                or f"{self.settings.device} opened but produced no frame within "
                f"{first_frame_timeout_s:.0f} s. Another process may be holding it."
            )

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and threading.current_thread() is not thread:
            thread.join(timeout=3.0)
        if self._capture is not None:
            try:
                self._capture.release()
            finally:
                self._capture = None
        self._preview_tmp.unlink(missing_ok=True)
        self._preview_path.unlink(missing_ok=True)

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def _open(self) -> Any:
        cv2 = self._cv2
        source = self.settings.source
        # V4L2 explicitly: the default backend on Linux may pick GStreamer,
        # which ignores the FourCC and resolution requests below.
        backend = getattr(cv2, "CAP_V4L2", 0) if isinstance(source, str) else getattr(cv2, "CAP_ANY", 0)
        capture = cv2.VideoCapture(source, backend)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open {self.settings.device}. Check that it exists and that no "
                "other page or recording is already using it."
            )

        # FourCC before the frame size: MJPG and YUYV support different
        # resolution sets, so a size chosen under the old codec can be silently
        # rejected once the codec changes.
        if self.settings.fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.settings.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
        capture.set(cv2.CAP_PROP_FPS, self.settings.fps)
        # One-frame buffer where the driver honours it: the point of this page
        # is that the saved still is the scene as it is now.
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001 - optional on most V4L2 drivers
            pass
        return capture

    # -- the reader thread --------------------------------------------------

    def _run(self) -> None:
        capture = self._capture
        assert capture is not None
        preview_period_s = 1.0 / self.settings.preview_hz
        next_preview = time.perf_counter()
        window_started = time.perf_counter()
        window_frames = 0
        failures = 0
        try:
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 30:
                        with self._lock:
                            self._last_error = (
                                f"{self.settings.device} stopped returning frames. "
                                "It may have been unplugged."
                            )
                        return
                    self._stop.wait(0.05)
                    continue

                failures = 0
                now = time.perf_counter()
                window_frames += 1
                with self._lock:
                    self._latest = frame
                    self._latest_at = now
                    self._frames += 1
                    if now - window_started >= 1.0:
                        self._measured_fps = window_frames / (now - window_started)
                        window_started, window_frames = now, 0
                self._first_frame.set()

                if now >= next_preview:
                    self._publish(frame)
                    next_preview += preview_period_s
                    if now >= next_preview:
                        # Encoding fell behind the preview rate; skip whole
                        # periods rather than trying to catch up frame by frame.
                        skipped = math.floor((now - next_preview) / preview_period_s) + 1
                        next_preview += skipped * preview_period_s
        except BaseException as exc:  # noqa: BLE001 - surfaced through status()
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"

    def _publish(self, frame: Any) -> None:
        """Write the preview JPEG atomically.

        The browser polls this file several times a second. Without the rename
        it would sooner or later fetch a half-written file and drop the frame.
        """
        cv2 = self._cv2
        ok, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality]
        )
        if not ok:
            return
        self._preview_tmp.write_bytes(buffer.tobytes())
        os.replace(self._preview_tmp, self._preview_path)

    # -- readers ------------------------------------------------------------

    def latest_frame(self) -> Any | None:
        """A copy of the newest frame, or ``None`` before the first one.

        The copy matters: the reader thread hands the same array to OpenCV again
        on the next iteration with some backends, so a caller holding the
        original could watch its still change underneath it.
        """
        with self._lock:
            frame = self._latest
        return None if frame is None else frame.copy()

    def status(self) -> CameraStatus:
        with self._lock:
            frame = self._latest
            age = time.perf_counter() - self._latest_at if frame is not None else None
            shape = (int(frame.shape[1]), int(frame.shape[0])) if frame is not None else None
            return CameraStatus(
                running=self.running,
                frames=self._frames,
                measured_fps=self._measured_fps,
                frame_shape=shape,
                last_frame_age_s=age,
                last_error=self._last_error,
            )

    # -- the actual point of the page ---------------------------------------

    def capture(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        prefix: str = "snapshot",
        image_format: str = "png",
        jpeg_quality: int = 95,
    ) -> Snapshot:
        """Save the current frame and return where it went.

        PNG by default. These stills end up as conditioning frames for the video
        models in ``src/video`` and ``src/cosmos``, and a JPEG generation
        artefact in a conditioning frame is indistinguishable from something the
        model was asked to reproduce.
        """
        image_format = image_format.lower().lstrip(".")
        if image_format not in IMAGE_FORMATS:
            raise ValueError(f"image_format must be one of {IMAGE_FORMATS}, got {image_format!r}")
        if not prefix.strip():
            raise ValueError("A file name prefix is required.")

        frame = self.latest_frame()
        if frame is None:
            raise RuntimeError("No camera frame has arrived yet; start the camera first.")

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        captured_at = time.time()
        path = _unique_path(directory, prefix.strip(), image_format, captured_at)

        cv2 = self._cv2
        params = (
            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)] if image_format == "jpg" else []
        )
        ok, buffer = cv2.imencode(f".{image_format}", frame, params)
        if not ok:
            raise RuntimeError(f"OpenCV could not encode the frame as {image_format.upper()}.")
        data = buffer.tobytes()
        # Same atomic write as the preview: the gallery below lists this
        # directory while captures are happening.
        tmp_path = path.with_name(f".{path.name}.tmp")
        tmp_path.write_bytes(data)
        os.replace(tmp_path, path)

        return Snapshot(
            path=path,
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
            captured_at=captured_at,
            bytes_written=len(data),
        )


def _unique_path(directory: Path, prefix: str, image_format: str, captured_at: float) -> Path:
    """``prefix_20260730-141502-337.png``, with a counter only if that collides.

    Millisecond stamps are unique in practice for a human pressing a button, but
    a burst driven by the countdown timer can land two captures in the same
    millisecond, and silently overwriting the first one would be the worst
    possible outcome for a page whose only job is to keep frames.
    """
    stamp = datetime.fromtimestamp(captured_at).strftime("%Y%m%d-%H%M%S-%f")[:-3]
    path = directory / f"{prefix}_{stamp}.{image_format}"
    counter = 1
    while path.exists():
        path = directory / f"{prefix}_{stamp}-{counter}.{image_format}"
        counter += 1
    return path
