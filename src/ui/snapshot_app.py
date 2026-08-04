#!/usr/bin/env python3
"""Streamlit page for photographing the camera view and keeping the frame.

The other pages in this directory drive the arm. This one does not: it opens a
camera, shows it, and writes single frames to disk. It exists because most of
the things built on top of this repository start from a still - a conditioning
frame for the Cosmos or diffusion video models, a goal image for a
goal-conditioned rollout, a reference photograph of a scene layout that has to
be rebuilt tomorrow - and taking one previously meant starting the whole
teleoperation stack with a robot attached.

The camera lives in ``snapshot_camera.py``, in one background thread, shared by
every browser session through ``st.cache_resource``. As on the teleoperation
page the browser fetches preview frames directly from the static file the
camera thread writes, so watching the preview costs no Streamlit reruns; only
pressing a button does.
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

try:
    from ui.snapshot_camera import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_STATIC_PATH,
        IMAGE_FORMATS,
        SnapshotCamera,
        SnapshotSettings,
        available_devices,
        existing_snapshots,
    )
except ModuleNotFoundError as exc:
    if exc.name != "ui":
        raise
    from snapshot_camera import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_STATIC_PATH,
        IMAGE_FORMATS,
        SnapshotCamera,
        SnapshotSettings,
        available_devices,
        existing_snapshots,
    )

PREVIEW_REFRESH_S = 0.1
STATUS_REFRESH_S = 1.0
COUNTDOWN_REFRESH_S = 0.1
FOURCC_OPTIONS = ("Auto (driver default)", "MJPG", "YUYV")
DELAY_OPTIONS = (0, 3, 5, 10)
GALLERY_COLUMNS = 4
GALLERY_LIMIT = 12

st.set_page_config(page_title="Camera snapshot", page_icon="📸", layout="wide")


# ---------------------------------------------------------------------------
# the shared camera
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _camera_holder() -> dict[str, SnapshotCamera | None]:
    """One slot, shared by every session in this server process.

    A second browser tab must not open a second handle to the same ``/dev/video``
    node - the driver would either refuse it or hand the two tabs alternating
    frames. The holder is mutable so that changing the resolution can replace
    what is inside it without invalidating the cache entry the other tabs hold.
    """
    return {"camera": None}


def _camera() -> SnapshotCamera | None:
    return _camera_holder()["camera"]


def _start_camera(settings: SnapshotSettings) -> None:
    holder = _camera_holder()
    _stop_camera()
    camera = SnapshotCamera(settings, preview_path=DEFAULT_STATIC_PATH)
    camera.start()
    holder["camera"] = camera


def _stop_camera() -> None:
    holder = _camera_holder()
    camera = holder["camera"]
    holder["camera"] = None
    if camera is not None:
        camera.stop()


def _remember(message: str, kind: str = "success") -> None:
    """Carry a result across the rerun that a button press triggers."""
    st.session_state["flash"] = (kind, message)


def _take_snapshot() -> None:
    camera = _camera()
    if camera is None:
        _remember("Start the camera before capturing.", "warning")
        return
    try:
        snapshot = camera.capture(
            output_dir=Path(st.session_state["output_dir"]).expanduser(),
            prefix=st.session_state["prefix"],
            image_format=st.session_state["image_format"],
            jpeg_quality=st.session_state["save_quality"],
        )
    except Exception as exc:  # noqa: BLE001 - every failure belongs on the page
        _remember(f"Could not save the snapshot: {exc}", "error")
        return
    st.session_state["last_snapshot"] = snapshot
    _remember(f"Saved {snapshot.path} ({snapshot.bytes_written / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

st.title("📸 Camera snapshot")

with st.sidebar:
    st.header("Camera")
    devices = available_devices()
    if devices:
        device = st.selectbox(
            "Device",
            devices,
            index=0,
            help="/dev/cam_* aliases come from config/ udev rules and survive a replug.",
        )
        device = st.text_input("Or type another device", value=device)
    else:
        device = st.text_input(
            "Device",
            value="/dev/video0",
            help="No /dev/video* node was found. A bare number is used as an OpenCV index.",
        )

    resolution = st.selectbox(
        "Resolution",
        ("640×480", "800×600", "1280×720", "1920×1080"),
        index=0,
    )
    width, height = (int(part) for part in resolution.split("×"))
    fps = st.number_input("Camera FPS", min_value=1.0, max_value=120.0, value=30.0, step=1.0)
    fourcc_choice = st.selectbox("FourCC", FOURCC_OPTIONS, index=0)
    fourcc = None if fourcc_choice.startswith("Auto") else fourcc_choice
    preview_quality = st.slider(
        "Preview JPEG quality",
        min_value=30,
        max_value=95,
        value=80,
        help="Preview only. Saved snapshots are written from the raw frame.",
    )

    st.divider()
    st.header("Saving")
    st.text_input("Output directory", value=str(DEFAULT_OUTPUT_DIR), key="output_dir")
    st.text_input("File name prefix", value="snapshot", key="prefix")
    st.selectbox(
        "Format",
        IMAGE_FORMATS,
        index=0,
        key="image_format",
        help="PNG is lossless, which is what the video models want from a conditioning frame.",
    )
    st.slider(
        "JPEG quality",
        min_value=50,
        max_value=100,
        value=95,
        key="save_quality",
        disabled=st.session_state.get("image_format", "png") != "jpg",
    )

settings = SnapshotSettings(
    device=device,
    width=width,
    height=height,
    fps=float(fps),
    fourcc=fourcc,
    jpeg_quality=int(preview_quality),
)

camera = _camera()
running = camera is not None and camera.running
settings_changed = running and camera is not None and camera.settings != settings

# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

flash = st.session_state.pop("flash", None)
if flash is not None:
    getattr(st, flash[0])(flash[1])

start_column, stop_column, capture_column, delay_column = st.columns([1, 1, 2, 1])

with start_column:
    if st.button(
        "Restart camera" if settings_changed else "Start camera",
        type="primary",
        disabled=running and not settings_changed,
        width="stretch",
    ):
        try:
            _start_camera(settings)
        except Exception as exc:  # noqa: BLE001 - a bad device is normal here
            _remember(str(exc), "error")
        st.rerun()

with stop_column:
    if st.button("Stop camera", disabled=not running, width="stretch"):
        _stop_camera()
        st.rerun()

with delay_column:
    delay_s = st.selectbox(
        "Delay",
        DELAY_OPTIONS,
        index=0,
        format_func=lambda value: "None" if value == 0 else f"{value} s",
        label_visibility="collapsed",
    )

with capture_column:
    pending = st.session_state.get("capture_at")
    if pending is None:
        if st.button(
            "📷 Take snapshot",
            type="primary",
            disabled=not running,
            width="stretch",
        ):
            if delay_s:
                # The countdown fragment below fires the capture. Doing it here
                # with a sleep would block this session's whole server thread,
                # and the preview would freeze exactly while the operator is
                # arranging the scene the delay exists to give them time for.
                st.session_state["capture_at"] = time.monotonic() + delay_s
            else:
                _take_snapshot()
            st.rerun()
    elif st.button("Cancel countdown", width="stretch"):
        st.session_state.pop("capture_at", None)
        st.rerun()

if settings_changed:
    st.info("The camera is running with different settings. Restart it to apply these.")

# ---------------------------------------------------------------------------
# preview, countdown, status
# ---------------------------------------------------------------------------


@st.fragment(run_every=COUNTDOWN_REFRESH_S)
def countdown_panel() -> None:
    """Only rendered while a countdown is pending, so the page is otherwise
    idle between button presses."""
    deadline = st.session_state.get("capture_at")
    if deadline is None:
        return
    remaining = deadline - time.monotonic()
    if remaining > 0:
        st.subheader(f"Capturing in {remaining:.1f} s")
        return
    st.session_state.pop("capture_at", None)
    _take_snapshot()
    # A full rerun rather than a fragment rerun: the result message and the
    # gallery both live outside this fragment.
    st.rerun()


def preview_panel() -> None:
    if not running:
        st.info("Start the camera to see the live view.")
        return
    refresh_ms = int(PREVIEW_REFRESH_S * 1000)
    components.html(
        f"""
        <style>
          html, body {{
            margin: 0;
            background: transparent;
            overflow: hidden;
          }}
          #live {{
            display: block;
            width: 100%;
            height: 520px;
            object-fit: contain;
            border-radius: 0.5rem;
            background: #111;
          }}
        </style>
        <img id="live" alt="Live camera">
        <script>
          const live = document.getElementById("live");
          // Self-scheduling rather than setInterval: over a forwarded SSH port a
          // frame can take longer to arrive than the refresh period, and a fixed
          // interval then queues requests faster than they complete.
          function refresh() {{
            const next = new Image();
            const done = (ok) => {{
              if (ok) live.src = next.src;
              window.setTimeout(refresh, {refresh_ms});
            }};
            next.onload = () => done(true);
            next.onerror = () => done(false);
            next.src = "/app/static/{DEFAULT_STATIC_PATH.name}?t=" + Date.now();
          }}
          refresh();
        </script>
        """,
        height=530,
        scrolling=False,
    )


@st.fragment(run_every=STATUS_REFRESH_S)
def status_panel() -> None:
    active = _camera()
    if active is None:
        st.caption("Camera stopped.")
        return
    status = active.status()
    left, right = st.columns(2)
    left.metric("Measured FPS", f"{status.measured_fps:.1f}")
    right.metric("Frames", f"{status.frames}")
    if status.frame_shape is not None:
        st.caption(f"Frame {status.frame_shape[0]}×{status.frame_shape[1]} from {active.settings.device}")
    if status.last_error:
        st.error(status.last_error, icon="🚨")
    elif not status.running:
        st.warning("The camera thread has stopped. Restart it.")


preview_column, side_column = st.columns([3, 1])
with preview_column:
    if st.session_state.get("capture_at") is not None:
        countdown_panel()
    preview_panel()
with side_column:
    st.subheader("Status")
    status_panel()
    last = st.session_state.get("last_snapshot")
    if last is not None and Path(last.path).is_file():
        st.subheader("Last snapshot")
        st.image(str(last.path), width="stretch")
        st.caption(last.label)
        st.download_button(
            "Download",
            data=Path(last.path).read_bytes(),
            file_name=last.path.name,
            mime=f"image/{'jpeg' if last.path.suffix == '.jpg' else 'png'}",
            width="stretch",
        )

# ---------------------------------------------------------------------------
# what has been saved
# ---------------------------------------------------------------------------

st.divider()
output_dir = Path(st.session_state["output_dir"]).expanduser()
saved = existing_snapshots(output_dir, limit=GALLERY_LIMIT)
total = len(existing_snapshots(output_dir))

st.subheader(f"Saved snapshots · {total} in {output_dir}")
if not saved:
    st.caption("Nothing captured into this directory yet.")
else:
    for row_start in range(0, len(saved), GALLERY_COLUMNS):
        row = saved[row_start : row_start + GALLERY_COLUMNS]
        for column, path in zip(st.columns(GALLERY_COLUMNS), row):
            with column:
                st.image(str(path), width="stretch")
                st.caption(path.name)
                st.download_button(
                    "Download",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=f"image/{'jpeg' if path.suffix == '.jpg' else 'png'}",
                    key=f"download-{path.name}",
                    width="stretch",
                )
    if total > len(saved):
        st.caption(f"Showing the {len(saved)} newest. The rest are in {output_dir}.")
