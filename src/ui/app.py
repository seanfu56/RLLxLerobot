#!/usr/bin/env python3
"""Streamlit supervisory UI for single-camera ROBOTIS-to-Piper teleoperation."""

from __future__ import annotations

import glob
import math
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

try:
    from ui.recording_control import RecordingCommandController
    from ui.teleop_runtime import (
        ACTION_KEYS,
        EEF_POSE_KEYS,
        DEFAULT_STATIC_PATH,
        CameraSettings,
        RecordingConfig,
        RuntimeConfig,
        RuntimeState,
        get_runtime,
    )
except ModuleNotFoundError as exc:
    if exc.name != "ui":
        raise
    from recording_control import RecordingCommandController
    from teleop_runtime import (
        ACTION_KEYS,
        EEF_POSE_KEYS,
        DEFAULT_STATIC_PATH,
        CameraSettings,
        RecordingConfig,
        RuntimeConfig,
        RuntimeState,
        get_runtime,
    )

# The recording panel drives the start countdown, so it stays responsive. The
# telemetry tables are read by a human and were the dominant rerun cost at 5 Hz.
UI_REFRESH_S = 0.2
TELEMETRY_REFRESH_S = 1.0
PREVIEW_REFRESH_S = 0.1
DEFAULT_OUTPUT_DIR = Path("outputs/teleop")
KNOWN_CAMERA_GLOBS = ("/dev/cam_*", "/dev/video*")
FOURCC_OPTIONS = ("Auto (same as CLI)", "MJPG", "YUYV")


@st.cache_data(ttl=10.0, show_spinner=False)
def _camera_devices() -> list[str]:
    devices: set[str] = set()
    for pattern in KNOWN_CAMERA_GLOBS:
        devices.update(glob.glob(pattern))
    return sorted(devices)


# The recording panel refreshes several times a second and needs this number on
# every pass. Scanning the output directory that often is pure I/O for a value
# that only changes when an episode is written, so the result is cached and
# explicitly invalidated by the writes: _episode_scan_token().
@st.cache_data(ttl=5.0, show_spinner=False)
def _next_episode_index(output_dir: Path, scan_token: int = 0) -> int:
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


def _episode_scan_token() -> int:
    return int(st.session_state.get("_episode_scan_token", 0))


def _invalidate_episode_scan() -> None:
    st.session_state["_episode_scan_token"] = _episode_scan_token() + 1


def _format_age(value: float | None) -> str:
    return "—" if value is None else f"{value * 1000:.1f} ms"


def _state_label(state: RuntimeState) -> str:
    icons = {
        RuntimeState.DISCONNECTED: "⚪",
        RuntimeState.CONNECTING: "🟡",
        RuntimeState.IDLE: "🟢",
        RuntimeState.RECORDING: "🔴",
        RuntimeState.STOPPING: "🟡",
        RuntimeState.ERROR: "🟠",
    }
    return f"{icons[state]} {state.value}"


def _recording_config(
    output_dir: str,
    task: str,
    action_sample_hz: float,
    episode_duration_s: float,
) -> RecordingConfig:
    return RecordingConfig(
        output_dir=Path(output_dir),
        task=task,
        action_sample_hz=action_sample_hz,
        duration_s=episode_duration_s,
    )


st.set_page_config(
    page_title="Single-camera teleoperation",
    page_icon="🦾",
    layout="wide",
)

runtime = get_runtime()
if not isinstance(
    st.session_state.get("_recording_command_controller"),
    RecordingCommandController,
):
    st.session_state["_recording_command_controller"] = RecordingCommandController()
recording_controller: RecordingCommandController = st.session_state[
    "_recording_command_controller"
]
initial_snapshot = runtime.get_snapshot()
hardware_locked = initial_snapshot.state is not RuntimeState.DISCONNECTED

st.title("Single-camera teleoperation")
st.caption(
    "ROBOTIS leader → Piper follower in a dedicated hardware process, with "
    "independently sampled actions and video."
)
st.warning(
    "Closing this browser or losing SSH does not stop the Ubuntu hardware process. "
    "Use Safe disconnect when practical and keep the physical emergency stop within reach.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Connection")
    can_port = st.text_input("Piper CAN interface", "piper_left", disabled=hardware_locked)
    leader_port = st.text_input(
        "ROBOTIS leader port",
        "/dev/robotis_left",
        disabled=hardware_locked,
    )
    speed_rate = st.slider(
        "Piper speed rate (%)",
        min_value=1,
        max_value=100,
        value=50,
        disabled=hardware_locked,
    )
    gripper_max_mm = st.selectbox(
        "Piper jaw maximum (mm)",
        options=(70.0, 100.0),
        index=1,
        disabled=hardware_locked,
        help="Use 100 mm for the installed large jaws; 70 mm is for the small-jaw gripper.",
    )
    limit_step = st.checkbox(
        "Limit relative target per control step",
        value=False,
        disabled=hardware_locked,
    )
    max_relative_target = (
        st.number_input(
            "Maximum joint change (degrees)",
            min_value=0.1,
            value=5.0,
            step=0.1,
            disabled=hardware_locked,
        )
        if limit_step
        else None
    )
    control_hz = st.number_input(
        "Control target (Hz)",
        min_value=1.0,
        max_value=500.0,
        value=200.0,
        step=5.0,
        disabled=hardware_locked,
        help="This rate remains unchanged before, during, and after recording.",
    )
    watchdog_timeout_s = st.number_input(
        "Stale-control watchdog (s)",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
        disabled=hardware_locked,
        help="Hard stop threshold after at least one command has completed.",
    )
    watchdog_startup_timeout_s = st.number_input(
        "Startup watchdog grace (s)",
        min_value=0.5,
        max_value=30.0,
        value=3.0,
        step=0.5,
        disabled=hardware_locked,
        help="Allows slow first hardware reads without weakening the steady-state watchdog.",
    )
    gripper_spring_enabled = st.checkbox(
        "Enable ROBOTIS gripper spring",
        value=False,
        disabled=hardware_locked,
        help=(
            "The spring adds a Dynamixel velocity read and current write to every leader "
            "sample. Leave it off while diagnosing a low control rate."
        ),
    )

    st.header("Camera")
    devices = _camera_devices()
    default_device = "/dev/video3"
    camera_options = devices + ["Custom path…"]
    default_index = devices.index(default_device) if default_device in devices else len(devices)
    selected_device = st.selectbox(
        "Device",
        camera_options,
        index=default_index,
        disabled=hardware_locked,
    )
    camera_device = (
        st.text_input("Camera path", default_device, disabled=hardware_locked)
        if selected_device == "Custom path…"
        else selected_device
    )
    width_column, height_column = st.columns(2)
    camera_width = width_column.number_input(
        "Width",
        min_value=1,
        value=640,
        step=1,
        disabled=hardware_locked,
    )
    camera_height = height_column.number_input(
        "Height",
        min_value=1,
        value=480,
        step=1,
        disabled=hardware_locked,
    )
    camera_fps = st.number_input(
        "Camera capture FPS",
        min_value=1.0,
        max_value=240.0,
        value=30.0,
        step=1.0,
        disabled=hardware_locked,
    )
    camera_fourcc = st.selectbox("Capture FourCC", FOURCC_OPTIONS, disabled=hardware_locked)
    preview_hz = st.number_input(
        "Preview publish rate (Hz)",
        min_value=1.0,
        max_value=30.0,
        value=10.0,
        step=1.0,
        disabled=hardware_locked,
        help=(
            "Every published frame is a JPEG the browser pulls over the SSH tunnel. "
            "Raise it only if the link is fast; it does not affect recorded video."
        ),
    )
    preview_enabled = st.checkbox("Publish browser preview", value=True)

    st.header("Recording")
    output_dir = st.text_input("Output directory", str(DEFAULT_OUTPUT_DIR))
    task = st.text_input("Task description", "pick up cube")
    number_of_episodes = st.number_input(
        "Target episode count",
        min_value=1,
        value=3,
        step=1,
    )
    episode_duration_s = st.number_input(
        "Episode duration (s)",
        min_value=1.0,
        value=15.0,
        step=1.0,
    )
    start_delay_s = st.number_input(
        "Start countdown (s)",
        min_value=0,
        max_value=30,
        value=5,
        step=1,
        help="Robot control continues during the countdown; data recording starts afterward.",
    )
    action_sample_hz = st.number_input(
        "Aligned action/video rate (Hz)",
        min_value=1.0,
        max_value=200.0,
        value=20.0,
        step=1.0,
        help=(
            "Each saved action is paired one-to-one with a video frame at this rate. "
            "This does not change the robot control rate."
        ),
    )

    st.divider()
    current = runtime.get_snapshot()
    st.write(f"Runtime: {_state_label(current.state)}")
    if runtime.process_pid is not None:
        st.caption(f"Dedicated hardware process PID: {runtime.process_pid}")
    if current.state is RuntimeState.DISCONNECTED:
        if st.button("🔌 Connect hardware", type="primary", width="stretch"):
            try:
                runtime.connect(
                    RuntimeConfig(
                        can_port=can_port,
                        leader_port=leader_port,
                        speed_rate=speed_rate,
                        gripper_max_mm=gripper_max_mm,
                        max_relative_target=max_relative_target,
                        control_hz=control_hz,
                        watchdog_timeout_s=watchdog_timeout_s,
                        watchdog_startup_timeout_s=watchdog_startup_timeout_s,
                        gripper_spring_enabled=gripper_spring_enabled,
                        camera=CameraSettings(
                            device=camera_device,
                            width=int(camera_width),
                            height=int(camera_height),
                            fps=camera_fps,
                            fourcc=(
                                None if camera_fourcc == FOURCC_OPTIONS[0] else camera_fourcc
                            ),
                            preview_hz=preview_hz,
                            preview_enabled=preview_enabled,
                        ),
                        preview_path=DEFAULT_STATIC_PATH,
                    )
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Connection failed: {exc}")
    else:
        if st.button("⏹ Safe disconnect (move to rest)", width="stretch"):
            try:
                runtime.disconnect()
                st.rerun()
            except Exception as exc:
                st.error(f"Safe disconnect failed: {exc}")
        if st.button("🛑 SOFTWARE EMERGENCY STOP", type="primary", width="stretch"):
            runtime.emergency_stop()
            st.rerun()
        st.caption(
            "The software emergency stop disables the arm without a rest move. "
            "It is not a substitute for the physical emergency stop."
        )


def _schedule_episode_start(
    source: str,
    *,
    discard_draft: bool = False,
) -> None:
    configured_output = Path(output_dir).expanduser()
    snapshot = runtime.get_snapshot()
    if (
        not discard_draft
        and snapshot.draft is None
        and _next_episode_index(configured_output, _episode_scan_token())
        >= int(number_of_episodes)
    ):
        recording_controller.reject(
            "start_ignored",
            f"{source.capitalize()} start ignored because the target episode count "
            "has already been reached.",
        )
        return
    recording_controller.request_start(
        runtime,
        _recording_config(
            output_dir,
            task,
            action_sample_hz,
            episode_duration_s,
        ),
        float(start_delay_s),
        source=source,
        discard_draft=discard_draft,
    )


try:
    runtime.set_preview_enabled(preview_enabled)
except Exception as exc:
    st.error(f"Could not change preview publication: {exc}")


def preview_panel() -> None:
    if runtime.state is RuntimeState.DISCONNECTED:
        st.info("Connect the hardware to start the local camera preview.")
        return
    if not preview_enabled:
        st.info("Preview publication is disabled. Video recording remains available.")
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
          // Self-scheduling instead of setInterval: over a forwarded SSH port a
          // frame can take longer to arrive than the refresh period, and a fixed
          // interval then queues requests faster than they complete until the tab
          // stalls. Each fetch is only scheduled once the previous one settles.
          function refresh() {{
            const next = new Image();
            const done = (ok) => {{
              if (ok) live.src = next.src;
              window.setTimeout(refresh, {refresh_ms});
            }};
            next.onload = () => done(true);
            next.onerror = () => done(false);
            next.src = "/app/static/live.jpg?t=" + Date.now();
          }}
          refresh();
        </script>
        """,
        height=530,
        scrolling=False,
    )


@st.fragment(run_every=TELEMETRY_REFRESH_S)
def telemetry_panel() -> None:
    snapshot = runtime.get_snapshot()
    if snapshot.last_error:
        st.error(
            f"{snapshot.error_source or 'runtime'}: {snapshot.last_error} "
            "Use Safe disconnect if the robot can still communicate; otherwise use "
            "the physical emergency-stop procedure.",
            icon="🚨",
        )

    metric_columns = st.columns(5)
    metric_columns[0].metric("State", snapshot.state.value)
    metric_columns[1].metric(
        "Control",
        f"{snapshot.effective_control_hz:.1f} Hz",
        f"target {snapshot.target_control_hz:.0f}",
    )
    metric_columns[2].metric("Command age", _format_age(snapshot.last_command_age_s))
    metric_columns[3].metric("Missed deadlines", snapshot.missed_control_deadlines)
    metric_columns[4].metric(
        "Camera",
        f"{snapshot.camera_capture_hz:.1f} Hz",
        f"age {_format_age(snapshot.camera_frame_age_s)}",
    )
    if (
        snapshot.control_period.count >= 50
        and snapshot.target_control_hz > 0
        and snapshot.effective_control_hz < snapshot.target_control_hz * 0.7
    ):
        recommended_hz = max(5.0, snapshot.effective_control_hz * 0.9)
        st.warning(
            f"The hardware path is sustaining {snapshot.effective_control_hz:.1f} Hz, "
            f"well below the {snapshot.target_control_hz:.0f} Hz target. A higher target "
            "does not queue or replay commands. For diagnosis, leave the gripper spring "
            f"off and reconnect near {recommended_hz:.0f} Hz, then inspect the latency table.",
            icon="⚠️",
        )

    timing_rows = []
    for label, timing in (
        ("Control period", snapshot.control_period),
        ("Robot observation", snapshot.observation_latency),
        ("Leader read", snapshot.leader_latency),
        ("Robot send", snapshot.robot_send_latency),
    ):
        timing_rows.append(
            {
                "Stage": label,
                "Mean ms": round(timing.mean_ms, 3),
                "p50 ms": round(timing.p50_ms, 3),
                "p95 ms": round(timing.p95_ms, 3),
                "p99 ms": round(timing.p99_ms, 3),
                "Max ms": round(timing.max_ms, 3),
            }
        )
    st.dataframe(timing_rows, hide_index=True, width="stretch")

    joint_rows = []
    for key in ACTION_KEYS:
        joint_rows.append(
            {
                "Signal": key,
                "Observed": snapshot.latest_observation.get(key),
                "Commanded": snapshot.latest_action.get(key),
            }
        )
    st.dataframe(joint_rows, hide_index=True, width="stretch")
    eef = snapshot.latest_observation
    eef_pose = [
        {
            "Axis": axis.upper(),
            "Value": eef.get(key),
            "Unit": "m" if axis in ("x", "y", "z") else "deg",
        }
        for axis, key in zip(
            ("x", "y", "z", "rx", "ry", "rz"), EEF_POSE_KEYS, strict=True
        )
    ]
    st.dataframe(eef_pose, hide_index=True, width="stretch")
    st.caption(
        f"minimum rolling effective rate {snapshot.minimum_effective_hz:.1f} Hz · "
        f"control stage {snapshot.control_stage} ({_format_age(snapshot.control_stage_age_s)}) · "
        f"preview encode {snapshot.preview_encode_ms:.1f} ms · "
        f"video queue {snapshot.video_queue_depth} · "
        f"dropped video frames {snapshot.video_dropped_frames}"
    )


@st.fragment(run_every=UI_REFRESH_S)
def recording_panel() -> None:
    try:
        recording_controller.advance(runtime)
    except Exception as exc:
        recording_controller.reject(
            "start_failed",
            f"Could not start recording: {exc}",
        )

    snapshot = runtime.get_snapshot()
    draft = snapshot.draft
    configured_output = Path(output_dir).expanduser()
    next_index = (
        draft.episode_index
        if draft is not None
        else _next_episode_index(configured_output, _episode_scan_token())
    )
    st.write(f"Episode {next_index + 1} / {int(number_of_episodes)}")

    recent_message = recording_controller.recent_message()
    if recent_message:
        st.info(recent_message)

    pending = recording_controller.pending
    if pending is not None:
        remaining_s = recording_controller.remaining_seconds()
        total_s = max(pending.deadline_s - pending.requested_at_s, 0.001)
        st.markdown(
            (
                '<div style="padding:1.25rem;text-align:center;border-radius:0.6rem;'
                'background:#fff3cd;color:#664d03;font-size:1.1rem;">'
                f'{pending.source.capitalize()} start accepted<br>'
                '<span style="font-size:3rem;font-weight:700;">'
                f'{math.ceil(remaining_s)}</span><br>seconds until recording'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.progress(
            min(max(1.0 - remaining_s / total_s, 0.0), 1.0),
            text=f"Recording begins in {remaining_s:.1f} s",
        )
        if st.button("Cancel pending start", width="stretch"):
            recording_controller.request_stop(runtime, source="button")
            st.rerun()
        return

    if snapshot.state is RuntimeState.RECORDING:
        duration = snapshot.recording_duration_s or episode_duration_s
        st.progress(
            min(snapshot.recording_elapsed_s / duration, 1.0),
            text=(
                f"Recording {snapshot.recording_elapsed_s:.1f}s / {duration:.1f}s · "
                f"{snapshot.action_samples} actions at {snapshot.action_sample_hz:.1f} Hz"
            ),
        )
        st.caption(
            f"{draft.video_frames if draft else 0} video frames · "
            f"{snapshot.video_queue_depth} queued · "
            f"{snapshot.video_dropped_frames} dropped · "
            f"{snapshot.missed_action_deadlines} sampler deadlines missed"
        )
        if st.button("⏹ Stop episode now", type="primary", width="stretch"):
            try:
                recording_controller.request_stop(runtime, source="button")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not stop episode: {exc}")
        return

    if draft is not None:
        st.success(
            f"Episode {draft.episode_index:03d} is ready: "
            f"{draft.action_samples} action samples, {draft.video_frames} video frames."
        )
        save_column, discard_column, rerecord_column = st.columns(3)
        if save_column.button("💾 Save", type="primary", width="stretch"):
            try:
                saved_path = runtime.save_episode()
                st.session_state["_last_saved_episode"] = str(saved_path)
                _invalidate_episode_scan()
                st.rerun()
            except Exception as exc:
                st.error(f"Save failed; the draft was retained: {exc}")
        if discard_column.button("🗑 Discard", width="stretch"):
            runtime.discard_episode()
            _invalidate_episode_scan()
            st.rerun()
        if rerecord_column.button("↺ Re-record", width="stretch"):
            try:
                _schedule_episode_start("button", discard_draft=True)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not restart the episode: {exc}")
        return

    if st.session_state.get("_last_saved_episode"):
        saved = st.session_state["_last_saved_episode"]
        st.success(f"Saved {saved}")
    if snapshot.state is RuntimeState.IDLE:
        if next_index >= int(number_of_episodes):
            st.success(
                f"Target reached: {int(number_of_episodes)} episode(s) are present in "
                f"{configured_output}."
            )
        elif st.button(
            f"▶ Start episode {next_index + 1}",
            type="primary",
            width="stretch",
        ):
            try:
                _schedule_episode_start("button")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not start recording: {exc}")
    elif snapshot.state is RuntimeState.DISCONNECTED:
        st.info("Connect hardware before recording. A completed draft can still be saved offline.")


left, right = st.columns([3, 2])
with left:
    st.subheader("Live camera")
    preview_panel()
with right:
    st.subheader("Recording")
    recording_panel()

st.subheader("Control telemetry")
telemetry_panel()
