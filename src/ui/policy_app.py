#!/usr/bin/env python3
"""Streamlit supervisory UI for running a trained policy on the Piper.

The page chooses a checkpoint, loads it into a dedicated hardware process, and
supervises timed rollouts. It is the inference counterpart of ``app.py`` and
follows the same rule: Streamlit never touches hardware or torch, it only sends
commands and renders immutable snapshots.
"""

from __future__ import annotations

import glob
import json
import math
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

try:
    from ui.policy_runtime import (
        DEFAULT_MODEL_ROOT,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PREVIEW_PATH,
        CameraSettings,
        PolicyInfo,
        PolicyRuntimeConfig,
        PolicySettings,
        PolicyState,
        RolloutConfig,
        get_runtime,
    )
    from ui.teleop_runtime import ACTION_KEYS
except ModuleNotFoundError as exc:
    if exc.name != "ui":
        raise
    from policy_runtime import (
        DEFAULT_MODEL_ROOT,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PREVIEW_PATH,
        CameraSettings,
        PolicyInfo,
        PolicyRuntimeConfig,
        PolicySettings,
        PolicyState,
        RolloutConfig,
        get_runtime,
    )
    from teleop_runtime import ACTION_KEYS

try:
    from policy.checkpoints import RunInfo, discover_runs
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from policy.checkpoints import RunInfo, discover_runs

UI_REFRESH_S = 0.2
# Runs written by src/policy/train.py land under models/simple. Older sweeps sit
# under models/sweeps; keeping it listed makes those runs visible even though
# their weights can no longer be loaded.
DEFAULT_MODEL_ROOTS = f"models/simple {DEFAULT_MODEL_ROOT}"
TELEMETRY_REFRESH_S = 1.0
PREVIEW_REFRESH_S = 0.1
KNOWN_CAMERA_GLOBS = ("/dev/cam_*", "/dev/video*")
FOURCC_OPTIONS = ("Auto (same as CLI)", "MJPG", "YUYV")

# Median initial measured state across the recorded demonstrations; the same
# pose every trial homes to before it starts.
DEFAULT_START_POSE = {
    "joint_1.pos": -1.125,
    "joint_2.pos": 1.602,
    "joint_3.pos": -0.932,
    "joint_4.pos": 0.0,
    "joint_5.pos": 22.798,
    "joint_6.pos": 20.760,
    "gripper.pos": 99.5,
}


def _split_roots(model_root: str) -> list[Path]:
    """Accept several roots in one box, comma or whitespace separated."""
    return [Path(entry) for entry in model_root.replace(",", " ").split() if entry]


@st.cache_data(ttl=30.0, show_spinner=False)
def _runs(model_root: str, scan_token: int = 0) -> list[RunInfo]:
    """Every run under every configured root, in the order the roots are given."""
    seen: set[Path] = set()
    runs: list[RunInfo] = []
    for root in _split_roots(model_root):
        for run in discover_runs(root):
            if run.directory.resolve() not in seen:
                seen.add(run.directory.resolve())
                runs.append(run)
    return runs


def _run_labels(runs: list[RunInfo]) -> list[str]:
    """Display names, disambiguated by root only where a name repeats."""
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.name] = counts.get(run.name, 0) + 1
    return [
        f"{run.directory.parent.name}/{run.name}" if counts[run.name] > 1 else run.name
        for run in runs
    ]


@st.cache_data(ttl=10.0, show_spinner=False)
def _camera_devices() -> list[str]:
    devices: set[str] = set()
    for pattern in KNOWN_CAMERA_GLOBS:
        devices.update(glob.glob(pattern))
    return sorted(devices)


def _format_age(value: float | None) -> str:
    return "—" if value is None else f"{value * 1000:.1f} ms"


def _state_label(state: PolicyState) -> str:
    icons = {
        PolicyState.DISCONNECTED: "⚪",
        PolicyState.CONNECTING: "🟡",
        PolicyState.IDLE: "🟢",
        PolicyState.RUNNING: "🔵",
        PolicyState.STOPPING: "🟡",
        PolicyState.ERROR: "🟠",
    }
    return f"{icons[state]} {state.value}"


st.set_page_config(page_title="Policy inference", page_icon="🤖", layout="wide")

runtime = get_runtime()
snapshot = runtime.get_snapshot()
hardware_locked = snapshot.state is not PolicyState.DISCONNECTED
driving = snapshot.state is PolicyState.RUNNING

st.title("Policy inference")
st.caption(
    "Run a trained checkpoint closed-loop on the Piper: overhead frame plus measured "
    "joint state in, joint targets out."
)
st.warning(
    "The arm is commanded by a neural network. Keep the physical emergency stop within "
    "reach, start with a low speed rate and a small per-step clamp, and never run this "
    "page and the teleoperation page against the same arm at the same time.",
    icon="⚠️",
)

# ---------------------------------------------------------------------------
# checkpoint selection
# ---------------------------------------------------------------------------


def checkpoint_panel() -> PolicySettings | None:
    """Pick a run and a checkpoint tag; return what the hardware process needs."""
    model_root = st.text_input(
        "Model roots",
        DEFAULT_MODEL_ROOTS,
        disabled=driving,
        help="One or more directories, separated by spaces or commas.",
    )
    if st.button("🔄 Rescan checkpoints", width="stretch", disabled=driving):
        st.session_state["_run_scan_token"] = st.session_state.get("_run_scan_token", 0) + 1

    runs = _runs(model_root, st.session_state.get("_run_scan_token", 0))
    if not runs:
        st.error(f"No run directory with a .pt checkpoint under {model_root}.")
        return None

    labels = _run_labels(runs)
    loaded = snapshot.policy
    default_index = next(
        (index for index, run in enumerate(runs) if loaded and run.name == loaded.run_name), 0
    )
    label = st.selectbox(
        "Run",
        labels,
        index=default_index,
        disabled=driving,
        help=(
            "One directory per training run, as written by src/policy/train.py."
        ),
    )
    run = runs[labels.index(label)]
    st.caption(run.summary())
    # The rollout task text defaults to what this run was trained on.
    st.session_state["_selected_run_task"] = run.tasks[0] if run.tasks else ""

    tags = [entry.tag for entry in run.checkpoints]
    tag = st.selectbox(
        "Checkpoint",
        tags,
        index=0,
        disabled=driving,
        format_func=lambda value: run.checkpoint(value).label,
        help="best.pt is the lowest validation loss; step_* are periodic snapshots.",
    )
    checkpoint = run.checkpoint(tag)

    config = run.config

    # Runs from the removed DINOv2 transformer policy are still listed - their
    # metrics remain readable through sweeps/collect_results.py - but nothing in
    # the tree can build that architecture any more, so loading would only fail
    # later with a state-dict mismatch.
    if run.policy not in ("simple", "?"):
        st.error(
            f"{run.name} was trained by the DINOv2 transformer policy (`{run.policy}`), which "
            "has been removed. Its metrics are still readable, but the weights can no longer "
            "be loaded. Pick a run from `models/simple`."
        )
        return None

    with st.expander("Run details"):
        st.write(
            {
                "policy": run.policy,
                "chunk_size": run.chunk_size,
                "image_size": run.image_size,
                "bundle": run.bundle,
                "tasks": list(run.tasks),
                "fps": run.fps,
                "best_val_loss": run.best_val_loss,
                "best_val_mae": run.best_val_mae,
                "last_step": run.last_step,
                "n_obs_steps": config.get("n_obs_steps"),
                "n_action_steps": config.get("n_action_steps"),
                "down_dims": config.get("down_dims"),
                "action_repr": config.get("action_repr"),
                "cond_dropout": config.get("cond_dropout"),
                "guidance_weight": config.get("guidance_weight"),
                "checkpoint": str(checkpoint.path),
            }
        )

    if int(config.get("n_obs_steps", 1)) > 1:
        st.caption(
            f"↺ Conditions on the last {config['n_obs_steps']} frames. The control rate must "
            "match the recording rate, or the motion between them is not what it trained on."
        )

    chunk_size = run.chunk_size or 8
    # The simple policy records the horizon it was trained to execute; the
    # transformer checkpoints predict relative to the measured state, where a
    # shorter horizon re-anchors more often.
    default_action_steps = int(config.get("n_action_steps") or min(4, chunk_size))
    action_steps = st.slider(
        "Actions executed per prediction",
        min_value=1,
        max_value=chunk_size,
        value=max(1, min(default_action_steps, chunk_size)),
        disabled=driving,
        help=(
            "LeRobot's n_action_steps: how much of each predicted chunk is executed "
            "before the policy re-plans from a fresh frame."
        ),
    )
    override_sampler = run.objective in ("diffusion", "flow") and st.checkbox(
        "Override sampler steps",
        value=False,
        disabled=driving,
        help="Fewer steps means lower latency per prediction and a coarser action.",
    )
    num_inference_steps = (
        int(
            st.number_input(
                "Sampler steps",
                min_value=1,
                max_value=200,
                value=int(config.get("num_inference_steps") or 10),
                step=1,
                disabled=driving,
            )
        )
        if override_sampler
        else None
    )

    # The weight itself is retuned live, after loading, in guidance_panel below.
    # Loading uses whatever the checkpoint was trained with.
    cond_dropout = float(config.get("cond_dropout") or 0.0)
    st.caption(
        f"🎛 Classifier-free guidance available (trained with cond dropout {cond_dropout:g}); "
        "set the weight after loading."
        if cond_dropout > 0.0
        else "Guidance unavailable: trained with --cond-dropout 0."
    )
    device = st.selectbox("Device", ("cuda", "cpu"), index=0, disabled=driving)
    use_ema = st.checkbox(
        "Use EMA weights",
        value=True,
        disabled=driving,
        help="Training tracks an exponential moving average; it is usually the better policy.",
    )

    try:
        return PolicySettings(
            checkpoint=checkpoint.path,
            action_steps=action_steps,
            num_inference_steps=num_inference_steps,
            device=device,
            use_ema=use_ema,
        )
    except ValueError as exc:
        st.error(str(exc))
        return None


def guidance_panel(loaded: PolicyInfo | None) -> None:
    """Retune classifier-free guidance on the policy already in the hardware process.

    Changing the weight here costs nothing but a queue flush; reloading the
    checkpoint would rebuild the CUDA context and repeat the warm-up. Only shown
    for checkpoints that were trained with conditioning dropout, because the
    runtime refuses any other weight on the rest.
    """
    if loaded is None or not loaded.supports_guidance:
        return

    st.header("Guidance")
    weight = float(
        st.slider(
            "Classifier-free guidance weight",
            min_value=1.0,
            max_value=5.0,
            value=float(loaded.guidance_weight),
            step=0.1,
            disabled=driving,
            help=(
                "1.0 is plain conditional sampling. Higher sharpens the policy onto what the "
                "current observation implies and narrows the range of behaviours it will "
                "commit to. Robotics weights are modest; 1.5-3.0 is the usual range."
            ),
        )
    )

    if abs(weight - loaded.guidance_weight) < 1e-9:
        st.caption(
            f"Active: {loaded.guidance_weight:g}"
            + (" (plain conditional)" if loaded.guidance_weight == 1.0 else "")
        )
    elif st.button(f"Apply guidance {weight:g}", width="stretch", disabled=driving):
        try:
            runtime.set_guidance_weight(weight)
            st.rerun()
        except Exception as exc:
            st.error(f"Could not change the guidance weight: {exc}")

    if weight > 1.0:
        st.caption(
            "Guidance runs the U-Net twice per sampler step. It is close to free on a GPU, "
            "but check the inference p95 against the control budget in the telemetry panel."
        )


with st.sidebar:
    st.header("Checkpoint")
    policy_settings = checkpoint_panel()

    loaded_policy = snapshot.policy
    if loaded_policy is not None:
        st.success(loaded_policy.headline())
        if loaded_policy.warmup_s is not None:
            st.caption(f"warm-up inference {loaded_policy.warmup_s * 1000:.0f} ms")
    else:
        st.info("No checkpoint is loaded in the hardware process yet.")

    load_label = "📦 Load checkpoint" if loaded_policy is None else "📦 Load / replace checkpoint"
    if st.button(load_label, type="primary", width="stretch", disabled=policy_settings is None or driving):
        assert policy_settings is not None
        try:
            with st.spinner("Loading weights and running one warm-up inference…"):
                runtime.load_policy(policy_settings)
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load the checkpoint: {exc}")

    guidance_panel(loaded_policy)

    st.header("Robot")
    can_port = st.text_input("Piper CAN interface", "piper_left", disabled=hardware_locked)
    speed_rate = st.slider(
        "Piper speed rate (%)",
        min_value=1,
        max_value=100,
        value=30,
        disabled=hardware_locked,
        help="Evaluation runs slower than teleoperation on purpose.",
    )
    gripper_max_mm = st.selectbox(
        "Piper jaw maximum (mm)",
        options=(70.0, 100.0),
        index=1,
        disabled=hardware_locked,
    )
    max_relative_target = st.number_input(
        "Maximum joint change per step (deg/mm)",
        min_value=0.1,
        max_value=45.0,
        value=2.0,
        step=0.1,
        disabled=hardware_locked,
        help="The plugin clamps every command against the previous one. This is the "
        "main protection against a bad prediction; raise it only deliberately.",
    )
    control_hz = st.number_input(
        "Control rate (Hz)",
        min_value=1.0,
        max_value=100.0,
        value=float(loaded_policy.fps if loaded_policy else 20.0),
        step=1.0,
        disabled=hardware_locked,
        help="Match the rate the demonstrations were recorded at (20 Hz).",
    )
    watchdog_timeout_s = st.number_input(
        "Stale-control watchdog (s)",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
        disabled=hardware_locked,
    )
    watchdog_startup_timeout_s = st.number_input(
        "Startup watchdog grace (s)",
        min_value=0.5,
        max_value=60.0,
        value=10.0,
        step=0.5,
        disabled=hardware_locked,
        help="Homing to the start pose happens inside connect, before the first control step.",
    )
    home_on_connect = st.checkbox(
        "Home to the start pose on connect",
        value=True,
        disabled=hardware_locked,
        help="Evaluation is only comparable from the pose the demonstrations started in.",
    )
    start_pose_text = st.text_area(
        "Start / rest pose (JSON)",
        json.dumps(DEFAULT_START_POSE),
        height=120,
        disabled=hardware_locked,
    )

    st.header("Camera")
    devices = _camera_devices()
    default_device = "/dev/video3"
    camera_options = devices + ["Custom path…"]
    default_index = devices.index(default_device) if default_device in devices else len(devices)
    selected_device = st.selectbox("Device", camera_options, index=default_index, disabled=hardware_locked)
    camera_device = (
        st.text_input("Camera path", default_device, disabled=hardware_locked)
        if selected_device == "Custom path…"
        else selected_device
    )
    width_column, height_column = st.columns(2)
    camera_width = width_column.number_input("Width", min_value=1, value=640, step=1, disabled=hardware_locked)
    camera_height = height_column.number_input("Height", min_value=1, value=480, step=1, disabled=hardware_locked)
    camera_fps = st.number_input(
        "Camera capture FPS", min_value=1.0, max_value=240.0, value=30.0, step=1.0, disabled=hardware_locked
    )
    camera_fourcc = st.selectbox("Capture FourCC", FOURCC_OPTIONS, disabled=hardware_locked)
    preview_hz = st.number_input(
        "Preview publish rate (Hz)", min_value=1.0, max_value=30.0, value=10.0, step=1.0, disabled=hardware_locked
    )
    preview_enabled = st.checkbox("Publish browser preview", value=True)

    st.header("Rollout")
    output_dir = st.text_input("Output directory", str(DEFAULT_OUTPUT_DIR))
    task = st.text_input(
        "Task description", st.session_state.get("_selected_run_task") or "pick up can"
    )
    rollout_duration_s = st.number_input(
        "Rollout duration (s)", min_value=1.0, max_value=600.0, value=20.0, step=1.0
    )
    start_delay_s = st.number_input(
        "Start countdown (s)",
        min_value=0,
        max_value=30,
        value=5,
        step=1,
        help="The arm is not commanded during the countdown.",
    )
    record_rollout = st.checkbox(
        "Record the rollout",
        value=True,
        help="Writes the same episode layout as the teleoperation page, plus a policy block in meta.json.",
    )
    return_to_start = st.checkbox(
        "Return to the start pose when the rollout ends",
        value=True,
        help=(
            "A policy leaves the arm wherever the episode ended. Parking it makes "
            "consecutive trials comparable and the per-step clamp start fresh. The "
            "start pose includes the gripper, so a grasped object is released."
        ),
    )
    if return_to_start:
        st.caption(
            "↩ The arm parks at the start pose above, gripper included: "
            "anything still held is released."
        )

    st.divider()
    st.write(f"Runtime: {_state_label(snapshot.state)}")
    if runtime.process_pid is not None:
        st.caption(f"Dedicated hardware process PID: {runtime.process_pid}")

    if snapshot.state is PolicyState.DISCONNECTED:
        if st.button("🔌 Connect hardware", type="primary", width="stretch"):
            try:
                start_pose = json.loads(start_pose_text)
                missing = [key for key in ACTION_KEYS if key not in start_pose]
                if missing:
                    raise ValueError(f"Start pose is missing {', '.join(missing)}.")
                runtime.connect(
                    PolicyRuntimeConfig(
                        can_port=can_port,
                        speed_rate=speed_rate,
                        gripper_max_mm=gripper_max_mm,
                        max_relative_target=float(max_relative_target),
                        control_hz=float(control_hz),
                        watchdog_timeout_s=float(watchdog_timeout_s),
                        watchdog_startup_timeout_s=float(watchdog_startup_timeout_s),
                        start_pose=start_pose,
                        home_on_connect=home_on_connect,
                        camera=CameraSettings(
                            device=camera_device,
                            width=int(camera_width),
                            height=int(camera_height),
                            fps=camera_fps,
                            fourcc=None if camera_fourcc == FOURCC_OPTIONS[0] else camera_fourcc,
                            preview_hz=preview_hz,
                            preview_enabled=preview_enabled,
                        ),
                        preview_path=DEFAULT_PREVIEW_PATH,
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


try:
    runtime.set_preview_enabled(preview_enabled)
except Exception as exc:
    st.error(f"Could not change preview publication: {exc}")


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------


def preview_panel() -> None:
    if snapshot.state is PolicyState.DISCONNECTED:
        st.info("Connect the hardware to start the local camera preview.")
        return
    if not preview_enabled:
        st.info("Preview publication is disabled. Rollout recording remains available.")
        return
    refresh_ms = int(PREVIEW_REFRESH_S * 1000)
    source = f"/app/static/{DEFAULT_PREVIEW_PATH.name}"
    components.html(
        f"""
        <style>
          html, body {{ margin: 0; background: transparent; overflow: hidden; }}
          #live {{
            display: block; width: 100%; height: 520px; object-fit: contain;
            border-radius: 0.5rem; background: #111;
          }}
        </style>
        <img id="live" alt="Live camera">
        <script>
          const live = document.getElementById("live");
          // Self-scheduling: over a forwarded SSH port a frame can take longer to
          // arrive than the refresh period, and a fixed interval then queues
          // requests faster than they complete until the tab stalls.
          function refresh() {{
            const next = new Image();
            const done = (ok) => {{
              if (ok) live.src = next.src;
              window.setTimeout(refresh, {refresh_ms});
            }};
            next.onload = () => done(true);
            next.onerror = () => done(false);
            next.src = "{source}?t=" + Date.now();
          }}
          refresh();
        </script>
        """,
        height=530,
        scrolling=False,
    )


def _pending_deadline() -> float | None:
    return st.session_state.get("_rollout_deadline_s")


def _schedule_start() -> None:
    st.session_state["_rollout_deadline_s"] = time.perf_counter() + float(start_delay_s)


def _cancel_start() -> None:
    st.session_state.pop("_rollout_deadline_s", None)


def _start_rollout() -> None:
    runtime.start_rollout(
        RolloutConfig(
            output_dir=Path(output_dir),
            task=task,
            duration_s=float(rollout_duration_s),
            record=record_rollout,
            return_to_start=return_to_start,
        )
    )


@st.fragment(run_every=UI_REFRESH_S)
def rollout_panel() -> None:
    current = runtime.get_snapshot()

    if current.returning_to_start:
        st.info("Returning the arm to the start pose…", icon="↩️")
        return

    deadline = _pending_deadline()
    if deadline is not None:
        if current.state is not PolicyState.IDLE:
            _cancel_start()
            st.warning(f"Pending start cancelled because the runtime is {current.state.value}.")
            return
        remaining_s = max(deadline - time.perf_counter(), 0.0)
        if remaining_s <= 0:
            _cancel_start()
            try:
                _start_rollout()
            except Exception as exc:
                st.error(f"Could not start the rollout: {exc}")
            st.rerun(scope="fragment")
            return
        st.markdown(
            (
                '<div style="padding:1.25rem;text-align:center;border-radius:0.6rem;'
                'background:#fff3cd;color:#664d03;font-size:1.1rem;">The arm starts moving in<br>'
                f'<span style="font-size:3rem;font-weight:700;">{math.ceil(remaining_s)}</span><br>seconds'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        if st.button("Cancel pending start", width="stretch"):
            _cancel_start()
            st.rerun(scope="fragment")
        return

    if current.state is PolicyState.RUNNING:
        duration = current.rollout_duration_s or rollout_duration_s
        st.progress(
            min(current.rollout_elapsed_s / duration, 1.0) if duration else 0.0,
            text=(
                f"Running {current.rollout_elapsed_s:.1f}s / {duration:.1f}s · "
                f"{current.rollout_samples} steps recorded · {current.plans} predictions"
            ),
        )
        guidance = (
            f" · guidance {current.policy.guidance_weight:g}"
            if current.policy is not None and current.policy.guidance_weight != 1.0
            else ""
        )
        st.caption(
            f"queue {current.queued_actions} action(s) · "
            f"inference p95 {current.inference_latency.p95_ms:.0f} ms · "
            f"{current.missed_rollout_frames} unrecorded step(s){guidance}"
        )
        if st.button("⏹ STOP ROLLOUT", type="primary", width="stretch"):
            try:
                with st.spinner("Stopping, then parking the arm at the start pose…"):
                    runtime.stop_rollout()
                st.rerun(scope="fragment")
            except Exception as exc:
                st.error(f"Could not stop the rollout: {exc}")
        return

    draft = current.draft
    if draft is not None and not draft.running:
        if draft.recorded:
            st.success(
                f"Rollout {draft.episode_index:03d} finished: {draft.samples} steps, "
                f"{draft.video_frames} video frames."
            )
            save_column, discard_column = st.columns(2)
            if save_column.button("💾 Save", type="primary", width="stretch"):
                try:
                    saved = runtime.save_rollout()
                    st.session_state["_last_saved_rollout"] = str(saved)
                    st.rerun(scope="fragment")
                except Exception as exc:
                    st.error(f"Save failed; the recording was retained: {exc}")
            if discard_column.button("🗑 Discard", width="stretch"):
                runtime.discard_rollout()
                st.rerun(scope="fragment")
        else:
            st.info(f"Rollout finished after {draft.duration_s:.1f}s; it was not recorded.")
            if st.button("Clear", width="stretch"):
                runtime.discard_rollout()
                st.rerun(scope="fragment")
        return

    if st.session_state.get("_last_saved_rollout"):
        st.success(f"Saved {st.session_state['_last_saved_rollout']}")

    if current.state is PolicyState.IDLE:
        if current.policy is None:
            st.info("Load a checkpoint before starting a rollout.")
            return
        st.caption(current.policy.headline())
        if st.button("▶ START ROLLOUT", type="primary", width="stretch"):
            _schedule_start()
            st.rerun(scope="fragment")
        if st.button("↩ Return to start pose", width="stretch"):
            try:
                with st.spinner("Parking the arm at the start pose…"):
                    runtime.return_to_start()
                st.rerun(scope="fragment")
            except Exception as exc:
                st.error(f"Could not park the arm: {exc}")
    elif current.state is PolicyState.DISCONNECTED:
        st.info("Connect the hardware to run the policy.")
    else:
        st.info(f"Runtime is {current.state.value}.")


@st.fragment(run_every=TELEMETRY_REFRESH_S)
def telemetry_panel() -> None:
    current = runtime.get_snapshot()
    if current.last_error:
        st.error(
            f"{current.error_source or 'runtime'}: {current.last_error} "
            "Use Safe disconnect if the robot can still communicate; otherwise use the "
            "physical emergency-stop procedure.",
            icon="🚨",
        )

    activity = (
        "parking"
        if current.returning_to_start
        else ("driving" if current.driving else "holding")
    )
    metrics = st.columns(5)
    metrics[0].metric("State", current.state.value, activity)
    metrics[1].metric(
        "Control",
        f"{current.effective_control_hz:.1f} Hz",
        f"target {current.target_control_hz:.0f}",
    )
    metrics[2].metric(
        "Inference p95",
        f"{current.inference_latency.p95_ms:.0f} ms",
        f"{current.plans} predictions",
    )
    metrics[3].metric("Command age", _format_age(current.last_command_age_s))
    metrics[4].metric(
        "Camera",
        f"{current.camera_capture_hz:.1f} Hz",
        f"age {_format_age(current.camera_frame_age_s)}",
    )

    budget_ms = 1000.0 / current.target_control_hz if current.target_control_hz else 0.0
    if budget_ms and current.inference_latency.count >= 5 and current.inference_latency.p95_ms > budget_ms:
        st.warning(
            f"A prediction takes {current.inference_latency.p95_ms:.0f} ms at p95 but one control "
            f"step is {budget_ms:.0f} ms, so the step it re-plans on runs long. Execute more "
            "actions per prediction, lower the sampler steps, or lower the control rate.",
            icon="⚠️",
        )

    timing_rows = [
        {
            "Stage": label,
            "Mean ms": round(timing.mean_ms, 3),
            "p50 ms": round(timing.p50_ms, 3),
            "p95 ms": round(timing.p95_ms, 3),
            "p99 ms": round(timing.p99_ms, 3),
            "Max ms": round(timing.max_ms, 3),
        }
        for label, timing in (
            ("Control period", current.control_period),
            ("Robot observation", current.observation_latency),
            ("Policy prediction", current.inference_latency),
            ("Robot send", current.robot_send_latency),
        )
    ]
    st.dataframe(timing_rows, hide_index=True, width="stretch")

    joint_rows = [
        {
            "Signal": key,
            "Measured": current.latest_observation.get(key),
            "Commanded": current.latest_action.get(key),
        }
        for key in ACTION_KEYS
    ]
    st.dataframe(joint_rows, hide_index=True, width="stretch")
    st.caption(
        f"minimum rolling effective rate {current.minimum_effective_hz:.1f} Hz · "
        f"control stage {current.control_stage} ({_format_age(current.control_stage_age_s)}) · "
        f"missed control deadlines {current.missed_control_deadlines} · "
        f"queued actions {current.queued_actions} · "
        f"video queue {current.video_queue_depth} · dropped {current.video_dropped_frames}"
    )


left, right = st.columns([3, 2])
with left:
    st.subheader("Live camera")
    preview_panel()
with right:
    st.subheader("Rollout")
    rollout_panel()

st.subheader("Control telemetry")
telemetry_panel()
