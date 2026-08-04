#!/usr/bin/env python3
"""Streamlit page for running a goal-conditioned policy on the Piper.

A behaviour-cloning rollout is one step: look, act. This one is two.

    1. A video model is shown the frame in front of the camera and predicts
       what should happen from it.
    2. One frame of that prediction becomes the goal, and the policy is driven
       closed-loop towards it.

Which is why this page exists at all rather than a checkbox on ``policy_app.py``:
the operator has to see the predicted video before the arm moves. A goal-
conditioned policy will drive confidently towards a wrong goal, and the
generated video is the only place that is visible - by the time the arm is
moving it looks like a policy failure.

The goal frame defaults to the third of four sampled evenly across the
generated video, which is the rule ``src/policy/goal.py`` trained under, and the
slider is there to override it per rollout.

    bash scripts/9_goal_policy_ui.sh

As in ``policy_app.py``: Streamlit never touches hardware or torch. It sends
commands to a dedicated process and renders immutable snapshots.
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
        count_outcomes,
        get_runtime,
    )
    from ui.teleop_runtime import ACTION_KEYS
    from ui.video_goal import (
        DEFAULT_BASE_CHECKPOINT,
        DEFAULT_COSMOS_URL,
        DEFAULT_SUPERRES_CHECKPOINT,
        GoalVideo,
        VideoSourceConfig,
    )
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
        count_outcomes,
        get_runtime,
    )
    from teleop_runtime import ACTION_KEYS
    from video_goal import (
        DEFAULT_BASE_CHECKPOINT,
        DEFAULT_COSMOS_URL,
        DEFAULT_SUPERRES_CHECKPOINT,
        GoalVideo,
        VideoSourceConfig,
    )

try:
    from policy.checkpoints import RunInfo, discover_runs
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from policy.checkpoints import RunInfo, discover_runs

UI_REFRESH_S = 0.2
TELEMETRY_REFRESH_S = 1.0
PREVIEW_REFRESH_S = 0.1
DEFAULT_MODEL_ROOTS = f"models/simple {DEFAULT_MODEL_ROOT}"
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
    return [Path(entry) for entry in model_root.replace(",", " ").split() if entry]


@st.cache_data(ttl=30.0, show_spinner=False)
def _runs(model_root: str, scan_token: int = 0) -> list[RunInfo]:
    """Only the goal-conditioned runs; this page cannot drive anything else."""
    seen: set[Path] = set()
    runs: list[RunInfo] = []
    for root in _split_roots(model_root):
        for run in discover_runs(root):
            if run.directory.resolve() in seen:
                continue
            seen.add(run.directory.resolve())
            if bool(run.config.get("goal_conditioned")):
                runs.append(run)
    return runs


@st.cache_data(ttl=10.0, show_spinner=False)
def _camera_devices() -> list[str]:
    devices: set[str] = set()
    for pattern in KNOWN_CAMERA_GLOBS:
        devices.update(glob.glob(pattern))
    return sorted(devices)


@st.cache_data(ttl=5.0, show_spinner=False)
def _outcome_counts(output_dir: str, save_token: int) -> dict[str, int]:
    return count_outcomes(Path(output_dir))


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


def _goal_video() -> GoalVideo | None:
    return st.session_state.get("_goal_video")


st.set_page_config(page_title="Goal-conditioned policy", page_icon="🎯", layout="wide")

runtime = get_runtime()
snapshot = runtime.get_snapshot()
hardware_locked = snapshot.state is not PolicyState.DISCONNECTED
driving = snapshot.state is PolicyState.RUNNING

st.title("Goal-conditioned policy")
st.caption(
    "Generate a video of what should happen, pick the frame to aim at, then run the policy "
    "towards it."
)
st.warning(
    "The arm is commanded by a neural network aiming at a goal a second network invented. "
    "Look at the generated video before starting: a wrong goal produces confident, wrong "
    "motion. Keep the physical emergency stop within reach, and never run this page and the "
    "teleoperation page against the same arm at the same time.",
    icon="⚠️",
)


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------


def checkpoint_panel() -> PolicySettings | None:
    """Pick a goal-conditioned run; the goal itself comes from the video model."""
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
        st.error(
            f"No goal-conditioned run under {model_root}. Train one with "
            "`python src/policy/train.py --goal-conditioned`, or use the plain policy page."
        )
        return None

    labels = [run.name for run in runs]
    loaded = snapshot.policy
    default_index = next(
        (index for index, run in enumerate(runs) if loaded and run.name == loaded.run_name), 0
    )
    label = st.selectbox("Run", labels, index=default_index, disabled=driving)
    run = runs[labels.index(label)]
    st.caption(run.summary())
    st.session_state["_selected_run_task"] = run.tasks[0] if run.tasks else ""

    tags = [entry.tag for entry in run.checkpoints]
    tag = st.selectbox(
        "Checkpoint",
        tags,
        index=0,
        disabled=driving,
        format_func=lambda value: run.checkpoint(value).label,
    )
    checkpoint = run.checkpoint(tag)
    config = run.config

    selection = config.get("goal_selection", "uniform4")
    frame_index = config.get("goal_frame_index", 2)
    frames = config.get("goal_frames", 4)
    if selection == "uniform4":
        st.caption(
            f"🎯 Trained on frame {frame_index} of {frames} sampled evenly across each episode - "
            "the same frame the video model below generates."
        )
    elif selection == "future_uniform":
        st.caption(
            "🎯 Trained with a goal drawn uniformly after the target action chunk through the "
            "episode's final frame. Choose a generated frame that is visibly beyond the next chunk."
        )
    else:
        st.caption(
            f"🎯 Trained on the episode's last frames (goal-selection `{selection}`). This page "
            "generates a mid-episode goal, so aim the slider at the video's final frame instead."
        )

    goal_dropout = float(config.get("goal_dropout") or 0.0)
    if goal_dropout <= 0.0:
        st.caption("This checkpoint has no null-goal branch: it will refuse to step without a goal.")

    # Both are retuned live in guidance_panel below; loading uses the trained
    # defaults. Which modes are offered is decided by the runner after loading -
    # this is only the heads-up, computed from the same two dropout rates.
    cond_dropout = float(config.get("cond_dropout") or 0.0)
    available = [
        label
        for label, trained in (
            (f"goal-only (goal dropout {goal_dropout:g})", goal_dropout > 0.0),
            (f"full (cond dropout {cond_dropout:g})", cond_dropout > 0.0),
        )
        if trained
    ]
    st.caption(
        f"🎛 Classifier-free guidance available: {', '.join(available)}. Set the mode and "
        "weight after loading."
        if available
        else "Guidance unavailable: trained with --cond-dropout 0 and --goal-dropout 0."
    )

    with st.expander("Run details"):
        st.write(
            {
                "policy": run.policy,
                "objective": run.objective,
                "chunk_size": run.chunk_size,
                "image_size": run.image_size,
                "bundle": run.bundle,
                "fps": run.fps,
                "best_val_loss": run.best_val_loss,
                "n_obs_steps": config.get("n_obs_steps"),
                "goal_selection": selection,
                "goal_frames": frames,
                "goal_frame_index": frame_index,
                "goal_window": config.get("goal_window"),
                "goal_dropout": goal_dropout,
                "cond_dropout": cond_dropout,
                "guidance_weight": config.get("guidance_weight"),
                "guidance_mode": config.get("guidance_mode", "full"),
                "checkpoint": str(checkpoint.path),
            }
        )

    chunk_size = run.chunk_size or 8
    action_steps = st.slider(
        "Actions executed per prediction",
        min_value=1,
        max_value=chunk_size,
        value=max(1, min(int(config.get("n_action_steps") or min(4, chunk_size)), chunk_size)),
        disabled=driving,
    )
    device = st.selectbox("Device", ("cuda", "cpu"), index=0, disabled=driving)
    use_ema = st.checkbox("Use EMA weights", value=True, disabled=driving)
    action_mode = st.radio(
        "Policy action mode",
        ("joint", "eef_ik"),
        format_func=lambda value: (
            "Joint targets"
            if value == "joint"
            else "EEF pose targets → numerical IK"
        ),
        disabled=driving,
        help=(
            "EEF IK requires a checkpoint trained with action columns "
            "eef.x/eef.y/eef.z/eef.rx/eef.ry/eef.rz/gripper.pos. "
            "Existing joint-target checkpoints must use Joint targets."
        ),
    )

    try:
        return PolicySettings(
            checkpoint=checkpoint.path,
            action_steps=action_steps,
            device=device,
            use_ema=use_ema,
            action_mode=action_mode,
        )
    except ValueError as exc:
        st.error(str(exc))
        return None


def video_source_panel() -> VideoSourceConfig | None:
    """Which video model predicts the goal, and how it is reached."""
    kind = st.radio(
        "Video model",
        ("cosmos", "diffusion"),
        format_func=lambda value: (
            "Cosmos3-Nano (HTTP server)" if value == "cosmos" else "src/diffusion cascade (local)"
        ),
        disabled=driving,
        help=(
            "Cosmos3 runs in its own environment behind src/cosmos/serve.py; the cascade is "
            "small enough to load in the hardware process next to the policy."
        ),
    )
    fields: dict = {"kind": kind}
    if kind == "cosmos":
        fields["url"] = st.text_input("Server URL", DEFAULT_COSMOS_URL, disabled=driving)
        prompt = st.text_input(
            "Prompt",
            "",
            disabled=driving,
            help="Blank uses the server's own default caption.",
        ).strip()
        fields["prompt"] = prompt or None
        fields["steps"] = int(
            st.number_input("Denoising steps", min_value=1, max_value=100, value=35, step=1,
                            disabled=driving)
        )
        st.caption("Start src/cosmos/serve.py first; a 93-frame clip takes a few seconds.")
    else:
        fields["base_checkpoint"] = Path(
            st.text_input("Base checkpoint", str(DEFAULT_BASE_CHECKPOINT), disabled=driving)
        )
        fields["superres_checkpoint"] = Path(
            st.text_input(
                "Super-resolution checkpoint", str(DEFAULT_SUPERRES_CHECKPOINT), disabled=driving
            )
        )
        fields["inference_steps"] = int(
            st.number_input("DDIM steps per stage", min_value=1, max_value=200, value=50, step=1,
                            disabled=driving)
        )
        st.caption("Generates four frames: the current one plus three futures.")

    if st.checkbox("Fix the sampling seed", value=False, disabled=driving):
        fields["seed"] = int(
            st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=0, step=1,
                            disabled=driving)
        )

    try:
        return VideoSourceConfig(**fields)
    except ValueError as exc:
        st.error(str(exc))
        return None


GUIDANCE_MODE_LABELS = {
    "goal": "goal only — amplify the goal, keep the frame",
    "full": "full — amplify everything the policy sees",
}


def guidance_panel(loaded: PolicyInfo | None) -> None:
    """Retune classifier-free guidance on the policy already in the hardware process.

    The mode is the interesting control on this page. Under ``full`` the guided
    branch is "no observation at all", so the weight sharpens the frame, the
    measured state and the goal together. Under ``goal`` only the goal is
    dropped: the branch is "this same scene, no goal in particular", and the
    weight scales how hard the rollout is pushed towards the picture chosen
    below while the policy keeps full sight of what is actually in front of it.
    That is the knob this page wants - and it needs only ``--goal-dropout``,
    which most goal-conditioned runs already have.

    Either way the weight only sharpens the goal that was set: a wrong goal
    driven at 3.0 is a worse rollout, not a better one. Judge the video first.

    Changing either setting costs a queue flush; reloading the checkpoint would
    rebuild the CUDA context and repeat the warm-up.
    """
    if loaded is None or not loaded.guidance_modes:
        return

    st.header("Guidance")
    modes = list(loaded.guidance_modes)
    mode = st.radio(
        "Guided branch drops",
        modes,
        index=modes.index(loaded.guidance_mode) if loaded.guidance_mode in modes else 0,
        format_func=lambda value: GUIDANCE_MODE_LABELS.get(value, value),
        disabled=driving,
        help=(
            "goal: the unconditional branch keeps this frame and this joint state and only "
            "blanks the goal, so the weight scales the goal's influence alone. full: it blanks "
            "the whole conditioning vector, so the weight scales everything at once."
        ),
    )
    if mode != loaded.guidance_mode:
        if st.button(f"Apply {mode} guidance", width="stretch", disabled=driving):
            try:
                runtime.set_guidance_mode(mode)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not change the guidance mode: {exc}")
        st.caption(f"Still running in **{loaded.guidance_mode}**; press to switch.")
        return
    if not loaded.supports_guidance:
        return

    weight = float(
        st.slider(
            "Classifier-free guidance weight",
            min_value=1.0,
            max_value=5.0,
            value=float(loaded.guidance_weight),
            step=0.1,
            disabled=driving,
            help=(
                "1.0 is plain conditional sampling. Higher narrows the range of behaviours the "
                "policy will commit to. Robotics weights are modest; 1.5-3.0 is the usual range."
            ),
        )
    )

    if abs(weight - loaded.guidance_weight) < 1e-9:
        st.caption(
            f"Active: {loaded.guidance_weight:g} ({loaded.guidance_mode})"
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
    if weight > 1.0 and loaded.guidance_mode == "goal" and not loaded.has_goal:
        # Both branches would be the null goal, so the weight would do nothing;
        # the runner refuses at the first step rather than drifting silently.
        st.warning(
            "Goal-only guidance needs a goal. Aim at a frame below, or the first step of the "
            "rollout will fail.",
            icon="🎯",
        )


with st.sidebar:
    st.header("Checkpoint")
    policy_settings = checkpoint_panel()

    loaded_policy = snapshot.policy
    if loaded_policy is not None:
        st.success(loaded_policy.headline())
    else:
        st.info("No checkpoint is loaded in the hardware process yet.")

    load_label = "📦 Load checkpoint" if loaded_policy is None else "📦 Load / replace checkpoint"
    if st.button(load_label, type="primary", width="stretch",
                 disabled=policy_settings is None or driving):
        assert policy_settings is not None
        try:
            with st.spinner("Loading weights and running one warm-up inference…"):
                runtime.load_policy(policy_settings)
            # A new policy has no goal, so the video on screen is not its goal.
            # The generated video stays on disk; only the aim is dropped.
            st.session_state.pop("_goal_video", None)
            st.session_state.pop("_goal_video_dir", None)
            st.session_state.pop("_aimed_goal", None)
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load the checkpoint: {exc}")

    guidance_panel(loaded_policy)

    st.header("Goal video")
    video_source = video_source_panel()
    keep_goal_videos = st.checkbox(
        "Keep every generated video",
        value=True,
        help="Writes each clip to `goal_videos/` under the rollout output directory, and the "
             "one actually aimed at into the episode that ran against it. Generation is not "
             "reproducible unless the seed is fixed, so an unsaved video is gone.",
    )

    st.header("Robot")
    can_port = st.text_input("Piper CAN interface", "piper_left", disabled=hardware_locked)
    speed_rate = st.slider("Piper speed rate (%)", 1, 100, 30, disabled=hardware_locked)
    gripper_max_mm = st.selectbox(
        "Piper jaw maximum (mm)", options=(70.0, 100.0), index=1, disabled=hardware_locked
    )
    max_relative_target = st.number_input(
        "Maximum joint change per step (deg/mm)",
        min_value=0.1, max_value=45.0, value=2.0, step=0.1, disabled=hardware_locked,
        help="The plugin clamps every command against the previous one. This is the main "
             "protection against a bad prediction; raise it only deliberately.",
    )
    control_hz = st.number_input(
        "Control rate (Hz)", min_value=1.0, max_value=100.0,
        value=float(loaded_policy.fps if loaded_policy else 20.0), step=1.0,
        disabled=hardware_locked, help="Match the rate the demonstrations were recorded at.",
    )
    watchdog_timeout_s = st.number_input(
        "Stale-control watchdog (s)", min_value=0.1, max_value=10.0, value=2.0, step=0.1,
        disabled=hardware_locked,
    )
    watchdog_startup_timeout_s = st.number_input(
        "Startup watchdog grace (s)", min_value=0.5, max_value=60.0, value=10.0, step=0.5,
        disabled=hardware_locked,
    )
    home_on_connect = st.checkbox(
        "Home to the start pose on connect", value=True, disabled=hardware_locked,
        help="The video model was trained on episodes that begin here, so the frame it is "
             "conditioned on has to begin here too.",
    )
    start_pose_text = st.text_area(
        "Start / rest pose (JSON)", json.dumps(DEFAULT_START_POSE), height=120,
        disabled=hardware_locked,
    )

    st.header("Camera")
    devices = _camera_devices()
    default_device = "/dev/video3"
    camera_options = devices + ["Custom path…"]
    default_index = devices.index(default_device) if default_device in devices else len(devices)
    selected_device = st.selectbox(
        "Device", camera_options, index=default_index, disabled=hardware_locked
    )
    camera_device = (
        st.text_input("Camera path", default_device, disabled=hardware_locked)
        if selected_device == "Custom path…"
        else selected_device
    )
    width_column, height_column = st.columns(2)
    camera_width = width_column.number_input(
        "Width", min_value=1, value=640, step=1, disabled=hardware_locked
    )
    camera_height = height_column.number_input(
        "Height", min_value=1, value=480, step=1, disabled=hardware_locked
    )
    camera_fps = st.number_input(
        "Camera capture FPS", min_value=1.0, max_value=240.0, value=30.0, step=1.0,
        disabled=hardware_locked,
    )
    camera_fourcc = st.selectbox("Capture FourCC", FOURCC_OPTIONS, disabled=hardware_locked)
    preview_hz = st.number_input(
        "Preview publish rate (Hz)", min_value=1.0, max_value=30.0, value=10.0, step=1.0,
        disabled=hardware_locked,
    )
    preview_enabled = st.checkbox("Publish browser preview", value=True)

    st.header("Rollout")
    output_dir = st.text_input("Output directory", str(DEFAULT_OUTPUT_DIR / "goal"))
    task = st.text_input(
        "Task description", st.session_state.get("_selected_run_task") or "pick up can"
    )
    rollout_duration_s = st.number_input(
        "Rollout duration (s)", min_value=1.0, max_value=600.0, value=20.0, step=1.0
    )
    start_delay_s = st.number_input(
        "Start countdown (s)", min_value=0, max_value=30, value=5, step=1,
        help="The arm is not commanded during the countdown.",
    )
    record_rollout = st.checkbox("Record the rollout", value=True)
    return_to_start = st.checkbox(
        "Return to the start pose when the rollout ends", value=True,
        help="Every rollout has to start from the pose the video model's conditioning frame "
             "was captured in, so parking is what makes consecutive trials comparable.",
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
            display: block; width: 100%; height: 360px; object-fit: contain;
            border-radius: 0.5rem; background: #111;
          }}
        </style>
        <img id="live" alt="Live camera">
        <script>
          const live = document.getElementById("live");
          // Self-scheduling: over a forwarded SSH port a frame can take longer
          // to arrive than the refresh period, and a fixed interval then queues
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
        height=370,
        scrolling=False,
    )


def _archive_goal_video(video: GoalVideo) -> Path:
    """Write a freshly generated video to its own directory under the output dir.

    Every generated video is kept, not only the one that ends up driving a
    rollout: rejecting a video is a judgement about the video model, and the
    rejected ones are the examples that judgement was made on.
    """
    base = Path(output_dir) / "goal_videos"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    directory = base / stamp
    attempt = 1
    while directory.exists():  # two generations inside one second
        directory = base / f"{stamp}_{attempt}"
        attempt += 1
    return video.save(directory, extra={"task": task, "generated_for": "goal_policy_app"})


def _save_goal_with_episode(episode_dir: Path) -> str | None:
    """Copy the goal the rollout ran against into the episode it produced.

    What is written is the video the operator pressed *Aim* on, not whatever is
    on screen now - a video generated after aiming, and not aimed at, never
    reached the policy. Returns a line to show the operator, or None when there
    was no goal to record.
    """
    aimed = st.session_state.get("_aimed_goal")
    if aimed is None:
        return None
    try:
        written = aimed["video"].save(
            episode_dir / "goal_video",
            chosen_index=aimed["index"],
            extra={
                "episode": episode_dir.name,
                "generated_archive": aimed["archive"],
                "note": "the goal this episode was driven towards; goal.png is the frame itself",
            },
        )
    except Exception as exc:
        # The episode is already committed on disk; losing the goal copy is bad
        # but it is not a reason to make the save look like it failed.
        return f"⚠️ The episode was saved but its goal video was not: {exc}"
    return f"🎯 Goal video written to `{written}`"


def goal_panel(policy: PolicyInfo | None) -> None:
    """Generate a video from the current frame, then choose the goal out of it.

    This is the step that makes the page: the generated video is the policy's
    entire idea of where the episode is going, and it is far cheaper to reject a
    bad one here than to watch the arm act on it.
    """
    st.subheader("Predicted video")
    if policy is None:
        st.info("Load a goal-conditioned checkpoint to generate a goal.")
        return
    if snapshot.state is PolicyState.DISCONNECTED:
        st.info("Connect the hardware: the video model conditions on the live camera frame.")
        return

    generate_column, clear_column = st.columns([3, 1])
    if generate_column.button(
        "🎬 Generate from the current frame",
        type="primary",
        width="stretch",
        disabled=video_source is None or driving,
    ):
        assert video_source is not None
        try:
            with st.spinner(f"Asking {video_source.describe()}…"):
                video = runtime.generate_goal_video(video_source)
            st.session_state["_goal_video"] = video
            st.session_state["_goal_frame_index"] = video.goal_index
            # Written before the rerun that displays it, so a video that is
            # looked at once and regenerated over is still on disk. Reported
            # through session state because the rerun clears anything drawn here.
            st.session_state["_goal_video_dir"] = None
            st.session_state["_goal_video_error"] = None
            if keep_goal_videos:
                try:
                    st.session_state["_goal_video_dir"] = str(_archive_goal_video(video))
                except Exception as exc:
                    st.session_state["_goal_video_error"] = str(exc)
            st.rerun()
        except Exception as exc:
            st.error("Generation failed. Complete traceback:")
            st.code(str(exc), language=None, wrap_lines=True)
    if clear_column.button("🗑 Clear", width="stretch", disabled=driving):
        st.session_state.pop("_goal_video", None)
        st.session_state.pop("_goal_video_dir", None)
        st.session_state.pop("_aimed_goal", None)
        try:
            runtime.set_goal_frame(None)
        except Exception as exc:
            st.error(f"Could not clear the goal: {exc}")
        st.rerun()

    video = _goal_video()
    if video is None:
        st.info(
            "No video yet. The arm should be at the start pose and the scene set up as it will "
            "be for the rollout: the prediction is a continuation of whatever the camera sees now."
        )
        return

    st.caption(
        f"{video.frame_count} frames from {video.source} in {video.seconds:.1f}s · "
        f"default goal frame {video.goal_index}"
    )
    if st.session_state.get("_goal_video_error"):
        st.warning(
            f"This video could not be written to disk: {st.session_state['_goal_video_error']}",
            icon="💾",
        )
    elif st.session_state.get("_goal_video_dir"):
        st.caption(f"💾 Kept in `{st.session_state['_goal_video_dir']}`")
    elif not keep_goal_videos:
        st.caption("💾 Not being kept: enable *Keep every generated video* in the sidebar.")
    # A short clip is a strip of stills - all four states at once is easier to
    # judge than four seconds of playback. A long one is a video.
    if video.frame_count <= 8:
        columns = st.columns(video.frame_count)
        for index, column in enumerate(columns):
            column.image(video.frames_png[index], caption=f"frame {index}", width="stretch")
    else:
        st.image(video.frames_png[video.goal_index], caption="the default goal frame", width=360)

    chosen = st.slider(
        "Goal frame",
        min_value=0,
        max_value=video.frame_count - 1,
        value=int(st.session_state.get("_goal_frame_index", video.goal_index)),
        disabled=driving,
        help=(
            "The policy aims at this picture for the whole rollout. Training used the third of "
            "four frames sampled evenly, which is what the default points at."
        ),
    )
    st.session_state["_goal_frame_index"] = chosen
    st.image(video.frames_png[chosen], caption=f"goal: frame {chosen}", width=280)

    if st.button(f"🎯 Aim at frame {chosen}", type="primary", width="stretch", disabled=driving):
        try:
            runtime.set_goal_frame(video.frames_png[chosen])
            # Remembered here rather than read back at save time: this is the
            # picture the policy is actually driving towards, and the page may
            # be showing a newer video by the time the rollout finishes.
            st.session_state["_aimed_goal"] = {
                "video": video,
                "index": int(chosen),
                "archive": st.session_state.get("_goal_video_dir"),
            }
            st.rerun()
        except Exception as exc:
            st.error(f"Could not set the goal: {exc}")

    if policy.has_goal:
        st.success("A goal is set. The policy will aim at it until it is replaced or cleared.")
    else:
        st.warning("No goal is set on the policy yet; press the button above.", icon="🎯")


def _pending_deadline() -> float | None:
    return st.session_state.get("_rollout_deadline_s")


def _cancel_start() -> None:
    st.session_state.pop("_rollout_deadline_s", None)


def outcome_panel(outcome: str | None) -> None:
    """Judge the finished episode: did the policy reach the goal or not?"""
    st.caption("Did this episode do the task?")
    success_column, failure_column = st.columns(2)
    if success_column.button(
        "✅ Success", width="stretch",
        type="primary" if outcome == "success" else "secondary",
        disabled=outcome == "success",
    ):
        runtime.set_rollout_outcome("success")
        st.rerun(scope="fragment")
    if failure_column.button(
        "❌ Failed", width="stretch",
        type="primary" if outcome == "failure" else "secondary",
        disabled=outcome == "failure",
    ):
        runtime.set_rollout_outcome("failure")
        st.rerun(scope="fragment")
    if outcome is None:
        st.caption("⚖️ Unjudged: Save is disabled until this episode is marked.")
        return
    st.caption(
        f"Marked **{'success' if outcome == 'success' else 'failed'}**; it is written to the "
        "episode's meta.json as `outcome`."
    )


def outcome_tally(directory: str) -> None:
    counts = _outcome_counts(directory, st.session_state.get("_saved_rollouts", 0))
    judged = counts["success"] + counts["failure"]
    if not judged and not counts["unjudged"]:
        return
    rate = f"{100.0 * counts['success'] / judged:.0f}%" if judged else "—"
    unjudged = f" · {counts['unjudged']} unjudged" if counts["unjudged"] else ""
    st.caption(
        f"📊 {rate} success over {judged} judged episode(s) in `{directory}` "
        f"({counts['success']} ✅ / {counts['failure']} ❌{unjudged})"
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
                runtime.start_rollout(
                    RolloutConfig(
                        output_dir=Path(output_dir),
                        task=task,
                        duration_s=float(rollout_duration_s),
                        record=record_rollout,
                        return_to_start=return_to_start,
                    )
                )
            except Exception as exc:
                st.error(f"Could not start the rollout: {exc}")
            st.rerun(scope="fragment")
            return
        st.markdown(
            '<div style="padding:1.25rem;text-align:center;border-radius:0.6rem;'
            'background:#fff3cd;color:#664d03;font-size:1.1rem;">The arm starts moving in<br>'
            f'<span style="font-size:3rem;font-weight:700;">{math.ceil(remaining_s)}</span><br>'
            "seconds</div>",
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
            f" · guidance {current.policy.guidance_weight:g} ({current.policy.guidance_mode})"
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
            outcome_panel(draft.outcome)
            save_column, discard_column = st.columns(2)
            if save_column.button(
                "💾 Save", type="primary", width="stretch",
                disabled=draft.outcome is None,
                help=None if draft.outcome else "Judge the episode first.",
            ):
                try:
                    saved = runtime.save_rollout()
                    st.session_state["_last_saved_rollout"] = str(saved)
                    st.session_state["_last_saved_goal"] = _save_goal_with_episode(saved)
                    st.session_state["_saved_rollouts"] = (
                        st.session_state.get("_saved_rollouts", 0) + 1
                    )
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
        if st.session_state.get("_last_saved_goal"):
            st.caption(st.session_state["_last_saved_goal"])
    outcome_tally(output_dir)

    if current.state is PolicyState.IDLE:
        if current.policy is None:
            st.info("Load a checkpoint before starting a rollout.")
            return
        st.caption(current.policy.headline())
        if not current.policy.has_goal:
            # A goal-dropout checkpoint would run against its null embedding
            # here, which is a different policy from the one this page is for.
            st.warning(
                "No goal is set. Generate a video and aim at one of its frames before starting.",
                icon="🎯",
            )
            return
        if st.button("▶ START ROLLOUT", type="primary", width="stretch"):
            st.session_state["_rollout_deadline_s"] = time.perf_counter() + float(start_delay_s)
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
        "parking" if current.returning_to_start else ("driving" if current.driving else "holding")
    )
    metrics = st.columns(5)
    metrics[0].metric("State", current.state.value, activity)
    metrics[1].metric(
        "Control", f"{current.effective_control_hz:.1f} Hz", f"target {current.target_control_hz:.0f}"
    )
    metrics[2].metric(
        "Inference p95", f"{current.inference_latency.p95_ms:.0f} ms", f"{current.plans} predictions"
    )
    metrics[3].metric("Command age", _format_age(current.last_command_age_s))
    metrics[4].metric(
        "Goal", "set" if current.policy and current.policy.has_goal else "none",
        current.policy.run_name if current.policy else "no policy",
    )

    budget_ms = 1000.0 / current.target_control_hz if current.target_control_hz else 0.0
    if budget_ms and current.inference_latency.count >= 5 and current.inference_latency.p95_ms > budget_ms:
        st.warning(
            f"A prediction takes {current.inference_latency.p95_ms:.0f} ms at p95 but one control "
            f"step is {budget_ms:.0f} ms. Execute more actions per prediction or lower the "
            "control rate.",
            icon="⚠️",
        )

    st.dataframe(
        [
            {
                "Stage": label,
                "Mean ms": round(timing.mean_ms, 3),
                "p50 ms": round(timing.p50_ms, 3),
                "p95 ms": round(timing.p95_ms, 3),
                "Max ms": round(timing.max_ms, 3),
            }
            for label, timing in (
                ("Control period", current.control_period),
                ("Robot observation", current.observation_latency),
                ("Policy prediction", current.inference_latency),
                ("Robot send", current.robot_send_latency),
            )
        ],
        hide_index=True,
        width="stretch",
    )
    st.dataframe(
        [
            {
                "Signal": key,
                "Measured": current.latest_observation.get(key),
                "Commanded": current.latest_action.get(key),
            }
            for key in ACTION_KEYS
        ],
        hide_index=True,
        width="stretch",
    )


left, right = st.columns([3, 2])
with left:
    st.subheader("Live camera")
    preview_panel()
    goal_panel(snapshot.policy)
with right:
    st.subheader("Rollout")
    rollout_panel()

st.subheader("Control telemetry")
telemetry_panel()
