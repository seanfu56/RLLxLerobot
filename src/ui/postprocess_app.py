#!/usr/bin/env python3
"""Streamlit UI for non-destructive temporal cropping of teleop episodes."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
from pathlib import Path

import streamlit as st

try:
    from ui.postprocess import (
        create_preview_clip,
        crop_episode,
        default_crop_name,
        discover_episodes,
        load_episode,
    )
except ModuleNotFoundError as exc:
    if exc.name != "ui":
        raise
    from postprocess import (
        create_preview_clip,
        crop_episode,
        default_crop_name,
        discover_episodes,
        load_episode,
    )


DEFAULT_SOURCE_ROOT = Path("outputs/teleop")
DEFAULT_OUTPUT_ROOT = Path("outputs/teleop_processed")
STATIC_ROOT = Path(__file__).resolve().parent / "static"
TIMELINE_STEP_S = 0.001


def _format_timecode(seconds: float) -> str:
    milliseconds = int(round(max(seconds, 0.0) * 1000))
    minutes, remainder = divmod(milliseconds, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def _timeline_changed() -> None:
    start_s, end_s = st.session_state["_editor_timeline"]
    st.session_state["_editor_in"] = start_s
    st.session_state["_editor_out"] = end_s


def _bounds_changed(duration_s: float, frame_step_s: float) -> None:
    start_s = min(max(float(st.session_state["_editor_in"]), 0.0), duration_s)
    end_s = min(max(float(st.session_state["_editor_out"]), 0.0), duration_s)
    if end_s <= start_s:
        end_s = min(start_s + frame_step_s, duration_s)
        if end_s <= start_s:
            start_s = max(end_s - frame_step_s, 0.0)
    st.session_state["_editor_in"] = start_s
    st.session_state["_editor_out"] = end_s
    st.session_state["_editor_timeline"] = (start_s, end_s)


def _nudge_boundary(
    boundary: str,
    delta_s: float,
    duration_s: float,
    frame_step_s: float,
) -> None:
    start_s = float(st.session_state["_editor_in"])
    end_s = float(st.session_state["_editor_out"])
    if boundary == "in":
        start_s = min(max(start_s + delta_s, 0.0), end_s - frame_step_s)
    else:
        end_s = max(
            min(end_s + delta_s, duration_s),
            start_s + frame_step_s,
        )
    st.session_state["_editor_in"] = start_s
    st.session_state["_editor_out"] = end_s
    st.session_state["_editor_timeline"] = (start_s, end_s)


def _publish_source_video(video_path: Path) -> tuple[str, str]:
    """Expose a stable source file to the browser-side scrubber."""
    source_path = video_path.resolve()
    source_stat = source_path.stat()
    cache_key = hashlib.sha256(
        f"{source_path}:{source_stat.st_mtime_ns}:{source_stat.st_size}".encode()
    ).hexdigest()[:20]
    extension = source_path.suffix.lower()
    public_path = STATIC_ROOT / f"source-{cache_key}{extension}"
    if not public_path.is_file():
        STATIC_ROOT.mkdir(parents=True, exist_ok=True)
        pending_path = STATIC_ROOT / f".source-{cache_key}.pending{extension}"
        pending_path.unlink(missing_ok=True)
        try:
            os.link(source_path, pending_path)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(source_path, pending_path)
        pending_path.replace(public_path)
    return f"/app/static/{public_path.name}", cache_key


def _source_scrubber(video_url: str, cache_key: str, duration_s: float) -> None:
    root_id = f"source-editor-{cache_key}"
    video_id = f"source-video-{cache_key}"
    scrubber_id = f"source-scrubber-{cache_key}"
    time_id = f"source-time-{cache_key}"
    st.html(
        f"""
        <style>
          #{root_id} {{
            width: 100%;
          }}
          #{video_id} {{
            display: block;
            width: 100%;
            height: 300px;
            object-fit: contain;
            border-radius: 0.5rem;
            background: #111;
          }}
          #{root_id} .scrub-row {{
            display: grid;
            grid-template-columns: minmax(0, 1fr) 7rem;
            gap: 0.75rem;
            align-items: center;
            margin-top: 0.55rem;
          }}
          #{scrubber_id} {{
            width: 100%;
            margin: 0;
          }}
          #{time_id} {{
            font-variant-numeric: tabular-nums;
            text-align: right;
          }}
        </style>
        <div id="{root_id}">
          <video id="{video_id}" controls preload="auto" src="{video_url}"></video>
          <div class="scrub-row">
            <input
              id="{scrubber_id}"
              type="range"
              min="0"
              max="{duration_s:.6f}"
              step="0.001"
              value="0"
              aria-label="Source video playhead"
            >
            <output id="{time_id}">0.000 s</output>
          </div>
        </div>
        <script>
          const video = document.getElementById("{video_id}");
          const scrubber = document.getElementById("{scrubber_id}");
          const time = document.getElementById("{time_id}");
          const storageKey = "teleop-source-playhead-{cache_key}";
          let dragging = false;

          function showTime(value) {{
            const numeric = Number(value);
            scrubber.value = numeric;
            time.textContent = numeric.toFixed(3) + " s";
            window.sessionStorage.setItem(storageKey, numeric.toFixed(6));
          }}

          video.addEventListener("loadedmetadata", () => {{
            const limit = Math.min(
              Number("{duration_s:.6f}"),
              Number.isFinite(video.duration) ? video.duration : Infinity
            );
            scrubber.max = limit.toFixed(6);
            const saved = Number(window.sessionStorage.getItem(storageKey) || 0);
            const restored = Math.min(Math.max(saved, 0), limit);
            video.currentTime = restored;
            showTime(restored);
          }});

          scrubber.addEventListener("pointerdown", () => {{ dragging = true; }});
          scrubber.addEventListener("input", () => {{
            video.pause();
            video.currentTime = Number(scrubber.value);
            showTime(scrubber.value);
          }});
          scrubber.addEventListener("pointerup", () => {{ dragging = false; }});
          video.addEventListener("timeupdate", () => {{
            if (!dragging) showTime(video.currentTime);
          }});
          video.addEventListener("seeked", () => showTime(video.currentTime));
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def _preview_matches(
    preview: dict[str, object] | None,
    episode_path: Path,
    start_s: float,
    end_s: float,
) -> bool:
    if preview is None:
        return False
    return (
        preview.get("episode") == str(episode_path.resolve())
        and abs(float(preview.get("start_s", -1.0)) - start_s) < 1e-6
        and abs(float(preview.get("end_s", -1.0)) - end_s) < 1e-6
        and Path(str(preview.get("video_path", ""))).is_file()
    )


st.set_page_config(
    page_title="Teleoperation data post-processing",
    page_icon="✂️",
    layout="wide",
)

st.title("Teleoperation data post-processing")
st.caption(
    "Create non-destructive temporal crops while keeping actions, frame timestamps, "
    "video frames, and metadata consistent."
)

with st.sidebar:
    st.header("Episode source")
    source_root_text = st.text_input("Collected data directory", str(DEFAULT_SOURCE_ROOT))
    output_root_text = st.text_input("Processed output directory", str(DEFAULT_OUTPUT_ROOT))
    source_root = Path(source_root_text).expanduser()
    output_root = Path(output_root_text).expanduser()
    episodes = discover_episodes(source_root)
    if not episodes:
        st.warning("No complete episodes were found in this directory.")
        st.stop()
    selected_path = st.selectbox(
        "Episode",
        episodes,
        format_func=lambda path: path.name,
    )

try:
    episode = load_episode(selected_path)
except Exception as exc:
    st.error(f"Could not load episode: {exc}")
    st.stop()

metric_columns = st.columns(4)
metric_columns[0].metric("Duration", f"{episode.duration_s:.2f} s")
metric_columns[1].metric("Actions", len(episode.actions))
metric_columns[2].metric("Frames", len(episode.frames))
metric_columns[3].metric("Alignment", "1:1" if episode.aligned else "Legacy")

if episode.aligned:
    st.success(
        "This episode is one-to-one aligned. Every selected action and video frame "
        "will retain the same new index.",
        icon="✅",
    )
else:
    st.warning(
        "This is a legacy unaligned episode. Actions and frames will both be cropped "
        "by timestamp, but their counts can remain different.",
        icon="⚠️",
    )

video_fps = float(
    episode.meta.get("video_fps_target")
    or episode.meta.get("action_sample_hz_target")
    or 20.0
)
frame_step_s = 1.0 / video_fps
editor_id = str(episode.path.resolve())
if st.session_state.get("_editor_episode") != editor_id:
    st.session_state["_editor_episode"] = editor_id
    st.session_state["_editor_in"] = 0.0
    st.session_state["_editor_out"] = float(episode.duration_s)
    st.session_state["_editor_timeline"] = (
        0.0,
        float(episode.duration_s),
    )
    st.session_state.pop("_editor_preview", None)

st.subheader("Source video")
_, source_monitor_column, _ = st.columns([1, 2, 1])
with source_monitor_column:
    try:
        browser_video_path = episode.video_path
        if browser_video_path.suffix.lower() != ".mp4":
            with st.spinner("Preparing browser-compatible source preview…"):
                browser_video_path = create_preview_clip(
                    episode,
                    0.0,
                    float(episode.duration_s),
                ).video_path
        source_video_url, source_cache_key = _publish_source_video(browser_video_path)
        _source_scrubber(
            source_video_url,
            source_cache_key,
            float(episode.duration_s),
        )
    except Exception as exc:
        st.error(f"Source video preview failed: {exc}")
st.caption(
    "The source scrubber runs entirely in the browser, so seeking does not rerun "
    "or clear the Streamlit page. Its floating-point readout has 0.001 s resolution."
)

st.divider()
st.subheader("Cut range")
st.slider(
    "Drag the handles to set the In and Out points",
    min_value=0.0,
    max_value=float(episode.duration_s),
    step=TIMELINE_STEP_S,
    format="%.3f s",
    key="_editor_timeline",
    on_change=_timeline_changed,
)
start_s = float(st.session_state["_editor_in"])
end_s = float(st.session_state["_editor_out"])

in_column, out_column, duration_column = st.columns(3)
in_column.number_input(
    "In point (seconds)",
    min_value=0.0,
    max_value=float(episode.duration_s),
    step=TIMELINE_STEP_S,
    format="%.3f",
    key="_editor_in",
    on_change=_bounds_changed,
    args=(float(episode.duration_s), frame_step_s),
)
out_column.number_input(
    "Out point (seconds)",
    min_value=0.0,
    max_value=float(episode.duration_s),
    step=TIMELINE_STEP_S,
    format="%.3f",
    key="_editor_out",
    on_change=_bounds_changed,
    args=(float(episode.duration_s), frame_step_s),
)
duration_column.metric(
    "Clip duration",
    _format_timecode(max(end_s - start_s, 0.0)),
    f"{max(int(round((end_s - start_s) * video_fps)), 0)} frames at {video_fps:g} Hz",
)

nudge_columns = st.columns(4)
nudge_columns[0].button(
    "◀ In −1 frame",
    on_click=_nudge_boundary,
    args=("in", -frame_step_s, float(episode.duration_s), frame_step_s),
    width="stretch",
)
nudge_columns[1].button(
    "In +1 frame ▶",
    on_click=_nudge_boundary,
    args=("in", frame_step_s, float(episode.duration_s), frame_step_s),
    width="stretch",
)
nudge_columns[2].button(
    "◀ Out −1 frame",
    on_click=_nudge_boundary,
    args=("out", -frame_step_s, float(episode.duration_s), frame_step_s),
    width="stretch",
)
nudge_columns[3].button(
    "Out +1 frame ▶",
    on_click=_nudge_boundary,
    args=("out", frame_step_s, float(episode.duration_s), frame_step_s),
    width="stretch",
)

# Button callbacks update session state before this rerun. Read the final values.
start_s = float(st.session_state["_editor_in"])
end_s = float(st.session_state["_editor_out"])

selected_actions = sum(
    start_s <= float(row["sample_timestamp_s"]) <= end_s
    for row in episode.actions
)
if episode.aligned:
    selected_frames = selected_actions
else:
    selected_frames = sum(
        start_s <= float(row["timestamp_s"]) <= end_s for row in episode.frames
    )

selection_columns = st.columns(4)
selection_columns[0].metric("In", _format_timecode(start_s))
selection_columns[1].metric("Out", _format_timecode(end_s))
selection_columns[2].metric("Selected actions", selected_actions)
selection_columns[3].metric("Selected frames", selected_frames)

preview = st.session_state.get("_editor_preview")
preview_matches = _preview_matches(
    preview if isinstance(preview, dict) else None,
    episode.path,
    start_s,
    end_s,
)

st.divider()
st.subheader("Cut preview")
st.caption(
    "Render the selected In/Out range, then review it here before saving."
)
preview_column, reset_column = st.columns([3, 1])
if preview_column.button(
    "▶ Preview selected cut",
    type="primary",
    width="stretch",
    disabled=selected_actions == 0 or selected_frames == 0 or end_s <= start_s,
):
    try:
        with st.spinner("Rendering exact frame preview…"):
            preview_result = create_preview_clip(episode, start_s, end_s)
        st.session_state["_editor_preview"] = {
            "episode": str(episode.path.resolve()),
            "start_s": start_s,
            "end_s": end_s,
            "video_path": str(preview_result.video_path),
            "action_samples": preview_result.action_samples,
            "video_frames": preview_result.video_frames,
        }
        st.rerun()
    except Exception as exc:
        st.error(f"Preview failed: {exc}")

if reset_column.button("Reset range", width="stretch"):
    st.session_state["_editor_in"] = 0.0
    st.session_state["_editor_out"] = float(episode.duration_s)
    st.session_state["_editor_timeline"] = (0.0, float(episode.duration_s))
    st.session_state.pop("_editor_preview", None)
    st.rerun()

if preview_matches and isinstance(preview, dict):
    preview_path = Path(str(preview["video_path"]))
    st.video(
        str(preview_path),
        autoplay=True,
        loop=True,
        muted=True,
        width="stretch",
    )
    st.caption(
        f"Previewing {_format_timecode(start_s)} → {_format_timecode(end_s)} · "
        f"{preview['action_samples']} actions · {preview['video_frames']} frames"
    )
else:
    st.info(
        "Set the floating-point In/Out points above, then click "
        "**Preview selected cut** to play only that range here. "
        "Saving is enabled only for the exact range you previewed."
    )

with st.expander("Episode metadata"):
    st.json(episode.meta, expanded=False)

st.divider()
st.subheader("Save cropped episode")
suggested_name = default_crop_name(episode, start_s, end_s)
output_name = st.text_input(
    "Output episode name",
    value=suggested_name,
    key=(
        f"crop_name_{episode.path.name}_"
        f"{int(round(start_s * 1000))}_{int(round(end_s * 1000))}"
    ),
)

if st.button(
    "💾 Save previewed cut",
    type="primary",
    width="stretch",
    disabled=not preview_matches,
):
    try:
        with st.status("Saving previewed cut…", expanded=True) as status:
            st.write("Selecting aligned CSV rows")
            st.write("Re-encoding the exact source video frame range")
            result = crop_episode(
                episode,
                start_s,
                end_s,
                output_root,
                output_name,
            )
            status.update(label="Cropped episode saved", state="complete", expanded=False)
        st.session_state["_last_crop"] = str(result.output_path)
        st.success(
            f"Created {result.output_path}: {result.action_samples} actions, "
            f"{result.video_frames} frames, {result.duration_s:.2f} s."
        )
    except Exception as exc:
        st.error(f"Crop failed: {exc}")

last_crop = st.session_state.get("_last_crop")
if last_crop:
    cropped_video = Path(last_crop) / "video.mp4"
    if cropped_video.is_file():
        st.subheader("Latest saved output")
        st.code(last_crop)
        st.video(str(cropped_video))
