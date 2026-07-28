"""Non-destructive temporal cropping for recorded teleoperation episodes."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeData:
    path: Path
    video_path: Path
    actions: tuple[dict[str, str], ...]
    observations: tuple[dict[str, str], ...]
    frames: tuple[dict[str, str], ...]
    meta: dict[str, Any]
    aligned: bool
    duration_s: float


@dataclass(frozen=True)
class CropResult:
    output_path: Path
    action_samples: int
    video_frames: int
    duration_s: float
    aligned: bool


@dataclass(frozen=True)
class PreviewResult:
    video_path: Path
    action_samples: int
    video_frames: int
    start_s: float
    end_s: float


@dataclass(frozen=True)
class _CropSelection:
    actions: tuple[dict[str, str], ...]
    observations: tuple[dict[str, str], ...]
    frames: tuple[dict[str, str], ...]
    first_frame_index: int
    last_frame_index: int
    origin_s: float


def discover_episodes(root: Path) -> list[Path]:
    root = Path(root).expanduser()
    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and (path / "actions.csv").is_file()
            and (path / "frame_timestamps.csv").is_file()
            and ((path / "video.mp4").is_file() or (path / "video.avi").is_file())
        ),
        key=lambda path: path.name,
    )


def load_episode(path: Path) -> EpisodeData:
    path = Path(path).expanduser()
    actions_path = path / "actions.csv"
    observations_path = path / "observations.csv"
    frames_path = path / "frame_timestamps.csv"
    meta_path = path / "meta.json"
    video_path = path / "video.mp4"
    if not video_path.is_file():
        video_path = path / "video.avi"

    missing = [
        candidate.name
        for candidate in (actions_path, frames_path, meta_path, video_path)
        if not candidate.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Episode {path} is missing required files: {', '.join(missing)}"
        )

    with actions_path.open(newline="") as handle:
        actions = tuple(dict(row) for row in csv.DictReader(handle))
    observations: tuple[dict[str, str], ...] = ()
    if observations_path.is_file():
        with observations_path.open(newline="") as handle:
            observations = tuple(dict(row) for row in csv.DictReader(handle))
    with frames_path.open(newline="") as handle:
        frames = tuple(dict(row) for row in csv.DictReader(handle))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not actions:
        raise ValueError(f"Episode {path} has no action rows.")
    if observations_path.is_file() and not observations:
        raise ValueError(f"Episode {path} has no observation rows.")
    if not frames:
        raise ValueError(f"Episode {path} has no frame timestamp rows.")

    action_times = [_as_float(row, "sample_timestamp_s") for row in actions]
    observation_times = [
        _as_float(row, "observation_timestamp_s") for row in observations
    ]
    frame_times = [_as_float(row, "timestamp_s") for row in frames]
    _require_monotonic(action_times, "action")
    _require_monotonic(observation_times, "observation")
    _require_monotonic(frame_times, "frame")

    aligned = len(actions) == len(frames) and (
        not observations or len(observations) == len(actions)
    )
    if aligned:
        for index, (action, frame) in enumerate(zip(actions, frames, strict=True)):
            try:
                row_aligned = (
                    _as_int(action, "sample_index") == index
                    and _as_int(action, "frame_index") == index
                    and _as_int(frame, "frame_index") == index
                    and action.get("capture_sequence")
                    == frame.get("capture_sequence")
                )
                if observations:
                    observation = observations[index]
                    row_aligned = (
                        row_aligned
                        and _as_int(observation, "sample_index") == index
                        and _as_int(observation, "frame_index") == index
                        and observation.get("source_control_sequence")
                        == action.get("source_control_sequence")
                    )
            except ValueError:
                row_aligned = False
            if not row_aligned:
                aligned = False
                break

    duration_values = [
        action_times[-1],
        frame_times[-1],
        float(meta.get("duration_s", 0.0) or 0.0),
    ]
    if observation_times:
        duration_values.append(observation_times[-1])
    duration_s = max(duration_values)
    return EpisodeData(
        path=path,
        video_path=video_path,
        actions=actions,
        observations=observations,
        frames=frames,
        meta=meta,
        aligned=aligned,
        duration_s=duration_s,
    )


def default_crop_name(episode: EpisodeData, start_s: float, end_s: float) -> str:
    start_ms = max(int(round(start_s * 1000)), 0)
    end_ms = max(int(round(end_s * 1000)), start_ms)
    return f"{episode.path.name}_crop_{start_ms:06d}_{end_ms:06d}"


def create_preview_clip(
    episode: EpisodeData,
    start_s: float,
    end_s: float,
    preview_root: Path | None = None,
) -> PreviewResult:
    selection = _select_episode_range(episode, start_s, end_s)
    if preview_root is None:
        preview_root = Path(tempfile.gettempdir()) / "rllxlerobot-teleop-previews"
    preview_root = Path(preview_root)
    preview_root.mkdir(parents=True, exist_ok=True)
    source_stat = episode.video_path.stat()
    cache_key = hashlib.sha256(
        (
            f"{episode.video_path.resolve()}:{source_stat.st_mtime_ns}:"
            f"{selection.first_frame_index}:{selection.last_frame_index}"
        ).encode()
    ).hexdigest()[:20]
    preview_path = preview_root / f"{episode.path.name}-{cache_key}.mp4"
    expected_frames = len(selection.frames)
    if not preview_path.is_file():
        pending_path = preview_path.with_suffix(".pending.mp4")
        pending_path.unlink(missing_ok=True)
        try:
            _crop_video_frames(
                episode.video_path,
                pending_path,
                selection.first_frame_index,
                selection.last_frame_index + 1,
            )
            actual_frames = _probe_video_frame_count(pending_path)
            if actual_frames != expected_frames:
                raise RuntimeError(
                    f"Preview contains {actual_frames} frames; expected {expected_frames}."
                )
            pending_path.replace(preview_path)
        except BaseException:
            pending_path.unlink(missing_ok=True)
            raise
    else:
        actual_frames = _probe_video_frame_count(preview_path)
        if actual_frames != expected_frames:
            preview_path.unlink(missing_ok=True)
            return create_preview_clip(episode, start_s, end_s, preview_root)

    return PreviewResult(
        video_path=preview_path,
        action_samples=len(selection.actions),
        video_frames=expected_frames,
        start_s=start_s,
        end_s=end_s,
    )


def crop_episode(
    episode: EpisodeData,
    start_s: float,
    end_s: float,
    output_root: Path,
    output_name: str | None = None,
) -> CropResult:
    selection = _select_episode_range(episode, start_s, end_s)
    action_rows = [dict(row) for row in selection.actions]
    observation_rows = [dict(row) for row in selection.observations]
    frame_rows = [dict(row) for row in selection.frames]
    first_frame_index = selection.first_frame_index
    last_frame_index = selection.last_frame_index
    origin_s = selection.origin_s
    rebased_actions = _rebase_actions(action_rows, origin_s, episode.aligned)
    rebased_observations = _rebase_observations(
        observation_rows,
        origin_s,
        episode.aligned,
    )
    rebased_frames = _rebase_frames(frame_rows, origin_s)

    output_root = Path(output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    name = output_name or default_crop_name(episode, start_s, end_s)
    _validate_output_name(name)
    final_path = output_root / name
    if final_path.exists():
        raise FileExistsError(f"Output already exists: {final_path}")
    pending_path = Path(
        tempfile.mkdtemp(prefix=f".{name}.pending-", dir=output_root)
    )

    try:
        output_video = pending_path / "video.mp4"
        _crop_video_frames(
            episode.video_path,
            output_video,
            first_frame_index,
            last_frame_index + 1,
        )
        expected_frames = len(rebased_frames)
        actual_frames = _probe_video_frame_count(output_video)
        if actual_frames != expected_frames:
            raise RuntimeError(
                f"Cropped video contains {actual_frames} frames; expected {expected_frames}."
            )

        _write_csv(pending_path / "actions.csv", rebased_actions)
        if rebased_observations:
            _write_csv(
                pending_path / "observations.csv",
                rebased_observations,
            )
        _write_csv(pending_path / "frame_timestamps.csv", rebased_frames)

        last_action_s = _as_float(rebased_actions[-1], "sample_timestamp_s")
        last_frame_s = _as_float(rebased_frames[-1], "timestamp_s")
        video_fps = float(episode.meta.get("video_fps_target", 0.0) or 0.0)
        frame_period_s = 1.0 / video_fps if video_fps > 0 else 0.0
        duration_s = max(last_action_s, last_frame_s + frame_period_s)
        meta = dict(episode.meta)
        meta.update(
            {
                "source_episode": episode.path.name,
                "source_path": str(episode.path.resolve()),
                "duration_s": round(duration_s, 6),
                "action_samples": len(rebased_actions),
                "observation_samples": len(rebased_observations),
                "video_frames": len(rebased_frames),
                "alignment": (
                    "one action and measured observation row per video frame; "
                    "frame_index is one-to-one"
                    if rebased_observations and episode.aligned
                    else "one action row per video frame; frame_index is one-to-one"
                    if episode.aligned
                    else "legacy timestamp crop; source was not one-to-one aligned"
                ),
                "temporal_crop": {
                    "requested_start_s": round(start_s, 6),
                    "requested_end_s": round(end_s, 6),
                    "source_first_frame_index": first_frame_index,
                    "source_last_frame_index": last_frame_index,
                    "timestamp_origin_s": round(origin_s, 6),
                },
            }
        )
        (pending_path / "meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        pending_path.rename(final_path)
        return CropResult(
            output_path=final_path,
            action_samples=len(rebased_actions),
            video_frames=len(rebased_frames),
            duration_s=duration_s,
            aligned=episode.aligned,
        )
    except BaseException:
        shutil.rmtree(pending_path, ignore_errors=True)
        raise


def _select_episode_range(
    episode: EpisodeData,
    start_s: float,
    end_s: float,
) -> _CropSelection:
    if start_s < 0:
        raise ValueError("Crop start must not be negative.")
    if end_s <= start_s:
        raise ValueError("Crop end must be greater than crop start.")
    if start_s > episode.duration_s:
        raise ValueError("Crop start is beyond the end of the episode.")

    action_rows = tuple(
        dict(row)
        for row in episode.actions
        if start_s <= _as_float(row, "sample_timestamp_s") <= end_s
    )
    if not action_rows:
        raise ValueError("The selected range contains no action samples.")

    observation_rows: tuple[dict[str, str], ...] = ()
    if episode.aligned:
        source_frame_indices = [_as_int(row, "frame_index") for row in action_rows]
        first_frame_index = source_frame_indices[0]
        last_frame_index = source_frame_indices[-1]
        frame_rows = tuple(
            dict(row)
            for row in episode.frames[first_frame_index : last_frame_index + 1]
        )
        if len(frame_rows) != len(action_rows):
            raise RuntimeError("Aligned source episode produced a mismatched crop.")
        if episode.observations:
            observation_rows = tuple(
                dict(row)
                for row in episode.observations[
                    first_frame_index : last_frame_index + 1
                ]
            )
            if len(observation_rows) != len(action_rows):
                raise RuntimeError(
                    "Aligned source episode produced mismatched observations."
                )
    else:
        frame_rows = tuple(
            dict(row)
            for row in episode.frames
            if start_s <= _as_float(row, "timestamp_s") <= end_s
        )
        if not frame_rows:
            raise ValueError("The selected range contains no video frames.")
        first_frame_index = _as_int(frame_rows[0], "frame_index")
        last_frame_index = _as_int(frame_rows[-1], "frame_index")
        if episode.observations:
            observation_rows = tuple(
                dict(row)
                for row in episode.observations
                if start_s
                <= _as_float(row, "observation_timestamp_s")
                <= end_s
            )

    origin_candidates = [
        _as_float(action_rows[0], "sample_timestamp_s"),
        _as_float(frame_rows[0], "timestamp_s"),
    ]
    if observation_rows:
        origin_candidates.append(
            _as_float(observation_rows[0], "observation_timestamp_s")
        )
    origin_s = min(origin_candidates)
    return _CropSelection(
        actions=action_rows,
        observations=observation_rows,
        frames=frame_rows,
        first_frame_index=first_frame_index,
        last_frame_index=last_frame_index,
        origin_s=origin_s,
    )


def _rebase_actions(
    rows: list[dict[str, str]],
    origin_s: float,
    aligned: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for new_index, source in enumerate(rows):
        row: dict[str, Any] = dict(source)
        row["source_sample_index"] = source.get("sample_index", "")
        row["sample_index"] = new_index
        if "frame_index" in row:
            row["source_frame_index"] = source.get("frame_index", "")
            row["frame_index"] = new_index if aligned else source.get("frame_index", "")
        for field in (
            "sample_timestamp_s",
            "source_control_timestamp_s",
            "frame_timestamp_s",
        ):
            if row.get(field) not in (None, ""):
                row[field] = round(float(row[field]) - origin_s, 6)
        result.append(row)
    return result


def _rebase_observations(
    rows: list[dict[str, str]],
    origin_s: float,
    aligned: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for new_index, source in enumerate(rows):
        row: dict[str, Any] = dict(source)
        row["source_sample_index"] = source.get("sample_index", "")
        row["sample_index"] = new_index
        if "frame_index" in row:
            row["source_frame_index"] = source.get("frame_index", "")
            row["frame_index"] = new_index if aligned else source.get("frame_index", "")
        for field in (
            "observation_timestamp_s",
            "source_control_timestamp_s",
        ):
            if row.get(field) not in (None, ""):
                row[field] = round(float(row[field]) - origin_s, 6)
        result.append(row)
    return result


def _rebase_frames(
    rows: list[dict[str, str]],
    origin_s: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for new_index, source in enumerate(rows):
        row: dict[str, Any] = dict(source)
        row["source_frame_index"] = source.get("frame_index", "")
        row["frame_index"] = new_index
        row["timestamp_s"] = round(
            _as_float(source, "timestamp_s") - origin_s,
            6,
        )
        result.append(row)
    return result


def _crop_video_frames(
    source: Path,
    destination: Path,
    start_frame: int,
    end_frame: int,
) -> None:
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source),
                "-vf",
                (
                    f"trim=start_frame={start_frame}:end_frame={end_frame},"
                    "setpts=PTS-STARTPTS"
                ),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-threads",
                "1",
                str(destination),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for temporal cropping.") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg crop failed: {result.stderr.strip()}")


def _probe_video_frame_count(video_path: Path) -> int:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required to validate cropped video.") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe returned an invalid frame count: {result.stdout.strip()!r}"
        ) from exc


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(row: dict[str, Any], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid or missing {field!r} value in CSV row.") from exc


def _as_int(row: dict[str, Any], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid or missing {field!r} value in CSV row.") from exc


def _require_monotonic(values: list[float], label: str) -> None:
    if any(current < previous for previous, current in zip(values, values[1:])):
        raise ValueError(f"{label.capitalize()} timestamps are not monotonic.")


def _validate_output_name(name: str) -> None:
    if name in {"", ".", ".."} or re.fullmatch(r"[A-Za-z0-9._-]+", name) is None:
        raise ValueError(
            "Output name may contain only letters, numbers, dots, underscores, and hyphens."
        )
