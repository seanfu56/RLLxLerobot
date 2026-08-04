"""Writing generated clips to disk without throwing any of them away.

Every synthetic clip in this repo is training data before it is anything else.
A DPO step is a difference of two per-element reconstruction errors, and the
condition-aligned pairing ranks candidates by DINOv2 distances that differ by
~0.02; both are differences of small numbers, so whatever the codec invents sits
in the same range as the signal and is not separable from it afterwards.

``diffusers.utils.export_to_video`` - what this module replaces - hands imageio
``quality=5`` on a 0-10 scale, which is variable-bitrate H.264 in ``yuv420p``.
On 256x256x93 clips that is roughly 550:1, about 42 dB PSNR, and measured
against the pairs in ``outputs/self_pairs``, one further encode moves a clip by
a median 39% of the winner-loser margin it is supposed to be ranked by. It also
defaults to ``macro_block_size=16``, which silently *rescales* a clip whose
dimensions are not a multiple of 16 rather than refusing.

``libx264rgb -qp 0`` is lossless and keeps the ``.mp4`` extension every reader
here globs for, so nothing downstream has to change. RGB rather than
``yuv444p``: at ``-qp 0`` the H.264 residual is exact either way, but the
RGB->YUV conversion in front of it rounds, and a round trip through
``yuv444p -qp 0`` comes back off by up to 2 counts per channel. FFV1 is equally
exact and about the same size, but only in Matroska or AVI.

Costs about 60x the bytes: ~2 MB for a 93-frame 256x256 clip against ~34 KB.
That is the price of the clip being usable as training data.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "LOSSLESS",
    "LOSSY_PREVIEW",
    "frames_to_array",
    "rebuild_from_frames",
    "save_video",
]

# -qp 0 is lossless in libx264's rate-control sense: the residual is coded
# exactly. libx264rgb skips the colourspace conversion that would round first.
LOSSLESS = ("-c:v", "libx264rgb", "-qp", "0", "-pix_fmt", "rgb24")
# For anything that is only ever looked at. Never for clips that get trained on.
LOSSY_PREVIEW = ("-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p")


def frames_to_array(frames: Iterable) -> np.ndarray:
    """``N x H x W x 3` uint8 RGB from PIL images, arrays, or a stacked array.

    Floating point input is taken to be in ``[0, 1]`` and is scaled; integer
    input is taken to be finished pixels and is not. ``export_to_video``
    multiplies *every* ndarray by 255, which silently wraps uint8 frames - the
    reason its callers here were passing PIL images to work around it.
    """
    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        stacked = frames
    else:
        items: Sequence = list(frames)
        if not items:
            raise ValueError("Refusing to write a video with no frames")
        stacked = np.stack([np.asarray(item) for item in items])

    if stacked.ndim != 4 or stacked.shape[-1] != 3:
        raise ValueError(
            f"Expected N x H x W x 3 RGB frames, got {tuple(stacked.shape)}"
        )
    if np.issubdtype(stacked.dtype, np.floating):
        stacked = np.rint(np.clip(stacked, 0.0, 1.0) * 255.0)
    elif stacked.dtype != np.uint8:
        stacked = np.clip(stacked, 0, 255)
    return np.ascontiguousarray(stacked.astype(np.uint8))


def save_video(
    frames: Iterable,
    path: str | Path,
    fps: float,
    *,
    lossless: bool = True,
) -> Path:
    """Write ``frames`` to ``path``. Lossless unless told otherwise.

    ``frames`` may be PIL images, a list of ``H x W x 3`` arrays, or one stacked
    ``N x H x W x 3`` array; see :func:`frames_to_array` for the dtype rules.
    """
    import imageio_ffmpeg

    array = frames_to_array(frames)
    count, height, width, _ = array.shape
    # H.264 codes in 16x16 macroblocks and pads to even dimensions internally.
    # Odd input is a caller mistake worth surfacing: the alternative every video
    # library picks is to resize, which loses more than the codec would.
    if height % 2 or width % 2:
        raise ValueError(
            f"H.264 needs even dimensions, got {width}x{height}. Crop or pad "
            "before calling; resizing here would defeat the point of the module."
        )
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    codec = LOSSLESS if lossless else LOSSY_PREVIEW
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", f"{fps:g}", "-i", "-",
        *codec,
        str(target),
    ]
    result = subprocess.run(command, input=array.tobytes(), capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed writing {target} ({count} frames, {width}x{height}):\n"
            f"{result.stderr.decode('utf-8', 'replace')}"
        )
    return target


def rebuild_from_frames(archive: Path, *, dry_run: bool = False) -> dict | None:
    """Rewrite one goal-video archive's clip losslessly from its PNG frames.

    A goal-video archive stores both ``frame_NNN.png`` - the model's output,
    lossless - and the ``video.mp4`` that ``finetune/goal_videos.py`` actually
    trains on, which was written lossily. The frames are authoritative, so the
    clip can be rebuilt exactly without regenerating anything on a GPU.

    Returns what changed, or ``None`` if the archive has no frames to rebuild
    from. Reports the PSNR of the clip being replaced so the size of what was
    being trained on is on the record.
    """
    import cv2

    frame_paths = sorted(archive.glob("frame_*.png"))
    meta_path = archive / "meta.json"
    if not frame_paths or not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    clip = archive / str(meta.get("clip") or "video.mp4")

    frames = np.stack([
        cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        for path in frame_paths
    ])

    before, psnr = None, None
    if clip.is_file():
        capture = cv2.VideoCapture(str(clip))
        decoded = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        before = clip.stat().st_size
        if len(decoded) == len(frames):
            error = np.mean(
                (np.stack(decoded).astype(np.float64) - frames.astype(np.float64)) ** 2
            )
            psnr = float(10 * np.log10(255.0**2 / max(float(error), 1e-12)))
        else:
            # Not fatal - the rebuild is what fixes it - but the old clip
            # disagreeing with the frames on length is worth saying out loud.
            psnr = float("nan")

    record = {
        "archive": str(archive),
        "frames": len(frames),
        "fps": float(meta.get("fps") or 20.0),
        "bytes_before": before,
        "psnr_before": psnr,
    }
    if dry_run:
        return record
    save_video(frames, clip, record["fps"])
    record["bytes_after"] = clip.stat().st_size
    return record


def _main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebuild goal-video clips losslessly from their PNG frames."
    )
    parser.add_argument("roots", nargs="+", type=Path,
                        help="ft_dataset roots, or any tree holding frame_*.png archives")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing")
    args = parser.parse_args(argv)

    archives = sorted({
        path.parent
        for root in args.roots
        for path in Path(root).rglob("frame_000.png")
    })
    if not archives:
        raise SystemExit(f"No frame_*.png archives found under {args.roots}")

    print(f"{'archive':22s} {'frames':>7} {'psnr_before':>12} {'KB before':>10} {'KB after':>9}")
    total_before = total_after = 0
    for archive in archives:
        record = rebuild_from_frames(archive, dry_run=args.dry_run)
        if record is None:
            continue
        before = record["bytes_before"] or 0
        after = record.get("bytes_after", 0)
        total_before += before
        total_after += after
        psnr = record["psnr_before"]
        print(f"{archive.name:22s} {record['frames']:7d} "
              f"{'-' if psnr is None else f'{psnr:12.2f}'} "
              f"{before / 1024:10.0f} {after / 1024:9.0f}")
    verb = "would rebuild" if args.dry_run else "rebuilt"
    print(f"\n{verb} {len(archives)} archive(s): "
          f"{total_before / 1e6:.1f} MB -> {total_after / 1e6:.1f} MB")


if __name__ == "__main__":
    _main()
