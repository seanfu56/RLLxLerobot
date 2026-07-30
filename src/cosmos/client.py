#!/usr/bin/env python3
"""Talk to a Cosmos3-Nano inference server from the robot process.

    from cosmos.client import CosmosClient

    cosmos = CosmosClient()                       # http://127.0.0.1:8501
    goal_bgr = cosmos.goal_frame(frame_bgr)       # (256, 256, 3) uint8 BGR
    runner.set_goal(goal_bgr)                     # policy/goal.py

This module deliberately imports nothing beyond the standard library, numpy and
OpenCV, all of which the `piper` environment already has.  That is the whole
reason the server exists: the model needs a diffusers/transformers stack that
`lerobot` declares itself incompatible with, so it lives in its own environment
and the robot only ever speaks JSON to it.

Frames go both ways as **BGR** uint8 arrays -- what `cv2.VideoCapture` hands
you and what `policy.inference.PolicyRunner.select_action` expects -- so there
is no channel swap to get wrong at the boundary.

As a CLI:

    python src/cosmos/client.py --health
    python src/cosmos/client.py --image frame.png --output-dir samples/live
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

DEFAULT_URL = "http://127.0.0.1:8501"
# A 93-frame generation takes ~7 s at 30 steps; the default leaves room for a
# cold server that is still loading 35 GB of weights.
DEFAULT_TIMEOUT = 300.0


class CosmosServerError(RuntimeError):
    """The server was reached but refused or failed the request."""


def _encode_png(frame_bgr: np.ndarray) -> str:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 BGR frame, got shape {frame_bgr.shape}")
    if frame_bgr.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 BGR frame, got dtype {frame_bgr.dtype}")
    ok, buffer = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed to encode a PNG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _decode_png(payload: str) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        raise CosmosServerError("Server returned an image that could not be decoded")
    return frame


class CosmosClient:
    """A thin JSON client for ``src/cosmos/serve.py``."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = DEFAULT_TIMEOUT):
        self.url = url.rstrip("/")
        self.timeout = float(timeout)

    def _post(self, endpoint: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except json.JSONDecodeError:
                pass
            raise CosmosServerError(f"HTTP {error.code} from {self.url}{endpoint}: {detail}") from error
        except urllib.error.URLError as error:
            raise CosmosServerError(
                f"Could not reach the Cosmos3 server at {self.url} ({error.reason}). "
                f"Start it with: python src/cosmos/serve.py"
            ) from error

    def health(self) -> dict:
        """Server configuration, or raise if it is not up."""
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.URLError as error:
            raise CosmosServerError(
                f"Could not reach the Cosmos3 server at {self.url} ({error.reason})"
            ) from error

    def _request(self, frame_bgr: np.ndarray, output: str, **kwargs) -> dict:
        payload = {"image": _encode_png(frame_bgr), "output": output}
        payload.update({key: value for key, value in kwargs.items() if value is not None})
        return self._post("/generate", payload)

    def goal_frame(self, frame_bgr: np.ndarray, **kwargs) -> np.ndarray:
        """Predicted end state as a BGR frame, ready for ``set_goal``.

        Only the final frame crosses the wire, which is what makes this cheap
        enough to call inside a rollout.
        """
        response = self._request(frame_bgr, "goal", **kwargs)
        if "goal" not in response:
            raise CosmosServerError(f"Server response has no 'goal' field: {sorted(response)}")
        return _decode_png(response["goal"])

    def generate(self, frame_bgr: np.ndarray, **kwargs) -> list[np.ndarray]:
        """The full predicted rollout as BGR frames."""
        response = self._request(frame_bgr, "frames", **kwargs)
        if "frames" not in response:
            raise CosmosServerError(f"Server response has no 'frames' field: {sorted(response)}")
        return [_decode_png(frame) for frame in response["frames"]]

    def video(self, frame_bgr: np.ndarray, **kwargs) -> bytes:
        """The predicted rollout as encoded MP4 bytes."""
        response = self._request(frame_bgr, "video", **kwargs)
        if "video" not in response:
            raise CosmosServerError(f"Server response has no 'video' field: {sorted(response)}")
        return base64.b64decode(response["video"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Command-line client for the Cosmos3-Nano inference server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--health", action="store_true", help="Print server config and exit")
    parser.add_argument("--image", type=Path, default=None, help="Conditioning frame")
    parser.add_argument(
        "--output",
        choices=("goal", "frames", "video"),
        default="video",
        help="What to ask the server for",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("samples/cosmos3-nano"))
    parser.add_argument("--name", default=None, help="Output basename; defaults to the image stem")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    client = CosmosClient(args.url, args.timeout)

    if args.health or args.image is None:
        print(json.dumps(client.health(), indent=2))
        if args.image is None and not args.health:
            print("\nPass --image to generate.")
        return

    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    options = {
        "prompt": args.prompt,
        "num_frames": args.num_frames,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "fps": args.fps,
        "seed": args.seed,
    }
    name = args.name or args.image.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.output == "goal":
        target = args.output_dir / f"{name}_goal.png"
        cv2.imwrite(str(target), client.goal_frame(frame, **options))
    elif args.output == "frames":
        target = args.output_dir / name
        target.mkdir(parents=True, exist_ok=True)
        for index, generated in enumerate(client.generate(frame, **options)):
            cv2.imwrite(str(target / f"{index:04d}.png"), generated)
    else:
        target = args.output_dir / f"{name}.mp4"
        target.write_bytes(client.video(frame, **options))
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
