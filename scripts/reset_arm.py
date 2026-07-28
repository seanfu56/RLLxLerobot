#!/usr/bin/env python3
"""Park a Piper arm at its configured rest pose and disable it safely.

This is a standalone recovery/maintenance command. It must not run while the
Streamlit teleoperation UI or another process is sending commands to the same
CAN interface. When the UI owns the arm, use its "Safe disconnect" button so
the control worker is stopped before the rest trajectory begins.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--can-port",
        default="piper_left",
        help="Piper CAN interface (default: piper_left).",
    )
    parser.add_argument(
        "--speed-rate",
        type=int,
        default=30,
        help="Firmware motion speed percentage during shutdown (default: 30).",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=2.0,
        help="Maximum per-command joint change in degrees (default: 2.0).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the shutdown plan without touching hardware.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.can_port:
        raise ValueError("--can-port must not be empty.")
    if not 1 <= args.speed_rate <= 100:
        raise ValueError("--speed-rate must be between 1 and 100.")
    if args.max_relative_target <= 0:
        raise ValueError("--max-relative-target must be positive.")


def find_streamlit_ui_processes(proc_root: Path = Path("/proc")) -> list[int]:
    """Return PIDs whose command line appears to run this repository's UI."""
    matches: list[int] = []
    own_pid = os.getpid()
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return matches

    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == own_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        arguments = [
            part.decode(errors="replace") for part in raw.split(b"\0") if part
        ]
        if not arguments:
            continue
        joined = " ".join(arguments)
        if "streamlit" in joined and any(
            argument == "ui/app.py" or argument.endswith("/ui/app.py")
            for argument in arguments
        ):
            matches.append(pid)
    return sorted(matches)


def confirm_shutdown(can_port: str) -> bool:
    prompt = (
        f"Piper on {can_port} will move to its configured rest pose and disable. "
        "Keep the emergency stop within reach.\n"
        "Type CLOSE to continue: "
    )
    try:
        return input(prompt).strip() == "CLOSE"
    except EOFError:
        return False


@contextlib.contextmanager
def defer_termination_signals() -> Iterator[None]:
    """Prevent Ctrl+C/SIGTERM from interrupting the rest-and-disable sequence."""
    handled = (signal.SIGINT, signal.SIGTERM)
    previous: dict[signal.Signals, Any] = {}
    received: list[signal.Signals] = []

    def defer(signum: int, _frame: Any) -> None:
        received.append(signal.Signals(signum))
        print(
            "\nShutdown is already in progress; deferring the termination signal "
            "until the arm is disabled.",
            file=sys.stderr,
            flush=True,
        )

    for current_signal in handled:
        previous[current_signal] = signal.getsignal(current_signal)
        signal.signal(current_signal, defer)
    try:
        yield
    finally:
        for current_signal, handler in previous.items():
            signal.signal(current_signal, handler)
        if received:
            print(
                "The arm shutdown completed after a termination signal was received.",
                file=sys.stderr,
                flush=True,
            )


def force_disable(robot: Any) -> None:
    """Best-effort fallback when normal PiperFollower.disconnect() raises."""
    piper = getattr(robot, "piper", None)
    if piper is None:
        return
    try:
        piper.DisableArm()
    finally:
        try:
            piper.GripperCtrl(0, 0, 0x00, 0)
        finally:
            if hasattr(robot, "_is_connected"):
                robot._is_connected = False


def close_arm(
    config: Any,
    robot_factory: Callable[[Any], Any],
) -> None:
    """Connect safely, run the plugin rest trajectory, and disable the arm."""
    robot = robot_factory(config)
    normal_disconnect_completed = False
    try:
        print(f"Connecting to Piper on {config.can_port}...", flush=True)
        robot.connect()
        print("Connected. Moving to the configured rest pose...", flush=True)
        robot.disconnect()
        normal_disconnect_completed = True
        print("Rest trajectory complete; arm and gripper are disabled.", flush=True)
    finally:
        if not normal_disconnect_completed and getattr(robot, "piper", None) is not None:
            print(
                "Normal safe disconnect failed; requesting immediate arm disable.",
                file=sys.stderr,
                flush=True,
            )
            force_disable(robot)


def main(
    argv: list[str] | None = None,
    *,
    proc_root: Path = Path("/proc"),
) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    ui_processes = find_streamlit_ui_processes(proc_root)
    if ui_processes:
        joined_pids = ", ".join(str(pid) for pid in ui_processes)
        print(
            "ERROR: ui/app.py appears to be running "
            f"(PID(s): {joined_pids}). Use the UI's Safe disconnect button so its "
            "control worker stops before the arm moves to rest.",
            file=sys.stderr,
        )
        return 3

    print("Safe shutdown plan:")
    print(f"  CAN interface:       {args.can_port}")
    print(f"  motion speed:        {args.speed_rate}%")
    print(f"  relative step limit: {args.max_relative_target} degrees")
    print("  sequence: connect/hold -> smooth rest move -> disable arm and gripper")

    if args.dry_run:
        print("Dry run complete; no hardware was accessed.")
        return 0
    if not args.yes and not confirm_shutdown(args.can_port):
        print("Cancelled; no hardware was accessed.")
        return 1

    try:
        from lerobot_robot_piper import PiperFollower, PiperFollowerConfig
    except ImportError as exc:
        print(
            "ERROR: Piper plugin is unavailable. Activate the piper environment "
            "and install plugins/lerobot_robot_piper.",
            file=sys.stderr,
        )
        return 4

    config = PiperFollowerConfig(
        can_port=args.can_port,
        speed_rate=args.speed_rate,
        max_relative_target=args.max_relative_target,
        cameras={},
    )

    try:
        with defer_termination_signals():
            close_arm(config, PiperFollower)
    except BaseException as exc:
        print(f"ERROR: arm shutdown failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Use the physical emergency stop and support the arm if it is not in "
            "the rest pose.",
            file=sys.stderr,
        )
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
