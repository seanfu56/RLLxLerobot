from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "close_robot_arm.py"
SPEC = importlib.util.spec_from_file_location("close_robot_arm", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
close_robot_arm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(close_robot_arm)


class FakePiper:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def DisableArm(self) -> None:
        self.calls.append(("DisableArm",))

    def GripperCtrl(self, *args) -> None:
        self.calls.append(("GripperCtrl", *args))


class FakeRobot:
    def __init__(self, config, *, fail_disconnect: bool = False) -> None:
        self.config = config
        self.fail_disconnect = fail_disconnect
        self.calls: list[str] = []
        self.piper = FakePiper()
        self._is_connected = False

    def connect(self) -> None:
        self.calls.append("connect")
        self._is_connected = True

    def disconnect(self) -> None:
        self.calls.append("disconnect")
        if self.fail_disconnect:
            raise RuntimeError("synthetic disconnect failure")
        self._is_connected = False


class CloseRobotArmTests(unittest.TestCase):
    def config(self):
        return SimpleNamespace(can_port="piper_left")

    def test_close_arm_uses_plugin_disconnect_after_connect(self) -> None:
        robot = FakeRobot(self.config())

        close_robot_arm.close_arm(robot.config, lambda _config: robot)

        self.assertEqual(robot.calls, ["connect", "disconnect"])
        self.assertEqual(robot.piper.calls, [])

    def test_disconnect_failure_falls_back_to_immediate_disable(self) -> None:
        robot = FakeRobot(self.config(), fail_disconnect=True)

        with self.assertRaisesRegex(RuntimeError, "synthetic disconnect failure"):
            close_robot_arm.close_arm(robot.config, lambda _config: robot)

        self.assertEqual(robot.calls, ["connect", "disconnect"])
        self.assertEqual(
            robot.piper.calls,
            [
                ("DisableArm",),
                ("GripperCtrl", 0, 0, 0x00, 0),
            ],
        )
        self.assertFalse(robot._is_connected)

    def test_detects_running_streamlit_ui(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc_root = Path(directory)
            process = proc_root / "4321"
            process.mkdir()
            (process / "cmdline").write_bytes(
                b"streamlit\0run\0/home/seanfu/RLLxLerobot/ui/app.py\0"
            )

            self.assertEqual(
                close_robot_arm.find_streamlit_ui_processes(proc_root),
                [4321],
            )

    def test_dry_run_never_imports_or_touches_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = close_robot_arm.main(
                ["--dry-run", "--yes"],
                proc_root=Path(directory),
            )

        self.assertEqual(result, 0)

    def test_invalid_motion_settings_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = close_robot_arm.main(
                ["--dry-run", "--speed-rate", "0"],
                proc_root=Path(directory),
            )

        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
