"""Numerical inverse kinematics for Piper using the Piper SDK FK model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

JOINT_KEYS = tuple(f"joint_{index}.pos" for index in range(1, 7))
EEF_POSE_KEYS = ("eef.x", "eef.y", "eef.z", "eef.rx", "eef.ry", "eef.rz")

# Piper limits from the follower plugin, in degrees.
JOINT_LIMITS_DEG = (
    (-150.0, 150.0),
    (0.0, 180.0),
    (-170.0, 0.0),
    (-100.0, 100.0),
    (-70.0, 70.0),
    (-120.0, 120.0),
)


def _angle_error_deg(actual: float, target: float) -> float:
    return (actual - target + 180.0) % 360.0 - 180.0


class PiperIK:
    """Solve Piper EEF poses to joint targets in degrees.

    The SDK's ``CalFK`` accepts radians and returns XYZ in millimetres plus
    Euler angles in degrees.  The policy-facing EEF representation uses
    metres for XYZ and degrees for RX/RY/RZ.
    """

    def __init__(self, *, dh_is_offset: int = 1) -> None:
        try:
            from piper_sdk import C_PiperForwardKinematics
            from scipy.optimize import least_squares
        except ImportError as exc:
            raise RuntimeError(
                "EEF IK requires piper-sdk and scipy in the active hardware environment."
            ) from exc
        self._fk = C_PiperForwardKinematics(dh_is_offset=dh_is_offset)
        self._least_squares = least_squares

    def solve(
        self,
        target: Mapping[str, float],
        seed: Mapping[str, float],
        *,
        position_tolerance_m: float = 0.005,
        rotation_tolerance_deg: float = 5.0,
    ) -> dict[str, float]:
        missing = [key for key in (*EEF_POSE_KEYS, "gripper.pos") if key not in target]
        if missing:
            raise ValueError(f"EEF policy action is missing: {', '.join(missing)}")
        missing_seed = [key for key in JOINT_KEYS if key not in seed]
        if missing_seed:
            raise ValueError(f"Cannot seed EEF IK; observation is missing: {', '.join(missing_seed)}")

        target_values = [float(target[key]) for key in EEF_POSE_KEYS]
        seed_rad = [math.radians(float(seed[key])) for key in JOINT_KEYS]
        lower = [math.radians(lo) for lo, _ in JOINT_LIMITS_DEG]
        upper = [math.radians(hi) for _, hi in JOINT_LIMITS_DEG]
        seed_rad = [min(max(value, lo), hi) for value, lo, hi in zip(seed_rad, lower, upper, strict=True)]

        def residual(joints_rad: Any) -> list[float]:
            pose = self._fk.CalFK(list(joints_rad))[-1]
            # Scale the dimensions so 5 mm and 5 degrees have comparable
            # influence during optimisation.
            return [
                (float(pose[index]) - target_values[index] * 1000.0) / 5.0
                for index in range(3)
            ] + [
                _angle_error_deg(float(pose[index]), target_values[index]) / 5.0
                for index in range(3, 6)
            ]

        result = self._least_squares(
            residual,
            seed_rad,
            bounds=(lower, upper),
            max_nfev=60,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )
        pose = self._fk.CalFK(result.x)[-1]
        position_error_m = max(
            abs(float(pose[index]) / 1000.0 - target_values[index]) for index in range(3)
        )
        rotation_error_deg = max(
            abs(_angle_error_deg(float(pose[index]), target_values[index]))
            for index in range(3, 6)
        )
        if not result.success or position_error_m > position_tolerance_m or rotation_error_deg > rotation_tolerance_deg:
            raise ValueError(
                "EEF IK did not reach the target: "
                f"position error={position_error_m * 1000.0:.1f} mm, "
                f"rotation error={rotation_error_deg:.1f} deg, "
                f"solver={result.message}"
            )

        return {
            key: math.degrees(float(value))
            for key, value in zip(JOINT_KEYS, result.x, strict=True)
        } | {"gripper.pos": float(target["gripper.pos"])}
