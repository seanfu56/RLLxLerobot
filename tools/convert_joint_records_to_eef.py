#!/usr/bin/env python3
"""Append Piper FK end-effector pose columns to recorded CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

from piper_sdk import C_PiperForwardKinematics

JOINT_COLUMNS = tuple(f"joint_{index}.pos" for index in range(1, 7))
EEF_COLUMNS = ("eef.x", "eef.y", "eef.z", "eef.rx", "eef.ry", "eef.rz")


def convert_file(path: Path, fk: C_PiperForwardKinematics) -> int:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        missing = [column for column in JOINT_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing joint columns: {missing}")
        if all(column in reader.fieldnames for column in EEF_COLUMNS):
            return 0

        fieldnames = [*reader.fieldnames, *[column for column in EEF_COLUMNS if column not in reader.fieldnames]]
        rows = []
        for row in reader:
            joints_rad = [math.radians(float(row[column])) for column in JOINT_COLUMNS]
            pose = fk.CalFK(joints_rad)[-1]
            row.update(
                {
                    **{
                        column: f"{float(value) / 1000.0:.9f}"
                        for column, value in zip(EEF_COLUMNS[:3], pose[:3], strict=True)
                    },
                    **{
                        column: f"{float(value):.9f}"
                        for column, value in zip(EEF_COLUMNS[3:], pose[3:], strict=True)
                    },
                }
            )
            rows.append(row)

    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("piper-data/raw"))
    parser.add_argument(
        "--dh-is-offset",
        type=int,
        choices=(0, 1),
        default=1,
        help="Piper SDK DH offset setting; use the setting matching the robot calibration.",
    )
    args = parser.parse_args()

    paths = sorted(
        [*args.root.glob("*/episode_*/actions.csv"), *args.root.glob("*/episode_*/observations.csv")]
    )
    if not paths:
        raise SystemExit(f"No observations.csv files found below {args.root}")

    fk = C_PiperForwardKinematics(dh_is_offset=args.dh_is_offset)
    converted = 0
    skipped = 0
    rows = 0
    for path in paths:
        count = convert_file(path, fk)
        if count:
            converted += 1
            rows += count
        else:
            skipped += 1
    print(f"Converted {converted} files / {rows} rows; skipped {skipped} already-converted files.")


if __name__ == "__main__":
    main()
