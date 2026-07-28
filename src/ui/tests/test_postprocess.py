from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ui.postprocess import (
    create_preview_clip,
    crop_episode,
    discover_episodes,
    load_episode,
)


class PostprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_aligned_temporal_crop_preserves_one_action_per_frame(self) -> None:
        source = self._make_episode("episode_000", action_count=12, frame_count=12)
        episode = load_episode(source)
        self.assertTrue(episode.aligned)

        result = crop_episode(
            episode,
            start_s=0.10,
            end_s=0.31,
            output_root=self.root / "processed",
            output_name="center_crop",
        )
        cropped = load_episode(result.output_path)

        self.assertTrue(result.aligned)
        self.assertTrue(cropped.aligned)
        self.assertEqual(result.action_samples, 5)
        self.assertEqual(result.video_frames, 5)
        self.assertEqual(len(cropped.actions), len(cropped.frames))
        self.assertEqual(len(cropped.observations), len(cropped.frames))
        self.assertEqual(cropped.actions[0]["source_sample_index"], "2")
        self.assertEqual(cropped.actions[-1]["source_sample_index"], "6")
        self.assertEqual(cropped.observations[0]["source_sample_index"], "2")
        self.assertEqual(cropped.observations[-1]["source_sample_index"], "6")
        self.assertEqual(cropped.frames[0]["source_frame_index"], "2")
        self.assertEqual(cropped.frames[-1]["source_frame_index"], "6")
        self.assertEqual(float(cropped.frames[0]["timestamp_s"]), 0.0)
        self.assertEqual(
            float(cropped.observations[0]["observation_timestamp_s"]),
            0.0,
        )
        self.assertEqual(float(cropped.observations[0]["joint_1.pos"]), 102.0)
        self.assertEqual(float(cropped.actions[0]["frame_timestamp_s"]), 0.0)
        self.assertAlmostEqual(
            float(cropped.actions[0]["sample_timestamp_s"]),
            0.01,
            places=5,
        )
        self.assertEqual(cropped.meta["action_samples"], 5)
        self.assertEqual(cropped.meta["observation_samples"], 5)
        self.assertEqual(cropped.meta["video_frames"], 5)
        self.assertEqual(
            cropped.meta["temporal_crop"]["source_first_frame_index"],
            2,
        )

    def test_legacy_episode_is_cropped_by_independent_timestamps(self) -> None:
        source = self._make_episode("episode_legacy", action_count=8, frame_count=12)
        episode = load_episode(source)
        self.assertFalse(episode.aligned)

        result = crop_episode(
            episode,
            start_s=0.10,
            end_s=0.30,
            output_root=self.root / "processed",
            output_name="legacy_crop",
        )
        cropped = load_episode(result.output_path)

        self.assertFalse(result.aligned)
        self.assertFalse(cropped.aligned)
        self.assertEqual(result.action_samples, 3)
        self.assertEqual(result.video_frames, 5)
        self.assertIn("legacy", cropped.meta["alignment"])

    def test_preview_uses_the_exact_same_frame_range_as_save(self) -> None:
        source = self._make_episode("episode_preview", action_count=12, frame_count=12)
        episode = load_episode(source)

        preview = create_preview_clip(
            episode,
            start_s=0.10,
            end_s=0.31,
            preview_root=self.root / "previews",
        )
        crop = crop_episode(
            episode,
            start_s=0.10,
            end_s=0.31,
            output_root=self.root / "processed",
            output_name="previewed_crop",
        )

        self.assertTrue(preview.video_path.is_file())
        self.assertEqual(preview.action_samples, crop.action_samples)
        self.assertEqual(preview.video_frames, crop.video_frames)
        self.assertEqual(preview.video_frames, 5)

    def test_discovery_ignores_incomplete_and_pending_directories(self) -> None:
        complete = self._make_episode("episode_001", action_count=4, frame_count=4)
        (self.root / "episode_incomplete").mkdir()
        pending = self._make_episode(".episode_002.pending", action_count=4, frame_count=4)

        self.assertEqual(discover_episodes(self.root), [complete])
        self.assertTrue(pending.is_dir())

    def _make_episode(
        self,
        name: str,
        *,
        action_count: int,
        frame_count: int,
    ) -> Path:
        path = self.root / name
        path.mkdir()
        video_path = path / "video.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (64, 48),
        )
        self.assertTrue(writer.isOpened())
        for index in range(frame_count):
            writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
        writer.release()

        aligned = action_count == frame_count
        action_rows = []
        for index in range(action_count):
            sample_time = (
                index * 0.05 + 0.01 if aligned else index * 0.075
            )
            row = {
                "sample_index": index,
                "sample_timestamp_s": round(sample_time, 6),
                "source_control_sequence": 100 + index,
                "source_control_timestamp_s": round(sample_time - 0.002, 6),
                "joint_1.pos": float(index),
            }
            if aligned:
                row.update(
                    {
                        "frame_index": index,
                        "capture_sequence": 1000 + index,
                        "frame_timestamp_s": round(index * 0.05, 6),
                        "action_frame_delta_ms": 8.0,
                    }
                )
            action_rows.append(row)

        frame_rows = [
            {
                "frame_index": index,
                "capture_sequence": 1000 + index,
                "timestamp_s": round(index * 0.05, 6),
            }
            for index in range(frame_count)
        ]
        self._write_csv(path / "actions.csv", action_rows)
        if aligned:
            observation_rows = [
                {
                    "sample_index": index,
                    "frame_index": index,
                    "source_control_sequence": 100 + index,
                    "observation_timestamp_s": round(index * 0.05, 6),
                    "source_control_timestamp_s": action_rows[index][
                        "source_control_timestamp_s"
                    ],
                    "observation_action_delta_ms": 8.0,
                    "joint_1.pos": 100.0 + index,
                }
                for index in range(action_count)
            ]
            self._write_csv(path / "observations.csv", observation_rows)
        self._write_csv(path / "frame_timestamps.csv", frame_rows)
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "task": "test",
                    "duration_s": max(
                        float(action_rows[-1]["sample_timestamp_s"]),
                        float(frame_rows[-1]["timestamp_s"]),
                    ),
                    "action_samples": action_count,
                    "observation_samples": action_count if aligned else 0,
                    "video_frames": frame_count,
                    "video_fps_target": 20.0,
                }
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
