from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from policy.data_prep import (
    BUNDLE_SPECS,
    JOINT_NAMES,
    build_shard,
    discover_tasks,
    load_episode,
    preprocess_frame,
    resolve_sources,
    write_bundle,
)


def write_episode(root: Path, index: int, frames: int, task: str = "pick up can") -> None:
    root.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(root / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (640, 480)
    )
    for frame_index in range(frames):
        frame = np.full((480, 640, 3), (frame_index * 5) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    action_columns = [
        "sample_index", "frame_index", "sample_timestamp_s", "source_control_sequence",
        *JOINT_NAMES,
    ]
    observation_columns = [
        "sample_index", "frame_index", "source_control_sequence", "observation_timestamp_s",
        *JOINT_NAMES,
    ]
    with (root / "actions.csv").open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=action_columns)
        csv_writer.writeheader()
        for frame_index in range(frames):
            row = {
                "sample_index": frame_index,
                "frame_index": frame_index,
                "sample_timestamp_s": frame_index / 20.0,
                "source_control_sequence": 1000 + frame_index,
            }
            row.update({name: float(frame_index + position) for position, name in enumerate(JOINT_NAMES)})
            csv_writer.writerow(row)
    with (root / "observations.csv").open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=observation_columns)
        csv_writer.writeheader()
        for frame_index in range(frames):
            row = {
                "sample_index": frame_index,
                "frame_index": frame_index,
                "source_control_sequence": 1000 + frame_index,
                "observation_timestamp_s": frame_index / 20.0,
            }
            row.update({name: float(frame_index) for name in JOINT_NAMES})
            csv_writer.writerow(row)
    with (root / "frame_timestamps.csv").open("w", newline="") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(["frame_index", "capture_sequence", "timestamp_s"])
        for frame_index in range(frames):
            csv_writer.writerow([frame_index, frame_index, frame_index / 20.0])
    (root / "meta.json").write_text(
        json.dumps(
            {
                "task": task,
                "episode_index": index,
                "video_fps_target": 20.0,
                "video_frames": frames,
                "observation_samples": frames,
            }
        ),
        encoding="utf-8",
    )


class PreprocessFrameTests(unittest.TestCase):
    def test_center_crop_keeps_the_middle_square_and_converts_to_rgb(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :80] = (255, 0, 0)  # blue margin that the crop must discard
        frame[:, 560:] = (255, 0, 0)
        frame[:, 80:560] = (0, 0, 255)  # red center in BGR

        processed = preprocess_frame(frame, 224)

        self.assertEqual(processed.shape, (224, 224, 3))
        self.assertGreater(int(processed[..., 0].mean()), 200)  # red channel first after RGB conversion
        self.assertLess(int(processed[..., 2].mean()), 55)

    def test_image_size_is_configurable(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(preprocess_frame(frame, 112).shape, (112, 112, 3))


class LoadEpisodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_returns_state_and_action_arrays(self) -> None:
        episode = self.root / "episode_000"
        write_episode(episode, 0, frames=6)

        state, action, meta = load_episode(episode)

        self.assertEqual(state.shape, (6, len(JOINT_NAMES)))
        self.assertEqual(action.shape, (6, len(JOINT_NAMES)))
        self.assertEqual(meta["task"], "pick up can")
        # observations are frame_index, actions are frame_index + column position
        np.testing.assert_allclose(action[3] - state[3], np.arange(len(JOINT_NAMES)))

    def test_rejects_row_count_mismatch(self) -> None:
        episode = self.root / "episode_000"
        write_episode(episode, 0, frames=6)
        rows = (episode / "actions.csv").read_text(encoding="utf-8").splitlines()
        (episode / "actions.csv").write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            load_episode(episode)

    def test_rejects_missing_observations(self) -> None:
        episode = self.root / "episode_000"
        write_episode(episode, 0, frames=4)
        (episode / "observations.csv").unlink()

        with self.assertRaises(FileNotFoundError):
            load_episode(episode)


class BuildShardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "raw" / "can"
        write_episode(self.source / "episode_000", 0, frames=5)
        write_episode(self.source / "episode_001", 1, frames=7)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_shard_concatenates_episodes_with_boundaries(self) -> None:
        shard = build_shard(self.source, self.root / "out" / "_shards" / "can", image_size=28, force=False)

        meta = json.loads((shard / "shard.json").read_text(encoding="utf-8"))
        frames = np.load(shard / "frames.npy", mmap_mode="r")
        self.assertEqual(meta["num_frames"], 12)
        self.assertEqual(frames.shape, (12, 28, 28, 3))
        self.assertEqual([(episode["start"], episode["end"]) for episode in meta["episodes"]], [(0, 5), (5, 12)])
        self.assertEqual(np.load(shard / "state.npy").shape, (12, len(JOINT_NAMES)))

    def test_rebuild_is_skipped_unless_forced(self) -> None:
        shard_dir = self.root / "out" / "_shards" / "can"
        build_shard(self.source, shard_dir, image_size=28, force=False)
        written = (shard_dir / "frames.npy").stat().st_mtime_ns
        build_shard(self.source, shard_dir, image_size=28, force=False)
        self.assertEqual((shard_dir / "frames.npy").stat().st_mtime_ns, written)

    def test_image_size_change_without_force_is_an_error(self) -> None:
        shard_dir = self.root / "out" / "_shards" / "can"
        build_shard(self.source, shard_dir, image_size=28, force=False)
        with self.assertRaises(ValueError):
            build_shard(self.source, shard_dir, image_size=56, force=False)

    def test_combined_bundle_references_both_shards(self) -> None:
        other = self.root / "raw" / "can-2"
        write_episode(other / "episode_000", 0, frames=4)
        output = self.root / "out"
        first = build_shard(self.source, output / "_shards" / "can", image_size=28, force=False)
        second = build_shard(other, output / "_shards" / "can-2", image_size=28, force=False)

        write_bundle(output / "can-all", "can-all", [first, second])

        manifest = json.loads((output / "can-all" / "bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["shards"], ["_shards/can", "_shards/can-2"])
        self.assertEqual(manifest["num_frames"], 16)
        self.assertEqual(manifest["num_episodes"], 3)
        self.assertEqual(len(manifest["stats"]["state"]["mean"]), len(JOINT_NAMES))


class BundleResolutionTests(unittest.TestCase):
    """Which raw directories a bundle name refers to.

    Bundles used to be a hardcoded table, which went stale the moment a
    recording session was named something new. They are discovered from disk
    now, so these tests cover the discovery instead of the table.
    """

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.raw = Path(self.temporary.name)
        for task in ("pick-can-m1", "pick-can-m2"):
            (self.raw / task / "episode_000").mkdir(parents=True)
        # Not a task: no episode_* inside, so it must not be offered as one.
        (self.raw / "notes").mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_only_directories_holding_episodes_count_as_tasks(self) -> None:
        self.assertEqual(discover_tasks(self.raw), ["pick-can-m1", "pick-can-m2"])

    def test_a_task_directory_needs_no_entry_in_the_table(self) -> None:
        # The whole point of dropping the table: a new session is usable the
        # moment it is recorded.
        tasks = discover_tasks(self.raw)
        self.assertNotIn("pick-can-m1", BUNDLE_SPECS)
        self.assertEqual(resolve_sources("pick-can-m1", self.raw, tasks), ("pick-can-m1",))

    def test_the_table_still_defines_bundles_that_merge_sessions(self) -> None:
        tasks = discover_tasks(self.raw)
        self.assertEqual(
            resolve_sources("pick-can-all", self.raw, tasks), ("pick-can-m1", "pick-can-m2")
        )

    def test_a_missing_source_fails_before_anything_is_decoded(self) -> None:
        # Failing late - after decoding several gigabytes of the other shards -
        # is the failure mode this check exists to prevent.
        tasks = discover_tasks(self.raw)
        with self.assertRaises(SystemExit) as error:
            resolve_sources("pick-scissors", self.raw, tasks)
        self.assertIn("pick-can-m1", str(error.exception))  # lists what is available

    def test_an_empty_raw_root_is_reported_rather_than_silently_empty(self) -> None:
        with self.assertRaises(SystemExit):
            discover_tasks(self.raw / "does-not-exist")


if __name__ == "__main__":
    unittest.main()
