from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from policy.checkpoints import discover_runs, load_run, resolve_checkpoint, summarise_log


def write_run(
    root: Path,
    name: str,
    *,
    tags: tuple[str, ...] = ("best", "final", "step_005000", "step_010000"),
    metadata: dict | None = None,
    log: list[dict] | None = None,
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    for tag in tags:
        (directory / f"{tag}.pt").write_bytes(b"x" * 1024)
    if metadata is not None:
        (directory / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    if log is not None:
        (directory / "log.jsonl").write_text(
            "\n".join(json.dumps(record) for record in log) + "\n", encoding="utf-8"
        )
    return directory


METADATA = {
    "run_name": "C_bc_diffusion_c8_perc_d44",
    "config": {
        "policy": "bc",
        "objective": "diffusion",
        "chunk_size": 8,
        "image_size": 224,
    },
    "bundle_manifest": {"name": "pick-bar", "fps": 20.0, "tasks": ["pick up bar"]},
}


class CheckpointDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_lists_runs_and_orders_tags(self) -> None:
        write_run(self.root, "B_run", metadata=METADATA)
        write_run(self.root, "A_run", metadata=METADATA)
        runs = discover_runs(self.root)
        self.assertEqual([run.name for run in runs], ["A_run", "B_run"])
        # best and final first, then snapshots oldest to newest.
        self.assertEqual(
            [entry.tag for entry in runs[0].checkpoints],
            ["best", "final", "step_005000", "step_010000"],
        )
        self.assertEqual(runs[0].preferred_checkpoint().tag, "best")
        self.assertEqual(runs[0].checkpoint("step_010000").step, 10000)
        self.assertIsNone(runs[0].checkpoint("best").step)

    def test_reads_metadata_and_log(self) -> None:
        write_run(
            self.root,
            "run",
            metadata=METADATA,
            log=[
                {"step": 250, "val_loss": 0.5, "val_mae": 2.0},
                {"step": 500, "val_loss": 0.2, "val_mae": 1.0},
                {"step": 750, "loss": 0.1},
            ],
        )
        run = discover_runs(self.root)[0]
        self.assertEqual(run.policy, "bc")
        self.assertEqual(run.objective, "diffusion")
        self.assertEqual(run.chunk_size, 8)
        self.assertEqual(run.bundle, "pick-bar")
        self.assertEqual(run.tasks, ("pick up bar",))
        self.assertEqual(run.fps, 20.0)
        self.assertAlmostEqual(run.best_val_loss or 0.0, 0.2)
        self.assertAlmostEqual(run.best_val_mae or 0.0, 1.0)
        self.assertEqual(run.last_step, 750)
        self.assertIn("val 0.2", run.summary())

    def test_run_without_metadata_still_lists(self) -> None:
        write_run(self.root, "bare", tags=("final",))
        run = discover_runs(self.root)[0]
        self.assertEqual(run.policy, "?")
        self.assertEqual(run.bundle, "?")
        self.assertIsNone(run.chunk_size)
        self.assertEqual(run.preferred_checkpoint().tag, "final")

    def test_the_policy_kind_is_reported_verbatim(self) -> None:
        # The picker refuses runs whose kind it cannot build, so this field has
        # to survive unchanged from run.json.
        metadata = {**METADATA, "config": {**METADATA["config"], "policy": "simple"}}
        write_run(self.root, "simple_run", metadata=metadata)
        self.assertEqual(discover_runs(self.root)[0].policy, "simple")

    def test_directory_without_checkpoints_is_skipped(self) -> None:
        (self.root / "empty").mkdir()
        write_run(self.root, "real", metadata=METADATA)
        self.assertEqual([run.name for run in discover_runs(self.root)], ["real"])
        self.assertIsNone(load_run(self.root / "empty"))

    def test_missing_root_returns_nothing(self) -> None:
        self.assertEqual(discover_runs(self.root / "nope"), [])

    def test_resolve_checkpoint_by_name_and_tag(self) -> None:
        write_run(self.root, "run_a", metadata=METADATA)
        entry = resolve_checkpoint(self.root, "run_a", "step_005000")
        self.assertEqual(entry.path, self.root / "run_a" / "step_005000.pt")
        self.assertEqual(entry.run_name, "run_a")
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(self.root, "run_a", "step_999999")
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(self.root, "missing_run")

    def test_corrupt_log_does_not_raise(self) -> None:
        directory = write_run(self.root, "run", metadata=METADATA)
        (directory / "log.jsonl").write_text("not json\n{\"step\": 10}\n", encoding="utf-8")
        self.assertEqual(summarise_log(directory), (None, None, 10))


if __name__ == "__main__":
    unittest.main()
