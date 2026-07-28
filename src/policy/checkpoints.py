"""Discover trained runs and their checkpoints without loading any weights.

``train.py`` writes one directory per run:

    models/sweeps/<run_name>/
        run.json        configuration, bundle manifest, episode split
        log.jsonl       one record per log/eval step
        best.pt         lowest validation loss so far
        final.pt        last step
        step_005000.pt  periodic snapshots

Everything a checkpoint picker needs is in ``run.json`` and ``log.jsonl``, so
this module never imports torch. That matters for the Streamlit page: it lists
checkpoints in the browser process while only the hardware process pays for a
CUDA context.

    for run in discover_runs(Path("models/sweeps")):
        print(run.name, run.summary(), [entry.tag for entry in run.checkpoints])
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL_ROOT = Path("models/sweeps")

# Ordering of the tag list offered to an operator: the two checkpoints worth
# running first, then periodic snapshots oldest to newest.
_TAG_PRIORITY = {"best": 0, "final": 1}


def _tag_of(path: Path) -> str:
    return path.stem


def _step_of(tag: str) -> int | None:
    if not tag.startswith("step_"):
        return None
    try:
        return int(tag.split("_", maxsplit=1)[1])
    except ValueError:
        return None


@dataclass(frozen=True)
class CheckpointInfo:
    """One ``.pt`` file inside a run directory."""

    path: Path
    run_name: str
    tag: str
    step: int | None
    size_bytes: int
    modified_s: float

    @property
    def label(self) -> str:
        size_mb = self.size_bytes / (1024 * 1024)
        if self.step is not None:
            return f"{self.tag} (step {self.step}, {size_mb:.0f} MB)"
        return f"{self.tag} ({size_mb:.0f} MB)"


@dataclass(frozen=True)
class RunInfo:
    """A run directory: its checkpoints plus whatever training recorded."""

    name: str
    directory: Path
    checkpoints: tuple[CheckpointInfo, ...]
    metadata: Mapping = field(default_factory=dict)
    best_val_loss: float | None = None
    best_val_mae: float | None = None
    last_step: int | None = None

    # ---- fields the picker shows; absent metadata must not break listing ----

    @property
    def config(self) -> Mapping:
        config = self.metadata.get("config")
        return config if isinstance(config, Mapping) else {}

    @property
    def policy(self) -> str:
        return str(self.config.get("policy", "?"))

    @property
    def objective(self) -> str:
        return str(self.config.get("objective", "?"))

    @property
    def chunk_size(self) -> int | None:
        value = self.config.get("chunk_size")
        return int(value) if isinstance(value, int) else None

    @property
    def image_size(self) -> int | None:
        value = self.config.get("image_size")
        return int(value) if isinstance(value, int) else None

    @property
    def bundle(self) -> str:
        manifest = self.metadata.get("bundle_manifest")
        if isinstance(manifest, Mapping) and manifest.get("name"):
            return str(manifest["name"])
        return str(self.metadata.get("bundle", "?"))

    @property
    def tasks(self) -> tuple[str, ...]:
        manifest = self.metadata.get("bundle_manifest")
        if isinstance(manifest, Mapping) and isinstance(manifest.get("tasks"), list):
            return tuple(str(task) for task in manifest["tasks"])
        return ()

    @property
    def fps(self) -> float | None:
        manifest = self.metadata.get("bundle_manifest")
        if isinstance(manifest, Mapping) and manifest.get("fps") is not None:
            return float(manifest["fps"])
        return None

    def checkpoint(self, tag: str) -> CheckpointInfo:
        for entry in self.checkpoints:
            if entry.tag == tag:
                return entry
        available = ", ".join(entry.tag for entry in self.checkpoints) or "none"
        raise FileNotFoundError(f"{self.name} has no checkpoint {tag!r}; available: {available}")

    def preferred_checkpoint(self) -> CheckpointInfo:
        """``best.pt`` when training evaluated, otherwise the newest snapshot."""
        if not self.checkpoints:
            raise FileNotFoundError(f"{self.name} contains no checkpoints")
        return self.checkpoints[0]

    def summary(self) -> str:
        parts = [self.policy, self.objective]
        if self.chunk_size is not None:
            parts.append(f"chunk {self.chunk_size}")
        parts.append(self.bundle)
        if self.best_val_loss is not None:
            parts.append(f"val {self.best_val_loss:.4f}")
        return " · ".join(parts)


def read_run_metadata(directory: Path) -> Mapping:
    path = directory / "run.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def summarise_log(directory: Path) -> tuple[float | None, float | None, int | None]:
    """Best ``val_loss`` with its MAE, and the last step training reached.

    Returned as a tuple rather than parsed lazily because the picker shows all
    three at once, and the file is a few hundred lines.
    """
    path = directory / "log.jsonl"
    if not path.is_file():
        return None, None, None
    best_loss: float | None = None
    best_mae: float | None = None
    last_step: int | None = None
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = record.get("step")
                if isinstance(step, int):
                    last_step = step if last_step is None else max(last_step, step)
                value = record.get("val_loss")
                if isinstance(value, (int, float)) and (best_loss is None or value < best_loss):
                    best_loss = float(value)
                    mae = record.get("val_mae")
                    best_mae = float(mae) if isinstance(mae, (int, float)) else None
    except OSError:
        return best_loss, best_mae, last_step
    return best_loss, best_mae, last_step


def load_run(directory: Path) -> RunInfo | None:
    """Read one run directory. ``None`` when it holds no checkpoint."""
    directory = Path(directory)
    if not directory.is_dir():
        return None

    entries: list[CheckpointInfo] = []
    for path in sorted(directory.glob("*.pt")):
        try:
            stat = path.stat()
        except OSError:
            continue
        tag = _tag_of(path)
        entries.append(
            CheckpointInfo(
                path=path,
                run_name=directory.name,
                tag=tag,
                step=_step_of(tag),
                size_bytes=stat.st_size,
                modified_s=stat.st_mtime,
            )
        )
    if not entries:
        return None

    entries.sort(key=lambda entry: (_TAG_PRIORITY.get(entry.tag, 2), entry.step or 0, entry.tag))
    best_loss, best_mae, last_step = summarise_log(directory)
    return RunInfo(
        name=directory.name,
        directory=directory,
        checkpoints=tuple(entries),
        metadata=read_run_metadata(directory),
        best_val_loss=best_loss,
        best_val_mae=best_mae,
        last_step=last_step,
    )


def discover_runs(root: Path = DEFAULT_MODEL_ROOT) -> list[RunInfo]:
    """Every run directly under ``root`` that contains at least one checkpoint."""
    root = Path(root).expanduser()
    if not root.is_dir():
        return []
    runs = [run for path in sorted(root.iterdir()) if (run := load_run(path)) is not None]
    return runs


def find_run(runs: list[RunInfo], name: str) -> RunInfo:
    for run in runs:
        if run.name == name:
            return run
    raise FileNotFoundError(f"No run named {name!r}")


def resolve_checkpoint(
    root: Path = DEFAULT_MODEL_ROOT,
    run_name: str | None = None,
    tag: str = "best",
) -> CheckpointInfo:
    """Locate one checkpoint by run and tag, for CLI-style callers."""
    runs = discover_runs(root)
    if not runs:
        raise FileNotFoundError(f"No runs with checkpoints under {root}")
    run = find_run(runs, run_name) if run_name else runs[0]
    return run.checkpoint(tag)
