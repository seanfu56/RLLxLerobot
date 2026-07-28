"""Shared start/stop command controller for button and voice recording input."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

try:
    from ui.teleop_runtime import RecordingConfig, RuntimeState
except ModuleNotFoundError as exc:
    if exc.name != "ui":
        raise
    from teleop_runtime import RecordingConfig, RuntimeState


@dataclass(frozen=True)
class CommandOutcome:
    accepted: bool
    action: str
    message: str


@dataclass(frozen=True)
class PendingEpisodeStart:
    config: RecordingConfig
    requested_at_s: float
    deadline_s: float
    discard_draft: bool
    source: str


class RecordingCommandController:
    """Serialize recording commands and maintain a non-blocking countdown."""

    def __init__(self) -> None:
        self.pending: PendingEpisodeStart | None = None
        self.last_outcome: CommandOutcome | None = None
        self.last_outcome_at_s = 0.0

    def request_start(
        self,
        runtime: Any,
        config: RecordingConfig,
        delay_s: float,
        *,
        source: str,
        discard_draft: bool = False,
        now_s: float | None = None,
    ) -> CommandOutcome:
        now = time.perf_counter() if now_s is None else now_s
        if self.pending is not None:
            return self._record(
                CommandOutcome(
                    False,
                    "start_ignored",
                    "A recording countdown is already active.",
                ),
                now,
            )

        snapshot = runtime.get_snapshot()
        if snapshot.state is not RuntimeState.IDLE:
            return self._record(
                CommandOutcome(
                    False,
                    "start_ignored",
                    f"Start ignored while runtime is {snapshot.state.value}.",
                ),
                now,
            )
        if snapshot.draft is not None and not discard_draft:
            return self._record(
                CommandOutcome(
                    False,
                    "start_ignored",
                    "Save, discard, or re-record the completed draft first.",
                ),
                now,
            )

        delay = max(float(delay_s), 0.0)
        self.pending = PendingEpisodeStart(
            config=config,
            requested_at_s=now,
            deadline_s=now + delay,
            discard_draft=discard_draft,
            source=source,
        )
        return self._record(
            CommandOutcome(
                True,
                "start_scheduled",
                f"{source.capitalize()} start accepted; recording begins in "
                f"{math.ceil(delay):d} second(s).",
            ),
            now,
        )

    def request_stop(
        self,
        runtime: Any,
        *,
        source: str,
        now_s: float | None = None,
    ) -> CommandOutcome:
        now = time.perf_counter() if now_s is None else now_s
        if self.pending is not None:
            self.pending = None
            return self._record(
                CommandOutcome(
                    True,
                    "countdown_cancelled",
                    f"{source.capitalize()} stop cancelled the pending recording.",
                ),
                now,
            )

        snapshot = runtime.get_snapshot()
        if snapshot.state is RuntimeState.RECORDING:
            runtime.stop_episode()
            return self._record(
                CommandOutcome(
                    True,
                    "recording_stopped",
                    f"{source.capitalize()} stop ended the episode.",
                ),
                now,
            )
        return self._record(
            CommandOutcome(
                False,
                "stop_ignored",
                f"Stop ignored while runtime is {snapshot.state.value}.",
            ),
            now,
        )

    def advance(
        self,
        runtime: Any,
        *,
        now_s: float | None = None,
    ) -> CommandOutcome | None:
        pending = self.pending
        if pending is None:
            return None
        now = time.perf_counter() if now_s is None else now_s
        if now < pending.deadline_s:
            return None

        self.pending = None
        snapshot = runtime.get_snapshot()
        if snapshot.state is not RuntimeState.IDLE:
            return self._record(
                CommandOutcome(
                    False,
                    "start_cancelled",
                    f"Pending start cancelled because runtime is {snapshot.state.value}.",
                ),
                now,
            )
        if pending.discard_draft:
            runtime.discard_episode()
        elif snapshot.draft is not None:
            return self._record(
                CommandOutcome(
                    False,
                    "start_cancelled",
                    "Pending start cancelled because a completed draft is present.",
                ),
                now,
            )

        runtime.start_episode(pending.config)
        return self._record(
            CommandOutcome(
                True,
                "recording_started",
                f"Recording started from {pending.source} command.",
            ),
            now,
        )

    def remaining_seconds(self, now_s: float | None = None) -> float:
        pending = self.pending
        if pending is None:
            return 0.0
        now = time.perf_counter() if now_s is None else now_s
        return max(pending.deadline_s - now, 0.0)

    def recent_message(
        self,
        *,
        max_age_s: float = 6.0,
        now_s: float | None = None,
    ) -> str | None:
        if self.last_outcome is None:
            return None
        now = time.perf_counter() if now_s is None else now_s
        if now - self.last_outcome_at_s > max_age_s:
            return None
        return self.last_outcome.message

    def reject(
        self,
        action: str,
        message: str,
        *,
        now_s: float | None = None,
    ) -> CommandOutcome:
        now = time.perf_counter() if now_s is None else now_s
        return self._record(CommandOutcome(False, action, message), now)

    def _record(self, outcome: CommandOutcome, now_s: float) -> CommandOutcome:
        self.last_outcome = outcome
        self.last_outcome_at_s = now_s
        return outcome
