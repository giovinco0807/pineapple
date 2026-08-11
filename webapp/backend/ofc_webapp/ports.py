"""Dependency ports for AI, scoring, and durable backup adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ofc_regular.hu_infoset import ScoringContext
from ofc_regular.state import Board

from .domain import ActionSubmission, DecisionObservation, FinalScore


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AIMetadata:
    """Audit metadata required for every AI decision."""

    evaluator: str
    weights_sha: tuple[str, ...]
    assembly_sha: str
    scores_topk: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.evaluator.strip():
            raise ValueError("AI evaluator name is required")
        if not self.weights_sha:
            raise ValueError("at least one AI weight SHA is required")
        for digest in (*self.weights_sha, self.assembly_sha):
            if not _SHA256_RE.fullmatch(digest):
                raise ValueError("AI metadata SHA values must be lowercase sha256")
        if len(self.scores_topk) != 3:
            raise ValueError("AI metadata must record exactly the top three scores")
        for candidate in self.scores_topk:
            if "score" not in candidate:
                raise ValueError("each top-k candidate requires a score")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "weights_sha": list(self.weights_sha),
            "assembly_sha": self.assembly_sha,
            "scores_topk": [dict(candidate) for candidate in self.scores_topk],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AIMetadata":
        raw_weights = payload.get("weights_sha", ())
        if isinstance(raw_weights, str):
            weights = (raw_weights,)
        elif isinstance(raw_weights, Sequence):
            weights = tuple(str(value) for value in raw_weights)
        else:
            raise ValueError("weights_sha must be a SHA or a sequence of SHAs")
        raw_topk = payload.get("scores_topk", ())
        if not isinstance(raw_topk, Sequence):
            raise ValueError("scores_topk must be a sequence")
        return cls(
            evaluator=str(payload.get("evaluator", "")),
            weights_sha=weights,
            assembly_sha=str(payload.get("assembly_sha", "")),
            scores_topk=tuple(dict(candidate) for candidate in raw_topk),
        )


@dataclass(frozen=True)
class AIDecision:
    action: ActionSubmission
    think_ms: int
    meta: AIMetadata

    def __post_init__(self) -> None:
        if (
            isinstance(self.think_ms, bool)
            or not isinstance(self.think_ms, int)
            or self.think_ms < 0
        ):
            raise ValueError("think_ms must be a non-negative integer")


@runtime_checkable
class AIDecisionPort(Protocol):
    """Select a normal or FL action from an information-safe observation."""

    def decide(self, observation: DecisionObservation) -> AIDecision:
        ...


@runtime_checkable
class FinalScorePort(Protocol):
    """Call the authoritative ``score_final`` implementation.

    The adapter is responsible for translating the native response into
    ``FinalScore``, including regular 14-card FL entry/stay flags.
    """

    def score_final(
        self,
        *,
        first_board: Board,
        second_board: Board,
        scoring: ScoringContext,
        first_in_fantasyland: bool,
        second_in_fantasyland: bool,
    ) -> FinalScore:
        ...


@runtime_checkable
class DatabaseBackupPort(Protocol):
    """Persist or restore the SQLite file outside ephemeral storage."""

    def restore(self, destination: Path) -> bool:
        """Restore into destination, returning whether a backup existed."""
        ...

    def backup(self, source: Path) -> None:
        ...


__all__ = [
    "AIDecision",
    "AIDecisionPort",
    "AIMetadata",
    "DatabaseBackupPort",
    "FinalScorePort",
]
