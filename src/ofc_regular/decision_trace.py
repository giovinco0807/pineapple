"""Post-decision replay-truth attachment for policy decision logs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hu_infoset import ReplayTruth


DECISION_LOG_ATTRIBUTES = (
    "hu_turn0_decision_log",
    "hu_turn1_decision_log",
    "hu_turn2_decision_log",
    "hu_turn3_decision_log",
    "hu_t4_decision_log",
    "topk_decision_log",
)
_ACTOR_VISIBLE_RECORD_FIELDS = frozenset(
    {
        "dead_cards",
        "visible_dead_cards",
        "hero_private_discards",
        "policy_observation",
    }
)


@dataclass(frozen=True)
class DecisionLogPosition:
    records: list[dict[str, Any]]
    start: int


def capture_decision_log_positions(policy: object) -> tuple[DecisionLogPosition, ...]:
    """Capture every unique in-memory decision log before policy execution."""
    positions: list[DecisionLogPosition] = []
    seen: set[int] = set()
    for attribute in DECISION_LOG_ATTRIBUTES:
        records = getattr(policy, attribute, None)
        if not isinstance(records, list) or id(records) in seen:
            continue
        seen.add(id(records))
        positions.append(DecisionLogPosition(records=records, start=len(records)))
    return tuple(positions)


def attach_replay_truth(
    positions: tuple[DecisionLogPosition, ...],
    truth: ReplayTruth,
) -> int:
    """Attach hidden state only to records created after the policy returned."""
    fields = truth.to_legacy_record_fields()
    unsafe = _ACTOR_VISIBLE_RECORD_FIELDS.intersection(fields)
    if unsafe:
        raise RuntimeError(
            "replay attachment attempted to overwrite actor-visible fields: "
            + ", ".join(sorted(unsafe))
        )
    attached = 0
    for position in positions:
        for record in position.records[position.start :]:
            if not isinstance(record, dict):
                raise TypeError("decision log entries must be dictionaries")
            record.update(fields)
            attached += 1
    return attached
