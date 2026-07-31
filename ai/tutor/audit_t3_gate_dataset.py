"""Audit an independent T3 precision-gate dataset for leakage and coverage.

The branch-expanded generator emits several correlated decisions per root.  A
precision gate is only useful when those decisions are unique, balanced across
positions, and absent from earlier teacher/holdout inputs.  This module creates
a compact, reproducible JSON audit before exact labels are generated or used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


JOKER_ALIASES = {"X", "X1", "X2", "JK", "Xj"}
ROW_ALIASES = {"mid": "middle", "bot": "bottom"}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected object in {path}:{line_number}")
            yield record


def _decision(record: dict[str, Any]) -> dict[str, Any]:
    state = record.get("state")
    return state if isinstance(state, dict) else record


def _cards(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(sorted(str(value) for value in values))


def _board(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        value = {}
    normalized: dict[str, tuple[str, ...]] = {}
    for source, target in (("top", "top"), ("middle", "middle"), ("mid", "middle"),
                           ("bottom", "bottom"), ("bot", "bottom")):
        if source in value:
            normalized[target] = _cards(value.get(source))
    return {row: normalized.get(row, ()) for row in ("top", "middle", "bottom")}


def decision_signature(record: dict[str, Any]) -> str:
    """Return an order-insensitive hash of fields that define a T3 decision."""
    decision = _decision(record)
    position = decision.get("position", record.get("position"))
    if position is None:
        is_btn = decision.get("is_btn", record.get("is_btn"))
        position = "btn" if bool(is_btn) else "bb"
    payload = {
        "turn": int(decision.get("turn", record.get("turn", -1))),
        "position": str(position).lower(),
        "board": _board(decision.get("board", decision.get("board_self"))),
        "opponent_board": _board(
            decision.get("opponent_board", decision.get("board_opponent"))
        ),
        "dealt": _cards(decision.get("dealt", decision.get("dealt_cards"))),
        "known_discards": _cards(
            decision.get("known_discards", decision.get("known_discards_self"))
        ),
        "exclude": _cards(decision.get("exclude")),
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _known_cards(record: dict[str, Any]) -> set[str]:
    decision = _decision(record)
    known: set[str] = set()
    for board_name in ("board", "opponent_board", "board_self", "board_opponent"):
        board = decision.get(board_name)
        if not isinstance(board, dict):
            continue
        for row in ("top", "middle", "mid", "bottom", "bot"):
            known.update(_cards(board.get(row)))
    for field in ("dealt", "dealt_cards", "known_discards", "known_discards_self", "exclude"):
        known.update(_cards(decision.get(field)))
    return known


def _joker_count(record: dict[str, Any]) -> int:
    jokers = {
        card for card in _known_cards(record)
        if card in JOKER_ALIASES or card.upper().startswith("X")
    }
    named = {card.upper() for card in jokers if card.upper() in {"X1", "X2"}}
    generic = bool({card.upper() for card in jokers} - named)
    return min(2, len(named) + int(generic))


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def load_signatures(path: Path) -> tuple[set[str], int]:
    signatures: set[str] = set()
    count = 0
    for record in iter_jsonl(path):
        signatures.add(decision_signature(record))
        count += 1
    return signatures, count


def audit_dataset(input_path: Path, compare_paths: Iterable[Path] = ()) -> dict[str, Any]:
    signatures: set[str] = set()
    duplicate_signatures: Counter[str] = Counter()
    positions: Counter[str] = Counter()
    roots: Counter[int] = Counter()
    roots_by_position: Counter[str] = Counter()
    visible_joker = 0
    future_joker_reachable = 0
    total = 0

    for record in iter_jsonl(input_path):
        total += 1
        signature = decision_signature(record)
        if signature in signatures:
            duplicate_signatures[signature] += 1
        signatures.add(signature)

        decision = _decision(record)
        position = str(
            decision.get("position", record.get("position", "unknown"))
        ).lower()
        positions[position] += 1

        branch = record.get("branch")
        if isinstance(branch, dict) and branch.get("root_index") is not None:
            root = int(branch["root_index"])
            roots[root] += 1
            roots_by_position[f"{root}:{position}"] += 1

        known_jokers = _joker_count(record)
        turn = int(decision.get("turn", record.get("turn", -1)))
        visible_joker += int(known_jokers > 0)
        future_joker_reachable += int(0 <= turn < 4 and known_jokers < 2)

    comparisons: list[dict[str, Any]] = []
    prior_union: set[str] = set()
    for compare_path in compare_paths:
        compare_signatures, compare_count = load_signatures(compare_path)
        overlap = signatures & compare_signatures
        prior_union.update(compare_signatures)
        comparisons.append(
            {
                "path": str(compare_path),
                "records": compare_count,
                "unique_decisions": len(compare_signatures),
                "overlap_decisions": len(overlap),
                "overlap_rate": len(overlap) / len(signatures) if signatures else 0.0,
            }
        )

    root_values = sorted(roots.values())
    duplicate_occurrences = sum(duplicate_signatures.values())
    result = {
        "input": str(input_path),
        "records": total,
        "unique_decisions": len(signatures),
        "duplicate_occurrences": duplicate_occurrences,
        "duplicate_signature_count": len(duplicate_signatures),
        "positions": _counter_dict(positions),
        "jokers": {
            "visible_records": visible_joker,
            "visible_rate": visible_joker / total if total else 0.0,
            "future_reachable_records": future_joker_reachable,
            "future_reachable_rate": future_joker_reachable / total if total else 0.0,
        },
        "roots": {
            "count": len(roots),
            "min_index": min(roots) if roots else None,
            "max_index": max(roots) if roots else None,
            "min_records": min(root_values) if root_values else 0,
            "max_records": max(root_values) if root_values else 0,
            "records_by_root": _counter_dict(roots),
            "records_by_root_position": _counter_dict(roots_by_position),
        },
        "comparisons": comparisons,
        "prior_union_unique_decisions": len(prior_union),
        "prior_union_overlap_decisions": len(signatures & prior_union),
        "prior_union_overlap_rate": (
            len(signatures & prior_union) / len(signatures) if signatures else 0.0
        ),
        "gate_checks": {
            "nonempty": total > 0,
            "all_decisions_unique": total == len(signatures),
            "both_positions_present": positions.get("bb", 0) > 0 and positions.get("btn", 0) > 0,
            "no_prior_overlap": not bool(signatures & prior_union),
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--compare", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="exit nonzero unless decisions are unique, balanced, and have no prior overlap",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_dataset(args.input, args.compare)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.require_clean and not all(result["gate_checks"].values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
