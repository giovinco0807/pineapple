"""Fail-closed audit for a complete HU Turn1 all-action teacher artifact."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--expected-seat", choices=("first", "second"), required=True)
    parser.add_argument("--future-samples", type=int, required=True)
    parser.add_argument("--profile", default="stage9f_p2")
    parser.add_argument("--opponent-profile", default="stage9f_p2")
    parser.add_argument("--t3-continuation", default="stage7_m5_r10")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"record at line {line_number} is not an object")
            yield row


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def board_cards(board: Any) -> list[str]:
    if not isinstance(board, dict):
        return []
    return [
        str(card)
        for row in ("top", "middle", "bottom")
        for card in (board.get(row) or ())
    ]


def canonical_action(action: dict[str, Any]) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    placements: list[tuple[str, str]] = []
    for item in action.get("placements", ()) or ():
        if isinstance(item, str):
            card, row = item.split(maxsplit=1)
        else:
            card, row = item
        placements.append((str(card), str(row)))
    return tuple(sorted(placements)), tuple(sorted(str(card) for card in action.get("discards", ()) or ()))


def audit_records(
    rows: Iterable[dict[str, Any]],
    *,
    expected_records: int,
    expected_seat: str,
    future_samples: int,
    profile: str,
    opponent_profile: str,
    t3_continuation: str,
) -> dict[str, Any]:
    errors: Counter[str] = Counter()
    sample_ids: set[str] = set()
    state_ids: set[tuple[str, str]] = set()
    records = 0
    total_actions = 0
    action_counts: list[int] = []

    for row in rows:
        records += 1
        sample_id = str(row.get("sample_id", ""))
        state_id = (str(row.get("hand_seed", "")), str(row.get("player", "")))
        if not sample_id or sample_id in sample_ids:
            errors["duplicate_or_missing_sample_id"] += 1
        sample_ids.add(sample_id)
        if not all(state_id) or state_id in state_ids:
            errors["duplicate_or_missing_state_id"] += 1
        state_ids.add(state_id)

        if row.get("seat") != expected_seat:
            errors["wrong_seat"] += 1
        if expected_seat == "first" and int(row.get("player", -1)) != 0:
            errors["wrong_player"] += 1
        if row.get("profile") != profile:
            errors["wrong_profile"] += 1
        if row.get("opponent_profile") != opponent_profile:
            errors["wrong_opponent_profile"] += 1
        if row.get("t2_continuation_profile") != profile:
            errors["wrong_t2_continuation"] += 1
        if row.get("t3_continuation") != t3_continuation:
            errors["wrong_t3_continuation"] += 1
        if int(row.get("future_samples", -1)) != future_samples:
            errors["wrong_future_samples"] += 1
        if bool(row.get("actions_truncated")):
            errors["actions_truncated"] += 1

        dead = sorted(str(card) for card in row.get("dead_cards", ()) or ())
        visible = sorted(str(card) for card in row.get("visible_dead_cards", ()) or ())
        opponent_cards = sorted(board_cards(row.get("opponent_board")))
        if dead != visible:
            errors["dead_cards_visibility_mismatch"] += 1
        if expected_seat == "first" and visible != opponent_cards:
            errors["first_seat_visible_cards_mismatch"] += 1
        if expected_seat == "first" and (row.get("true_dead_cards") or ()):
            errors["unexpected_first_seat_private_discards"] += 1

        actions = row.get("actions")
        if not isinstance(actions, list) or not actions:
            errors["missing_actions"] += 1
            continue
        action_count = len(actions)
        action_counts.append(action_count)
        total_actions += action_count
        declared = (
            int(row.get("action_count", -1)),
            int(row.get("evaluated_action_count", -1)),
            int(row.get("total_legal_actions", -1)),
        )
        if declared != (action_count, action_count, action_count):
            errors["all_action_count_mismatch"] += 1

        original_indices: set[int] = set()
        action_keys: set[tuple[tuple[tuple[str, str], ...], tuple[str, ...]]] = set()
        scores: list[float] = []
        for action in actions:
            if not isinstance(action, dict):
                errors["invalid_action_object"] += 1
                continue
            if not all(finite(action.get(key)) for key in ("score", "ev", "se")):
                errors["non_finite_action_value"] += 1
            if int(action.get("rollout_count", -1)) != future_samples:
                errors["wrong_action_rollout_count"] += 1
            original_index = int(action.get("original_index", action.get("action_index", -1)))
            if original_index in original_indices:
                errors["duplicate_action_index"] += 1
            original_indices.add(original_index)
            try:
                key = canonical_action(action)
            except (TypeError, ValueError):
                errors["invalid_action_encoding"] += 1
            else:
                if key in action_keys:
                    errors["duplicate_action_encoding"] += 1
                action_keys.add(key)
            if finite(action.get("score")):
                scores.append(float(action["score"]))
        if original_indices != set(range(action_count)):
            errors["incomplete_action_index_set"] += 1

        best_index = row.get("best_action")
        if not isinstance(best_index, int) or not 0 <= best_index < action_count:
            errors["illegal_best_action"] += 1
        elif scores and abs(float(actions[best_index]["score"]) - max(scores)) > 1e-8:
            errors["best_action_not_max"] += 1

    if records != expected_records:
        errors["record_count_mismatch"] += abs(records - expected_records) or 1
    summary = {
        "schema": "hu_turn1_teacher_audit_v1",
        "status": "pass" if not errors else "fail",
        "records": records,
        "expected_records": expected_records,
        "unique_sample_ids": len(sample_ids),
        "unique_state_ids": len(state_ids),
        "total_actions": total_actions,
        "action_count_min": min(action_counts) if action_counts else 0,
        "action_count_max": max(action_counts) if action_counts else 0,
        "action_count_mean": (sum(action_counts) / len(action_counts)) if action_counts else 0.0,
        "expected_seat": expected_seat,
        "future_samples": future_samples,
        "profile": profile,
        "opponent_profile": opponent_profile,
        "t3_continuation": t3_continuation,
        "errors": dict(sorted(errors.items())),
    }
    return summary


def main() -> None:
    args = parse_args()
    try:
        summary = audit_records(
            read_jsonl(args.input),
            expected_records=args.expected_records,
            expected_seat=args.expected_seat,
            future_samples=args.future_samples,
            profile=args.profile,
            opponent_profile=args.opponent_profile,
            t3_continuation=args.t3_continuation,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
