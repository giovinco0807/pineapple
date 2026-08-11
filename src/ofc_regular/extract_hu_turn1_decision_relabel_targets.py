"""Extract replay-ready HU Turn1 decision-log rows for stronger relabeling."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--label",
        action="append",
        choices=("positive", "hard_negative", "neutral", "all"),
        default=None,
        help="Fired label to include. Repeatable. Default: positive + hard_negative.",
    )
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--max-per-label", type=int, default=0)
    parser.add_argument(
        "--seat",
        action="append",
        choices=("first", "second"),
        default=None,
        help="Seat to include. Repeatable. Default: both seats.",
    )
    parser.add_argument("--min-abs-delta", type=float, default=0.0)
    parser.add_argument(
        "--include-non-fired-candidates",
        action="store_true",
        help="Also include replay-ready non-fired candidate-vs-baseline rows when the actions differ.",
    )
    parser.add_argument(
        "--only-non-fired-candidates",
        action="store_true",
        help="Extract only non-fired candidate rows; useful for boundary teacher relabeling.",
    )
    parser.add_argument(
        "--include-all-replay-ready-states",
        action="store_true",
        help="Include every replay-ready state for all-legal-action relabeling, even when runtime kept baseline.",
    )
    parser.add_argument(
        "--min-predicted-margin",
        type=float,
        default=0.0,
        help="Minimum HU Turn1 predicted margin for non-fired candidate rows.",
    )
    parser.add_argument("--require-replay-ready", action="store_true", default=True)
    parser.add_argument("--allow-replay-ineligible", action="store_false", dest="require_replay_ready")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed and parsed not in (float("inf"), float("-inf")) else default


def label_for_row(row: dict[str, Any]) -> str:
    existing = row.get("hu_turn1_safe_override_label")
    if existing in {"positive", "hard_negative", "neutral"}:
        return str(existing)
    delta = safe_float(row.get("hu_turn1_realized_delta", row.get("realized_candidate_seat_delta")))
    if delta > 0.0:
        return "positive"
    if delta < 0.0:
        return "hard_negative"
    return "neutral"


def player_from_row(row: dict[str, Any]) -> int:
    if row.get("player") in (0, 1, "0", "1"):
        return int(row["player"])
    return 1 if row.get("seat") == "second" else 0


def state_key(row: dict[str, Any]) -> str:
    payload = {
        "hand_seed": row.get("hand_seed", row.get("hand_id")),
        "seat_swap": row.get("seat_swap"),
        "seat": row.get("seat"),
        "board": row.get("hero_board"),
        "opponent_board": row.get("opponent_board"),
        "dealt": row.get("cards_to_place"),
        "true_dead_cards": row.get("true_dead_cards"),
        "hero_private_discards": row.get("hero_private_discards"),
        "opponent_private_discards": row.get("opponent_private_discards"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def action_signature(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    placements = tuple((str(card), str(row)) for card, row in action.get("placements", ()))
    discards = tuple(str(card) for card in action.get("discards", ()))
    return json.dumps({"placements": placements, "discards": discards}, sort_keys=True, separators=(",", ":"))


def to_target(row: dict[str, Any], *, target_id: int) -> dict[str, Any]:
    label = label_for_row(row)
    delta = safe_float(row.get("hu_turn1_realized_delta", row.get("realized_candidate_seat_delta")))
    candidate = row.get("hu_turn1_action") or row.get("final_action")
    baseline = row.get("baseline_action") or row.get("fallback_action")
    source_actions = []
    if isinstance(candidate, dict):
        source_actions.append(
            {
                **candidate,
                "score": row.get("candidate_score", 0.0),
                "action_index": row.get("candidate_action_index"),
                "source_role": "runtime_candidate",
            }
        )
    if isinstance(baseline, dict) and action_signature(baseline) != action_signature(candidate):
        source_actions.append(
            {
                **baseline,
                "score": row.get("fallback_score", 0.0),
                "action_index": row.get("fallback_action_index"),
                "source_role": "runtime_baseline",
            }
        )
    return {
        "sample_id": target_id,
        "target_id": target_id,
        "rule_set": "regular",
        "schema": "hu_turn1_stage1_decision_relabel_target_v1",
        "phase": "hu_turn1_5card",
        "source_schema": "hu_turn1_decision_log",
        "source_kind": (
            "runtime_non_fired_candidate_decision"
            if not row.get("override_fired")
            else "runtime_fired_decision"
        ),
        "hand_seed": int(row.get("hand_seed", row.get("hand_id", 0)) or 0),
        "hand_id": row.get("hand_id"),
        "game_id": row.get("game_id"),
        "paired_index": row.get("paired_index"),
        "seat_swap": row.get("seat_swap"),
        "player": player_from_row(row),
        "seat": row.get("seat", "first"),
        "board": row.get("hero_board"),
        "opponent_board": row.get("opponent_board"),
        "dealt": row.get("cards_to_place", ()),
        "dead_cards": row.get("visible_dead_cards", row.get("dead_cards", ())),
        "visible_dead_cards": row.get("visible_dead_cards", row.get("dead_cards", ())),
        "true_dead_cards": row.get("true_dead_cards", ()),
        "hero_private_discards": row.get("hero_private_discards", ()),
        "opponent_private_discards": row.get("opponent_private_discards", ()),
        "replay_ready": bool(row.get("replay_ready")),
        "visibility_model": row.get("visibility_model", "unknown"),
        "discard_visibility": row.get("discard_visibility", "unknown"),
        "action_count": row.get("action_count"),
        "best_action": 0,
        "actions": source_actions,
        "runtime_candidate_action": candidate,
        "runtime_baseline_action": baseline,
        "runtime_candidate_action_signature": action_signature(candidate),
        "runtime_baseline_action_signature": action_signature(baseline),
        "runtime_candidate_action_index": row.get("candidate_action_index"),
        "runtime_baseline_action_index": row.get("fallback_action_index", row.get("baseline_action_index")),
        "runtime_predicted_margin": row.get("hu_turn1_predicted_margin"),
        "runtime_candidate_score": row.get("candidate_score"),
        "runtime_baseline_score": row.get("fallback_score"),
        "realized_delta": delta,
        "realized_delta_basis": row.get("realized_delta_basis"),
        "safe_override_label": label,
        "safe_override_label_id": 1 if label == "positive" else 0 if label == "hard_negative" else -1,
        "selection_reasons": [
            "runtime_non_fired_candidate"
            if not row.get("override_fired")
            else f"runtime_{label}"
        ],
        "state_key": state_key(row),
    }


def is_action_different(row: dict[str, Any]) -> bool:
    candidate = row.get("hu_turn1_action") or row.get("final_action")
    baseline = row.get("baseline_action") or row.get("fallback_action")
    return bool(action_signature(candidate) and action_signature(candidate) != action_signature(baseline))


def selected_rows(
    rows: Iterable[dict[str, Any]],
    *,
    labels: set[str],
    min_abs_delta: float,
    require_replay_ready: bool,
    include_non_fired_candidates: bool = False,
    only_non_fired_candidates: bool = False,
    include_all_replay_ready_states: bool = False,
    min_predicted_margin: float = 0.0,
    seats: set[str] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        if seats is not None and str(row.get("seat")) not in seats:
            skipped["seat_excluded"] += 1
            continue
        if require_replay_ready and not row.get("replay_ready"):
            skipped["not_replay_ready"] += 1
            continue
        if include_all_replay_ready_states:
            key = state_key(row)
            if key in seen:
                skipped["duplicate_state"] += 1
                continue
            seen.add(key)
            selected.append(row)
            continue
        if not row.get("override_fired"):
            if not include_non_fired_candidates and not only_non_fired_candidates:
                skipped["not_fired"] += 1
                continue
            if not is_action_different(row):
                skipped["non_fired_same_as_baseline"] += 1
                continue
            if safe_float(row.get("hu_turn1_predicted_margin")) < min_predicted_margin:
                skipped["below_min_predicted_margin"] += 1
                continue
            key = state_key(row)
            if key in seen:
                skipped["duplicate_state"] += 1
                continue
            seen.add(key)
            selected.append(row)
            continue
        if only_non_fired_candidates:
            skipped["fired_excluded"] += 1
            continue
        if not row.get("realized_delta_valid", True):
            skipped["invalid_realized_delta"] += 1
            continue
        label = label_for_row(row)
        if "all" not in labels and label not in labels:
            skipped[f"label_{label}_excluded"] += 1
            continue
        delta = safe_float(row.get("hu_turn1_realized_delta", row.get("realized_candidate_seat_delta")))
        if abs(delta) < min_abs_delta:
            skipped["below_min_abs_delta"] += 1
            continue
        key = state_key(row)
        if key in seen:
            skipped["duplicate_state"] += 1
            continue
        seen.add(key)
        selected.append(row)
    selected.sort(
        key=lambda row: (
            0 if not row.get("override_fired") else 1,
            0 if label_for_row(row) == "hard_negative" else 1,
            -safe_float(row.get("hu_turn1_predicted_margin")),
            -abs(safe_float(row.get("hu_turn1_realized_delta", row.get("realized_candidate_seat_delta")))),
        )
    )
    return selected, skipped


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def limit_per_label(rows: list[dict[str, Any]], max_per_label: int) -> list[dict[str, Any]]:
    if max_per_label <= 0:
        return rows
    counts: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for row in rows:
        label = label_for_row(row)
        if counts[label] >= max_per_label:
            continue
        counts[label] += 1
        output.append(row)
    return output


def main() -> None:
    args = parse_args()
    if args.max_targets < 0:
        raise SystemExit("--max-targets must be non-negative")
    if args.max_per_label < 0:
        raise SystemExit("--max-per-label must be non-negative")
    labels = set(args.label or ("positive", "hard_negative"))
    rows = read_jsonl(args.input)
    selected, skipped = selected_rows(
        rows,
        labels=labels,
        min_abs_delta=args.min_abs_delta,
        require_replay_ready=args.require_replay_ready,
        include_non_fired_candidates=args.include_non_fired_candidates,
        only_non_fired_candidates=args.only_non_fired_candidates,
        include_all_replay_ready_states=args.include_all_replay_ready_states,
        min_predicted_margin=args.min_predicted_margin,
        seats=set(args.seat) if args.seat else None,
    )
    selected = limit_per_label(selected, args.max_per_label)
    if args.max_targets > 0:
        selected = selected[: args.max_targets]
    targets = [to_target(row, target_id=index) for index, row in enumerate(selected)]
    written = write_jsonl(args.output, targets)
    label_counts = Counter(str(target.get("safe_override_label")) for target in targets)
    seat_counts = Counter(str(target.get("seat")) for target in targets)
    source_kind_counts = Counter(str(target.get("source_kind")) for target in targets)
    summary = {
        "schema": "hu_turn1_stage1_decision_relabel_target_summary_v1",
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(rows),
        "written": written,
        "labels_requested": sorted(labels),
        "seats_requested": sorted(set(args.seat)) if args.seat else ["first", "second"],
        "label_counts": dict(sorted(label_counts.items())),
        "seat_counts": dict(sorted(seat_counts.items())),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "require_replay_ready": args.require_replay_ready,
        "min_abs_delta": args.min_abs_delta,
        "include_non_fired_candidates": args.include_non_fired_candidates,
        "only_non_fired_candidates": args.only_non_fired_candidates,
        "include_all_replay_ready_states": args.include_all_replay_ready_states,
        "min_predicted_margin": args.min_predicted_margin,
        "max_targets": args.max_targets,
        "max_per_label": args.max_per_label,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
