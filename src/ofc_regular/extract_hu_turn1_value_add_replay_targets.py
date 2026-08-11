"""Extract HU Turn1 value-add replay targets from runtime decision logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .extract_hu_turn1_decision_relabel_targets import (
    action_signature,
    read_jsonl,
    safe_float,
    state_key,
    to_target,
)


BOUNDARY_REASONS = {
    "above_confirm_se",
    "below_confirm_delta",
    "below_confirm_se",
    "below_safe_selector",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--exclude-input",
        type=Path,
        action="append",
        default=None,
        help="Decision JSONL whose state/action pairs must not be selected again.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--seat", action="append", choices=("first", "second"), default=None)
    parser.add_argument("--max-neutral", type=int, default=200)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--min-abs-delta", type=float, default=0.0)
    parser.add_argument(
        "--fired-only",
        action="store_true",
        help="Select only runtime overrides that actually fired.",
    )
    parser.add_argument("--require-replay-ready", action="store_true", default=True)
    parser.add_argument("--allow-replay-ineligible", action="store_false", dest="require_replay_ready")
    return parser.parse_args()


def value_add_label(row: dict[str, Any]) -> str:
    existing = row.get("hu_turn1_value_add_label")
    if existing == "positive":
        return "positive"
    if existing in {"negative", "hard_negative"}:
        return "hard_negative"
    if existing == "neutral":
        return "neutral"
    delta = safe_float(row.get("hu_turn1_realized_delta", row.get("realized_candidate_seat_delta")))
    if delta > 0.0:
        return "positive"
    if delta < 0.0:
        return "hard_negative"
    return "neutral"


def action_pair_key(row: dict[str, Any]) -> str:
    candidate = row.get("hu_turn1_action") or row.get("final_action")
    baseline = row.get("baseline_action") or row.get("fallback_action")
    return json.dumps(
        {
            "state": state_key(row),
            "candidate": action_signature(candidate),
            "baseline": action_signature(baseline),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def action_pair_differs(row: dict[str, Any]) -> bool:
    candidate = row.get("hu_turn1_action") or row.get("final_action")
    baseline = row.get("baseline_action") or row.get("fallback_action")
    candidate_sig = action_signature(candidate)
    baseline_sig = action_signature(baseline)
    return bool(candidate_sig and baseline_sig and candidate_sig != baseline_sig)


def selection_bucket(row: dict[str, Any]) -> str | None:
    label = value_add_label(row)
    reason = str(row.get("no_override_reason") or ("override_fired" if row.get("override_fired") else ""))
    if row.get("override_fired"):
        return f"{label}_fired"
    if label in {"positive", "hard_negative"} and reason in BOUNDARY_REASONS:
        return f"{label}_near_miss_{reason}"
    if label == "neutral" and reason in BOUNDARY_REASONS:
        return f"neutral_boundary_{reason}"
    return None


def target_sort_key(row: dict[str, Any]) -> tuple[int, int, float, float]:
    label = value_add_label(row)
    bucket = selection_bucket(row) or ""
    label_rank = {"hard_negative": 0, "positive": 1, "neutral": 2}.get(label, 3)
    fired_rank = 0 if row.get("override_fired") else 1
    if "near_miss" in bucket:
        fired_rank = 0
    delta = abs(safe_float(row.get("hu_turn1_realized_delta", row.get("realized_candidate_seat_delta"))))
    confirm_delta = safe_float(row.get("confirm_delta"))
    return (label_rank, fired_rank, -delta, -confirm_delta)


def selected_rows(
    rows: Iterable[dict[str, Any]],
    *,
    seats: set[str] | None = None,
    max_neutral: int = 200,
    min_abs_delta: float = 0.0,
    require_replay_ready: bool = True,
    fired_only: bool = False,
    exclude_action_pair_keys: set[str] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    neutral_candidates: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        if fired_only and not row.get("override_fired"):
            skipped["not_fired"] += 1
            continue
        if seats is not None and str(row.get("seat")) not in seats:
            skipped["seat_excluded"] += 1
            continue
        if require_replay_ready and not row.get("replay_ready"):
            skipped["not_replay_ready"] += 1
            continue
        if not action_pair_differs(row):
            skipped["same_action_or_missing_action"] += 1
            continue
        bucket = selection_bucket(row)
        if bucket is None:
            skipped["not_value_add_target"] += 1
            continue
        key = action_pair_key(row)
        if exclude_action_pair_keys is not None and key in exclude_action_pair_keys:
            skipped["excluded_action_pair"] += 1
            continue
        if key in seen:
            skipped["duplicate_action_pair"] += 1
            continue
        label = value_add_label(row)
        delta = abs(safe_float(row.get("hu_turn1_realized_delta", row.get("realized_candidate_seat_delta"))))
        if label != "neutral" and delta < min_abs_delta:
            skipped["below_min_abs_delta"] += 1
            continue
        seen.add(key)
        if label == "neutral":
            neutral_candidates.append(row)
        else:
            selected.append(row)

    neutral_candidates.sort(key=target_sort_key)
    if max_neutral > 0:
        selected.extend(neutral_candidates[:max_neutral])
        skipped["neutral_over_limit"] += max(0, len(neutral_candidates) - max_neutral)
    elif max_neutral == 0:
        skipped["neutral_over_limit"] += len(neutral_candidates)
    else:
        selected.extend(neutral_candidates)
    selected.sort(key=target_sort_key)
    return selected, skipped


def enriched_target(row: dict[str, Any], *, target_id: int) -> dict[str, Any]:
    target = to_target(row, target_id=target_id)
    label = value_add_label(row)
    bucket = selection_bucket(row) or "unknown"
    target["schema"] = "hu_turn1_value_add_replay_target_v1"
    target["safe_override_label"] = label
    target["safe_override_label_id"] = 1 if label == "positive" else 0 if label == "hard_negative" else -1
    target["value_add_selection_bucket"] = bucket
    target["source_bucket"] = bucket
    target["source_bucket_group"] = "hu_turn1_value_add"
    target["stage10_source_run"] = row.get("stage10_source_run")
    target["source_run"] = row.get("stage10_source_run", row.get("source_run"))
    target["confirm_delta"] = row.get("confirm_delta")
    target["confirm_delta_se"] = row.get("confirm_delta_se")
    target["stage_a_delta"] = row.get("stage_a_delta")
    target["stage_a_delta_se"] = row.get("stage_a_delta_se")
    target["safe_selector_score"] = row.get("safe_selector_score")
    target["runtime_no_override_reason"] = row.get("no_override_reason")
    target["selection_reasons"] = [bucket, *list(target.get("selection_reasons", ()) or ())]
    return target


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    if args.max_neutral < -1:
        raise SystemExit("--max-neutral must be -1 or greater")
    if args.max_targets < 0:
        raise SystemExit("--max-targets must be non-negative")
    rows: list[dict[str, Any]] = []
    for path in args.input:
        rows.extend(read_jsonl(path))
    excluded_rows: list[dict[str, Any]] = []
    for path in args.exclude_input or ():
        excluded_rows.extend(read_jsonl(path))
    exclude_action_pair_keys = {
        action_pair_key(row) for row in excluded_rows if action_pair_differs(row)
    }
    selected, skipped = selected_rows(
        rows,
        seats=set(args.seat) if args.seat else None,
        max_neutral=args.max_neutral,
        min_abs_delta=args.min_abs_delta,
        require_replay_ready=args.require_replay_ready,
        fired_only=args.fired_only,
        exclude_action_pair_keys=exclude_action_pair_keys,
    )
    if args.max_targets > 0:
        selected = selected[: args.max_targets]
    targets = [enriched_target(row, target_id=index) for index, row in enumerate(selected)]
    written = write_jsonl(args.output, targets)
    summary = {
        "schema": "hu_turn1_value_add_replay_target_summary_v1",
        "inputs": [str(path) for path in args.input],
        "exclude_inputs": [str(path) for path in args.exclude_input or ()],
        "exclude_action_pair_count": len(exclude_action_pair_keys),
        "output": str(args.output),
        "rows": len(rows),
        "written": written,
        "label_counts": dict(sorted(Counter(str(row.get("safe_override_label")) for row in targets).items())),
        "bucket_counts": dict(sorted(Counter(str(row.get("value_add_selection_bucket")) for row in targets).items())),
        "seat_counts": dict(sorted(Counter(str(row.get("seat")) for row in targets).items())),
        "skipped": dict(sorted(skipped.items())),
        "max_neutral": args.max_neutral,
        "max_targets": args.max_targets,
        "min_abs_delta": args.min_abs_delta,
        "require_replay_ready": args.require_replay_ready,
        "fired_only": args.fired_only,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
