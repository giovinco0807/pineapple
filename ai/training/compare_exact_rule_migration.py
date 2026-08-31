"""Compare exact-teacher JSONL before and after a scoring-rule migration.

The late-turn Rust solver writes one record per decision and one metrics object
per legal action.  This module aligns those records by ``record_index`` and
actions by their semantic placement/discard, then reports how much the teacher
policy and its score/bust/FL targets changed.

When the original decision input is supplied, records are also classified as:

* ``visible_joker``: a Joker is already in known cards;
* ``future_joker_reachable``: the decision is before T4 and at least one Joker
  is not known, so a later exact draw can still contain it.

That distinction matters because a rule change can affect early labels even
when no Joker is currently visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


METRICS = ("score", "raw_score", "royalty", "bust_rate", "fl_rate")
ROW_ALIASES = {"mid": "middle", "bot": "bottom"}
JOKER_ALIASES = {"X", "X1", "X2", "JK", "Xj"}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def action_key(action: dict[str, Any] | None) -> tuple[tuple[tuple[str, str], ...], str | None]:
    action = action or {}
    placements = tuple(
        sorted(
            (str(card), ROW_ALIASES.get(str(row), str(row)))
            for card, row in action.get("placements", [])
        )
    )
    discard = action.get("discard")
    return placements, None if discard is None else str(discard)


def action_json(action: dict[str, Any] | None) -> dict[str, Any]:
    key = action_key(action)
    return {
        "placements": [[card, row] for card, row in key[0]],
        "discard": key[1],
    }


def result_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    return list(record.get("candidates") or [])


def result_best(record: dict[str, Any]) -> dict[str, Any] | None:
    best = record.get("best")
    if isinstance(best, dict):
        return best
    candidates = result_candidates(record)
    return candidates[0] if candidates else None


def metrics(candidate: dict[str, Any] | None) -> dict[str, float]:
    candidate = candidate or {}
    values = candidate.get("metrics") or candidate.get("exact") or candidate
    return {name: float(values.get(name, 0.0)) for name in METRICS}


def _known_decision_cards(record: dict[str, Any]) -> set[str]:
    decision = record.get("state") if isinstance(record.get("state"), dict) else record
    cards: set[str] = set()

    for board_name in ("board", "opponent_board", "board_self", "board_opponent"):
        board = decision.get(board_name)
        if not isinstance(board, dict):
            continue
        for row in ("top", "middle", "mid", "bottom", "bot"):
            values = board.get(row)
            if isinstance(values, list):
                cards.update(str(card) for card in values)

    for field in (
        "dealt",
        "dealt_cards",
        "known_discards",
        "known_discards_self",
        "exclude",
    ):
        values = decision.get(field)
        if isinstance(values, list):
            cards.update(str(card) for card in values)
    return cards


def classify_input(record: dict[str, Any]) -> dict[str, Any]:
    decision = record.get("state") if isinstance(record.get("state"), dict) else record
    known_cards = _known_decision_cards(record)
    known_jokers = {card for card in known_cards if card in JOKER_ALIASES or card.startswith("X")}

    named = {card for card in known_jokers if card in {"X1", "X2"}}
    generic_known = bool(known_jokers - named)
    known_count = min(2, len(named) + int(generic_known))
    turn = int(decision.get("turn", record.get("turn", -1)))
    visible = known_count > 0
    future_reachable = 0 <= turn < 4 and known_count < 2
    return {
        "turn": turn,
        "visible_joker": visible,
        "known_joker_count": known_count,
        "future_joker_reachable": future_reachable,
        "joker_relevant": visible or future_reachable,
    }


def load_by_record_index(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for ordinal, record in enumerate(iter_jsonl(path)):
        index = int(record.get("record_index", ordinal))
        if index in records:
            raise ValueError(f"duplicate record_index={index} in {path}")
        records[index] = record
    return records


def iter_indexed_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream a monotonically indexed JSONL without retaining nested labels."""
    previous = -1
    for ordinal, record in enumerate(iter_jsonl(path)):
        index = int(record.get("record_index", ordinal))
        if index <= previous:
            raise ValueError(
                f"record_index must be strictly increasing in {path}: "
                f"previous={previous}, current={index}"
            )
        previous = index
        yield index, record


def compare_records(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    classification: dict[str, Any] | None = None,
    threshold: float = 1e-9,
) -> dict[str, Any]:
    baseline_best = result_best(baseline)
    candidate_best = result_best(candidate)
    baseline_key = action_key((baseline_best or {}).get("action"))
    candidate_key = action_key((candidate_best or {}).get("action"))

    baseline_by_action = {
        action_key(item.get("action")): item for item in result_candidates(baseline)
    }
    candidate_by_action = {
        action_key(item.get("action")): item for item in result_candidates(candidate)
    }
    common = baseline_by_action.keys() & candidate_by_action.keys()

    baseline_best_metrics = metrics(baseline_best)
    candidate_best_metrics = metrics(candidate_best)
    metric_deltas = {
        name: candidate_best_metrics[name] - baseline_best_metrics[name]
        for name in METRICS
    }
    max_action_metric_delta = {name: 0.0 for name in METRICS}
    changed_actions = 0
    for key in common:
        old_metrics = metrics(baseline_by_action[key])
        new_metrics = metrics(candidate_by_action[key])
        changed = False
        for name in METRICS:
            delta = abs(new_metrics[name] - old_metrics[name])
            max_action_metric_delta[name] = max(max_action_metric_delta[name], delta)
            changed = changed or delta > threshold
        changed_actions += int(changed)

    old_choice_under_new = candidate_by_action.get(baseline_key)
    policy_regret = 0.0
    if old_choice_under_new is not None:
        policy_regret = (
            candidate_best_metrics["score"] - metrics(old_choice_under_new)["score"]
        )

    out = {
        "best_action_changed": baseline_key != candidate_key,
        "baseline_best_action": action_json((baseline_best or {}).get("action")),
        "candidate_best_action": action_json((candidate_best or {}).get("action")),
        "baseline_best_metrics": baseline_best_metrics,
        "candidate_best_metrics": candidate_best_metrics,
        "best_metric_deltas": metric_deltas,
        "common_actions": len(common),
        "baseline_actions": len(baseline_by_action),
        "candidate_actions": len(candidate_by_action),
        "changed_actions": changed_actions,
        "max_action_metric_abs_delta": max_action_metric_delta,
        "old_policy_regret_under_candidate": policy_regret,
        "candidate_policy_regret_vs_baseline": max(
            0.0,
            baseline_best_metrics["score"] - candidate_best_metrics["score"],
        ),
    }
    if classification:
        out.update(classification)
    return out


def summarize(
    baseline_path: Path,
    candidate_path: Path,
    *,
    input_path: Path | None = None,
    threshold: float = 1e-9,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_iter = iter(iter_indexed_jsonl(baseline_path))
    candidate_iter = iter(iter_indexed_jsonl(candidate_path))
    input_iter = iter(iter_indexed_jsonl(input_path)) if input_path else None
    baseline_item = next(baseline_iter, None)
    candidate_item = next(candidate_iter, None)
    input_item = next(input_iter, None) if input_iter else None
    baseline_count = int(baseline_item is not None)
    candidate_count = int(candidate_item is not None)
    missing_from_candidate = 0
    missing_from_baseline = 0
    rows: list[dict[str, Any]] = []
    while baseline_item is not None or candidate_item is not None:
        if candidate_item is None or (
            baseline_item is not None and baseline_item[0] < candidate_item[0]
        ):
            missing_from_candidate += 1
            baseline_item = next(baseline_iter, None)
            baseline_count += int(baseline_item is not None)
            continue
        if baseline_item is None or candidate_item[0] < baseline_item[0]:
            missing_from_baseline += 1
            candidate_item = next(candidate_iter, None)
            candidate_count += int(candidate_item is not None)
            continue

        index = baseline_item[0]
        classification = None
        if input_iter is not None:
            while input_item is not None and input_item[0] < index:
                input_item = next(input_iter, None)
            if input_item is not None and input_item[0] == index:
                classification = classify_input(input_item[1])
                input_item = next(input_iter, None)
        row = compare_records(
            baseline_item[1],
            candidate_item[1],
            classification=classification,
            threshold=threshold,
        )
        row["record_index"] = index
        rows.append(row)
        baseline_item = next(baseline_iter, None)
        candidate_item = next(candidate_iter, None)
        baseline_count += int(baseline_item is not None)
        candidate_count += int(candidate_item is not None)

    changed_best = sum(int(row["best_action_changed"]) for row in rows)
    old_policy_suboptimal = sum(
        int(row["old_policy_regret_under_candidate"] > threshold) for row in rows
    )
    candidate_policy_suboptimal = sum(
        int(row["candidate_policy_regret_vs_baseline"] > threshold) for row in rows
    )
    any_metric_changed = sum(
        int(any(value > threshold for value in row["max_action_metric_abs_delta"].values()))
        for row in rows
    )
    summary: dict[str, Any] = {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "input": str(input_path) if input_path else None,
        "threshold": threshold,
        "baseline_records": baseline_count,
        "candidate_records": candidate_count,
        "matched_records": len(rows),
        "missing_from_candidate": missing_from_candidate,
        "missing_from_baseline": missing_from_baseline,
        "best_action_changed": changed_best,
        "best_action_changed_rate": changed_best / max(len(rows), 1),
        "old_policy_suboptimal_under_candidate": old_policy_suboptimal,
        "old_policy_suboptimal_under_candidate_rate": old_policy_suboptimal
        / max(len(rows), 1),
        "candidate_policy_suboptimal_vs_baseline": candidate_policy_suboptimal,
        "candidate_policy_suboptimal_vs_baseline_rate": candidate_policy_suboptimal
        / max(len(rows), 1),
        "records_with_any_metric_change": any_metric_changed,
        "records_with_any_metric_change_rate": any_metric_changed / max(len(rows), 1),
        "avg_old_policy_regret_under_candidate": sum(
            row["old_policy_regret_under_candidate"] for row in rows
        )
        / max(len(rows), 1),
        "max_old_policy_regret_under_candidate": max(
            (row["old_policy_regret_under_candidate"] for row in rows),
            default=0.0,
        ),
        "avg_candidate_policy_regret_vs_baseline": sum(
            row["candidate_policy_regret_vs_baseline"] for row in rows
        )
        / max(len(rows), 1),
        "max_candidate_policy_regret_vs_baseline": max(
            (row["candidate_policy_regret_vs_baseline"] for row in rows),
            default=0.0,
        ),
        "best_metric_delta": {
            name: {
                "mean": sum(row["best_metric_deltas"][name] for row in rows)
                / max(len(rows), 1),
                "mean_abs": sum(abs(row["best_metric_deltas"][name]) for row in rows)
                / max(len(rows), 1),
                "max_abs": max(
                    (abs(row["best_metric_deltas"][name]) for row in rows),
                    default=0.0,
                ),
            }
            for name in METRICS
        },
    }

    if input_path:
        for field in ("visible_joker", "future_joker_reachable", "joker_relevant"):
            selected = [row for row in rows if row.get(field)]
            summary[field] = {
                "records": len(selected),
                "best_action_changed": sum(
                    int(row["best_action_changed"]) for row in selected
                ),
                "records_with_any_metric_change": sum(
                    int(
                        any(
                            value > threshold
                            for value in row["max_action_metric_abs_delta"].values()
                        )
                    )
                    for row in selected
                ),
            }
    return summary, rows


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare exact labels before and after a scoring-rule change"
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--details-output")
    parser.add_argument("--threshold", type=float, default=1e-9)
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary, rows = summarize(
        Path(args.baseline),
        Path(args.candidate),
        input_path=Path(args.input) if args.input else None,
        threshold=args.threshold,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.details_output:
        details = Path(args.details_output)
        details.parent.mkdir(parents=True, exist_ok=True)
        with details.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
