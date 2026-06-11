"""Build HU Turn3 teacher samples from override counterfactual traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-wins", action="store_true")
    parser.add_argument("--min-abs-delta", type=float, default=0.0)
    parser.add_argument("--score-scale", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def build_sample(row: dict[str, Any], sample_id: int, *, score_scale: float) -> dict[str, Any] | None:
    delta = float(row.get("counterfactual_delta_vs_baseline", 0.0))
    chosen = dict(row["chosen_action"])
    baseline = dict(row["baseline_action"])
    chosen_key = _action_key(chosen)
    baseline_key = _action_key(baseline)
    if chosen_key == baseline_key:
        return None

    chosen["score"] = float(row["counterfactual_chosen_score"]) * score_scale
    chosen["future_count"] = 1
    chosen["non_bust_future_count"] = int(chosen["score"] > -1000.0)
    baseline["score"] = float(row["counterfactual_baseline_score"]) * score_scale
    baseline["future_count"] = 1
    baseline["non_bust_future_count"] = int(baseline["score"] > -1000.0)
    actions = [chosen, baseline]
    actions.sort(key=lambda action: float(action["score"]), reverse=True)
    return {
        "sample_id": sample_id,
        "rule_set": "regular",
        "schema": "hu_stage1",
        "phase": "hu_turn3_9card",
        "seat": row.get("seat", "first"),
        "to_act_order": row.get("to_act_order", row.get("seat", "first")),
        "board": row["board"],
        "opponent_board": row["opponent_board"],
        "dead_cards": row.get("dead_cards", ()),
        "dealt": row["dealt"],
        "best_action": 0,
        "score_gap": abs(delta) * score_scale,
        "source": "override_counterfactual",
        "trace_seed": row.get("seed"),
        "trace_turn": row.get("turn"),
        "trace_delta_vs_baseline": delta,
        "trace_predicted_margin": row.get("predicted_margin"),
        "trace_self_regret_vs_baseline": row.get("self_regret_vs_baseline"),
        "actions": actions,
    }


def _action_key(action: dict[str, Any]) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    placements = tuple((str(card), str(row)) for card, row in action.get("placements", ()))
    discards = tuple(str(card) for card in action.get("discards", ()))
    return placements, discards


def main() -> None:
    args = parse_args()
    if args.score_scale <= 0:
        raise SystemExit("--score-scale must be positive")
    output_count = 0
    skipped_win = 0
    skipped_delta = 0
    skipped_duplicate = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.trace.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as output:
        for line in source:
            row = json.loads(line)
            if not row.get("override", False):
                continue
            delta = float(row.get("counterfactual_delta_vs_baseline", 0.0))
            if delta > 0 and not args.include_wins:
                skipped_win += 1
                continue
            if abs(delta) < args.min_abs_delta:
                skipped_delta += 1
                continue
            sample = build_sample(row, output_count, score_scale=args.score_scale)
            if sample is None:
                skipped_duplicate += 1
                continue
            output.write(json.dumps(sample, separators=(",", ":")) + "\n")
            output_count += 1
            if args.max_samples is not None and output_count >= args.max_samples:
                break
    summary = {
        "trace": str(args.trace),
        "output": str(args.output),
        "samples": output_count,
        "include_wins": args.include_wins,
        "min_abs_delta": args.min_abs_delta,
        "score_scale": args.score_scale,
        "skipped_win": skipped_win,
        "skipped_delta": skipped_delta,
        "skipped_duplicate": skipped_duplicate,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
