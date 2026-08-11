"""Extract replay-ready fired HU T0 states from whole-game counterfactual logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


ROWS = ("top", "middle", "bottom")


def _normalized_board(value: Any) -> dict[str, list[str]]:
    board = value if isinstance(value, dict) else {}
    return {row: list(board.get(row, ())) for row in ROWS}


def _target_id(event: dict[str, Any]) -> str:
    decision = event["decision"]
    payload = {
        "seat": event["seat"],
        "hero_board": _normalized_board(decision.get("hero_board")),
        "opponent_board": _normalized_board(decision.get("opponent_board")),
        "cards_to_place": list(decision["cards_to_place"]),
        "baseline_action_index": int(event["fallback_action_index"]),
        "candidate_action_index": int(event["candidate_action_index"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def extract_fired_targets(
    sources: Sequence[tuple[Path, str]],
    *,
    candidate_model: str,
    candidate_topk: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets_by_id: dict[str, dict[str, Any]] = {}
    input_counts: dict[str, int] = {}
    fired_counts: dict[str, int] = {}
    duplicate_count = 0
    for path, config_id in sources:
        rows = 0
        fires = 0
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                rows += 1
                event = json.loads(line)
                if str(event.get("config_id")) != config_id:
                    continue
                if not bool(event.get("override_fired")):
                    continue
                fires += 1
                decision = event.get("decision") or {}
                if not bool(decision.get("replay_ready")):
                    raise ValueError(f"{path}:{line_number} is not replay-ready")
                baseline_index = int(event["fallback_action_index"])
                candidate_index = int(event["candidate_action_index"])
                final_index = int(event["final_action_index"])
                if candidate_index == baseline_index or final_index != candidate_index:
                    raise ValueError(f"{path}:{line_number} has inconsistent fired actions")
                if not 0 <= baseline_index < 232 or not 0 <= candidate_index < 232:
                    raise ValueError(f"{path}:{line_number} has an invalid action index")
                if str(decision.get("visibility_model")) != "hidden_discard":
                    raise ValueError(f"{path}:{line_number} has the wrong visibility model")
                if str(decision.get("discard_visibility")) != "own_private_only":
                    raise ValueError(f"{path}:{line_number} exposes private discards")

                target_id = _target_id(event)
                target = {
                    "schema": "hu_turn0_fired_replay_target_v1",
                    "target_id": target_id,
                    "source_path": str(path),
                    "source_line": line_number,
                    "source_config_id": config_id,
                    "source_event_id": event["event_id"],
                    "hand_seed": int(event["seed"]),
                    "seat": str(event["seat"]),
                    "hero_board": _normalized_board(decision.get("hero_board")),
                    "opponent_board": _normalized_board(decision.get("opponent_board")),
                    "cards_to_place": list(decision["cards_to_place"]),
                    "dead_cards": list(decision.get("dead_cards", ())),
                    "visible_dead_cards": list(decision.get("visible_dead_cards", ())),
                    "baseline_action_index": baseline_index,
                    "candidate_action_index": candidate_index,
                    "baseline_action": decision["baseline_action"],
                    "candidate_action": decision["hu_turn0_action"],
                    "predicted_margin": float(event["hu_turn0_predicted_margin"]),
                    "realized_whole_game_delta": float(event["realized_delta"]),
                    "candidate_model": candidate_model,
                    "candidate_topk": int(candidate_topk),
                    "t1_continuation": "stage18_p1",
                    "t2_continuation": "stage9f_p2",
                    "t3_continuation": "stage7_m5_r10",
                    "visibility_model": "hidden_discard",
                    "discard_visibility": "own_private_only",
                    "replay_ready": True,
                }
                if target_id in targets_by_id:
                    duplicate_count += 1
                    continue
                targets_by_id[target_id] = target
        input_counts[str(path)] = input_counts.get(str(path), 0) + rows
        fired_counts[f"{path}:{config_id}"] = fires

    targets = sorted(targets_by_id.values(), key=lambda row: row["target_id"])
    summary = {
        "schema": "hu_turn0_fired_replay_targets_summary_v1",
        "candidate_model": candidate_model,
        "candidate_topk": int(candidate_topk),
        "input_counts": input_counts,
        "source_fired_counts": fired_counts,
        "targets": len(targets),
        "duplicates_dropped": duplicate_count,
        "seat_counts": dict(sorted(Counter(row["seat"] for row in targets).items())),
        "source_config_counts": dict(
            sorted(Counter(row["source_config_id"] for row in targets).items())
        ),
        "replay_ready": sum(bool(row["replay_ready"]) for row in targets),
        "missing_action_mapping": sum(
            row["baseline_action_index"] == row["candidate_action_index"] for row in targets
        ),
        "fixed_continuations": {
            "t1": "stage18_p1",
            "t2": "stage9f_p2",
            "t3": "stage7_m5_r10",
        },
    }
    return targets, summary


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_source(value: str) -> tuple[Path, str]:
    try:
        path_text, config_id = value.rsplit("::", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("source must be PATH::CONFIG_ID") from exc
    if not path_text or not config_id:
        raise argparse.ArgumentTypeError("source must be PATH::CONFIG_ID")
    return Path(path_text), config_id


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=parse_source, action="append", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--candidate-topk", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    targets, summary = extract_fired_targets(
        args.source,
        candidate_model=args.candidate_model,
        candidate_topk=args.candidate_topk,
    )
    _write_jsonl(args.output, targets)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
