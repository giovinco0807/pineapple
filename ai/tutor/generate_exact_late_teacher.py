"""Generate exact T3/T4 teacher labels from self-play target JSONL.

T4 BTN actions are terminal and scored directly.  For T4 BB actions, every BTN
draw and BTN's best legal reply are enumerated.  T3 is evaluated by enumerating
every possible Hero T4 draw and choosing Hero's exact best T4 placement.

The output schema intentionally matches the flat candidate teacher JSONL used
by ``ai.training.convert_action_value_teacher``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.action_space import encode_action, get_turn_actions
from ai.tutor.exact_late import (
    action_key,
    action_to_dict,
    apply_action,
    board_card_count,
    board_to_dict,
    exact_candidate_metrics,
    evaluate_late_position,
    normalize_position_payload,
)


def _candidate_record(action_idx: int, action: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    score = float(metrics.get("score", metrics.get("ev", 0.0)) or 0.0)
    bust_rate = float(metrics.get("bust_rate", 0.0) or 0.0)
    fl_rate = float(metrics.get("fl_rate", 0.0) or 0.0)
    fl_type_rates = {
        "qq": float((metrics.get("fl_type_rates") or {}).get("qq", 0.0) or 0.0),
        "kk": float((metrics.get("fl_type_rates") or {}).get("kk", 0.0) or 0.0),
        "aa": float((metrics.get("fl_type_rates") or {}).get("aa", 0.0) or 0.0),
        "trips": float((metrics.get("fl_type_rates") or {}).get("trips", 0.0) or 0.0),
    }
    action_dict = action_to_dict(action)
    return {
        **action_dict,
        "action_idx": int(action_idx),
        "target_score": score,
        "ev": score,
        "expected_royalty": float(metrics.get("royalty", 0.0) or 0.0),
        "raw_score": float(metrics.get("raw_score", score) or 0.0),
        "bust_prob": bust_rate,
        "fl_rate": fl_rate,
        "fl_type_rates": fl_type_rates,
        "exact": metrics,
        # Keep an MC-shaped copy so older evaluators that look under ``mc``
        # still see FL-type rates and candidate-level probabilities.
        "mc": {
            "avg_score": score,
            "bust_rate": bust_rate,
            "fl_rate": fl_rate,
            "fl_type_rates": fl_type_rates,
            "avg_royalty": float(metrics.get("royalty", 0.0) or 0.0),
            "n_rollouts": int(metrics.get("samples", 0) or 0),
        },
    }


def evaluate_target(payload: tuple[int, dict[str, Any]]) -> dict[str, Any] | None:
    index, target = payload
    turn = int(target.get("turn", -1))
    if turn not in (3, 4):
        return None

    board, opponent, dealt, exclude = normalize_position_payload(target)
    expected_count = 9 if turn == 3 else 11
    if board_card_count(board) != expected_count:
        raise ValueError(
            f"turn {turn} requires {expected_count} board cards, got {board_card_count(board)}"
        )
    if len(dealt) != 3:
        raise ValueError(f"turn {turn} requires 3 dealt cards, got {len(dealt)}")

    actions = get_turn_actions(dealt, board)
    rust_metrics_by_action: dict[str, dict[str, Any]] = {}
    if turn == 4 and board_card_count(opponent) == 11:
        exact_result = evaluate_late_position(
            board,
            dealt,
            4,
            opponent_board=opponent,
            exclude=exclude,
            top_n=max(1, len(actions)),
            prefer_rust=True,
        )
        rust_metrics_by_action = {
            action_key(candidate.get("action") or {}): candidate.get("metrics") or {}
            for candidate in (exact_result.get("candidates") or [])
        }
        if len(rust_metrics_by_action) != len(actions):
            raise RuntimeError(
                "T4 exact solver did not return every legal BB action: "
                f"{len(rust_metrics_by_action)}/{len(actions)}"
            )
    candidates: list[dict[str, Any]] = []
    total_samples = 0
    for action in actions:
        action_idx = int(encode_action(action, actions, turn=turn, dealt_cards=dealt))
        metrics = rust_metrics_by_action.get(action_key(action))
        if metrics is None:
            metrics = exact_candidate_metrics(
                board,
                dealt,
                action,
                turn,
                opponent_board=opponent,
                exclude=exclude,
            )
        total_samples += int(metrics.get("samples", 0) or 0)
        candidate = _candidate_record(action_idx, action, metrics)
        candidate["board"] = board_to_dict(apply_action(board, action))
        candidates.append(candidate)

    candidates.sort(key=lambda item: float(item.get("target_score", 0.0)), reverse=True)
    position = str(target.get("position") or ("btn" if target.get("is_btn") else "bb"))
    if turn == 3:
        exact_scope = "t3_self_board_all_t4_draws_best_t4"
    elif board_card_count(opponent) == 11:
        exact_scope = "t4_all_opponent_draws_best_response_given_exclude"
    else:
        exact_scope = "t4_terminal_actions"
    return {
        "source": target.get("source"),
        "source_line": target.get("source_line"),
        "target_index": int(index),
        "active_reasons": list(target.get("reasons", [])),
        "turn": turn,
        "board": board_to_dict(board),
        "opponent_board": board_to_dict(opponent),
        "dealt": list(dealt),
        "known_discards": list(target.get("known_discards") or []),
        "exclude": list(target.get("exclude") or []),
        "is_btn": bool(target.get("is_btn", position == "btn")),
        "position": position,
        "n_candidates": len(candidates),
        "candidates": candidates,
        "best_idx": 0,
        "eval_mode": "exact",
        "elapsed_s": 0.0,
        "candidate_source": "exact_late_all_actions",
        "exact_scope": exact_scope,
        "hu_exact": bool(turn == 4 and board_card_count(opponent) == 13),
        "information_model": (
            "exclude_conditioned_physical_or_uniform_unspecified"
            if turn == 4 and board_card_count(opponent) == 11
            else "terminal_public_state"
        ),
        "total_exact_samples": int(total_samples),
        "original_n_actions": int(target.get("n_actions") or len(actions)),
        "original_evaluated_actions": int(target.get("evaluated_actions") or len(actions)),
        "original_eval_mode": target.get("eval_mode"),
        "original_sims": int(target.get("sims") or 0),
    }


def iter_targets(path: Path, include_turns: set[int]) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            target = json.loads(line)
            if int(target.get("turn", -1)) in include_turns:
                yield target


def load_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    include_turns = {int(part) for part in args.turns.split(",") if part.strip()}
    unsupported = include_turns - {3, 4}
    if unsupported:
        raise ValueError(f"exact late teacher only supports turns 3 and 4, got {sorted(unsupported)}")
    targets = list(iter_targets(Path(args.input), include_turns))
    if args.max_records > 0:
        targets = targets[: args.max_records]
    return targets


def generate(args: argparse.Namespace) -> dict[str, Any]:
    targets = load_targets(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "input": str(args.input),
        "output": str(output),
        "targets": len(targets),
        "written": 0,
        "skipped": 0,
        "turns": {},
        "candidates_by_turn": {},
        "samples_by_turn": {},
        "elapsed_s": 0.0,
        "source": "exact_late",
    }
    start = time.time()
    work = [(i, target) for i, target in enumerate(targets, start=1)]

    with output.open("w", encoding="utf-8") as dst:
        if args.workers <= 1:
            iterator = (evaluate_target(item) for item in work)
            for i, record in enumerate(iterator, start=1):
                if record is None:
                    stats["skipped"] += 1
                    continue
                _write_record(dst, record, stats)
                if i % args.progress_every == 0:
                    dst.flush()
                    print(f"  {i:,}/{len(work):,} written={stats['written']:,}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(evaluate_target, item) for item in work]
                for i, future in enumerate(as_completed(futures), start=1):
                    try:
                        record = future.result()
                    except Exception as exc:
                        stats["skipped"] += 1
                        if args.print_errors:
                            print(f"  error: {exc}", flush=True)
                        continue
                    if record is None:
                        stats["skipped"] += 1
                        continue
                    _write_record(dst, record, stats)
                    if i % args.progress_every == 0:
                        dst.flush()
                        print(f"  {i:,}/{len(work):,} written={stats['written']:,}", flush=True)

    stats["elapsed_s"] = round(time.time() - start, 3)
    output.with_suffix(".summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return stats


def _write_record(dst: Any, record: dict[str, Any], stats: dict[str, Any]) -> None:
    dst.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats["written"] += 1
    turn_key = str(record["turn"])
    stats["turns"][turn_key] = int(stats["turns"].get(turn_key, 0)) + 1
    stats["candidates_by_turn"][turn_key] = int(stats["candidates_by_turn"].get(turn_key, 0)) + int(
        record.get("n_candidates") or 0
    )
    stats["samples_by_turn"][turn_key] = int(stats["samples_by_turn"].get(turn_key, 0)) + int(
        record.get("total_exact_samples") or 0
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate exact T3/T4 teacher labels")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", default="3,4")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--print-errors", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(generate(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
