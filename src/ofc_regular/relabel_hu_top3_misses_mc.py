"""Relabel HU Turn3 top-3 miss candidates with a larger MC rollout count."""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .ai_profiles import DEFAULT_TURN3_MODEL
from .hu_self_play_teacher_data import evaluate_hu_self_play_turn3_actions
from .hu_turn3_model import load_hu_action_value_model, sample_to_matrix
from .policy import RegularAiPolicy
from .state import Board
from .turn3_model import load_action_value_model

_TURN3_MODEL: Any | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miss-summary", type=Path, required=True)
    parser.add_argument("--teacher-data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--future-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=9204800)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def action_key(action: dict[str, Any]) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    return (
        tuple((str(card), str(row)) for card, row in action.get("placements", ())),
        tuple(str(card) for card in action.get("discards", ())),
    )


def action_text(action: dict[str, Any]) -> str:
    placements = " ".join(f"{card}->{row}" for card, row in action.get("placements", ()))
    discards = ",".join(action.get("discards", ())) or "-"
    return f"{placements}; discard {discards}"


def _init_worker(turn3_model_path: str) -> None:
    global _TURN3_MODEL
    _TURN3_MODEL = load_action_value_model(turn3_model_path)


def _relabel_one(sample: dict[str, Any], line_index: int, future_samples: int, seed: int) -> dict[str, Any]:
    if _TURN3_MODEL is None:
        raise RuntimeError("worker turn3 model is not loaded")
    board = Board.from_rows(**sample["board"])
    opponent_board = Board.from_rows(**sample["opponent_board"])
    opponent_cards = set(opponent_board.all_cards())
    prior_dead = tuple(card for card in sample.get("dead_cards", ()) if card not in opponent_cards)
    hero_seat = str(sample.get("seat", "first"))
    opponent_seat = "second" if hero_seat == "first" else "first"
    hero_policy = RegularAiPolicy(turn3_model=_TURN3_MODEL, seed=seed * 2, seat=hero_seat)
    opponent_policy = RegularAiPolicy(turn3_model=_TURN3_MODEL, seed=seed * 2 + 1, seat=opponent_seat)
    started_at = time.time()
    ranked = evaluate_hu_self_play_turn3_actions(
        board=board,
        dealt_cards=sample["dealt"],
        opponent_board=opponent_board,
        dead_cards=prior_dead,
        hero_seat=hero_seat,
        hero_policy=hero_policy,
        opponent_policy=opponent_policy,
        self_regret_model=_TURN3_MODEL,
        self_regret_penalty_weight=0.25,
        self_regret_free=0.0,
        future_samples=future_samples,
        rng=random.Random(seed),
    )
    return {
        "line_index": int(line_index),
        "elapsed_seconds": time.time() - started_at,
        "ranked": ranked,
    }


def load_selected_samples(teacher_data: Path, line_indices: list[int]) -> dict[int, dict[str, Any]]:
    wanted = set(line_indices)
    samples: dict[int, dict[str, Any]] = {}
    with teacher_data.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index not in wanted:
                continue
            samples[line_index] = json.loads(line)
            if len(samples) == len(wanted):
                break
    missing = sorted(wanted - set(samples))
    if missing:
        raise ValueError(f"missing teacher lines: {missing[:10]}")
    return samples


def model_topk_for_sample(model: Any, sample: dict[str, Any], top_k: int) -> dict[str, Any]:
    features, _targets = sample_to_matrix(sample)
    predictions = model.predict_matrix(features)
    k = min(top_k, len(predictions))
    top_indices = [int(index) for index in np.argsort(predictions)[-k:][::-1]]
    return {
        "predictions": [float(value) for value in predictions],
        "top_indices": top_indices,
        "top_keys": [action_key(sample["actions"][index]) for index in top_indices],
    }


def summarize_result(
    *,
    sample: dict[str, Any],
    line_index: int,
    relabel: dict[str, Any],
    model_info: dict[str, Any],
) -> dict[str, Any]:
    mc_by_key = {action_key(action): action for action in relabel["ranked"]}
    if not relabel["ranked"]:
        raise ValueError(f"line {line_index} produced no MC actions")
    mc_best = relabel["ranked"][0]
    mc_best_key = action_key(mc_best)
    original_best_key = action_key(sample["actions"][0])
    model_top1_key = model_info["top_keys"][0]
    model_top3_keys = set(model_info["top_keys"])
    model_top1_mc = mc_by_key[model_top1_key]
    mc_best_score = float(mc_best["score"])
    model_top1_mc_score = float(model_top1_mc["score"])
    return {
        "line_index": int(line_index),
        "seat": sample.get("seat"),
        "dealt": sample.get("dealt"),
        "board": sample.get("board"),
        "opponent_board": sample.get("opponent_board"),
        "actions_count": len(sample.get("actions", ())),
        "future_samples": int(mc_best.get("future_count", 0)),
        "elapsed_seconds": float(relabel["elapsed_seconds"]),
        "mc_top3_miss": mc_best_key not in model_top3_keys,
        "original_mc128_best_still_best": mc_best_key == original_best_key,
        "mc_top1_regret_for_model_top1": mc_best_score - model_top1_mc_score,
        "mc_best": {
            "text": action_text(mc_best),
            "score": mc_best_score,
            "raw_score": float(mc_best.get("raw_score", 0.0)),
            "self_model_score": float(mc_best.get("self_model_score", 0.0)),
            "self_regret": float(mc_best.get("self_regret", 0.0)),
            "next_board": mc_best.get("next_board"),
        },
        "original_mc128_best": {
            "text": action_text(sample["actions"][0]),
            "score": float(mc_by_key[original_best_key]["score"]),
            "original_score": float(sample["actions"][0]["score"]),
            "mc2048_rank": next(
                index + 1
                for index, action in enumerate(relabel["ranked"])
                if action_key(action) == original_best_key
            ),
        },
        "model_top3": [
            {
                "rank": rank + 1,
                "index": int(index),
                "text": action_text(sample["actions"][index]),
                "model_pred_ev": float(model_info["predictions"][index]),
                "mc_score": float(mc_by_key[action_key(sample["actions"][index])]["score"]),
                "mc_rank": next(
                    mc_rank + 1
                    for mc_rank, action in enumerate(relabel["ranked"])
                    if action_key(action) == action_key(sample["actions"][index])
                ),
                "next_board": sample["actions"][index].get("next_board"),
            }
            for rank, index in enumerate(model_info["top_indices"])
        ],
        "mc_top5": [
            {
                "rank": rank + 1,
                "text": action_text(action),
                "score": float(action["score"]),
                "raw_score": float(action.get("raw_score", 0.0)),
                "self_model_score": float(action.get("self_model_score", 0.0)),
                "self_regret": float(action.get("self_regret", 0.0)),
                "next_board": action.get("next_board"),
            }
            for rank, action in enumerate(relabel["ranked"][:5])
        ],
    }


def main() -> None:
    args = parse_args()
    if args.future_samples <= 0:
        raise SystemExit("--future-samples must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")

    miss_summary = json.loads(args.miss_summary.read_text(encoding="utf-8"))
    examples = miss_summary.get("examples_sorted_by_top1_regret", [])
    if args.limit is not None:
        examples = examples[: args.limit]
    line_indices = [int(example["sample_line_index"]) for example in examples]
    samples_by_line = load_selected_samples(args.teacher_data, line_indices)

    hu_model = load_hu_action_value_model(args.model)
    model_infos = {
        line_index: model_topk_for_sample(hu_model, samples_by_line[line_index], args.top_k)
        for line_index in line_indices
    }

    started_at = time.time()
    relabels: dict[int, dict[str, Any]] = {}
    if args.workers == 1:
        _init_worker(str(args.turn3_model))
        for ordinal, line_index in enumerate(line_indices, 1):
            relabels[line_index] = _relabel_one(
                samples_by_line[line_index],
                line_index,
                args.future_samples,
                args.seed + line_index,
            )
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "done": ordinal,
                        "total": len(line_indices),
                        "line_index": line_index,
                        "elapsed_seconds": time.time() - started_at,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(str(args.turn3_model),),
        ) as executor:
            futures = {
                executor.submit(
                    _relabel_one,
                    samples_by_line[line_index],
                    line_index,
                    args.future_samples,
                    args.seed + line_index,
                ): line_index
                for line_index in line_indices
            }
            for ordinal, future in enumerate(as_completed(futures), 1):
                line_index = futures[future]
                relabels[line_index] = future.result()
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "done": ordinal,
                            "total": len(line_indices),
                            "line_index": line_index,
                            "elapsed_seconds": time.time() - started_at,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

    results = [
        summarize_result(
            sample=samples_by_line[line_index],
            line_index=line_index,
            relabel=relabels[line_index],
            model_info=model_infos[line_index],
        )
        for line_index in line_indices
    ]
    results.sort(key=lambda item: item["mc_top1_regret_for_model_top1"], reverse=True)
    mc_misses = [item for item in results if item["mc_top3_miss"]]
    output = {
        "model": str(args.model),
        "miss_summary": str(args.miss_summary),
        "teacher_data": str(args.teacher_data),
        "turn3_model": str(args.turn3_model),
        "future_samples": args.future_samples,
        "seed": args.seed,
        "workers": args.workers,
        "input_candidates": len(line_indices),
        "mc_top3_miss_count": len(mc_misses),
        "mc_top3_miss_rate_among_candidates": len(mc_misses) / len(line_indices) if line_indices else 0.0,
        "original_mc128_best_still_best_count": sum(
            1 for item in results if item["original_mc128_best_still_best"]
        ),
        "elapsed_seconds": time.time() - started_at,
        "mc_top3_misses": mc_misses,
        "all_relabels": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in (
        "input_candidates",
        "mc_top3_miss_count",
        "mc_top3_miss_rate_among_candidates",
        "original_mc128_best_still_best_count",
        "elapsed_seconds",
    )}, indent=2))


if __name__ == "__main__":
    main()
