"""Analyze HU T0/T1 candidate-model coverage against all-action teacher rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol

from .action_space import Action
from .hu_turn0_candidate import load_turn0_candidate_model
from .hu_turn3_model import hu_policy_sample
from .state import Board


class CandidateModel(Protocol):
    def predict_sample(self, sample: Any) -> list[float]: ...


@dataclass(frozen=True)
class CandidateModelSpec:
    name: str
    model: CandidateModel
    topk: int | None = None


UnionMode = Literal[
    "min_rank",
    "rank_sum",
    "reciprocal_rank_sum",
    "mean_score",
    "max_score",
    "mean_z_score",
    "max_z_score",
]


def _board_from_json(data: dict[str, Any]) -> Board:
    return Board.from_rows(data.get("top", ()), data.get("middle", ()), data.get("bottom", ()))


def _action_from_json(data: dict[str, Any]) -> Action:
    return Action(
        placements=tuple((str(card), str(row)) for card, row in data.get("placements", ())),
        discards=tuple(str(card) for card in data.get("discards", ())),
    )


def _action_original_index(data: dict[str, Any]) -> int:
    return int(data.get("original_index", data.get("action_index")))


def _action_ev(data: dict[str, Any]) -> float:
    return float(data.get("ev", data.get("score")))


def _parse_ks(values: Iterable[str]) -> list[int]:
    ks: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                k = int(part)
                if k <= 0:
                    raise ValueError("all K values must be positive")
                ks.append(k)
    return sorted(set(ks))


def _single_model_order(
    *,
    model: CandidateModel,
    sample: dict[str, Any],
    original_indices: list[int],
) -> list[int]:
    predictions = model.predict_sample(sample)
    if len(predictions) != len(original_indices):
        raise ValueError(f"prediction count mismatch: expected {len(original_indices)}, got {len(predictions)}")
    return [
        original_indices[index]
        for index in sorted(range(len(original_indices)), key=lambda index: (-float(predictions[index]), original_indices[index]))
    ]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _z_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance <= 0.0:
        return [0.0 for _value in values]
    std = math.sqrt(variance)
    return [(value - mean) / std for value in values]


def _union_model_order(
    *,
    specs: list[CandidateModelSpec],
    sample: dict[str, Any],
    original_indices: list[int],
    union_mode: UnionMode = "min_rank",
) -> tuple[list[int], list[dict[str, Any]]]:
    union_ranks: dict[int, int] = {}
    rank_by_model: list[dict[int, int]] = []
    score_by_model: list[dict[int, float]] = []
    z_score_by_model: list[dict[int, float]] = []
    per_model: list[dict[str, Any]] = []
    for model_index, spec in enumerate(specs):
        predictions = spec.model.predict_sample(sample)
        if len(predictions) != len(original_indices):
            raise ValueError(
                f"prediction count mismatch for {spec.name}: "
                f"expected {len(original_indices)}, got {len(predictions)}"
            )
        order_offsets = sorted(
            range(len(original_indices)),
            key=lambda index: (-float(predictions[index]), original_indices[index]),
        )
        rank_map = {
            original_indices[offset]: rank
            for rank, offset in enumerate(order_offsets)
        }
        score_map = {
            original_indices[offset]: float(predictions[offset])
            for offset in range(len(original_indices))
        }
        z_values = _z_scores([float(value) for value in predictions])
        z_score_map = {
            original_indices[offset]: float(z_values[offset])
            for offset in range(len(original_indices))
        }
        rank_by_model.append(rank_map)
        score_by_model.append(score_map)
        z_score_by_model.append(z_score_map)
        topk = len(order_offsets) if spec.topk is None else min(spec.topk, len(order_offsets))
        selected_offsets = order_offsets[:topk]
        for rank, offset in enumerate(selected_offsets):
            original_index = original_indices[offset]
            previous = union_ranks.get(original_index)
            if previous is None or rank < previous:
                union_ranks[original_index] = rank
        per_model.append(
            {
                "model_index": model_index,
                "name": spec.name,
                "topk": int(topk),
                "selected_count": len(selected_offsets),
            }
        )
    if union_mode == "min_rank":
        sort_key = lambda original_index: (union_ranks[original_index], original_index)
    elif union_mode == "rank_sum":
        penalty_rank = len(original_indices)
        sort_key = lambda original_index: (
            sum(rank_map.get(original_index, penalty_rank) for rank_map in rank_by_model),
            union_ranks[original_index],
            original_index,
        )
    elif union_mode == "reciprocal_rank_sum":
        sort_key = lambda original_index: (
            -sum(1.0 / (rank_map[original_index] + 1.0) for rank_map in rank_by_model if original_index in rank_map),
            union_ranks[original_index],
            original_index,
        )
    elif union_mode == "mean_score":
        sort_key = lambda original_index: (
            -_mean([score_map[original_index] for score_map in score_by_model]),
            union_ranks[original_index],
            original_index,
        )
    elif union_mode == "max_score":
        sort_key = lambda original_index: (
            -max(score_map[original_index] for score_map in score_by_model),
            union_ranks[original_index],
            original_index,
        )
    elif union_mode == "mean_z_score":
        sort_key = lambda original_index: (
            -_mean([score_map[original_index] for score_map in z_score_by_model]),
            union_ranks[original_index],
            original_index,
        )
    elif union_mode == "max_z_score":
        sort_key = lambda original_index: (
            -max(score_map[original_index] for score_map in z_score_by_model),
            union_ranks[original_index],
            original_index,
        )
    else:
        raise ValueError(f"unknown union mode: {union_mode}")
    return sorted(union_ranks, key=sort_key), per_model


def analyze_rows(rows: Iterable[dict[str, Any]], model: CandidateModel, ks: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return analyze_rows_union(rows, [CandidateModelSpec("candidate_model", model, None)], ks)


def analyze_rows_union(
    rows: Iterable[dict[str, Any]],
    model_specs: list[CandidateModelSpec],
    ks: list[int],
    union_mode: UnionMode = "min_rank",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not model_specs:
        raise ValueError("at least one candidate model is required")
    stats = {k: {"hit": 0, "miss": 0, "regret_sum": 0.0, "max_regret": 0.0} for k in ks}
    detail_rows: list[dict[str, Any]] = []
    record_count = 0
    for row in rows:
        actions_json = list(row.get("actions", ()))
        if not actions_json:
            continue
        record_count += 1
        board = _board_from_json(row["board"])
        opponent_board = _board_from_json(row["opponent_board"])
        pairs = sorted(
            ((_action_original_index(action_json), _action_from_json(action_json)) for action_json in actions_json),
            key=lambda item: item[0],
        )
        original_indices = [original_index for original_index, _action in pairs]
        actions = [action for _original_index, action in pairs]
        ev_by_original = {_action_original_index(action_json): _action_ev(action_json) for action_json in actions_json}
        sample = hu_policy_sample(
            board,
            row["dealt"],
            actions,
            opponent_board=opponent_board,
            dead_cards=tuple(row.get("visible_dead_cards") or row.get("dead_cards") or ()),
            seat=row.get("seat"),
            to_act_order=row.get("seat"),
        )
        if len(model_specs) == 1 and model_specs[0].topk is None:
            predicted_order = _single_model_order(
                model=model_specs[0].model,
                sample=sample,
                original_indices=original_indices,
            )
            per_model = [{"model_index": 0, "name": model_specs[0].name, "topk": len(actions), "selected_count": len(actions)}]
        else:
            predicted_order, per_model = _union_model_order(
                specs=model_specs,
                sample=sample,
                original_indices=original_indices,
                union_mode=union_mode,
            )
        best_action_offset = int(row["best_action"])
        best_original = _action_original_index(actions_json[best_action_offset])
        best_ev = ev_by_original[best_original]
        detail = {
            "sample_id": row.get("sample_id"),
            "hand_seed": row.get("hand_seed"),
            "seat": row.get("seat"),
            "action_count": len(actions),
            "best_original_index": best_original,
            "best_ev": best_ev,
            "score_gap": row.get("score_gap"),
            "candidate_union_size": len(predicted_order),
            "candidate_model_count": len(model_specs),
            "candidate_union_mode": union_mode,
        }
        for model_row in per_model:
            prefix = f"model{model_row['model_index']}"
            detail[f"{prefix}_selected_count"] = model_row["selected_count"]
        for k in ks:
            top_originals = predicted_order[: min(k, len(predicted_order))]
            top_best_ev = max((ev_by_original[original] for original in top_originals), default=-math.inf)
            regret = max(0.0, best_ev - top_best_ev)
            hit = best_original in top_originals
            stats[k]["hit"] += int(hit)
            stats[k]["miss"] += int(not hit)
            stats[k]["regret_sum"] += regret
            stats[k]["max_regret"] = max(stats[k]["max_regret"], regret)
            detail[f"top{k}_hit"] = hit
            detail[f"top{k}_regret"] = regret
        detail_rows.append(detail)
    summary_rows: list[dict[str, Any]] = []
    for k in ks:
        stat = stats[k]
        summary_rows.append(
            {
                "k": k,
                "records": record_count,
                "model_count": len(model_specs),
                "hit": stat["hit"],
                "miss": stat["miss"],
                "recall": stat["hit"] / record_count if record_count else 0.0,
                "avg_topk_regret": stat["regret_sum"] / record_count if record_count else 0.0,
                "max_topk_regret": stat["max_regret"],
            }
        )
    return summary_rows, detail_rows


def summarize_details_by_seat(
    detail_rows: list[dict[str, Any]], ks: list[int]
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    seats = sorted({str(row.get("seat", "unknown")) for row in detail_rows})
    for seat in seats:
        rows = [row for row in detail_rows if str(row.get("seat", "unknown")) == seat]
        for k in ks:
            regrets = [float(row[f"top{k}_regret"]) for row in rows]
            hits = sum(int(bool(row[f"top{k}_hit"])) for row in rows)
            summary.append(
                {
                    "seat": seat,
                    "k": k,
                    "records": len(rows),
                    "hit": hits,
                    "miss": len(rows) - hits,
                    "recall": hits / len(rows) if rows else 0.0,
                    "avg_topk_regret": _mean(regrets),
                    "max_topk_regret": max(regrets, default=0.0),
                }
            )
    return summary


def infer_artifact_stage(rows: list[dict[str, Any]]) -> str:
    if any(str(row.get("phase", "")).startswith("hu_turn0") for row in rows) or any(
        str(row.get("schema", "")).startswith("hu_turn0") for row in rows
    ):
        return "hu_turn0"
    return "hu_turn1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path)
    parser.add_argument("--candidate-models", type=Path, nargs="+")
    parser.add_argument("--candidate-topk", type=int, default=0)
    parser.add_argument(
        "--union-mode",
        choices=[
            "min_rank",
            "rank_sum",
            "reciprocal_rank_sum",
            "mean_score",
            "max_score",
            "mean_z_score",
            "max_z_score",
        ],
        default="min_rank",
    )
    parser.add_argument("--ks", nargs="+", default=["1,3,5,8,10,12,15,20,27"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    ks = _parse_ks(args.ks)
    if args.candidate_model is not None and args.candidate_models:
        raise SystemExit("--candidate-model and --candidate-models are mutually exclusive")
    if args.candidate_model is None and not args.candidate_models:
        raise SystemExit("one of --candidate-model or --candidate-models is required")
    if args.candidate_models and args.candidate_topk <= 0:
        raise SystemExit("--candidate-topk must be positive when --candidate-models is set")
    if args.candidate_topk < 0:
        raise SystemExit("--candidate-topk must be non-negative")
    if args.candidate_models:
        model_specs = [
            CandidateModelSpec(str(path), load_turn0_candidate_model(path), args.candidate_topk)
            for path in args.candidate_models
        ]
    else:
        model_specs = [
            CandidateModelSpec(str(args.candidate_model), load_turn0_candidate_model(args.candidate_model), None)
        ]
    input_rows = _read_jsonl(args.input)
    artifact_stage = infer_artifact_stage(input_rows)
    summary_rows, detail_rows = analyze_rows_union(input_rows, model_specs, ks, union_mode=args.union_mode)
    seat_summary_rows = summarize_details_by_seat(detail_rows, ks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": f"{artifact_stage}_candidate_coverage_v1",
        "artifact_stage": artifact_stage,
        "input": str(args.input),
        "candidate_model": str(args.candidate_model) if args.candidate_model else None,
        "candidate_models": [str(path) for path in args.candidate_models or ()],
        "candidate_topk": int(args.candidate_topk),
        "union_mode": args.union_mode,
        "records": summary_rows[0]["records"] if summary_rows else 0,
        "ks": summary_rows,
        "seat_ks": seat_summary_rows,
    }
    summary_path = args.summary_output or (args.output_dir / "coverage_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "coverage_summary.csv", summary_rows)
    _write_csv(args.output_dir / "coverage_seat_summary.csv", seat_summary_rows)
    _write_csv(args.output_dir / "coverage_rows.csv", detail_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
