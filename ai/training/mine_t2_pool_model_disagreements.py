"""Mine T2 pool-reranker disagreement and high-loss groups.

The existing miss-weighting path only needs ``dataset``, ``group_id``,
``target_k`` and ``ev_loss``.  This helper writes that shape for cases where
two or more pool rerankers choose different Top1 actions, or where any supplied
pool reranker loses EV against the teacher best.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_t2_selector_feature_union import parse_feature_selector
from ai.training.train_t2_selector_feature_pool_reranker import load_dataset, predict_dataset
from ai.training.train_t2_selector_feature_ranker import (
    parse_lgbm_selector,
    parse_model,
    parse_named_path,
    parse_selector,
)


@dataclass(frozen=True)
class PoolModelSpec:
    name: str
    path: Path
    kind: str
    pool_k: int
    scope: str
    pool_local_features: bool = False


def parse_pool_model(value: str) -> PoolModelSpec:
    name, rest = value.split("=", 1)
    items = [part.strip() for part in rest.split(",")]
    if len(items) not in {4, 5}:
        raise ValueError("--pool-model must be name=path,kind,pool_k,scope[,pool_local]")
    scope = items[3].lower()
    if scope not in {"both", "selector", "state"}:
        raise ValueError("pool model scope must be both, selector, or state")
    return PoolModelSpec(
        name=name.strip(),
        path=Path(items[0]),
        kind=items[1],
        pool_k=int(items[2]),
        scope=scope,
        pool_local_features=(len(items) == 5 and items[4].lower() in {"1", "true", "yes", "pool_local"}),
    )


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_record_data(values: list[str]) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {}
    for value in values:
        spec = parse_named_path(value)
        records[spec.name] = list(iter_jsonl(spec.path))
    return records


def compact_action(candidate: dict | None) -> dict | None:
    if not candidate:
        return None
    action = candidate.get("action") if isinstance(candidate.get("action"), dict) else candidate
    return {
        "placements": action.get("placements") or [],
        "discard": action.get("discard"),
    }


def compact_source_context(row: dict | None) -> dict | None:
    if not row:
        return None
    best = row.get("best") or {}
    branch = row.get("branch") or {}
    return {
        "source": row.get("source"),
        "source_line": row.get("source_line"),
        "position": row.get("position"),
        "is_btn": row.get("is_btn"),
        "turn": row.get("turn"),
        "board": row.get("board"),
        "opponent_board": row.get("opponent_board"),
        "dealt": row.get("dealt"),
        "known_discards": row.get("known_discards"),
        "future_hidden_deals": row.get("future_hidden_deals"),
        "branch": {
            key: branch.get(key)
            for key in (
                "root_index",
                "route_index",
                "branch_id",
                "target_position",
                "target_t0_rank",
                "target_t1_rank",
            )
            if key in branch
        },
        "root_deals": row.get("root_deals"),
        "source_best_idx": row.get("best_idx"),
        "source_best_action": compact_action(best.get("action") if isinstance(best.get("action"), dict) else best),
        "source_best_score": best.get("model_t3_value_score") or best.get("t3_model_ev"),
        "source_candidate_count": row.get("candidate_count") or len(row.get("candidates") or []),
    }


def compact_teacher_context(row: dict | None, best_local_index: int | None) -> dict | None:
    if not row:
        return None
    candidates = row.get("candidates") or []
    best = None
    if best_local_index is not None and 0 <= int(best_local_index) < len(candidates):
        best = candidates[int(best_local_index)]
    return {
        "source_global_index": row.get("source_global_index"),
        "exact_record_index": row.get("exact_record_index"),
        "position": row.get("position"),
        "is_btn": row.get("is_btn"),
        "turn": row.get("turn"),
        "board": row.get("board"),
        "opponent_board": row.get("opponent_board"),
        "dealt": row.get("dealt"),
        "known_discards": row.get("known_discards"),
        "cap": row.get("cap"),
        "source_candidate_top_k": row.get("source_candidate_top_k"),
        "exact_elapsed_ms": row.get("exact_elapsed_ms"),
        "n_candidates": row.get("n_candidates") or len(candidates),
        "teacher_best_action": compact_action(best),
        "teacher_best_score": best.get("score") if isinstance(best, dict) else None,
        "teacher_best_bust": best.get("bust_rate") if isinstance(best, dict) else None,
        "teacher_best_fl": best.get("fl_rate") if isinstance(best, dict) else None,
    }


def model_top(pred: np.ndarray, start: int, end: int) -> int:
    return start + int(np.argmax(pred[start:end]))


def summarize(rows: list[dict], all_groups: int) -> dict:
    by_reason = Counter(reason for row in rows for reason in row.get("reasons", []))
    by_dataset: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        dataset = str(row.get("dataset"))
        by_dataset[dataset]["rows"] += 1
        for reason in row.get("reasons", []):
            by_dataset[dataset][reason] += 1
    losses = [float(row["ev_loss"]) for row in rows]
    return {
        "groups": int(all_groups),
        "rows": len(rows),
        "datasets": dict(Counter(str(row.get("dataset")) for row in rows)),
        "reasons": dict(by_reason),
        "by_dataset": {k: dict(v) for k, v in sorted(by_dataset.items())},
        "ev_loss_mean": float(np.mean(losses)) if losses else 0.0,
        "ev_loss_max": float(np.max(losses)) if losses else 0.0,
    }


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_specs = [parse_named_path(value) for value in args.data]
    model_specs = {spec.name: spec for spec in [parse_model(value) for value in args.model]}
    selectors = [parse_selector(value) for value in args.selector]
    lgbm_selectors = [parse_lgbm_selector(value) for value in args.lgbm_selector]
    feature_selectors = [parse_feature_selector(value) for value in args.feature_selector]
    pool_specs = [parse_pool_model(value) for value in args.pool_model]
    source_records = load_record_data(args.source_data)
    teacher_records = load_record_data(args.teacher_data)

    loaded_models = {name: joblib.load(spec.path) for name, spec in model_specs.items()}
    loaded_lgbm = {name: joblib.load(path) for name, path in lgbm_selectors}
    loaded_feature = {spec.name: joblib.load(spec.path) for spec in feature_selectors}
    loaded_pool = {spec.name: joblib.load(spec.path) for spec in pool_specs}

    rows: list[dict] = []
    all_groups = 0
    for data_spec in data_specs:
        ds = load_dataset(
            data_spec,
            model_specs,
            selectors,
            lgbm_selectors,
            args.npy_selector,
            loaded_models,
            loaded_lgbm,
            feature_selectors,
            loaded_feature,
            include_base=not args.no_base,
        )
        pred_by_model = {
            spec.name: predict_dataset(
                ds,
                loaded_pool[spec.name],
                spec.kind,
                spec.pool_k,
                spec.scope,
                spec.pool_local_features,
            )
            for spec in pool_specs
        }
        for group_id, (start, end) in enumerate(ds.bounds):
            all_groups += 1
            teacher_best = start + int(np.argmax(ds.scores[start:end]))
            teacher_score = float(ds.scores[teacher_best])
            choices: dict[str, dict] = {}
            unique_choices: set[int] = set()
            losses: list[float] = []
            hits = 0
            for spec in pool_specs:
                top = model_top(pred_by_model[spec.name], start, end)
                loss = max(0.0, teacher_score - float(ds.scores[top]))
                hits += int(top == teacher_best)
                losses.append(loss)
                unique_choices.add(top)
                choices[spec.name] = {
                    "index": int(top),
                    "local_index": int(top - start),
                    "score": float(ds.scores[top]),
                    "ev_loss": float(loss),
                    "hit": bool(top == teacher_best),
                }

            reasons: list[str] = []
            if len(unique_choices) > 1:
                reasons.append("model_disagreement")
            if hits > 0 and hits < len(pool_specs):
                reasons.append("one_model_hits")
            if any(loss >= float(args.min_model_loss) for loss in losses):
                reasons.append("model_high_loss")
            if not reasons:
                continue
            ev_loss = max(losses)
            if ev_loss < float(args.min_ev_loss):
                continue
            row = {
                "dataset": data_spec.name,
                "group_id": int(group_id),
                "target_k": int(args.target_k),
                "ev_loss": float(ev_loss),
                "teacher_best_index": int(teacher_best),
                "teacher_best_local_index": int(teacher_best - start),
                "teacher_best_score": teacher_score,
                "size": int(end - start),
                "reasons": reasons,
                "choices": choices,
            }
            if data_spec.name in source_records:
                records = source_records[data_spec.name]
                if group_id < len(records):
                    row["source_context"] = compact_source_context(records[group_id])
            if data_spec.name in teacher_records:
                records = teacher_records[data_spec.name]
                if group_id < len(records):
                    row["teacher_context"] = compact_teacher_context(records[group_id], teacher_best - start)
            rows.append(row)

    rows.sort(key=lambda row: (float(row["ev_loss"]), len(row.get("reasons", []))), reverse=True)
    rows_path = out_dir / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    summary = {
        "data": {spec.name: str(spec.path) for spec in data_specs},
        "pool_models": {
            spec.name: {
                "path": str(spec.path),
                "kind": spec.kind,
                "pool_k": spec.pool_k,
                "scope": spec.scope,
            }
            for spec in pool_specs
        },
        "target_k": int(args.target_k),
        "source_data": {name: len(records) for name, records in source_records.items()},
        "teacher_data": {name: len(records) for name, records in teacher_records.items()},
        "min_ev_loss": float(args.min_ev_loss),
        "min_model_loss": float(args.min_model_loss),
        "rows_path": str(rows_path),
        "summary": summarize(rows, all_groups),
        "preview": rows[: min(len(rows), int(args.preview))],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "rows": len(rows)}, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="append", required=True, help="name=path")
    parser.add_argument("--model", action="append", default=[], help="name=path,target")
    parser.add_argument("--selector", action="append", default=[], help="name=model_a+model_b,gamma")
    parser.add_argument("--lgbm-selector", action="append", default=[], help="name=path")
    parser.add_argument("--feature-selector", action="append", default=[], help="name=path,selector|state")
    parser.add_argument("--pool-model", action="append", required=True, help="name=path,kind,pool_k,scope[,pool_local]")
    parser.add_argument(
        "--npy-selector",
        action="append",
        default=[],
        help="Selector score stored as <data_dir>/selector_scores/<name>.npy",
    )
    parser.add_argument("--source-data", action="append", default=[], help="Optional original source JSONL, name=path")
    parser.add_argument("--teacher-data", action="append", default=[], help="Optional exact teacher JSONL, name=path")
    parser.add_argument("--target-k", type=int, default=0)
    parser.add_argument("--min-ev-loss", type=float, default=0.0)
    parser.add_argument("--min-model-loss", type=float, default=0.25)
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--preview", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if len(args.pool_model) < 1:
        raise ValueError("at least one --pool-model value is required")
    run(args)


if __name__ == "__main__":
    main()
