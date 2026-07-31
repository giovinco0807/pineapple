"""Evaluate the configured T3 runtime candidate pool against saved exact EVs.

Unlike :mod:`ai.tutor.t3_runtime`, this command does not invoke the Rust exact
solver.  It builds the production candidate pool, then looks up each pooled
action in an already generated exact-teacher JSONL file.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.evaluate_t3_exact_teacher_models import stats
from ai.tutor.exact_late import action_key
from ai.tutor.t3_runtime import DEFAULT_CONFIG, T3UnionCandidatePool, observation_from_t3_payload


DEFAULT_TIE_TOLERANCE = 1.0e-9


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit > 0 and line_number > limit:
                break
            if line.strip():
                yield json.loads(line)


def candidate_action(candidate: dict[str, Any]) -> dict[str, Any]:
    nested = candidate.get("action")
    source = nested if isinstance(nested, dict) else candidate
    return {
        "placements": source.get("placements") or [],
        "discard": source.get("discard"),
    }


def candidate_ev(candidate: dict[str, Any]) -> float:
    for field in ("ev", "target_score", "score"):
        value = candidate.get(field)
        if value is not None:
            score = float(value)
            if not math.isfinite(score):
                raise ValueError(f"teacher candidate has non-finite {field}: {value!r}")
            return score

    exact = candidate.get("exact")
    if isinstance(exact, dict):
        for field in ("ev", "score"):
            value = exact.get(field)
            if value is not None:
                score = float(value)
                if not math.isfinite(score):
                    raise ValueError(f"teacher candidate has non-finite exact.{field}: {value!r}")
                return score
    raise ValueError("teacher candidate is missing ev/target_score/score")


def teacher_action_table(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for teacher_index, candidate in enumerate(row.get("candidates") or []):
        action = candidate_action(candidate)
        key = action_key(action)
        ev = candidate_ev(candidate)
        previous = table.get(key)
        if previous is None or ev > float(previous["ev"]):
            table[key] = {
                "action": action,
                "ev": ev,
                # The converted teacher preserves the Rust solver's complete
                # quality ordering, including its secondary tie breakers.
                "teacher_rank": int(teacher_index + 1),
            }
    if not table:
        raise ValueError("teacher row has no scored candidates")
    return table


def evaluate_teacher_row(
    row: dict[str, Any],
    pooler: Any,
    *,
    row_index: int,
    per_source_top_k: int | None = None,
    insurance_candidates: int | None = None,
    insurance_mode: str | None = None,
    near_full_expand_gap: int | None = None,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> dict[str, Any]:
    obs, _board, _opponent, _dealt, _exclude, position = observation_from_t3_payload(row)
    started = time.perf_counter()
    pool = pooler.build_pool(
        obs,
        position=position,
        per_source_top_k=per_source_top_k,
        insurance_candidates=insurance_candidates,
        insurance_mode=insurance_mode,
        near_full_expand_gap=near_full_expand_gap,
    )
    pool_elapsed_ms = (time.perf_counter() - started) * 1000.0

    teacher_by_key = teacher_action_table(row)
    teacher_best_ev = max(float(item["ev"]) for item in teacher_by_key.values())
    teacher_best_keys = {
        key
        for key, item in teacher_by_key.items()
        if teacher_best_ev - float(item["ev"]) <= tie_tolerance
    }

    seen_pool_keys: set[str] = set()
    matched: list[dict[str, Any]] = []
    unmatched_keys: list[str] = []
    pool_candidates: list[dict[str, Any]] = []
    for candidate in pool.get("candidates") or []:
        action = candidate_action(candidate)
        key = action_key(action)
        if key in seen_pool_keys:
            continue
        seen_pool_keys.add(key)
        teacher_item = teacher_by_key.get(key)
        compact = {
            "action_key": key,
            "action": action,
            "teacher_ev": None if teacher_item is None else float(teacher_item["ev"]),
            "teacher_rank": None if teacher_item is None else int(teacher_item["teacher_rank"]),
            "best_source_rank": candidate.get("best_source_rank"),
            "source_ranks": candidate.get("source_ranks") or {},
        }
        pool_candidates.append(compact)
        if teacher_item is None:
            unmatched_keys.append(key)
        else:
            matched.append(compact)

    pool_best = min(
        matched,
        key=lambda item: (
            -float(item["teacher_ev"]),
            int(item["teacher_rank"]),
            str(item["action_key"]),
        ),
        default=None,
    )
    pool_best_ev = None if pool_best is None else float(pool_best["teacher_ev"])
    ev_loss = None if pool_best_ev is None else max(0.0, teacher_best_ev - pool_best_ev)
    exact_best_hit = bool(teacher_best_keys.intersection(seen_pool_keys))

    return {
        "row_index": int(row_index),
        "source": row.get("source"),
        "source_line": row.get("source_line"),
        "target_index": row.get("target_index"),
        "position": position,
        "dealt": row.get("dealt") or [],
        "legal_actions": int(pool.get("legal_actions", len(teacher_by_key)) or 0),
        "teacher_candidate_count": len(teacher_by_key),
        "candidate_pool_size": int(pool.get("candidate_pool_size", len(seen_pool_keys)) or 0),
        "unique_pool_actions": len(seen_pool_keys),
        "matched_pool_actions": len(matched),
        "unmatched_pool_actions": len(unmatched_keys),
        "unmatched_pool_action_keys": unmatched_keys,
        "teacher_best_ev": teacher_best_ev,
        "teacher_best_action_keys": sorted(teacher_best_keys),
        "teacher_best_actions": [teacher_by_key[key]["action"] for key in sorted(teacher_best_keys)],
        "exact_best_hit": exact_best_hit,
        "pool_best_ev": pool_best_ev,
        "pool_best_action": None if pool_best is None else pool_best["action"],
        "ev_loss": ev_loss,
        "zero_ev_loss": bool(ev_loss is not None and ev_loss <= tie_tolerance),
        "pool_elapsed_ms": pool_elapsed_ms,
        "pool_policy": pool.get("pool_policy"),
        "per_source_top_k": pool.get("per_source_top_k"),
        "insurance_candidates": pool.get("insurance_candidates", 0),
        "insurance_mode": pool.get("insurance_mode"),
        "near_full_expand_gap": pool.get("near_full_expand_gap", 0),
        "sources": pool.get("sources") or [],
        "pool_candidates": pool_candidates,
    }


def pool_size_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": int(max(values)),
    }


def latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    losses = [float(row["ev_loss"]) for row in rows if row.get("ev_loss") is not None]
    pool_sizes = [int(row["candidate_pool_size"]) for row in rows]
    pool_latencies = [float(row["pool_elapsed_ms"]) for row in rows]
    total_pool_actions = sum(int(row["unique_pool_actions"]) for row in rows)
    unmatched_counts = [int(row["unmatched_pool_actions"]) for row in rows]
    hits = sum(1 for row in rows if row.get("exact_best_hit"))
    zero_loss = sum(1 for row in rows if row.get("zero_ev_loss"))
    evaluable = len(losses)
    return {
        "rows": len(rows),
        "evaluable_rows": evaluable,
        "exact_best_hits": hits,
        "exact_best_recall": hits / len(rows) if rows else 0.0,
        "zero_ev_loss_rows": zero_loss,
        "zero_ev_loss_rate": zero_loss / evaluable if evaluable else 0.0,
        "ev_loss_missing_rows": len(rows) - evaluable,
        "ev_loss": stats(losses),
        "pool_size": pool_size_stats(pool_sizes),
        "pool_elapsed_ms": latency_stats(pool_latencies),
        "unmatched_pool_actions": {
            "total": sum(unmatched_counts),
            "rows": sum(1 for count in unmatched_counts if count > 0),
            "max_per_row": max(unmatched_counts, default=0),
            "rate": sum(unmatched_counts) / total_pool_actions if total_pool_actions else 0.0,
        },
    }


def evaluate_teacher_rows(
    rows: Iterable[dict[str, Any]],
    pooler: Any,
    *,
    per_source_top_k: int | None = None,
    insurance_candidates: int | None = None,
    insurance_mode: str | None = None,
    near_full_expand_gap: int | None = None,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details = [
        evaluate_teacher_row(
            row,
            pooler,
            row_index=row_index,
            per_source_top_k=per_source_top_k,
            insurance_candidates=insurance_candidates,
            insurance_mode=insurance_mode,
            near_full_expand_gap=near_full_expand_gap,
            tie_tolerance=tie_tolerance,
        )
        for row_index, row in enumerate(rows)
    ]
    summary = summarize_bucket(details)
    summary["by_position"] = {
        position: summarize_bucket([row for row in details if row["position"] == position])
        for position in sorted({str(row["position"]) for row in details})
    }
    return summary, details


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the configured T3 runtime pool against saved exact teacher EVs (no Rust)",
    )
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pool-policy", default="fast")
    parser.add_argument("--pool-k", type=int, default=-1, help="Per-source TopK; -1 uses the config policy, 0 uses all legal actions")
    parser.add_argument("--insurance-candidates", type=int, default=-1, help="-1 uses the config policy")
    parser.add_argument("--insurance-mode", choices=("default", "general", "category"), default="default")
    parser.add_argument("--near-full-expand-gap", type=int, default=-1, help="-1 uses the config policy")
    parser.add_argument("--tie-tolerance", type=float, default=DEFAULT_TIE_TOLERANCE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--rows-output", default="")
    return parser


def main(argv: Iterable[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    if args.tie_tolerance < 0.0:
        raise SystemExit("--tie-tolerance must be non-negative")

    started = time.perf_counter()
    teacher_path = Path(args.teacher)
    config_path = Path(args.config)
    pooler = T3UnionCandidatePool(
        config_path=config_path,
        device=args.device,
        pool_policy=args.pool_policy,
    )
    per_source_top_k = None if int(args.pool_k) < 0 else int(args.pool_k)
    insurance_candidates = None if int(args.insurance_candidates) < 0 else int(args.insurance_candidates)
    insurance_mode = None if args.insurance_mode == "default" else str(args.insurance_mode)
    near_full_expand_gap = None if int(args.near_full_expand_gap) < 0 else int(args.near_full_expand_gap)

    summary, row_details = evaluate_teacher_rows(
        iter_jsonl(teacher_path, limit=max(0, int(args.limit))),
        pooler,
        per_source_top_k=per_source_top_k,
        insurance_candidates=insurance_candidates,
        insurance_mode=insurance_mode,
        near_full_expand_gap=near_full_expand_gap,
        tie_tolerance=float(args.tie_tolerance),
    )
    summary.update(
        {
            "teacher": str(teacher_path),
            "config": str(config_path),
            "pool_policy": str(args.pool_policy),
            "overrides": {
                "per_source_top_k": per_source_top_k,
                "insurance_candidates": insurance_candidates,
                "insurance_mode": insurance_mode,
                "near_full_expand_gap": near_full_expand_gap,
            },
            "tie_tolerance": float(args.tie_tolerance),
            "device": str(args.device),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "rust_exact_calls": 0,
        }
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.rows_output:
        rows_path = Path(args.rows_output)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        with rows_path.open("w", encoding="utf-8") as handle:
            for detail in row_details:
                handle.write(json.dumps(detail, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
