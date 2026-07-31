"""Audit T1/T2 hybrid shortlist recall on candidate-level holdouts.

The input format is the NumPy dataset produced by
``convert_action_value_teacher.py``.  This script measures the model-only
Top-K recall and the hybrid ``Top15 + insurance`` pool recall, then writes the
pool misses to JSONL for active data generation.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_reranker import ActionValueReranker, BlendedActionValueReranker
from ai.tutor.evaluate_action_value_dataset import FL_KEYS, group_ranges, predict
from ai.tutor.evaluate_teacher_model import resolve_model_path
from ai.tutor.hybrid_t1t2 import HybridConfig, _sync_refinement_candidates, build_shortlist


DEFAULT_MODEL = "ai/models/candidate_runs/tutor-route10-top20-prune-active8x-ft-20260523/model/action_value_best.pt"
TOP_NS = (1, 3, 5, 10, 15)


@dataclass
class AuditStats:
    top_hits: dict[int, int] = field(default_factory=lambda: {k: 0 for k in TOP_NS})
    top_regret_sum: dict[int, float] = field(default_factory=lambda: {k: 0.0 for k in TOP_NS})
    top_regret_max: dict[int, float] = field(default_factory=lambda: {k: 0.0 for k in TOP_NS})
    decisions: int = 0
    candidates: int = 0
    pool_hits: int = 0
    pool_regret_sum: float = 0.0
    pool_regret_max: float = 0.0
    sync_top3_hits: int = 0
    sync_top3_regret_sum: float = 0.0
    sync_top3_regret_max: float = 0.0
    sync_high_bust_violations: int = 0

    def as_dict(self) -> dict[str, Any]:
        denom = max(self.decisions, 1)
        return {
            "decisions": self.decisions,
            "candidates": self.candidates,
            "avg_candidates": self.candidates / denom,
            "top_recall": {f"top_{k}": self.top_hits[k] / denom for k in TOP_NS},
            "topk_exact_rerank_avg_regret": {
                f"top_{k}": self.top_regret_sum[k] / denom for k in TOP_NS
            },
            "topk_exact_rerank_max_regret": {f"top_{k}": self.top_regret_max[k] for k in TOP_NS},
            "hybrid_pool_recall": self.pool_hits / denom,
            "hybrid_pool_avg_regret": self.pool_regret_sum / denom,
            "hybrid_pool_max_regret": self.pool_regret_max,
            "sync_top3_guarded_recall": self.sync_top3_hits / denom,
            "sync_top3_guarded_avg_regret": self.sync_top3_regret_sum / denom,
            "sync_top3_guarded_max_regret": self.sync_top3_regret_max,
            "sync_high_bust_violations": self.sync_high_bust_violations,
        }


def load_jsonl_index(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists() or path.is_dir():
        return {}
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for index, line in enumerate(f):
            if line.strip():
                record = json.loads(line)
                records[index] = {
                    "source": record.get("source"),
                    "source_line": record.get("source_line"),
                    "turn": record.get("turn"),
                    "board": record.get("board"),
                    "opponent_board": record.get("opponent_board"),
                    "dealt": record.get("dealt"),
                    "known_discards": record.get("known_discards"),
                    "is_btn": record.get("is_btn"),
                    "position": record.get("position"),
                    "best_idx": record.get("best_idx"),
                    "eval_mode": record.get("eval_mode"),
                }
    return records


def read_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_audit_model(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    model_path = resolve_model_path(args.model)
    if getattr(args, "model_b", ""):
        model_b_path = resolve_model_path(args.model_b)
        model = BlendedActionValueReranker.from_checkpoints(
            model_path,
            model_b_path,
            model_b_weight=float(args.model_b_weight),
            map_location=device,
        )
        model.to(device)
        model.eval()
        return model, {
            "kind": "blend",
            "model_a": str(model_path),
            "model_b": str(model_b_path),
            "model_b_weight": float(args.model_b_weight),
            "model_a_weight": 1.0 - float(args.model_b_weight),
        }
    model = ActionValueReranker.from_checkpoint(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model, {"kind": "single", "model": str(model_path)}


def weighted_model_score(
    preds: dict[str, np.ndarray],
    *,
    bust_weight: float,
    fl_any_weight: float,
    fl_qq_weight: float,
    fl_kk_weight: float,
    fl_aa_weight: float,
    fl_trips_weight: float,
) -> np.ndarray:
    score = np.asarray(preds["score"], dtype=np.float32).copy()
    score -= float(bust_weight) * np.asarray(preds["bust"], dtype=np.float32)
    score += float(fl_any_weight) * np.asarray(preds["fl"], dtype=np.float32)
    type_weights = np.asarray(
        [fl_qq_weight, fl_kk_weight, fl_aa_weight, fl_trips_weight],
        dtype=np.float32,
    )
    if np.any(type_weights):
        score += np.asarray(preds["fl_types"], dtype=np.float32) @ type_weights
    return score


def adjusted_model_score(preds: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    return weighted_model_score(
        preds,
        bust_weight=args.bust_weight,
        fl_any_weight=args.fl_any_weight,
        fl_qq_weight=args.fl_qq_weight,
        fl_kk_weight=args.fl_kk_weight,
        fl_aa_weight=args.fl_aa_weight,
        fl_trips_weight=args.fl_trips_weight,
    )


def sync_adjusted_model_score(preds: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    return weighted_model_score(
        preds,
        bust_weight=args.bust_weight if args.sync_bust_weight is None else args.sync_bust_weight,
        fl_any_weight=args.fl_any_weight if args.sync_fl_any_weight is None else args.sync_fl_any_weight,
        fl_qq_weight=args.fl_qq_weight if args.sync_fl_qq_weight is None else args.sync_fl_qq_weight,
        fl_kk_weight=args.fl_kk_weight if args.sync_fl_kk_weight is None else args.sync_fl_kk_weight,
        fl_aa_weight=args.fl_aa_weight if args.sync_fl_aa_weight is None else args.sync_fl_aa_weight,
        fl_trips_weight=args.fl_trips_weight
        if args.sync_fl_trips_weight is None
        else args.sync_fl_trips_weight,
    )


def effective_sync_weights(args: argparse.Namespace) -> dict[str, float]:
    return {
        "bust": float(args.bust_weight if args.sync_bust_weight is None else args.sync_bust_weight),
        "fl_any": float(args.fl_any_weight if args.sync_fl_any_weight is None else args.sync_fl_any_weight),
        "qq": float(args.fl_qq_weight if args.sync_fl_qq_weight is None else args.sync_fl_qq_weight),
        "kk": float(args.fl_kk_weight if args.sync_fl_kk_weight is None else args.sync_fl_kk_weight),
        "aa": float(args.fl_aa_weight if args.sync_fl_aa_weight is None else args.sync_fl_aa_weight),
        "trips": float(
            args.fl_trips_weight if args.sync_fl_trips_weight is None else args.sync_fl_trips_weight
        ),
    }


def candidate_summary(
    local_idx: int,
    absolute_idx: int,
    action_indices: np.ndarray,
    scores: np.ndarray,
    adjusted: np.ndarray,
    sync_adjusted: np.ndarray,
    preds: dict[str, np.ndarray],
    rank: int,
    sync_rank: int | None = None,
) -> dict[str, Any]:
    return {
        "local_idx": int(local_idx),
        "absolute_idx": int(absolute_idx),
        "action_idx": int(action_indices[absolute_idx]),
        "teacher_score": float(scores[absolute_idx]),
        "model_score": float(adjusted[absolute_idx]),
        "model_rank": int(rank),
        "sync_score": float(sync_adjusted[absolute_idx]),
        "sync_rank": None if sync_rank is None else int(sync_rank),
        "predicted_bust": float(preds["bust"][absolute_idx]),
        "predicted_fl": float(preds["fl"][absolute_idx]),
        "predicted_fl_types": {
            key: float(preds["fl_types"][absolute_idx, i]) for i, key in enumerate(FL_KEYS)
        },
    }


def miss_summary(
    *,
    kind: str,
    top_k: int,
    regret: float,
    data_dir: Path,
    group_id: int,
    turn: int,
    start: int,
    end: int,
    teacher_best_local: int,
    model_top_local: int,
    order_local: np.ndarray,
    action_indices: np.ndarray,
    scores: np.ndarray,
    adjusted: np.ndarray,
    sync_adjusted: np.ndarray,
    preds: dict[str, np.ndarray],
    rank_by_local: dict[int, int],
    sync_rank_by_local: dict[int, int],
    source_record: dict[str, Any] | None,
) -> dict[str, Any]:
    kept_local = [int(idx) for idx in order_local[: min(top_k, len(order_local))]]
    return {
        "kind": kind,
        "top_k": int(top_k),
        "regret": float(regret),
        "data": str(data_dir),
        "group_id": int(group_id),
        "turn": int(turn),
        "n_candidates": int(end - start),
        "teacher_best": candidate_summary(
            teacher_best_local,
            start + teacher_best_local,
            action_indices,
            scores,
            adjusted,
            sync_adjusted,
            preds,
            rank_by_local[teacher_best_local],
            sync_rank_by_local[teacher_best_local],
        ),
        "model_top1": candidate_summary(
            model_top_local,
            start + model_top_local,
            action_indices,
            scores,
            adjusted,
            sync_adjusted,
            preds,
            1,
            sync_rank_by_local[model_top_local],
        ),
        "kept_action_indices": [int(action_indices[start + idx]) for idx in kept_local],
        "source_record": source_record,
    }


def audit_dataset(args: argparse.Namespace, data_dir: Path, output_dir: Path, model: ActionValueReranker) -> dict[str, Any]:
    metadata = read_metadata(data_dir)
    states_mm = np.load(data_dir / "states.npy", mmap_mode="r")
    n_samples = min(int(metadata.get("n_samples", states_mm.shape[0])), states_mm.shape[0])
    if args.limit_samples:
        n_samples = min(n_samples, int(args.limit_samples))

    states = states_mm[:n_samples]
    scores = np.asarray(np.load(data_dir / "scores.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    turns = np.asarray(np.load(data_dir / "turns.npy", mmap_mode="r")[:n_samples], dtype=np.int16)
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    action_indices = np.asarray(np.load(data_dir / "action_indices.npy", mmap_mode="r")[:n_samples], dtype=np.int64)

    device = next(model.parameters()).device
    preds = predict(model, states, turns, device, args.batch_size)
    adjusted = adjusted_model_score(preds, args)
    sync_adjusted = sync_adjusted_model_score(preds, args)

    config = HybridConfig(
        shortlist_k=args.shortlist_k,
        insurance_k=args.insurance_k,
        sync_exact_k=args.sync_exact_k,
        max_pool=args.max_pool,
        time_budget_ms=args.time_budget_ms,
        high_bust_threshold=args.high_bust_threshold,
        enable_sync_refinement=False,
    )
    source_records = load_jsonl_index(Path(metadata["source"]) if metadata.get("source") else None)
    stats = AuditStats()
    misses: list[dict[str, Any]] = []
    top1_misses: list[dict[str, Any]] = []
    top3_misses: list[dict[str, Any]] = []
    high_regret_misses: list[dict[str, Any]] = []

    for start, end in group_ranges(group_ids):
        if end <= start:
            continue
        local_scores = scores[start:end]
        local_adjusted = adjusted[start:end]
        local_sync_adjusted = sync_adjusted[start:end]
        order_local = np.argsort(-local_adjusted)
        sync_order_local = np.argsort(-local_sync_adjusted)
        teacher_best_local = int(np.argmax(local_scores))
        teacher_best_abs = start + teacher_best_local
        teacher_best_action = int(action_indices[teacher_best_abs])
        teacher_best_score = float(scores[teacher_best_abs])

        stats.decisions += 1
        stats.candidates += int(end - start)

        rank_by_local = {int(local_idx): int(rank) + 1 for rank, local_idx in enumerate(order_local)}
        sync_rank_by_local = {int(local_idx): int(rank) + 1 for rank, local_idx in enumerate(sync_order_local)}
        model_top = int(order_local[0])
        source_record = source_records.get(int(group_ids[start]))
        for k in TOP_NS:
            kept_local = [int(idx) for idx in order_local[: min(k, len(order_local))]]
            kept_actions = {int(action_indices[start + idx]) for idx in kept_local}
            if teacher_best_action in kept_actions:
                stats.top_hits[k] += 1
            best_kept = max(float(scores[start + idx]) for idx in kept_local) if kept_local else float("-inf")
            regret = max(0.0, teacher_best_score - best_kept)
            stats.top_regret_sum[k] += regret
            stats.top_regret_max[k] = max(stats.top_regret_max[k], regret)
            if regret > 0.0 and k in (1, 3):
                row = miss_summary(
                    kind=f"top{k}_miss",
                    top_k=k,
                    regret=regret,
                    data_dir=data_dir,
                    group_id=int(group_ids[start]),
                    turn=int(turns[start]),
                    start=start,
                    end=end,
                    teacher_best_local=teacher_best_local,
                    model_top_local=model_top,
                    order_local=order_local,
                    action_indices=action_indices,
                    scores=scores,
                    adjusted=adjusted,
                    sync_adjusted=sync_adjusted,
                    preds=preds,
                    rank_by_local=rank_by_local,
                    sync_rank_by_local=sync_rank_by_local,
                    source_record=source_record,
                )
                if k == 1:
                    top1_misses.append(row)
                else:
                    top3_misses.append(row)
            if k == 1 and regret >= float(args.high_regret_threshold):
                high_regret_misses.append(
                    miss_summary(
                        kind="high_regret_top1_miss",
                        top_k=k,
                        regret=regret,
                        data_dir=data_dir,
                        group_id=int(group_ids[start]),
                        turn=int(turns[start]),
                        start=start,
                        end=end,
                        teacher_best_local=teacher_best_local,
                        model_top_local=model_top,
                        order_local=order_local,
                        action_indices=action_indices,
                        scores=scores,
                        adjusted=adjusted,
                        sync_adjusted=sync_adjusted,
                        preds=preds,
                        rank_by_local=rank_by_local,
                        sync_rank_by_local=sync_rank_by_local,
                        source_record=source_record,
                    )
                )

        candidates: list[dict[str, Any]] = []
        for local_idx in range(end - start):
            absolute_idx = start + local_idx
            fl_types = preds["fl_types"][absolute_idx]
            candidates.append(
                {
                    "action_idx": int(action_indices[absolute_idx]),
                    "model_score": float(adjusted[absolute_idx]),
                    "model_rank": rank_by_local[local_idx],
                    "sync_score": float(sync_adjusted[absolute_idx]),
                    "sync_rank": sync_rank_by_local[local_idx],
                    "predicted_bust": float(preds["bust"][absolute_idx]),
                    "predicted_fl": float(preds["fl"][absolute_idx]),
                    "predicted_fl_types": {
                        key: float(fl_types[i]) for i, key in enumerate(FL_KEYS)
                    },
                }
            )

        pool = build_shortlist(candidates, config=config)
        for item in pool:
            item["refinement_rank"] = int(item.get("sync_rank", item.get("refinement_rank", item["model_rank"])))
        pool_actions = {int(item["action_idx"]) for item in pool}
        if teacher_best_action in pool_actions:
            stats.pool_hits += 1
        else:
            best_pool_score = max(
                (float(scores[start + idx]) for idx in range(end - start) if int(action_indices[start + idx]) in pool_actions),
                default=float("-inf"),
            )
            regret = max(0.0, teacher_best_score - best_pool_score)
            stats.pool_regret_sum += regret
            stats.pool_regret_max = max(stats.pool_regret_max, regret)
            misses.append(
                {
                    "data": str(data_dir),
                    "group_id": int(group_ids[start]),
                    "turn": int(turns[start]),
                    "n_candidates": int(end - start),
                    "teacher_best": candidate_summary(
                        teacher_best_local,
                        teacher_best_abs,
                        action_indices,
                        scores,
                        adjusted,
                        sync_adjusted,
                        preds,
                        rank_by_local[teacher_best_local],
                        sync_rank_by_local[teacher_best_local],
                    ),
                    "model_top1": candidate_summary(
                        model_top,
                        start + model_top,
                        action_indices,
                        scores,
                        adjusted,
                        sync_adjusted,
                        preds,
                        1,
                        sync_rank_by_local[model_top],
                    ),
                    "top15_action_indices": [
                        int(action_indices[start + idx]) for idx in order_local[: min(args.shortlist_k, len(order_local))]
                    ],
                    "hybrid_pool_action_indices": sorted(pool_actions),
                    "hybrid_pool_regret": regret,
                    "source_record": source_record,
                }
            )

        sync_candidates = _sync_refinement_candidates(pool, config, turn=int(turns[start]))
        sync_actions = {int(item["action_idx"]) for item in sync_candidates}
        if teacher_best_action in sync_actions:
            stats.sync_top3_hits += 1
        best_sync_score = max(
            (float(scores[start + idx]) for idx in range(end - start) if int(action_indices[start + idx]) in sync_actions),
            default=float("-inf"),
        )
        sync_regret = max(0.0, teacher_best_score - best_sync_score)
        stats.sync_top3_regret_sum += sync_regret
        stats.sync_top3_regret_max = max(stats.sync_top3_regret_max, sync_regret)
        has_safe = any(float(item["predicted_bust"]) < args.high_bust_threshold for item in pool)
        if has_safe and any(float(item["predicted_bust"]) >= args.high_bust_threshold for item in sync_candidates):
            stats.sync_high_bust_violations += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = data_dir.name
    misses_path = output_dir / f"{stem}.misses.jsonl"
    with misses_path.open("w", encoding="utf-8") as f:
        for item in misses:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    top1_misses.sort(key=lambda item: float(item["regret"]), reverse=True)
    top3_misses.sort(key=lambda item: float(item["regret"]), reverse=True)
    high_regret_misses.sort(key=lambda item: float(item["regret"]), reverse=True)
    top1_misses_path = output_dir / f"{stem}.top1_misses.jsonl"
    top3_misses_path = output_dir / f"{stem}.top3_misses.jsonl"
    high_regret_path = output_dir / f"{stem}.high_regret_top1_misses.jsonl"
    for path, rows in (
        (top1_misses_path, top1_misses),
        (top3_misses_path, top3_misses),
        (high_regret_path, high_regret_misses),
    ):
        with path.open("w", encoding="utf-8") as f:
            for item in rows[: int(args.weak_spots)]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "data": str(data_dir),
        "metadata": metadata,
        "n_samples": int(n_samples),
        "shortlist_k": args.shortlist_k,
        "insurance_k": args.insurance_k,
        "max_pool": args.max_pool,
        "sync_exact_k": args.sync_exact_k,
        "high_bust_threshold": args.high_bust_threshold,
        "bust_weight": args.bust_weight,
        "fl_any_weight": args.fl_any_weight,
        "fl_type_weights": {
            "qq": args.fl_qq_weight,
            "kk": args.fl_kk_weight,
            "aa": args.fl_aa_weight,
            "trips": args.fl_trips_weight,
        },
        "sync_score_weights": effective_sync_weights(args),
        "stats": stats.as_dict(),
        "miss_count": len(misses),
        "misses_path": str(misses_path),
        "top1_miss_count": len(top1_misses),
        "top3_miss_count": len(top3_misses),
        "high_regret_top1_miss_count": len(high_regret_misses),
        "top1_misses_path": str(top1_misses_path),
        "top3_misses_path": str(top3_misses_path),
        "high_regret_top1_misses_path": str(high_regret_path),
    }
    (output_dir / f"{stem}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Hybrid T1/T2 Shortlist Audit",
        "",
        f"- model: `{report['model']}`",
        f"- output: `{report['output']}`",
        "",
        "| data | decisions | top1 | top3 | top5 | top10 | top15 | top15+insurance | sync top3 | top1 misses | top3 misses | pool misses | max pool regret | bust violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["datasets"]:
        stats = item["stats"]
        top = stats["top_recall"]
        lines.append(
            "| "
            + " | ".join(
                [
                    Path(item["data"]).name,
                    str(stats["decisions"]),
                    f"{top['top_1']:.1%}",
                    f"{top['top_3']:.1%}",
                    f"{top['top_5']:.1%}",
                    f"{top['top_10']:.1%}",
                    f"{top['top_15']:.1%}",
                    f"{stats['hybrid_pool_recall']:.1%}",
                    f"{stats['sync_top3_guarded_recall']:.1%}",
                    str(item.get("top1_miss_count", 0)),
                    str(item.get("top3_miss_count", 0)),
                    str(item["miss_count"]),
                    f"{stats['hybrid_pool_max_regret']:.3f}",
                    str(stats["sync_high_bust_violations"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit hybrid T1/T2 Top15+insurance shortlist recall")
    parser.add_argument("--data", action="append", required=True, help="Candidate-level dataset directory")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-b", default="", help="Optional second checkpoint for inference-time score/component blend")
    parser.add_argument("--model-b-weight", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--shortlist-k", type=int, default=15)
    parser.add_argument("--insurance-k", type=int, default=5)
    parser.add_argument("--sync-exact-k", type=int, default=3)
    parser.add_argument("--max-pool", type=int, default=20)
    parser.add_argument("--time-budget-ms", type=int, default=5000)
    parser.add_argument("--high-bust-threshold", type=float, default=0.999)
    parser.add_argument("--bust-weight", type=float, default=0.0)
    parser.add_argument("--fl-any-weight", type=float, default=0.0)
    parser.add_argument("--fl-qq-weight", type=float, default=0.0)
    parser.add_argument("--fl-kk-weight", type=float, default=0.0)
    parser.add_argument("--fl-aa-weight", type=float, default=0.0)
    parser.add_argument("--fl-trips-weight", type=float, default=0.0)
    parser.add_argument("--sync-bust-weight", type=float, default=None)
    parser.add_argument("--sync-fl-any-weight", type=float, default=None)
    parser.add_argument("--sync-fl-qq-weight", type=float, default=None)
    parser.add_argument("--sync-fl-kk-weight", type=float, default=None)
    parser.add_argument("--sync-fl-aa-weight", type=float, default=None)
    parser.add_argument("--sync-fl-trips-weight", type=float, default=None)
    parser.add_argument("--high-regret-threshold", type=float, default=5.0)
    parser.add_argument("--weak-spots", type=int, default=50)
    args = parser.parse_args(list(argv) if argv is not None else None)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, model_report = load_audit_model(args, device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [audit_dataset(args, Path(data), output_dir, model) for data in args.data]
    report = {
        "model": model_report.get("model", model_report.get("model_a")),
        "model_report": model_report,
        "device": str(device),
        "output": str(output_dir),
        "datasets": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output_dir / "summary.md", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
