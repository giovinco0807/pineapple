"""Build stable T2 oracle teacher labels from two capped-exact solver outputs.

The exact T2 path can be expensive, so we first compare two caps (for example
cap100 and cap200).  A row is emitted as a hard teacher label only when both
caps choose the same Top1 action.  Unstable rows are written separately for
later higher-cap or full-exact verification.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ai.tutor.run_t2_exact_oracle import (
    action_key,
    candidate_action,
    candidate_metric,
    candidate_score,
    load_jsonl,
)


def exact_best_action_key(record: dict[str, Any]) -> str:
    best = record.get("best") or {}
    return action_key(best.get("action") or {})


def exact_records_by_index(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for record in load_jsonl(path):
        records[int(record.get("record_index", -1))] = record
    return records


def cap_name(path: Path) -> str:
    name = path.name
    for part in name.split("_"):
        if part.startswith("cap") or part == "exact":
            return part
    return path.stem


def metric_block(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = candidate.get("metrics") or {}
    fl_types = metrics.get("fl_type_rates") or {}
    return {
        "ev": candidate_score(candidate),
        "score": candidate_score(candidate),
        "raw_score": float(metrics.get("raw_score", candidate_score(candidate)) or 0.0),
        "expected_royalty": float(metrics.get("royalty", 0.0) or 0.0),
        "bust_rate": candidate_metric(candidate, "bust"),
        "fl_rate": candidate_metric(candidate, "fl"),
        "fl_type_rates": {
            "qq": float(fl_types.get("qq", 0.0) or 0.0),
            "kk": float(fl_types.get("kk", 0.0) or 0.0),
            "aa": float(fl_types.get("aa", 0.0) or 0.0),
            "trips": float(fl_types.get("trips", 0.0) or 0.0),
        },
        "samples": int(metrics.get("samples", 0) or 0),
        "source": str(metrics.get("source", "")),
        "forced_bust": bool(metrics.get("forced_bust", False)),
    }


def teacher_candidate(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    action = candidate_action(candidate)
    metrics = metric_block(candidate)
    board = candidate.get("board") or {}
    return {
        **action,
        "oracle_rank": int(rank),
        "action_key": action_key(action),
        "board": board,
        "target_score": metrics["score"],
        "ev": metrics["ev"],
        "raw_score": metrics["raw_score"],
        "expected_royalty": metrics["expected_royalty"],
        "bust_prob": metrics["bust_rate"],
        "fl_rate": metrics["fl_rate"],
        "fl_type_rates": metrics["fl_type_rates"],
        "exact": metrics,
        "mc": {
            "avg_score": metrics["score"],
            "bust_rate": metrics["bust_rate"],
            "fl_rate": metrics["fl_rate"],
            "fl_type_rates": metrics["fl_type_rates"],
            "avg_royalty": metrics["expected_royalty"],
            "n_rollouts": metrics["samples"],
        },
    }


def source_context(source: dict[str, Any]) -> dict[str, Any]:
    position = str(source.get("position") or ("btn" if source.get("is_btn") else "bb"))
    return {
        "source": source.get("source"),
        "source_line": source.get("source_line"),
        "active_reasons": list(source.get("active_reasons") or source.get("reasons") or []),
        "turn": int(source.get("turn", 2)),
        "board": source.get("board") or {},
        "opponent_board": source.get("opponent_board") or source.get("board_opponent") or {},
        "dealt": list(source.get("dealt") or []),
        "known_discards": list(source.get("known_discards") or []),
        "exclude": list(source.get("exclude") or []),
        "is_btn": bool(source.get("is_btn", position == "btn")),
        "position": position,
    }


def build_label(
    *,
    source: dict[str, Any],
    weak: dict[str, Any],
    strong: dict[str, Any],
    weak_path: Path,
    strong_path: Path,
) -> dict[str, Any]:
    strong_candidates = list(strong.get("candidates") or [])
    candidates = [
        teacher_candidate(candidate, rank)
        for rank, candidate in enumerate(strong_candidates, start=1)
    ]
    strong_best = strong.get("best") or (strong_candidates[0] if strong_candidates else {})
    weak_best = weak.get("best") or {}
    strong_metrics = metric_block(strong_best)
    weak_metrics = metric_block(weak_best)
    best_key = exact_best_action_key(strong)
    best_idx = next(
        (idx for idx, candidate in enumerate(candidates) if str(candidate.get("action_key")) == best_key),
        0,
    )
    label = {
        **source_context(source),
        "record_index": int(strong.get("record_index", weak.get("record_index", -1))),
        "n_candidates": len(candidates),
        "candidates": candidates,
        "best_idx": int(best_idx),
        "eval_mode": "stable_capped_exact_t2",
        "candidate_source": "t2_stable_capped_exact",
        "exact_scope": "t2_capped_exact_top1_stable",
        "teacher_caps": {
            "weak": cap_name(weak_path),
            "strong": cap_name(strong_path),
            "weak_path": str(weak_path),
            "strong_path": str(strong_path),
        },
        "stability": {
            "same_top1": True,
            "top1_action_key": best_key,
            "weak_score": weak_metrics["score"],
            "strong_score": strong_metrics["score"],
            "score_abs_diff": abs(weak_metrics["score"] - strong_metrics["score"]),
            "weak_bust_rate": weak_metrics["bust_rate"],
            "strong_bust_rate": strong_metrics["bust_rate"],
            "bust_abs_diff": abs(weak_metrics["bust_rate"] - strong_metrics["bust_rate"]),
            "weak_samples": weak_metrics["samples"],
            "strong_samples": strong_metrics["samples"],
            "weak_elapsed_ms": float(weak.get("elapsed_ms", 0.0) or 0.0),
            "strong_elapsed_ms": float(strong.get("elapsed_ms", 0.0) or 0.0),
        },
    }
    return label


def unstable_record(
    *,
    source: dict[str, Any] | None,
    weak: dict[str, Any] | None,
    strong: dict[str, Any] | None,
    weak_path: Path,
    strong_path: Path,
    reason: str,
    record_index: int,
) -> dict[str, Any]:
    weak_key = exact_best_action_key(weak) if weak else None
    strong_key = exact_best_action_key(strong) if strong else None
    row: dict[str, Any] = {
        "record_index": record_index,
        "reason": reason,
        "teacher_caps": {
            "weak": cap_name(weak_path),
            "strong": cap_name(strong_path),
            "weak_path": str(weak_path),
            "strong_path": str(strong_path),
        },
        "weak_top1_action_key": weak_key,
        "strong_top1_action_key": strong_key,
        "weak_best": (weak or {}).get("best"),
        "strong_best": (strong or {}).get("best"),
    }
    if source is not None:
        row.update(source_context(source))
    return row


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_records = load_jsonl(Path(args.source))
    weak_path = Path(args.weak)
    strong_path = Path(args.strong)
    weak_records = exact_records_by_index(weak_path)
    strong_records = exact_records_by_index(strong_path)
    output = Path(args.output)
    unstable_output = Path(args.unstable_output) if args.unstable_output else output.with_suffix(".unstable.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    unstable_output.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    labels: list[dict[str, Any]] = []
    unstable: list[dict[str, Any]] = []
    record_indices = sorted(set(weak_records) | set(strong_records))
    if args.max_records > 0:
        record_indices = record_indices[: args.max_records]

    for record_index in record_indices:
        stats["records"] += 1
        weak = weak_records.get(record_index)
        strong = strong_records.get(record_index)
        source = source_records[record_index] if 0 <= record_index < len(source_records) else None
        if weak is None or strong is None or source is None:
            reason = "missing_weak" if weak is None else "missing_strong" if strong is None else "missing_source"
            unstable.append(
                unstable_record(
                    source=source,
                    weak=weak,
                    strong=strong,
                    weak_path=weak_path,
                    strong_path=strong_path,
                    reason=reason,
                    record_index=record_index,
                )
            )
            stats[reason] += 1
            continue
        stats["common_records"] += 1
        weak_key = exact_best_action_key(weak)
        strong_key = exact_best_action_key(strong)
        if weak_key != strong_key:
            unstable.append(
                unstable_record(
                    source=source,
                    weak=weak,
                    strong=strong,
                    weak_path=weak_path,
                    strong_path=strong_path,
                    reason="top1_changed_between_caps",
                    record_index=record_index,
                )
            )
            stats["top1_changed_between_caps"] += 1
            continue
        label = build_label(
            source=source,
            weak=weak,
            strong=strong,
            weak_path=weak_path,
            strong_path=strong_path,
        )
        labels.append(label)
        stats["stable_top1"] += 1
        stats[f"turn_{label['turn']}"] += 1

    with output.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False, separators=(",", ":")) + "\n")
    with unstable_output.open("w", encoding="utf-8") as f:
        for row in unstable:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "source": str(args.source),
        "weak": str(weak_path),
        "strong": str(strong_path),
        "output": str(output),
        "unstable_output": str(unstable_output),
        "max_records": int(args.max_records),
        "records_compared": int(stats["records"]),
        "common_records": int(stats["common_records"]),
        "stable_top1": int(stats["stable_top1"]),
        "unstable": len(unstable),
        "stable_top1_rate": float(stats["stable_top1"]) / max(int(stats["records"]), 1),
        "stable_top1_rate_common": float(stats["stable_top1"]) / max(int(stats["common_records"]), 1),
        "counts": dict(stats),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build stable T2 capped-exact teacher labels")
    parser.add_argument("--source", required=True, help="Original source/MC teacher JSONL")
    parser.add_argument("--weak", required=True, help="Lower-cap exact JSONL, e.g. cap100")
    parser.add_argument("--strong", required=True, help="Higher-cap exact JSONL, e.g. cap200")
    parser.add_argument("--output", required=True)
    parser.add_argument("--unstable-output", default="")
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(build(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
