"""Collect T2 runtime high-loss payloads for targeted action-value retraining.

The T2 runtime benchmark stores the original teacher payload in each
``results.jsonl`` row.  This script filters rows where the runtime-selected
action has a known teacher EV loss above a threshold and writes those original
payloads back out with runtime metadata attached.  The output can be fed
directly into ``ai/training/convert_action_value_teacher.py``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            yield line_no, json.loads(text)


def resolve_results_path(path: str) -> Path:
    raw = Path(path)
    if raw.is_dir():
        raw = raw / "results.jsonl"
    if not raw.exists():
        raise FileNotFoundError(raw)
    return raw


def parse_set_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = resolve_results_path(spec)
        return path.parent.name, path
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty set name in --set {spec!r}")
    return name, resolve_results_path(raw_path.strip())


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def payload_key(payload: dict[str, Any], teacher: dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload.get("source"),
        payload.get("source_line"),
        payload.get("source_global_index"),
        teacher.get("chosen_action_key"),
    )


def collect(args: argparse.Namespace) -> dict[str, Any]:
    sets = [parse_set_spec(spec) for spec in args.set]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    per_set: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[tuple[Any, ...]] = set()
    rows_out: list[dict[str, Any]] = []

    for set_name, results_path in sets:
        for line_no, row in iter_jsonl(results_path):
            counts["rows_seen"] += 1
            per_set[set_name]["rows_seen"] += 1

            payload = row.get("payload")
            teacher = row.get("teacher") or {}
            result = row.get("result") or {}
            if not isinstance(payload, dict):
                counts["missing_payload"] += 1
                per_set[set_name]["missing_payload"] += 1
                continue

            chosen_loss = numeric(teacher.get("chosen_ev_loss"))
            if chosen_loss is None:
                counts["unknown_loss"] += 1
                per_set[set_name]["unknown_loss"] += 1
                continue
            if chosen_loss < args.threshold:
                counts["below_threshold"] += 1
                per_set[set_name]["below_threshold"] += 1
                continue

            key = payload_key(payload, teacher)
            if not args.allow_duplicates and key in seen:
                counts["duplicate_skipped"] += 1
                per_set[set_name]["duplicate_skipped"] += 1
                continue
            seen.add(key)

            copied = dict(payload)
            copied["runtime_eval_set"] = set_name
            copied["runtime_source"] = str(results_path)
            copied["runtime_source_line"] = int(row.get("source_line") or line_no)
            copied["runtime_mode"] = result.get("mode")
            copied["runtime_chosen_ev_loss"] = chosen_loss
            copied["runtime_teacher_best_action_key"] = teacher.get("teacher_best_action_key")
            copied["runtime_chosen_action_key"] = teacher.get("chosen_action_key")
            copied["runtime_model_top1_action_key"] = teacher.get("model_top1_action_key")
            copied["runtime_model_top1_ev_loss"] = teacher.get("model_top1_ev_loss")
            copied["runtime_model_top1_action_idx"] = result.get("model_top1_action_idx")
            copied["runtime_selected_action_idx"] = (
                result.get("best", {}).get("action_idx")
                if isinstance(result.get("best"), dict)
                else None
            )
            copied["runtime_elapsed_ms"] = result.get("elapsed_ms")
            copied["runtime_refinement_error"] = result.get("refinement_error")
            copied["runtime_candidate_pool_size"] = result.get("candidate_pool_size")
            copied["runtime_exact_evaluated"] = result.get("exact_evaluated")
            copied["runtime_threshold"] = args.threshold
            rows_out.append(copied)
            counts["selected"] += 1
            per_set[set_name]["selected"] += 1

    with output.open("w", encoding="utf-8", newline="\n") as fh:
        for record in rows_out:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    losses = [float(row["runtime_chosen_ev_loss"]) for row in rows_out]
    summary = {
        "output": str(output),
        "threshold": args.threshold,
        "set_count": len(sets),
        "counts": dict(counts),
        "per_set": {key: dict(value) for key, value in sorted(per_set.items())},
        "loss_min": min(losses) if losses else None,
        "loss_mean": sum(losses) / len(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
    }
    if args.summary:
        summary_path = Path(args.summary)
    else:
        summary_path = output.with_suffix(output.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    summary["summary"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        action="append",
        required=True,
        help="NAME=runtime_dir_or_results_jsonl. May be repeated.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Keep duplicate source/source_line/chosen-action payloads.",
    )
    args = parser.parse_args()
    summary = collect(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
