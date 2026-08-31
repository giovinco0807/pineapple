"""Build action-value teacher JSONL from Rust T2 exact/capped-exact output.

The Rust T2 evaluator writes ranked exact candidates, but the action-value
converter needs the original decision context: board, opponent board, dealt
cards, position, and discards.  This script joins those pieces back together and
emits the standard candidate shape consumed by convert_action_value_teacher.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def metric(candidate: dict[str, Any], key: str, default: float = 0.0) -> float:
    metrics = candidate.get("metrics") or {}
    return float(metrics.get(key, default) or default)


def normalize_candidate(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    action = candidate.get("action") or {}
    metrics = candidate.get("metrics") or {}
    out = {
        "placements": action.get("placements") or [],
        "discard": action.get("discard"),
        "ev": metric(candidate, "score"),
        "score": metric(candidate, "score"),
        "raw_score": metric(candidate, "raw_score"),
        "royalty": metric(candidate, "royalty"),
        "bust_prob": metric(candidate, "bust_rate"),
        "bust_rate": metric(candidate, "bust_rate"),
        "fl_rate": metric(candidate, "fl_rate"),
        "fl_type_rates": dict(metrics.get("fl_type_rates") or {}),
        "samples": int(metrics.get("samples") or 0),
        "metric_source": metrics.get("source"),
        "forced_bust": bool(metrics.get("forced_bust", False)),
        "teacher_rank": int(rank),
    }
    return out


def action_cards(candidate: dict[str, Any]) -> set[str]:
    cards = {str(card) for card, _row in (candidate.get("placements") or [])}
    discard = candidate.get("discard")
    if discard not in (None, ""):
        cards.add(str(discard))
    return cards


def should_use_row_order(source_rows: list[dict[str, Any]], exact_rows: list[dict[str, Any]]) -> bool:
    """Detect concatenated chunk outputs whose record_index restarts per chunk."""

    if len(source_rows) != len(exact_rows):
        return False
    if any(row.get("global_index") is not None for row in exact_rows):
        return False
    record_indices = [row.get("record_index") for row in exact_rows]
    if any(index is None for index in record_indices):
        return True
    try:
        indices = [int(index) for index in record_indices]
    except (TypeError, ValueError):
        return True
    return len(set(indices)) != len(indices)


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = list(iter_jsonl(Path(args.source)))
    exact_rows = list(iter_jsonl(Path(args.exact)))
    use_row_order = bool(args.prefer_row_order or should_use_row_order(source_rows, exact_rows))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    candidate_count = 0
    invalid_rows = 0
    invalid_candidates = 0
    with out_path.open("w", encoding="utf-8") as out:
        for fallback_idx, exact in enumerate(exact_rows):
            source_idx = exact.get("global_index")
            if source_idx is None and not use_row_order:
                source_idx = exact.get("record_index")
            if source_idx is None:
                source_idx = fallback_idx
            source_idx = int(source_idx)
            if source_idx < 0 or source_idx >= len(source_rows):
                skipped += 1
                continue
            source = source_rows[source_idx]
            candidates = [
                normalize_candidate(candidate, rank)
                for rank, candidate in enumerate(exact.get("candidates") or [], start=1)
            ]
            if not candidates:
                skipped += 1
                continue
            dealt = {str(card) for card in (source.get("dealt") or [])}
            row_invalid_candidates = sum(1 for candidate in candidates if action_cards(candidate) != dealt)
            if row_invalid_candidates:
                invalid_rows += 1
                invalid_candidates += row_invalid_candidates
            row = {
                key: source.get(key)
                for key in (
                    "source",
                    "source_line",
                    "active_reasons",
                    "turn",
                    "board",
                    "opponent_board",
                    "dealt",
                    "known_discards",
                    "exclude",
                    "is_btn",
                    "position",
                    "future_hidden_deals",
                    "original_row_id",
                    "runtime_dataset",
                    "runtime_output",
                    "runtime_mode",
                    "runtime_selected_action_key",
                )
                if key in source
            }
            row.update(
                {
                    "source_global_index": source_idx,
                    "exact_record_index": exact.get("record_index"),
                    "eval_mode": args.eval_mode,
                    "teacher_eval_mode": args.eval_mode,
                    "best_idx": 0,
                    "n_candidates": len(candidates),
                    "legal_actions": exact.get("legal_actions"),
                    "evaluated_actions": exact.get("evaluated_actions"),
                    "requested_actions": exact.get("requested_actions"),
                    "exact_elapsed_ms": exact.get("elapsed_ms"),
                    "source_candidate_top_k": exact.get("source_candidate_top_k", args.source_candidate_top_k),
                    "cap": exact.get("cap", args.cap),
                    "candidates": candidates,
                }
            )
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            candidate_count += len(candidates)

    summary = {
        "source": str(args.source),
        "exact": str(args.exact),
        "output": str(out_path),
        "written": written,
        "skipped": skipped,
        "candidate_count": candidate_count,
        "join_mode": "row_order" if use_row_order else "record_index",
        "invalid_rows": invalid_rows,
        "invalid_candidates": invalid_candidates,
        "eval_mode": args.eval_mode,
        "cap": args.cap,
        "source_candidate_top_k": int(args.source_candidate_top_k),
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original T2 input JSONL with board/dealt context.")
    parser.add_argument("--exact", required=True, help="Rust T2 exact/capped output JSONL.")
    parser.add_argument("--output", required=True, help="Output teacher JSONL.")
    parser.add_argument("--eval-mode", default="exact_t2_capped")
    parser.add_argument("--cap", default="cap50")
    parser.add_argument("--source-candidate-top-k", type=int, default=10)
    parser.add_argument(
        "--prefer-row-order",
        action="store_true",
        help="Join exact rows to source rows by file order instead of record_index.",
    )
    args = parser.parse_args(argv)
    print(json.dumps(build(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
