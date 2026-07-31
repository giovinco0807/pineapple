"""Convert Rust T3/T4 exact solver output into action-value teacher JSONL.

``t3_exact_solver`` writes compact per-position results keyed by the input
line index.  The existing action-value converter expects each record to carry
the original board/dealt context and flattened candidate metrics, so this
adapter joins the Rust output back to its input JSONL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


FL_TYPE_KEYS = ("qq", "kk", "aa", "trips")


def _board_card_count(raw: Any) -> int:
    if not isinstance(raw, dict):
        return 0
    return sum(
        len(raw.get(name) or [])
        for name in ("top", "middle", "mid", "bottom", "bot")
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def metric_float(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    return float(value if value is not None else default)


def convert_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    action = candidate.get("action") or {}
    metrics = candidate.get("metrics") or {}
    fl_type_rates = {
        key: metric_float(metrics.get("fl_type_rates") or {}, key)
        for key in FL_TYPE_KEYS
    }
    score = metric_float(metrics, "score", metric_float(metrics, "ev"))
    bust_rate = metric_float(metrics, "bust_rate")
    fl_rate = metric_float(metrics, "fl_rate")
    royalty = metric_float(metrics, "royalty")
    raw_score = metric_float(metrics, "raw_score", score)
    samples = int(metrics.get("samples") or 0)
    return {
        "placements": list(action.get("placements") or []),
        "discard": action.get("discard") or "",
        "target_score": score,
        "ev": score,
        "expected_royalty": royalty,
        "raw_score": raw_score,
        "bust_prob": bust_rate,
        "fl_rate": fl_rate,
        "fl_type_rates": fl_type_rates,
        "board": candidate.get("board") or {},
        "exact": {
            **metrics,
            "fl_type_rates": fl_type_rates,
            "samples": samples,
        },
        "mc": {
            "avg_score": score,
            "bust_rate": bust_rate,
            "fl_rate": fl_rate,
            "fl_type_rates": fl_type_rates,
            "avg_royalty": royalty,
            "n_rollouts": samples,
        },
    }


def convert_record(result: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    candidates = [convert_candidate(c) for c in (result.get("candidates") or [])]
    turn = int(result.get("turn", source.get("turn", 3)))
    total_samples = sum(int((c.get("exact") or {}).get("samples") or 0) for c in candidates)
    if turn == 3:
        exact_scope = "t3_self_board_all_t4_draws_best_t4"
    elif any(
        bool((candidate.get("exact") or {}).get("opponent_response"))
        or (candidate.get("exact") or {}).get("source") == "exact_hu_response"
        for candidate in candidates
    ):
        exact_scope = "t4_all_opponent_draws_best_response_given_exclude"
    else:
        exact_scope = "t4_terminal_actions"
    opponent_cards = _board_card_count(
        source.get("opponent_board") or source.get("board_opponent") or {}
    )
    return {
        "source": source.get("source"),
        "source_line": source.get("source_line"),
        "target_index": int(result.get("record_index", source.get("target_index", 0))),
        "active_reasons": list(source.get("reasons", source.get("active_reasons", [])) or []),
        "turn": turn,
        "board": source.get("board") or {},
        "opponent_board": source.get("opponent_board") or source.get("board_opponent") or {},
        "dealt": list(source.get("dealt") or []),
        "known_discards": list(source.get("known_discards") or []),
        "exclude": list(source.get("exclude") or []),
        "is_btn": bool(source.get("is_btn", True)),
        "position": source.get("position") or ("btn" if source.get("is_btn", True) else "bb"),
        "n_candidates": len(candidates),
        "candidates": candidates,
        "best_idx": 0,
        "eval_mode": "exact",
        "elapsed_s": metric_float(result, "elapsed_ms") / 1000.0,
        "candidate_source": "rust_t3_exact_solver",
        "exact_scope": exact_scope,
        "hu_exact": bool(turn == 4 and opponent_cards == 13),
        "information_model": (
            "exclude_conditioned_physical_or_uniform_unspecified"
            if exact_scope == "t4_all_opponent_draws_best_response_given_exclude"
            else "terminal_public_state"
        ),
        "total_exact_samples": int(total_samples),
        "original_n_actions": int(result.get("legal_actions") or len(candidates)),
        "original_evaluated_actions": int(result.get("legal_actions") or len(candidates)),
        "original_eval_mode": source.get("eval_mode"),
        "original_sims": int(source.get("sims") or 0),
    }


def convert(args: argparse.Namespace) -> dict[str, Any]:
    input_records = list(iter_jsonl(Path(args.input)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "input": str(args.input),
        "rust_output": str(args.rust_output),
        "output": str(output),
        "input_records": len(input_records),
        "written": 0,
        "skipped": 0,
        "candidates": 0,
    }
    with output.open("w", encoding="utf-8") as dst:
        for result in iter_jsonl(Path(args.rust_output)):
            idx = int(result.get("record_index", -1))
            if idx < 0 or idx >= len(input_records):
                stats["skipped"] += 1
                continue
            record = convert_record(result, input_records[idx])
            if not record["candidates"]:
                stats["skipped"] += 1
                continue
            dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["written"] += 1
            stats["candidates"] += len(record["candidates"])
    output.with_suffix(".summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return stats


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert Rust T3 exact output to teacher JSONL")
    parser.add_argument("--input", required=True, help="Original input JSONL passed to t3_exact_solver")
    parser.add_argument("--rust-output", required=True, help="JSONL produced by t3_exact_solver")
    parser.add_argument("--output", required=True, help="Teacher JSONL output path")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(convert(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
