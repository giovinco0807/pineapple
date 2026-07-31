"""Run the Rust T2 exact evaluator and compare it with existing teacher labels.

The current T2 training files are mostly MC labels.  This harness creates a
small exact/capped-exact oracle slice from the same JSONL and writes a compact
comparison report so we can decide which misses should become training data.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from ai.engine.encoding import ALL_CARDS


ROW_NAMES = {"middle": "middle", "mid": "middle", "bottom": "bottom", "bot": "bottom", "top": "top"}


def row_name(value: Any) -> str:
    return ROW_NAMES.get(str(value), str(value))


def action_key(action: dict[str, Any] | None) -> str:
    action = action or {}
    placements = tuple(
        sorted((str(card), row_name(row)) for card, row in (action.get("placements") or []))
    )
    discard = action.get("discard")
    return json.dumps(
        {"placements": placements, "discard": None if discard is None else str(discard)},
        sort_keys=True,
        separators=(",", ":"),
    )


def candidate_key(candidate: dict[str, Any] | None) -> str:
    return action_key(candidate_action(candidate))


def candidate_score(candidate: dict[str, Any] | None) -> float:
    candidate = candidate or {}
    metrics = candidate.get("metrics") or {}
    if "score" in metrics:
        return float(metrics["score"])
    mc = candidate.get("mc") or {}
    if "avg_score" in mc:
        return float(mc["avg_score"])
    for key in ("target_score", "score", "ev", "t3_model_ev", "model_t3_value_score", "model_t3_priority_score"):
        if key in candidate and candidate[key] is not None:
            return float(candidate[key])
    return float("-inf")


def candidate_metric(candidate: dict[str, Any] | None, name: str) -> float:
    candidate = candidate or {}
    metrics = candidate.get("metrics") or {}
    mc = candidate.get("mc") or {}
    if name == "bust":
        return float(metrics.get("bust_rate", mc.get("bust_rate", candidate.get("model_t3_value_bust", 0.0))) or 0.0)
    if name == "fl":
        return float(metrics.get("fl_rate", mc.get("fl_rate", candidate.get("model_t3_value_fl", 0.0))) or 0.0)
    raise ValueError(name)


def candidate_action(candidate: dict[str, Any] | None) -> dict[str, Any]:
    candidate = candidate or {}
    action = candidate.get("action")
    if isinstance(action, dict):
        return {
            "placements": action.get("placements") or [],
            "discard": action.get("discard"),
        }
    return {
        "placements": candidate.get("placements") or [],
        "discard": candidate.get("discard"),
    }


def compact_candidate(
    candidate: dict[str, Any] | None,
    *,
    rank: int | None,
    label_source: str,
) -> dict[str, Any] | None:
    if candidate is None:
        return None
    action = candidate_action(candidate)
    metrics = candidate.get("metrics") or {}
    mc = candidate.get("mc") or {}
    return {
        "rank": rank,
        "label_source": label_source,
        "action_key": action_key(action),
        "action": action,
        "score": candidate_score(candidate),
        "fl_rate": candidate_metric(candidate, "fl"),
        "bust_rate": candidate_metric(candidate, "bust"),
        "samples": metrics.get("samples", mc.get("simulations")),
        "metric_source": metrics.get("source", candidate.get("eval_mode", "")),
        "forced_bust": metrics.get("forced_bust"),
        "fl_type_rates": metrics.get("fl_type_rates", mc.get("fl_type_rates", {})),
    }


def board_cards(board: dict[str, Any] | None) -> list[str]:
    board = board or {}
    return [
        str(card)
        for key in ("top", "mid", "middle", "bot", "bottom")
        for card in (board.get(key) or [])
        if str(card)
    ]


def full_t2_draw_count(source: dict[str, Any], action: dict[str, Any] | None) -> int:
    action = action or {}
    used = set(board_cards(source.get("board") or {}))
    used.update(board_cards(source.get("opponent_board") or source.get("board_opponent") or {}))
    used.update(str(card) for card in (source.get("known_discards") or []) if str(card))
    used.update(str(card) for card in (source.get("exclude") or []) if str(card))
    for card, _row in action.get("placements") or []:
        used.add(str(card))
    discard = action.get("discard")
    if discard:
        used.add(str(discard))
    remaining = len([card for card in ALL_CARDS if card not in used])
    return math.comb(remaining, 3) if remaining >= 3 else 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def source_best(record: dict[str, Any]) -> dict[str, Any] | None:
    candidates = list(record.get("candidates") or [])
    if not candidates:
        return None
    best_idx = record.get("best_idx")
    if best_idx is not None:
        try:
            idx = int(best_idx)
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except (TypeError, ValueError):
            pass
    return max(candidates, key=candidate_score)


def ranked_source(record: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = list(record.get("candidates") or [])
    return sorted(candidates, key=candidate_score, reverse=True)


def find_candidate_by_key(candidates: Iterable[dict[str, Any]], key: str) -> tuple[int | None, dict[str, Any] | None]:
    for rank, candidate in enumerate(candidates, start=1):
        if candidate_key(candidate) == key:
            return rank, candidate
    return None, None


def source_context(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source.get("source"),
        "source_line": source.get("source_line"),
        "active_reasons": source.get("active_reasons"),
        "position": source.get("position"),
        "is_btn": source.get("is_btn"),
        "board": source.get("board"),
        "opponent_board": source.get("opponent_board") or source.get("board_opponent"),
        "dealt": source.get("dealt"),
        "known_discards": source.get("known_discards"),
        "exclude": source.get("exclude"),
    }


def run_solver(args: argparse.Namespace, exact_path: Path) -> None:
    if args.binary:
        cmd = [str(Path(args.binary))]
    else:
        cmd = [
            "cargo",
            "run",
            "--release",
            "--manifest-path",
            str(Path(args.manifest)),
            "--",
        ]
    cmd.extend(
        [
            "--input",
            str(Path(args.input)),
            "--output",
            str(exact_path),
            "--limit",
            str(args.limit),
            "--skip",
            str(args.skip),
            "--top-n",
            str(args.top_n),
            "--t2-draw-limit",
            str(args.t2_draw_limit),
            "--fl-config",
            str(Path(args.fl_config)),
        ]
    )
    if int(args.source_candidate_top_k) > 0:
        cmd.extend(["--source-candidate-top-k", str(args.source_candidate_top_k)])
    subprocess.run(cmd, check=True)


def compare(input_path: Path, exact_path: Path, miss_top_k: int = 5) -> dict[str, Any]:
    source_records = load_jsonl(input_path)
    exact_records = load_jsonl(exact_path)
    rows: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    for exact in exact_records:
        record_index = int(exact.get("record_index", -1))
        if record_index < 0 or record_index >= len(source_records):
            rows.append({"record_index": record_index, "error": "source_record_missing"})
            continue
        source = source_records[record_index]
        source_ranked = ranked_source(source)
        source_best_candidate = source_best(source)
        exact_best = exact.get("best") or {}
        exact_candidates = list(exact.get("candidates") or [])
        source_best_key = candidate_key(source_best_candidate)
        exact_best_key = action_key((exact_best.get("action") or {}))
        exact_by_action = {
            action_key(candidate.get("action") or {}): (rank, candidate)
            for rank, candidate in enumerate(exact_candidates, start=1)
        }
        source_best_exact_rank, source_best_exact = exact_by_action.get(source_best_key, (None, None))
        source_by_action = {
            candidate_key(candidate): (rank, candidate)
            for rank, candidate in enumerate(source_ranked, start=1)
        }
        exact_best_source_rank, exact_best_source = source_by_action.get(exact_best_key, (None, None))
        exact_best_score = candidate_score(exact_best)
        source_best_exact_score = candidate_score(source_best_exact)
        source_best_score = candidate_score(source_best_candidate)
        exact_best_source_score = candidate_score(exact_best_source)
        exact_samples = int(((exact_best.get("metrics") or {}).get("samples") or 0))
        full_draws = full_t2_draw_count(source, exact_best.get("action") or {})
        elapsed_ms = float(exact.get("elapsed_ms", 0.0) or 0.0)
        estimated_full_ms = (
            elapsed_ms * full_draws / exact_samples
            if exact_samples > 0 and full_draws > exact_samples
            else elapsed_ms
        )
        rows.append(
            {
                "record_index": record_index,
                "turn": int(exact.get("turn", source.get("turn", -1))),
                "legal_actions": int(exact.get("legal_actions", len(exact_candidates))),
                "exact_elapsed_ms": elapsed_ms,
                "exact_samples": exact_samples,
                "estimated_full_t2_draws": full_draws,
                "estimated_full_t2_elapsed_ms_from_cap": estimated_full_ms,
                "exact_source": str((exact_best.get("metrics") or {}).get("source", "")),
                "source_eval_mode": str(source.get("eval_mode", "")),
                "same_top1": source_best_key == exact_best_key,
                "source_best_exact_rank": source_best_exact_rank,
                "exact_best_source_rank": exact_best_source_rank,
                "source_best_score": source_best_score,
                "exact_best_score": exact_best_score,
                "source_best_exact_score": source_best_exact_score,
                "exact_best_source_score": exact_best_source_score,
                "exact_regret_of_source_top1": max(0.0, exact_best_score - source_best_exact_score)
                if source_best_exact is not None
                else None,
                "source_regret_of_exact_top1": max(0.0, source_best_score - exact_best_source_score)
                if exact_best_source is not None
                else None,
                "source_best_fl": candidate_metric(source_best_candidate, "fl"),
                "exact_best_fl": candidate_metric(exact_best, "fl"),
                "source_best_bust": candidate_metric(source_best_candidate, "bust"),
                "exact_best_bust": candidate_metric(exact_best, "bust"),
                "source_best_action": source_best_candidate,
                "exact_best_action": exact_best.get("action"),
            }
        )
        if source_best_key != exact_best_key:
            misses.append(
                {
                    "record_index": record_index,
                    "turn": int(exact.get("turn", source.get("turn", -1))),
                    "reason": "source_top1_differs_from_oracle_top1",
                    "input": str(input_path),
                    "oracle_output": str(exact_path),
                    **source_context(source),
                    "legal_actions": int(exact.get("legal_actions", len(exact_candidates))),
                    "source_eval_mode": str(source.get("eval_mode", "")),
                    "oracle_source": str((exact_best.get("metrics") or {}).get("source", "")),
                    "oracle_samples": exact_samples,
                    "estimated_full_t2_draws": full_draws,
                    "exact_elapsed_ms": elapsed_ms,
                    "estimated_full_t2_elapsed_ms_from_cap": estimated_full_ms,
                    "source_best_exact_rank": source_best_exact_rank,
                    "oracle_best_source_rank": exact_best_source_rank,
                    "exact_regret_of_source_top1": max(0.0, exact_best_score - source_best_exact_score)
                    if source_best_exact is not None
                    else None,
                    "source_regret_of_oracle_top1": max(0.0, source_best_score - exact_best_source_score)
                    if exact_best_source is not None
                    else None,
                    "source_best": compact_candidate(source_best_candidate, rank=1, label_source="source"),
                    "oracle_best": compact_candidate(exact_best, rank=1, label_source="oracle"),
                    "source_best_under_oracle": compact_candidate(
                        source_best_exact,
                        rank=source_best_exact_rank,
                        label_source="oracle",
                    ),
                    "oracle_best_under_source": compact_candidate(
                        exact_best_source,
                        rank=exact_best_source_rank,
                        label_source="source",
                    ),
                    "source_top_candidates": [
                        compact_candidate(candidate, rank=rank, label_source="source")
                        for rank, candidate in enumerate(source_ranked[:miss_top_k], start=1)
                    ],
                    "oracle_top_candidates": [
                        compact_candidate(candidate, rank=rank, label_source="oracle")
                        for rank, candidate in enumerate(exact_candidates[:miss_top_k], start=1)
                    ],
                }
            )
    valid = [row for row in rows if "error" not in row]
    same = sum(1 for row in valid if row["same_top1"])
    changed = len(valid) - same
    avg_elapsed = sum(float(row["exact_elapsed_ms"]) for row in valid) / max(len(valid), 1)
    avg_est_full = sum(float(row["estimated_full_t2_elapsed_ms_from_cap"]) for row in valid) / max(len(valid), 1)
    avg_regret = sum(float(row["exact_regret_of_source_top1"] or 0.0) for row in valid) / max(len(valid), 1)
    max_regret = max([float(row["exact_regret_of_source_top1"] or 0.0) for row in valid] or [0.0])
    return {
        "input": str(input_path),
        "exact_output": str(exact_path),
        "records": len(valid),
        "same_top1": same,
        "changed_top1": changed,
        "same_top1_rate": same / max(len(valid), 1),
        "avg_exact_elapsed_ms": avg_elapsed,
        "avg_estimated_full_t2_elapsed_ms_from_cap": avg_est_full,
        "avg_exact_regret_of_source_top1": avg_regret,
        "max_exact_regret_of_source_top1": max_regret,
        "miss_count": len(misses),
        "misses": misses,
        "rows": rows,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# T2 Exact Oracle Check",
        "",
        f"- input: `{Path(report['input']).name}`",
        f"- exact output: `{Path(report['exact_output']).name}`",
        f"- records: {report['records']}",
        f"- same Top1: {report['same_top1']} ({report['same_top1_rate']:.1%})",
        f"- changed Top1: {report['changed_top1']}",
        f"- misses written: {report.get('miss_count', 0)}",
        f"- avg exact elapsed: {report['avg_exact_elapsed_ms']:.1f} ms",
        f"- avg estimated full T2 elapsed from cap: {report['avg_estimated_full_t2_elapsed_ms_from_cap']:.1f} ms",
        f"- avg exact regret of source Top1: {report['avg_exact_regret_of_source_top1']:+.4f}",
        f"- max exact regret of source Top1: {report['max_exact_regret_of_source_top1']:+.4f}",
        "",
        "| record | same | source mode | exact source | exact samples | full draws est | exact ms | full ms est | source exact rank | exact source rank | exact regret source top1 | source FL | exact FL | source bust | exact bust |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        if "error" in row:
            lines.append(f"| {row.get('record_index')} | {row['error']} |  |  |  |  |  |  |  |  |  |  |  |  |  |")
            continue
        regret = row["exact_regret_of_source_top1"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["record_index"]),
                    "yes" if row["same_top1"] else "no",
                    row["source_eval_mode"],
                    row["exact_source"],
                    str(row["exact_samples"]),
                    str(row["estimated_full_t2_draws"]),
                    f"{row['exact_elapsed_ms']:.1f}",
                    f"{row['estimated_full_t2_elapsed_ms_from_cap']:.1f}",
                    "" if row["source_best_exact_rank"] is None else str(row["source_best_exact_rank"]),
                    "" if row["exact_best_source_rank"] is None else str(row["exact_best_source_rank"]),
                    "" if regret is None else f"{regret:+.4f}",
                    f"{row['source_best_fl']:.1%}",
                    f"{row['exact_best_fl']:.1%}",
                    f"{row['source_best_bust']:.1%}",
                    f"{row['exact_best_bust']:.1%}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run and compare Rust T2 exact oracle labels")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--t2-draw-limit", type=int, default=1)
    parser.add_argument("--manifest", default="ai/rust_solver/t3_exact_solver/Cargo.toml")
    parser.add_argument("--binary", default="")
    parser.add_argument("--fl-config", default="ai/config/fl_ev.json")
    parser.add_argument("--miss-top-k", type=int, default=5)
    parser.add_argument(
        "--source-candidate-top-k",
        type=int,
        default=0,
        help="If input rows include ranked candidates, exact only the first K candidates.",
    )
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "exact" if args.t2_draw_limit == 0 else f"cap{args.t2_draw_limit}"
    skip_part = f"_skip{args.skip}" if args.skip else ""
    exact_path = out_dir / f"t2_oracle_{suffix}{skip_part}_limit{args.limit}.jsonl"
    if not args.skip_run:
        run_solver(args, exact_path)
    report = compare(Path(args.input), exact_path, miss_top_k=args.miss_top_k)
    report_path = out_dir / f"t2_oracle_{suffix}{skip_part}_limit{args.limit}.summary.json"
    md_path = out_dir / f"t2_oracle_{suffix}{skip_part}_limit{args.limit}.summary.md"
    misses_path = out_dir / f"t2_oracle_{suffix}{skip_part}_limit{args.limit}.misses.jsonl"
    with misses_path.open("w", encoding="utf-8") as f:
        for miss in report["misses"]:
            f.write(json.dumps(miss, ensure_ascii=False, separators=(",", ":")) + "\n")
    report["misses_path"] = str(misses_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(md_path, report)
    print(
        json.dumps(
            {k: report[k] for k in report if k not in {"rows", "misses"}}
            | {"summary": str(report_path), "markdown": str(md_path), "misses_path": str(misses_path)},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
