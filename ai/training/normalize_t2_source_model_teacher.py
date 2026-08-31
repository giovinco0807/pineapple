"""Normalize T2 T3-model source rows into action-value teacher shape.

Rows produced by ``build_t2_t3_model_teacher_from_t0t1_topk`` store candidate
placements under ``candidate["action"]`` and use T3-model estimates as scores.
That is useful for cheap preselection, but ``convert_action_value_teacher.py``
expects flat candidate fields.  This converter emits a pseudo-teacher JSONL
using the source model estimates, so selector/pool models can score newly
generated source rows before we spend local exact time on them.

The output is not an exact label.  Use it only for preselecting rows to exact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


CONTEXT_KEYS = (
    "source",
    "source_line",
    "turn",
    "board",
    "opponent_board",
    "dealt",
    "known_discards",
    "exclude",
    "is_btn",
    "position",
    "future_hidden_deals",
    "branch",
    "root_deals",
    "trace",
    "generator_config",
)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def estimate_score(candidate: dict) -> float:
    for key in ("model_t3_value_score", "t3_model_ev", "model_t3_priority_score", "score"):
        value = candidate.get(key)
        if value is not None:
            return float(value)
    return 0.0


def normalize_candidate(candidate: dict, rank: int) -> dict:
    action = candidate.get("action") or {}
    score = estimate_score(candidate)
    bust = float(candidate.get("model_t3_value_bust") or candidate.get("bust_rate") or 0.0)
    fl = float(candidate.get("model_t3_value_fl") or candidate.get("fl_rate") or 0.0)
    return {
        "placements": action.get("placements") or candidate.get("placements") or [],
        "discard": action.get("discard", candidate.get("discard")),
        "ev": score,
        "score": score,
        "raw_score": score,
        "royalty": 0.0,
        "bust_prob": bust,
        "bust_rate": bust,
        "fl_rate": fl,
        "fl_type_rates": dict(candidate.get("model_t3_value_fl_types") or candidate.get("fl_type_rates") or {}),
        "samples": int(candidate.get("t3_states_scored") or candidate.get("samples") or 0),
        "metric_source": "t3_model_source",
        "forced_bust": bool(candidate.get("forced_bust", False)),
        "teacher_rank": int(rank),
        "source_index": candidate.get("source_index"),
        "source_rank": candidate.get("rank"),
        "source_action_key": candidate.get("action_key"),
    }


def convert(args: argparse.Namespace) -> dict:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    candidate_count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for source_idx, row in enumerate(iter_jsonl(input_path)):
            if args.limit and written >= args.limit:
                break
            if int(row.get("turn", -1)) != 2:
                skipped += 1
                continue
            candidates = list(row.get("candidates") or [])
            if not candidates:
                skipped += 1
                continue
            candidates.sort(key=estimate_score, reverse=True)
            norm_candidates = [normalize_candidate(candidate, rank) for rank, candidate in enumerate(candidates, start=1)]
            out_row = {key: row.get(key) for key in CONTEXT_KEYS if key in row}
            out_row.update(
                {
                    "source_global_index": source_idx,
                    "eval_mode": "t3_model_source",
                    "teacher_eval_mode": "t3_model_source",
                    "best_idx": 0,
                    "n_candidates": len(norm_candidates),
                    "estimated": True,
                    "exact": False,
                    "candidates": norm_candidates,
                }
            )
            out.write(json.dumps(out_row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            candidate_count += len(norm_candidates)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "written": written,
        "skipped": skipped,
        "candidate_count": candidate_count,
        "eval_mode": "t3_model_source",
        "note": "Estimated source labels only. Use for preselection, not final training labels.",
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(convert(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
