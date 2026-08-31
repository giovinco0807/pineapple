"""Compare shortlist T2 exact-rerank output against all-legal exact output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def action_key(action: dict[str, Any] | None) -> str:
    action = action or {}
    placements = tuple(sorted((str(card), str(row)) for card, row in (action.get("placements") or [])))
    discard = action.get("discard")
    return json.dumps(
        {"placements": placements, "discard": None if discard is None else str(discard)},
        sort_keys=True,
        separators=(",", ":"),
    )


def score(candidate: dict[str, Any] | None) -> float:
    metrics = (candidate or {}).get("metrics") or {}
    return float(metrics.get("score", candidate.get("score", float("-inf")) if candidate else float("-inf")))


def load_by_index(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    for fallback_idx, row in enumerate(iter_jsonl(path)):
        idx = int(row.get("global_index", row.get("record_index", fallback_idx)))
        rows[idx] = row
    return rows


def compare(args: argparse.Namespace) -> dict[str, Any]:
    alllegal = load_by_index(Path(args.alllegal_exact))
    shortlist = load_by_index(Path(args.shortlist_exact))
    out_rows: list[dict[str, Any]] = []
    losses: list[float] = []
    hits = 0
    total_elapsed = 0.0
    total_evaluated = 0
    for idx in sorted(shortlist):
        short = shortlist[idx]
        full = alllegal.get(idx)
        if full is None:
            out_rows.append({"record_index": idx, "error": "missing_alllegal_row"})
            continue
        full_best = full.get("best") or {}
        short_best = short.get("best") or {}
        full_best_key = action_key(full_best.get("action") or {})
        short_best_key = action_key(short_best.get("action") or {})
        full_score = score(full_best)
        short_score = score(short_best)
        loss = max(0.0, full_score - short_score)
        losses.append(loss)
        if full_best_key == short_best_key:
            hits += 1
        elapsed = float(short.get("elapsed_ms", 0.0) or 0.0)
        evaluated = int(short.get("evaluated_actions", 0) or 0)
        total_elapsed += elapsed
        total_evaluated += evaluated
        out_rows.append(
            {
                "record_index": idx,
                "hit": full_best_key == short_best_key,
                "ev_loss": loss,
                "alllegal_best_score": full_score,
                "shortlist_best_score": short_score,
                "alllegal_best_action": full_best.get("action"),
                "shortlist_best_action": short_best.get("action"),
                "shortlist_elapsed_ms": elapsed,
                "shortlist_evaluated_actions": evaluated,
                "alllegal_evaluated_actions": int(full.get("evaluated_actions", 0) or 0),
            }
        )
    n = max(len(losses), 1)
    summary = {
        "alllegal_exact": str(Path(args.alllegal_exact)),
        "shortlist_exact": str(Path(args.shortlist_exact)),
        "rows": len(losses),
        "hit_count": hits,
        "hit_rate": hits / n,
        "ev_loss_mean": sum(losses) / n,
        "ev_loss_max": max(losses) if losses else 0.0,
        "ev_loss_gt_0p05": sum(1 for loss in losses if loss > 0.05),
        "ev_loss_gt_0p1": sum(1 for loss in losses if loss > 0.1),
        "ev_loss_gt_0p25": sum(1 for loss in losses if loss > 0.25),
        "avg_elapsed_ms": total_elapsed / n,
        "total_elapsed_ms": total_elapsed,
        "avg_evaluated_actions": total_evaluated / n,
        "rows_path": str(Path(args.output_rows)) if args.output_rows else None,
    }
    if args.output_rows:
        out_path = Path(args.output_rows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as out:
            for row in out_rows:
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alllegal-exact", required=True)
    parser.add_argument("--shortlist-exact", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-rows")
    args = parser.parse_args(argv)
    print(json.dumps(compare(args), indent=2))


if __name__ == "__main__":
    main()
