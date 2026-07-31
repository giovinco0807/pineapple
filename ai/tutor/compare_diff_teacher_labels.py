"""Compare baseline/candidate changed actions against generated teacher labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def row_name(value: str) -> str:
    return "middle" if value in {"mid", "middle"} else ("bottom" if value in {"bot", "bottom"} else value)


def action_key(action: dict[str, Any]) -> tuple[tuple[tuple[str, str], ...], str | None]:
    placements = tuple(sorted((str(card), row_name(str(pos))) for card, pos in action.get("placements", [])))
    discard = action.get("discard")
    return placements, None if discard is None else str(discard)


def candidate_key(candidate: dict[str, Any]) -> tuple[tuple[tuple[str, str], ...], str | None]:
    placements = tuple(sorted((str(card), row_name(str(pos))) for card, pos in candidate.get("placements", [])))
    discard = candidate.get("discard")
    return placements, None if discard is None else str(discard)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def candidate_ev(candidate: dict[str, Any], eval_mode: str) -> float:
    if eval_mode.startswith("mc") or eval_mode == "t0_ladder":
        return float((candidate.get("mc") or {}).get("avg_score", 0.0))
    return float(candidate.get("ev", candidate.get("score", 0.0)))


def candidate_bust(candidate: dict[str, Any], eval_mode: str) -> float:
    if eval_mode.startswith("mc") or eval_mode == "t0_ladder":
        return float((candidate.get("mc") or {}).get("bust_rate", 0.0))
    return float(candidate.get("bust_prob", candidate.get("bust_rate", 0.0)))


def candidate_fl(candidate: dict[str, Any], eval_mode: str) -> float:
    if eval_mode.startswith("mc") or eval_mode == "t0_ladder":
        return float((candidate.get("mc") or {}).get("fl_rate", 0.0))
    return float(candidate.get("fl_rate", 0.0))


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    targets = {int(item["source_line"]): item for item in load_jsonl(Path(args.targets))}
    labels = load_jsonl(Path(args.labels))
    rows: list[dict[str, Any]] = []
    missing = 0
    for label in labels:
        source_line = int(label.get("source_line", -1))
        target = targets.get(source_line)
        if target is None:
            missing += 1
            continue
        eval_mode = str(label.get("eval_mode", "exact"))
        candidates = list(label.get("candidates") or [])
        by_action = {candidate_key(candidate): (rank, candidate) for rank, candidate in enumerate(candidates, start=1)}
        baseline_item = by_action.get(action_key(target["baseline_action"]))
        candidate_item = by_action.get(action_key(target["candidate_action"]))
        if baseline_item is None or candidate_item is None:
            rows.append(
                {
                    "source_line": source_line,
                    "hand": target.get("hand"),
                    "turn": target.get("turn"),
                    "error": "action_not_found",
                    "baseline_found": baseline_item is not None,
                    "candidate_found": candidate_item is not None,
                }
            )
            continue
        baseline_rank, baseline_cand = baseline_item
        candidate_rank, candidate_cand = candidate_item
        best_ev = candidate_ev(candidates[0], eval_mode)
        baseline_ev = candidate_ev(baseline_cand, eval_mode)
        candidate_score = candidate_ev(candidate_cand, eval_mode)
        rows.append(
            {
                "source_line": source_line,
                "hand": int(target.get("hand", 0)),
                "record": int(target.get("record", source_line)),
                "player": int(target.get("player", 0)),
                "turn": int(target.get("turn", -1)),
                "eval_mode": eval_mode,
                "n_candidates": int(label.get("n_candidates", len(candidates))),
                "best_ev": best_ev,
                "baseline_rank": baseline_rank,
                "baseline_ev": baseline_ev,
                "baseline_regret": best_ev - baseline_ev,
                "baseline_bust": candidate_bust(baseline_cand, eval_mode),
                "baseline_fl": candidate_fl(baseline_cand, eval_mode),
                "candidate_rank": candidate_rank,
                "candidate_ev": candidate_score,
                "candidate_regret": best_ev - candidate_score,
                "candidate_bust": candidate_bust(candidate_cand, eval_mode),
                "candidate_fl": candidate_fl(candidate_cand, eval_mode),
                "delta_candidate_minus_baseline": candidate_score - baseline_ev,
                "winner": (
                    "candidate"
                    if candidate_score - baseline_ev > args.tie_threshold
                    else ("baseline" if baseline_ev - candidate_score > args.tie_threshold else "tie")
                ),
                "baseline_action": target["baseline_action"],
                "candidate_action": target["candidate_action"],
            }
        )

    valid_rows = [row for row in rows if "error" not in row]
    candidate_wins = sum(1 for row in valid_rows if row["winner"] == "candidate")
    baseline_wins = sum(1 for row in valid_rows if row["winner"] == "baseline")
    ties = sum(1 for row in valid_rows if row["winner"] == "tie")
    delta_sum = sum(float(row["delta_candidate_minus_baseline"]) for row in valid_rows)
    report = {
        "targets": args.targets,
        "labels": args.labels,
        "tie_threshold": args.tie_threshold,
        "labels_read": len(labels),
        "targets_read": len(targets),
        "missing_targets": missing,
        "valid": len(valid_rows),
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "avg_delta_candidate_minus_baseline": delta_sum / max(len(valid_rows), 1),
        "sum_delta_candidate_minus_baseline": delta_sum,
        "rows": rows,
    }
    return report


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Teacher Check For Changed Actions",
        "",
        f"- targets: `{Path(report['targets']).name}`",
        f"- labels: `{Path(report['labels']).name}`",
        f"- valid changed states: {report['valid']}",
        f"- candidate wins: {report['candidate_wins']}",
        f"- baseline wins: {report['baseline_wins']}",
        f"- ties: {report['ties']}",
        f"- avg candidate-baseline EV delta: {report['avg_delta_candidate_minus_baseline']:+.3f}",
        f"- sum candidate-baseline EV delta: {report['sum_delta_candidate_minus_baseline']:+.3f}",
        "",
        "| hand | rec | turn | winner | delta | base rank | cand rank | base EV | cand EV | best EV | base FL | cand FL | base bust | cand bust |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        if "error" in row:
            lines.append(
                f"| {row.get('hand')} | {row['source_line']} | {row.get('turn')} | {row['error']} |  |  |  |  |  |  |  |  |  |  |"
            )
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["hand"]),
                    str(row["record"]),
                    f"T{row['turn']}",
                    row["winner"],
                    f"{row['delta_candidate_minus_baseline']:+.3f}",
                    str(row["baseline_rank"]),
                    str(row["candidate_rank"]),
                    f"{row['baseline_ev']:+.3f}",
                    f"{row['candidate_ev']:+.3f}",
                    f"{row['best_ev']:+.3f}",
                    f"{row['baseline_fl']:.1%}",
                    f"{row['candidate_fl']:.1%}",
                    f"{row['baseline_bust']:.1%}",
                    f"{row['candidate_bust']:.1%}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare changed actions against teacher labels")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--tie-threshold", type=float, default=0.10)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = summarize(args)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_md, report)
    print(json.dumps({k: report[k] for k in report if k != "rows"}, indent=2, ensure_ascii=False))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
