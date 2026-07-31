"""Replay a T1 override gate on saved hybrid-refinement result rows.

This is cheaper than re-running recursive refinement.  It assumes each result
row already contains the model Top1 action, the recursive-refinement candidate,
teacher scores, and ``t1_override_features`` from a no-gate or loose-gate run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.hybrid_t1t2 import _gate_accepts_override, load_t1_override_gate


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def threshold_grid(step: float) -> list[float]:
    step = max(float(step), 0.001)
    count = int(round(1.0 / step))
    return sorted({round(i * step, 10) for i in range(0, count + 1)})


def row_accepts(row: dict[str, Any], gate: dict[str, Any] | None, threshold: float, override_margin: float) -> tuple[bool, float | None, str | None]:
    features = row.get("t1_override_features")
    if not isinstance(features, dict):
        return False, None, "no_features"
    if int(row.get("best_action_idx", -1)) == int(row.get("model_top1_action_idx", -2)):
        return False, None, "same_action"
    if float(features.get("refined_delta", 0.0)) < float(override_margin):
        return False, None, "margin"
    if gate is None:
        return True, None, None
    accepted, probability = _gate_accepts_override(
        gate,
        {name: float(value) for name, value in features.items()},
        threshold=threshold,
    )
    if not accepted:
        return False, probability, "gate"
    return True, probability, None


def summarize(rows: list[dict[str, Any]], gate: dict[str, Any] | None, threshold: float, override_margin: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "threshold": threshold,
        "override_margin": override_margin,
        "rows": 0,
        "rows_with_features": 0,
        "accepted": 0,
        "rejected": 0,
        "reject_reasons": {},
        "model_top1": 0,
        "always_accept_top1": 0,
        "gated_top1": 0,
        "accepted_final_top1": 0,
        "rejected_model_top1": 0,
        "accepted_bad_top1": 0,
        "rejected_missed_top1": 0,
        "model_regret": 0.0,
        "always_accept_regret": 0.0,
        "gated_regret": 0.0,
        "accepted_gain": 0.0,
        "bad_override_regret": 0.0,
    }
    reject_reasons: dict[str, int] = {}
    for row in rows:
        if int(row.get("turn", -1)) != 1:
            continue
        out["rows"] += 1
        if isinstance(row.get("t1_override_features"), dict):
            out["rows_with_features"] += 1
        accepted, _probability, reject_reason = row_accepts(row, gate, threshold, override_margin)
        if accepted:
            out["accepted"] += 1
        else:
            out["rejected"] += 1
            reject_reasons[str(reject_reason or "rejected")] = reject_reasons.get(str(reject_reason or "rejected"), 0) + 1

        model_hit = bool(row.get("model_top1_hit"))
        final_hit = bool(row.get("final_top1_hit"))
        selected_hit = final_hit if accepted else model_hit
        if model_hit:
            out["model_top1"] += 1
        if final_hit:
            out["always_accept_top1"] += 1
        if selected_hit:
            out["gated_top1"] += 1
        if accepted and final_hit:
            out["accepted_final_top1"] += 1
        if not accepted and model_hit:
            out["rejected_model_top1"] += 1
        if accepted and model_hit and not final_hit:
            out["accepted_bad_top1"] += 1
        if not accepted and final_hit and not model_hit:
            out["rejected_missed_top1"] += 1

        best_score = float(row.get("teacher_best_score", 0.0))
        model_score = float(row.get("model_teacher_score", row.get("model_score", 0.0)) or 0.0)
        final_score = float(row.get("final_teacher_score", row.get("final_score", 0.0)) or 0.0)
        selected_score = final_score if accepted else model_score
        out["model_regret"] += max(0.0, best_score - model_score)
        out["always_accept_regret"] += max(0.0, best_score - final_score)
        out["gated_regret"] += max(0.0, best_score - selected_score)
        if accepted:
            gain = final_score - model_score
            out["accepted_gain"] += gain
            if gain < 0.0:
                out["bad_override_regret"] += -gain

    out["reject_reasons"] = reject_reasons
    denom = max(int(out["rows"]), 1)
    out["model_top1_recall"] = out["model_top1"] / denom
    out["always_accept_top1_recall"] = out["always_accept_top1"] / denom
    out["gated_top1_recall"] = out["gated_top1"] / denom
    out["top1_delta_vs_model"] = (out["gated_top1"] - out["model_top1"]) / denom
    out["top1_delta_vs_always_accept"] = (out["gated_top1"] - out["always_accept_top1"]) / denom
    out["model_avg_regret"] = out["model_regret"] / denom
    out["always_accept_avg_regret"] = out["always_accept_regret"] / denom
    out["gated_avg_regret"] = out["gated_regret"] / denom
    return out


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay a T1 override gate on saved hybrid-refinement results")
    parser.add_argument("--results", required=True)
    parser.add_argument("--gate", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--override-margin", type=float, default=0.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = list(iter_jsonl(Path(args.results)))
    gate = load_t1_override_gate(args.gate) if args.gate else None
    if args.sweep:
        threshold_rows = [
            summarize(rows, gate, threshold, args.override_margin)
            for threshold in threshold_grid(args.threshold_step)
        ]
        best = max(
            threshold_rows,
            key=lambda row: (
                row["gated_top1"],
                -row["gated_regret"],
                -row["accepted_bad_top1"],
                -row["rejected_missed_top1"],
            ),
        )
        report: dict[str, Any] = {
            "results": args.results,
            "gate": args.gate,
            "override_margin": args.override_margin,
            "threshold_step": args.threshold_step,
            "best": best,
            "threshold_eval": threshold_rows,
        }
    else:
        report = summarize(rows, gate, args.threshold, args.override_margin)
        report["results"] = args.results
        report["gate"] = args.gate

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
