"""Evaluate an action-value reranker on recursive tutor teacher data.

The output is meant to drive data generation: it reports top-N recall, model
regret, score/FL/bust calibration error, and a JSONL of the weakest decisions.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import encode_state
from ai.models.action_value_reranker import ActionValueReranker
from ai.training.action_feature_encoding import adapt_np_state_with_action
from ai.training.convert_recursive_teacher_to_reranker import (
    candidate_metrics,
    post_action_observation,
    top_candidate_to_regular,
    trace_candidate_to_regular,
)


TOP_NS = (1, 3, 5, 10, 15, 20, 24, 64)
FL_KEYS = ("qq", "kk", "aa", "trips")


@dataclass
class MetricBucket:
    decisions: int = 0
    candidates: int = 0
    skipped_candidates: int = 0
    top_hits: Dict[int, int] = field(default_factory=lambda: {k: 0 for k in TOP_NS})
    topk_regret_sum: Dict[int, float] = field(default_factory=lambda: {k: 0.0 for k in TOP_NS})
    topk_regret_max: Dict[int, float] = field(default_factory=lambda: {k: 0.0 for k in TOP_NS})
    regret_sum: float = 0.0
    regret_max: float = 0.0
    score_abs_sum: float = 0.0
    fl_abs_sum: float = 0.0
    bust_abs_sum: float = 0.0
    fl_type_abs_sum: Dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in FL_KEYS})

    def add_candidate_errors(
        self,
        n: int,
        score_abs: float,
        fl_abs: float,
        bust_abs: float,
        fl_type_abs: Dict[str, float],
    ) -> None:
        self.candidates += n
        self.score_abs_sum += score_abs
        self.fl_abs_sum += fl_abs
        self.bust_abs_sum += bust_abs
        for key in FL_KEYS:
            self.fl_type_abs_sum[key] += fl_type_abs.get(key, 0.0)

    def add_decision(self, rank_1_based: int, regret: float, topk_regrets: Optional[Dict[int, float]] = None) -> None:
        self.decisions += 1
        topk_regrets = topk_regrets or {}
        for k in TOP_NS:
            if rank_1_based <= k:
                self.top_hits[k] += 1
            topk_regret = float(topk_regrets.get(k, regret))
            self.topk_regret_sum[k] += topk_regret
            self.topk_regret_max[k] = max(self.topk_regret_max[k], topk_regret)
        self.regret_sum += regret
        self.regret_max = max(self.regret_max, regret)

    def as_dict(self) -> Dict[str, Any]:
        denom_decisions = max(self.decisions, 1)
        denom_candidates = max(self.candidates, 1)
        return {
            "decisions": self.decisions,
            "candidates": self.candidates,
            "skipped_candidates": self.skipped_candidates,
            "top_recall": {f"top_{k}": self.top_hits[k] / denom_decisions for k in TOP_NS},
            "topk_exact_rerank_avg_regret": {
                f"top_{k}": self.topk_regret_sum[k] / denom_decisions for k in TOP_NS
            },
            "topk_exact_rerank_max_regret": {
                f"top_{k}": self.topk_regret_max[k] for k in TOP_NS
            },
            "avg_regret": self.regret_sum / denom_decisions,
            "max_regret": self.regret_max,
            "score_mae": self.score_abs_sum / denom_candidates,
            "fl_rate_mae": self.fl_abs_sum / denom_candidates,
            "bust_rate_mae": self.bust_abs_sum / denom_candidates,
            "fl_type_mae": {
                key: self.fl_type_abs_sum[key] / denom_candidates
                for key in FL_KEYS
            },
        }


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def resolve_model_path(model: str) -> Path:
    if model != "auto":
        path = Path(model)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    root = Path("ai/models/candidate_runs")
    candidates = sorted(
        root.rglob("action_value_best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No action_value_best.pt found under ai/models/candidate_runs")
    return candidates[0]


def iter_decisions(record: Dict[str, Any], include_turns: set[int]) -> Iterator[Dict[str, Any]]:
    if "turn" in record and record.get("candidates"):
        turn = int(record.get("turn", -1))
        if turn in include_turns:
            position = str(record.get("position") or record.get("seat") or ("btn" if record.get("is_btn") else "bb"))
            yield {
                "turn": turn,
                "position": position,
                "is_btn": bool(record.get("is_btn", position == "btn")),
                "board": record.get("board") or record.get("board_before") or {},
                "dealt": list(record.get("dealt", []) or []),
                "known_discards": list(record.get("known_discards", []) or record.get("exclude", []) or []),
                "opponent_board": record.get("opponent_board") or record.get("board_opponent") or {},
                "candidates": [trace_candidate_to_regular(c) for c in (record.get("candidates") or [])],
                "source": "flat",
            }
        return

    position = str(record.get("position") or record.get("seat") or "bb")
    t0_dealt = list(record.get("dealt", []) or [])
    if 0 in include_turns:
        candidates = record.get("candidates") or []
        if candidates:
            yield {
                "turn": 0,
                "position": position,
                "is_btn": position == "btn",
                "board": {},
                "dealt": t0_dealt,
                "known_discards": list(record.get("known_discards", []) or []),
                "opponent_board": record.get("opponent_board") or {},
                "candidates": [top_candidate_to_regular(c, t0_dealt) for c in candidates],
                "source": "t0",
            }

    for trace_index, trace in enumerate(record.get("turn_traces") or []):
        turn = int(trace.get("turn", -1))
        if turn not in include_turns:
            continue
        candidates = trace.get("candidates") or []
        if not candidates:
            continue
        trace_position = str(trace.get("position") or position)
        yield {
            "turn": turn,
            "position": trace_position,
            "is_btn": trace_position == "btn",
            "board": trace.get("board_before") or {},
            "dealt": list(trace.get("dealt", []) or []),
            "known_discards": list(trace.get("dead_before", []) or []),
            "opponent_board": trace.get("opponent_board") or record.get("opponent_board") or {},
            "t0_dealt": t0_dealt,
            "candidates": [trace_candidate_to_regular(c) for c in candidates],
            "source": "trace",
            "trace_index": trace_index,
        }


def true_metric(candidate: Dict[str, Any]) -> Dict[str, Any]:
    score, bust, fl, fl_types = candidate_metrics(candidate)
    return {
        "score": float(score),
        "bust_rate": float(bust),
        "fl_rate": float(fl),
        "fl_type_rates": {key: float(fl_types[i]) for i, key in enumerate(FL_KEYS)},
    }


def metric_gap(best: Dict[str, Any], second: Optional[Dict[str, Any]]) -> float:
    if second is None:
        return 0.0
    return float(best["score"] - second["score"])


def score_candidates(
    model: ActionValueReranker,
    decision: Dict[str, Any],
    device: torch.device,
) -> tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    states = []
    skipped = 0

    for idx, candidate in enumerate(decision.get("candidates") or []):
        try:
            obs, normalized = post_action_observation(decision, candidate)
            states.append(
                adapt_np_state_with_action(
                    np.asarray(encode_state(obs), dtype=np.float32),
                    int(model.input_dim),
                    decision,
                    normalized,
                )
            )
            rows.append(
                {
                    "candidate_index": idx,
                    "candidate": normalized,
                    "true": true_metric(candidate),
                }
            )
        except Exception:
            skipped += 1

    if not states:
        return [], skipped

    with torch.no_grad():
        tensor = torch.from_numpy(np.stack(states)).to(device)
        turn_tensor = torch.full(
            (tensor.shape[0],),
            int(decision.get("turn", 0)),
            dtype=torch.long,
            device=device,
        )
        try:
            out = model.predict_components(tensor, turn=turn_tensor)
        except TypeError:
            out = model.predict_components(tensor)
        pred_score = out["score"].detach().cpu().numpy()
        pred_bust = out["bust_prob"].detach().cpu().numpy()
        pred_fl = out["fl_prob"].detach().cpu().numpy()
        pred_fl_types = out["fl_type_probs"].detach().cpu().numpy()

    for i, row in enumerate(rows):
        row["pred"] = {
            "score": float(pred_score[i]),
            "bust_rate": float(pred_bust[i]),
            "fl_rate": float(pred_fl[i]),
            "fl_type_rates": {
                key: float(pred_fl_types[i, j])
                for j, key in enumerate(FL_KEYS)
            },
        }
    return rows, skipped


def action_label(candidate: Dict[str, Any]) -> str:
    if candidate.get("action"):
        return str(candidate["action"])
    placements = candidate.get("placements") or []
    discard = candidate.get("discard")
    placed = "; ".join(f"{card}->{row}" for card, row in placements)
    if discard:
        return f"{placed}; discard {discard}"
    return placed


def summarize_decision(
    record_index: int,
    decision_index: int,
    decision: Dict[str, Any],
    scored: List[Dict[str, Any]],
    skipped: int,
) -> Optional[Dict[str, Any]]:
    if not scored:
        return None

    true_order = sorted(scored, key=lambda r: r["true"]["score"], reverse=True)
    pred_order = sorted(scored, key=lambda r: r["pred"]["score"], reverse=True)
    best_true = true_order[0]
    second_true = true_order[1] if len(true_order) > 1 else None
    pred_pick = pred_order[0]
    true_best_idx = best_true["candidate_index"]
    pred_rank = next(i for i, row in enumerate(pred_order, start=1) if row["candidate_index"] == true_best_idx)
    regret = best_true["true"]["score"] - pred_pick["true"]["score"]
    topk_regrets = {}
    for k in TOP_NS:
        kept = pred_order[: min(k, len(pred_order))]
        best_kept = max(row["true"]["score"] for row in kept) if kept else pred_pick["true"]["score"]
        topk_regrets[str(k)] = float(best_true["true"]["score"] - best_kept)

    score_abs = 0.0
    fl_abs = 0.0
    bust_abs = 0.0
    fl_type_abs = {key: 0.0 for key in FL_KEYS}
    for row in scored:
        score_abs += abs(row["pred"]["score"] - row["true"]["score"])
        fl_abs += abs(row["pred"]["fl_rate"] - row["true"]["fl_rate"])
        bust_abs += abs(row["pred"]["bust_rate"] - row["true"]["bust_rate"])
        for key in FL_KEYS:
            fl_type_abs[key] += abs(
                row["pred"]["fl_type_rates"][key] - row["true"]["fl_type_rates"][key]
            )

    return {
        "record_index": record_index,
        "decision_index": decision_index,
        "turn": int(decision["turn"]),
        "position": decision.get("position"),
        "source": decision.get("source"),
        "board": decision.get("board"),
        "dealt": decision.get("dealt"),
        "opponent_board": decision.get("opponent_board"),
        "known_discards": decision.get("known_discards"),
        "candidate_count": len(scored),
        "skipped_candidates": skipped,
        "true_best_pred_rank": pred_rank,
        "regret": regret,
        "topk_exact_rerank_regret": topk_regrets,
        "teacher_gap": metric_gap(best_true["true"], second_true["true"] if second_true else None),
        "score_mae": score_abs / len(scored),
        "fl_rate_mae": fl_abs / len(scored),
        "bust_rate_mae": bust_abs / len(scored),
        "fl_type_mae": {key: fl_type_abs[key] / len(scored) for key in FL_KEYS},
        "teacher_best": {
            "candidate_index": best_true["candidate_index"],
            "action": action_label(best_true["candidate"]),
            "true": best_true["true"],
            "pred": best_true["pred"],
        },
        "model_pick": {
            "candidate_index": pred_pick["candidate_index"],
            "action": action_label(pred_pick["candidate"]),
            "true": pred_pick["true"],
            "pred": pred_pick["pred"],
        },
        "teacher_top3": [
            {
                "candidate_index": row["candidate_index"],
                "action": action_label(row["candidate"]),
                "score": row["true"]["score"],
                "fl_rate": row["true"]["fl_rate"],
                "bust_rate": row["true"]["bust_rate"],
            }
            for row in true_order[:3]
        ],
        "model_top3": [
            {
                "candidate_index": row["candidate_index"],
                "action": action_label(row["candidate"]),
                "pred_score": row["pred"]["score"],
                "true_score": row["true"]["score"],
                "true_fl_rate": row["true"]["fl_rate"],
                "true_bust_rate": row["true"]["bust_rate"],
            }
            for row in pred_order[:3]
        ],
        "tags": decision_tags(best_true["true"], regret, pred_rank, metric_gap(best_true["true"], second_true["true"] if second_true else None)),
    }


def decision_tags(best: Dict[str, Any], regret: float, pred_rank: int, gap: float) -> List[str]:
    tags: List[str] = []
    if pred_rank > 20:
        tags.append("miss_top20")
    elif pred_rank > 15:
        tags.append("miss_top15")
    elif pred_rank > 10:
        tags.append("miss_top10")
    elif pred_rank > 3:
        tags.append("miss_top3")
    elif pred_rank > 1:
        tags.append("miss_top1")
    if regret >= 2.0:
        tags.append("high_regret")
    if gap <= 0.5:
        tags.append("close_teacher_margin")
    if best["fl_rate"] >= 0.3:
        tags.append("fl_heavy")
    if best["bust_rate"] >= 0.2:
        tags.append("bust_risky")
    if best["fl_type_rates"].get("aa", 0.0) >= 0.2:
        tags.append("aa_route")
    if best["fl_type_rates"].get("kk", 0.0) >= 0.2:
        tags.append("kk_route")
    return tags


def update_bucket(bucket: MetricBucket, summary: Dict[str, Any]) -> None:
    bucket.skipped_candidates += int(summary["skipped_candidates"])
    topk_regrets = {
        int(k): float(v)
        for k, v in (summary.get("topk_exact_rerank_regret") or {}).items()
    }
    bucket.add_decision(int(summary["true_best_pred_rank"]), float(summary["regret"]), topk_regrets)
    n = int(summary["candidate_count"])
    bucket.add_candidate_errors(
        n,
        float(summary["score_mae"]) * n,
        float(summary["fl_rate_mae"]) * n,
        float(summary["bust_rate_mae"]) * n,
        {key: float(summary["fl_type_mae"][key]) * n for key in FL_KEYS},
    )


def priority_score(summary: Dict[str, Any]) -> float:
    return (
        float(summary["regret"]) * 10.0
        + max(0, int(summary["true_best_pred_rank"]) - 1)
        + float(summary["fl_rate_mae"]) * 3.0
        + float(summary["bust_rate_mae"]) * 2.0
    )


def write_markdown(path: Path, report: Dict[str, Any], weak_spots: List[Dict[str, Any]]) -> None:
    lines = [
        "# Tutor Teacher Model Evaluation",
        "",
        f"- input: `{report['input']}`",
        f"- model: `{report['model']}`",
        f"- decisions: {report['overall']['decisions']}",
        f"- candidates: {report['overall']['candidates']}",
        "",
        "## Overall",
        "",
    ]
    overall = report["overall"]
    for key, value in overall["top_recall"].items():
        lines.append(f"- {key}: {value:.1%}")
    lines.extend(
        [
            f"- avg_regret: {overall['avg_regret']:.3f}",
            f"- max_regret: {overall['max_regret']:.3f}",
            f"- score_mae: {overall['score_mae']:.3f}",
            f"- fl_rate_mae: {overall['fl_rate_mae']:.1%}",
            f"- bust_rate_mae: {overall['bust_rate_mae']:.1%}",
            "",
            "## By Turn",
            "",
        ]
    )
    for turn, stats in report["by_turn"].items():
        lines.append(
            f"- T{turn}: decisions={stats['decisions']} "
            f"top1={stats['top_recall']['top_1']:.1%} "
            f"top3={stats['top_recall']['top_3']:.1%} "
            f"top10={stats['top_recall']['top_10']:.1%} "
            f"top15={stats['top_recall']['top_15']:.1%} "
            f"top20={stats['top_recall']['top_20']:.1%} "
            f"top24={stats['top_recall']['top_24']:.1%} "
            f"top64={stats['top_recall']['top_64']:.1%} "
            f"regret={stats['avg_regret']:.3f} "
            f"score_mae={stats['score_mae']:.3f} "
            f"FL_MAE={stats['fl_rate_mae']:.1%} "
            f"bust_MAE={stats['bust_rate_mae']:.1%}"
        )

    lines.extend(["", "## Weak Spots", ""])
    for item in weak_spots[:10]:
        lines.append(
            f"- T{item['turn']} {item['position']} rank={item['true_best_pred_rank']} "
            f"regret={item['regret']:.3f} tags={','.join(item['tags']) or '-'} "
            f"dealt={item['dealt']}"
        )
        lines.append(f"  - teacher: {item['teacher_best']['action']}")
        lines.append(f"  - model:   {item['model_pick']['action']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_turns = {int(v) for v in args.turns.split(",") if v.strip()}

    model_path = resolve_model_path(args.model)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ActionValueReranker.from_checkpoint(model_path, map_location=device)
    model.to(device)
    model.eval()

    overall = MetricBucket()
    by_turn: Dict[int, MetricBucket] = {}
    weak_spots: List[Dict[str, Any]] = []
    decision_count = 0
    record_count = 0

    for record_index, record in enumerate(iter_jsonl(input_path), start=1):
        record_count += 1
        for decision in iter_decisions(record, include_turns):
            scored, skipped = score_candidates(model, decision, device)
            summary = summarize_decision(record_index, decision_count + 1, decision, scored, skipped)
            if summary is None:
                overall.skipped_candidates += skipped
                continue
            decision_count += 1
            update_bucket(overall, summary)
            turn = int(summary["turn"])
            by_turn.setdefault(turn, MetricBucket())
            update_bucket(by_turn[turn], summary)
            weak_spots.append(summary)

        if args.limit_records and record_count >= args.limit_records:
            break

    weak_spots.sort(key=priority_score, reverse=True)
    weak_path = output_dir / "weak_spots.jsonl"
    with weak_path.open("w", encoding="utf-8") as f:
        for item in weak_spots[: args.weak_spots]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "input": str(input_path),
        "model": str(model_path),
        "device": str(device),
        "records": record_count,
        "turns": sorted(include_turns),
        "metric_note": "score is the teacher target_score used by the action-value reranker; FL and bust are candidate-level probabilities.",
        "overall": overall.as_dict(),
        "by_turn": {str(turn): bucket.as_dict() for turn, bucket in sorted(by_turn.items())},
        "weak_spots_path": str(weak_path),
        "recommended_generation_focus": build_recommendations(by_turn, weak_spots),
    }
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "summary.md", report, weak_spots[: args.weak_spots])
    return report


def build_recommendations(by_turn: Dict[int, MetricBucket], weak_spots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    recommendations = []
    for turn, bucket in sorted(by_turn.items()):
        stats = bucket.as_dict()
        miss_top3 = 1.0 - stats["top_recall"]["top_3"]
        priority = stats["avg_regret"] + miss_top3 * 3.0 + stats["fl_rate_mae"] + stats["bust_rate_mae"]
        recommendations.append(
            {
                "turn": turn,
                "priority_score": priority,
                "reason": {
                    "top3_miss_rate": miss_top3,
                    "avg_regret": stats["avg_regret"],
                    "fl_rate_mae": stats["fl_rate_mae"],
                    "bust_rate_mae": stats["bust_rate_mae"],
                },
            }
        )
    tag_counts: Dict[str, int] = {}
    for item in weak_spots[:50]:
        for tag in item.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    recommendations.sort(key=lambda item: item["priority_score"], reverse=True)
    return [
        {
            "focus": "turn_data",
            "items": recommendations,
        },
        {
            "focus": "weak_spot_tags",
            "items": sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True),
        },
    ]


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate action-value reranker against recursive teacher data")
    parser.add_argument("input", help="Recursive tutor teacher JSONL")
    parser.add_argument("--model", default="auto", help="'auto' or path to action_value_best.pt")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--turns", default="0,1,2,3", help="Turns to evaluate")
    parser.add_argument("--device", default=None, help="cpu/cuda override")
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--weak-spots", type=int, default=50)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = evaluate(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
