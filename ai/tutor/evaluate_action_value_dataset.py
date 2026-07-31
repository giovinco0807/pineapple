"""Evaluate an action-value reranker on candidate-level teacher data.

This evaluates datasets produced by convert_action_value_teacher.py.  It is
useful for fixed holdout sets because it reports top-N recall by decision group
instead of only candidate-level losses.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_reranker import ActionValueReranker
from ai.tutor.evaluate_teacher_model import resolve_model_path


TOP_NS = (1, 3, 5, 10, 20)
FL_KEYS = ("qq", "kk", "aa", "trips")


@dataclass
class Bucket:
    top_ns: tuple[int, ...] = TOP_NS
    decisions: int = 0
    candidates: int = 0
    top_hits: dict[int, int] = field(init=False)
    topk_regret_sum: dict[int, float] = field(init=False)
    topk_regret_max: dict[int, float] = field(init=False)
    regret_sum: float = 0.0
    regret_max: float = 0.0
    score_abs_sum: float = 0.0
    fl_abs_sum: float = 0.0
    bust_abs_sum: float = 0.0
    fl_type_abs_sum: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in FL_KEYS})

    def __post_init__(self) -> None:
        self.top_hits = {k: 0 for k in self.top_ns}
        self.topk_regret_sum = {k: 0.0 for k in self.top_ns}
        self.topk_regret_max = {k: 0.0 for k in self.top_ns}

    def add_decision(self, rank: int, regret: float, n_candidates: int, topk_regrets: dict[int, float]) -> None:
        self.decisions += 1
        self.candidates += n_candidates
        for k in self.top_ns:
            if rank <= k:
                self.top_hits[k] += 1
            topk_regret = float(topk_regrets.get(k, regret))
            self.topk_regret_sum[k] += topk_regret
            self.topk_regret_max[k] = max(self.topk_regret_max[k], topk_regret)
        self.regret_sum += regret
        self.regret_max = max(self.regret_max, regret)

    def add_candidate_errors(
        self,
        score_abs: float,
        fl_abs: float,
        bust_abs: float,
        fl_type_abs: dict[str, float],
    ) -> None:
        self.score_abs_sum += score_abs
        self.fl_abs_sum += fl_abs
        self.bust_abs_sum += bust_abs
        for key in FL_KEYS:
            self.fl_type_abs_sum[key] += fl_type_abs.get(key, 0.0)

    def as_dict(self) -> dict[str, Any]:
        denom_d = max(self.decisions, 1)
        denom_c = max(self.candidates, 1)
        return {
            "decisions": self.decisions,
            "candidates": self.candidates,
            "top_recall": {f"top_{k}": self.top_hits[k] / denom_d for k in self.top_ns},
            "topk_exact_rerank_avg_regret": {
                f"top_{k}": self.topk_regret_sum[k] / denom_d for k in self.top_ns
            },
            "topk_exact_rerank_max_regret": {f"top_{k}": self.topk_regret_max[k] for k in self.top_ns},
            "avg_regret": self.regret_sum / denom_d,
            "max_regret": self.regret_max,
            "score_mae": self.score_abs_sum / denom_c,
            "fl_rate_mae": self.fl_abs_sum / denom_c,
            "bust_rate_mae": self.bust_abs_sum / denom_c,
            "fl_type_mae": {key: self.fl_type_abs_sum[key] / denom_c for key in FL_KEYS},
        }


def load_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def group_ranges(group_ids: np.ndarray) -> list[tuple[int, int]]:
    if len(group_ids) == 0:
        return []
    starts = [0]
    for i in range(1, len(group_ids)):
        if group_ids[i] != group_ids[i - 1]:
            starts.append(i)
    starts_arr = np.asarray(starts, dtype=np.int64)
    ends_arr = np.concatenate([starts_arr[1:], np.asarray([len(group_ids)], dtype=np.int64)])
    return [(int(s), int(e)) for s, e in zip(starts_arr, ends_arr)]


def parse_top_ns(value: str | Iterable[int] | None) -> tuple[int, ...]:
    if value is None:
        return TOP_NS
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        top_ns = [int(part) for part in parts if part]
    else:
        top_ns = [int(part) for part in value]
    cleaned = sorted({k for k in top_ns if k > 0})
    if not cleaned:
        raise ValueError("At least one positive top-N value is required")
    return tuple(cleaned)


def predict(
    model: ActionValueReranker,
    states: np.ndarray,
    turns: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    pred_score = np.zeros(states.shape[0], dtype=np.float32)
    pred_fl = np.zeros(states.shape[0], dtype=np.float32)
    pred_bust = np.zeros(states.shape[0], dtype=np.float32)
    pred_fl_types = np.zeros((states.shape[0], len(FL_KEYS)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, states.shape[0], batch_size):
            end = min(start + batch_size, states.shape[0])
            batch = torch.from_numpy(np.array(states[start:end], dtype=np.float32, copy=True)).to(device)
            turn = torch.from_numpy(np.asarray(turns[start:end], dtype=np.int64)).to(device)
            out = model.predict_components(batch, turn=turn)
            pred_score[start:end] = out["score"].detach().cpu().numpy()
            pred_fl[start:end] = out["fl_prob"].detach().cpu().numpy()
            pred_bust[start:end] = out["bust_prob"].detach().cpu().numpy()
            pred_fl_types[start:end] = out["fl_type_probs"].detach().cpu().numpy()
    return {
        "score": pred_score,
        "fl": pred_fl,
        "bust": pred_bust,
        "fl_types": pred_fl_types,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(data_dir)
    top_ns = parse_top_ns(getattr(args, "top_ns", None))

    states_mm = np.load(data_dir / "states.npy", mmap_mode="r")
    n_samples = min(int(metadata.get("n_samples", states_mm.shape[0])), states_mm.shape[0])
    if args.limit_samples:
        n_samples = min(n_samples, int(args.limit_samples))
    # Keep states memory-mapped.  Large candidate datasets can be several GB,
    # while prediction already copies only the current batch to the device.
    states = states_mm[:n_samples]
    scores = np.asarray(np.load(data_dir / "scores.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    fl = np.asarray(np.load(data_dir / "fl.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    bust = np.asarray(np.load(data_dir / "bust.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    turns = np.asarray(np.load(data_dir / "turns.npy", mmap_mode="r")[:n_samples], dtype=np.int16)
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    fl_types = (
        np.asarray(np.load(data_dir / "fl_types.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
        if (data_dir / "fl_types.npy").exists()
        else np.zeros((n_samples, len(FL_KEYS)), dtype=np.float32)
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_path = resolve_model_path(args.model)
    model = ActionValueReranker.from_checkpoint(model_path, map_location=device)
    model.to(device)
    preds = predict(model, states, turns, device, args.batch_size)

    overall = Bucket(top_ns=top_ns)
    by_turn: dict[int, Bucket] = {}
    weak: list[dict[str, Any]] = []
    for start, end in group_ranges(group_ids):
        if end <= start:
            continue
        true = scores[start:end]
        pred = preds["score"][start:end]
        order = np.argsort(-pred)
        teacher_best = int(np.argmax(true))
        pred_best = int(order[0])
        rank = int(np.where(order == teacher_best)[0][0]) + 1
        regret = float(true[teacher_best] - true[pred_best])
        topk_regrets = {}
        for k in top_ns:
            kept = order[: min(k, len(order))]
            best_kept = float(np.max(true[kept])) if len(kept) else float(true[pred_best])
            topk_regrets[k] = float(true[teacher_best] - best_kept)
        turn = int(turns[start])
        n = int(end - start)

        score_abs = float(np.abs(pred - true).sum())
        fl_abs = float(np.abs(preds["fl"][start:end] - fl[start:end]).sum())
        bust_abs = float(np.abs(preds["bust"][start:end] - bust[start:end]).sum())
        fl_type_abs = {
            key: float(np.abs(preds["fl_types"][start:end, i] - fl_types[start:end, i]).sum())
            for i, key in enumerate(FL_KEYS)
        }

        overall.add_decision(rank, regret, n, topk_regrets)
        overall.add_candidate_errors(score_abs, fl_abs, bust_abs, fl_type_abs)
        bucket = by_turn.get(turn)
        if bucket is None:
            bucket = Bucket(top_ns=top_ns)
            by_turn[turn] = bucket
        bucket.add_decision(rank, regret, n, topk_regrets)
        bucket.add_candidate_errors(score_abs, fl_abs, bust_abs, fl_type_abs)
        weak.append(
            {
                "group_id": int(group_ids[start]),
                "turn": turn,
                "rank": rank,
                "regret": regret,
                "n_candidates": n,
                "teacher_score": float(true[teacher_best]),
                "predicted_choice_teacher_score": float(true[pred_best]),
                "predicted_score": float(pred[pred_best]),
                "topk_exact_rerank_regret": {str(k): topk_regrets[k] for k in top_ns},
            }
        )

    weak.sort(key=lambda item: (item["regret"], item["rank"]), reverse=True)
    weak_path = output_dir / "weak_groups.jsonl"
    with weak_path.open("w", encoding="utf-8") as f:
        for item in weak[: args.weak_groups]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "data": str(data_dir),
        "model": str(model_path),
        "device": str(device),
        "n_samples": int(n_samples),
        "metadata": metadata,
        "top_ns": list(top_ns),
        "overall": overall.as_dict(),
        "by_turn": {str(turn): bucket.as_dict() for turn, bucket in sorted(by_turn.items())},
        "weak_groups_path": str(weak_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "summary.md", report)
    return report


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    top_ns = tuple(int(k) for k in report.get("top_ns", TOP_NS))
    top_columns = [f"top{k}" for k in top_ns]
    lines = [
        "# Action-Value Holdout Evaluation",
        "",
        f"- data: `{report['data']}`",
        f"- model: `{report['model']}`",
        f"- samples: {report['n_samples']:,}",
        "",
        "| scope | decisions | "
        + " | ".join(top_columns)
        + " | avg_regret | score_mae | FL_MAE | bust_MAE |",
        "|---|---:|" + "---:|" * len(top_columns) + "---:|---:|---:|---:|",
    ]
    rows = [("overall", report["overall"])] + [
        (f"T{turn}", stats) for turn, stats in sorted(report["by_turn"].items(), key=lambda kv: int(kv[0]))
    ]
    for name, stats in rows:
        top = stats["top_recall"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(stats["decisions"]),
                    *[pct(top[f"top_{k}"]) for k in top_ns],
                    f"{stats['avg_regret']:.3f}",
                    f"{stats['score_mae']:.3f}",
                    pct(stats["fl_rate_mae"]),
                    pct(stats["bust_rate_mae"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Exact Rerank After Pruning",
            "",
            "| scope | "
            + " | ".join([f"top{k} avg regret" for k in top_ns])
            + " | "
            + " | ".join([f"top{k} max regret" for k in top_ns])
            + " |",
            "|---|" + "---:|" * (len(top_ns) * 2),
        ]
    )
    for name, stats in rows:
        avg = stats["topk_exact_rerank_avg_regret"]
        max_ = stats["topk_exact_rerank_max_regret"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    *[f"{avg[f'top_{k}']:.3f}" for k in top_ns],
                    *[f"{max_[f'top_{k}']:.3f}" for k in top_ns],
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a reranker on candidate-level teacher data")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--weak-groups", type=int, default=100)
    parser.add_argument(
        "--top-ns",
        default="1,3,5,10,20",
        help="Comma-separated prune budgets to report, e.g. 1,3,5,10,20,24",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = evaluate(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
