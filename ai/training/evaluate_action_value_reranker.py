"""Evaluate an action-value reranker on candidate-level teacher data."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_reranker import ActionValueReranker


def parse_topks(value: str) -> list[int]:
    topks = sorted({int(part) for part in value.split(",") if part.strip()})
    return [k for k in topks if k > 0]


def parse_weights(value: str, n: int) -> list[float]:
    if not value.strip():
        return [1.0 / max(n, 1)] * n
    weights = [float(part) for part in value.split(",") if part.strip()]
    if len(weights) != n:
        raise ValueError(f"--weights has {len(weights)} values but {n} checkpoints were provided")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("--weights must sum to a positive value")
    return [w / total for w in weights]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    xx = x.astype(np.float64, copy=True)
    yy = y.astype(np.float64, copy=True)
    xx -= xx.mean()
    yy -= yy.mean()
    denom = math.sqrt(float((xx * xx).sum() * (yy * yy).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((xx * yy).sum() / denom)


def load_metadata(data_dir: Path) -> dict:
    path = data_dir / "metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_source_records(path: str | Path | None, max_records: int) -> list[dict]:
    if not path:
        return []
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if len(records) >= max_records:
                break
    return records


def build_group_bounds(group_ids: np.ndarray) -> list[tuple[int, int]]:
    if len(group_ids) == 0:
        return []
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(group_ids)):
        if group_ids[i] != group_ids[i - 1]:
            bounds.append((start, i))
            start = i
    bounds.append((start, len(group_ids)))
    return bounds


def filter_group_bounds_by_position(
    bounds: list[tuple[int, int]],
    positions: np.ndarray | None,
    position: str,
) -> list[tuple[int, int]]:
    """Keep complete candidate groups for one table position."""
    if position == "all":
        return bounds
    if positions is None:
        raise ValueError("--position requires positions.npy in the data directory")
    expected = 1 if position == "btn" else 0
    selected: list[tuple[int, int]] = []
    for start, end in bounds:
        group_positions = np.asarray(positions[start:end], dtype=np.int8)
        if len(group_positions) == 0:
            continue
        if not np.all(group_positions == group_positions[0]):
            raise ValueError(f"mixed positions within candidate group at [{start}:{end}]")
        if int(group_positions[0]) == expected:
            selected.append((start, end))
    return selected


def candidate_label(candidate: dict) -> str:
    placements = [
        f"{card}->{row}"
        for card, row in candidate.get("placements", []) or []
    ]
    discard = candidate.get("discard")
    if discard not in (None, ""):
        placements.append(f"discard {discard}")
    return "; ".join(placements)


@torch.no_grad()
def predict(
    model: ActionValueReranker,
    states: np.ndarray,
    turns: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    n = int(states.shape[0])
    pred_score = np.zeros(n, dtype=np.float32)
    pred_bust = np.zeros(n, dtype=np.float32)
    pred_fl = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        state = torch.from_numpy(np.array(states[start:end], dtype=np.float32, copy=True)).to(device)
        turn = torch.from_numpy(np.asarray(turns[start:end], dtype=np.int64)).to(device)
        out = model.predict_components(state, turn=turn)
        pred_score[start:end] = out["score"].detach().cpu().numpy()
        pred_bust[start:end] = out["bust_prob"].detach().cpu().numpy()
        pred_fl[start:end] = out["fl_prob"].detach().cpu().numpy()
    return pred_score, pred_bust, pred_fl


def predict_ensemble(
    checkpoint_paths: list[str],
    weights: list[float],
    states: np.ndarray,
    turns: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_score: np.ndarray | None = None
    pred_bust: np.ndarray | None = None
    pred_fl: np.ndarray | None = None
    for checkpoint, weight in zip(checkpoint_paths, weights):
        model = ActionValueReranker.from_checkpoint(checkpoint, map_location=device).to(device)
        score_i, bust_i, fl_i = predict(model, states, turns, device, batch_size)
        del model
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()
        if pred_score is None:
            pred_score = np.zeros_like(score_i, dtype=np.float32)
            pred_bust = np.zeros_like(bust_i, dtype=np.float32)
            pred_fl = np.zeros_like(fl_i, dtype=np.float32)
        pred_score += float(weight) * score_i
        pred_bust += float(weight) * bust_i
        pred_fl += float(weight) * fl_i
    if pred_score is None or pred_bust is None or pred_fl is None:
        raise ValueError("No checkpoints were provided")
    return pred_score, pred_bust, pred_fl


def summarize_groups(
    scores: np.ndarray,
    bust: np.ndarray,
    fl: np.ndarray,
    pred_score: np.ndarray,
    pred_bust: np.ndarray,
    pred_fl: np.ndarray,
    bounds: list[tuple[int, int]],
    topks: list[int],
    bust_full_threshold: float,
) -> tuple[dict, list[dict]]:
    hits = {k: 0 for k in topks}
    rerank_regret = {k: 0.0 for k in topks}
    pred_top1_regret = 0.0
    teacher_ranks: list[int] = []
    bad_full_bust = 0
    rows: list[dict] = []

    for group_id, (start, end) in enumerate(bounds):
        idx = np.arange(start, end, dtype=np.int64)
        true = np.asarray(scores[idx], dtype=np.float64)
        pred = np.asarray(pred_score[idx], dtype=np.float64)
        true_best_local = int(np.argmax(true))
        ordered = np.argsort(-pred)
        pred_best_local = int(ordered[0])
        true_rank = int(np.where(ordered == true_best_local)[0][0]) + 1
        teacher_ranks.append(true_rank)
        true_best_score = float(true[true_best_local])
        pred_best_true_score = float(true[pred_best_local])
        pred_top1_regret += true_best_score - pred_best_true_score
        if (
            float(bust[idx[pred_best_local]]) >= bust_full_threshold
            and float(np.min(bust[idx])) < bust_full_threshold
        ):
            bad_full_bust += 1
        for k in topks:
            kk = min(k, len(idx))
            if true_rank <= kk:
                hits[k] += 1
            best_in_topk = float(true[ordered[:kk]].max())
            rerank_regret[k] += true_best_score - best_in_topk
        rows.append(
            {
                "group_id": group_id,
                "size": int(len(idx)),
                "teacher_rank_by_model": true_rank,
                "teacher_best_index": int(idx[true_best_local]),
                "model_best_index": int(idx[pred_best_local]),
                "teacher_best_score": true_best_score,
                "model_best_true_score": pred_best_true_score,
                "model_best_pred_score": float(pred[pred_best_local]),
                "regret": true_best_score - pred_best_true_score,
                "teacher_best_bust": float(bust[idx[true_best_local]]),
                "model_best_bust": float(bust[idx[pred_best_local]]),
                "teacher_best_fl": float(fl[idx[true_best_local]]),
                "model_best_fl": float(fl[idx[pred_best_local]]),
                "model_pred_bust": float(pred_bust[idx[pred_best_local]]),
                "model_pred_fl": float(pred_fl[idx[pred_best_local]]),
                "ordered_indices": [int(idx[i]) for i in ordered],
            }
        )

    n_groups = max(len(bounds), 1)
    rank_arr = np.asarray(teacher_ranks, dtype=np.float64)
    selected_indices = (
        np.concatenate(
            [np.arange(start, end, dtype=np.int64) for start, end in bounds]
        )
        if bounds
        else np.asarray([], dtype=np.int64)
    )
    selected_scores = np.asarray(scores[selected_indices])
    selected_pred_scores = np.asarray(pred_score[selected_indices])
    selected_bust = np.asarray(bust[selected_indices])
    selected_pred_bust = np.asarray(pred_bust[selected_indices])
    selected_fl = np.asarray(fl[selected_indices])
    selected_pred_fl = np.asarray(pred_fl[selected_indices])
    metrics = {
        "groups": int(len(bounds)),
        "samples": int(len(selected_indices)),
        "score_mae": (
            float(np.mean(np.abs(selected_pred_scores - selected_scores)))
            if len(selected_indices)
            else 0.0
        ),
        "score_corr": pearson_corr(selected_pred_scores, selected_scores),
        "bust_mae": (
            float(np.mean(np.abs(selected_pred_bust - selected_bust)))
            if len(selected_indices)
            else 0.0
        ),
        "fl_mae": (
            float(np.mean(np.abs(selected_pred_fl - selected_fl)))
            if len(selected_indices)
            else 0.0
        ),
        "group_top1_regret": pred_top1_regret / n_groups,
        "teacher_rank_mean": float(rank_arr.mean()) if len(rank_arr) else 0.0,
        "teacher_rank_p95": float(np.percentile(rank_arr, 95)) if len(rank_arr) else 0.0,
        "model_chose_full_bust_when_avoidable": int(bad_full_bust),
    }
    for k in topks:
        metrics[f"group_top{k}"] = hits[k] / n_groups
        metrics[f"group_top{k}_rerank_regret"] = rerank_regret[k] / n_groups
    return metrics, rows


def write_misses(
    path: Path,
    rows: list[dict],
    records: list[dict],
    pred_score: np.ndarray,
    scores: np.ndarray,
    bust: np.ndarray,
    fl: np.ndarray,
    miss_topk: int,
    top_predictions: int,
) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if int(row["teacher_rank_by_model"]) <= miss_topk:
                continue
            group_id = int(row["group_id"])
            raw = records[group_id] if group_id < len(records) else {}
            candidates = raw.get("candidates") or []
            ordered = row.get("ordered_indices", [])
            top_pred = []
            for global_idx in ordered[:top_predictions]:
                candidate = candidates[global_idx - min(ordered)] if candidates and 0 <= global_idx - min(ordered) < len(candidates) else {}
                top_pred.append(
                    {
                        "sample_index": int(global_idx),
                        "action": candidate_label(candidate) if candidate else "",
                        "pred_score": float(pred_score[global_idx]),
                        "teacher_score": float(scores[global_idx]),
                        "teacher_bust": float(bust[global_idx]),
                        "teacher_fl": float(fl[global_idx]),
                    }
                )
            teacher_best_idx = int(row["teacher_best_index"])
            model_best_idx = int(row["model_best_index"])
            base = min(ordered) if ordered else 0
            teacher_candidate = candidates[teacher_best_idx - base] if candidates and 0 <= teacher_best_idx - base < len(candidates) else {}
            model_candidate = candidates[model_best_idx - base] if candidates and 0 <= model_best_idx - base < len(candidates) else {}
            out = {
                "group_id": group_id,
                "source": raw.get("source"),
                "source_line": raw.get("source_line"),
                "turn": raw.get("turn"),
                "board": raw.get("board"),
                "opponent_board": raw.get("opponent_board"),
                "dealt": raw.get("dealt"),
                "teacher_rank_by_model": row["teacher_rank_by_model"],
                "regret": row["regret"],
                "teacher_best": {
                    "action": candidate_label(teacher_candidate) if teacher_candidate else "",
                    "score": row["teacher_best_score"],
                    "bust": row["teacher_best_bust"],
                    "fl": row["teacher_best_fl"],
                },
                "model_best": {
                    "action": candidate_label(model_candidate) if model_candidate else "",
                    "teacher_score": row["model_best_true_score"],
                    "pred_score": row["model_best_pred_score"],
                    "bust": row["model_best_bust"],
                    "fl": row["model_best_fl"],
                },
                "top_predictions": top_pred,
            }
            f.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def write_summary_md(path: Path, summary: dict, topks: list[int]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# Action-Value Reranker Evaluation",
        "",
        f"- label: `{summary['label']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- checkpoints: {len(summary.get('checkpoints', []))}",
        f"- data: `{summary['data']}`",
        f"- groups: {metrics['groups']:,}",
        f"- samples: {metrics['samples']:,}",
        f"- score MAE/corr: {metrics['score_mae']:.3f} / {metrics['score_corr']:.3f}",
        f"- bust MAE: {metrics['bust_mae']:.3f}",
        f"- FL MAE: {metrics['fl_mae']:.3f}",
        f"- teacher rank mean/p95: {metrics['teacher_rank_mean']:.2f} / {metrics['teacher_rank_p95']:.1f}",
        f"- full-bust chosen when avoidable: {metrics['model_chose_full_bust_when_avoidable']}",
        "",
        "## Group Recall",
        "",
        "| K | recall | rerank regret |",
        "|---:|---:|---:|",
    ]
    for k in topks:
        lines.append(
            f"| {k} | {metrics[f'group_top{k}']:.1%} | {metrics[f'group_top{k}_rerank_regret']:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    start_time = time.time()
    data_dir = Path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topks = parse_topks(args.topks)
    meta = load_metadata(data_dir)
    checkpoint_paths = list(args.checkpoints or [])
    if args.checkpoint:
        checkpoint_paths.insert(0, args.checkpoint)
    if not checkpoint_paths:
        raise ValueError("Provide --checkpoint or --checkpoints")
    weights = parse_weights(args.weights, len(checkpoint_paths))

    states = np.load(data_dir / "states.npy", mmap_mode="r")
    n_samples = min(int(meta.get("n_samples", states.shape[0])), int(states.shape[0]))
    states = states[:n_samples]
    scores = np.asarray(np.load(data_dir / "scores.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    bust = np.asarray(np.load(data_dir / "bust.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    fl = np.asarray(np.load(data_dir / "fl.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    turns = np.asarray(np.load(data_dir / "turns.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    bounds = build_group_bounds(group_ids)
    positions_path = data_dir / "positions.npy"
    positions = (
        np.load(positions_path, mmap_mode="r")[:n_samples]
        if positions_path.exists()
        else None
    )
    bounds = filter_group_bounds_by_position(bounds, positions, args.position)
    if args.max_groups > 0:
        bounds = bounds[: args.max_groups]
        n_samples = bounds[-1][1] if bounds else 0
        states = states[:n_samples]
        scores = scores[:n_samples]
        bust = bust[:n_samples]
        fl = fl[:n_samples]
        turns = turns[:n_samples]

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    pred_score, pred_bust, pred_fl = predict_ensemble(
        checkpoint_paths,
        weights,
        states,
        turns,
        device,
        args.batch_size,
    )

    metrics, rows = summarize_groups(
        scores,
        bust,
        fl,
        pred_score,
        pred_bust,
        pred_fl,
        bounds,
        topks,
        args.bust_full_threshold,
    )
    records = load_source_records(args.source_jsonl, len(bounds))
    misses_path = out_dir / "misses.jsonl"
    misses = write_misses(
        misses_path,
        rows,
        records,
        pred_score,
        scores,
        bust,
        fl,
        args.miss_topk,
        args.miss_top_predictions,
    )
    summary = {
        "label": args.label,
        "checkpoint": str(checkpoint_paths[0]) if len(checkpoint_paths) == 1 else "ensemble",
        "checkpoints": [str(path) for path in checkpoint_paths],
        "weights": weights,
        "data": str(data_dir),
        "source_jsonl": str(args.source_jsonl) if args.source_jsonl else "",
        "topks": topks,
        "position": args.position,
        "metrics": metrics,
        "miss_topk": int(args.miss_topk),
        "misses": int(misses),
        "misses_path": str(misses_path),
        "elapsed_seconds": time.time() - start_time,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_summary_md(out_dir / "summary.md", summary, topks)

    print("Action-value reranker evaluation")
    print(f"  label: {args.label}")
    if len(checkpoint_paths) == 1:
        print(f"  checkpoint: {checkpoint_paths[0]}")
    else:
        print("  checkpoints:")
        for checkpoint, weight in zip(checkpoint_paths, weights):
            print(f"    {weight:.3f} {checkpoint}")
    print(f"  data: {data_dir}")
    print(f"  position: {args.position}")
    print(f"  groups={metrics['groups']:,} samples={metrics['samples']:,}")
    print(
        f"  score_mae={metrics['score_mae']:.3f} corr={metrics['score_corr']:.3f} "
        f"bust_mae={metrics['bust_mae']:.3f} fl_mae={metrics['fl_mae']:.3f}"
    )
    for k in topks:
        print(
            f"  top{k}={metrics[f'group_top{k}']:.1%} "
            f"top{k}_regret={metrics[f'group_top{k}_rerank_regret']:.3f}"
        )
    print(f"  top1_regret={metrics['group_top1_regret']:.3f}")
    print(f"  full_bust_when_avoidable={metrics['model_chose_full_bust_when_avoidable']}")
    print(f"  misses>{args.miss_topk}: {misses} saved={misses_path}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate an action-value reranker checkpoint")
    parser.add_argument("--data", required=True, help="Candidate-level reranker data directory")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path")
    parser.add_argument("--checkpoints", nargs="*", default=[], help="Additional checkpoint paths for weighted ensemble")
    parser.add_argument("--weights", default="", help="Comma-separated ensemble weights; defaults to uniform")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--label", default="model")
    parser.add_argument("--source-jsonl", default="", help="Optional source teacher JSONL for readable misses")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--miss-topk", type=int, default=1)
    parser.add_argument("--miss-top-predictions", type=int, default=5)
    parser.add_argument("--bust-full-threshold", type=float, default=0.999)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--position", choices=("all", "bb", "btn"), default="all")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
