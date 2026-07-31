"""Train a T2 sync-candidate selector from MC-labeled candidate pools.

The live T2 path cannot refine many candidates inside five seconds.  This
selector learns which Pool20 candidates should enter the small synchronous
refinement set, using only features available before refinement starts.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.benchmark_t2_t3_union_runtime import (
    DEFAULT_INPUT,
    DEFAULT_RUNTIME_CONFIG,
    candidate_teacher_score,
    extract_t2_payload,
)
from ai.tutor.exact_late import action_key
from ai.tutor.hybrid_t1t2 import (
    HybridConfig,
    _runtime_config_mode_payload,
    evaluate_hybrid_position,
    make_action_value_evaluator,
    t2_sync_selector_features,
)

DEFAULT_FEATURES = [
    "model_score",
    "model_raw_score",
    "neg_model_rank",
    "inv_model_rank",
    "rank_le_1",
    "rank_le_3",
    "rank_le_6",
    "rank_le_10",
    "rank_le_15",
    "neg_refinement_rank",
    "inv_refinement_rank",
    "refinement_rank_le_6",
    "refinement_rank_le_10",
    "predicted_bust",
    "predicted_safe",
    "predicted_fl",
    "candidate_fl_score",
    "predicted_qq",
    "predicted_kk",
    "predicted_aa",
    "predicted_trips",
    "premium_fl_sum",
    "insurance_low_bust",
    "insurance_fl",
    "forced_bust",
    "not_forced_bust",
    "model_score_x_inv_rank",
    "model_score_x_not_forced_bust",
    "fl_score_x_safe",
    "placed_top_count",
    "placed_middle_count",
    "placed_bottom_count",
    "final_top_count",
    "final_middle_count",
    "final_bottom_count",
    "existing_top_count",
    "existing_middle_count",
    "existing_bottom_count",
    "bottom_sparse_two_placed",
    "middle_sparse_two_placed",
    "top_full_after",
    "middle_full_after",
    "bottom_full_after",
    "final_top_pair",
    "final_top_trips",
    "final_middle_pair",
    "final_middle_trips",
    "final_bottom_pair",
    "final_bottom_trips",
    "placed_top_has_a",
    "placed_top_has_k",
    "placed_top_has_q",
    "placed_has_joker",
    "discard_a",
    "discard_k",
    "discard_q",
    "discard_joker",
    "opp_top_count",
    "opp_middle_count",
    "opp_bottom_count",
    "opp_top_pair",
    "opp_top_trips",
    "opp_middle_pair",
    "opp_middle_trips",
    "opp_bottom_pair",
    "opp_bottom_trips",
    "dead_a",
    "own_visible_a",
    "opp_visible_a",
    "remaining_a",
    "dead_k",
    "own_visible_k",
    "opp_visible_k",
    "remaining_k",
    "dead_q",
    "own_visible_q",
    "opp_visible_q",
    "remaining_q",
    "dead_j",
    "own_visible_j",
    "opp_visible_j",
    "remaining_j",
    "dead_t",
    "own_visible_t",
    "opp_visible_t",
    "remaining_t",
    "dead_2",
    "own_visible_2",
    "opp_visible_2",
    "remaining_2",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(path: str | Path) -> Path:
    out = Path(path)
    if out.is_absolute():
        return out
    return repo_root() / out


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[tuple[int, dict[str, Any]]]:
    emitted = 0
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            yield line_no, json.loads(line)
            emitted += 1
            if limit > 0 and emitted >= limit:
                break


def source_teacher_scores(record: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for candidate in record.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        score = candidate_teacher_score(candidate)
        if score is None:
            continue
        action = {
            "placements": candidate.get("placements") or [],
            "discard": candidate.get("discard"),
        }
        out[action_key(action)] = float(score)
    return out


def config_from_runtime(runtime_config: Path, runtime_mode: str | None = None) -> tuple[str, dict[str, Any], HybridConfig, str]:
    payload = json.loads(runtime_config.read_text(encoding="utf-8-sig"))
    mode_name, mode_payload = _runtime_config_mode_payload(payload, runtime_mode)
    model_path = (payload.get("models") or {}).get("action_value")
    if not model_path:
        raise ValueError("Runtime config must provide models.action_value")
    defaults = HybridConfig()
    values = {field.name: mode_payload.get(field.name, getattr(defaults, field.name)) for field in fields(HybridConfig)}
    values["enable_sync_refinement"] = False
    values["t2_sync_selection_policy"] = "rank"
    values["t2_sync_selector"] = None
    return mode_name, mode_payload, HybridConfig(**values), str(model_path)


def build_selector_rows(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    input_path = resolve_path(args.input)
    runtime_config = resolve_path(args.runtime_config)
    mode_name, _mode_payload, config, model_path = config_from_runtime(runtime_config, args.runtime_mode or None)
    if args.model:
        model_path = args.model
    evaluator = make_action_value_evaluator(model_path=model_path, device=args.device)
    groups: dict[str, list[dict[str, Any]]] = {}
    output_rows_path = Path(args.rows_output) if args.rows_output else None
    if output_rows_path is not None:
        output_rows_path.parent.mkdir(parents=True, exist_ok=True)
        rows_file = output_rows_path.open("w", encoding="utf-8")
    else:
        rows_file = None
    try:
        processed = 0
        for source_line, record in iter_jsonl(input_path, args.limit):
            payload = extract_t2_payload(record)
            if payload is None:
                continue
            teacher_scores = source_teacher_scores(payload)
            if not teacher_scores:
                continue
            result = evaluate_hybrid_position(payload, evaluator=evaluator, config=config)
            pool_rows = []
            full_teacher_key, full_teacher_score = max(teacher_scores.items(), key=lambda item: item[1])
            for candidate in result.get("candidates") or []:
                action = candidate.get("action")
                if not isinstance(action, dict):
                    continue
                key = action_key(action)
                if key not in teacher_scores:
                    continue
                fl_types = candidate.get("predicted_fl_types") or {}
                row = {
                    "group_id": f"{source_line}",
                    "source_line": int(source_line),
                    "turn": 2,
                    "runtime_mode": mode_name,
                    "position": payload.get("position") or ("btn" if payload.get("is_btn") else "bb"),
                    "action_key": key,
                    "action": action,
                    "action_idx": int(candidate.get("action_idx", -1)),
                    "model_score": float(candidate.get("model_score", 0.0) or 0.0),
                    "model_raw_score": float(candidate.get("model_raw_score", candidate.get("model_score", 0.0)) or 0.0),
                    "model_rank": int(candidate.get("model_rank", 999) or 999),
                    "refinement_rank": int(candidate.get("refinement_rank", candidate.get("model_rank", 999)) or 999),
                    "predicted_bust": float(candidate.get("predicted_bust", 0.0) or 0.0),
                    "predicted_fl": float(candidate.get("predicted_fl", 0.0) or 0.0),
                    "predicted_qq": float(fl_types.get("qq", 0.0) or 0.0),
                    "predicted_kk": float(fl_types.get("kk", 0.0) or 0.0),
                    "predicted_aa": float(fl_types.get("aa", 0.0) or 0.0),
                    "predicted_trips": float(fl_types.get("trips", 0.0) or 0.0),
                    "insurance_reason": str(candidate.get("insurance_reason") or ""),
                    "forced_bust": bool(candidate.get("forced_bust", False)),
                    "teacher_score": float(teacher_scores[key]),
                    "full_teacher_best_score": float(full_teacher_score),
                    "full_teacher_best_action_key": full_teacher_key,
                    "is_full_teacher_best": key == full_teacher_key,
                    "full_teacher_in_pool": False,
                    "pool_teacher_best_score": 0.0,
                    "is_pool_teacher_best": False,
                    "teacher_regret": max(0.0, float(full_teacher_score) - float(teacher_scores[key])),
                    "dealt": payload.get("dealt"),
                    "board": payload.get("board"),
                    "opponent_board": payload.get("opponent_board"),
                    "known_discards": payload.get("known_discards") or [],
                }
                pool_rows.append(row)
            if not pool_rows:
                continue
            pool_best = max(pool_rows, key=lambda row: float(row["teacher_score"]))
            full_in_pool = any(bool(row["is_full_teacher_best"]) for row in pool_rows)
            for row in pool_rows:
                row["full_teacher_in_pool"] = bool(full_in_pool)
                row["pool_teacher_best_score"] = float(pool_best["teacher_score"])
                row["is_pool_teacher_best"] = row["action_key"] == pool_best["action_key"]
                row["pool_teacher_regret"] = max(0.0, float(pool_best["teacher_score"]) - float(row["teacher_score"]))
                if rows_file is not None:
                    rows_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            groups[str(source_line)] = pool_rows
            processed += 1
            if args.progress_every > 0 and processed % args.progress_every == 0:
                print(f"processed_groups={processed}", flush=True)
    finally:
        if rows_file is not None:
            rows_file.close()
    return groups


def split_groups(groups: dict[str, list[dict[str, Any]]], dev_ratio: float, seed: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    dev_n = max(1, int(round(len(keys) * dev_ratio))) if len(keys) > 1 and dev_ratio > 0.0 else 0
    dev_keys = set(keys[:dev_n])
    train = {key: rows for key, rows in groups.items() if key not in dev_keys}
    dev = {key: rows for key, rows in groups.items() if key in dev_keys}
    return train, dev


def vectorize(
    groups: dict[str, list[dict[str, Any]]],
    features: list[str],
    means: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
) -> tuple[list[torch.Tensor], list[int], list[torch.Tensor], dict[str, float], dict[str, float]]:
    raw_vectors: list[list[float]] = []
    group_vectors: list[list[list[float]]] = []
    group_regrets: list[list[float]] = []
    labels: list[int] = []
    for rows in groups.values():
        vectors = [[t2_sync_selector_features(row).get(name, 0.0) for name in features] for row in rows]
        label = next((idx for idx, row in enumerate(rows) if bool(row.get("is_pool_teacher_best"))), 0)
        regrets = [float(row.get("pool_teacher_regret", 0.0) or 0.0) for row in rows]
        group_vectors.append(vectors)
        group_regrets.append(regrets)
        raw_vectors.extend(vectors)
        labels.append(label)
    if means is None or scales is None:
        matrix = torch.tensor(raw_vectors, dtype=torch.float32)
        mean_tensor = matrix.mean(dim=0)
        scale_tensor = matrix.std(dim=0, unbiased=False)
        scale_tensor = torch.where(scale_tensor < 1e-9, torch.ones_like(scale_tensor), scale_tensor)
        means = {name: float(mean_tensor[i].item()) for i, name in enumerate(features)}
        scales = {name: float(scale_tensor[i].item()) for i, name in enumerate(features)}
    mean = torch.tensor([means[name] for name in features], dtype=torch.float32)
    scale = torch.tensor([max(scales[name], 1e-9) for name in features], dtype=torch.float32)
    tensors = [(torch.tensor(vectors, dtype=torch.float32) - mean) / scale for vectors in group_vectors]
    regret_tensors = [torch.tensor(regrets, dtype=torch.float32) for regrets in group_regrets]
    return tensors, labels, regret_tensors, means, scales


def train_selector(
    groups: dict[str, list[dict[str, Any]]],
    *,
    features: list[str],
    epochs: int,
    lr: float,
    l2: float,
    regret_weight: float,
    regret_scale: float,
    seed: int,
) -> tuple[dict[str, Any], list[float]]:
    random.seed(seed)
    torch.manual_seed(seed)
    tensors, labels, regrets, means, scales = vectorize(groups, features)
    weights = torch.zeros(len(features), dtype=torch.float32, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights, intercept], lr=lr)
    order = list(range(len(tensors)))
    losses: list[float] = []
    for _epoch in range(max(1, epochs)):
        random.shuffle(order)
        total = torch.zeros((), dtype=torch.float32)
        optimizer.zero_grad()
        for idx in order:
            scores = tensors[idx].mv(weights) + intercept
            ce_loss = torch.logsumexp(scores, dim=0) - scores[int(labels[idx])]
            regret_loss = (torch.softmax(scores, dim=0) * regrets[idx]).sum() / max(float(regret_scale), 1e-9)
            total = total + ce_loss + float(regret_weight) * regret_loss
        loss = total / max(len(tensors), 1) + float(l2) * weights.pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    selector = {
        "kind": "t2_sync_linear_selector",
        "features": features,
        "weights": [float(value) for value in weights.detach().tolist()],
        "intercept": float(intercept.detach().item()),
        "means": means,
        "scales": scales,
        "epochs": int(epochs),
        "lr": float(lr),
        "l2": float(l2),
        "regret_weight": float(regret_weight),
        "regret_scale": float(regret_scale),
        "seed": int(seed),
        "final_loss": losses[-1] if losses else None,
    }
    return selector, losses


def selector_score(row: dict[str, Any], selector: dict[str, Any]) -> float:
    features = t2_sync_selector_features(row)
    total = float(selector.get("intercept", 0.0))
    for name, weight in zip(selector["features"], selector["weights"]):
        raw = float(features.get(name, 0.0))
        raw = (raw - float(selector["means"].get(name, 0.0))) / max(float(selector["scales"].get(name, 1.0)), 1e-9)
        total += float(weight) * raw
    return total


def selected_rows(rows: list[dict[str, Any]], ranked_rows: list[dict[str, Any]], k: int, high_bust: float) -> list[dict[str, Any]]:
    viable = [row for row in ranked_rows if not bool(row.get("forced_bust", False))]
    if viable:
        ranked_rows = viable
    if any(float(row.get("predicted_bust", 0.0) or 0.0) < high_bust for row in ranked_rows):
        ranked_rows = [row for row in ranked_rows if float(row.get("predicted_bust", 0.0) or 0.0) < high_bust]
    return ranked_rows[:k] if ranked_rows else rows[:k]


def evaluate_groups(
    groups: dict[str, list[dict[str, Any]]],
    *,
    selector: dict[str, Any] | None,
    sync_ks: list[int],
    high_bust: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {"decisions": len(groups)}
    policies = {
        "model_rank": lambda rows: sorted(rows, key=lambda row: (int(row.get("refinement_rank", row["model_rank"])), int(row["model_rank"]))),
    }
    if selector is not None:
        policies["selector"] = lambda rows: sorted(
            rows,
            key=lambda row: (selector_score(row, selector), -int(row.get("model_rank", 999))),
            reverse=True,
        )
    for policy_name, ranker in policies.items():
        policy_report: dict[str, Any] = {}
        for k in sync_ks:
            pool_hits = 0
            full_hits = 0
            losses: list[float] = []
            pool_losses: list[float] = []
            for rows in groups.values():
                selected = selected_rows(rows, ranker(rows), k, high_bust)
                pool_hits += int(any(bool(row.get("is_pool_teacher_best")) for row in selected))
                full_hits += int(any(bool(row.get("is_full_teacher_best")) for row in selected))
                best_selected_score = max(float(row["teacher_score"]) for row in selected)
                full_best_score = max(float(row["full_teacher_best_score"]) for row in rows)
                pool_best_score = max(float(row["pool_teacher_best_score"]) for row in rows)
                losses.append(max(0.0, full_best_score - best_selected_score))
                pool_losses.append(max(0.0, pool_best_score - best_selected_score))
            policy_report[f"pool_hit_at_{k}"] = pool_hits / max(len(groups), 1)
            policy_report[f"full_hit_at_{k}"] = full_hits / max(len(groups), 1)
            policy_report[f"ev_loss_mean_at_{k}"] = sum(losses) / max(len(losses), 1)
            policy_report[f"ev_loss_max_at_{k}"] = max(losses) if losses else 0.0
            policy_report[f"ev_loss_ge_0_5_at_{k}"] = sum(loss >= 0.5 for loss in losses)
            policy_report[f"ev_loss_ge_1_0_at_{k}"] = sum(loss >= 1.0 for loss in losses)
            policy_report[f"pool_ev_loss_mean_at_{k}"] = sum(pool_losses) / max(len(pool_losses), 1)
        report[policy_name] = policy_report
    return report


def parse_ks(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--runtime-config", default=str(DEFAULT_RUNTIME_CONFIG))
    parser.add_argument("--runtime-mode", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows-output", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--regret-weight", type=float, default=0.5)
    parser.add_argument("--regret-scale", type=float, default=3.0)
    parser.add_argument("--dev-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sync-ks", default="3,4,5,10")
    parser.add_argument("--high-bust-threshold", type=float, default=0.999)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args(list(argv) if argv is not None else None)

    groups = build_selector_rows(args)
    if not groups:
        raise SystemExit("No T2 selector groups were built")
    train_groups, dev_groups = split_groups(groups, float(args.dev_ratio), int(args.seed))
    features = [part.strip() for part in str(args.features).split(",") if part.strip()]
    selector, _losses = train_selector(
        train_groups,
        features=features,
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        regret_weight=float(args.regret_weight),
        regret_scale=float(args.regret_scale),
        seed=int(args.seed),
    )
    output = Path(args.output)
    selector.update(
        {
            "name": output.stem,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "train_inputs": [str(resolve_path(args.input))],
            "runtime_config": str(resolve_path(args.runtime_config)),
            "train_groups": len(train_groups),
            "dev_groups": len(dev_groups),
            "metrics": {
                "train": evaluate_groups(
                    train_groups,
                    selector=selector,
                    sync_ks=parse_ks(args.sync_ks),
                    high_bust=float(args.high_bust_threshold),
                ),
                "dev": evaluate_groups(
                    dev_groups,
                    selector=selector,
                    sync_ks=parse_ks(args.sync_ks),
                    high_bust=float(args.high_bust_threshold),
                )
                if dev_groups
                else {},
                "all": evaluate_groups(
                    groups,
                    selector=selector,
                    sync_ks=parse_ks(args.sync_ks),
                    high_bust=float(args.high_bust_threshold),
                ),
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selector, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(selector, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
