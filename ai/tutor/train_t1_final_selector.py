"""Train a T1 final selector over already-refined candidates."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


BLOCKER_RANKS = ("A", "K", "Q", "J", "T", "2")

DEFAULT_FEATURES = [
    "refined_score",
    "model_score",
    "model_raw_score",
    "refined_minus_model",
    "refined_plus_model",
    "neg_model_rank",
    "inv_model_rank",
    "neg_refined_rank",
    "inv_refined_rank",
    "refined_rank_le_1",
    "refined_rank_le_3",
    "refined_rank_le_5",
    "predicted_bust",
    "predicted_safe",
    "predicted_fl",
    "candidate_fl_score",
    "predicted_qq",
    "predicted_kk",
    "predicted_aa",
    "predicted_trips",
    "premium_fl_sum",
    "samples",
    "log_samples",
    "sync_selector_score",
    "refined_x_safe",
    "refined_x_fl",
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


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def group_margin(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return float(rows[0].get("teacher_margin", 0.0) or 0.0)


def group_rows(
    paths: list[Path],
    *,
    require_teacher_refined: bool,
    min_teacher_margin: float = 0.0,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path_index, path in enumerate(paths):
        for row in iter_jsonl(path):
            if int(row.get("turn", -1)) != 1 or not bool(row.get("is_refined")):
                continue
            grouped[f"{path_index}:{int(row.get('line', 0))}"].append(row)
    out = {}
    for key, rows in grouped.items():
        if group_margin(rows) < float(min_teacher_margin):
            continue
        has_teacher = any(bool(row.get("is_teacher_best")) for row in rows)
        if has_teacher or not require_teacher_refined:
            out[key] = rows
    return out


def split_groups(
    groups: dict[str, list[dict[str, Any]]],
    *,
    heldout_fraction: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    if heldout_fraction <= 0.0 or len(groups) < 2:
        return groups, {}
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    heldout_count = min(len(keys) - 1, max(1, int(round(len(keys) * heldout_fraction))))
    heldout_keys = set(keys[:heldout_count])
    train = {key: rows for key, rows in groups.items() if key not in heldout_keys}
    heldout = {key: rows for key, rows in groups.items() if key in heldout_keys}
    return train, heldout


def _rank(row: dict[str, Any], key: str, default: int = 999) -> float:
    try:
        return max(float(row.get(key, default) or default), 1.0)
    except (TypeError, ValueError):
        return float(default)


def _card_rank(card: str) -> str:
    card = str(card)
    if card.startswith("X"):
        return "X"
    return card[:-1]


def _row_cards(board: dict[str, Any], row: str) -> list[str]:
    if row == "middle":
        return [str(card) for card in (board.get("middle", []) or board.get("mid", []) or [])]
    if row == "bottom":
        return [str(card) for card in (board.get("bottom", []) or board.get("bot", []) or [])]
    return [str(card) for card in (board.get(row, []) or [])]


def _placed_cards(row: dict[str, Any], target_row: str) -> list[str]:
    action = row.get("action") or {}
    return [
        str(card)
        for card, placed_row in (action.get("placements") or [])
        if placed_row == target_row
    ]


def _has_kind_with_jokers(cards: list[str], n: int) -> bool:
    jokers = sum(1 for card in cards if _card_rank(card) == "X")
    counts: dict[str, int] = {}
    for card in cards:
        rank = _card_rank(card)
        if rank != "X":
            counts[rank] = counts.get(rank, 0) + 1
    return any(count + jokers >= n for count in counts.values())


def action_shape_features(row: dict[str, Any]) -> dict[str, float]:
    board = row.get("board") or {}
    action = row.get("action") or {}
    discard = str(action.get("discard") or "")
    placed_top = _placed_cards(row, "top")
    placed_middle = _placed_cards(row, "middle")
    placed_bottom = _placed_cards(row, "bottom")
    top_cards = _row_cards(board, "top")
    middle_cards = _row_cards(board, "middle")
    bottom_cards = _row_cards(board, "bottom")

    placed_top_count = float(len(placed_top))
    placed_middle_count = float(len(placed_middle))
    placed_bottom_count = float(len(placed_bottom))
    top_count = float(len(top_cards))
    middle_count = float(len(middle_cards))
    bottom_count = float(len(bottom_cards))
    existing_top_count = max(0.0, top_count - placed_top_count)
    existing_middle_count = max(0.0, middle_count - placed_middle_count)
    existing_bottom_count = max(0.0, bottom_count - placed_bottom_count)
    placed_ranks = [_card_rank(card) for card in placed_top + placed_middle + placed_bottom]
    discard_rank = _card_rank(discard) if discard else ""

    return {
        "placed_top_count": placed_top_count,
        "placed_middle_count": placed_middle_count,
        "placed_bottom_count": placed_bottom_count,
        "final_top_count": top_count,
        "final_middle_count": middle_count,
        "final_bottom_count": bottom_count,
        "existing_top_count": existing_top_count,
        "existing_middle_count": existing_middle_count,
        "existing_bottom_count": existing_bottom_count,
        "bottom_sparse_two_placed": 1.0
        if existing_bottom_count <= 1.0 and placed_bottom_count == 2.0
        else 0.0,
        "middle_sparse_two_placed": 1.0
        if existing_middle_count <= 1.0 and placed_middle_count == 2.0
        else 0.0,
        "top_full_after": 1.0 if top_count >= 3.0 else 0.0,
        "middle_full_after": 1.0 if middle_count >= 5.0 else 0.0,
        "bottom_full_after": 1.0 if bottom_count >= 5.0 else 0.0,
        "final_top_pair": 1.0 if _has_kind_with_jokers(top_cards, 2) else 0.0,
        "final_top_trips": 1.0 if _has_kind_with_jokers(top_cards, 3) else 0.0,
        "final_middle_pair": 1.0 if _has_kind_with_jokers(middle_cards, 2) else 0.0,
        "final_middle_trips": 1.0 if _has_kind_with_jokers(middle_cards, 3) else 0.0,
        "final_bottom_pair": 1.0 if _has_kind_with_jokers(bottom_cards, 2) else 0.0,
        "final_bottom_trips": 1.0 if _has_kind_with_jokers(bottom_cards, 3) else 0.0,
        "placed_top_has_a": 1.0 if "A" in [_card_rank(card) for card in placed_top] else 0.0,
        "placed_top_has_k": 1.0 if "K" in [_card_rank(card) for card in placed_top] else 0.0,
        "placed_top_has_q": 1.0 if "Q" in [_card_rank(card) for card in placed_top] else 0.0,
        "placed_has_joker": 1.0 if "X" in placed_ranks else 0.0,
        "discard_a": 1.0 if discard_rank == "A" else 0.0,
        "discard_k": 1.0 if discard_rank == "K" else 0.0,
        "discard_q": 1.0 if discard_rank == "Q" else 0.0,
        "discard_joker": 1.0 if discard_rank == "X" else 0.0,
    }


def blocker_features(row: dict[str, Any]) -> dict[str, float]:
    board = row.get("board") or {}
    opponent_board = row.get("opponent_board") or {}
    known_discards = [str(card) for card in (row.get("known_discards") or []) if card]
    action = row.get("action") or {}
    discard = str(action.get("discard") or "")

    own_cards = _row_cards(board, "top") + _row_cards(board, "middle") + _row_cards(board, "bottom")
    opp_top = _row_cards(opponent_board, "top")
    opp_middle = _row_cards(opponent_board, "middle")
    opp_bottom = _row_cards(opponent_board, "bottom")
    opp_cards = opp_top + opp_middle + opp_bottom
    dead_cards = list(own_cards) + list(opp_cards) + list(known_discards)
    if discard:
        dead_cards.append(discard)

    def count_rank(cards: list[str], rank: str) -> float:
        return float(sum(1 for card in cards if _card_rank(card) == rank))

    features = {
        "opp_top_count": float(len(opp_top)),
        "opp_middle_count": float(len(opp_middle)),
        "opp_bottom_count": float(len(opp_bottom)),
        "opp_top_pair": 1.0 if _has_kind_with_jokers(opp_top, 2) else 0.0,
        "opp_top_trips": 1.0 if _has_kind_with_jokers(opp_top, 3) else 0.0,
        "opp_middle_pair": 1.0 if _has_kind_with_jokers(opp_middle, 2) else 0.0,
        "opp_middle_trips": 1.0 if _has_kind_with_jokers(opp_middle, 3) else 0.0,
        "opp_bottom_pair": 1.0 if _has_kind_with_jokers(opp_bottom, 2) else 0.0,
        "opp_bottom_trips": 1.0 if _has_kind_with_jokers(opp_bottom, 3) else 0.0,
    }
    for rank in BLOCKER_RANKS:
        key = rank.lower()
        dead = count_rank(dead_cards, rank)
        own = count_rank(own_cards, rank)
        opp = count_rank(opp_cards, rank)
        features[f"dead_{key}"] = dead
        features[f"own_visible_{key}"] = own
        features[f"opp_visible_{key}"] = opp
        features[f"remaining_{key}"] = max(0.0, 4.0 - dead)
    return features


def row_features(row: dict[str, Any]) -> dict[str, float]:
    model_rank = _rank(row, "model_rank")
    refined_rank = _rank(row, "refined_rank")
    refined_score = float(row.get("refined_score", 0.0) or 0.0)
    model_score = float(row.get("model_score", 0.0) or 0.0)
    model_raw_score = float(row.get("model_raw_score", 0.0) or 0.0)
    predicted_bust = float(row.get("predicted_bust", 0.0) or 0.0)
    predicted_fl = float(row.get("predicted_fl", 0.0) or 0.0)
    predicted_qq = float(row.get("predicted_qq", 0.0) or 0.0)
    predicted_kk = float(row.get("predicted_kk", 0.0) or 0.0)
    predicted_aa = float(row.get("predicted_aa", 0.0) or 0.0)
    predicted_trips = float(row.get("predicted_trips", 0.0) or 0.0)
    fl_score = max(predicted_fl, predicted_qq, predicted_kk, predicted_aa, predicted_trips)
    samples = float(row.get("samples", 0.0) or 0.0)
    features = {
        "refined_score": refined_score,
        "model_score": model_score,
        "model_raw_score": model_raw_score,
        "refined_minus_model": refined_score - model_score,
        "refined_plus_model": refined_score + model_score,
        "neg_model_rank": -model_rank,
        "inv_model_rank": 1.0 / model_rank,
        "neg_refined_rank": -refined_rank,
        "inv_refined_rank": 1.0 / refined_rank,
        "refined_rank_le_1": 1.0 if refined_rank <= 1.0 else 0.0,
        "refined_rank_le_3": 1.0 if refined_rank <= 3.0 else 0.0,
        "refined_rank_le_5": 1.0 if refined_rank <= 5.0 else 0.0,
        "predicted_bust": predicted_bust,
        "predicted_safe": 1.0 - predicted_bust,
        "predicted_fl": predicted_fl,
        "candidate_fl_score": fl_score,
        "predicted_qq": predicted_qq,
        "predicted_kk": predicted_kk,
        "predicted_aa": predicted_aa,
        "predicted_trips": predicted_trips,
        "premium_fl_sum": predicted_kk + predicted_aa + predicted_trips,
        "samples": samples,
        "log_samples": 0.0 if samples <= 0.0 else math.log1p(samples),
        "sync_selector_score": float(row.get("sync_selector_score", 0.0) or 0.0),
        "refined_x_safe": refined_score * (1.0 - predicted_bust),
        "refined_x_fl": refined_score * fl_score,
    }
    features.update(action_shape_features(row))
    features.update(blocker_features(row))
    return features


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
        vectors = [[row_features(row).get(name, 0.0) for name in features] for row in rows]
        label = next((idx for idx, row in enumerate(rows) if bool(row.get("is_teacher_best"))), 0)
        best_score = float(rows[0].get("teacher_best_score", 0.0) or 0.0)
        regrets = [
            max(0.0, best_score - float(row.get("teacher_score", best_score) or best_score))
            for row in rows
        ]
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


def train(groups: dict[str, list[dict[str, Any]]], features: list[str], args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tensors, labels, regret_tensors, means, scales = vectorize(groups, features)
    weights = torch.zeros(len(features), dtype=torch.float32, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights, intercept], lr=args.lr)
    order = list(range(len(tensors)))
    last_loss = 0.0
    for _epoch in range(max(1, args.epochs)):
        random.shuffle(order)
        total = torch.zeros((), dtype=torch.float32)
        optimizer.zero_grad()
        for idx in order:
            scores = tensors[idx].mv(weights) + intercept
            ce_loss = torch.logsumexp(scores, dim=0) - scores[int(labels[idx])]
            regret_loss = torch.zeros((), dtype=torch.float32)
            if float(args.regret_weight) > 0.0:
                regrets = regret_tensors[idx]
                if float(args.regret_cap) > 0.0:
                    regrets = torch.clamp(regrets, max=float(args.regret_cap))
                probs = torch.softmax(scores, dim=0)
                regret_loss = (probs * regrets).sum() / max(float(args.regret_scale), 1e-9)
            total = total + ce_loss + float(args.regret_weight) * regret_loss
        loss = total / max(len(tensors), 1) + float(args.l2) * weights.pow(2).mean()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())
    return {
        "name": Path(args.output).stem,
        "kind": "t1_final_linear_selector",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "weights": [float(value) for value in weights.detach().tolist()],
        "intercept": float(intercept.detach().item()),
        "means": means,
        "scales": scales,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "l2": float(args.l2),
        "regret_weight": float(args.regret_weight),
        "regret_cap": float(args.regret_cap),
        "regret_scale": float(args.regret_scale),
        "seed": int(args.seed),
        "final_loss": last_loss,
    }


def selector_score(row: dict[str, Any], selector: dict[str, Any]) -> float:
    features = row_features(row)
    total = float(selector.get("intercept", 0.0))
    for name, weight in zip(selector["features"], selector["weights"]):
        raw = float(features.get(name, 0.0))
        raw = (raw - float(selector["means"].get(name, 0.0))) / max(float(selector["scales"].get(name, 1.0)), 1e-9)
        total += float(weight) * raw
    return total


def teacher_best_score(group: list[dict[str, Any]]) -> float:
    if not group:
        return 0.0
    return float(group[0].get("teacher_best_score", 0.0) or 0.0)


def evaluate_policy(groups: dict[str, list[dict[str, Any]]], selector: dict[str, Any] | None) -> dict[str, Any]:
    policies = {
        "refined_score": lambda rows: max(rows, key=lambda row: (float(row.get("refined_score", float("-inf"))), float(row.get("model_score", float("-inf"))))),
        "model_score": lambda rows: max(rows, key=lambda row: float(row.get("model_score", float("-inf")))),
    }
    if selector is not None:
        policies["selector"] = lambda rows: max(rows, key=lambda row: selector_score(row, selector))
    report = {}
    for name, picker in policies.items():
        hits = 0
        regret = 0.0
        max_regret = 0.0
        for rows in groups.values():
            row = picker(rows)
            if bool(row.get("is_teacher_best")):
                hits += 1
            row_score = row.get("teacher_score")
            row_regret = 0.0 if row_score is None else max(0.0, teacher_best_score(rows) - float(row_score))
            regret += row_regret
            max_regret = max(max_regret, row_regret)
        denom = max(len(groups), 1)
        report[name] = {
            "decisions": len(groups),
            "top1": hits / denom,
            "hits": hits,
            "avg_regret": regret / denom,
            "max_regret": max_regret,
        }
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a T1 final selector")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--eval", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--regret-weight", type=float, default=0.0)
    parser.add_argument("--regret-cap", type=float, default=0.0)
    parser.add_argument("--regret-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-teacher-margin", type=float, default=0.0)
    parser.add_argument("--heldout-fraction", type=float, default=0.0)
    parser.add_argument("--heldout-seed", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    features = [part.strip() for part in args.features.split(",") if part.strip()]
    train_paths = [Path(path) for path in args.train]
    all_train_groups = group_rows(
        train_paths,
        require_teacher_refined=True,
        min_teacher_margin=args.min_teacher_margin,
    )
    train_groups, heldout_groups = split_groups(
        all_train_groups,
        heldout_fraction=args.heldout_fraction,
        seed=args.heldout_seed,
    )
    if not train_groups:
        raise SystemExit("No refined T1 groups with teacher best found in training rows")
    selector = train(train_groups, features, args)
    selector["train_inputs"] = [str(path) for path in train_paths]
    selector["min_teacher_margin"] = float(args.min_teacher_margin)
    selector["heldout_fraction"] = float(args.heldout_fraction)
    selector["heldout_seed"] = int(args.heldout_seed)
    selector["group_counts"] = {
        "train_total_after_margin": len(all_train_groups),
        "train_fit": len(train_groups),
        "heldout": len(heldout_groups),
    }
    selector["metrics"] = {
        "train_fit": evaluate_policy(train_groups, selector),
    }
    if heldout_groups:
        selector["metrics"]["heldout"] = evaluate_policy(heldout_groups, selector)
    for eval_path_raw in args.eval:
        eval_path = Path(eval_path_raw)
        eval_groups = group_rows(
            [eval_path],
            require_teacher_refined=False,
            min_teacher_margin=args.min_teacher_margin,
        )
        selector["metrics"][str(eval_path)] = evaluate_policy(eval_groups, selector)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selector, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(selector, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
