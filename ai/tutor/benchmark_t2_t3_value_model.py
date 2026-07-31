"""Benchmark T2 evaluation through a fast T3 action-value model.

This is a local harness for the proposed runtime path:

    T2 action -> T3 draw samples -> model-score all legal T3 actions -> average

It compares the resulting T2 candidate ranking with a capped/exact T2 JSONL
when one is provided.  The script is intentionally diagnostic; it does not
replace the existing Rust exact oracle.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.action_space import Action, get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board, Observation, encode_state
from ai.models.action_value_reranker import ActionValueReranker, BlendedActionValueReranker
from ai.training.action_feature_encoding import adapt_np_state_with_action
from ai.tutor.exact_late import CardNormalizer, apply_action, board_to_dict, normalize_board
from ai.tutor.run_t2_exact_oracle import action_key, candidate_action, candidate_score


ROW_ALIASES = {"top": "top", "mid": "middle", "middle": "middle", "bot": "bottom", "bottom": "bottom"}


def row_name(row: Any) -> str:
    return ROW_ALIASES.get(str(row), str(row))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def board_cards(board: Board) -> set[str]:
    return set(board.top + board.middle + board.bottom)


def compact_board_cards(board: dict[str, Any] | None) -> set[str]:
    board = board or {}
    cards: set[str] = set()
    for key in ("top", "middle", "mid", "bottom", "bot"):
        cards.update(str(card) for card in (board.get(key) or []) if str(card))
    return cards


def make_action(raw: dict[str, Any], normalizer: CardNormalizer | None = None) -> Action:
    normalizer = normalizer or CardNormalizer()
    return Action(
        placements=[
            (normalizer.card(str(card)), row_name(row))
            for card, row in (raw.get("placements") or [])
        ],
        discard=(
            None
            if raw.get("discard") in (None, "")
            else normalizer.card(str(raw.get("discard")))
        ),
    )


def source_candidate_key(candidate: dict[str, Any]) -> str:
    return action_key(candidate_action(candidate))


def loose_card_key(card: Any) -> str:
    card = str(card)
    return "X" if card in {"Xj", "JK", "X1", "X2"} or card.startswith("X") else card


def loose_action_key(action: dict[str, Any] | None) -> str:
    action = action or {}
    placements = tuple(
        sorted((loose_card_key(card), row_name(row)) for card, row in (action.get("placements") or []))
    )
    discard = action.get("discard")
    return json.dumps(
        {
            "placements": placements,
            "discard": None if discard is None else loose_card_key(discard),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def legal_t2_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for idx, candidate in enumerate(record.get("candidates") or []):
        raw = candidate_action(candidate)
        if not raw.get("placements"):
            continue
        candidates.append(
            {
                "source_index": idx,
                "action": raw,
                "key": action_key(raw),
                "source_score": candidate_score(candidate),
            }
        )
    return candidates


def known_dead_cards(record: dict[str, Any], hero_board: Board, opponent_board: Board, extra: Iterable[str]) -> list[str]:
    used_visible = board_cards(hero_board) | board_cards(opponent_board)
    dead = []
    seen = set()
    for card in list(record.get("known_discards") or []) + list(record.get("exclude") or []) + list(extra):
        card = str(card)
        if not card or card in used_visible or card in seen:
            continue
        seen.add(card)
        dead.append(card)
    return dead


def normalized_dead_cards(
    record: dict[str, Any],
    hero_board: Board,
    opponent_board: Board,
    extra: Iterable[str],
    normalizer: CardNormalizer,
) -> list[str]:
    raw = list(record.get("known_discards") or []) + list(record.get("exclude") or []) + list(extra)
    normalized = normalizer.cards([str(card) for card in raw if str(card)])
    used_visible = board_cards(hero_board) | board_cards(opponent_board)
    dead: list[str] = []
    seen: set[str] = set()
    for card in normalized:
        if card in used_visible or card in seen:
            continue
        seen.add(card)
        dead.append(card)
    return dead


def remaining_deck(hero_board: Board, opponent_board: Board, dead: Iterable[str]) -> list[str]:
    used = board_cards(hero_board) | board_cards(opponent_board) | {str(card) for card in dead if str(card)}
    return [card for card in ALL_CARDS if card not in used]


def choose_draws(deck: list[str], limit: int, seed: int) -> list[tuple[str, str, str]]:
    all_count = len(deck) * (len(deck) - 1) * (len(deck) - 2) // 6
    if limit <= 0 or limit >= all_count:
        return list(combinations(deck, 3))
    rng = random.Random(seed)
    seen: set[tuple[str, str, str]] = set()
    draws: list[tuple[str, str, str]] = []
    while len(draws) < limit:
        draw = tuple(sorted(rng.sample(deck, 3)))
        if draw in seen:
            continue
        seen.add(draw)
        draws.append(draw)
    return draws


def encode_post_t3_candidates(
    board: Board,
    opponent_board: Board,
    draw: Iterable[str],
    dead_before_t3: list[str],
    *,
    is_btn: bool,
    target_dim: int = 520,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    draw_list = list(draw)
    decision = {
        "turn": 3,
        "board": board_to_dict(board),
        "opponent_board": board_to_dict(opponent_board),
        "dealt": draw_list,
        "known_discards": list(dead_before_t3),
        "exclude": list(dead_before_t3),
        "is_btn": bool(is_btn),
    }
    states: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for action in get_turn_actions(draw_list, board):
        post_board = apply_action(board, action)
        known = list(dead_before_t3)
        if action.discard and action.discard not in known:
            known.append(action.discard)
        obs = Observation(
            board_self=post_board,
            board_opponent=opponent_board,
            dealt_cards=[],
            known_discards_self=known,
            turn=3,
            is_btn=is_btn,
            is_fl=False,
            opp_is_fl=False,
            chips_self=200,
            chips_opponent=200,
        )
        action_payload = {"placements": action.placements, "discard": action.discard}
        state = adapt_np_state_with_action(
            np.asarray(encode_state(obs), dtype=np.float32),
            int(target_dim),
            decision,
            action_payload,
        )
        states.append(np.asarray(state, dtype=np.float32))
        rows.append({"action": action_payload, "board": board_to_dict(post_board)})
    return states, rows


def predict_scores(
    model: ActionValueReranker,
    states: list[np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    bust: list[np.ndarray] = []
    fl: list[np.ndarray] = []
    if not states:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    turn = None
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = torch.from_numpy(np.stack(states[start : start + batch_size])).to(device)
            turn = torch.full((batch.shape[0],), 3, dtype=torch.long, device=device)
            out = model.predict_components(batch, turn=turn)
            scores.append(out["score"].detach().cpu().numpy())
            bust.append(out["bust_prob"].detach().cpu().numpy())
            fl.append(out["fl_prob"].detach().cpu().numpy())
    return np.concatenate(scores), np.concatenate(bust), np.concatenate(fl)


def adjusted_t3_priority_score(
    scores: np.ndarray,
    bust: np.ndarray,
    fl: np.ndarray,
    *,
    bust_weight: float,
    fl_any_weight: float,
) -> np.ndarray:
    priority = np.asarray(scores, dtype=np.float32).copy()
    priority -= float(bust_weight) * np.asarray(bust, dtype=np.float32)
    priority += float(fl_any_weight) * np.asarray(fl, dtype=np.float32)
    return priority


def evaluate_t2_candidate(
    record: dict[str, Any],
    candidate: dict[str, Any],
    model: ActionValueReranker,
    *,
    device: torch.device,
    draw_limit: int,
    seed: int,
    batch_size: int,
    t3_rank_bust_weight: float,
    t3_rank_fl_any_weight: float,
) -> dict[str, Any]:
    normalizer = CardNormalizer()
    hero_board = normalize_board(record.get("board") or {}, normalizer)
    opponent_board = normalize_board(record.get("opponent_board") or record.get("board_opponent") or {}, normalizer)
    action = make_action(candidate["action"], normalizer)
    board_after_t2 = apply_action(hero_board, action)
    dead = normalized_dead_cards(
        record,
        board_after_t2,
        opponent_board,
        [action.discard] if action.discard else [],
        normalizer,
    )
    deck = remaining_deck(board_after_t2, opponent_board, dead)
    draws = choose_draws(deck, draw_limit, seed)

    all_states: list[np.ndarray] = []
    row_ranges: list[tuple[int, int]] = []
    for draw in draws:
        start = len(all_states)
        states, _rows = encode_post_t3_candidates(
            board_after_t2,
            opponent_board,
            draw,
            dead,
            is_btn=bool(record.get("is_btn", False)),
            target_dim=int(getattr(model, "input_dim", 520)),
        )
        all_states.extend(states)
        row_ranges.append((start, len(all_states)))

    scores, bust, fl = predict_scores(model, all_states, device=device, batch_size=batch_size)
    priority_scores = adjusted_t3_priority_score(
        scores,
        bust,
        fl,
        bust_weight=t3_rank_bust_weight,
        fl_any_weight=t3_rank_fl_any_weight,
    )
    draw_best_scores: list[float] = []
    draw_best_priority_scores: list[float] = []
    draw_best_bust: list[float] = []
    draw_best_fl: list[float] = []
    draw_raw_top1_matches: list[bool] = []
    for start, end in row_ranges:
        if end <= start:
            continue
        local = priority_scores[start:end]
        best_offset = int(np.argmax(local))
        best_idx = start + best_offset
        raw_best_idx = start + int(np.argmax(scores[start:end]))
        draw_best_scores.append(float(scores[best_idx]))
        draw_best_priority_scores.append(float(priority_scores[best_idx]))
        draw_best_bust.append(float(bust[best_idx]))
        draw_best_fl.append(float(fl[best_idx]))
        draw_raw_top1_matches.append(best_idx == raw_best_idx)

    return {
        "action": candidate["action"],
        "action_key": candidate["key"],
        "source_index": candidate["source_index"],
        "source_score": candidate["source_score"],
        "model_t3_value_score": float(np.mean(draw_best_scores)) if draw_best_scores else float("-inf"),
        "model_t3_priority_score": float(np.mean(draw_best_priority_scores)) if draw_best_priority_scores else float("-inf"),
        "model_t3_value_bust": float(np.mean(draw_best_bust)) if draw_best_bust else 0.0,
        "model_t3_value_fl": float(np.mean(draw_best_fl)) if draw_best_fl else 0.0,
        "t3_raw_top1_match_rate": float(np.mean(draw_raw_top1_matches)) if draw_raw_top1_matches else 0.0,
        "draws": len(draws),
        "t3_states_scored": len(all_states),
    }


def exact_reference_by_record(exact_path: Path | None) -> dict[int, dict[str, Any]]:
    if exact_path is None:
        return {}
    return {int(row.get("record_index", idx)): row for idx, row in enumerate(load_jsonl(exact_path))}


def add_exact_comparison(row: dict[str, Any], exact_record: dict[str, Any] | None) -> None:
    if not exact_record:
        return
    exact_candidates = list(exact_record.get("candidates") or [])
    exact_by_key = {action_key(candidate_action(candidate)): candidate for candidate in exact_candidates}
    exact_by_loose_key = {
        loose_action_key(candidate_action(candidate)): candidate for candidate in exact_candidates
    }
    exact_ranked = sorted(exact_candidates, key=candidate_score, reverse=True)
    for candidate in row["candidates"]:
        exact = exact_by_key.get(candidate["action_key"])
        if exact is None:
            exact = exact_by_loose_key.get(loose_action_key(candidate["action"]))
        if exact is not None:
            candidate["exact_score"] = candidate_score(exact)
            candidate["exact_rank"] = next(
                (
                    idx
                    for idx, exact_candidate in enumerate(exact_ranked, start=1)
                    if action_key(candidate_action(exact_candidate)) == action_key(candidate_action(exact))
                ),
                None,
            )

    exact_best = exact_record.get("best") or (exact_candidates[0] if exact_candidates else None)
    exact_best_key = action_key(candidate_action(exact_best)) if exact_best else None
    exact_best_loose_key = loose_action_key(candidate_action(exact_best)) if exact_best else None
    model_order = row["candidates"]
    exact_scores = [candidate.get("exact_score") for candidate in model_order if candidate.get("exact_score") is not None]
    if not exact_best_key or not exact_scores:
        return

    model_best = model_order[0]
    exact_best_score = candidate_score(exact_best)
    model_best_exact = model_best.get("exact_score")
    exact_top1_model_rank = next(
        (
            idx
            for idx, candidate in enumerate(model_order, start=1)
            if candidate["action_key"] == exact_best_key
            or loose_action_key(candidate["action"]) == exact_best_loose_key
        ),
        None,
    )
    row["exact_comparison"] = {
        "exact_best_action_key": exact_best_key,
        "model_best_action_key": model_best["action_key"],
        "exact_top1_model_rank": exact_top1_model_rank,
        "model_top1_exact_regret": None if model_best_exact is None else float(exact_best_score - model_best_exact),
        "topk_recall": {
            str(k): bool(exact_top1_model_rank is not None and exact_top1_model_rank <= k)
            for k in (1, 3, 5, 10, 20)
        },
    }


def add_source_comparison(row: dict[str, Any]) -> None:
    candidates = list(row.get("candidates") or [])
    if not candidates:
        return
    source_ranked = sorted(candidates, key=lambda item: float(item.get("source_score", float("-inf"))), reverse=True)
    source_best = source_ranked[0]
    source_best_key = str(source_best.get("action_key") or "")
    model_best = candidates[0]
    source_top1_model_rank = next(
        (
            rank
            for rank, candidate in enumerate(candidates, start=1)
            if str(candidate.get("action_key") or "") == source_best_key
        ),
        None,
    )
    source_best_score = float(source_best.get("source_score", float("-inf")))
    model_best_source_score = float(model_best.get("source_score", float("-inf")))
    row["source_comparison"] = {
        "source_best_action_key": source_best_key,
        "model_best_action_key": str(model_best.get("action_key") or ""),
        "source_top1_model_rank": source_top1_model_rank,
        "model_top1_source_regret": max(0.0, source_best_score - model_best_source_score),
        "topk_recall": {
            str(k): bool(source_top1_model_rank is not None and source_top1_model_rank <= k)
            for k in (1, 3, 5, 10, 20)
        },
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(input_path)
    if args.limit > 0:
        records = records[: args.limit]
    exact_refs = exact_reference_by_record(Path(args.exact) if args.exact else None)

    device = torch.device(args.device)
    if args.model_b:
        model = BlendedActionValueReranker.from_checkpoints(
            args.model,
            args.model_b,
            model_b_weight=args.model_b_weight,
            map_location=device,
        ).to(device)
        model_report = {
            "kind": "blend",
            "model_a": str(args.model),
            "model_b": str(args.model_b),
            "model_a_weight": 1.0 - float(args.model_b_weight),
            "model_b_weight": float(args.model_b_weight),
        }
    else:
        model = ActionValueReranker.from_checkpoint(args.model, map_location=device).to(device)
        model_report = {"kind": "single", "model": str(args.model)}
    model.eval()

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    total_states = 0
    for record_index, record in enumerate(records):
        candidates = legal_t2_candidates(record)
        if args.max_candidates > 0:
            candidates = sorted(candidates, key=lambda item: item["source_score"], reverse=True)[: args.max_candidates]
        cand_rows = []
        record_start = time.perf_counter()
        for cand_index, candidate in enumerate(candidates):
            cand_rows.append(
                evaluate_t2_candidate(
                    record,
                    candidate,
                    model,
                    device=device,
                    draw_limit=args.draw_limit,
                    seed=args.seed + record_index * 1009 + cand_index * 37,
                    batch_size=args.batch_size,
                    t3_rank_bust_weight=args.t3_rank_bust_weight,
                    t3_rank_fl_any_weight=args.t3_rank_fl_any_weight,
                )
            )
        sort_key = "model_t3_priority_score" if args.t2_sort_score == "priority" else "model_t3_value_score"
        cand_rows.sort(key=lambda item: item[sort_key], reverse=True)
        for rank, candidate in enumerate(cand_rows, start=1):
            candidate["model_t3_value_rank"] = rank
            candidate["model_t3_rank_score_key"] = sort_key
            total_states += int(candidate.get("t3_states_scored") or 0)
        row = {
            "record_index": record_index,
            "source_line": record.get("source_line"),
            "turn": int(record.get("turn", -1)),
            "position": record.get("position"),
            "board": record.get("board"),
            "opponent_board": record.get("opponent_board") or record.get("board_opponent"),
            "dealt": record.get("dealt"),
            "draw_limit": args.draw_limit,
            "candidate_count": len(cand_rows),
            "elapsed_ms": (time.perf_counter() - record_start) * 1000.0,
            "candidates": cand_rows,
        }
        add_source_comparison(row)
        add_exact_comparison(row, exact_refs.get(record_index))
        rows.append(row)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    summary = {
        "input": str(input_path),
        "model": str(args.model),
        "model_report": model_report,
        "exact": args.exact,
        "records": len(rows),
        "draw_limit": args.draw_limit,
        "max_candidates": args.max_candidates,
        "batch_size": args.batch_size,
        "t2_sort_score": args.t2_sort_score,
        "t3_rank_bust_weight": args.t3_rank_bust_weight,
        "t3_rank_fl_any_weight": args.t3_rank_fl_any_weight,
        "elapsed_ms": elapsed_ms,
        "avg_record_elapsed_ms": elapsed_ms / max(len(rows), 1),
        "total_t3_states_scored": total_states,
        "t3_states_per_second": total_states / max(elapsed_ms / 1000.0, 1e-9),
    }
    source_comparable = [row for row in rows if row.get("source_comparison")]
    if source_comparable:
        summary["source_top1_model_rank_mean"] = sum(
            float(row["source_comparison"]["source_top1_model_rank"] or 9999)
            for row in source_comparable
        ) / len(source_comparable)
        summary["model_top1_source_regret_mean"] = sum(
            float(row["source_comparison"]["model_top1_source_regret"] or 0.0)
            for row in source_comparable
        ) / len(source_comparable)
        summary["source_topk_recall"] = {
            str(k): sum(1 for row in source_comparable if row["source_comparison"]["topk_recall"][str(k)])
            / len(source_comparable)
            for k in (1, 3, 5, 10, 20)
        }
    comparable = [row for row in rows if row.get("exact_comparison")]
    if comparable:
        summary["exact_top1_model_rank_mean"] = sum(
            float(row["exact_comparison"]["exact_top1_model_rank"] or 9999) for row in comparable
        ) / len(comparable)
        summary["model_top1_exact_regret_mean"] = sum(
            float(row["exact_comparison"]["model_top1_exact_regret"] or 0.0) for row in comparable
        ) / len(comparable)
        summary["topk_recall"] = {
            str(k): sum(1 for row in comparable if row["exact_comparison"]["topk_recall"][str(k)]) / len(comparable)
            for k in (1, 3, 5, 10, 20)
        }

    rows_path = output_dir / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "summary.md", summary)
    return summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# T2 via T3 Value Model Benchmark",
        "",
        f"- input: `{Path(summary['input']).name}`",
        f"- model: `{summary['model']}`",
        f"- exact reference: `{summary.get('exact')}`",
        f"- records: {summary['records']}",
        f"- draw limit per T2 candidate: {summary['draw_limit']}",
        f"- max candidates: {summary['max_candidates']}",
        f"- T2 sort score: `{summary.get('t2_sort_score', 'value')}`",
        f"- T3 rank weights: bust={summary.get('t3_rank_bust_weight', 0.0)}, FL={summary.get('t3_rank_fl_any_weight', 0.0)}",
        f"- elapsed: {summary['elapsed_ms']:.1f} ms",
        f"- avg record elapsed: {summary['avg_record_elapsed_ms']:.1f} ms",
        f"- T3 states scored: {summary['total_t3_states_scored']:,}",
        f"- T3 states/sec: {summary['t3_states_per_second']:.0f}",
    ]
    if "topk_recall" in summary:
        lines.extend(["", "## Exact Reference", ""])
        lines.append(f"- exact Top1 model-rank mean: {summary['exact_top1_model_rank_mean']:.2f}")
        lines.append(f"- model Top1 exact-regret mean: {summary['model_top1_exact_regret_mean']:.3f}")
        lines.append("")
        lines.append("| K | recall |")
        lines.append("|---:|---:|")
        for key, value in summary["topk_recall"].items():
            lines.append(f"| {key} | {value:.1%} |")
    if "source_topk_recall" in summary:
        lines.extend(["", "## Source Teacher Reference", ""])
        lines.append(f"- source Top1 model-rank mean: {summary['source_top1_model_rank_mean']:.2f}")
        lines.append(f"- model Top1 source-regret mean: {summary['model_top1_source_regret_mean']:.3f}")
        lines.append("")
        lines.append("| K | recall |")
        lines.append("|---:|---:|")
        for key, value in summary["source_topk_recall"].items():
            lines.append(f"| {key} | {value:.1%} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark T2 evaluation via a T3 value model")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-b", default="", help="Optional second T3 checkpoint for inference-time blend")
    parser.add_argument("--model-b-weight", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exact", default="")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--draw-limit", type=int, default=30)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--t3-rank-bust-weight", type=float, default=0.0)
    parser.add_argument("--t3-rank-fl-any-weight", type=float, default=0.0)
    parser.add_argument("--t2-sort-score", choices=["value", "priority"], default="value")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(evaluate(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
