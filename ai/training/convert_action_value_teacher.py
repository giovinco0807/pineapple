"""Convert MC teacher JSONL to candidate-level action-value data.

Unlike the policy converter, this script creates one sample per candidate
placement.  The input features are the encoded board after the candidate action;
the targets are the teacher's rollout EV, bust rate, FL rate, and FL type mix.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.action_space import Action
from ai.engine.encoding import Board, Observation, encode_state
from ai.engine.turn_order import normalize_position, validate_decision_board_counts
from ai.training.action_feature_encoding import (
    ACTION_FEATURE_DIM,
    BOARD_CONTEXT_FEATURE_DIM,
    FL_OPPORTUNITY_FEATURE_DIM,
    adapt_np_state,
    adapt_np_state_with_action,
)
from ai.training.convert_mc_teacher import (
    action_to_index_t0,
    action_to_index_t1plus,
    get_candidate_bust,
    get_candidate_ev,
    get_candidate_fl,
    normalize_joker_refs,
)


FL_TYPE_KEYS = ("qq", "kk", "aa", "trips")
ROUTE_TAGS = {
    "fl_positive": 1,
    "t0t1_high_top": 2,
    "fl_shape_but_foul": 4,
    "teacher_best": 8,
    "decision_high_gap": 16,
    "t0_safe_fl_route": 32,
    "t0_risky_fl_route": 64,
    "t0_dead_high_route": 128,
    "trips_fl_positive": 256,
}


def _adapt_np_state(state: np.ndarray, target_dim: int) -> np.ndarray:
    return adapt_np_state(state, target_dim)


def _row_name(pos: str) -> str:
    if pos == "mid":
        return "middle"
    if pos == "bot":
        return "bottom"
    return pos


def board_from_dict(board: dict) -> Board:
    return Board(
        top=list(board.get("top", [])),
        middle=list(board.get("middle", [])) + list(board.get("mid", [])),
        bottom=list(board.get("bottom", [])) + list(board.get("bot", [])),
    )


def build_board(record: dict) -> Board:
    return board_from_dict(record.get("board", {}))


def opponent_board_for_record(record: dict) -> Board:
    board = record.get("opponent_board") or record.get("board_opponent") or {}
    return board_from_dict(board)


def canonical_record_position(record: dict, *, validate_counts: bool = True) -> str:
    """Resolve one teacher record to the canonical BB/BTN role."""
    raw_position = record.get("position")
    if raw_position in (None, ""):
        raw_position = record.get("player_position")
    has_position = raw_position not in (None, "")
    has_is_btn = "is_btn" in record and record.get("is_btn") is not None
    if not has_position and not has_is_btn:
        raise ValueError("HU teacher record requires explicit position or is_btn")
    position = normalize_position(
        raw_position if has_position else None,
        is_btn=record["is_btn"] if has_is_btn else None,
    )
    if validate_counts:
        validate_decision_board_counts(
            int(record.get("turn", 0)),
            position,
            len(build_board(record).all_cards()),
            len(opponent_board_for_record(record).all_cards()),
        )
    return position


def _real_discard(discard: str | None) -> str | None:
    if discard in (None, "", "Xj"):
        return None
    return discard


def apply_candidate(board: Board, candidate: dict, dealt_cards: Sequence[str]) -> tuple[Board, str | None]:
    placements = [(card, _row_name(pos)) for card, pos in candidate.get("placements", [])]
    placements, discard = normalize_joker_refs(
        placements,
        _real_discard(candidate.get("discard")),
        list(dealt_cards),
    )

    next_board = board.copy()
    for card, pos in placements:
        getattr(next_board, _row_name(pos)).append(card)
    return next_board, _real_discard(discard)


def candidate_action_cards(candidate: dict, dealt_cards: Sequence[str]) -> Counter:
    placements = [(card, _row_name(pos)) for card, pos in candidate.get("placements", [])]
    placements, discard = normalize_joker_refs(
        placements,
        candidate.get("discard"),
        list(dealt_cards),
    )
    cards = [str(card) for card, _pos in placements]
    discard = _real_discard(discard)
    if discard is not None:
        cards.append(str(discard))
    return Counter(cards)


def candidate_matches_dealt(record: dict, candidate: dict) -> bool:
    dealt_cards = [str(card) for card in (record.get("dealt") or [])]
    if not dealt_cards:
        return True
    return candidate_action_cards(candidate, dealt_cards) == Counter(dealt_cards)


def filter_valid_candidate_indices(record: dict, candidates: Sequence[dict], indices: Sequence[int]) -> tuple[list[int], int]:
    valid = []
    invalid = 0
    for idx in indices:
        if idx < 0 or idx >= len(candidates):
            invalid += 1
            continue
        if candidate_matches_dealt(record, candidates[idx]):
            valid.append(int(idx))
        else:
            invalid += 1
    return valid, invalid


def valid_candidate_indices(record: dict, candidates: Sequence[dict]) -> list[int]:
    return [i for i, candidate in enumerate(candidates) if candidate_matches_dealt(record, candidate)]


def post_action_observation(record: dict, candidate: dict) -> Observation:
    position = canonical_record_position(record)
    dealt = list(record.get("dealt", []))
    board, discard = apply_candidate(build_board(record), candidate, dealt)
    opponent_board = opponent_board_for_record(record)
    encode_opponent_shape = not (int(record.get("turn", 0)) == 0 and position == "bb")
    encoded_opponent_board = opponent_board if encode_opponent_shape else Board()

    # New teacher records preserve opponent row placement.  Older records only
    # have ``exclude`` as opponent cards + self discards, so keep the old
    # discard-like fallback for compatibility.
    unavailable = list(record.get("known_discards", []))
    seen = set(board.all_cards())
    seen.update(encoded_opponent_board.all_cards())
    for card in record.get("exclude", []):
        if card not in seen and card not in unavailable:
            unavailable.append(card)
    if discard and discard not in unavailable:
        unavailable.append(discard)

    return Observation(
        board_self=board,
        board_opponent=encoded_opponent_board,
        dealt_cards=[],
        known_discards_self=unavailable,
        turn=int(record.get("turn", 0)),
        is_btn=position == "btn",
        is_fl=False,
        opp_is_fl=False,
        chips_self=200,
        chips_opponent=200,
    )


def select_candidate_indices(n: int, limit: int) -> List[int]:
    if limit <= 0 or n <= limit:
        return list(range(n))
    top_n = max(1, min(n, limit // 2))
    selected = list(range(top_n))
    remaining = limit - len(selected)
    if remaining > 0 and top_n < n:
        spread = np.linspace(top_n, n - 1, num=remaining, dtype=np.int64).tolist()
        for idx in spread:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
    return selected[:limit]


def _rank(card: str) -> str:
    if str(card).startswith("X"):
        return "X"
    return str(card)[0] if card else ""


def _has_fl_ready_top(board: Board) -> bool:
    ranks = [_rank(c) for c in board.top]
    jokers = ranks.count("X")
    counts = Counter(r for r in ranks if r != "X")
    for r in ("Q", "K", "A"):
        if counts.get(r, 0) + jokers >= 2:
            return True
    for r, count in counts.items():
        if count + jokers >= 3:
            return True
    return jokers >= 2


def _has_high_top_route(board: Board) -> bool:
    return any(_rank(c) in ("Q", "K", "A", "X") for c in board.top)


def _route_tag(
    turn: int,
    board: Board,
    fl_rate: float,
    bust_rate: float,
    rank: int,
    decision_gap: float,
    fl_vec: np.ndarray,
    args: argparse.Namespace,
) -> int:
    tag = 0
    high_top = _has_high_top_route(board)
    if fl_rate > 0.0:
        tag |= ROUTE_TAGS["fl_positive"]
    if turn <= 1 and high_top:
        tag |= ROUTE_TAGS["t0t1_high_top"]
    if (_has_fl_ready_top(board) or high_top) and bust_rate >= 0.35:
        tag |= ROUTE_TAGS["fl_shape_but_foul"]
    if rank == 0:
        tag |= ROUTE_TAGS["teacher_best"]
    if decision_gap >= 2.0:
        tag |= ROUTE_TAGS["decision_high_gap"]
    if turn == 0 and high_top:
        if fl_rate >= args.t0_safe_fl_min and bust_rate <= args.t0_safe_bust_max:
            tag |= ROUTE_TAGS["t0_safe_fl_route"]
        if fl_rate >= args.t0_risky_fl_min and bust_rate >= args.t0_risky_bust_min:
            tag |= ROUTE_TAGS["t0_risky_fl_route"]
        if fl_rate <= args.t0_dead_fl_max and bust_rate >= args.t0_dead_bust_min:
            tag |= ROUTE_TAGS["t0_dead_high_route"]
    if len(fl_vec) > 3 and float(fl_vec[3]) > 0.0:
        tag |= ROUTE_TAGS["trips_fl_positive"]
    return tag


def _sample_weight(tag: int, fl_rate: float, bust_rate: float, candidate_gap: float, args: argparse.Namespace) -> float:
    weight = 1.0
    if tag & ROUTE_TAGS["fl_positive"]:
        weight += args.fl_positive_weight
    if tag & ROUTE_TAGS["t0t1_high_top"]:
        weight += args.high_route_weight
    if tag & ROUTE_TAGS["fl_shape_but_foul"]:
        weight += args.fl_shape_but_foul_weight
    if tag & ROUTE_TAGS["teacher_best"]:
        weight += args.teacher_best_weight
    if tag & ROUTE_TAGS["t0_safe_fl_route"]:
        weight += args.t0_safe_route_weight
    if tag & ROUTE_TAGS["t0_risky_fl_route"]:
        weight += args.t0_risky_route_weight
    if tag & ROUTE_TAGS["t0_dead_high_route"]:
        weight += args.t0_dead_route_weight
    if tag & ROUTE_TAGS["trips_fl_positive"]:
        weight += args.trips_fl_weight
    # Keep high-gap negatives visible to the ranker, but cap their influence.
    weight += min(max(candidate_gap, 0.0) / 10.0, 2.0) * args.gap_weight
    if fl_rate >= 0.25 and bust_rate <= 0.35:
        weight += args.clean_fl_weight
    return float(min(weight, args.max_sample_weight))


def _score_adjustment(tag: int, args: argparse.Namespace) -> float:
    adjustment = 0.0
    if tag & ROUTE_TAGS["t0_safe_fl_route"]:
        adjustment += args.t0_safe_route_score_bonus
    if tag & ROUTE_TAGS["t0_risky_fl_route"]:
        adjustment -= args.t0_risky_route_score_penalty
    if tag & ROUTE_TAGS["t0_dead_high_route"]:
        adjustment -= args.t0_dead_route_score_penalty
    if tag & ROUTE_TAGS["trips_fl_positive"]:
        adjustment += args.trips_fl_score_bonus
    return float(adjustment)


def candidate_limit_for_turn(turn: int, args: argparse.Namespace) -> int:
    if turn == 0:
        return args.t0_max_candidates
    return args.regular_max_candidates


def count_samples(input_path: Path, include_turns: set[int], args: argparse.Namespace) -> tuple[int, dict]:
    total = 0
    records = 0
    invalid_records = 0
    invalid_selected_candidates = 0
    turns: dict[int, int] = {}
    with input_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            d = json.loads(line)
            turn = int(d.get("turn", -1))
            candidates = d.get("candidates") or []
            if turn not in include_turns or not candidates:
                continue
            canonical_record_position(d)
            if turn == 0 and args.keep_t0_fl_shaped and args.t0_max_candidates > 0:
                chosen = select_candidate_indices(len(candidates), args.t0_max_candidates)
                board = build_board(d)
                for i, cand in enumerate(candidates):
                    if i in chosen:
                        continue
                    try:
                        next_board, _discard = apply_candidate(board, cand, d.get("dealt", []))
                    except Exception:
                        continue
                    if _has_high_top_route(next_board) or _has_fl_ready_top(next_board):
                        chosen.append(i)
            else:
                chosen = select_candidate_indices(len(candidates), candidate_limit_for_turn(turn, args))
            chosen, invalid = filter_valid_candidate_indices(d, candidates, sorted(set(chosen)))
            invalid_selected_candidates += invalid
            if not chosen:
                invalid_records += 1
                continue
            total += len(chosen)
            records += 1
            turns[turn] = turns.get(turn, 0) + 1
            if args.limit_records and records >= args.limit_records:
                break
    return total, {
        "records": records,
        "turns": turns,
        "invalid_records": invalid_records,
        "invalid_selected_candidates": invalid_selected_candidates,
    }


def action_index(record: dict, candidate: dict) -> int:
    turn = int(record.get("turn", 0))
    dealt = list(record.get("dealt", []))
    try:
        if turn == 0:
            return int(action_to_index_t0(candidate.get("placements", []), dealt))
        return int(action_to_index_t1plus(candidate, dealt))
    except Exception:
        return -1


def fl_type_vector(candidate: dict, eval_mode: str) -> np.ndarray:
    mc = candidate.get("mc", {}) if eval_mode.startswith("mc") or eval_mode == "t0_ladder" else candidate
    rates = mc.get("fl_type_rates", {}) or {}
    return np.array([float(rates.get(k, 0.0)) for k in FL_TYPE_KEYS], dtype=np.float32)


def convert(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    include_turns = {int(t) for t in args.turns.split(",") if t.strip()}

    print(f"Reading: {input_path}")
    print(f"Output:  {out_dir}")
    print(f"Turns:   {sorted(include_turns)}")
    print(f"State:   {args.state_dim} dims")
    t0_limit = "all" if args.t0_max_candidates <= 0 else str(args.t0_max_candidates)
    reg_limit = "all" if args.regular_max_candidates <= 0 else str(args.regular_max_candidates)
    print(f"Limits:  T0={t0_limit}, regular={reg_limit}, keep_t0_fl_shaped={args.keep_t0_fl_shaped}")

    n_total, count_meta = count_samples(input_path, include_turns, args)
    if n_total <= 0:
        raise SystemExit("No samples matched the requested filters.")

    print(f"Records: {count_meta['records']:,}")
    print(f"Samples: {n_total:,}")

    states = np.lib.format.open_memmap(
        out_dir / "states.npy", mode="w+", dtype=np.float32, shape=(n_total, args.state_dim)
    )
    scores = np.lib.format.open_memmap(out_dir / "scores.npy", mode="w+", dtype=np.float32, shape=(n_total,))
    bust = np.lib.format.open_memmap(out_dir / "bust.npy", mode="w+", dtype=np.float32, shape=(n_total,))
    fl = np.lib.format.open_memmap(out_dir / "fl.npy", mode="w+", dtype=np.float32, shape=(n_total,))
    fl_types = np.lib.format.open_memmap(out_dir / "fl_types.npy", mode="w+", dtype=np.float32, shape=(n_total, 4))
    turns = np.lib.format.open_memmap(out_dir / "turns.npy", mode="w+", dtype=np.int16, shape=(n_total,))
    positions = np.lib.format.open_memmap(out_dir / "positions.npy", mode="w+", dtype=np.int8, shape=(n_total,))
    action_indices = np.lib.format.open_memmap(out_dir / "action_indices.npy", mode="w+", dtype=np.int16, shape=(n_total,))
    candidate_ranks = np.lib.format.open_memmap(out_dir / "candidate_ranks.npy", mode="w+", dtype=np.int16, shape=(n_total,))
    group_ids = np.lib.format.open_memmap(out_dir / "group_ids.npy", mode="w+", dtype=np.int64, shape=(n_total,))
    route_tags = np.lib.format.open_memmap(out_dir / "route_tags.npy", mode="w+", dtype=np.int16, shape=(n_total,))
    sample_weights = np.lib.format.open_memmap(out_dir / "sample_weights.npy", mode="w+", dtype=np.float32, shape=(n_total,))
    teacher_gaps = np.lib.format.open_memmap(out_dir / "teacher_gaps.npy", mode="w+", dtype=np.float32, shape=(n_total,))

    idx = 0
    group_id = 0
    skipped = 0
    invalid_records = 0
    invalid_selected_candidates = 0
    written_by_turn: dict[int, int] = {}
    route_counts = {name: 0 for name in ROUTE_TAGS}
    fl_type_sums = np.zeros(len(FL_TYPE_KEYS), dtype=np.float64)
    t0 = time.time()

    with input_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            record = json.loads(line)
            turn = int(record.get("turn", -1))
            candidates = record.get("candidates") or []
            if turn not in include_turns or not candidates:
                continue
            position = canonical_record_position(record)
            canonical_record = dict(record)
            canonical_record["position"] = position
            canonical_record["is_btn"] = position == "btn"

            eval_mode = record.get("eval_mode", "exact")
            chosen = select_candidate_indices(len(candidates), candidate_limit_for_turn(turn, args))
            if turn == 0 and args.keep_t0_fl_shaped and args.t0_max_candidates > 0:
                chosen_set = set(chosen)
                base_board = build_board(record)
                for i, cand in enumerate(candidates):
                    if i in chosen_set:
                        continue
                    try:
                        next_board, _discard = apply_candidate(base_board, cand, record.get("dealt", []))
                    except Exception:
                        continue
                    if _has_high_top_route(next_board) or _has_fl_ready_top(next_board):
                        chosen_set.add(i)
                chosen = sorted(chosen_set)
            chosen, invalid = filter_valid_candidate_indices(record, candidates, chosen)
            invalid_selected_candidates += invalid
            all_valid = valid_candidate_indices(record, candidates)
            if not chosen or not all_valid:
                invalid_records += 1
                continue
            best_score = float(get_candidate_ev(candidates[all_valid[0]], eval_mode))
            second_score = (
                float(get_candidate_ev(candidates[all_valid[1]], eval_mode))
                if len(all_valid) > 1
                else best_score
            )
            decision_gap = best_score - second_score

            for rank in chosen:
                candidate = candidates[rank]
                try:
                    obs = post_action_observation(canonical_record, candidate)
                    state = adapt_np_state_with_action(encode_state(obs), args.state_dim, canonical_record, candidate)
                except Exception:
                    skipped += 1
                    continue

                raw_ev = float(get_candidate_ev(candidate, eval_mode))
                bust_rate = float(get_candidate_bust(candidate, eval_mode))
                fl_rate = float(get_candidate_fl(candidate, eval_mode))
                fl_vec = fl_type_vector(candidate, eval_mode)
                candidate_gap = best_score - raw_ev
                tag = _route_tag(
                    turn,
                    obs.board_self,
                    fl_rate,
                    bust_rate,
                    int(rank),
                    decision_gap,
                    fl_vec,
                    args,
                )
                ev = raw_ev + _score_adjustment(tag, args)

                states[idx] = state
                scores[idx] = ev
                bust[idx] = bust_rate
                fl[idx] = fl_rate
                fl_types[idx] = fl_vec
                turns[idx] = turn
                positions[idx] = 1 if obs.is_btn else 0
                action_indices[idx] = action_index(record, candidate)
                candidate_ranks[idx] = int(rank)
                group_ids[idx] = group_id
                route_tags[idx] = tag
                sample_weights[idx] = _sample_weight(tag, fl_rate, bust_rate, candidate_gap, args)
                teacher_gaps[idx] = candidate_gap
                fl_type_sums += fl_vec
                for name, bit in ROUTE_TAGS.items():
                    if tag & bit:
                        route_counts[name] += 1
                idx += 1
                written_by_turn[turn] = written_by_turn.get(turn, 0) + 1

            group_id += 1
            if group_id % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  records={group_id:,} samples={idx:,} skipped={skipped:,} ({elapsed:.0f}s)", flush=True)
            if args.limit_records and group_id >= args.limit_records:
                break

    for arr in (
        states, scores, bust, fl, fl_types, turns, positions, action_indices,
        candidate_ranks, group_ids, route_tags, sample_weights, teacher_gaps,
    ):
        arr.flush()

    metadata = {
        "source": str(input_path),
        "n_allocated": int(n_total),
        "n_samples": int(idx),
        "n_records": int(group_id),
        "state_dim": int(args.state_dim),
        "action_feature_dim": int(ACTION_FEATURE_DIM if args.state_dim not in (520, 522, 822) else 0),
        "fl_opportunity_feature_dim": int(
            FL_OPPORTUNITY_FEATURE_DIM
            if args.state_dim not in (520, 522, 822, 520 + ACTION_FEATURE_DIM, 522 + ACTION_FEATURE_DIM, 822 + ACTION_FEATURE_DIM)
            else 0
        ),
        "board_context_feature_dim": int(
            BOARD_CONTEXT_FEATURE_DIM
            if args.state_dim
            not in (
                520,
                522,
                822,
                520 + ACTION_FEATURE_DIM,
                522 + ACTION_FEATURE_DIM,
                822 + ACTION_FEATURE_DIM,
                520 + ACTION_FEATURE_DIM + FL_OPPORTUNITY_FEATURE_DIM,
                522 + ACTION_FEATURE_DIM + FL_OPPORTUNITY_FEATURE_DIM,
                822 + ACTION_FEATURE_DIM + FL_OPPORTUNITY_FEATURE_DIM,
            )
            else 0
        ),
        "turns": {str(k): int(v) for k, v in sorted(written_by_turn.items())},
        "record_turns": {str(k): int(v) for k, v in sorted(count_meta["turns"].items())},
        "skipped": int(skipped),
        "invalid_records": int(invalid_records),
        "invalid_selected_candidates": int(invalid_selected_candidates),
        "count_invalid_records": int(count_meta.get("invalid_records", 0)),
        "count_invalid_selected_candidates": int(count_meta.get("invalid_selected_candidates", 0)),
        "t0_max_candidates": int(args.t0_max_candidates),
        "regular_max_candidates": int(args.regular_max_candidates),
        "score_mean": float(np.asarray(scores[:idx]).mean()) if idx else 0.0,
        "score_std": float(np.asarray(scores[:idx]).std()) if idx else 1.0,
        "bust_mean": float(np.asarray(bust[:idx]).mean()) if idx else 0.0,
        "fl_mean": float(np.asarray(fl[:idx]).mean()) if idx else 0.0,
        "positions": {
            "bb": int((np.asarray(positions[:idx]) == 0).sum()) if idx else 0,
            "btn": int((np.asarray(positions[:idx]) == 1).sum()) if idx else 0,
        },
        "sample_weight_mean": float(np.asarray(sample_weights[:idx]).mean()) if idx else 1.0,
        "sample_weight_max": float(np.asarray(sample_weights[:idx]).max()) if idx else 1.0,
        "route_tag_bits": ROUTE_TAGS,
        "route_counts": {str(k): int(v) for k, v in route_counts.items()},
        "fl_type_keys": list(FL_TYPE_KEYS),
        "fl_type_sums": {k: float(fl_type_sums[i]) for i, k in enumerate(FL_TYPE_KEYS)},
        "route_safety_config": {
            "t0_safe_fl_min": float(args.t0_safe_fl_min),
            "t0_safe_bust_max": float(args.t0_safe_bust_max),
            "t0_risky_fl_min": float(args.t0_risky_fl_min),
            "t0_risky_bust_min": float(args.t0_risky_bust_min),
            "t0_dead_fl_max": float(args.t0_dead_fl_max),
            "t0_dead_bust_min": float(args.t0_dead_bust_min),
            "t0_safe_route_score_bonus": float(args.t0_safe_route_score_bonus),
            "t0_risky_route_score_penalty": float(args.t0_risky_route_score_penalty),
            "t0_dead_route_score_penalty": float(args.t0_dead_route_score_penalty),
            "trips_fl_score_bonus": float(args.trips_fl_score_bonus),
        },
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  samples={idx:,} skipped={skipped:,}")
    print(f"  score={metadata['score_mean']:+.3f} +/- {metadata['score_std']:.3f}")
    print(f"  bust={metadata['bust_mean']:.1%} fl={metadata['fl_mean']:.1%}")
    print(f"  weight_mean={metadata['sample_weight_mean']:.2f} max={metadata['sample_weight_max']:.2f}")
    print(f"  routes={metadata['route_counts']}")
    print(f"  saved={out_dir}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert teacher JSONL to action-value samples")
    parser.add_argument("input", help="Input teacher JSONL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--turns", default="0,1,2,3,4", help="Turns to include")
    parser.add_argument("--state-dim", type=int, default=520)
    parser.add_argument("--t0-max-candidates", type=int, default=0,
                        help="0 keeps all T0 candidates")
    parser.add_argument("--regular-max-candidates", type=int, default=27)
    parser.add_argument("--keep-t0-fl-shaped", action="store_true",
                        help="When limiting T0, always keep extra candidates that place Q/K/A/Joker on top")
    parser.add_argument("--fl-positive-weight", type=float, default=2.0)
    parser.add_argument("--high-route-weight", type=float, default=1.0)
    parser.add_argument("--fl-shape-but-foul-weight", type=float, default=1.5)
    parser.add_argument("--teacher-best-weight", type=float, default=0.5)
    parser.add_argument("--gap-weight", type=float, default=0.5)
    parser.add_argument("--clean-fl-weight", type=float, default=1.0)
    parser.add_argument("--t0-safe-fl-min", type=float, default=0.20)
    parser.add_argument("--t0-safe-bust-max", type=float, default=0.35)
    parser.add_argument("--t0-risky-fl-min", type=float, default=0.20)
    parser.add_argument("--t0-risky-bust-min", type=float, default=0.35)
    parser.add_argument("--t0-dead-fl-max", type=float, default=0.05)
    parser.add_argument("--t0-dead-bust-min", type=float, default=0.25)
    parser.add_argument("--t0-safe-route-weight", type=float, default=0.0)
    parser.add_argument("--t0-risky-route-weight", type=float, default=0.0)
    parser.add_argument("--t0-dead-route-weight", type=float, default=0.0)
    parser.add_argument("--trips-fl-weight", type=float, default=0.0)
    parser.add_argument("--t0-safe-route-score-bonus", type=float, default=0.0)
    parser.add_argument("--t0-risky-route-score-penalty", type=float, default=0.0)
    parser.add_argument("--t0-dead-route-score-penalty", type=float, default=0.0)
    parser.add_argument("--trips-fl-score-bonus", type=float, default=0.0)
    parser.add_argument("--max-sample-weight", type=float, default=8.0)
    parser.add_argument("--limit-records", type=int, default=0)
    args = parser.parse_args(argv)
    convert(args)


if __name__ == "__main__":
    main()
