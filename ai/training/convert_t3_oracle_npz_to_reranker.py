"""Convert fixed-slot T3 oracle NPZ shards to action-value reranker data.

The T3 oracle shards store one pre-action state per decision plus EVs for the
27 semantic regular-turn actions.  The action-value reranker expects one sample
per candidate action, using the post-action state.  This converter decodes the
pre-action state, applies each valid semantic action, and writes the same NPY
layout used by ``convert_action_value_teacher.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.action_space import (
    REGULAR_TURN_ACTIONS,
    get_action_from_semantic_index_if_valid,
)
from ai.engine.encoding import (
    ALL_CARDS,
    LOC_IN_HAND,
    LOC_MY_BOT,
    LOC_MY_DISCARD,
    LOC_MY_MID,
    LOC_MY_TOP,
    LOC_OPP_BOT,
    LOC_OPP_MID,
    LOC_OPP_TOP,
    Board,
    Observation,
    encode_state,
)
from ai.training.action_feature_encoding import ACTION_FEATURE_DIM, adapt_np_state_with_action


TEACHER_BEST_TAG = 8


def _npz_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ shards found in {input_dir}")
    return files


def _row_cards(card_matrix: np.ndarray, loc: int, threshold: float) -> list[str]:
    indices = np.flatnonzero(card_matrix[:, loc] > threshold)
    return [ALL_CARDS[int(i)] for i in indices]


def decode_observation(state: np.ndarray, threshold: float = 0.5) -> Observation:
    """Decode the card-location part of an encoded pre-action observation."""
    card_matrix = np.asarray(state[:486], dtype=np.float32).reshape(54, 9)
    turn = int(round(float(state[486]) * 4.0)) if len(state) > 486 else 3
    is_btn = bool(float(state[487]) >= 0.5) if len(state) > 487 else True
    is_fl = bool(float(state[488]) >= 0.5) if len(state) > 488 else False
    opp_is_fl = bool(float(state[489]) >= 0.5) if len(state) > 489 else False
    chips_self = int(round(float(state[490]) * 200.0)) if len(state) > 490 else 200
    chips_opponent = int(round(float(state[491]) * 200.0)) if len(state) > 491 else 200
    return Observation(
        board_self=Board(
            top=_row_cards(card_matrix, LOC_MY_TOP, threshold),
            middle=_row_cards(card_matrix, LOC_MY_MID, threshold),
            bottom=_row_cards(card_matrix, LOC_MY_BOT, threshold),
        ),
        board_opponent=Board(
            top=_row_cards(card_matrix, LOC_OPP_TOP, threshold),
            middle=_row_cards(card_matrix, LOC_OPP_MID, threshold),
            bottom=_row_cards(card_matrix, LOC_OPP_BOT, threshold),
        ),
        dealt_cards=_row_cards(card_matrix, LOC_IN_HAND, threshold),
        known_discards_self=_row_cards(card_matrix, LOC_MY_DISCARD, threshold),
        turn=turn,
        is_btn=is_btn,
        is_fl=is_fl,
        opp_is_fl=opp_is_fl,
        chips_self=max(chips_self, 1),
        chips_opponent=max(chips_opponent, 1),
    )


def apply_action_observation(obs: Observation, action_idx: int) -> tuple[Observation, dict] | None:
    action = get_action_from_semantic_index_if_valid(action_idx, obs.dealt_cards, obs.board_self)
    if action is None:
        return None
    board = obs.board_self.copy()
    placements = [(card, row) for card, row in action.placements]
    for card, row in placements:
        getattr(board, row).append(card)
    discards = list(obs.known_discards_self)
    if action.discard and action.discard not in discards:
        discards.append(action.discard)
    next_obs = Observation(
        board_self=board,
        board_opponent=obs.board_opponent.copy(),
        dealt_cards=[],
        known_discards_self=discards,
        turn=obs.turn,
        is_btn=obs.is_btn,
        is_fl=obs.is_fl,
        opp_is_fl=obs.opp_is_fl,
        chips_self=obs.chips_self,
        chips_opponent=obs.chips_opponent,
    )
    candidate = {
        "placements": [[card, row] for card, row in placements],
        "discard": action.discard,
    }
    return next_obs, candidate


def decision_for_features(obs: Observation) -> dict:
    return {
        "turn": obs.turn,
        "board": obs.board_self.to_dict(),
        "dealt": list(obs.dealt_cards),
        "is_btn": obs.is_btn,
    }


def _valid_action_indices(valid_mask: np.ndarray, evs: np.ndarray) -> np.ndarray:
    mask = np.asarray(valid_mask[:REGULAR_TURN_ACTIONS], dtype=bool)
    finite = np.isfinite(evs[:REGULAR_TURN_ACTIONS]) & (evs[:REGULAR_TURN_ACTIONS] > -9999.0)
    return np.flatnonzero(mask & finite).astype(np.int16)


def count_samples(input_dir: Path, limit_records: int = 0) -> tuple[int, int, int]:
    total_samples = 0
    total_records = 0
    total_files = 0
    for path in _npz_files(input_dir):
        shard = np.load(path)
        evs = shard["action_evs"]
        masks = shard["valid_masks"] if "valid_masks" in shard else shard["action_masks"]
        remaining = evs.shape[0] if limit_records <= 0 else max(limit_records - total_records, 0)
        if remaining <= 0:
            break
        n = min(evs.shape[0], remaining)
        for i in range(n):
            total_samples += int(len(_valid_action_indices(masks[i], evs[i])))
        total_records += n
        total_files += 1
    return total_samples, total_records, total_files


def convert(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total, n_records, n_files = count_samples(input_dir, args.limit_records)
    if n_total <= 0:
        raise SystemExit("No valid T3 samples found")

    print(f"Reading: {input_dir}")
    print(f"Output:  {out_dir}")
    print(f"Files:   {n_files}")
    print(f"Records: {n_records:,}")
    print(f"Samples: {n_total:,}")
    print(f"State:   {args.state_dim} dims")

    states = np.lib.format.open_memmap(
        out_dir / "states.npy", mode="w+", dtype=np.float32, shape=(n_total, args.state_dim)
    )
    scores = np.lib.format.open_memmap(out_dir / "scores.npy", mode="w+", dtype=np.float32, shape=(n_total,))
    bust = np.lib.format.open_memmap(out_dir / "bust.npy", mode="w+", dtype=np.float32, shape=(n_total,))
    fl = np.lib.format.open_memmap(out_dir / "fl.npy", mode="w+", dtype=np.float32, shape=(n_total,))
    fl_types = np.lib.format.open_memmap(out_dir / "fl_types.npy", mode="w+", dtype=np.float32, shape=(n_total, 4))
    turns = np.lib.format.open_memmap(out_dir / "turns.npy", mode="w+", dtype=np.int16, shape=(n_total,))
    action_indices = np.lib.format.open_memmap(out_dir / "action_indices.npy", mode="w+", dtype=np.int16, shape=(n_total,))
    candidate_ranks = np.lib.format.open_memmap(out_dir / "candidate_ranks.npy", mode="w+", dtype=np.int16, shape=(n_total,))
    group_ids = np.lib.format.open_memmap(out_dir / "group_ids.npy", mode="w+", dtype=np.int64, shape=(n_total,))
    route_tags = np.lib.format.open_memmap(out_dir / "route_tags.npy", mode="w+", dtype=np.int16, shape=(n_total,))
    sample_weights = np.lib.format.open_memmap(out_dir / "sample_weights.npy", mode="w+", dtype=np.float32, shape=(n_total,))
    teacher_gaps = np.lib.format.open_memmap(out_dir / "teacher_gaps.npy", mode="w+", dtype=np.float32, shape=(n_total,))

    idx = 0
    group_id = 0
    skipped = 0
    t0 = time.time()
    for path in _npz_files(input_dir):
        if args.limit_records and group_id >= args.limit_records:
            break
        shard = np.load(path)
        raw_states = shard["states"]
        evs = shard["action_evs"]
        masks = shard["valid_masks"] if "valid_masks" in shard else shard["action_masks"]
        n = raw_states.shape[0]
        for row in range(n):
            if args.limit_records and group_id >= args.limit_records:
                break
            action_ids = _valid_action_indices(masks[row], evs[row])
            if len(action_ids) == 0:
                group_id += 1
                continue
            obs = decode_observation(raw_states[row], threshold=args.decode_threshold)
            decision = decision_for_features(obs)
            valid_evs = np.asarray(evs[row, action_ids], dtype=np.float32)
            order = np.argsort(-valid_evs)
            rank_by_action = {int(action_ids[int(pos)]): rank for rank, pos in enumerate(order)}
            best_ev = float(valid_evs[int(order[0])])

            for action_idx in action_ids:
                action_idx_int = int(action_idx)
                applied = apply_action_observation(obs, action_idx_int)
                if applied is None:
                    skipped += 1
                    continue
                next_obs, candidate = applied
                state = adapt_np_state_with_action(
                    encode_state(next_obs),
                    args.state_dim,
                    decision,
                    candidate,
                )
                ev = float(evs[row, action_idx_int])
                rank = int(rank_by_action[action_idx_int])
                gap = best_ev - ev

                states[idx] = state
                scores[idx] = ev
                bust[idx] = 0.0
                fl[idx] = 0.0
                fl_types[idx] = 0.0
                turns[idx] = int(obs.turn)
                action_indices[idx] = action_idx_int
                candidate_ranks[idx] = rank
                group_ids[idx] = group_id
                route_tags[idx] = TEACHER_BEST_TAG if rank == 0 else 0
                sample_weights[idx] = min(
                    1.0 + (args.teacher_best_weight if rank == 0 else 0.0)
                    + min(max(gap, 0.0) / 10.0, 2.0) * args.gap_weight,
                    args.max_sample_weight,
                )
                teacher_gaps[idx] = gap
                idx += 1

            group_id += 1
            if group_id % args.progress_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  records={group_id:,} samples={idx:,} skipped={skipped:,} "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )

    for arr in (
        states, scores, bust, fl, fl_types, turns, action_indices,
        candidate_ranks, group_ids, route_tags, sample_weights, teacher_gaps,
    ):
        arr.flush()

    metadata = {
        "source": str(input_dir),
        "format": "t3_oracle_npz",
        "n_allocated": int(n_total),
        "n_samples": int(idx),
        "n_records": int(group_id),
        "state_dim": int(args.state_dim),
        "action_feature_dim": int(ACTION_FEATURE_DIM if args.state_dim not in (520, 522, 822) else 0),
        "turns": {"3": int(idx)},
        "record_turns": {"3": int(group_id)},
        "skipped": int(skipped),
        "regular_max_candidates": int(REGULAR_TURN_ACTIONS),
        "score_mean": float(np.asarray(scores[:idx]).mean()) if idx else 0.0,
        "score_std": float(np.asarray(scores[:idx]).std()) if idx else 1.0,
        "bust_mean": 0.0,
        "fl_mean": 0.0,
        "sample_weight_mean": float(np.asarray(sample_weights[:idx]).mean()) if idx else 1.0,
        "sample_weight_max": float(np.asarray(sample_weights[:idx]).max()) if idx else 1.0,
        "route_tag_bits": {"teacher_best": TEACHER_BEST_TAG},
        "route_counts": {"teacher_best": int(np.asarray(candidate_ranks[:idx] == 0).sum())},
        "label_note": "EV labels are exact/teacher action_evs from T3 oracle NPZ; FL/bust labels are unavailable and set to zero.",
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  samples={idx:,} skipped={skipped:,}")
    print(f"  score={metadata['score_mean']:+.3f} +/- {metadata['score_std']:.3f}")
    print(f"  saved={out_dir}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert T3 oracle NPZ shards to action-value samples")
    parser.add_argument("input", help="Input directory with T3 oracle NPZ shards")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--state-dim", type=int, default=520 + ACTION_FEATURE_DIM)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--decode-threshold", type=float, default=0.5)
    parser.add_argument("--teacher-best-weight", type=float, default=0.5)
    parser.add_argument("--gap-weight", type=float, default=0.5)
    parser.add_argument("--max-sample-weight", type=float, default=4.0)
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args(argv)
    convert(args)


if __name__ == "__main__":
    main()
