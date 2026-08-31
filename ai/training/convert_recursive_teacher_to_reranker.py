"""Convert recursive T0/T1-T3 teacher JSONL to action-value reranker data.

The recursive teacher generator writes one JSONL record per T0 hand.  Each
record contains a ranked T0 candidate pool plus optional sampled T1-T3 trace
decisions.  The action-value reranker trainer expects one sample per candidate
and one group per decision, so this converter flattens that nested format into
the same numpy files produced by ``convert_action_value_teacher.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state
from ai.training.convert_action_value_teacher import (
    FL_TYPE_KEYS,
    ROUTE_TAGS,
    _adapt_np_state,
    _route_tag,
    _sample_weight,
    _score_adjustment,
    select_candidate_indices,
)
from ai.training.convert_mc_teacher import action_to_index_t0, action_to_index_t1plus


ROW_ALIASES = {
    "top": "top",
    "mid": "middle",
    "middle": "middle",
    "bot": "bottom",
    "bottom": "bottom",
}


def _row_name(row: str) -> str:
    return ROW_ALIASES.get(row, row)


def _is_joker(card: object) -> bool:
    return str(card).startswith("X")


def _is_concrete_joker(card: object) -> bool:
    return str(card) in {"X1", "X2"}


class JokerNormalizer:
    """Map generic ``Xj`` joker refs to concrete ``X1``/``X2`` ids."""

    def __init__(self, preferred: Sequence[str]):
        pref = [c for c in preferred if _is_concrete_joker(c)]
        self.preferred = list(dict.fromkeys(pref + ["X1", "X2"]))
        self.used: set[str] = set()

    def card(self, card: str | None) -> str | None:
        if card is None:
            return None
        if not _is_joker(card):
            return card
        if _is_concrete_joker(card):
            self.used.add(card)
            return card
        for candidate in self.preferred:
            if candidate not in self.used:
                self.used.add(candidate)
                return candidate
        return self.preferred[0]

    def cards(self, cards: Sequence[str]) -> list[str]:
        return [c for c in (self.card(card) for card in cards) if c is not None]

    def placements(self, placements: Sequence[Sequence[str]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for item in placements:
            if len(item) != 2:
                continue
            card, row = item
            norm_card = self.card(str(card))
            if norm_card is None:
                continue
            out.append((norm_card, _row_name(str(row))))
        return out


def board_from_dict_norm(board: dict, normalizer: JokerNormalizer) -> Board:
    return Board(
        top=normalizer.cards(list(board.get("top", []))),
        middle=normalizer.cards(list(board.get("middle", [])) + list(board.get("mid", []))),
        bottom=normalizer.cards(list(board.get("bottom", [])) + list(board.get("bot", []))),
    )


def board_cards(board: dict) -> list[str]:
    cards: list[str] = []
    for row in ("top", "middle", "bottom", "mid", "bot"):
        cards.extend(str(card) for card in board.get(row, []) or [])
    return cards


def board_to_placements(board: dict, normalizer: JokerNormalizer) -> list[tuple[str, str]]:
    placements: list[tuple[str, str]] = []
    for row in ("top", "middle", "bottom"):
        for card in board.get(row, []) + board.get({"middle": "mid", "bottom": "bot"}.get(row, ""), []):
            norm_card = normalizer.card(str(card))
            if norm_card is not None:
                placements.append((norm_card, row))
    return placements


def board_card_count(board: dict) -> int:
    return sum(len(board.get(k, [])) for k in ("top", "middle", "bottom", "mid", "bot"))


def candidate_metrics(candidate: dict) -> tuple[float, float, float, np.ndarray]:
    recursive = candidate.get("recursive") or {}
    mc = recursive.get("mc") or candidate.get("mc") or {}
    pe = candidate.get("pe") or {}
    score = float(candidate.get("target_score", mc.get("avg_score", pe.get("ev", candidate.get("ev", 0.0))) or 0.0))
    bust = float(mc.get("bust_rate", pe.get("bust_prob", candidate.get("bust_prob", 0.0)) or 0.0))
    fl = float(mc.get("fl_rate", pe.get("fl_rate", candidate.get("fl_rate", 0.0)) or 0.0))
    type_rates = mc.get("fl_type_rates", {}) or {}
    fl_types = np.array([float(type_rates.get(k, 0.0)) for k in FL_TYPE_KEYS], dtype=np.float32)
    return score, bust, fl, fl_types


def top_candidate_to_regular(candidate: dict, t0_dealt: Sequence[str]) -> dict:
    normalizer = JokerNormalizer(t0_dealt)
    placements = board_to_placements(candidate.get("board", {}) or {}, normalizer)
    return {
        **candidate,
        "placements": placements,
        "discard": None,
    }


def trace_candidate_to_regular(candidate: dict) -> dict:
    placements: list[tuple[str, str]] = []
    for item in candidate.get("placements", []) or []:
        if len(item) != 2:
            continue
        card, row = item
        placements.append((str(card), _row_name(str(row))))
    return {
        **candidate,
        "placements": placements,
        "discard": candidate.get("discard") if candidate.get("discard") not in (None, "") else None,
    }


def decision_candidates(record: dict, args: argparse.Namespace) -> Iterator[dict]:
    t0_dealt = list(record.get("dealt", []))
    position = str(record.get("position") or args.position)
    opponent_board = record.get("opponent_board") or record.get("board_opponent") or {}
    top_known = list(record.get("known_discards", []) or record.get("exclude", []) or [])
    if 0 in args.include_turns:
        candidates = record.get("candidates") or []
        if candidates:
            yield {
                "turn": 0,
                "is_btn": position == "btn",
                "board": {},
                "dealt": t0_dealt,
                "known_discards": top_known,
                "opponent_board": opponent_board,
                "candidates": [top_candidate_to_regular(c, t0_dealt) for c in candidates],
                "source": "t0",
                "position": position,
            }

    for trace in record.get("turn_traces") or []:
        turn = int(trace.get("turn", -1))
        if turn not in args.include_turns:
            continue
        candidates = trace.get("candidates") or []
        if not candidates:
            continue
        dealt = list(trace.get("dealt", []))
        dead = list(trace.get("dead_before", []))
        regular_candidates = [trace_candidate_to_regular(cand) for cand in candidates]
        yield {
            "turn": turn,
            "is_btn": position == "btn",
            "board": trace.get("board_before", {}) or {},
            "dealt": dealt,
            "known_discards": dead,
            "opponent_board": opponent_board,
            "t0_dealt": t0_dealt,
            "candidates": regular_candidates,
            "source": "trace",
            "position": position,
        }


def candidate_limit(turn: int, args: argparse.Namespace) -> int:
    return args.t0_max_candidates if turn == 0 else args.regular_max_candidates


def count_samples(input_path: Path, args: argparse.Namespace) -> tuple[int, dict]:
    total = 0
    bad_lines = 0
    hands = 0
    decisions = 0
    turns: dict[int, int] = {}
    candidates_by_turn: dict[int, int] = {}
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            hands += 1
            for decision in decision_candidates(record, args):
                turn = int(decision["turn"])
                candidates = decision.get("candidates") or []
                chosen = select_candidate_indices(len(candidates), candidate_limit(turn, args))
                total += len(chosen)
                decisions += 1
                turns[turn] = turns.get(turn, 0) + 1
                candidates_by_turn[turn] = candidates_by_turn.get(turn, 0) + len(chosen)
            if args.limit_hands and hands >= args.limit_hands:
                break
    return total, {
        "hands": hands,
        "decisions": decisions,
        "turns": turns,
        "candidates_by_turn": candidates_by_turn,
        "bad_lines": bad_lines,
    }


def post_action_observation(decision: dict, candidate: dict) -> tuple[Observation, dict]:
    dealt = list(decision.get("dealt", []))
    known = list(decision.get("known_discards", []))
    preferred_jokers = list(decision.get("t0_dealt", [])) + dealt + known
    for raw_card, _row in candidate.get("placements", []) or []:
        preferred_jokers.append(raw_card)
    if candidate.get("discard"):
        preferred_jokers.append(candidate["discard"])

    normalizer = JokerNormalizer(preferred_jokers)
    board = board_from_dict_norm(decision.get("board", {}) or {}, normalizer)
    opponent_board = board_from_dict_norm(decision.get("opponent_board", {}) or {}, normalizer)
    placements = normalizer.placements(candidate.get("placements", []) or [])
    discard = normalizer.card(candidate.get("discard")) if candidate.get("discard") else None
    known_discards = normalizer.cards(known)
    opponent_cards = set(opponent_board.all_cards())
    known_discards = [card for card in known_discards if card not in opponent_cards]

    for card, row in placements:
        getattr(board, _row_name(row)).append(card)
    if discard and discard not in known_discards:
        known_discards.append(discard)

    normalized_candidate = {
        **candidate,
        "placements": placements,
        "discard": discard,
    }
    obs = Observation(
        board_self=board,
        board_opponent=opponent_board,
        dealt_cards=[],
        known_discards_self=known_discards,
        turn=int(decision.get("turn", 0)),
        is_btn=bool(decision.get("is_btn", False)),
        is_fl=False,
        opp_is_fl=False,
        chips_self=200,
        chips_opponent=200,
    )
    return obs, normalized_candidate


def action_index(decision: dict, candidate: dict) -> int:
    try:
        turn = int(decision.get("turn", 0))
        if turn == 0:
            return int(action_to_index_t0(candidate.get("placements", []), decision.get("dealt", [])))
        return int(action_to_index_t1plus(candidate, decision.get("dealt", [])))
    except Exception:
        return -1


def write_summary_md(path: Path, metadata: dict) -> None:
    lines = [
        "# Recursive Teacher Reranker Data",
        "",
        f"- source: `{metadata['source']}`",
        f"- samples: {metadata['n_samples']:,}",
        f"- decisions: {metadata['n_decisions']:,}",
        f"- hands: {metadata['n_hands']:,}",
        f"- skipped: {metadata['skipped']:,}",
        f"- bad JSONL lines: {metadata['bad_lines']:,}",
        f"- score mean/std: {metadata['score_mean']:+.3f} / {metadata['score_std']:.3f}",
        f"- bust mean: {metadata['bust_mean']:.1%}",
        f"- FL mean: {metadata['fl_mean']:.1%}",
        "",
        "## Samples By Turn",
        "",
    ]
    for turn, count in metadata["turns"].items():
        lines.append(f"- T{turn}: {count:,}")
    lines.extend(["", "## Route Counts", ""])
    for name, count in metadata["route_counts"].items():
        lines.append(f"- {name}: {count:,}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.include_turns = {int(t) for t in args.turns.split(",") if t.strip()}

    print(f"Reading: {input_path}")
    print(f"Output:  {out_dir}")
    print(f"Turns:   {sorted(args.include_turns)}")
    print(f"State:   {args.state_dim} dims")

    n_total, count_meta = count_samples(input_path, args)
    if n_total <= 0:
        raise SystemExit("No samples matched the requested filters.")

    states = np.lib.format.open_memmap(out_dir / "states.npy", mode="w+", dtype=np.float32, shape=(n_total, args.state_dim))
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
    hands = 0
    bad_lines = 0
    written_by_turn: dict[int, int] = {}
    record_turns: dict[int, int] = {}
    route_counts = {name: 0 for name in ROUTE_TAGS}
    fl_type_sums = np.zeros(len(FL_TYPE_KEYS), dtype=np.float64)
    start = time.time()

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            hands += 1
            for decision in decision_candidates(record, args):
                turn = int(decision["turn"])
                candidates = decision.get("candidates") or []
                chosen = select_candidate_indices(len(candidates), candidate_limit(turn, args))
                if not chosen:
                    continue
                best_score, _best_bust, _best_fl, _best_types = candidate_metrics(candidates[chosen[0]])
                second_score = best_score
                if len(chosen) > 1:
                    second_score = candidate_metrics(candidates[chosen[1]])[0]
                decision_gap = best_score - second_score

                wrote_group = False
                for rank in chosen:
                    candidate = candidates[rank]
                    try:
                        obs, norm_candidate = post_action_observation(decision, candidate)
                        state = _adapt_np_state(encode_state(obs), args.state_dim)
                    except Exception:
                        skipped += 1
                        continue

                    raw_ev, bust_rate, fl_rate, fl_vec = candidate_metrics(candidate)
                    candidate_gap = best_score - raw_ev
                    tag = _route_tag(turn, obs.board_self, fl_rate, bust_rate, int(rank), decision_gap, fl_vec, args)
                    ev = raw_ev + _score_adjustment(tag, args)

                    states[idx] = state
                    scores[idx] = ev
                    bust[idx] = bust_rate
                    fl[idx] = fl_rate
                    fl_types[idx] = fl_vec
                    turns[idx] = turn
                    positions[idx] = 1 if bool(decision.get("is_btn", False)) else 0
                    action_indices[idx] = action_index(decision, norm_candidate)
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
                    wrote_group = True
                    written_by_turn[turn] = written_by_turn.get(turn, 0) + 1

                if wrote_group:
                    record_turns[turn] = record_turns.get(turn, 0) + 1
                    group_id += 1
                    if group_id % 5000 == 0:
                        elapsed = time.time() - start
                        print(f"  decisions={group_id:,} samples={idx:,} skipped={skipped:,} ({elapsed:.0f}s)", flush=True)
            if args.limit_hands and hands >= args.limit_hands:
                break

    for arr in (
        states, scores, bust, fl, fl_types, turns, positions, action_indices,
        candidate_ranks, group_ids, route_tags, sample_weights, teacher_gaps,
    ):
        arr.flush()

    score_view = np.asarray(scores[:idx])
    bust_view = np.asarray(bust[:idx])
    fl_view = np.asarray(fl[:idx])
    weight_view = np.asarray(sample_weights[:idx])
    metadata = {
        "source": str(input_path),
        "source_format": "recursive_t0_random",
        "output": str(out_dir),
        "n_allocated": int(n_total),
        "n_samples": int(idx),
        "n_hands": int(hands),
        "n_decisions": int(group_id),
        "bad_lines": int(bad_lines),
        "state_dim": int(args.state_dim),
        "position": args.position,
        "position_note": "Per-record position is used when the input contains a position field; this value is only the fallback.",
        "turns": {str(k): int(v) for k, v in sorted(written_by_turn.items())},
        "record_turns": {str(k): int(v) for k, v in sorted(record_turns.items())},
        "skipped": int(skipped),
        "t0_max_candidates": int(args.t0_max_candidates),
        "regular_max_candidates": int(args.regular_max_candidates),
        "score_mean": float(score_view.mean()) if idx else 0.0,
        "score_std": float(score_view.std()) if idx else 1.0,
        "bust_mean": float(bust_view.mean()) if idx else 0.0,
        "fl_mean": float(fl_view.mean()) if idx else 0.0,
        "sample_weight_mean": float(weight_view.mean()) if idx else 1.0,
        "sample_weight_max": float(weight_view.max()) if idx else 1.0,
        "positions": {
            "bb": int((np.asarray(positions[:idx]) == 0).sum()) if idx else 0,
            "btn": int((np.asarray(positions[:idx]) == 1).sum()) if idx else 0,
        },
        "route_tag_bits": ROUTE_TAGS,
        "route_counts": {str(k): int(v) for k, v in route_counts.items()},
        "fl_type_keys": list(FL_TYPE_KEYS),
        "fl_type_sums": {k: float(fl_type_sums[i]) for i, k in enumerate(FL_TYPE_KEYS)},
        "notes": [
            "T0 candidates are the generated recursive teacher pool, not all 232 legal T0 placements unless the source was generated that way.",
            "This run is position-labeled by --position; current GCP recursive generator used bb.",
        ],
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    write_summary_md(out_dir / "summary.md", metadata)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  hands={hands:,} decisions={group_id:,} samples={idx:,} skipped={skipped:,}")
    print(f"  score={metadata['score_mean']:+.3f} +/- {metadata['score_std']:.3f}")
    print(f"  bust={metadata['bust_mean']:.1%} fl={metadata['fl_mean']:.1%}")
    print(f"  saved={out_dir}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert recursive teacher JSONL to action-value reranker samples")
    parser.add_argument("input", help="Input recursive teacher JSONL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--turns", default="0,1,2,3", help="Turns to include")
    parser.add_argument("--position", choices=["bb", "btn"], default="bb")
    parser.add_argument("--state-dim", type=int, default=520, choices=[520, 522, 822])
    parser.add_argument("--t0-max-candidates", type=int, default=0, help="0 keeps all generated T0 candidates")
    parser.add_argument("--regular-max-candidates", type=int, default=0, help="0 keeps all generated T1+ candidates")
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
    parser.add_argument("--limit-hands", type=int, default=0)
    args = parser.parse_args(argv)
    convert(args)


if __name__ == "__main__":
    main()
