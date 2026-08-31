#!/usr/bin/env python3
"""Generate T2 oracle data using a T3 value oracle.

Sources:
  - t0-jsonl: reuse solved T0 placements from JSONL shards
  - rule-random: deal random cards and use the existing rule/heuristic policy for T0/T1
"""
import argparse
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai.engine.action_space import (
    REGULAR_TURN_ACTIONS,
    get_initial_actions,
    get_semantic_action_index,
    get_turn_actions,
)
from ai.engine.encoding import ALL_CARDS, Board, Observation, encode_state
from ai.training.generate_data import heuristic_score, select_action
from ai.training.generate_t2_oracle_data import apply_placements, load_t3_oracle


ROW_RE = re.compile(r"Top\[(.*?)\]\s+Mid\[(.*?)\]\s+Bot\[(.*?)\]")


def parse_cards(s: str) -> list[str]:
    s = s.strip()
    return [] if not s else s.split()


def normalize_cards(cards: list[str]) -> list[str]:
    joker_id = 1
    normalized = []
    for card in cards:
        if card == "JK":
            normalized.append(f"X{joker_id}")
            joker_id += 1
        else:
            normalized.append(card)
    return normalized


def parse_t0_placement(desc: str) -> Board:
    m = ROW_RE.search(desc)
    if not m:
        raise ValueError(f"cannot parse placement: {desc}")
    cards = parse_cards(m.group(1)) + parse_cards(m.group(2)) + parse_cards(m.group(3))
    normalized = normalize_cards(cards)
    top_n = len(parse_cards(m.group(1)))
    mid_n = len(parse_cards(m.group(2)))
    return Board(
        top=normalized[:top_n],
        middle=normalized[top_n:top_n + mid_n],
        bottom=normalized[top_n + mid_n:],
    )


def iter_t0_boards(jsonl_dir: Path, limit: int = 0):
    count = 0
    for path in sorted(jsonl_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                placements = row.get("placements") or []
                if not placements:
                    continue
                yield parse_t0_placement(placements[0]["p"])
                count += 1
                if limit and count >= limit:
                    return


def choose_action(dealt_cards: list[str], board: Board, turn: int, policy: str):
    if policy == "first":
        actions = get_turn_actions(dealt_cards, board)
        return actions[0]
    if policy in ("random", "heuristic"):
        return select_action(dealt_cards, board, turn, strategy=policy)
    actions = get_turn_actions(dealt_cards, board)
    return random.choice(actions)


def card_rank(card: str) -> str | None:
    if card in ("X1", "X2"):
        return None
    return card[0]


def find_non_ace_trip(cards: list[str]) -> list[str] | None:
    ranks = {}
    for card in cards:
        rank = card_rank(card)
        if rank is None or rank == "A":
            continue
        ranks.setdefault(rank, []).append(card)
    trips = [same_rank[:3] for same_rank in ranks.values() if len(same_rank) >= 3]
    if not trips:
        return None
    return max(trips, key=lambda cs: "23456789TJQK".index(card_rank(cs[0]) or "2"))


def choose_from_actions(actions, board: Board, policy: str):
    if not actions:
        raise ValueError("No valid actions!")
    if policy == "random":
        return random.choice(actions)
    if policy == "heuristic":
        scored = [(heuristic_score(a, board), random.random(), a) for a in actions]
        scored.sort(reverse=True)
        return scored[0][2]
    return actions[0]


def generate_trip_t0_actions(dealt_cards: list[str], policy: str) -> list:
    trip_cards = find_non_ace_trip(dealt_cards)
    if not trip_cards:
        return []

    actions = get_initial_actions(dealt_cards, Board())
    branched = []
    for row in ("top", "bottom", "middle"):
        candidates = []
        for action in actions:
            by_card = {card: pos for card, pos in action.placements}
            if all(by_card.get(card) == row for card in trip_cards):
                candidates.append(action)
        if candidates:
            branched.append(choose_from_actions(candidates, Board(), policy))
    return branched


def finish_t1_state(board: Board, deck: list[str], t1_policy: str):
    t1_deal = [deck.pop() for _ in range(3)]
    t1_action = choose_action(t1_deal, board, turn=1, policy=t1_policy)
    board = apply_placements(board, t1_action.placements)
    discards = [t1_action.discard] if t1_action.discard else []

    t2_deal = [deck.pop() for _ in range(3)]
    return board, discards, t2_deal


def generate_rule_random_t1_states(t0_policy: str, t1_policy: str, branch_trips: bool):
    deck = list(ALL_CARDS)
    random.shuffle(deck)

    t0_deal = [deck.pop() for _ in range(5)]
    t0_actions = generate_trip_t0_actions(t0_deal, t0_policy) if branch_trips else []
    if not t0_actions:
        t0_actions = [select_action(t0_deal, Board(), turn=0, strategy=t0_policy)]

    states = []
    for t0_action in t0_actions:
        branch_deck = list(deck)
        board = apply_placements(Board(), t0_action.placements)
        states.append(finish_t1_state(board, branch_deck, t1_policy))
    return states


def evaluate_t2_state(board: Board, deal: list[str], discards: list[str], model, device, n_samples: int):
    actions = get_turn_actions(deal, board)
    if not actions:
        return None

    used = set(board.all_cards() + deal + discards)
    rem_deck = [c for c in ALL_CARDS if c not in used]
    action_evs = np.full(REGULAR_TURN_ACTIONS, -1.0e4, dtype=np.float32)
    valid_mask = np.zeros(REGULAR_TURN_ACTIONS, dtype=bool)

    t3_states = []
    t3_indices = []
    for action in actions:
        next_board = apply_placements(board, action.placements)
        action_discards = discards + ([action.discard] if action.discard else [])
        state_indices = []
        for _ in range(n_samples):
            t3_deal = random.sample(rem_deck, 3)
            obs = Observation(
                board_self=next_board,
                board_opponent=Board(),
                dealt_cards=t3_deal,
                known_discards_self=action_discards,
                turn=3,
                is_btn=True,
            )
            state_vec = encode_state(obs)[: model.input_proj[0].in_features]
            state_indices.append(len(t3_states))
            t3_states.append(state_vec)
        t3_indices.append((action, state_indices))

    batch = torch.as_tensor(np.asarray(t3_states, dtype=np.float32), device=device)
    with torch.no_grad():
        _logits, values = model(batch, masks=None)
    values = values.detach().cpu().numpy()

    for action, state_indices in t3_indices:
        idx = get_semantic_action_index(action, deal)
        action_evs[idx] = float(values[state_indices].mean())
        valid_mask[idx] = True

    obs = Observation(
        board_self=board,
        board_opponent=Board(),
        dealt_cards=deal,
        known_discards_self=discards,
        turn=2,
        is_btn=True,
    )
    state = encode_state(obs)[: model.input_proj[0].in_features].astype(np.float16)
    best_idx = int(np.where(valid_mask, action_evs, -1.0e9).argmax())
    best_ev = float(action_evs[best_idx])
    return {
        "states": state,
        "action_evs": action_evs.astype(np.float16),
        "action_masks": valid_mask,
        "valid_masks": valid_mask,
        "best_evs": np.float32(best_ev),
        "actions": np.int64(best_idx),
        "rewards": np.float32(best_ev),
    }


def write_chunk(records, out_dir: Path, chunk_id: int):
    arrays = {k: np.stack([r[k] for r in records], axis=0) for k in records[0]}
    out = out_dir / f"t2_oracle_{chunk_id:06d}.npz"
    np.savez_compressed(out, **arrays)
    return out


def upload_to_gcs(path: Path, gcs_output: str):
    if not gcs_output:
        return
    dest = gcs_output.rstrip("/") + "/" + path.name
    subprocess.run(["gcloud", "storage", "cp", str(path), dest], check=False)


def main():
    parser = argparse.ArgumentParser(description="Generate T2 oracle NPZ")
    parser.add_argument("--source", choices=["t0-jsonl", "rule-random"], default="t0-jsonl")
    parser.add_argument("--t0-jsonl-dir", default="ai/data/t0_full_results_v2")
    parser.add_argument("--t3-model", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--states", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t0-policy", choices=["random", "heuristic"], default="heuristic")
    parser.add_argument("--t1-policy", choices=["random", "heuristic", "first"], default="heuristic")
    parser.add_argument("--branch-non-ace-trips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--gcs-output", default="")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_t3_oracle(args.t3_model, device)

    records = []
    written = 0
    attempts = 0
    chunk_id = 0
    pending_rule_states = []
    t0 = time.time()

    if args.source == "t0-jsonl":
        source_iter = iter_t0_boards(Path(args.t0_jsonl_dir))
    else:
        source_iter = iter(int, 1)

    for source_item in source_iter:
        if written >= args.states:
            break
        if args.max_seconds > 0 and time.time() - t0 >= args.max_seconds:
            break
        attempts += 1
        if args.source == "t0-jsonl":
            t0_board = source_item
            used = set(t0_board.all_cards())
            rem = [c for c in ALL_CARDS if c not in used]
            if len(rem) < 6:
                continue

            t1_deal = random.sample(rem, 3)
            t1_actions = get_turn_actions(t1_deal, t0_board)
            if not t1_actions:
                continue
            t1_action = choose_action(t1_deal, t0_board, turn=1, policy=args.t1_policy)
            t1_board = apply_placements(t0_board, t1_action.placements)
            discards = [t1_action.discard] if t1_action.discard else []
            used.update(t1_deal)
            rem = [c for c in ALL_CARDS if c not in used]
            if len(rem) < 3:
                continue

            t2_deal = random.sample(rem, 3)
        else:
            if not pending_rule_states:
                pending_rule_states = generate_rule_random_t1_states(
                    args.t0_policy,
                    args.t1_policy,
                    args.branch_non_ace_trips,
                )
            t1_board, discards, t2_deal = pending_rule_states.pop(0)

        rec = evaluate_t2_state(t1_board, t2_deal, discards, model, device, args.n_samples)
        if rec is None:
            continue
        records.append(rec)
        written += 1

        if len(records) >= args.chunk_size:
            out = write_chunk(records, out_dir, chunk_id)
            print(f"wrote {out} records={len(records)} total={written}")
            upload_to_gcs(out, args.gcs_output)
            records.clear()
            chunk_id += 1

        if written % 100 == 0:
            elapsed = time.time() - t0
            print(f"written={written} attempts={attempts} speed={written / max(elapsed, 1e-6):.2f} states/s")

    if records:
        out = write_chunk(records, out_dir, chunk_id)
        print(f"wrote {out} records={len(records)} total={written}")
        upload_to_gcs(out, args.gcs_output)

    print(f"done written={written} attempts={attempts} elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
