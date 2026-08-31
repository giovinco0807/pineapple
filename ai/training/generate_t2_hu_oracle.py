#!/usr/bin/env python3
"""Generate heads-up T2 oracle data with opponent board and BTN/BB position.

The older T2 generator creates a useful single-board bootstrap dataset. This
script creates real heads-up decision states:

* BB T2: BB acts first with BTN visible after BTN T1.
* BTN T2: BB has already acted on T2, then BTN acts with BB T2 board visible.

Each candidate T2 action is evaluated by sampling T3 deals and reading the T3
value head, preserving the acting seat's position flag and visible opponent
board.
"""
import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai.engine.action_space import (  # noqa: E402
    REGULAR_TURN_ACTIONS,
    get_semantic_action_index,
    get_turn_actions,
)
from ai.engine.encoding import ALL_CARDS, Board, Observation, encode_state  # noqa: E402
from ai.training.generate_data import select_action  # noqa: E402
from ai.training.generate_t2_oracle_data import apply_placements, load_t3_oracle  # noqa: E402


def draw(deck: list[str], n: int) -> list[str]:
    return [deck.pop() for _ in range(n)]


def choose_turn_action(dealt_cards: list[str], board: Board, turn: int, policy: str):
    if policy == "random":
        actions = get_turn_actions(dealt_cards, board)
        return random.choice(actions) if actions else None
    return select_action(dealt_cards, board, turn=turn, strategy=policy)


def apply_regular_turn(board: Board, discards: list[str], dealt_cards: list[str], turn: int, policy: str):
    action = choose_turn_action(dealt_cards, board, turn, policy)
    if action is None:
        return None, discards
    next_board = apply_placements(board, action.placements)
    next_discards = list(discards)
    if action.discard:
        next_discards.append(action.discard)
    return next_board, next_discards


def build_t2_hu_state(position: str, t0_policy: str, t1_policy: str, t2_opp_policy: str):
    deck = list(ALL_CARDS)
    random.shuffle(deck)

    bb_t0_deal = draw(deck, 5)
    btn_t0_deal = draw(deck, 5)
    bb_board = apply_placements(Board(), select_action(bb_t0_deal, Board(), turn=0, strategy=t0_policy).placements)
    btn_board = apply_placements(Board(), select_action(btn_t0_deal, Board(), turn=0, strategy=t0_policy).placements)
    btn_discards: list[str] = []
    bb_discards: list[str] = []

    bb_t1_deal = draw(deck, 3)
    bb_board, bb_discards = apply_regular_turn(bb_board, bb_discards, bb_t1_deal, 1, t1_policy)
    if bb_board is None:
        return None

    btn_t1_deal = draw(deck, 3)
    btn_board, btn_discards = apply_regular_turn(btn_board, btn_discards, btn_t1_deal, 1, t1_policy)
    if btn_board is None:
        return None

    bb_t2_deal = draw(deck, 3)
    if position == "bb":
        return {
            "board": bb_board,
            "opponent_board": btn_board,
            "deal": bb_t2_deal,
            "discards": bb_discards,
            "is_btn": False,
            "deck": deck,
        }

    bb_board, bb_discards = apply_regular_turn(bb_board, bb_discards, bb_t2_deal, 2, t2_opp_policy)
    if bb_board is None:
        return None

    btn_t2_deal = draw(deck, 3)
    return {
        "board": btn_board,
        "opponent_board": bb_board,
        "deal": btn_t2_deal,
        "discards": btn_discards,
        "is_btn": True,
        "deck": deck,
    }


def evaluate_t2_state(
    board: Board,
    opponent_board: Board,
    deal: list[str],
    discards: list[str],
    is_btn: bool,
    model,
    device,
    n_samples: int,
):
    actions = get_turn_actions(deal, board)
    if not actions:
        return None

    visible = set(board.all_cards() + opponent_board.all_cards() + deal + discards)
    rem_deck = [c for c in ALL_CARDS if c not in visible]
    if len(rem_deck) < 3:
        return None

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
                board_opponent=opponent_board,
                dealt_cards=t3_deal,
                known_discards_self=action_discards,
                turn=3,
                is_btn=is_btn,
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
        board_opponent=opponent_board,
        dealt_cards=deal,
        known_discards_self=discards,
        turn=2,
        is_btn=is_btn,
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
        "is_btn": np.bool_(is_btn),
    }


def write_chunk(records, out_dir: Path, chunk_id: int):
    arrays = {k: np.stack([r[k] for r in records], axis=0) for k in records[0]}
    out = out_dir / f"t2_hu_oracle_{chunk_id:06d}.npz"
    np.savez_compressed(out, **arrays)
    return out


def upload_to_gcs(path: Path, gcs_output: str):
    if not gcs_output:
        return
    dest = gcs_output.rstrip("/") + "/" + path.name
    subprocess.run(["gcloud", "storage", "cp", str(path), dest], check=False)


def main():
    parser = argparse.ArgumentParser(description="Generate HU T2 oracle NPZ")
    parser.add_argument("--position", choices=["btn", "bb", "mixed"], default="bb")
    parser.add_argument("--t3-model", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--states", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t0-policy", choices=["random", "heuristic"], default="heuristic")
    parser.add_argument("--t1-policy", choices=["random", "heuristic"], default="heuristic")
    parser.add_argument("--t2-opp-policy", choices=["random", "heuristic"], default="heuristic")
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
    t0 = time.time()

    while written < args.states:
        if args.max_seconds > 0 and time.time() - t0 >= args.max_seconds:
            break
        attempts += 1
        position = random.choice(["btn", "bb"]) if args.position == "mixed" else args.position
        state = build_t2_hu_state(position, args.t0_policy, args.t1_policy, args.t2_opp_policy)
        if state is None:
            continue
        rec = evaluate_t2_state(
            state["board"],
            state["opponent_board"],
            state["deal"],
            state["discards"],
            state["is_btn"],
            model,
            device,
            args.n_samples,
        )
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
