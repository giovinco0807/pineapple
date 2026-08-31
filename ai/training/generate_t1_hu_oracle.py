#!/usr/bin/env python3
"""Generate heads-up T1 oracle data using T2 BB/BTN value models."""
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
    create_regular_turn_mask,
    get_action_from_semantic_index,
    get_semantic_action_index,
    get_turn_actions,
)
from ai.engine.encoding import ALL_CARDS, Board, Observation, encode_state  # noqa: E402
from ai.training.generate_data import select_action  # noqa: E402
from ai.training.generate_t2_oracle_data import apply_placements  # noqa: E402
from ai.training.train_t2_oracle import T2PolicyValueNet  # noqa: E402


def draw(deck: list[str], n: int) -> list[str]:
    return [deck.pop() for _ in range(n)]


def load_t2_model(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dim = ckpt.get("state_dim", 522)
    n_actions = ckpt.get("n_actions", 27)
    hidden = ckpt.get("hidden", 1024)
    n_blocks = ckpt.get("n_blocks", 4)
    model = T2PolicyValueNet(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=hidden,
        n_blocks=n_blocks,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def apply_regular_turn(board: Board, discards: list[str], dealt_cards: list[str], turn: int, policy: str):
    if policy == "random":
        actions = get_turn_actions(dealt_cards, board)
        action = random.choice(actions) if actions else None
    else:
        action = select_action(dealt_cards, board, turn=turn, strategy=policy)
    if action is None:
        return None, discards
    next_board = apply_placements(board, action.placements)
    next_discards = list(discards)
    if action.discard:
        next_discards.append(action.discard)
    return next_board, next_discards


def choose_t2_model_action(model, board: Board, opponent_board: Board, deal: list[str], discards: list[str], is_btn: bool, device: str):
    mask = create_regular_turn_mask(deal, board)
    obs = Observation(
        board_self=board,
        board_opponent=opponent_board,
        dealt_cards=deal,
        known_discards_self=discards,
        turn=2,
        is_btn=is_btn,
    )
    state = encode_state(obs)[: model.input_proj[0].in_features]
    with torch.no_grad():
        states_t = torch.as_tensor(state[None, :], dtype=torch.float32, device=device)
        masks_t = torch.as_tensor(mask[None, :], dtype=torch.bool, device=device)
        logits, _value = model(states_t, masks_t)
        idx = int(torch.argmax(logits, dim=-1).item())
    return get_action_from_semantic_index(idx, deal)


def choose_t2_model_actions_batch(model, items: list[tuple[Board, Board, list[str], list[str], bool]], device: str):
    """Choose T2 actions for many independent states in one model call."""
    if not items:
        return []
    states = []
    masks = []
    for board, opponent_board, deal, discards, is_btn in items:
        mask = create_regular_turn_mask(deal, board)
        obs = Observation(
            board_self=board,
            board_opponent=opponent_board,
            dealt_cards=deal,
            known_discards_self=discards,
            turn=2,
            is_btn=is_btn,
        )
        states.append(encode_state(obs)[: model.input_proj[0].in_features])
        masks.append(mask)
    with torch.no_grad():
        states_t = torch.as_tensor(np.asarray(states, dtype=np.float32), device=device)
        masks_t = torch.as_tensor(np.asarray(masks, dtype=bool), device=device)
        logits, _value = model(states_t, masks_t)
        idxs = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
    return [
        get_action_from_semantic_index(int(idx), deal)
        for idx, (_board, _opponent_board, deal, _discards, _is_btn) in zip(idxs, items)
    ]


def build_t1_hu_state(position: str, t0_policy: str, t1_opp_policy: str):
    deck = list(ALL_CARDS)
    random.shuffle(deck)

    bb_t0_deal = draw(deck, 5)
    btn_t0_deal = draw(deck, 5)
    bb_board = apply_placements(Board(), select_action(bb_t0_deal, Board(), turn=0, strategy=t0_policy).placements)
    btn_board = apply_placements(Board(), select_action(btn_t0_deal, Board(), turn=0, strategy=t0_policy).placements)
    bb_discards: list[str] = []
    btn_discards: list[str] = []

    bb_t1_deal = draw(deck, 3)
    if position == "bb":
        return {
            "board": bb_board,
            "opponent_board": btn_board,
            "deal": bb_t1_deal,
            "discards": bb_discards,
            "opp_discards": btn_discards,
            "is_btn": False,
            "deck": deck,
        }

    bb_board, bb_discards = apply_regular_turn(bb_board, bb_discards, bb_t1_deal, 1, t1_opp_policy)
    if bb_board is None:
        return None
    btn_t1_deal = draw(deck, 3)
    return {
        "board": btn_board,
        "opponent_board": bb_board,
        "deal": btn_t1_deal,
        "discards": btn_discards,
        "opp_discards": bb_discards,
        "is_btn": True,
        "deck": deck,
    }


def evaluate_t1_state(
    board: Board,
    opponent_board: Board,
    deal: list[str],
    discards: list[str],
    opp_discards: list[str],
    is_btn: bool,
    t2_bb_model,
    t2_btn_model,
    device: str,
    n_samples: int,
    t1_opp_policy: str,
):
    actions = get_turn_actions(deal, board)
    if not actions:
        return None

    action_evs = np.full(REGULAR_TURN_ACTIONS, -1.0e4, dtype=np.float32)
    valid_mask = np.zeros(REGULAR_TURN_ACTIONS, dtype=bool)
    value_states = []
    action_state_indices = []

    for action in actions:
        next_board = apply_placements(board, action.placements)
        next_discards = discards + ([action.discard] if action.discard else [])
        state_indices = []

        if is_btn:
            # BTN T1 -> choose many BB T2 actions in one T2-BB batch, then
            # evaluate the resulting BTN T2 states in one T2-BTN batch later.
            pending_bb_items = []
            pending_rem2 = []
            pending_bb_deals = []
            for _ in range(n_samples):
                used = set(next_board.all_cards() + opponent_board.all_cards() + deal + next_discards + opp_discards)
                rem = [c for c in ALL_CARDS if c not in used]
                if len(rem) < 6:
                    continue
                bb_t2_deal = random.sample(rem, 3)
                pending_bb_items.append((opponent_board, next_board, bb_t2_deal, list(opp_discards), False))
                pending_bb_deals.append(bb_t2_deal)

            bb_t2_actions = choose_t2_model_actions_batch(t2_bb_model, pending_bb_items, device)
            for bb_t2_deal, bb_t2_action in zip(pending_bb_deals, bb_t2_actions):
                bb_board = apply_placements(opponent_board, bb_t2_action.placements)
                bb_discards = list(opp_discards)
                if bb_t2_action.discard:
                    bb_discards.append(bb_t2_action.discard)
                used2 = set(next_board.all_cards() + bb_board.all_cards() + next_discards + bb_discards + bb_t2_deal)
                rem2 = [c for c in ALL_CARDS if c not in used2]
                if len(rem2) < 3:
                    continue
                t2_deal = random.sample(rem2, 3)
                obs = Observation(
                    board_self=next_board,
                    board_opponent=bb_board,
                    dealt_cards=t2_deal,
                    known_discards_self=next_discards,
                    turn=2,
                    is_btn=True,
                )
                state_vec = encode_state(obs)[: t2_btn_model.input_proj[0].in_features]
                state_indices.append(len(value_states))
                value_states.append((state_vec, t2_btn_model))
            action_state_indices.append((action, state_indices))
            continue

        for _ in range(n_samples):
            used = set(next_board.all_cards() + opponent_board.all_cards() + deal + next_discards + opp_discards)
            rem = [c for c in ALL_CARDS if c not in used]
            if len(rem) < 6:
                continue

            # BB T1 -> sample BTN T1 -> evaluate BB T2 value.
            btn_t1_deal = random.sample(rem, 3)
            btn_board, btn_discards = apply_regular_turn(
                opponent_board,
                list(opp_discards),
                btn_t1_deal,
                1,
                t1_opp_policy,
            )
            if btn_board is None:
                continue
            used2 = set(next_board.all_cards() + btn_board.all_cards() + next_discards + btn_discards + btn_t1_deal)
            rem2 = [c for c in ALL_CARDS if c not in used2]
            if len(rem2) < 3:
                continue
            t2_deal = random.sample(rem2, 3)
            obs = Observation(
                board_self=next_board,
                board_opponent=btn_board,
                dealt_cards=t2_deal,
                known_discards_self=next_discards,
                turn=2,
                is_btn=False,
            )
            model = t2_bb_model

            state_vec = encode_state(obs)[: model.input_proj[0].in_features]
            state_indices.append(len(value_states))
            value_states.append((state_vec, model))

        action_state_indices.append((action, state_indices))

    if not value_states:
        return None

    # Evaluate by model group to avoid mixing BB/BTN models in one batch.
    values = np.zeros(len(value_states), dtype=np.float32)
    for model in (t2_bb_model, t2_btn_model):
        idxs = [i for i, (_state, m) in enumerate(value_states) if m is model]
        if not idxs:
            continue
        batch = np.asarray([value_states[i][0] for i in idxs], dtype=np.float32)
        with torch.no_grad():
            _logits, v = model(torch.as_tensor(batch, device=device), masks=None)
        values[idxs] = v.detach().cpu().numpy()

    for action, state_indices in action_state_indices:
        if not state_indices:
            continue
        idx = get_semantic_action_index(action, deal)
        action_evs[idx] = float(values[state_indices].mean())
        valid_mask[idx] = True

    if not valid_mask.any():
        return None

    obs = Observation(
        board_self=board,
        board_opponent=opponent_board,
        dealt_cards=deal,
        known_discards_self=discards,
        turn=1,
        is_btn=is_btn,
    )
    state = encode_state(obs)[: t2_bb_model.input_proj[0].in_features].astype(np.float16)
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
    out = out_dir / f"t1_hu_oracle_{chunk_id:06d}.npz"
    np.savez_compressed(out, **arrays)
    return out


def upload_to_gcs(path: Path, gcs_output: str):
    if not gcs_output:
        return
    dest = gcs_output.rstrip("/") + "/" + path.name
    subprocess.run(["gcloud", "storage", "cp", str(path), dest], check=False)


def main():
    parser = argparse.ArgumentParser(description="Generate HU T1 oracle NPZ")
    parser.add_argument("--position", choices=["bb", "btn", "mixed"], default="bb")
    parser.add_argument("--t2-bb-model", required=True)
    parser.add_argument("--t2-btn-model", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--states", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t0-policy", choices=["random", "heuristic"], default="heuristic")
    parser.add_argument("--t1-opp-policy", choices=["random", "heuristic"], default="heuristic")
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
    t2_bb_model = load_t2_model(args.t2_bb_model, device)
    t2_btn_model = load_t2_model(args.t2_btn_model, device)

    records = []
    written = 0
    attempts = 0
    chunk_id = 0
    t0 = time.time()

    while written < args.states:
        if args.max_seconds > 0 and time.time() - t0 >= args.max_seconds:
            break
        attempts += 1
        position = random.choice(["bb", "btn"]) if args.position == "mixed" else args.position
        state = build_t1_hu_state(position, args.t0_policy, args.t1_opp_policy)
        if state is None:
            continue
        rec = evaluate_t1_state(
            state["board"],
            state["opponent_board"],
            state["deal"],
            state["discards"],
            state["opp_discards"],
            state["is_btn"],
            t2_bb_model,
            t2_btn_model,
            device,
            args.n_samples,
            args.t1_opp_policy,
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
