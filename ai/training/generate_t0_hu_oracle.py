#!/usr/bin/env python3
"""Generate heads-up T0 oracle data using T1 BB/BTN value models.

This is the first bootstrapping pass for T0.  BB acts first:

* BB T0: evaluate each initial placement by sampling BTN T0 responses, then
  valuing the resulting BB T1 state with the T1-BB model.
* BTN T0: BB T0 is already on board; evaluate each placement by sampling BB T1
  responses chosen by the T1-BB model, then valuing BTN T1 with the T1-BTN model.
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
    MAX_ACTIONS,
    create_regular_turn_mask,
    get_action_from_semantic_index,
    get_initial_actions,
)
from ai.engine.encoding import ALL_CARDS, Board, Observation, encode_state  # noqa: E402
from ai.training.generate_data import select_action  # noqa: E402
from ai.training.generate_t2_oracle_data import apply_placements  # noqa: E402
from ai.training.train_t2_oracle import T2PolicyValueNet  # noqa: E402


def draw(deck: list[str], n: int) -> list[str]:
    return [deck.pop() for _ in range(n)]


def load_model(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = T2PolicyValueNet(
        state_dim=ckpt.get("state_dim", 522),
        n_actions=ckpt.get("n_actions", 27),
        hidden=ckpt.get("hidden", 1024),
        n_blocks=ckpt.get("n_blocks", 4),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def choose_regular_model_action(model, board: Board, opponent_board: Board, deal: list[str], discards: list[str], is_btn: bool, device: str):
    mask = create_regular_turn_mask(deal, board)
    obs = Observation(
        board_self=board,
        board_opponent=opponent_board,
        dealt_cards=deal,
        known_discards_self=discards,
        turn=1,
        is_btn=is_btn,
    )
    state = encode_state(obs)[: model.input_proj[0].in_features]
    with torch.no_grad():
        states_t = torch.as_tensor(state[None, :], dtype=torch.float32, device=device)
        masks_t = torch.as_tensor(mask[None, :], dtype=torch.bool, device=device)
        logits, _value = model(states_t, masks_t)
        idx = int(torch.argmax(logits, dim=-1).item())
    return get_action_from_semantic_index(idx, deal)


def choose_regular_model_actions_batch(model, items: list[tuple[Board, Board, list[str], list[str], bool]], device: str):
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
            turn=1,
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


def choose_initial_action(
    deal: list[str],
    is_btn: bool,
    model,
    device: str,
    policy: str,
    opponent_board: Board | None = None,
):
    if policy != "model":
        return select_action(deal, Board(), turn=0, strategy=policy)
    if model is None:
        raise ValueError("T0 model policy requested but no model was provided")
    opponent_board = opponent_board or Board()
    actions = get_initial_actions(deal, Board())
    mask = np.zeros(MAX_ACTIONS, dtype=bool)
    mask[: min(len(actions), MAX_ACTIONS)] = True
    obs = Observation(
        board_self=Board(),
        board_opponent=opponent_board,
        dealt_cards=deal,
        known_discards_self=[],
        turn=0,
        is_btn=is_btn,
    )
    state = encode_state(obs)[: model.input_proj[0].in_features]
    with torch.no_grad():
        states_t = torch.as_tensor(state[None, :], dtype=torch.float32, device=device)
        masks_t = torch.as_tensor(mask[None, :], dtype=torch.bool, device=device)
        logits, _value = model(states_t, masks_t)
        idx = int(torch.argmax(logits, dim=-1).item())
    if idx >= len(actions):
        idx = int(np.flatnonzero(mask)[0])
    return actions[idx]


def choose_initial_actions_batch(
    items: list[tuple[list[str], bool, Board]],
    model,
    device: str,
):
    if not items:
        return []
    states = []
    masks = []
    action_lists = []
    for deal, is_btn, opponent_board in items:
        actions = get_initial_actions(deal, Board())
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        mask[: min(len(actions), MAX_ACTIONS)] = True
        obs = Observation(
            board_self=Board(),
            board_opponent=opponent_board,
            dealt_cards=deal,
            known_discards_self=[],
            turn=0,
            is_btn=is_btn,
        )
        states.append(encode_state(obs)[: model.input_proj[0].in_features])
        masks.append(mask)
        action_lists.append(actions)
    with torch.no_grad():
        states_t = torch.as_tensor(np.asarray(states, dtype=np.float32), device=device)
        masks_t = torch.as_tensor(np.asarray(masks, dtype=bool), device=device)
        logits, _value = model(states_t, masks_t)
        idxs = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
    result = []
    for idx, actions in zip(idxs, action_lists):
        if idx >= len(actions):
            idx = 0
        result.append(actions[int(idx)])
    return result


def value_states(model, states: list[np.ndarray], device: str) -> np.ndarray:
    if not states:
        return np.zeros(0, dtype=np.float32)
    batch = np.asarray(states, dtype=np.float32)
    values = []
    with torch.no_grad():
        for i in range(0, len(batch), 4096):
            x = torch.as_tensor(batch[i:i + 4096], device=device)
            _logits, v = model(x, masks=None)
            values.append(v.detach().cpu().numpy())
    return np.concatenate(values, axis=0).astype(np.float32)


def build_t0_hu_state(position: str, t0_bb_policy: str, t0_bb_model, device: str):
    deck = list(ALL_CARDS)
    random.shuffle(deck)

    bb_deal = draw(deck, 5)
    if position == "bb":
        return {
            "board": Board(),
            "opponent_board": Board(),
            "deal": bb_deal,
            "is_btn": False,
            "deck": deck,
        }

    bb_action = choose_initial_action(bb_deal, False, t0_bb_model, device, t0_bb_policy)
    bb_board = apply_placements(Board(), bb_action.placements)
    btn_deal = draw(deck, 5)
    return {
        "board": Board(),
        "opponent_board": bb_board,
        "deal": btn_deal,
        "is_btn": True,
        "deck": deck,
    }


def evaluate_t0_bb(
    deal: list[str],
    t1_bb_model,
    device: str,
    n_samples: int,
    t0_opp_policy: str,
    t0_opp_model,
):
    actions = get_initial_actions(deal, Board())
    action_evs = np.full(MAX_ACTIONS, -1.0e4, dtype=np.float32)
    valid_mask = np.zeros(MAX_ACTIONS, dtype=bool)
    value_inputs = []
    action_state_indices: list[tuple[int, list[int]]] = []
    pending_model_responses = []
    pending_model_meta = []

    for action_idx, action in enumerate(actions[:MAX_ACTIONS]):
        bb_board = apply_placements(Board(), action.placements)
        state_indices = []
        for _ in range(n_samples):
            used = set(deal)
            rem = [c for c in ALL_CARDS if c not in used]
            btn_t0_deal = random.sample(rem, 5)
            if t0_opp_policy == "model":
                pending_model_responses.append((btn_t0_deal, True, bb_board))
                pending_model_meta.append((action_idx, state_indices, btn_t0_deal, bb_board))
                continue
            btn_action = choose_initial_action(btn_t0_deal, True, t0_opp_model, device, t0_opp_policy, bb_board)
            btn_board = apply_placements(Board(), btn_action.placements)
            used2 = set(deal + btn_t0_deal)
            rem2 = [c for c in ALL_CARDS if c not in used2]
            if len(rem2) < 3:
                continue
            bb_t1_deal = random.sample(rem2, 3)
            obs = Observation(
                board_self=bb_board,
                board_opponent=btn_board,
                dealt_cards=bb_t1_deal,
                known_discards_self=[],
                turn=1,
                is_btn=False,
            )
            state_indices.append(len(value_inputs))
            value_inputs.append(encode_state(obs)[: t1_bb_model.input_proj[0].in_features])
        action_state_indices.append((action_idx, state_indices))

    if pending_model_responses:
        if t0_opp_model is None:
            raise ValueError("T0 opponent model policy requested but no model was provided")
        btn_actions = choose_initial_actions_batch(pending_model_responses, t0_opp_model, device)
        for (_action_idx, state_indices, btn_t0_deal, bb_board), btn_action in zip(pending_model_meta, btn_actions):
            btn_board = apply_placements(Board(), btn_action.placements)
            used2 = set(deal + btn_t0_deal)
            rem2 = [c for c in ALL_CARDS if c not in used2]
            if len(rem2) < 3:
                continue
            bb_t1_deal = random.sample(rem2, 3)
            obs = Observation(
                board_self=bb_board,
                board_opponent=btn_board,
                dealt_cards=bb_t1_deal,
                known_discards_self=[],
                turn=1,
                is_btn=False,
            )
            state_indices.append(len(value_inputs))
            value_inputs.append(encode_state(obs)[: t1_bb_model.input_proj[0].in_features])

    values = value_states(t1_bb_model, value_inputs, device)
    for action_idx, state_indices in action_state_indices:
        if not state_indices:
            continue
        action_evs[action_idx] = float(values[state_indices].mean())
        valid_mask[action_idx] = True

    return action_evs, valid_mask


def evaluate_t0_btn(
    bb_board: Board,
    deal: list[str],
    t1_bb_model,
    t1_btn_model,
    device: str,
    n_samples: int,
):
    actions = get_initial_actions(deal, Board())
    action_evs = np.full(MAX_ACTIONS, -1.0e4, dtype=np.float32)
    valid_mask = np.zeros(MAX_ACTIONS, dtype=bool)
    pending_choices = []
    pending_meta = []

    for action_idx, action in enumerate(actions[:MAX_ACTIONS]):
        btn_board = apply_placements(Board(), action.placements)
        for _ in range(n_samples):
            used = set(bb_board.all_cards() + deal)
            rem = [c for c in ALL_CARDS if c not in used]
            if len(rem) < 6:
                continue
            bb_t1_deal = random.sample(rem, 3)
            pending_choices.append((
                bb_board,
                btn_board,
                bb_t1_deal,
                [],
                False,
            ))
            pending_meta.append((action_idx, btn_board, bb_t1_deal))

    bb_t1_actions = choose_regular_model_actions_batch(t1_bb_model, pending_choices, device)
    value_inputs = []
    action_state_indices: dict[int, list[int]] = {i: [] for i in range(min(len(actions), MAX_ACTIONS))}
    for (action_idx, btn_board, bb_t1_deal), bb_t1_action in zip(pending_meta, bb_t1_actions):
            bb_next = apply_placements(bb_board, bb_t1_action.placements)
            bb_discards = [bb_t1_action.discard] if bb_t1_action.discard else []
            used2 = set(bb_next.all_cards() + btn_board.all_cards() + deal + bb_t1_deal + bb_discards)
            rem2 = [c for c in ALL_CARDS if c not in used2]
            if len(rem2) < 3:
                continue
            btn_t1_deal = random.sample(rem2, 3)
            obs = Observation(
                board_self=btn_board,
                board_opponent=bb_next,
                dealt_cards=btn_t1_deal,
                known_discards_self=[],
                turn=1,
                is_btn=True,
            )
            action_state_indices[action_idx].append(len(value_inputs))
            value_inputs.append(encode_state(obs)[: t1_btn_model.input_proj[0].in_features])

    values = value_states(t1_btn_model, value_inputs, device)
    for action_idx, state_indices in action_state_indices.items():
        if not state_indices:
            continue
        action_evs[action_idx] = float(values[state_indices].mean())
        valid_mask[action_idx] = True

    return action_evs, valid_mask


def evaluate_t0_state(
    state: dict,
    t1_bb_model,
    t1_btn_model,
    device: str,
    n_samples: int,
    t0_opp_policy: str,
    t0_opp_model,
):
    if state["is_btn"]:
        action_evs, valid_mask = evaluate_t0_btn(
            state["opponent_board"],
            state["deal"],
            t1_bb_model,
            t1_btn_model,
            device,
            n_samples,
        )
    else:
        action_evs, valid_mask = evaluate_t0_bb(
            state["deal"],
            t1_bb_model,
            device,
            n_samples,
            t0_opp_policy,
            t0_opp_model,
        )

    if not valid_mask.any():
        return None

    obs = Observation(
        board_self=state["board"],
        board_opponent=state["opponent_board"],
        dealt_cards=state["deal"],
        known_discards_self=[],
        turn=0,
        is_btn=state["is_btn"],
    )
    state_vec = encode_state(obs)[: t1_bb_model.input_proj[0].in_features].astype(np.float16)
    best_idx = int(np.where(valid_mask, action_evs, -1.0e9).argmax())
    best_ev = float(action_evs[best_idx])
    return {
        "states": state_vec,
        "action_evs": action_evs.astype(np.float16),
        "action_masks": valid_mask,
        "valid_masks": valid_mask,
        "best_evs": np.float32(best_ev),
        "actions": np.int64(best_idx),
        "rewards": np.float32(best_ev),
        "is_btn": np.bool_(state["is_btn"]),
    }


def write_chunk(records, out_dir: Path, chunk_id: int):
    arrays = {k: np.stack([r[k] for r in records], axis=0) for k in records[0]}
    out = out_dir / f"t0_hu_oracle_{chunk_id:06d}.npz"
    np.savez_compressed(out, **arrays)
    return out


def upload_to_gcs(path: Path, gcs_output: str):
    if not gcs_output:
        return
    dest = gcs_output.rstrip("/") + "/" + path.name
    subprocess.run(["gcloud", "storage", "cp", str(path), dest], check=False)


def main():
    parser = argparse.ArgumentParser(description="Generate HU T0 oracle NPZ")
    parser.add_argument("--position", choices=["bb", "btn", "mixed"], default="bb")
    parser.add_argument("--t1-bb-model", required=True)
    parser.add_argument("--t1-btn-model", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--states", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t0-bb-policy", choices=["random", "heuristic", "model"], default="heuristic")
    parser.add_argument("--t0-opp-policy", choices=["random", "heuristic", "model"], default="heuristic")
    parser.add_argument("--t0-bb-model", default=None)
    parser.add_argument("--t0-opp-model", default=None)
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
    t1_bb_model = load_model(args.t1_bb_model, device)
    t1_btn_model = load_model(args.t1_btn_model, device)
    t0_bb_model = load_model(args.t0_bb_model, device) if args.t0_bb_model else None
    t0_opp_model = load_model(args.t0_opp_model, device) if args.t0_opp_model else None

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
        state = build_t0_hu_state(position, args.t0_bb_policy, t0_bb_model, device)
        rec = evaluate_t0_state(
            state,
            t1_bb_model,
            t1_btn_model,
            device,
            args.n_samples,
            args.t0_opp_policy,
            t0_opp_model,
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

        if written % 10 == 0:
            elapsed = time.time() - t0
            print(f"written={written} attempts={attempts} speed={written / max(elapsed, 1e-6):.2f} states/s")

    if records:
        out = write_chunk(records, out_dir, chunk_id)
        print(f"wrote {out} records={len(records)} total={written}")
        upload_to_gcs(out, args.gcs_output)
    print(f"done written={written} attempts={attempts} elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
