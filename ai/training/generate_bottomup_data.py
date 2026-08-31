"""
Bottom-up per-turn BC training data generation.

For T2: uses T3 BC + T4 BC for playout, evaluates all T2 actions with N3 samples.
For T1: uses T2 BC + T3 BC + T4 BC for playout, evaluates all T1 actions with N2 samples.

GPU batch inference for speed: collects all playout states, runs BC in batch,
then unpacks results.

Usage:
    # T2 data (requires T3+T4 BC models)
    python -m ai.training.generate_bottomup_data \
        --turn 2 --n-samples 30 --n-quick 3 \
        --bc-t3 ai/models/bc_t3/bc_policy_best.pt \
        --bc-t4 ai/models/bc_t4/bc_policy_best.pt \
        --json-dir data/expectimax_results_v3 \
        --save data/t2_bottomup

    # T1 data (requires T2+T3+T4 BC models)
    python -m ai.training.generate_bottomup_data \
        --turn 1 --n-samples 50 --n-quick 3 \
        --bc-t2 ai/models/bc_t2/bc_policy_best.pt \
        --bc-t3 ai/models/bc_t3/bc_policy_best.pt \
        --bc-t4 ai/models/bc_t4/bc_policy_best.pt \
        --json-dir data/expectimax_results_v3 \
        --save data/t1_bottomup
"""
import sys
import json
import random
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.engine.encoding import (
    Board, Observation, ALL_CARDS, encode_state, STATE_DIM
)
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, create_action_mask,
    get_semantic_action_index
)
from ai.engine.game_engine import Hand, evaluate_hand
from ai.models.networks import PolicyNetwork

MAX_ACTIONS = 250
ROW_MAP = {"T": "top", "M": "middle", "B": "bottom"}
ARROW = "\u2192"


def load_bc_model(path, device='cpu'):
    """Load a BC policy model."""
    model = PolicyNetwork()
    ck = torch.load(path, map_location=device, weights_only=True)
    sd = ck.get('model_state_dict', ck)
    model.load_state_dict(sd)
    model.eval()
    model.to(device)
    return model


def parse_turn_action_desc(desc):
    """Parse 'd:Qs 6h→B 6c→B' → (discard, [(card, pos), ...])"""
    parts = desc.split()
    discard = parts[0][2:]
    placements = []
    for part in parts[1:]:
        card, row_code = part.split(ARROW)
        placements.append((card, ROW_MAP[row_code]))
    return discard, placements


def parse_t0_action_desc(desc):
    """Parse 'Ah→M Kh→T ...' → [(card, pos), ...]"""
    placements = []
    for part in desc.split():
        card, row_code = part.split(ARROW)
        placements.append((card, ROW_MAP[row_code]))
    return placements


def board_from_json(top, mid, bot):
    """Create Board from JSON arrays."""
    return Board(top=list(top), middle=list(mid), bottom=list(bot))


def apply_placements(board, placements):
    """Apply placements to board, return new board."""
    b = Board(
        top=list(board.top),
        middle=list(board.middle),
        bottom=list(board.bottom),
    )
    for card, pos in placements:
        getattr(b, pos).append(card)
    return b


def board_cards(board):
    """Get all cards on a board."""
    return board.top + board.middle + board.bottom


def remaining_deck(board, deal, discards=None):
    """Compute remaining deck cards."""
    used = set(board_cards(board) + deal)
    if discards:
        used.update(discards)
    return [c for c in ALL_CARDS if c not in used]


def bc_greedy_action(model, obs, actions, device='cpu'):
    """Pick action greedily using BC model."""
    if len(actions) <= 1:
        return actions[0] if actions else None
    state = encode_state(obs)
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    mask = create_action_mask(actions, turn=obs.turn, dealt_cards=obs.dealt_cards)
    mask_t = torch.BoolTensor(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(state_t, mask_t).squeeze(0).cpu().numpy()
    best_action = max(
        actions,
        key=lambda a: probs[get_semantic_action_index(a, obs.dealt_cards)]
    )
    return best_action


def evaluate_final_board(board):
    """Evaluate a complete board: royalties + bust check + FL."""
    from ai.engine.game_engine import (
        get_top_royalty, get_middle_royalty, get_bottom_royalty,
        check_fl_entry, evaluate_board_with_joker_constraint
    )
    
    eval_res = evaluate_board_with_joker_constraint(board.top, board.middle, board.bottom)

    if eval_res["busted"]:
        return -6.0  # bust penalty

    # Royalties
    top_r = get_top_royalty(eval_res["top"])
    mid_r = get_middle_royalty(eval_res["middle"])
    bot_r = get_bottom_royalty(eval_res["bottom"])
    total = top_r + mid_r + bot_r

    # FL bonus
    fl_ev_map = {14: 14.0, 15: 27.9, 16: 52.4, 17: 104.5}
    fl_entry, fl_cards = check_fl_entry(eval_res["top"])
    if fl_entry and fl_cards > 0:
        total += fl_ev_map.get(fl_cards, 0.0)

    return float(total)


def playout_from_board(board, remaining, turn, bc_models, device='cpu',
                       discards=None):
    """Play out from a board state to completion using per-turn BC models.

    Args:
        board: Current board state
        remaining: Remaining deck cards
        turn: Current turn (the board has already been updated for this turn)
        bc_models: dict {turn_num: PolicyNetwork}
        device: torch device
        discards: known discards so far

    Returns:
        Final board score (royalties + FL - bust penalty)
    """
    current_board = Board(
        top=list(board.top),
        middle=list(board.middle),
        bottom=list(board.bottom),
    )
    rem = list(remaining)
    disc = list(discards) if discards else []

    # Play remaining turns
    for t in range(turn, 5):  # turns 1-4
        if current_board.card_count() >= 13:
            break

        # Sample 3 cards
        if len(rem) < 3:
            break
        deal = random.sample(rem, 3)
        for c in deal:
            rem.remove(c)

        actions = get_turn_actions(deal, current_board)
        if not actions:
            break

        if len(actions) == 1:
            chosen = actions[0]
        elif t == 4:
            # T4: enumerate all placements, pick best by evaluation
            best_score = float('-inf')
            chosen = actions[0]
            for a in actions:
                test_board = apply_placements(current_board, a.placements)
                score = evaluate_final_board(test_board)
                if score > best_score:
                    best_score = score
                    chosen = a
        else:
            # Use per-turn BC model
            model = bc_models.get(t, bc_models.get('default'))
            if model is not None:
                obs = Observation(
                    board_self=current_board,
                    board_opponent=Board(),
                    dealt_cards=deal,
                    known_discards_self=disc,
                    turn=t,
                    is_btn=True,
                )
                chosen = bc_greedy_action(model, obs, actions, device)
            else:
                chosen = actions[0]  # fallback

        # Apply action
        current_board = apply_placements(current_board, chosen.placements)
        if chosen.discard:
            disc.append(chosen.discard)

    return evaluate_final_board(current_board)


def turn_action_to_index(action, dealt_cards):
    """Map T1+ action to fixed 27-slot semantic index."""
    return get_semantic_action_index(action, dealt_cards)


def load_patterns_from_json(json_dir, target_turn, file_start=0, file_end=0):
    """Load board states at target_turn from existing Expectimax JSON results.

    Args:
        file_start: Start index of JSON files (0-based, inclusive)
        file_end: End index of JSON files (exclusive, 0=all)
    """
    json_path = Path(json_dir)
    json_files = sorted(json_path.glob("*.json"))
    total_files = len(json_files)
    if file_end > 0:
        json_files = json_files[file_start:file_end]
    elif file_start > 0:
        json_files = json_files[file_start:]
    print(f"  Files: [{file_start}:{file_end if file_end > 0 else total_files}] "
          f"({len(json_files)} of {total_files})")

    states = []
    n_files = 0

    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                result = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue

        if "choice_records" not in result:
            continue

        t0_hand = result["t0_hand"]

        # Get best T0 action
        best_t0_desc = result["all_actions"][0]["action_desc"]
        t0_placements = parse_t0_action_desc(best_t0_desc)

        # Build board after T0
        board_t0 = Board()
        for card, pos in t0_placements:
            getattr(board_t0, pos).append(card)

        # Collect choice records at target_turn
        for cr in result["choice_records"]:
            if cr["turn"] == target_turn:
                board = board_from_json(cr["top"], cr["mid"], cr["bot"])
                states.append({
                    "board": board,
                    "deal": cr["deal"],
                    "t0_hand": t0_hand,
                })

        n_files += 1

    print(f"Loaded {len(states)} T{target_turn} states from {n_files} files")
    return states


def batch_bc_inference(model, obs_list, actions_list, device):
    """Batch BC inference for multiple observations.

    Args:
        model: PolicyNetwork
        obs_list: list of Observation objects (some may be None)
        actions_list: list of action lists (parallel to obs_list)
        device: torch device

    Returns:
        List of chosen actions (parallel to obs_list)
    """
    # Find indices needing inference (>1 action and obs is not None)
    need_idx = []
    state_vecs = []
    masks = []
    for i, (obs, actions) in enumerate(zip(obs_list, actions_list)):
        if obs is not None and len(actions) > 1:
            need_idx.append(i)
            state_vecs.append(encode_state(obs))
            masks.append(create_action_mask(
                actions, turn=obs.turn, dealt_cards=obs.dealt_cards))

    # Default: pick first action
    chosen = []
    for actions in actions_list:
        chosen.append(actions[0] if actions else None)

    if not need_idx:
        return chosen

    # Batch forward pass
    batch_states = torch.FloatTensor(np.array(state_vecs)).to(device)
    batch_masks = torch.BoolTensor(np.array(masks)).to(device)

    with torch.no_grad():
        logits = model(batch_states, batch_masks)
        probs = logits.cpu().numpy()

    for bi, si in enumerate(need_idx):
        actions = actions_list[si]
        obs = obs_list[si]
        chosen[si] = max(
            actions,
            key=lambda a: probs[bi, get_semantic_action_index(a, obs.dealt_cards)]
        )

    return chosen


def generate_data(target_turn, states, n_samples, bc_models, device,
                  n_quick=3):
    """Generate training data with batched GPU inference.

    For T2: batches T3 inference, then T4 inference.
    For T1: batches T2, T3, T4 inference sequentially.
    """
    records = []
    total_playouts = 0
    t0 = time.time()
    CHUNK = 500  # T2 states per batch

    for chunk_start in range(0, len(states), CHUNK):
        chunk_end = min(chunk_start + CHUNK, len(states))
        chunk = states[chunk_start:chunk_end]

        # ---- Phase 1: Enumerate all (action, sample) → scenarios ----
        scenarios = []  # flat list of scenario dicts
        chunk_actions = []  # actions per state in chunk

        for si, state_info in enumerate(chunk):
            board = state_info["board"]
            deal = state_info["deal"]
            actions = get_turn_actions(deal, board)
            chunk_actions.append(actions)

            if not actions:
                continue

            rem = remaining_deck(board, deal)

            for ai, action in enumerate(actions):
                new_board = apply_placements(board, action.placements)
                disc = [action.discard] if action.discard else []
                action_rem = [c for c in rem if c not in deal]

                for sample_i in range(n_samples):
                    scenarios.append({
                        'si': si, 'ai': ai,
                        'board': new_board,
                        'rem': action_rem,
                        'disc': disc,
                    })

        if not scenarios:
            continue

        # ---- Phase 2+: Playout remaining turns with batched inference ----
        # Start from target_turn+1 and go to turn 4
        current_boards = [s['board'] for s in scenarios]
        current_rems = [list(s['rem']) for s in scenarios]
        current_discs = [list(s['disc']) for s in scenarios]

        for turn in range(target_turn + 1, 5):
            # Sample deals
            deals = []
            has_deal = []
            for i in range(len(scenarios)):
                b = current_boards[i]
                if b.card_count() >= 13 or len(current_rems[i]) < 3:
                    deals.append(None)
                    has_deal.append(False)
                    continue
                deal = random.sample(current_rems[i], 3)
                deals.append(deal)
                has_deal.append(True)

            # Get actions for all scenarios
            all_actions = []
            all_obs = []
            for i in range(len(scenarios)):
                if not has_deal[i]:
                    all_actions.append([])
                    all_obs.append(None)
                    continue
                actions = get_turn_actions(deals[i], current_boards[i])
                all_actions.append(actions)
                if len(actions) <= 1:
                    all_obs.append(None)
                else:
                    all_obs.append(Observation(
                        board_self=current_boards[i],
                        board_opponent=Board(),
                        dealt_cards=deals[i],
                        known_discards_self=current_discs[i],
                        turn=turn,
                        is_btn=True,
                    ))

            # T4: enumerate all placements, pick best by evaluation
            if turn == 4:
                chosen = []
                for i in range(len(scenarios)):
                    actions = all_actions[i]
                    if not actions:
                        chosen.append(None)
                        continue
                    best_score = float('-inf')
                    best_a = actions[0]
                    for a in actions:
                        tb = apply_placements(current_boards[i], a.placements)
                        sc = evaluate_final_board(tb)
                        if sc > best_score:
                            best_score = sc
                            best_a = a
                    chosen.append(best_a)
            else:
                # Batch inference
                model = bc_models.get(turn)
                if model is not None:
                    chosen = batch_bc_inference(
                        model, all_obs, all_actions, device)
                else:
                    chosen = [a[0] if a else None for a in all_actions]

            # Apply chosen actions
            for i in range(len(scenarios)):
                if not has_deal[i] or chosen[i] is None:
                    continue
                current_boards[i] = apply_placements(
                    current_boards[i], chosen[i].placements)
                if chosen[i].discard:
                    current_discs[i].append(chosen[i].discard)
                # Update remaining deck
                for c in deals[i]:
                    if c in current_rems[i]:
                        current_rems[i].remove(c)

        # ---- Phase 3: Evaluate final boards ----
        final_scores = [evaluate_final_board(b) for b in current_boards]
        total_playouts += len(scenarios)

        # ---- Phase 4: Aggregate scores per (state, action) ----
        score_map = defaultdict(list)
        for i, sc in enumerate(scenarios):
            score_map[(sc['si'], sc['ai'])].append(final_scores[i])

        # ---- Phase 5: Build records ----
        for si, state_info in enumerate(chunk):
            board = state_info["board"]
            deal = state_info["deal"]
            actions = chunk_actions[si]

            if not actions:
                continue

            action_evs = np.full(MAX_ACTIONS, -100.0, dtype=np.float32)
            valid_mask = np.zeros(MAX_ACTIONS, dtype=bool)

            for ai, action in enumerate(actions):
                scores = score_map.get((si, ai), [])
                if scores:
                    ev = float(np.mean(scores))
                    idx = turn_action_to_index(action, deal)
                    if ev > action_evs[idx]:
                        action_evs[idx] = ev
                    valid_mask[idx] = True

            obs = Observation(
                board_self=board,
                board_opponent=Board(),
                dealt_cards=deal,
                known_discards_self=[],
                turn=target_turn,
                is_btn=True,
            )
            state_vec = encode_state(obs)
            best_idx = int(np.argmax(action_evs))

            records.append({
                "state": state_vec.astype(np.float16),
                "action": best_idx,
                "ev": float(action_evs[best_idx]),
                "turn": target_turn,
                "valid_mask": valid_mask,
                "action_evs": action_evs.astype(np.float16),
            })

        elapsed = time.time() - t0
        rate = total_playouts / elapsed if elapsed > 0 else 0
        print(f"  [{chunk_end}/{len(states)}] {len(records)} records, "
              f"{total_playouts/1e6:.1f}M playouts, {rate:.0f}/s, "
              f"{elapsed:.0f}s")

    return records


def _worker_process_states(args):
    """Worker function for multiprocessing. Each worker handles a subset of
    states using CPU inference."""
    (target_turn, worker_states, n_samples, bc_model_paths, seed,
     worker_id) = args

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # Avoid thread contention between workers

    # Load BC models on CPU
    bc_models = {}
    for t, path in bc_model_paths.items():
        if path and Path(path).exists():
            bc_models[t] = load_bc_model(path, 'cpu')

    records = []
    total_playouts = 0
    t0 = time.time()
    report_interval = max(len(worker_states) // 10, 1)

    for si, state_info in enumerate(worker_states):
        board = state_info["board"]
        deal = state_info["deal"]

        actions = get_turn_actions(deal, board)
        if not actions:
            continue

        rem = remaining_deck(board, deal)
        action_evs_arr = np.full(MAX_ACTIONS, -100.0, dtype=np.float32)
        valid_mask = np.zeros(MAX_ACTIONS, dtype=bool)

        for action in actions:
            new_board = apply_placements(board, action.placements)
            disc = [action.discard] if action.discard else []
            action_rem = [c for c in rem if c not in deal]

            scores = []
            for _ in range(n_samples):
                score = playout_from_board(
                    new_board, action_rem,
                    turn=target_turn + 1,
                    bc_models=bc_models,
                    device='cpu',
                    discards=disc,
                )
                scores.append(score)
                total_playouts += 1

            ev = float(np.mean(scores))
            idx = turn_action_to_index(action, deal)
            if ev > action_evs_arr[idx]:
                action_evs_arr[idx] = ev
            valid_mask[idx] = True

        obs = Observation(
            board_self=board,
            board_opponent=Board(),
            dealt_cards=deal,
            known_discards_self=[],
            turn=target_turn,
            is_btn=True,
        )
        state_vec = encode_state(obs)
        best_idx = int(np.argmax(action_evs_arr))

        records.append({
            "state": state_vec.astype(np.float16),
            "action": best_idx,
            "ev": float(action_evs_arr[best_idx]),
            "turn": target_turn,
            "valid_mask": valid_mask,
            "action_evs": action_evs_arr.astype(np.float16),
        })

        if (si + 1) % report_interval == 0:
            elapsed = time.time() - t0
            rate = total_playouts / elapsed if elapsed > 0 else 0
            print(f"  [W{worker_id}] {si+1}/{len(worker_states)} "
                  f"{rate:.0f} playout/s", flush=True)

    return records


def generate_data_parallel(target_turn, states, n_samples, bc_model_paths,
                           n_workers):
    """Generate data using multiprocessing with CPU inference."""
    import os
    from multiprocessing import Pool

    # Workers only need CPU torch - prevent CUDA DLL loading to save memory
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Split states across workers
    chunk_size = (len(states) + n_workers - 1) // n_workers
    worker_args = []
    for i in range(n_workers):
        start = i * chunk_size
        end = min(start + chunk_size, len(states))
        if start >= len(states):
            break
        worker_args.append((
            target_turn,
            states[start:end],
            n_samples,
            bc_model_paths,
            42 + i * 1000,
            i,
        ))

    print(f"  Starting {len(worker_args)} workers "
          f"({chunk_size} states each)...")
    t0 = time.time()

    with Pool(len(worker_args)) as pool:
        results = pool.map(_worker_process_states, worker_args)

    records = []
    for r in results:
        records.extend(r)

    elapsed = time.time() - t0
    print(f"  Done: {len(records)} records in {elapsed:.0f}s "
          f"({len(records)/elapsed:.0f} records/s)")
    return records


def save_records(records, save_dir):
    """Save records to BC training format."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    N = len(records)
    states = np.array([r["state"] for r in records], dtype=np.float16)
    actions = np.array([r["action"] for r in records], dtype=np.int64)
    valid_masks = np.array([r["valid_mask"] for r in records], dtype=bool)
    action_evs = np.array([r["action_evs"] for r in records], dtype=np.float16)
    rewards = np.array([r["ev"] for r in records], dtype=np.float32)
    busted = np.zeros(N, dtype=np.float32)
    fl_entry = np.zeros(N, dtype=np.float32)
    royalties = np.zeros(N, dtype=np.float32)

    np.save(save_path / "states.npy", states)
    np.save(save_path / "actions.npy", actions)
    np.save(save_path / "valid_masks.npy", valid_masks)
    np.save(save_path / "action_evs.npy", action_evs)
    np.save(save_path / "rewards.npy", rewards)
    np.save(save_path / "busted.npy", busted)
    np.save(save_path / "fl_entry.npy", fl_entry)
    np.save(save_path / "royalties.npy", royalties)

    print(f"\nSaved {N} records to {save_path}:")
    print(f"  states:   {states.shape}")
    print(f"  EV range: [{rewards.min():.1f}, {rewards.max():.1f}], "
          f"mean={rewards.mean():.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Bottom-up per-turn BC data generation"
    )
    parser.add_argument("--turn", type=int, required=True,
                        choices=[1, 2], help="Target turn to generate data for")
    parser.add_argument("--n-samples", type=int, default=30,
                        help="Number of playout samples per action")
    parser.add_argument("--n-quick", type=int, default=3,
                        help="Quick eval samples for T1/T2 action selection")
    parser.add_argument("--bc-t1", default=None)
    parser.add_argument("--bc-t2", default=None)
    parser.add_argument("--bc-t3", default=None)
    parser.add_argument("--bc-t4", default=None)
    parser.add_argument("--bc-default", default=None,
                        help="Fallback BC model for any turn without specific model")
    parser.add_argument("--json-dir", default="data/expectimax_results_v3",
                        help="Directory with Expectimax JSON results")
    parser.add_argument("--save", required=True, help="Output directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (>1 uses CPU multiprocessing)")
    parser.add_argument("--max-states", type=int, default=0,
                        help="Limit number of states to process (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--file-start", type=int, default=0,
                        help="Start index of JSON files (0-based)")
    parser.add_argument("--file-end", type=int, default=0,
                        help="End index of JSON files (exclusive, 0=all)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Bottom-Up T{args.turn} Data Generation ===")
    print(f"  Device: {device}, Workers: {args.workers}")
    print(f"  N_samples: {args.n_samples}")

    # Load states from existing Expectimax data
    states = load_patterns_from_json(args.json_dir, args.turn,
                                     args.file_start, args.file_end)

    if args.max_states > 0:
        states = states[:args.max_states]
        print(f"  Limited to {len(states)} states")

    # Collect BC model paths for workers
    bc_model_paths = {}
    for t, path in [(1, args.bc_t1), (2, args.bc_t2),
                     (3, args.bc_t3), (4, args.bc_t4)]:
        if path:
            bc_model_paths[t] = path
    if args.bc_default:
        bc_model_paths['default'] = args.bc_default

    if args.workers > 1:
        # Multiprocessing with CPU inference
        records = generate_data_parallel(
            target_turn=args.turn,
            states=states,
            n_samples=args.n_samples,
            bc_model_paths=bc_model_paths,
            n_workers=args.workers,
        )
    else:
        # Single process with GPU batching
        bc_models = {}
        for t, path in bc_model_paths.items():
            if Path(path).exists():
                bc_models[t] = load_bc_model(path, device)
                print(f"  Loaded T{t} BC: {path}")

        records = generate_data(
            target_turn=args.turn,
            states=states,
            n_samples=args.n_samples,
            bc_models=bc_models,
            device=device,
            n_quick=args.n_quick,
        )

    # Save
    save_records(records, args.save)

    print(f"\n=== T{args.turn} Data Generation Complete ===")


if __name__ == "__main__":
    main()
