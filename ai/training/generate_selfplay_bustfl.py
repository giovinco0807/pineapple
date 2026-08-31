"""
Generate Self-Play Bust/FL Training Data for VN

Plays BC-vs-BC games, records every intermediate board state,
and back-propagates exact bust/FL labels from the completed board.

Key advantage over Expectimax off-policy:
  - bust/FL labels are EXACT (not MC estimates)
  - States match BC's actual distribution (no distribution shift)
  - EV labels are noisy (game outcome), but we only need bust/FL

Output: NPZ compatible with train_value.py (--data-extra)

Usage:
    python -m ai.training.generate_selfplay_bustfl --games 5000 --output data/selfplay_bustfl
    python -m ai.training.generate_selfplay_bustfl --games 5000 --output data/selfplay_bustfl --workers 8
"""
import sys
import random
import time
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from ai.engine.encoding import (
    Board, Observation, encode_state, ALL_CARDS, STATE_DIM
)
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, create_action_mask
)
from ai.engine.game_engine import (
    Hand, evaluate_hand, check_fl_entry,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    evaluate_board_with_joker_constraint,
)
from ai.models.networks import PolicyNetwork


# ── Worker globals ──────────────────────────────────────────

_policy_net = None


def init_worker(model_path):
    """Initialize BC policy in each worker."""
    global _policy_net
    _policy_net = PolicyNetwork()
    _policy_net.load_state_dict(
        torch.load(model_path, map_location='cpu', weights_only=True))
    _policy_net.eval()


def bc_select_t0(obs):
    """BC greedy action selection for T0."""
    valid_actions = get_initial_actions(obs.dealt_cards, obs.board_self)
    if not valid_actions:
        return None
    if len(valid_actions) == 1:
        return valid_actions[0]
    state_vec = encode_state(obs)
    state_t = torch.FloatTensor(state_vec).unsqueeze(0)
    mask = create_action_mask(valid_actions)
    mask_t = torch.BoolTensor(mask).unsqueeze(0)
    with torch.no_grad():
        probs = _policy_net(state_t, mask_t).squeeze(0).numpy()
    return valid_actions[int(np.argmax(probs[:len(valid_actions)]))]


def bc_select(obs):
    """BC greedy action selection for T1+."""
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    if not valid_actions:
        return None
    if len(valid_actions) == 1:
        return valid_actions[0]
    state_vec = encode_state(obs)
    state_t = torch.FloatTensor(state_vec).unsqueeze(0)
    mask = create_action_mask(valid_actions)
    mask_t = torch.BoolTensor(mask).unsqueeze(0)
    with torch.no_grad():
        probs = _policy_net(state_t, mask_t).squeeze(0).numpy()
    return valid_actions[int(np.argmax(probs[:len(valid_actions)]))]


# ── Game play ───────────────────────────────────────────────

def play_one_game(_game_idx):
    """Play one BC-vs-BC game, return per-turn state snapshots with bust/FL labels.

    Returns list of dicts: {state, turn, busted, fl_entry, score}
    """
    deck = list(ALL_CARDS)
    random.shuffle(deck)
    hand = Hand(deck=deck, btn=0)

    # Collect snapshots: (seat, turn, encoded_state)
    snapshots = []

    # T0
    for seat in [hand.btn, 1 - hand.btn]:
        obs = hand.get_observation(seat)
        action = bc_select_t0(obs)
        if action is None:
            continue

        # Encode POST-action state
        new_board = obs.board_self.copy()
        for card, pos in action.placements:
            getattr(new_board, pos).append(card)
        post_obs = Observation(
            board_self=new_board,
            board_opponent=obs.board_opponent,
            dealt_cards=[],
            known_discards_self=list(obs.known_discards_self),
            turn=0,
            is_btn=obs.is_btn,
        )
        state = encode_state(post_obs)
        snapshots.append((seat, 0, state))
        hand.apply_action(seat, action)

    # T1-T4
    for turn_num in range(1, 9):
        if hand.is_hand_complete():
            break
        hand.deal_next_turn()
        for seat in [hand.btn, 1 - hand.btn]:
            cards = hand.dealt_cards[seat]
            if not cards or hand.boards[seat].is_complete():
                continue
            obs = hand.get_observation(seat)
            action = bc_select(obs)
            if action is None:
                continue

            # Encode POST-action state
            new_board = obs.board_self.copy()
            new_discards = list(obs.known_discards_self)
            for card, pos in action.placements:
                getattr(new_board, pos).append(card)
            if action.discard:
                new_discards.append(action.discard)
            post_obs = Observation(
                board_self=new_board,
                board_opponent=obs.board_opponent,
                dealt_cards=[],
                known_discards_self=new_discards,
                turn=turn_num,
                is_btn=obs.is_btn,
            )
            state = encode_state(post_obs)
            snapshots.append((seat, turn_num, state))
            hand.apply_action(seat, action)

    # Game complete: determine bust/FL for each seat through the canonical
    # bottom-up Joker evaluator.
    seat_info = {}
    for seat in [0, 1]:
        board = hand.boards[seat]
        evaluated = evaluate_board_with_joker_constraint(
            board.top, board.middle, board.bottom
        )
        is_busted = bool(evaluated["busted"])
        is_fl = bool(evaluated["fl_entry"])
        seat_info[seat] = {
            'busted': is_busted,
            'fl_entry': is_fl,
        }

    # Compute score (seat 0 perspective)
    from ai.mcts.rollout_evaluator import RolloutEvaluator
    score_0 = RolloutEvaluator.compute_score_raw(hand.boards[0], hand.boards[1])

    # Build records: back-propagate bust/FL labels
    records = []
    for seat, turn, state in snapshots:
        info = seat_info[seat]
        score = score_0 if seat == 0 else -score_0
        records.append({
            'state': state.astype(np.float16),
            'turn': turn,
            'busted': 1.0 if info['busted'] else 0.0,
            'fl_entry': 1.0 if info['fl_entry'] else 0.0,
            'score': float(score),
        })

    n_bust = sum(1 for s in [0, 1] if seat_info[s]['busted'])
    n_fl = sum(1 for s in [0, 1] if seat_info[s]['fl_entry'])
    return records, n_bust, n_fl


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate self-play bust/FL data for VN training")
    parser.add_argument("--model",
                        default="ai/models/expectimax_bc_v3/bc_policy_best.pt",
                        help="BC policy model path")
    parser.add_argument("--games", type=int, default=5000,
                        help="Number of games to play")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0=auto)")
    parser.add_argument("--output", default="data/selfplay_bustfl",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    n_workers = args.workers or max(1, mp.cpu_count() - 2)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Self-Play Bust/FL Data Generator")
    print(f"  Model: {args.model}")
    print(f"  Games: {args.games}")
    print(f"  Workers: {n_workers}")
    print(f"  Output: {output_path}")
    print()

    # Collect all records
    all_states = []
    all_scores = []
    all_turns = []
    all_busted = []
    all_fl = []
    total_busts = 0
    total_fl = 0
    completed = 0

    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(args.model,),
    ) as pool:
        futures = {pool.submit(play_one_game, i): i for i in range(args.games)}
        for future in as_completed(futures):
            try:
                records, n_bust, n_fl = future.result()
                for r in records:
                    all_states.append(r['state'])
                    all_scores.append(r['score'])
                    all_turns.append(r['turn'])
                    all_busted.append(r['busted'])
                    all_fl.append(r['fl_entry'])
                total_busts += n_bust
                total_fl += n_fl
                completed += 1

                if completed % 100 == 0 or completed == args.games:
                    elapsed = time.time() - t0
                    bust_rate = total_busts / max(completed * 2, 1)
                    fl_rate = total_fl / max(completed * 2, 1)
                    speed = completed / elapsed
                    eta = (args.games - completed) / speed if speed > 0 else 0
                    print(f"  [{completed:>5}/{args.games}] "
                          f"samples={len(all_states):>7,}  "
                          f"bust={bust_rate:.1%}  fl={fl_rate:.1%}  "
                          f"{speed:.1f} g/s  ETA {eta:.0f}s")
            except Exception as e:
                completed += 1
                print(f"  [ERROR] Game {futures[future]}: {e}")

    elapsed = time.time() - t0
    N = len(all_states)
    print(f"\nGenerated {N:,} samples from {completed} games in {elapsed:.1f}s")

    if N == 0:
        print("No samples generated!")
        return

    # Convert to arrays
    obs = np.stack(all_states)
    score = np.array(all_scores, dtype=np.float32)
    turn = np.array(all_turns, dtype=np.int32)
    busted = np.array(all_busted, dtype=np.float32)
    fl_entry = np.array(all_fl, dtype=np.float32)

    # Stats
    print(f"\n{'='*60}")
    print(f"  Self-play bust/FL data: {N:,} samples")
    print(f"{'='*60}")
    turn_counts = Counter(turn.tolist())
    for t in sorted(turn_counts):
        mask = turn == t
        avg_score = float(np.mean(score[mask]))
        bust_r = float(np.mean(busted[mask]))
        fl_r = float(np.mean(fl_entry[mask]))
        print(f"  T{t}: {turn_counts[t]:>7,}  "
              f"score={avg_score:>+6.1f}  "
              f"bust={bust_r:.1%}  FL={fl_r:.1%}")
    print(f"  {'─'*54}")
    print(f"  Total: {N:>7,}  "
          f"score={float(np.mean(score)):>+6.1f}  "
          f"bust={float(np.mean(busted)):.1%}  "
          f"FL={float(np.mean(fl_entry)):.1%}")

    # Save
    npz_path = output_path / "value_train.npz"
    print(f"\n  Saving to {npz_path} ...")
    np.savez_compressed(npz_path, obs=obs, score=score, turn=turn,
                        busted=busted, fl_entry=fl_entry)
    file_size = npz_path.stat().st_size
    print(f"  File: {file_size / 1e6:.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
