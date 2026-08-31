"""
Multiprocess DAgger BC data collection.

Splits games across N workers for ~Nx speedup on multi-core machines.
Each worker plays games independently, evaluates all T1+ actions via rollouts,
and returns records. Results are merged and saved in BC training format.

Usage:
    python -m ai.training.collect_dagger_mp \
        --games 5000 --rollouts 50 --workers 20 \
        --bc-model ai/models/expectimax_bc_v4/bc_policy_best.pt \
        --save data/dagger_bc_v2
"""
import sys
import random
import argparse
import time
import numpy as np
import torch
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.engine.encoding import (
    Board, Observation, ALL_CARDS, encode_state, STATE_DIM
)
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, create_action_mask,
    get_semantic_action_index
)
from ai.engine.game_engine import Hand
from ai.models.networks import PolicyNetwork
from ai.mcts.rollout_evaluator import RolloutEvaluator

MAX_ACTIONS = 250


def bc_greedy(policy_net, obs, actions=None):
    if actions is None:
        actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    if not actions:
        return None, []
    if len(actions) == 1:
        return actions[0], actions
    state_t = torch.FloatTensor(encode_state(obs)).unsqueeze(0)
    mask = create_action_mask(actions, turn=obs.turn, dealt_cards=obs.dealt_cards)
    mask_t = torch.BoolTensor(mask).unsqueeze(0)
    with torch.no_grad():
        probs = policy_net(state_t, mask_t).squeeze(0).numpy()
    if obs.turn == 0:
        return actions[int(np.argmax(probs[:len(actions)]))], actions
    action = max(
        actions,
        key=lambda a: probs[get_semantic_action_index(a, obs.dealt_cards)]
    )
    return action, actions


def action_to_index(action, dealt_cards):
    return get_semantic_action_index(action, dealt_cards)


def evaluate_all_actions(evaluator, obs, actions, n_rollouts):
    action_evs = np.full(MAX_ACTIONS, -100.0, dtype=np.float32)
    valid_mask = np.zeros(MAX_ACTIONS, dtype=bool)
    bust_rates = np.zeros(MAX_ACTIONS, dtype=np.float32)

    for action in actions:
        idx = action_to_index(action, obs.dealt_cards)
        results = [evaluator._single_rollout_detailed(obs, action)
                   for _ in range(n_rollouts)]
        ev = float(np.mean([r.score for r in results]))
        bust = sum(1 for r in results if r.busted) / len(results)
        if ev > action_evs[idx]:
            action_evs[idx] = ev
            bust_rates[idx] = bust
        valid_mask[idx] = True

    best_idx = int(np.argmax(action_evs))
    return action_evs, valid_mask, best_idx, bust_rates


def worker_fn(args_tuple):
    """Worker: play games and collect DAgger data."""
    worker_id, n_games, rollouts, bc_model_path, seed = args_tuple

    random.seed(seed + worker_id * 10000)
    np.random.seed(seed + worker_id * 10000)
    torch.manual_seed(seed + worker_id * 10000)

    # Load BC policy (per-worker)
    policy = PolicyNetwork()
    ck = torch.load(bc_model_path, map_location='cpu', weights_only=True)
    policy.load_state_dict(ck.get('model_state_dict', ck))
    policy.eval()

    evaluator = RolloutEvaluator(
        policy_net=policy, n_rollouts=rollouts, top_k=20,
        device='cpu', use_policy_playout=False,
    )

    records = []
    total_rollouts = 0
    t0 = time.time()

    for game_idx in range(n_games):
        deck = list(ALL_CARDS)
        random.shuffle(deck)
        hand = Hand(deck=list(deck), btn=0)

        # T0: play, don't collect
        for seat in [hand.btn, 1 - hand.btn]:
            obs = hand.get_observation(seat)
            actions = get_initial_actions(obs.dealt_cards, obs.board_self)
            chosen, _ = bc_greedy(policy, obs, actions)
            if chosen:
                hand.apply_action(seat, chosen)

        # T1+: play and collect
        for turn_num in range(1, 9):
            if hand.is_hand_complete():
                break
            hand.deal_next_turn()
            for seat in [hand.btn, 1 - hand.btn]:
                cards = hand.dealt_cards[seat]
                if not cards or hand.boards[seat].is_complete():
                    continue
                obs = hand.get_observation(seat)
                cc = obs.board_self.card_count()

                # Skip last turn (too few actions, exact eval)
                if cc >= 11:
                    actions = get_turn_actions(obs.dealt_cards, obs.board_self)
                    if actions:
                        hand.apply_action(seat, actions[0])
                    continue

                actions = get_turn_actions(obs.dealt_cards, obs.board_self)
                if len(actions) <= 1:
                    if actions:
                        hand.apply_action(seat, actions[0])
                    continue

                state = encode_state(obs)
                action_evs, valid_mask, best_idx, bust_rates = \
                    evaluate_all_actions(evaluator, obs, actions, rollouts)
                total_rollouts += len(actions) * rollouts

                records.append({
                    "state": state.astype(np.float16),
                    "action": best_idx,
                    "ev": float(action_evs[best_idx]),
                    "turn": obs.turn,
                    "valid_mask": valid_mask,
                    "action_evs": action_evs.astype(np.float16),
                    "busted": float(bust_rates[best_idx]),
                    "fl_entry": 0.0,
                })

                chosen, _ = bc_greedy(policy, obs, actions)
                if chosen:
                    hand.apply_action(seat, chosen)

        if (game_idx + 1) % max(n_games // 5, 1) == 0:
            elapsed = time.time() - t0
            rate = total_rollouts / elapsed if elapsed > 0 else 0
            print(f"  [W{worker_id}] {game_idx+1}/{n_games} games, "
                  f"{len(records)} states, {rate:.0f} rollouts/s")

    elapsed = time.time() - t0
    print(f"  [W{worker_id}] Done: {len(records)} states, "
          f"{total_rollouts/1e6:.1f}M rollouts in {elapsed:.0f}s")
    return records, total_rollouts


def main():
    parser = argparse.ArgumentParser(description="Multiprocess DAgger BC")
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of worker processes (0=auto)")
    parser.add_argument("--bc-model",
                        default="ai/models/expectimax_bc_v4/bc_policy_best.pt")
    parser.add_argument("--save", default="data/dagger_bc_v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_workers = args.workers if args.workers > 0 else mp.cpu_count()
    games_per_worker = args.games // n_workers
    remainder = args.games % n_workers

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Multiprocess DAgger BC Data Collection")
    print("=" * 60)
    print(f"  Games: {args.games}")
    print(f"  Workers: {n_workers}")
    print(f"  Games/worker: {games_per_worker} (+{remainder} extra)")
    print(f"  Rollouts/action: {args.rollouts}")
    print(f"  BC model: {args.bc_model}")
    print(f"  Save: {args.save}")
    print()

    # Build worker args
    worker_args = []
    for i in range(n_workers):
        n = games_per_worker + (1 if i < remainder else 0)
        worker_args.append((i, n, args.rollouts, args.bc_model, args.seed))

    t0 = time.time()

    with mp.Pool(n_workers) as pool:
        results = pool.map(worker_fn, worker_args)

    # Merge results
    all_records = []
    total_rollouts = 0
    for records, rollouts in results:
        all_records.extend(records)
        total_rollouts += rollouts

    total_time = time.time() - t0

    N = len(all_records)
    if N == 0:
        print("No records collected!")
        return

    states = np.array([r["state"] for r in all_records], dtype=np.float16)
    actions = np.array([r["action"] for r in all_records], dtype=np.int64)
    valid_masks = np.array([r["valid_mask"] for r in all_records], dtype=bool)
    action_evs = np.array([r["action_evs"] for r in all_records], dtype=np.float16)
    royalties = np.zeros(N, dtype=np.float32)
    busted = np.array([r["busted"] for r in all_records], dtype=np.float32)
    fl_entry = np.array([r["fl_entry"] for r in all_records], dtype=np.float32)
    rewards = np.array([r["ev"] for r in all_records], dtype=np.float32)
    turns = np.array([r["turn"] for r in all_records], dtype=np.int32)

    np.save(save_dir / "states.npy", states)
    np.save(save_dir / "actions.npy", actions)
    np.save(save_dir / "valid_masks.npy", valid_masks)
    np.save(save_dir / "action_evs.npy", action_evs)
    np.save(save_dir / "royalties.npy", royalties)
    np.save(save_dir / "busted.npy", busted)
    np.save(save_dir / "fl_entry.npy", fl_entry)
    np.save(save_dir / "rewards.npy", rewards)

    # VN-format data
    np.savez_compressed(
        save_dir / "value_train.npz",
        obs=states, score=rewards, turn=turns,
        busted=busted, fl=fl_entry,
    )

    print(f"\n{'='*60}")
    print(f"  Multiprocess DAgger BC Results")
    print(f"{'='*60}")
    print(f"  Workers: {n_workers}")
    print(f"  Total states: {N}")
    print(f"  Total rollouts: {total_rollouts/1e6:.1f}M")
    print(f"  EV range: [{rewards.min():.1f}, {rewards.max():.1f}], "
          f"mean={rewards.mean():.2f}")
    print(f"  Bust rate: {(busted > 0.5).sum()} ({100*(busted > 0.5).mean():.1f}%)")
    print(f"  Turn distribution: {np.bincount(turns, minlength=5).tolist()}")
    print(f"  Time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Throughput: {total_rollouts/total_time:.0f} rollouts/s")
    print(f"  Saved: {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
