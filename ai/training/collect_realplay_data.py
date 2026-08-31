"""
Collect real-play VN/BC training data via rollouts.

Play games with BC-greedy, then label each state via rollouts to get
accurate EV, bust_prob, and fl_prob from the actual gameplay distribution.
This fixes the distribution shift problem in Expectimax-only VN training.

Usage:
    python -m ai.training.collect_realplay_data \
        --games 1000 --rollouts 100 --alt-actions 3 \
        --bc-model ai/models/expectimax_bc_v4/bc_policy_best.pt \
        --save data/realplay_vn_v1
"""
import sys
import random
import argparse
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.engine.encoding import (
    Board, Observation, ALL_CARDS, encode_state, STATE_DIM
)
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, create_action_mask,
    get_semantic_action_index
)
from ai.engine.game_engine import (
    GameEngine, Hand, evaluate_hand,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
)
from ai.models.networks import PolicyNetwork
from ai.mcts.rollout_evaluator import RolloutEvaluator


def bc_greedy(policy_net, obs, actions=None):
    """Select action greedily using BC policy."""
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


def label_with_rollouts(evaluator, obs, action, n_rollouts):
    """Run rollouts for one (state, action) pair, return (score, bust_rate, fl_rate)."""
    results = [evaluator._single_rollout_detailed(obs, action)
               for _ in range(n_rollouts)]
    scores = [r.score for r in results]
    bust_rate = sum(1 for r in results if r.busted) / len(results)
    fl_rate = sum(1 for r in results if r.fl_qualified) / len(results)
    return float(np.mean(scores)), bust_rate, fl_rate


def encode_post_action(obs, action):
    """Encode the state after applying action."""
    new_board = obs.board_self.copy()
    for card, pos in action.placements:
        getattr(new_board, pos).append(card)

    discards = list(obs.known_discards_self)
    if action.discard:
        discards.append(action.discard)

    post_obs = Observation(
        board_self=new_board,
        board_opponent=obs.board_opponent,
        dealt_cards=[],
        known_discards_self=discards,
        turn=obs.turn,
        is_btn=obs.is_btn,
        chips_self=obs.chips_self,
        chips_opponent=obs.chips_opponent,
    )
    return encode_state(post_obs), post_obs


def main():
    parser = argparse.ArgumentParser(description="Collect real-play VN data")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--rollouts", type=int, default=100,
                        help="Rollouts per state-action pair for labeling")
    parser.add_argument("--alt-actions", type=int, default=3,
                        help="Random alternative actions to label per state")
    parser.add_argument("--bc-model",
                        default="ai/models/expectimax_bc_v4/bc_policy_best.pt")
    parser.add_argument("--save", default="data/realplay_vn_v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Real-Play VN Data Collection")
    print("=" * 60)
    print(f"  Games: {args.games}")
    print(f"  Rollouts/state: {args.rollouts}")
    print(f"  Alt actions: {args.alt_actions}")
    print(f"  BC model: {args.bc_model}")
    print(f"  Save: {args.save}")
    print()

    # Load BC policy
    policy = PolicyNetwork()
    ck = torch.load(args.bc_model, map_location='cpu', weights_only=True)
    policy.load_state_dict(ck.get('model_state_dict', ck))
    policy.eval()

    # Create evaluator for rollouts (rule_playout, no VN needed)
    evaluator = RolloutEvaluator(
        policy_net=policy, n_rollouts=args.rollouts, top_k=20,
        device='cpu', use_policy_playout=False,
    )

    # Phase 1: Play games and collect states
    print("Phase 1: Playing games and collecting states...")
    t0 = time.time()

    # Collect: list of (obs, action, turn, is_chosen)
    state_records = []

    for game_idx in range(args.games):
        deck = list(ALL_CARDS)
        random.shuffle(deck)
        hand = Hand(deck=list(deck), btn=0)

        # T0: initial placement
        for seat in [hand.btn, 1 - hand.btn]:
            obs = hand.get_observation(seat)
            if obs.turn == 0:
                actions = get_initial_actions(obs.dealt_cards, obs.board_self)
            else:
                actions = get_turn_actions(obs.dealt_cards, obs.board_self)

            chosen, all_actions = bc_greedy(policy, obs, actions)

            if chosen:
                # Record chosen action
                state_records.append((obs, chosen, obs.turn, True))

                # Record alternative actions (for diverse VN data)
                if args.alt_actions > 0 and len(all_actions) > 1:
                    others = [a for a in all_actions if a is not chosen]
                    random.shuffle(others)
                    for alt in others[:args.alt_actions]:
                        state_records.append((obs, alt, obs.turn, False))

                hand.apply_action(seat, chosen)

        # T1+
        for turn_num in range(1, 9):
            if hand.is_hand_complete():
                break
            hand.deal_next_turn()
            for seat in [hand.btn, 1 - hand.btn]:
                cards = hand.dealt_cards[seat]
                if not cards or hand.boards[seat].is_complete():
                    continue
                obs = hand.get_observation(seat)
                actions = get_turn_actions(obs.dealt_cards, obs.board_self)
                chosen, all_actions = bc_greedy(policy, obs, actions)

                if chosen:
                    state_records.append((obs, chosen, obs.turn, True))

                    if args.alt_actions > 0 and len(all_actions) > 1:
                        others = [a for a in all_actions if a is not chosen]
                        random.shuffle(others)
                        for alt in others[:args.alt_actions]:
                            state_records.append((obs, alt, obs.turn, False))

                    hand.apply_action(seat, chosen)

        if (game_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{game_idx+1}/{args.games}] {len(state_records)} states "
                  f"({elapsed:.0f}s, {elapsed/(game_idx+1)*1000:.0f}ms/game)")

    play_time = time.time() - t0
    print(f"\n  Phase 1 done: {len(state_records)} states from {args.games} games "
          f"({play_time:.0f}s)")

    # Phase 2: Label states with rollouts
    print(f"\nPhase 2: Labeling {len(state_records)} states with {args.rollouts} rollouts each...")
    t1 = time.time()

    obs_list = []
    score_list = []
    turn_list = []
    bust_list = []
    fl_list = []

    for i, (obs, action, turn, is_chosen) in enumerate(state_records):
        encoded, post_obs = encode_post_action(obs, action)
        score, bust_rate, fl_rate = label_with_rollouts(
            evaluator, obs, action, args.rollouts)

        obs_list.append(encoded.astype(np.float16))
        score_list.append(score)
        turn_list.append(turn)
        bust_list.append(bust_rate)
        fl_list.append(fl_rate)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t1
            rate = (i + 1) / elapsed
            eta = (len(state_records) - i - 1) / rate
            n_bust = sum(1 for b in bust_list if b > 0.5)
            n_fl = sum(1 for f in fl_list if f > 0.5)
            print(f"  [{i+1}/{len(state_records)}] {elapsed:.0f}s "
                  f"({rate:.1f} states/s, ETA {eta:.0f}s) "
                  f"bust>50%: {n_bust} ({100*n_bust/(i+1):.1f}%) "
                  f"fl>50%: {n_fl} ({100*n_fl/(i+1):.1f}%)")

    label_time = time.time() - t1

    # Save as NPZ
    obs_arr = np.array(obs_list, dtype=np.float16)
    score_arr = np.array(score_list, dtype=np.float32)
    turn_arr = np.array(turn_list, dtype=np.int32)
    bust_arr = np.array(bust_list, dtype=np.float32)
    fl_arr = np.array(fl_list, dtype=np.float32)

    npz_path = save_dir / "value_train.npz"
    np.savez_compressed(
        npz_path,
        obs=obs_arr, score=score_arr, turn=turn_arr,
        busted=bust_arr, fl=fl_arr,
    )

    total_time = time.time() - t0

    # Stats
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"  Total states: {len(obs_list)}")
    print(f"  Score range: [{score_arr.min():.1f}, {score_arr.max():.1f}]")
    print(f"  Score mean: {score_arr.mean():.2f}")
    print(f"  Bust rate (>50%): {(bust_arr > 0.5).sum()} ({100*(bust_arr > 0.5).mean():.1f}%)")
    print(f"  FL rate (>50%): {(fl_arr > 0.5).sum()} ({100*(fl_arr > 0.5).mean():.1f}%)")
    turns = np.bincount(turn_arr, minlength=5)
    print(f"  Turn distribution: {turns.tolist()}")
    print(f"  Play time: {play_time:.0f}s")
    print(f"  Label time: {label_time:.0f}s")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  Saved: {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
