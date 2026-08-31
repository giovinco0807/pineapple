"""
DAgger-style BC data collection from real play.

Play games with BC-greedy, then at each T1+ state evaluate ALL valid actions
via rollouts to get per-action EVs for soft-label BC training.

This fixes distribution shift: BC learns from states that actually occur in play,
not just Expectimax-generated states.

T0 is skipped (Expectimax T0 data is high quality and T0 has ~1000 actions).

Usage:
    python -m ai.training.collect_dagger_bc \
        --games 1000 --rollouts 50 \
        --bc-model ai/models/expectimax_bc_v4/bc_policy_best.pt \
        --save data/dagger_bc_v1
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
from ai.engine.game_engine import Hand
from ai.models.networks import PolicyNetwork
from ai.mcts.rollout_evaluator import RolloutEvaluator

MAX_ACTIONS = 250  # Must match convert_expectimax.py


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


def action_to_index(action, dealt_cards):
    """Map T1+ action to fixed 27-slot semantic index."""
    return get_semantic_action_index(action, dealt_cards)


def evaluate_all_actions(evaluator, obs, actions, n_rollouts):
    """Evaluate all valid actions via rollouts. Returns (action_evs, valid_mask, best_idx)."""
    action_evs = np.full(MAX_ACTIONS, -100.0, dtype=np.float32)
    valid_mask = np.zeros(MAX_ACTIONS, dtype=bool)
    bust_rates = np.zeros(MAX_ACTIONS, dtype=np.float32)

    for action in actions:
        idx = action_to_index(action, obs.dealt_cards)
        results = [evaluator._single_rollout_detailed(obs, action)
                   for _ in range(n_rollouts)]
        ev = float(np.mean([r.score for r in results]))
        bust = sum(1 for r in results if r.busted) / len(results)

        # Keep best EV if multiple actions map to same index
        if ev > action_evs[idx]:
            action_evs[idx] = ev
            bust_rates[idx] = bust
        valid_mask[idx] = True

    best_idx = int(np.argmax(action_evs))
    return action_evs, valid_mask, best_idx, bust_rates


def main():
    parser = argparse.ArgumentParser(description="DAgger BC data collection")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--rollouts", type=int, default=50,
                        help="Rollouts per action for EV estimation")
    parser.add_argument("--bc-model",
                        default="ai/models/expectimax_bc_v4/bc_policy_best.pt")
    parser.add_argument("--save", default="data/dagger_bc_v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DAgger BC Data Collection (T1+ only)")
    print("=" * 60)
    print(f"  Games: {args.games}")
    print(f"  Rollouts/action: {args.rollouts}")
    print(f"  BC model: {args.bc_model}")
    print(f"  Save: {args.save}")
    print()

    # Load BC policy
    policy = PolicyNetwork()
    ck = torch.load(args.bc_model, map_location='cpu', weights_only=True)
    policy.load_state_dict(ck.get('model_state_dict', ck))
    policy.eval()

    # Evaluator for rollouts (rule_playout)
    evaluator = RolloutEvaluator(
        policy_net=policy, n_rollouts=args.rollouts, top_k=20,
        device='cpu', use_policy_playout=False,
    )

    # Collect states and label them
    records = []
    t0 = time.time()
    total_rollouts = 0

    for game_idx in range(args.games):
        deck = list(ALL_CARDS)
        random.shuffle(deck)
        hand = Hand(deck=list(deck), btn=0)

        # T0: just play, don't collect (Expectimax T0 data is better)
        for seat in [hand.btn, 1 - hand.btn]:
            obs = hand.get_observation(seat)
            actions = get_initial_actions(obs.dealt_cards, obs.board_self)
            chosen, _ = bc_greedy(policy, obs, actions)
            if chosen:
                hand.apply_action(seat, chosen)

        # T1+: play and collect DAgger data
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
                if cc >= 11:
                    # Last turn: exact eval, no need for DAgger data
                    from eval_mcts import direct_eval_last_turn
                    action = direct_eval_last_turn(obs)
                    if action:
                        hand.apply_action(seat, action)
                    continue

                actions = get_turn_actions(obs.dealt_cards, obs.board_self)
                if len(actions) <= 1:
                    if actions:
                        hand.apply_action(seat, actions[0])
                    continue

                # Encode pre-action state
                state = encode_state(obs)

                # Evaluate ALL valid actions via rollouts
                action_evs, valid_mask, best_idx, bust_rates = \
                    evaluate_all_actions(evaluator, obs, actions, args.rollouts)
                total_rollouts += len(actions) * args.rollouts

                records.append({
                    "state": state.astype(np.float16),
                    "action": best_idx,
                    "ev": float(action_evs[best_idx]),
                    "turn": obs.turn,
                    "valid_mask": valid_mask,
                    "action_evs": action_evs.astype(np.float16),
                    "busted": float(bust_rates[best_idx]),
                    "fl_entry": 0.0,  # TODO: could compute from rollouts
                })

                # Play the BC-chosen action (not the rollout-best) to maintain
                # the on-policy distribution
                chosen, _ = bc_greedy(policy, obs, actions)
                if chosen:
                    hand.apply_action(seat, chosen)

        if (game_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = total_rollouts / elapsed if elapsed > 0 else 0
            eta = (args.games - game_idx - 1) / ((game_idx + 1) / elapsed) if elapsed > 0 else 0
            print(f"  [{game_idx+1}/{args.games}] {len(records)} states, "
                  f"{total_rollouts/1e6:.1f}M rollouts ({rate:.0f}/s, "
                  f"ETA {eta:.0f}s)")

    total_time = time.time() - t0

    # Save in BC format
    N = len(records)
    if N == 0:
        print("No records collected!")
        return

    states = np.array([r["state"] for r in records], dtype=np.float16)
    actions = np.array([r["action"] for r in records], dtype=np.int64)
    valid_masks = np.array([r["valid_mask"] for r in records], dtype=bool)
    action_evs = np.array([r["action_evs"] for r in records], dtype=np.float16)
    royalties = np.zeros(N, dtype=np.float32)
    busted = np.array([r["busted"] for r in records], dtype=np.float32)
    fl_entry = np.array([r["fl_entry"] for r in records], dtype=np.float32)
    rewards = np.array([r["ev"] for r in records], dtype=np.float32)
    turns = np.array([r["turn"] for r in records], dtype=np.int32)

    np.save(save_dir / "states.npy", states)
    np.save(save_dir / "actions.npy", actions)
    np.save(save_dir / "valid_masks.npy", valid_masks)
    np.save(save_dir / "action_evs.npy", action_evs)
    np.save(save_dir / "royalties.npy", royalties)
    np.save(save_dir / "busted.npy", busted)
    np.save(save_dir / "fl_entry.npy", fl_entry)
    np.save(save_dir / "rewards.npy", rewards)

    # Also save VN-format data
    vn_scores = np.array([r["ev"] for r in records], dtype=np.float32)
    np.savez_compressed(
        save_dir / "value_train.npz",
        obs=states, score=vn_scores, turn=turns,
        busted=busted, fl=fl_entry,
    )

    print(f"\n{'='*60}")
    print(f"  DAgger BC Results")
    print(f"{'='*60}")
    print(f"  Total states: {N}")
    print(f"  Total rollouts: {total_rollouts/1e6:.1f}M")
    evs = rewards
    print(f"  EV range: [{evs.min():.1f}, {evs.max():.1f}], mean={evs.mean():.2f}")
    print(f"  Bust rate: {(busted > 0.5).sum()} ({100*(busted > 0.5).mean():.1f}%)")
    print(f"  Turn distribution: {np.bincount(turns, minlength=5).tolist()}")
    print(f"  Time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Saved: {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
