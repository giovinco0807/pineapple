"""
OFC Pineapple - Automated MCTS Parameter Optimization via Optuna

Bayesian optimization over MCTS hyperparameters:
  - bust_penalty: VN bust_prob coefficient
  - fl_ev_scale: FL EV chain value multiplier
  - c_puct: UCB exploration constant
  - vn_top_k: VN prefilter candidate count

Usage:
    python -m ai.auto_tune --trials 30 --games-per-trial 50 --seed 42
    python -m ai.auto_tune --trials 50 --games-per-trial 50 --seed 42 --n-jobs 2
"""
import sys
import random
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import optuna
from optuna.samplers import TPESampler

from ai.engine.encoding import Board, Observation, ALL_CARDS, encode_state
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, create_action_mask
)
from ai.engine.game_engine import (
    GameEngine, Hand, evaluate_hand,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    check_fl_entry, RANK_VALUES,
)
from ai.models.networks import PolicyNetwork, ValueNetwork
from ai.mcts.rollout_evaluator import RolloutEvaluator
from ai.mcts.multi_turn_mcts import MultiTurnMCTS, MultiTurnConfig
from ai.rust_solver_wrapper import RustFLSolver
from ai.engine.scoring import check_fl_stay_from_cards


# ─── Shared game logic (from eval_mcts.py) ──────────────────────────

SUIT_MAP = {'s': 0, 'h': 1, 'd': 2, 'c': 3}


def card_str_to_rust(card_str):
    if card_str.startswith('X'):
        return (0, 4)
    rank = RANK_VALUES.get(card_str[:-1], 0)
    suit = SUIT_MAP.get(card_str[-1], 0)
    return (rank, suit)


def rust_card_to_str(rank, suit):
    if suit == 4 or rank == 0:
        return 'X1'
    rank_map = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
                9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
    suit_map = {0: 's', 1: 'h', 2: 'd', 3: 'c'}
    return rank_map.get(rank, '?') + suit_map.get(suit, '?')


def play_fl_round(fl_solver, fl_card_count, available_cards):
    if len(available_cards) < fl_card_count:
        return 0, [], [], [], [], False, 0
    dealt = available_cards[:fl_card_count]
    rust_cards = [card_str_to_rust(c) for c in dealt]
    result = fl_solver.solve(rust_cards)
    if result is None:
        return 0, [], [], [], [], False, 0
    royalty = (result.get('top_royalty', 0) + result.get('middle_royalty', 0)
               + result.get('bottom_royalty', 0))
    top_strs = [rust_card_to_str(c['rank'], c['suit']) for c in result.get('top', [])]
    mid_strs = [rust_card_to_str(c['rank'], c['suit']) for c in result.get('middle', [])]
    bot_strs = [rust_card_to_str(c['rank'], c['suit']) for c in result.get('bottom', [])]
    can_stay, next_fl = check_fl_stay_from_cards(
        top_strs, bot_strs, fl_card_count, middle_cards=mid_strs
    )
    return royalty, top_strs, mid_strs, bot_strs, [], can_stay, next_fl


def play_fl_chain(fl_solver, initial_fl_cards, deck, deck_idx):
    total_royalty = 0
    total_rounds = 0
    fl_boards = []
    fl_cards = initial_fl_cards
    pool = list(deck[deck_idx:])
    total_consumed = len(pool)
    while fl_cards > 0:
        random.shuffle(pool)
        royalty, top, mid, bot, _, can_stay, new_fl = play_fl_round(
            fl_solver, fl_cards, pool)
        if not top:
            break
        total_royalty += royalty
        total_rounds += 1
        fl_boards.append((top, mid, bot, royalty))
        placed = set(top + mid + bot)
        pool = [c for c in pool if c not in placed]
        if can_stay and new_fl > 0:
            fl_cards = new_fl
        else:
            break
    return total_royalty, total_rounds, fl_boards, total_consumed


def bc_greedy(policy_net, obs, actions=None):
    if actions is None:
        actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    if not actions:
        return None
    if len(actions) == 1:
        return actions[0]
    dev = next(policy_net.parameters()).device
    state_t = torch.FloatTensor(encode_state(obs)).unsqueeze(0).to(dev)
    mask = create_action_mask(actions)
    mask_t = torch.BoolTensor(mask).unsqueeze(0).to(dev)
    with torch.no_grad():
        probs = policy_net(state_t, mask_t).squeeze(0).cpu().numpy()
    return actions[int(np.argmax(probs[:len(actions)]))]


def direct_eval_last_turn(obs):
    valid = get_turn_actions(obs.dealt_cards, obs.board_self)
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    best_s = float("-inf")
    best_a = valid[0]
    for action in valid:
        t = list(obs.board_self.top)
        m = list(obs.board_self.middle)
        b = list(obs.board_self.bottom)
        for c, p in action.placements:
            if p == 'top': t.append(c)
            elif p == 'middle': m.append(c)
            else: b.append(c)
        my_b = Board(top=t, middle=m, bottom=b)
        s = RolloutEvaluator._compute_score(my_b, obs.board_opponent)
        if s > best_s:
            best_s = s
            best_a = action
    return best_a


def play_game(deck, mcts, rollout_eval, policy_net, fl_solver):
    """Play one game. Hero: MCTS(T0) + Rollout(T1+). Opponent: BC greedy."""
    hand = Hand(deck=list(deck), btn=0)
    hero = 0

    # T0
    for seat in [hand.btn, 1 - hand.btn]:
        obs = hand.get_observation(seat)
        if seat == hero:
            _, action = mcts.select_action(obs)
        else:
            va = get_initial_actions(obs.dealt_cards, obs.board_self)
            action = bc_greedy(policy_net, obs, actions=va)
        if action:
            hand.apply_action(seat, action)

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
            cc = obs.board_self.card_count()
            if seat == hero:
                if cc == 11:
                    action = direct_eval_last_turn(obs)
                else:
                    _, action = rollout_eval.select_action(obs)
            else:
                if cc == 11:
                    action = direct_eval_last_turn(obs)
                else:
                    action = bc_greedy(policy_net, obs)
            if action:
                hand.apply_action(seat, action)

    # Score
    result = GameEngine.compute_result(hand)
    normal_score = result.raw_score[hero]

    # FL rounds
    fl_deck = list(ALL_CARDS)
    used = set()
    for seat in range(2):
        used.update(hand.boards[seat].all_cards())
        used.update(hand.discards[seat])
    fl_deck = [c for c in fl_deck if c not in used]
    random.shuffle(fl_deck)

    hero_fl_rounds = opp_fl_rounds = 0
    hero_fl_boards = opp_fl_boards = []
    fl_idx = 0

    if result.fl_entry[hero] and not result.busted[hero]:
        _, hero_fl_rounds, hero_fl_boards, used_n = play_fl_chain(
            fl_solver, result.fl_card_count[hero], fl_deck, fl_idx)
        fl_idx += used_n
    if result.fl_entry[1 - hero] and not result.busted[1 - hero]:
        _, opp_fl_rounds, opp_fl_boards, _ = play_fl_chain(
            fl_solver, result.fl_card_count[1 - hero], fl_deck, fl_idx)

    fl_delta = 0
    if hero_fl_rounds > 0 and opp_fl_rounds > 0:
        # Both entered FL - simplified scoring
        h_roy = sum(r for _, _, _, r in hero_fl_boards)
        o_roy = sum(r for _, _, _, r in opp_fl_boards)
        fl_delta = (6 + h_roy) * hero_fl_rounds - (6 + o_roy) * opp_fl_rounds
    elif hero_fl_rounds > 0:
        for _, _, _, r in hero_fl_boards:
            fl_delta += 6 + r
    elif opp_fl_rounds > 0:
        for _, _, _, r in opp_fl_boards:
            fl_delta -= (6 + r)

    return {
        'busted': result.busted[hero],
        'fl_entry': result.fl_entry[hero] and not result.busted[hero],
        'total_score': normal_score + fl_delta,
    }


# ─── Optuna Objective ──────────────────────────────────────────

def create_objective(policy, vn, ns, decks, t1_rollouts, mcts_sims, device='cpu'):
    """Create Optuna objective function with shared models and decks."""
    fl_solver = RustFLSolver()

    rollout_t1 = RolloutEvaluator(
        policy_net=policy, n_rollouts=t1_rollouts, top_k=20,
        device=device, value_net=vn, vn_top_k=10, norm_stats=ns,
    )

    def objective(trial):
        # Sample parameters
        bust_penalty = trial.suggest_float("bust_penalty", 0.0, 20.0)
        fl_ev_scale = trial.suggest_float("fl_ev_scale", 0.3, 3.0)
        c_puct = trial.suggest_float("c_puct", 0.5, 3.0)
        vn_top_k = trial.suggest_int("vn_top_k", 6, 16)

        config = MultiTurnConfig(
            num_simulations=mcts_sims,
            c_puct=c_puct,
            vn_top_k=vn_top_k,
            bust_penalty=bust_penalty,
            fl_ev_scale=fl_ev_scale,
        )
        mcts = MultiTurnMCTS(
            policy_net=policy, value_net=vn,
            config=config, device=device, norm_stats=ns,
        )

        # Play games
        busts = 0
        scores = []
        fl_entries = 0

        for i, deck in enumerate(decks):
            r = play_game(deck, mcts, rollout_t1, policy, fl_solver)
            if r['busted']:
                busts += 1
            if r['fl_entry']:
                fl_entries += 1
            scores.append(r['total_score'])

            # Report intermediate for pruning
            if (i + 1) % 10 == 0:
                trial.report(np.mean(scores), i + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        n = len(decks)
        avg_score = np.mean(scores)
        bust_rate = busts / n
        fl_rate = fl_entries / n

        # Log metrics
        trial.set_user_attr("avg_score", float(avg_score))
        trial.set_user_attr("bust_rate", float(bust_rate))
        trial.set_user_attr("fl_rate", float(fl_rate))

        # Objective: maximize score, penalize high bust rate
        # bust_rate > 40% gets penalized proportionally
        bust_excess = max(0, bust_rate - 0.40)
        objective_value = avg_score - 50.0 * bust_excess

        return objective_value

    return objective


# ─── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MCTS Auto-Tune via Optuna")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--games-per-trial", type=int, default=50)
    parser.add_argument("--mcts-sims", type=int, default=400)
    parser.add_argument("--t1-rollouts", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bc-model", default="ai/models/selfplay_iter17/bc_policy_best.pt")
    parser.add_argument("--vn-model", default="ai/models/value_v1/value_best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--db", default=None, help="Optuna DB URL for persistence")
    parser.add_argument("--study-name", default="mcts_tune")
    args = parser.parse_args()

    print("=" * 60)
    print("  MCTS Auto-Tune (Optuna)")
    print("=" * 60)

    device = args.device
    print(f"  Device: {device}")

    # Load models
    policy = PolicyNetwork()
    ck = torch.load(args.bc_model, map_location='cpu', weights_only=True)
    policy.load_state_dict(ck.get('model_state_dict', ck))
    policy.eval().to(device)

    vn = ValueNetwork()
    vk = torch.load(args.vn_model, map_location='cpu', weights_only=True)
    vn.load_state_dict(vk.get('model_state_dict', vk))
    vn.eval().to(device)
    ns = vk.get('norm_stats', None)

    # Pre-generate decks (same decks for all trials → fair comparison)
    rng = random.Random(args.seed)
    decks = []
    for _ in range(args.games_per_trial):
        d = list(ALL_CARDS)
        rng.shuffle(d)
        decks.append(d)

    print(f"  Trials: {args.trials}")
    print(f"  Games/trial: {args.games_per_trial}")
    print(f"  MCTS sims: {args.mcts_sims}")
    print(f"  T1+ rollouts: {args.t1_rollouts}")
    print(f"  Seed: {args.seed}")
    print()

    # Create study
    storage = args.db if args.db else None
    sampler = TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    objective = create_objective(
        policy, vn, ns, decks, args.t1_rollouts, args.mcts_sims, device=device)

    t_start = time.time()

    def trial_callback(study, trial):
        elapsed = time.time() - t_start
        if trial.value is None:
            print(f"  Trial {trial.number:3d} [{elapsed:.0f}s] PRUNED", flush=True)
            return
        attrs = trial.user_attrs
        score = attrs.get('avg_score', 0.0)
        bust = attrs.get('bust_rate', 0.0)
        fl = attrs.get('fl_rate', 0.0)
        print(f"  Trial {trial.number:3d} [{elapsed:.0f}s] "
              f"obj={trial.value:+.2f}  score={score:+.2f}  "
              f"bust={bust:.1%}  fl={fl:.1%}  "
              f"params={trial.params}", flush=True)

    study.optimize(objective, n_trials=args.trials, callbacks=[trial_callback])

    total_time = time.time() - t_start

    # Report
    print()
    print("=" * 60)
    print(f"  Optimization Complete ({total_time:.0f}s)")
    print("=" * 60)

    best = study.best_trial
    print(f"  Best Trial: #{best.number}")
    print(f"  Objective:  {best.value:+.2f}")
    print(f"  Score:      {best.user_attrs['avg_score']:+.2f}")
    print(f"  Bust Rate:  {best.user_attrs['bust_rate']:.1%}")
    print(f"  FL Rate:    {best.user_attrs['fl_rate']:.1%}")
    print(f"  Parameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print()

    # Top 5 trials
    print("  Top 5 Trials:")
    trials = sorted(study.trials, key=lambda t: t.value if t.value else -999,
                    reverse=True)
    for t in trials[:5]:
        if t.value is None:
            continue
        attrs = t.user_attrs
        print(f"    #{t.number:3d} obj={t.value:+.2f}  "
              f"score={attrs.get('avg_score', 0):+.2f}  "
              f"bust={attrs.get('bust_rate', 0):.1%}  "
              f"fl={attrs.get('fl_rate', 0):.1%}  "
              f"bp={t.params.get('bust_penalty', 0):.1f} "
              f"fl_s={t.params.get('fl_ev_scale', 0):.2f} "
              f"cp={t.params.get('c_puct', 0):.2f} "
              f"vk={t.params.get('vn_top_k', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
