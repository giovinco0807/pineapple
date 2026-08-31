"""
Self-Play v3: Full FL playthrough with Rust FL Solver.

Complete game flow:
- Normal hand played (Rollout T0 + BC T1-4)
- FL rounds played out with Rust solver
- Both-player FL supported
- Opponent plays normal hand during FL
- Accurate scoring: lines + scoop + royalties via compute_score_raw

Usage:
    python ai/self_play.py --games 50 --rollouts 20 --workers 12 --output data/selfplay.jsonl
"""
import sys
import random
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from ai.engine.encoding import Board, Observation, encode_state, ALL_CARDS, RANKS, SUITS
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, create_action_mask, encode_action
)
from ai.engine.game_engine import (
    GameEngine, Hand, evaluate_hand, check_fl_entry as _check_fl_entry_tuple,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    evaluate_board_with_joker_constraint,
)
from ai.engine.scoring import check_fl_stay_from_cards
from ai.engine.turn_order import action_order
from ai.models.networks import PolicyNetwork
from ai.models.action_value_reranker import ActionValueReranker
from ai.mcts.rollout_evaluator import RolloutEvaluator


# ─── Card format conversion ─────────────────────────────────────────

RANK_TO_INT = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
               'T':10,'J':11,'Q':12,'K':13,'A':14}
SUIT_TO_INT = {'s':0,'h':1,'d':2,'c':3}
INT_TO_RANK = {v: k for k, v in RANK_TO_INT.items()}
INT_TO_SUIT = {v: k for k, v in SUIT_TO_INT.items()}


def card_str_to_tuple(card_str):
    """Convert 'Ah' -> (14,1), 'X1' -> (0,4), 'X2' -> (0,4)."""
    if card_str in ("X1", "X2"):
        return (0, 4)  # Joker
    rank = RANK_TO_INT.get(card_str[:-1], 0)
    suit = SUIT_TO_INT.get(card_str[-1], 0)
    return (rank, suit)


def tuple_to_card_str(rank, suit):
    """Convert (14,1) -> 'Ah', (0,4) -> 'X1'."""
    if suit == 4 or rank == 0:
        return "X1"  # Joker
    r = INT_TO_RANK.get(rank, '?')
    s = INT_TO_SUIT.get(suit, '?')
    return f"{r}{s}"


def solver_row_to_cards(row_list):
    """Convert solver output row [{"rank":14,"suit":1}, ...] to card strings."""
    return [tuple_to_card_str(c['rank'], c['suit']) for c in row_list]


def build_board_from_solver(placement):
    """Build a Board object from Rust FL solver placement result."""
    board = Board()
    board.top = solver_row_to_cards(placement['top'])
    board.middle = solver_row_to_cards(placement['middle'])
    board.bottom = solver_row_to_cards(placement['bottom'])
    return board


# ─── FL detection (delegated to engine) ──────────────────────────────

def check_fl_entry(top_cards):
    """Check if top row qualifies for Fantasyland (QQ+ pair or trips)."""
    return _check_fl_entry_tuple(top_cards)[0]


def get_fl_card_count(top_cards):
    """QQ=14, KK=15, AA=16, Trips=17."""
    return _check_fl_entry_tuple(top_cards)[1]


# ─── Worker functions ────────────────────────────────────────────────

def init_worker(
    model_path,
    n_rollouts,
    top_k,
    fl_enrich=0.0,
    deep_all_turns=False,
    full_width=False,
    policy_playout=False,
    record_action_scores=False,
    base_seed=None,
    reranker_path=None,
    reranker_turn_models="",
    reranker_top_k=0,
    reranker_t0_top_k=0,
    reranker_t2_top_k=0,
    reranker_bust_weight=0.0,
    reranker_fl_weight=0.0,
    reranker_fl_any_weight=0.0,
    reranker_fl_qq_weight=0.0,
    reranker_fl_kk_weight=0.0,
    reranker_fl_aa_weight=0.0,
    reranker_fl_trips_weight=0.0,
    reranker_suit_ensemble_turns="",
    reranker_suit_ensemble_size=8,
):
    """Initialize model + FL solver in each worker."""
    global _policy_net, _rollout_eval, _rollout_eval_light, _fl_solver, _action_value_net
    global _action_value_nets_by_turn, _reranker_turn_model_paths
    global _fl_enrich_rate, _deep_all_turns, _record_action_scores, _base_seed
    global _eval_rollouts, _full_width_candidates, _reranker_path
    global _reranker_top_k, _reranker_top_k_by_turn
    _fl_enrich_rate = fl_enrich
    _deep_all_turns = deep_all_turns
    _record_action_scores = record_action_scores
    _base_seed = base_seed
    _eval_rollouts = int(n_rollouts)
    _full_width_candidates = bool(full_width)
    _reranker_path = reranker_path
    _reranker_turn_model_paths = _parse_reranker_turn_models(reranker_turn_models)
    _reranker_top_k = int(reranker_top_k or 0)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    input_dim = state_dict.get('net.0.weight', torch.empty(0, 522)).shape[1]
    _policy_net = PolicyNetwork(input_dim=input_dim)
    _policy_net.load_state_dict(state_dict)
    _policy_net.eval()
    _action_value_net = None
    if reranker_path:
        _action_value_net = ActionValueReranker.from_checkpoint(reranker_path, map_location='cpu')
        _action_value_net.eval()
    _action_value_nets_by_turn = {}
    for turn, path in _reranker_turn_model_paths.items():
        net = ActionValueReranker.from_checkpoint(path, map_location='cpu')
        net.eval()
        _action_value_nets_by_turn[int(turn)] = net
    reranker_top_k_by_turn = {}
    if reranker_top_k > 0 and reranker_t0_top_k > 0:
        reranker_top_k_by_turn[0] = reranker_t0_top_k
    if reranker_top_k > 0 and reranker_t2_top_k > 0:
        reranker_top_k_by_turn[2] = reranker_t2_top_k
    _reranker_top_k_by_turn = dict(reranker_top_k_by_turn)
    if isinstance(reranker_suit_ensemble_turns, str):
        suit_ensemble_turns = {
            int(turn) for turn in reranker_suit_ensemble_turns.split(",") if turn.strip()
        }
    else:
        suit_ensemble_turns = {int(turn) for turn in (reranker_suit_ensemble_turns or [])}
    _rollout_eval = RolloutEvaluator(
        policy_net=_policy_net,
        n_rollouts=n_rollouts,
        top_k=top_k,
        device='cpu',
        use_policy_playout=policy_playout,
        full_width=full_width,
        action_value_net=_action_value_net,
        action_value_nets_by_turn=_action_value_nets_by_turn,
        action_value_top_k=reranker_top_k,
        action_value_top_k_by_turn=reranker_top_k_by_turn,
        action_value_bust_weight=reranker_bust_weight,
        action_value_fl_weight=reranker_fl_weight,
        action_value_fl_any_weight=reranker_fl_any_weight,
        action_value_fl_qq_weight=reranker_fl_qq_weight,
        action_value_fl_kk_weight=reranker_fl_kk_weight,
        action_value_fl_aa_weight=reranker_fl_aa_weight,
        action_value_fl_trips_weight=reranker_fl_trips_weight,
        action_value_suit_ensemble_turns=suit_ensemble_turns,
        action_value_suit_ensemble_size=reranker_suit_ensemble_size,
    )
    _rollout_eval_light = RolloutEvaluator(
        policy_net=_policy_net,
        n_rollouts=0 if n_rollouts <= 0 else max(n_rollouts // 2, 10),
        top_k=top_k,
        device='cpu',
        use_policy_playout=policy_playout,
        full_width=full_width,
        action_value_net=_action_value_net,
        action_value_nets_by_turn=_action_value_nets_by_turn,
        action_value_top_k=reranker_top_k,
        action_value_top_k_by_turn=reranker_top_k_by_turn,
        action_value_bust_weight=reranker_bust_weight,
        action_value_fl_weight=reranker_fl_weight,
        action_value_fl_any_weight=reranker_fl_any_weight,
        action_value_fl_qq_weight=reranker_fl_qq_weight,
        action_value_fl_kk_weight=reranker_fl_kk_weight,
        action_value_fl_aa_weight=reranker_fl_aa_weight,
        action_value_fl_trips_weight=reranker_fl_trips_weight,
        action_value_suit_ensemble_turns=suit_ensemble_turns,
        action_value_suit_ensemble_size=reranker_suit_ensemble_size,
    )
    from ai.rust_solver_wrapper import RustFLSolver
    _fl_solver = RustFLSolver()


def _parse_reranker_turn_models(value):
    """Parse "turn=checkpoint,turn=checkpoint" into a turn-path mapping."""
    if not value:
        return {}
    if isinstance(value, dict):
        return {int(turn): str(path) for turn, path in value.items() if str(path)}
    mapping = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --reranker-turn-models item: {item!r}")
        turn_text, path = item.split("=", 1)
        mapping[int(turn_text.strip())] = path.strip()
    return mapping


def bc_select(obs):
    """BC greedy action selection."""
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


def _select_last_turn_direct(obs):
    """Direct evaluation for the last turn (no rollout needed).

    Board has 11 cards. Place 2 cards, discard 1 → complete board.
    Uses full scoring including opponent comparison and FL EV.
    """
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    if not valid_actions:
        return None
    if len(valid_actions) == 1:
        return valid_actions[0]

    best_score = float("-inf")
    best_action = valid_actions[0]

    for action in valid_actions:
        # Build completed board
        top = list(obs.board_self.top)
        mid = list(obs.board_self.middle)
        bot = list(obs.board_self.bottom)
        for card, pos in action.placements:
            if pos == 'top':
                top.append(card)
            elif pos == 'middle':
                mid.append(card)
            else:
                bot.append(card)

        my_board = Board(top=top, middle=mid, bottom=bot)
        score = RolloutEvaluator._compute_score(my_board, obs.board_opponent)

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def _select_last_turn_direct_with_scores(obs):
    """Direct last-turn evaluation with per-action scores for teacher data."""
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    if not valid_actions:
        return -1, None, []
    scores = [float("-inf")] * len(valid_actions)
    best_idx = 0
    best_score = float("-inf")

    for i, action in enumerate(valid_actions):
        top = list(obs.board_self.top)
        mid = list(obs.board_self.middle)
        bot = list(obs.board_self.bottom)
        for card, pos in action.placements:
            if pos == 'top':
                top.append(card)
            elif pos == 'middle':
                mid.append(card)
            else:
                bot.append(card)

        my_board = Board(top=top, middle=mid, bottom=bot)
        score = RolloutEvaluator._compute_score(my_board, obs.board_opponent)
        scores[i] = score
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx, valid_actions[best_idx], scores


def _active_reranker_budget(turn):
    budget = int(globals().get("_reranker_top_k", 0) or 0)
    turn_budget = int(globals().get("_reranker_top_k_by_turn", {}).get(int(turn), 0) or 0)
    if turn_budget > 0:
        budget = max(budget, turn_budget)
    return budget


def _active_reranker_path(turn):
    turn_paths = globals().get("_reranker_turn_model_paths", {})
    return turn_paths.get(int(turn), globals().get("_reranker_path", None))


def _sims_from_eval_mode(eval_mode, default=0):
    if isinstance(eval_mode, str) and eval_mode.startswith("mc"):
        try:
            return int(eval_mode[2:])
        except ValueError:
            return int(default or 0)
    return 0


def _teacher_payload(obs, action, scores=None, eval_mode=None):
    """Return action index and optional finite EV labels aligned to action ids."""
    if obs.turn == 0:
        valid_actions = get_initial_actions(obs.dealt_cards, obs.board_self)
    else:
        valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)

    action_idx = encode_action(
        action,
        valid_actions,
        turn=obs.turn,
        dealt_cards=obs.dealt_cards,
    )
    eval_rollouts = int(globals().get("_eval_rollouts", 0) or 0)
    full_width = bool(globals().get("_full_width_candidates", False))
    if eval_mode is None:
        active_path = _active_reranker_path(obs.turn)
        if active_path and eval_rollouts <= 0:
            eval_mode = "action_value"
        elif eval_rollouts > 0:
            eval_mode = f"mc{eval_rollouts}"
        else:
            eval_mode = "policy"
    reranker_path = _active_reranker_path(obs.turn)
    uses_reranker_budget = eval_mode not in ("exact", "policy")
    reranker_budget = (
        _active_reranker_budget(obs.turn)
        if reranker_path and not full_width and uses_reranker_budget
        else 0
    )

    payload = {
        "action_idx": int(action_idx),
        "n_actions": len(valid_actions),
        "eval_mode": eval_mode,
        "sims": _sims_from_eval_mode(eval_mode, eval_rollouts),
        "estimated": eval_mode != "exact",
        "pruned_top_k": reranker_budget if reranker_budget > 0 else 0,
        "prune_model": str(reranker_path) if reranker_path and reranker_budget > 0 else None,
    }
    if scores is not None:
        evs = []
        idxs = []
        for i, score in enumerate(scores):
            if i >= len(valid_actions) or not np.isfinite(score):
                continue
            idx = encode_action(
                valid_actions[i],
                valid_actions,
                turn=obs.turn,
                dealt_cards=obs.dealt_cards,
            )
            idxs.append(int(idx))
            evs.append(float(score))
        payload["evaluated_actions"] = len(idxs)
        payload["top_k_indices"] = idxs
        payload["action_evs"] = evs
    else:
        payload["evaluated_actions"] = 0
    return payload


def play_normal_hand_full(deck):
    """Play one normal hand. Returns (turn_records, boards, fl_info)."""
    hand = Hand(deck=list(deck), btn=0)
    turn_records = []

    # Turn 0 (rollout)
    for seat in action_order(hand.btn):
        obs = hand.get_observation(seat)
        scores = None
        if _record_action_scores:
            _, action, scores = _rollout_eval.select_action_with_scores(obs)
        else:
            _, action = _rollout_eval.select_action(obs)
        if action:
            teacher = _teacher_payload(obs, action, scores)
            turn_records.append({
                'seat': seat, 'turn': 0,
                'board_self': obs.board_self.to_dict(),
                'board_opponent': obs.board_opponent.to_dict(),
                'dealt_cards': list(obs.dealt_cards),
                'discards_self': list(obs.known_discards_self),
                'is_btn': seat == hand.btn,
                'placements': [[c, p] for c, p in action.placements],
                'discard': action.discard,
                **teacher,
            })
            hand.apply_action(seat, action)

    # Turns 1-4 (Rollout, with direct eval on last turn)
    for turn_num in range(1, 9):
        if hand.is_hand_complete():
            break
        hand.deal_next_turn()
        for seat in action_order(hand.btn):
            cards = hand.dealt_cards[seat]
            if not cards or hand.boards[seat].is_complete():
                continue
            obs = hand.get_observation(seat)
            board = obs.board_self
            card_count = len(board.top) + len(board.middle) + len(board.bottom)
            scores = None
            if card_count == 11:
                # Last turn: direct evaluation (no rollout needed)
                if _record_action_scores:
                    _, action, scores = _select_last_turn_direct_with_scores(obs)
                else:
                    action = _select_last_turn_direct(obs)
                eval_mode = "exact"
            elif _deep_all_turns:
                if _record_action_scores:
                    _, action, scores = _rollout_eval.select_action_with_scores(obs)
                else:
                    _, action = _rollout_eval.select_action(obs)
                eval_mode = None
            else:
                if _record_action_scores:
                    _, action, scores = _rollout_eval_light.select_action_with_scores(obs)
                else:
                    _, action = _rollout_eval_light.select_action(obs)
                light_rollouts = int(globals().get("_eval_rollouts", 0) or 0)
                light_rollouts = 0 if light_rollouts <= 0 else max(light_rollouts // 2, 10)
                eval_mode = f"mc{light_rollouts}" if light_rollouts > 0 else None
            if action:
                teacher = _teacher_payload(obs, action, scores, eval_mode=eval_mode)
                turn_records.append({
                    'seat': seat, 'turn': turn_num,
                    'board_self': obs.board_self.to_dict(),
                    'board_opponent': obs.board_opponent.to_dict(),
                    'dealt_cards': list(obs.dealt_cards),
                    'discards_self': list(obs.known_discards_self),
                    'is_btn': seat == hand.btn,
                    'placements': [[c, p] for c, p in action.placements],
                    'discard': action.discard,
                    **teacher,
                })
                hand.apply_action(seat, action)

    # Compute FL info per seat
    fl_info = {}
    for seat in [0, 1]:
        board = hand.boards[seat]
        evaluated = evaluate_board_with_joker_constraint(
            board.top, board.middle, board.bottom
        )
        is_busted = bool(evaluated["busted"])
        is_fl = bool(evaluated["fl_entry"])
        fl_cards = int(evaluated["fl_card_count"])
        roy = int(evaluated["royalties"]["total"])
        fl_info[seat] = {
            'busted': is_busted,
            'fl_entry': is_fl,
            'fl_cards': fl_cards,
            'royalties': roy,
        }

    return turn_records, hand.boards, fl_info


def solve_fl_round(fl_card_count):
    """Deal and solve one FL round. Returns (board, can_stay, fl_card_count_next)."""
    # Deal random FL hand
    deck = list(ALL_CARDS)
    random.shuffle(deck)
    fl_hand_strs = deck[:fl_card_count]
    fl_tuples = [card_str_to_tuple(c) for c in fl_hand_strs]

    result = _fl_solver.solve(fl_tuples)
    if result is None:
        # Solver failure: return busted board
        board = Board()
        return board, False, 0

    board = build_board_from_solver(result)
    can_stay, next_fl_cards = check_fl_stay_from_cards(
        board.top,
        board.bottom,
        fl_card_count,
        middle_cards=board.middle,
    )

    return board, can_stay, next_fl_cards


_fl_enrich_rate = 0.0  # Set by init_worker
_base_seed = None  # Set by init_worker


def _enrich_deck_for_fl(deck):
    """Rejection sampling: re-shuffle until T0 deal contains A or K for at least one player.
    Gives up after 20 attempts to avoid infinite loops."""
    for _ in range(20):
        random.shuffle(deck)
        # T0 deals 5 cards to each player (positions 0-4 and 5-9)
        p0_cards = deck[:5]
        p1_cards = deck[5:10]
        has_ak = any(c[0] in ('A', 'K') for c in p0_cards + p1_cards if len(c) >= 2 and c[0] != 'X')
        if has_ak:
            return
    # Fallback: use whatever shuffle we have


def play_one_game(_game_idx):
    """
    Play one full game including FL chains.

    Game flow:
      1. Normal hand → score → FL check
      2. If FL: play FL rounds (FL vs normal, FL vs FL)
      3. Accumulate total score
      4. reward = total score from each player's perspective
    """
    if _base_seed is not None:
        random.seed(_base_seed + _game_idx)
        np.random.seed((_base_seed + _game_idx) % (2**32 - 1))

    deck = list(ALL_CARDS)
    if _fl_enrich_rate > 0 and random.random() < _fl_enrich_rate:
        _enrich_deck_for_fl(deck)
    else:
        random.shuffle(deck)

    # ─── Round 1: Normal hand ───
    turn_records, boards, fl_info = play_normal_hand_full(deck)

    # Score the normal hand (no FL_EV)
    score_0 = RolloutEvaluator.compute_score_raw(boards[0], boards[1])
    total_scores = {0: score_0, 1: -score_0}

    # Build hand_result for logging
    hand_result = {
        "royalties": {str(s): {"total": fl_info[s]['royalties']} for s in [0, 1]},
        "busted": {str(s): fl_info[s]['busted'] for s in [0, 1]},
        "fl_entry": {str(s): fl_info[s]['fl_entry'] for s in [0, 1]},
        "fl_cards": {str(s): fl_info[s]['fl_cards'] for s in [0, 1]},
    }

    total_fl_rounds = 0
    total_fl_bonus = 0.0

    # ─── FL chain ───
    # Track which seats are in FL, with their current card count
    fl_active = {}
    for seat in [0, 1]:
        if fl_info[seat]['fl_entry']:
            fl_active[seat] = fl_info[seat]['fl_cards']

    max_fl_chain = 20  # Safety limit
    fl_round = 0

    while fl_active and fl_round < max_fl_chain:
        fl_round += 1
        total_fl_rounds += 1

        # ─── Case C: Both in FL ───
        if len(fl_active) == 2:
            fl_boards = {}
            fl_stay = {}
            fl_next = {}
            for seat in [0, 1]:
                board, can_stay, next_cards = solve_fl_round(fl_active[seat])
                fl_boards[seat] = board
                fl_stay[seat] = can_stay
                fl_next[seat] = next_cards

            # Score FL vs FL
            round_score = RolloutEvaluator.compute_score_raw(
                fl_boards[0], fl_boards[1]
            )
            total_scores[0] += round_score
            total_scores[1] -= round_score
            total_fl_bonus += abs(round_score)

            # Update FL active
            new_active = {}
            for seat in [0, 1]:
                if fl_stay[seat] and fl_next[seat] > 0:
                    new_active[seat] = fl_next[seat]
            fl_active = new_active

        # ─── Case B: One in FL ───
        else:
            fl_seat = list(fl_active.keys())[0]
            normal_seat = 1 - fl_seat

            # FL seat: solver
            fl_board, can_stay, next_cards = solve_fl_round(fl_active[fl_seat])

            # Normal seat: play a normal hand
            normal_deck = list(ALL_CARDS)
            random.shuffle(normal_deck)
            normal_hand = Hand(deck=normal_deck, btn=0)

            # Play normal hand with rollout T0 + BC T1-4
            for seat_order in [0, 1]:
                obs = normal_hand.get_observation(seat_order)
                _, action = _rollout_eval.select_action(obs)
                if action:
                    normal_hand.apply_action(seat_order, action)

            for turn_num in range(1, 9):
                if normal_hand.is_hand_complete():
                    break
                normal_hand.deal_next_turn()
                for seat_order in [0, 1]:
                    cards = normal_hand.dealt_cards[seat_order]
                    if not cards or normal_hand.boards[seat_order].is_complete():
                        continue
                    obs = normal_hand.get_observation(seat_order)
                    action = bc_select(obs)
                    if action:
                        normal_hand.apply_action(seat_order, action)

            # Use seat 0 of the normal hand as the normal player's board
            normal_board = normal_hand.boards[0]

            # Score FL vs Normal
            if fl_seat == 0:
                round_score = RolloutEvaluator.compute_score_raw(
                    fl_board, normal_board
                )
            else:
                round_score = -RolloutEvaluator.compute_score_raw(
                    normal_board, fl_board
                )

            total_scores[0] += round_score
            total_scores[1] -= round_score
            total_fl_bonus += abs(round_score)

            # Update FL active
            if can_stay and next_cards > 0:
                fl_active = {fl_seat: next_cards}
            else:
                fl_active = {}

    # ─── Build JSONL output ───
    lines = []
    for tr in turn_records:
        seat = tr['seat']
        reward = total_scores[seat]
        record = {
            "turn_log": {
                "board_self": tr['board_self'],
                "board_opponent": tr['board_opponent'],
                "dealt_cards": tr['dealt_cards'],
                "discards_self": tr['discards_self'],
                "turn": tr['turn'],
                "is_btn": tr['is_btn'],
                "action": {"placements": tr['placements'], "discard": tr['discard']},
                "player": seat,
                "action_idx": tr.get("action_idx"),
                "n_actions": tr.get("n_actions"),
                "evaluated_actions": tr.get("evaluated_actions"),
                "eval_mode": tr.get("eval_mode"),
                "sims": tr.get("sims"),
                "estimated": tr.get("estimated"),
                "pruned_top_k": tr.get("pruned_top_k"),
                "prune_model": tr.get("prune_model"),
                "top_k_indices": tr.get("top_k_indices"),
                "action_evs": tr.get("action_evs"),
            },
            "hand_result": hand_result,
            "reward": reward,
            "fl_rounds": total_fl_rounds,
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    bust_count = sum(1 for s in [0, 1] if fl_info[s]['busted'])
    fl_count = sum(1 for s in [0, 1] if fl_info[s]['fl_entry'])
    return lines, bust_count, fl_count, total_fl_rounds, total_fl_bonus


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ai/models/selfplay_iter18/bc_policy_best.pt')
    parser.add_argument('--games', type=int, default=50)
    parser.add_argument('--game-start', type=int, default=0,
                        help='First deterministic game index when using --seed')
    parser.add_argument('--rollouts', type=int, default=20)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--output', default='data/selfplay.jsonl')
    parser.add_argument('--fl-enrich', type=float, default=0.0,
                        help='Fraction of games with FL-enriched deals (0.0-1.0)')
    parser.add_argument('--deep-all-turns', action='store_true',
                        help='Use the main rollout evaluator for T1-T3, not the light evaluator')
    parser.add_argument('--full-width', action='store_true',
                        help='Evaluate every legal candidate instead of policy/FL prefiltered candidates')
    parser.add_argument('--policy-playout', action='store_true',
                        help='Use BC-guided playout for future non-T4 turns inside rollouts')
    parser.add_argument('--record-action-scores', action='store_true',
                        help='Save per-candidate rollout/direct scores for soft-label training')
    parser.add_argument('--seed', type=int, default=None,
                        help='Base seed for deterministic per-game shuffles')
    parser.add_argument('--reranker', default=None,
                        help='Action-value reranker checkpoint for candidate scoring/prefiltering')
    parser.add_argument('--reranker-turn-models', default='',
                        help='Optional comma-separated turn=checkpoint overrides, e.g. 3=path/to/t3.pt')
    parser.add_argument('--reranker-top-k', type=int, default=0,
                        help='Keep this many reranker-scored candidates before rollout; with --rollouts 0, play reranker directly')
    parser.add_argument('--reranker-t0-top-k', type=int, default=0,
                        help='When --reranker-top-k is active, keep at least this many T0 candidates; 0 uses --reranker-top-k')
    parser.add_argument('--reranker-t2-top-k', type=int, default=24,
                        help='When --reranker-top-k is active, keep at least this many T2 candidates before rollout; 0 disables the T2 safety budget')
    parser.add_argument('--reranker-bust-weight', type=float, default=0.0,
                        help='Optional extra penalty applied to reranker bust probability')
    parser.add_argument('--reranker-fl-weight', type=float, default=0.0,
                        help='Backward-compatible alias for --reranker-fl-any-weight')
    parser.add_argument('--reranker-fl-any-weight', type=float, default=0.0,
                        help='Bonus applied to reranker FL-any probability')
    parser.add_argument('--reranker-fl-qq-weight', type=float, default=0.0,
                        help='Bonus applied to reranker QQ FL probability')
    parser.add_argument('--reranker-fl-kk-weight', type=float, default=0.0,
                        help='Bonus applied to reranker KK FL probability')
    parser.add_argument('--reranker-fl-aa-weight', type=float, default=0.0,
                        help='Bonus applied to reranker AA FL probability')
    parser.add_argument('--reranker-fl-trips-weight', type=float, default=0.0,
                        help='Bonus applied to reranker trips FL probability')
    parser.add_argument('--reranker-suit-ensemble-turns', default='',
                        help='Comma-separated turns where reranker scores are averaged over all 24 suit permutations, e.g. 2')
    parser.add_argument('--reranker-suit-ensemble-size', type=int, default=8,
                        help='Number of suit permutations to average on ensemble turns; use 24 for full ensemble')
    args = parser.parse_args()

    n_workers = args.workers or max(1, mp.cpu_count() - 2)
    print(f"Self-Play v3 (FL Solver integration)")
    print(f"  BC model: {args.model}")
    print(f"  Rollout: {args.rollouts} sims, top-{args.top_k}")
    print(f"  Deep all turns: {args.deep_all_turns}")
    print(f"  Full-width candidates: {args.full_width}")
    print(f"  Policy playout: {args.policy_playout}")
    print(f"  Record action scores: {args.record_action_scores}")
    print(f"  Reranker: {args.reranker or 'off'}")
    turn_model_paths = _parse_reranker_turn_models(args.reranker_turn_models)
    if turn_model_paths:
        formatted = ", ".join(f"T{turn}={path}" for turn, path in sorted(turn_model_paths.items()))
        print(f"  Reranker turn overrides: {formatted}")
    if args.reranker or turn_model_paths:
        mode = "direct" if args.rollouts <= 0 else f"prefilter top-{args.reranker_top_k}"
        print(
            f"  Reranker mode: {mode}, bust_w={args.reranker_bust_weight}, "
            f"fl_any_w={args.reranker_fl_weight + args.reranker_fl_any_weight}, "
            f"qq={args.reranker_fl_qq_weight}, kk={args.reranker_fl_kk_weight}, "
            f"aa={args.reranker_fl_aa_weight}, trips={args.reranker_fl_trips_weight}"
        )
        if args.rollouts > 0 and args.reranker_top_k > 0 and args.reranker_t0_top_k > 0:
            print(f"  Reranker T0 safety budget: top-{max(args.reranker_top_k, args.reranker_t0_top_k)}")
        if args.rollouts > 0 and args.reranker_top_k > 0 and args.reranker_t2_top_k > 0:
            print(f"  Reranker T2 safety budget: top-{max(args.reranker_top_k, args.reranker_t2_top_k)}")
        if args.reranker_suit_ensemble_turns:
            print(
                f"  Reranker suit ensemble turns: {args.reranker_suit_ensemble_turns} "
                f"(size={args.reranker_suit_ensemble_size})"
            )
    print(f"  Workers: {n_workers}")
    print(f"  Games: {args.games}")
    print(f"  Game start: {args.game_start}")
    print(f"  FL enrich: {args.fl_enrich:.0%}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {args.output}\n")

    t0 = time.time()
    total_turns = 0
    total_busts = 0
    total_fl = 0
    total_fl_rounds = 0
    total_fl_bonus = 0.0
    completed = 0

    with open(args.output, 'w', encoding='utf-8') as f:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_worker,
            initargs=(
                args.model,
                args.rollouts,
                args.top_k,
                args.fl_enrich,
                args.deep_all_turns,
                args.full_width,
                args.policy_playout,
                args.record_action_scores,
                args.seed,
                args.reranker,
                args.reranker_turn_models,
                args.reranker_top_k,
                args.reranker_t0_top_k,
                args.reranker_t2_top_k,
                args.reranker_bust_weight,
                args.reranker_fl_weight,
                args.reranker_fl_any_weight,
                args.reranker_fl_qq_weight,
                args.reranker_fl_kk_weight,
                args.reranker_fl_aa_weight,
                args.reranker_fl_trips_weight,
                args.reranker_suit_ensemble_turns,
                args.reranker_suit_ensemble_size,
            ),
        ) as pool:
            futures = {
                pool.submit(play_one_game, i): i
                for i in range(args.game_start, args.game_start + args.games)
            }
            for future in as_completed(futures):
                try:
                    lines, busts, fls, fl_rnds, fl_bonus = future.result()
                    for line in lines:
                        f.write(line + '\n')
                    total_turns += len(lines)
                    total_busts += busts
                    total_fl += fls
                    total_fl_rounds += fl_rnds
                    total_fl_bonus += fl_bonus
                    completed += 1
                    if completed % 10 == 0 or completed == args.games:
                        elapsed = time.time() - t0
                        bust_rate = total_busts / max(completed * 2, 1) * 100
                        fl_rate = total_fl / max(completed * 2, 1) * 100
                        rate = completed / elapsed * 60
                        print(f"  [{completed}/{args.games}] "
                              f"turns={total_turns} bust={bust_rate:.0f}% "
                              f"fl={fl_rate:.0f}% fl_rnds={total_fl_rounds} "
                              f"fl_bonus={total_fl_bonus:.0f} "
                              f"{rate:.1f} g/min ({elapsed:.0f}s)",
                              flush=True)
                except Exception as e:
                    completed += 1
                    print(f"  [ERROR] Game {futures[future]}: {e}", flush=True)

    elapsed = time.time() - t0
    bust_rate = total_busts / max(completed * 2, 1) * 100
    fl_rate = total_fl / max(completed * 2, 1) * 100
    rate = completed / elapsed * 60
    print(f"\nDone! {completed} games, {total_turns} turns")
    print(f"  bust={bust_rate:.0f}%, fl={fl_rate:.0f}%")
    print(f"  fl_rounds={total_fl_rounds}, fl_bonus_total={total_fl_bonus:.0f}")
    print(f"  {rate:.1f} games/min, {elapsed:.0f}s")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
