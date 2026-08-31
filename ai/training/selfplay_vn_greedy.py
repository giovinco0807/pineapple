"""
VN-Greedy Self-Play Data Generation

Plays hands using VN (Value Network) as the sole decision maker:
  - For each turn, enumerate all valid actions
  - Apply each action to the board, encode resulting state
  - VN scores each resulting state → pick the highest value
  - Record the trajectory with final hand outcomes

No PolicyNet or MCTS required — just VN greedy play.

Output: JSONL compatible with preprocess_mc_teacher.py

Usage:
    python -m ai.training.selfplay_vn_greedy \
        --model models/vn_v3_s200/value_best.pt \
        --norm models/vn_v3_s200/norm_stats.json \
        --hands 10000 --output data/selfplay_vn.jsonl
"""
import sys
import json
import random
import time
import copy
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import (
    Board, Observation, encode_state, ALL_CARDS, STATE_DIM,
)
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, Action,
)
from ai.engine.game_engine import GameEngine, Hand
from ai.engine.turn_order import action_order
from ai.models.networks import ValueNetworkV3


def apply_action_to_board(board: Board, action: Action) -> Board:
    """Apply an action to a board and return the new board (copy)."""
    new_top = list(board.top)
    new_mid = list(board.middle)
    new_bot = list(board.bottom)

    for card, pos in action.placements:
        if pos == "top":
            new_top.append(card)
        elif pos == "middle":
            new_mid.append(card)
        elif pos == "bottom":
            new_bot.append(card)

    return Board(top=new_top, middle=new_mid, bottom=new_bot)


class VNGreedyPlayer:
    """Plays OFC hands using VN greedy evaluation."""

    def __init__(self, model: ValueNetworkV3, device: str = "cpu",
                 norm_stats: dict = None, temperature: float = 0.0):
        self.model = model
        self.device = device
        self.model.eval()
        self.temperature = temperature  # 0 = greedy, > 0 = softmax sampling

        self.score_mean = 0.0
        self.score_std = 1.0
        if norm_stats:
            self.score_mean = norm_stats.get("mean", norm_stats.get("score_mean", 0.0))
            self.score_std = norm_stats.get("std", norm_stats.get("score_std", 1.0))

    @torch.no_grad()
    def select_action(self, obs: Observation) -> Tuple[int, Action, float]:
        """Select the best action by VN evaluation of resulting states.

        Returns:
            (best_index, best_action, best_vn_score)
        """
        if obs.turn == 0:
            actions = get_initial_actions(obs.dealt_cards, obs.board_self)
        else:
            actions = get_turn_actions(obs.dealt_cards, obs.board_self)

        if not actions:
            raise ValueError("No valid actions")
        if len(actions) == 1:
            return 0, actions[0], 0.0

        # Build resulting obs for each action and batch-evaluate
        states = []
        turns = []
        for action in actions:
            new_board = apply_action_to_board(obs.board_self, action)
            # Discard goes to known_discards
            new_discards = list(obs.known_discards_self)
            if action.discard:
                new_discards.append(action.discard)

            new_obs = Observation(
                board_self=new_board,
                board_opponent=obs.board_opponent,
                dealt_cards=[],  # Cards already placed
                known_discards_self=new_discards,
                turn=obs.turn,
                is_btn=obs.is_btn,
            )
            state = encode_state(new_obs)
            states.append(state)
            turns.append(obs.turn)

        # Batch inference
        state_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        turn_tensor = torch.tensor(turns, dtype=torch.long).to(self.device)

        output = self.model(state_tensor, turn_tensor)
        values = output["value"].squeeze(-1).cpu().numpy()

        # Denormalize values
        scores = values * self.score_std + self.score_mean

        # Incorporate bust/FL predictions
        bust_probs = output["bust_prob"].squeeze(-1).cpu().numpy()
        fl_probs = output["fl_prob"].squeeze(-1).cpu().numpy()

        # Adjusted score: penalize bust, reward FL
        # bust → expected score is ~-8 (typical bust penalty)
        # FL → expected bonus (FL entry typically adds 10-20 points)
        adjusted = scores * (1 - bust_probs) + (-8.0) * bust_probs + 10.0 * fl_probs

        if self.temperature <= 0:
            # Greedy
            best_idx = int(np.argmax(adjusted))
        else:
            # Softmax sampling
            logits = adjusted / self.temperature
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            best_idx = int(np.random.choice(len(actions), p=probs))

        return best_idx, actions[best_idx], float(adjusted[best_idx])


def play_one_hand(player: VNGreedyPlayer, seed: int = None) -> dict:
    """Play one full hand using VN-greedy and return trajectory data."""
    deck = list(ALL_CARDS)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(deck)
    else:
        random.shuffle(deck)

    hand = Hand(deck=deck, btn=random.randint(0, 1))
    records = []

    # Play through all turns for seat 0 (hero)
    seat = 0

    # Turn 0: initial 5-card placement
    for s in action_order(hand.btn):
        obs = hand.get_observation(s)
        if s == seat:
            idx, action, vn_score = player.select_action(obs)
            records.append({
                "turn": 0,
                "board": _board_to_dict(obs.board_self),
                "dealt": list(obs.dealt_cards),
                "exclude": list(obs.known_discards_self),
                "vn_score": vn_score,
                "board_after": _board_to_dict(apply_action_to_board(obs.board_self, action)),
            })
            hand.apply_action(s, action)
        else:
            # Opponent plays randomly (greedy from a random perspective)
            actions = get_initial_actions(obs.dealt_cards, obs.board_self)
            action = random.choice(actions) if actions else None
            if action:
                hand.apply_action(s, action)

    # Turns 1-8
    for turn_num in range(1, 9):
        if hand.is_hand_complete():
            break
        hand.deal_next_turn()

        for s in action_order(hand.btn):
            cards = hand.dealt_cards[s]
            if not cards:
                continue
            if hand.boards[s].is_complete():
                continue

            obs = hand.get_observation(s)
            if s == seat:
                idx, action, vn_score = player.select_action(obs)
                records.append({
                    "turn": turn_num,
                    "board": _board_to_dict(obs.board_self),
                    "dealt": list(obs.dealt_cards),
                    "exclude": list(obs.known_discards_self),
                    "vn_score": vn_score,
                    "board_after": _board_to_dict(apply_action_to_board(obs.board_self, action)),
                })
                hand.apply_action(s, action)
            else:
                actions = get_turn_actions(obs.dealt_cards, obs.board_self)
                action = random.choice(actions) if actions else None
                if action:
                    hand.apply_action(s, action)

    # Score the hand
    result = GameEngine.compute_result(hand)

    # Final record (Turn -1)
    final_board = hand.boards[seat]
    final_record = {
        "turn": -1,
        "score": float(result.raw_score[seat]),
        "busted": bool(result.busted[seat]),
        "fl_entry": bool(result.fl_entry[seat]),
        "final_board": _board_to_dict(final_board),
    }

    return {
        "records": records,
        "final": final_record,
        "score": float(result.raw_score[seat]),
        "busted": bool(result.busted[seat]),
        "fl_entry": bool(result.fl_entry[seat]),
    }


def _board_to_dict(board: Board) -> dict:
    return {
        "top": list(board.top),
        "mid": list(board.middle),
        "bot": list(board.bottom),
    }


def generate_selfplay_data(
    model_path: str,
    norm_path: str,
    n_hands: int = 1000,
    output_path: str = "data/selfplay_vn.jsonl",
    device: str = "auto",
    temperature: float = 0.1,
    input_dim: int = STATE_DIM,
):
    """Generate self-play data using VN-greedy evaluation."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading model from {model_path}...")
    model = ValueNetworkV3(input_dim=input_dim).to(device)
    ck = torch.load(model_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck)
    model.load_state_dict(sd)
    model.eval()

    # Load norm stats
    norm_stats = None
    if norm_path and Path(norm_path).exists():
        with open(norm_path) as f:
            norm_stats = json.load(f)
        print(f"  Norm stats: mean={norm_stats.get('mean', norm_stats.get('score_mean', 0)):.2f}, "
              f"std={norm_stats.get('std', norm_stats.get('score_std', 1)):.2f}")

    player = VNGreedyPlayer(model, device=device, norm_stats=norm_stats,
                            temperature=temperature)

    # Generate hands
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  VN-Greedy Self-Play")
    print(f"{'='*60}")
    print(f"  Hands:       {n_hands:,}")
    print(f"  Temperature: {temperature}")
    print(f"  Device:      {device}")
    print(f"  Output:      {output_path}")
    print()

    total_score = 0.0
    n_bust = 0
    n_fl = 0
    n_records = 0
    start_time = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        for hand_idx in range(n_hands):
            try:
                result = play_one_hand(player, seed=hand_idx)

                # Write in MC teacher format (compatible with preprocess_mc_teacher.py)
                hand_id = hand_idx
                for rec in result["records"]:
                    line = {
                        "hand_id": hand_id,
                        "turn": rec["turn"],
                        "board": rec["board"],
                        "dealt": rec["dealt"],
                        "exclude": rec["exclude"],
                        "n_candidates": 1,
                        "best_idx": 0,
                        "eval_mode": "vn_greedy",
                        "board_after": rec["board_after"],
                        "candidates": [{
                            "mc": {
                                "avg_score": rec["vn_score"],
                                "bust_rate": 0.0,
                                "fl_rate": 0.0,
                            }
                        }],
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    n_records += 1

                # Final record (Turn -1)
                final_line = {
                    "hand_id": hand_id,
                    "turn": -1,
                    "score": result["score"],
                    "busted": result["busted"],
                    "fl_entry": result["fl_entry"],
                    "final_board": result["final"]["final_board"],
                }
                f.write(json.dumps(final_line, ensure_ascii=False) + "\n")
                n_records += 1

                total_score += result["score"]
                if result["busted"]:
                    n_bust += 1
                if result["fl_entry"]:
                    n_fl += 1

            except Exception as e:
                print(f"  [WARN] Hand {hand_idx} failed: {e}")
                import traceback
                traceback.print_exc()

            if (hand_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (hand_idx + 1) / elapsed
                avg = total_score / (hand_idx + 1)
                bust_r = n_bust / (hand_idx + 1) * 100
                fl_r = n_fl / (hand_idx + 1) * 100
                print(f"  {hand_idx+1:,}/{n_hands:,} hands  "
                      f"({rate:.1f}/s)  avg={avg:.1f}  "
                      f"bust={bust_r:.1f}%  fl={fl_r:.1f}%")

    elapsed = time.time() - start_time
    avg_score = total_score / max(n_hands, 1)
    bust_rate = n_bust / max(n_hands, 1)
    fl_rate = n_fl / max(n_hands, 1)

    print(f"\n{'='*60}")
    print(f"  Self-Play Complete")
    print(f"{'='*60}")
    print(f"  Hands:      {n_hands:,}")
    print(f"  Records:    {n_records:,}")
    print(f"  Time:       {elapsed:.1f}s ({n_hands/elapsed:.1f} hands/s)")
    print(f"  Avg score:  {avg_score:.2f}")
    print(f"  Bust rate:  {bust_rate*100:.1f}%")
    print(f"  FL rate:    {fl_rate*100:.1f}%")
    print(f"  Output:     {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VN-Greedy Self-Play Data Generation")
    parser.add_argument("--model", required=True,
                        help="Path to VN model checkpoint (value_best.pt)")
    parser.add_argument("--norm", default=None,
                        help="Path to norm_stats.json")
    parser.add_argument("--hands", type=int, default=1000,
                        help="Number of hands to play")
    parser.add_argument("--output", default="data/selfplay_vn.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Action selection temperature (0=greedy, >0=exploration)")
    parser.add_argument("--input-dim", type=int, default=STATE_DIM,
                        help="VN input dimension (522 or 822)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    generate_selfplay_data(
        model_path=args.model,
        norm_path=args.norm,
        n_hands=args.hands,
        output_path=args.output,
        device=args.device,
        temperature=args.temperature,
        input_dim=args.input_dim,
    )
