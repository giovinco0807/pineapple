"""
OFC Pineapple - Self-Play Data Generator

Generates training data by running random/heuristic AI self-play
and writing the results directly as JSONL for the training pipeline.

Usage:
    python ai/training/generate_data.py --games 1000 --output data/train.jsonl
    python -m ai.training.preprocess data/train.jsonl --output data/processed
    python -m ai.training.behavior_cloning --data data/processed --epochs 100
"""
import sys
import json
import random
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state
from ai.engine.action_space import get_initial_actions, get_turn_actions, Action
from ai.engine.game_engine import GameEngine, Hand, Session, HandResult, ALL_CARDS


def heuristic_score(action: Action, board: Board) -> float:
    """
    Simple heuristic for action ranking (better than random).
    Prefers: pairs together, high cards bottom, low cards top.
    """
    from ai.engine.game_engine import RANK_VALUES
    score = 0.0

    for card, pos in action.placements:
        if card in ("X1", "X2"):
            rank = 14  # Joker as high value
        else:
            rank = RANK_VALUES.get(card[0], 7)

        if pos == "bottom":
            score += rank * 0.5   # Prefer high cards bottom
        elif pos == "top":
            score -= rank * 0.3   # Penalize high cards top (save for pairs)
        else:
            score += rank * 0.1   # Small bonus for middle

    # Bonus for placing same-rank cards together
    placed_by_pos = {}
    for card, pos in action.placements:
        placed_by_pos.setdefault(pos, []).append(card)

    for pos, cards in placed_by_pos.items():
        existing = getattr(board, pos, [])
        all_cards = list(existing) + cards
        ranks = []
        for c in all_cards:
            if c not in ("X1", "X2"):
                ranks.append(c[0])
        from collections import Counter
        rc = Counter(ranks)
        for _, count in rc.items():
            if count >= 2:
                score += 5.0 * count  # Reward pairs/trips in same row

    return score


def select_action(dealt_cards: list, board: Board, turn: int,
                  strategy: str = "heuristic") -> Action:
    """Select an action using the given strategy."""
    if turn == 0:
        actions = get_initial_actions(dealt_cards, board)
    else:
        actions = get_turn_actions(dealt_cards, board)

    if not actions:
        raise ValueError("No valid actions!")

    if strategy == "random":
        return random.choice(actions)
    elif strategy == "heuristic":
        scored = [(heuristic_score(a, board), random.random(), a) for a in actions]
        scored.sort(reverse=True)
        # Softmax-like: pick from top-3 with temperature
        top = scored[:min(3, len(scored))]
        return random.choice(top)[2]
    else:
        return random.choice(actions)


def generate_game(strategy: str = "heuristic") -> list:
    """Play one full hand and collect training samples."""
    deck = list(ALL_CARDS)
    random.shuffle(deck)
    hand = Hand(deck=deck, btn=random.randint(0, 1))

    turn_logs = []

    # Turn 0
    for seat in [hand.btn, 1 - hand.btn]:
        cards = hand.dealt_cards[seat]
        board = Board(
            top=list(hand.boards[seat].top),
            middle=list(hand.boards[seat].middle),
            bottom=list(hand.boards[seat].bottom),
        )
        action = select_action(cards, board, 0, strategy)

        turn_logs.append({
            "turn": 0,
            "player": seat,
            "is_btn": seat == hand.btn,
            "board_self": board.to_dict(),
            "board_opponent": hand.boards[1 - seat].to_dict() if hasattr(hand.boards[1 - seat], 'to_dict') else {"top": [], "middle": [], "bottom": []},
            "dealt_cards": list(cards),
            "discards_self": [],
            "action": {
                "placements": [(c, p) for c, p in action.placements],
                "discard": action.discard,
            },
        })
        hand.apply_action(seat, action)

    # Turns 1-8
    for turn_num in range(1, 9):
        if hand.is_hand_complete():
            break
        hand.deal_next_turn()

        for seat in [hand.btn, 1 - hand.btn]:
            cards = hand.dealt_cards[seat]
            if not cards:
                continue
            board = hand.boards[seat].copy()
            actions = get_turn_actions(cards, board)
            if not actions:
                continue
            action = select_action(cards, board, turn_num, strategy)

            turn_logs.append({
                "turn": turn_num,
                "player": seat,
                "is_btn": seat == hand.btn,
                "board_self": board.to_dict(),
                "board_opponent": hand.boards[1 - seat].to_dict(),
                "dealt_cards": list(cards),
                "discards_self": list(hand.discards[seat]),
                "action": {
                    "placements": [(c, p) for c, p in action.placements],
                    "discard": action.discard,
                },
            })
            hand.apply_action(seat, action)

    # Score hand
    result = GameEngine.compute_result(hand)
    hand_result = {
        "busted": result.busted,
        "royalties": result.royalties,
        "fl_entry": result.fl_entry,
        "raw_score": result.raw_score,
    }

    return [{"turn_log": tl, "hand_result": hand_result} for tl in turn_logs]


def generate_dataset(num_games: int, output_path: str,
                     strategy: str = "heuristic"):
    """Generate training data from self-play."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    busts = 0
    fl_entries = 0
    total_royalty = 0.0
    games_done = 0

    with open(output, "w", encoding="utf-8") as f:
        for i in range(num_games):
            try:
                samples = generate_game(strategy)
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    total_samples += 1

                    hr = sample["hand_result"]
                    player = sample["turn_log"]["player"]
                    if hr["busted"][player]:
                        busts += 1
                    if hr["fl_entry"][player]:
                        fl_entries += 1
                    total_royalty += hr["royalties"][player]["total"]
                games_done += 1
            except Exception as e:
                print(f"  [WARN] Game {i} failed: {e}")

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{num_games} games done ({total_samples} samples)")

    bust_rate = busts / max(total_samples, 1)
    fl_rate = fl_entries / max(total_samples, 1)
    avg_royalty = total_royalty / max(total_samples, 1)

    print(f"\n--- Data Generation Complete ---")
    print(f"  Games: {games_done}")
    print(f"  Samples: {total_samples}")
    print(f"  Bust rate: {bust_rate:.1%}")
    print(f"  FL rate: {fl_rate:.1%}")
    print(f"  Avg royalty: {avg_royalty:.1f}")
    print(f"  Output: {output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate self-play training data")
    parser.add_argument("--games", type=int, default=1000, help="Number of games")
    parser.add_argument("--output", default="data/train.jsonl", help="Output JSONL")
    parser.add_argument("--strategy", choices=["random", "heuristic"], default="heuristic")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Generating {args.games} games with '{args.strategy}' strategy...")
    generate_dataset(args.games, args.output, args.strategy)
