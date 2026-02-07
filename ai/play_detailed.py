"""
Detailed game logger for OFC Pineapple.

Plays games and logs every decision for analysis.
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO

from ai.rl_env import OFCPineappleEnv
from game.card import Card
from game.board import Row
from game.hand_evaluator import (
    evaluate_3_card_hand, evaluate_5_card_hand,
    hand_rank_name, hand_rank_3_name
)


def card_to_str(card: Card) -> str:
    """Convert card to readable string."""
    if card.is_joker:
        return "🃏"
    suit_symbols = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    return f"{card.rank}{suit_symbols.get(card.suit, card.suit)}"


def cards_to_str(cards: List[Card]) -> str:
    """Convert list of cards to readable string."""
    return " ".join(card_to_str(c) for c in cards)


def evaluate_row(cards: List[Card], row_type: str) -> str:
    """Evaluate a row and return hand description."""
    if not cards:
        return "Empty"
    
    if row_type == "top" and len(cards) == 3:
        rank, kickers = evaluate_3_card_hand(cards)
        return hand_rank_3_name(rank)
    elif row_type in ["middle", "bottom"] and len(cards) == 5:
        rank, kickers = evaluate_5_card_hand(cards)
        return hand_rank_name(rank)
    else:
        return f"Incomplete ({len(cards)} cards)"


def play_detailed_game(model_path: str = None, log_file: str = None):
    """
    Play a game with detailed logging.
    
    Args:
        model_path: Path to trained model (None for random play)
        log_file: Path to save log (None for console only)
    """
    env = OFCPineappleEnv()
    
    if model_path:
        print(f"Loading model: {model_path}")
        model = PPO.load(model_path)
    else:
        model = None
        print("Playing with random actions")
    
    # Game log
    game_log = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "rounds": []
    }
    
    obs, info = env.reset()
    done = False
    step = 0
    
    round_log = {
        "round_number": env.game.round_number,
        "actions": []
    }
    
    print("\n" + "="*60)
    print("GAME START")
    print("="*60)
    
    while not done:
        player = env.game.players[env.current_player]
        phase = env.game.phase.value
        
        # Log current state
        action_log = {
            "step": step,
            "phase": phase,
            "hand": cards_to_str(player.hand),
            "board_before": {
                "top": cards_to_str(player.board.top),
                "middle": cards_to_str(player.board.middle),
                "bottom": cards_to_str(player.board.bottom),
            }
        }
        
        print(f"\n--- Step {step} ({phase}) ---")
        print(f"Hand: {cards_to_str(player.hand)}")
        print(f"Board:")
        print(f"  Top:    [{cards_to_str(player.board.top)}] {evaluate_row(player.board.top, 'top')}")
        print(f"  Middle: [{cards_to_str(player.board.middle)}] {evaluate_row(player.board.middle, 'middle')}")
        print(f"  Bottom: [{cards_to_str(player.board.bottom)}] {evaluate_row(player.board.bottom, 'bottom')}")
        
        # Get action
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Random valid action
            mask = env.action_masks()
            valid_actions = np.where(mask == 1)[0]
            if len(valid_actions) == 0:
                print("No valid actions!")
                break
            action = np.random.choice(valid_actions)
        
        action = int(action)
        card_idx = action // 3
        row_idx = action % 3
        row_name = ["top", "middle", "bottom"][row_idx]
        
        if card_idx < len(player.hand):
            card = player.hand[card_idx]
            print(f"Action: Place {card_to_str(card)} -> {row_name}")
            action_log["action"] = {
                "card": card_to_str(card),
                "row": row_name,
                "action_id": action
            }
        else:
            print(f"Action: Invalid card index {card_idx}")
            action_log["action"] = {"error": f"Invalid card index {card_idx}"}
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        action_log["reward"] = float(reward)
        action_log["done"] = done
        
        if reward != 0:
            print(f"Reward: {reward:.1f}")
        
        round_log["actions"].append(action_log)
        step += 1
        
        # If round ended, log final state
        if done:
            print("\n" + "="*60)
            print("ROUND END")
            print("="*60)
            
            # Final board state
            for pid in ["player", "ai"]:
                p = env.game.players[pid]
                print(f"\n{p.name}'s Final Board:")
                print(f"  Top:    [{cards_to_str(p.board.top)}] {evaluate_row(p.board.top, 'top')}")
                print(f"  Middle: [{cards_to_str(p.board.middle)}] {evaluate_row(p.board.middle, 'middle')}")
                print(f"  Bottom: [{cards_to_str(p.board.bottom)}] {evaluate_row(p.board.bottom, 'bottom')}")
                
                if p.board.is_complete():
                    royalties = p.board.get_royalties()
                    print(f"  Royalties: {royalties['total']} (T:{royalties['top']}, M:{royalties['middle']}, B:{royalties['bottom']})")
                    print(f"  Bust: {p.board.is_bust()}")
                    if p.in_fantasyland:
                        print(f"  Fantasyland: {p.fantasyland_cards} cards")
            
            # Scores
            if env.game.last_scores:
                scores = env.game.last_scores
                print(f"\nScores:")
                print(f"  Player net: {scores['player_net']}")
                print(f"  AI net: {scores['ai_net']}")
                if scores['scoop_bonus'] != 0:
                    print(f"  Scoop: {scores['scoop_bonus']}")
            
            round_log["final_state"] = {
                "player_board": {
                    "top": cards_to_str(env.game.players["player"].board.top),
                    "middle": cards_to_str(env.game.players["player"].board.middle),
                    "bottom": cards_to_str(env.game.players["player"].board.bottom),
                },
                "scores": env.game.last_scores
            }
    
    game_log["rounds"].append(round_log)
    game_log["total_reward"] = sum(a.get("reward", 0) for a in round_log["actions"])
    
    # Save log
    if log_file:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(game_log, f, indent=2, ensure_ascii=False)
        print(f"\nLog saved to: {log_file}")
    
    return game_log


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Play OFC Pineapple with detailed logging")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--log", type=str, default=None, help="Path to save JSON log")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    
    args = parser.parse_args()
    
    for i in range(args.games):
        print(f"\n{'#'*60}")
        print(f"# GAME {i+1}/{args.games}")
        print(f"{'#'*60}")
        
        log_file = None
        if args.log:
            base, ext = os.path.splitext(args.log)
            log_file = f"{base}_{i+1}{ext}" if args.games > 1 else args.log
        
        play_detailed_game(args.model, log_file)


if __name__ == "__main__":
    main()
