"""Detailed test script for trained model."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.rl_env import OFCPineappleEnv
from stable_baselines3 import PPO
from game.hand_evaluator import evaluate_3_card_hand, evaluate_5_card_hand, hand_rank_name, hand_rank_3_name
import numpy as np

def cards_str(cards):
    result = []
    for c in cards:
        if c.is_joker:
            result.append("🃏")
        else:
            suits = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
            result.append(f"{c.rank}{suits.get(c.suit, c.suit)}")
    return " ".join(result)

def hand_str(cards, row_type):
    if not cards:
        return "Empty"
    if row_type == "top" and len(cards) == 3:
        rank, _ = evaluate_3_card_hand(cards)
        return f"{cards_str(cards)} [{hand_rank_3_name(rank)}]"
    elif row_type in ["middle", "bottom"] and len(cards) == 5:
        rank, _ = evaluate_5_card_hand(cards)
        return f"{cards_str(cards)} [{hand_rank_name(rank)}]"
    return f"{cards_str(cards)} [Incomplete]"

model = PPO.load('ai/models/ppo_ofc_20260125_023738_final')
env = OFCPineappleEnv()

print('='*60)
print('Playing 10 games with trained model')
print('='*60)

total_wins = 0
total_losses = 0
total_busts = 0

for game in range(10):
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 50:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
    
    player = env.game.players['player']
    
    print(f'\nGame {game+1}:')
    print(f'  Top:    {hand_str(player.board.top, "top")}')
    print(f'  Middle: {hand_str(player.board.middle, "middle")}')
    print(f'  Bottom: {hand_str(player.board.bottom, "bottom")}')
    
    if player.board.is_complete():
        royalties = player.board.get_royalties()
        bust = player.board.is_bust()
        print(f'  Royalties: {royalties["total"]} | Bust: {bust}')
        if bust:
            total_busts += 1
    
    if env.game.last_scores:
        s = env.game.last_scores
        net = s["player_net"]
        print(f'  Net Score: {net}')
        if net > 0:
            total_wins += 1
        elif net < 0:
            total_losses += 1

print('\n' + '='*60)
print(f'Summary: Wins={total_wins}, Losses={total_losses}, Busts={total_busts}')
print('='*60)
