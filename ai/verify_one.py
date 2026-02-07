"""1ハンドだけ検証用"""
import sys
import argparse
sys.path.insert(0, '.')
from game.card import Deck, RANKS
from game.hand_evaluator import evaluate_3_card_hand, evaluate_5_card_hand, hand_rank_name, hand_rank_3_name
from game.royalty import get_top_royalty, get_middle_royalty, get_bottom_royalty
from ai.fl_solver_v2 import cards_to_str, solve_fantasyland

import random

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--hands', type=int, default=1, help='Number of hands')
parser.add_argument('--cards', type=int, default=14, help='Cards per hand')
parser.add_argument('--jokers', action='store_true', help='Include jokers')
args = parser.parse_args()

for hand_num in range(args.hands):
    seed = args.seed if args.seed else random.randint(0, 99999)
    if args.hands > 1:
        seed = seed + hand_num
    
    random.seed(seed)
    deck = Deck(include_jokers=args.jokers)
    deck.shuffle()
    hand = deck.deal(args.cards)

    print('='*70)
    print(f'【ハンド #{hand_num+1}】 seed={seed}, {args.cards}枚' + (' (ジョーカーあり)' if args.jokers else ''))
    print('='*70)
    print(cards_to_str(hand))
    print()

    # ランク順表示
    sorted_hand = sorted(hand, key=lambda c: (RANKS.index(c.rank) if not c.is_joker else 99, c.suit if not c.is_joker else 'z'), reverse=True)
    print('【ランク順】')
    print(cards_to_str(sorted_hand))
    print()

    # ソルバー実行
    solutions = solve_fantasyland(hand, max_solutions=3)

    print(f'【見つかった解: {len(solutions)}個】')

    for i, sol in enumerate(solutions):
        top_rank, _ = evaluate_3_card_hand(sol.opt_top if sol.opt_top else sol.top)
        mid_rank, _ = evaluate_5_card_hand(sol.opt_middle if sol.opt_middle else sol.middle)
        bot_rank, _ = evaluate_5_card_hand(sol.opt_bottom if sol.opt_bottom else sol.bottom)
        
        print()
        print(f'--- 解 #{i+1} (スコア: {sol.score:.0f}点) ---')
        print(f'  Top:    {cards_to_str(sol.top):30} [{hand_rank_3_name(top_rank):15}] Roy:{get_top_royalty(sol.opt_top if sol.opt_top else sol.top):2}')
        print(f'  Middle: {cards_to_str(sol.middle):30} [{hand_rank_name(mid_rank):15}] Roy:{get_middle_royalty(sol.opt_middle if sol.opt_middle else sol.middle):2}')
        print(f'  Bottom: {cards_to_str(sol.bottom):30} [{hand_rank_name(bot_rank):15}] Roy:{get_bottom_royalty(sol.opt_bottom if sol.opt_bottom else sol.bottom):2}')
        print(f'  Discard: {cards_to_str(sol.discards)}')
        print(f'  合計: {sol.royalties}pts | FL残留: {sol.can_stay} | Bust: {sol.is_bust}')
    
    print()

