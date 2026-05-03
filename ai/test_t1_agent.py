import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.game_state import GameState, GamePhase
from game.board import Row
from game.card import Card
from ai.t1_agent import T1Agent
from ai.random_ai import RandomAI

def card_to_str(c: Card) -> str:
    if c is None: return ""
    return str(c.rank) + c.suit

def get_row_name(r: Row) -> str:
    if r == Row.TOP: return "Top"
    if r == Row.MIDDLE: return "Mid"
    return "Bot"

def get_row_enum(r_str: str) -> Row:
    if r_str == "Top": return Row.TOP
    if r_str == "Mid": return Row.MIDDLE
    return Row.BOTTOM

def main():
    print("Loading T1 Agent...")
    t1_agent = T1Agent("ai/models/t1_placement_net_v1.pt")
    random_ai = RandomAI()

    games_to_play = 5
    for i in range(games_to_play):
        print(f"\n--- Game {i+1} ---")
        game = GameState()
        game.start_new_round()
        
        while game.phase != GamePhase.ROUND_END and game.phase != GamePhase.SCORING:
            player = game.players["player"]
            
            if game.phase == GamePhase.INITIAL_DEAL:
                # Use Random AI for T0
                placements = random_ai.place_initial_cards(player.hand, player.board)
                game.place_cards("player", placements)
                game.check_phase_complete()
                
            elif game.phase == GamePhase.PINEAPPLE:
                # Is it T1? T1 means the board has 5 cards.
                board_count = len(player.board.top) + len(player.board.middle) + len(player.board.bottom)
                if board_count == 5:
                    # It's T1! Use our T1Agent
                    top_strs = [card_to_str(c) for c in player.board.top]
                    mid_strs = [card_to_str(c) for c in player.board.middle]
                    bot_strs = [card_to_str(c) for c in player.board.bottom]
                    hand_strs = [card_to_str(c) for c in player.hand]
                    
                    print(f"T1 Situation for player:")
                    print(f"  Top: {top_strs}")
                    print(f"  Mid: {mid_strs}")
                    print(f"  Bot: {bot_strs}")
                    print(f"  Hand: {hand_strs}")
                    
                    placements_str, ev = t1_agent.get_action(top_strs, mid_strs, bot_strs, hand_strs)
                    print(f"  T1Agent EV: {ev:.3f}")
                    
                    # Convert to actual cards and row enums
                    placements = []
                    discard = None
                    for c_str, r_str in placements_str:
                        # find card in hand
                        matched_card = next((c for c in player.hand if card_to_str(c) == c_str), None)
                        if r_str == "Discard":
                            discard = matched_card
                            print(f"  Discarding: {c_str}")
                        else:
                            placements.append((matched_card, get_row_enum(r_str)))
                            print(f"  Placing: {c_str} -> {r_str}")
                    
                    game.place_cards("player", placements)
                    if discard:
                        game.discard_card("player", discard)
                    game.check_phase_complete()
                    
                else:
                    # T2, T3, T4 use Random AI
                    placements, discard = random_ai.place_pineapple_cards(player.hand, player.board)
                    game.place_cards("player", placements)
                    game.discard_card("player", discard)
                    game.check_phase_complete()
            
            # Dummy process AI
            ai_player = game.players["ai"]
            if ai_player.hand:
                if game.phase == GamePhase.INITIAL_DEAL:
                    placements = random_ai.place_initial_cards(ai_player.hand, ai_player.board)
                    game.place_cards("ai", placements)
                elif game.phase == GamePhase.PINEAPPLE:
                    placements, discard = random_ai.place_pineapple_cards(ai_player.hand, ai_player.board)
                    game.place_cards("ai", placements)
                    game.discard_card("ai", discard)
            
            game.check_phase_complete()
            
            if game.phase == GamePhase.SCORING:
                game.calculate_scores()
                game.finalize_round()

if __name__ == "__main__":
    main()
