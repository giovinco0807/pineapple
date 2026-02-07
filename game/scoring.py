"""Scoring logic for OFC Pineapple."""
from typing import Dict
from .board import Board, Row
from .hand_evaluator import compare_hands_3, compare_hands_5


def calculate_round_score(
    player_board: Board,
    ai_board: Board,
    player_stack: int,
    ai_stack: int
) -> Dict:
    """
    Calculate the score for a round.
    
    Scoring rules:
    - Each row: winner gets +1 point
    - Scoop (win all 3 rows): +3 bonus points
    - Royalties: added to winner's score
    - Bust: player loses all rows, no royalties
    - Both bust: draw (0 points)
    - Points capped at min(player_stack, opponent_stack)
    
    Returns:
        Dict with detailed scoring breakdown
    """
    result = {
        'player_bust': player_board.is_bust() if player_board.is_complete() else False,
        'ai_bust': ai_board.is_bust() if ai_board.is_complete() else False,
        'rows': {'top': 0, 'middle': 0, 'bottom': 0},
        'player_royalties': {'top': 0, 'middle': 0, 'bottom': 0, 'total': 0},
        'ai_royalties': {'top': 0, 'middle': 0, 'bottom': 0, 'total': 0},
        'scoop_bonus': 0,
        'raw_score': 0,  # Player's raw score (before cap)
        'stack_cap': min(player_stack, ai_stack),
        'player_net': 0,  # Final points for player
        'ai_net': 0       # Final points for AI
    }
    
    player_bust = result['player_bust']
    ai_bust = result['ai_bust']
    
    # Both bust = draw
    if player_bust and ai_bust:
        return result
    
    # One player bust = other wins all
    if player_bust:
        # AI wins all rows
        result['rows'] = {'top': -1, 'middle': -1, 'bottom': -1}
        if ai_board.is_complete():
            result['ai_royalties'] = ai_board.get_royalties()
        result['scoop_bonus'] = -3
        raw_score = -3 - 3 - result['ai_royalties']['total']  # -3 rows, -3 scoop
        result['raw_score'] = raw_score
        
    elif ai_bust:
        # Player wins all rows
        result['rows'] = {'top': 1, 'middle': 1, 'bottom': 1}
        if player_board.is_complete():
            result['player_royalties'] = player_board.get_royalties()
        result['scoop_bonus'] = 3
        raw_score = 3 + 3 + result['player_royalties']['total']  # +3 rows, +3 scoop
        result['raw_score'] = raw_score
        
    else:
        # Normal comparison
        if player_board.is_complete() and ai_board.is_complete():
            # Compare each row
            top_result = compare_hands_3(player_board.top, ai_board.top)
            mid_result = compare_hands_5(player_board.middle, ai_board.middle)
            bot_result = compare_hands_5(player_board.bottom, ai_board.bottom)
            
            result['rows'] = {
                'top': top_result,
                'middle': mid_result,
                'bottom': bot_result
            }
            
            # Calculate row points
            row_points = sum(result['rows'].values())
            
            # Check for scoop
            player_wins = sum(1 for v in result['rows'].values() if v > 0)
            ai_wins = sum(1 for v in result['rows'].values() if v < 0)
            
            if player_wins == 3:
                result['scoop_bonus'] = 3
            elif ai_wins == 3:
                result['scoop_bonus'] = -3
            
            # Get royalties
            result['player_royalties'] = player_board.get_royalties()
            result['ai_royalties'] = ai_board.get_royalties()
            
            # Calculate raw score
            raw_score = (
                row_points +
                result['scoop_bonus'] +
                result['player_royalties']['total'] -
                result['ai_royalties']['total']
            )
            result['raw_score'] = raw_score
    
    # Apply stack cap
    stack_cap = result['stack_cap']
    capped_score = max(-stack_cap, min(stack_cap, result['raw_score']))
    result['player_net'] = capped_score
    result['ai_net'] = -capped_score
    
    return result


def format_score_summary(result: Dict) -> str:
    """Format scoring result for display."""
    lines = []
    
    if result['player_bust'] and result['ai_bust']:
        lines.append("Both players bust - Draw!")
        return '\n'.join(lines)
    
    if result['player_bust']:
        lines.append("Player bust! AI wins all rows.")
    elif result['ai_bust']:
        lines.append("AI bust! Player wins all rows.")
    else:
        # Row results
        row_names = {'top': 'Top', 'middle': 'Middle', 'bottom': 'Bottom'}
        for row, val in result['rows'].items():
            if val > 0:
                lines.append(f"{row_names[row]}: Player wins")
            elif val < 0:
                lines.append(f"{row_names[row]}: AI wins")
            else:
                lines.append(f"{row_names[row]}: Tie")
    
    # Scoop
    if result['scoop_bonus'] > 0:
        lines.append(f"Player scoops! (+{result['scoop_bonus']})")
    elif result['scoop_bonus'] < 0:
        lines.append(f"AI scoops! ({result['scoop_bonus']})")
    
    # Royalties
    if result['player_royalties']['total'] > 0:
        lines.append(f"Player royalties: +{result['player_royalties']['total']}")
    if result['ai_royalties']['total'] > 0:
        lines.append(f"AI royalties: +{result['ai_royalties']['total']}")
    
    # Net result
    net = result['player_net']
    if net > 0:
        lines.append(f"\nPlayer wins {net} points!")
    elif net < 0:
        lines.append(f"\nAI wins {-net} points!")
    else:
        lines.append("\nRound is a draw!")
    
    return '\n'.join(lines)
