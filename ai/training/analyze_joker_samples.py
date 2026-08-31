import json
import sys
sys.path.append('.')
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board

def is_joker(state):
    drawn = state.get('dealt', [])
    board = state.get('board', {})
    
    for c in drawn:
        if c in ['X1', 'X2', 'X']:
            return True
    
    for row_name, row in board.items():
        for c in row:
            if c in ['X1', 'X2', 'X']:
                return True
    return False

def action_to_str(action):
    placements = []
    for card, pos in action.placements:
        placements.append(f"{card}->{pos}")
    return f"Place: {', '.join(placements)}, Discard: {action.discard}"

if __name__ == '__main__':
    with open('ai/data/t3_oracle/t3_p99_samples.jsonl', 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    
    joker_samples = [s for s in samples if is_joker(s['state'])]
    print(f"Joker samples count: {len(joker_samples)} out of {len(samples)}")
        
    print("\n--- Top 5 Joker Samples Detailed Actions ---")
    for i, s in enumerate(joker_samples[:5]):
        print(f"\nSample {i+1}")
        print(f"Regret: {s['regret']}")
        print(f"Best EV: {s['best_ev']}, Pred EV: {s['pred_ev']}, Value Pred: {s['value_pred']}")
        
        state = s['state']
        dealt = state['dealt']
        board_dict = state['board']
        print(f"Dealt: {dealt}")
        print(f"Board: {board_dict}")
        
        board = Board()
        board.top = board_dict.get('top', [])
        board.middle = board_dict.get('middle', [])
        board.bottom = board_dict.get('bottom', [])
        
        valid_actions = get_turn_actions(dealt, board)
        
        print("\nTrue Top 3 Actions by EV:")
        for ta in s['true_top3_by_ev']:
            idx = ta['action_idx']
            ev = ta['ev']
            if idx < len(valid_actions):
                print(f"  [{idx}] EV: {ev:8.2f}  |  {action_to_str(valid_actions[idx])}")
            else:
                print(f"  [{idx}] EV: {ev:8.2f}  |  INVALID INDEX")
                
        print("\nPred Top 3 Actions by Logit:")
        for pa in s['pred_top3_by_logit']:
            idx = pa['action_idx']
            logit = pa['logit']
            ev = pa['ev']
            if idx < len(valid_actions):
                print(f"  [{idx}] EV: {ev:8.2f}, Logit: {logit:5.2f}  |  {action_to_str(valid_actions[idx])}")
            else:
                print(f"  [{idx}] EV: {ev:8.2f}, Logit: {logit:5.2f}  |  INVALID INDEX")
