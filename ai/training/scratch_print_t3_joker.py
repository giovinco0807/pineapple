import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.engine.action_space import get_action_from_semantic_index

def parse_samples():
    with open('ai/data/t3_oracle/t3_p99_samples.jsonl') as f:
        found = 0
        for i, line in enumerate(f):
            d = json.loads(line)
            state = d['state']
            dealt = state.get('dealt', [])
            
            if 'X2' in dealt:
                true_action = get_action_from_semantic_index(d['true_action_idx'], dealt)
                
                # Find best action where X2 is placed on top
                x2_top_actions = []
                for act_info in d['true_top3_by_ev']:
                    act = get_action_from_semantic_index(act_info['action_idx'], dealt)
                    for card, pos in act.placements:
                        if card == 'X2' and pos == 'top':
                            x2_top_actions.append((act, act_info['ev']))
                
                x2_pos = None
                for card, pos in true_action.placements:
                    if card == 'X2':
                        x2_pos = pos
                
                if x2_pos != 'top':
                    print(f"--- Sample {i} ---")
                    print(f"Board: {state.get('board')}")
                    print(f"Dealt: {dealt}")
                    print(f"True Action (solver): {true_action} (EV: {d['best_ev']:.3f})")
                    if x2_top_actions:
                        print(f"Best X2-on-top in top3: {x2_top_actions[0][0]} (EV: {x2_top_actions[0][1]:.3f})")
                    else:
                        print("X2-on-top action not in top 3.")
                        
                    # Also print the full top 3
                    print("Top 3 actions by EV:")
                    for rank, act_info in enumerate(d['true_top3_by_ev']):
                        act = get_action_from_semantic_index(act_info['action_idx'], dealt)
                        print(f"  {rank+1}: {act} (EV: {act_info['ev']:.3f})")
                        
                    found += 1
                    if found >= 5:
                        break

if __name__ == '__main__':
    parse_samples()
