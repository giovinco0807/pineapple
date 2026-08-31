import json

def format_action(idx):
    # Action index is pos0 * 3 + pos1 (0-8)
    # The discarded card is implied.
    pos_map = {0: "Top", 1: "Middle", 2: "Bottom"}
    pos0 = pos_map[idx // 3]
    pos1 = pos_map[idx % 3]
    return f"[{pos0}, {pos1}]"

with open('ai/data/t3_oracle/t3_p99_samples.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5: # show top 5
            break
        data = json.loads(line)
        print(f"=== Rank {data['tail_rank']} (Regret: {data['regret']:.2f}) ===")
        print(f"Best EV: {data['best_ev']:.2f} | Pred EV: {data['pred_ev']:.2f} | Model Value Output: {data['value_pred']:.2f}")
        
        board = data['state']['board']
        dealt = data['state']['dealt']
        print(f"Board:  Top {board['top']} | Mid {board['middle']} | Bot {board['bottom']}")
        print(f"Dealt:  {dealt}")
        
        print("True Top 3 Actions:")
        for a in data['true_top3_by_ev']:
            print(f"  Action {a['action_idx']} {format_action(a['action_idx'])}: EV {a['ev']:.2f}")
            
        print("Predicted Top 3 Actions:")
        for a in data['pred_top3_by_logit']:
            print(f"  Action {a['action_idx']} {format_action(a['action_idx'])}: EV {a['ev']:.2f} (Logit: {a['logit']:.2f})")
        print()
