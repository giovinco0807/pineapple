import json
import random

filepath = r'D:\ofc_data\mc_teacher\mc_s50_merged.jsonl'
hands = []

with open(filepath, 'r') as f:
    for line in f:
        hands.append(json.loads(line))

# Group hands by hand_id
hands_by_id = {}
for h in hands:
    hid = h.get('hand_id')
    if hid not in hands_by_id:
        hands_by_id[hid] = []
    hands_by_id[hid].append(h)

# Find a hand that has all turns 0, 1, 2, 3, 4
complete_hands = [hid for hid, hlist in hands_by_id.items() if len(hlist) == 5]

if not complete_hands:
    print("No complete 5-turn hands found in the early shard.")
else:
    # Pick a random complete hand
    sampled_id = random.choice(complete_hands)
    hand_seq = sorted(hands_by_id[sampled_id], key=lambda x: x.get('turn', 0))
    
    print(f"\n================ FULL HAND PROGRESSION (Hand ID: {sampled_id}) ================\n")
    for hand in hand_seq:
        turn = hand.get('turn')
        print(f"--- Turn {turn} ---")
        
        board = hand.get('board', {}) if turn < 4 else hand.get('final_board', [[], [], []])
        dealt = hand.get('dealt', [])
        
        if turn < 4:
            print(f"Board Before: Top {board.get('top', [])} | Mid {board.get('mid', [])} | Bot {board.get('bot', [])}")
            print(f"Dealt Cards: {dealt}")
            
            best_idx = hand.get('best_idx', 0)
            candidates = hand.get('candidates', [])
            if candidates and len(candidates) > best_idx:
                best_cand = candidates[best_idx]
                placements = best_cand.get('placements', [])
                print(f"Action taken: {placements}")
                
                stats = best_cand.get('stats', best_cand.get('mc', {}))
                ev = stats.get('ev', stats.get('mean', 0.0))
                bust = stats.get('bust_prob', stats.get('bust_rate', 0.0))
                fl = stats.get('fl_rate', 0.0)
                print(f"EV: {ev:.3f} | Bust: {bust*100:.1f}% | FL: {fl*100:.1f}%\n")
        else:
            print(f"Final Board Reached!")
            top = board[0]
            mid = board[1]
            bot = board[2]
            print(f"Top Array: {top}")
            print(f"Mid Array: {mid}")
            print(f"Bot Array: {bot}")
            
            score = hand.get("score", "N/A")
            busted = hand.get("busted", "N/A")
            fl_entry = hand.get("fl_entry", "N/A")
            royalty = hand.get("eval_result", {}).get("royalty", "N/A")
            print(f"Final Outcome -> Score: {score} | Busted? {busted} | FL Entry? {fl_entry} | Total Royalty: {royalty}\n")
    print("================================================================================")
