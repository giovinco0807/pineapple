import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from engine.encoding import Board
from engine.action_space import get_turn_actions, encode_action, Action

line = '{"sample_id":1002876,"bb_top":["Th","Ks","7h"],"bb_middle":["2s","3d","Jc","5s"],"bb_bottom":["6d","Tc","9s","2c"],"bb_drawn":["9h","Kh","6s"],"placements":[{"cards_placed":["9h","Kh"],"slots":["mid","bot"],"ev":-5.269172932330827},{"cards_placed":["9h","Kh"],"slots":["bot","mid"],"ev":-0.5736842105263158},{"cards_placed":["9h","6s"],"slots":["mid","bot"],"ev":-0.5736842105263158},{"cards_placed":["9h","6s"],"slots":["bot","mid"],"ev":-0.5736842105263158},{"cards_placed":["Kh","6s"],"slots":["mid","bot"],"ev":-0.5736842105263158},{"cards_placed":["Kh","6s"],"slots":["bot","mid"],"ev":-5.269172932330827}],"best_ev":-0.5736842105263158}'

sol = json.loads(line)
top = sol["bb_top"]
mid = sol["bb_middle"]
bot = sol["bb_bottom"]
board_obj = Board(top=top, middle=mid, bottom=bot)
bb_drawn = sol["bb_drawn"]

valid_actions = get_turn_actions(bb_drawn, board_obj)
print(f"Valid actions count: {len(valid_actions)}")
for a in valid_actions[:3]:
    print(a)

row_map = {"top": "top", "mid": "middle", "bot": "bottom"}
for p in sol.get("placements", []):
    cards_placed = p["cards_placed"]
    slots = p["slots"]
    discard = next(c for c in bb_drawn if c not in cards_placed)
    placements = [(cards_placed[0], row_map[slots[0]]), (cards_placed[1], row_map[slots[1]])]
    py_action = Action(placements=placements, discard=discard)
    
    try:
        idx = encode_action(py_action, valid_actions, turn=4, dealt_cards=bb_drawn)
        print(f"Success: {py_action} -> {idx}")
    except Exception as e:
        print(f"Failed to encode: {py_action} - {e}")
