import sys
from pathlib import Path
AI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_DIR))
sys.path.insert(0, str(AI_DIR.parent))

from engine.encoding import Board
from engine.action_space import get_turn_actions, encode_action, Action

board_obj = Board(top=["Qh","3c","Qd"], middle=["7h","Jh","Tc"], bottom=["4s","4h","4d","2d","2c"])
bb_drawn = ["7d","Ad","6s"]
valid_actions = get_turn_actions(bb_drawn, board_obj)
print("Valid actions:", len(valid_actions))
print("Valid actions list:")
for a in valid_actions:
    print(a.placements, a.discard)

action = Action(placements=[("7d", "middle"), ("Ad", "middle")], discard="6s")
print("Target:", action.placements, action.discard)
try:
    idx = encode_action(action, valid_actions, turn=4, dealt_cards=bb_drawn)
    print("Idx:", idx)
except Exception as e:
    print("Error:", e)
