import sys; sys.path.insert(0, '.')
from backend.main import evaluate_hand, hand_category, _evaluate_with_joker_constraint, _B

# Seat 0's exact board from debug
top = ['9c', 'Ks', '7d']
middle = ['4c', '4s', '5d', '5h', '4h']
bottom = ['Qc', 'Qh', 'Qs', 'Js', 'Jc']

bot_val = evaluate_hand(bottom, 5)
mid_val = _evaluate_with_joker_constraint(middle, 5, max_value=bot_val)
top_val = _evaluate_with_joker_constraint(top, 3, max_value=mid_val)

print(f"top_val={top_val}, cat={hand_category(top_val)}")
print(f"mid_val={mid_val}, cat={hand_category(mid_val)}")
print(f"bot_val={bot_val}, cat={hand_category(bot_val)}")

# FL entry check
top_cat = hand_category(top_val)
top_r1 = (top_val // (_B ** 4)) % _B
print(f"FL check: top_cat={top_cat}, top_r1={top_r1}")
if top_cat >= 3:
    print("  -> Trips! FL entry 17")
elif top_cat == 1:
    print(f"  -> Pair r1={top_r1}")
    if top_r1 >= 12:
        print(f"  -> FL entry! cards={14 + (top_r1 - 12)}")
    else:
        print("  -> Pair too low, no FL")
else:
    print("  -> No pair, no FL entry")

# Seat 1
top1 = ['X2', 'Kh', 'As']
mid1 = ['7s', '6c', 'X1', '9h', '7h']
bot1 = ['8h', '8s', '3c', '8c', '7c']

bot_val1 = evaluate_hand(bot1, 5)
mid_val1 = _evaluate_with_joker_constraint(mid1, 5, max_value=bot_val1)
top_val1 = _evaluate_with_joker_constraint(top1, 3, max_value=mid_val1)

top_cat1 = hand_category(top_val1)
top_r11 = (top_val1 // (_B ** 4)) % _B
print(f"\nSeat1: top_cat={top_cat1}, top_r1={top_r11}")
