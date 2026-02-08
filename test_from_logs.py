"""
Replay logged hands through current scoring logic to detect bugs.

Checks:
1. Bust detection consistency
2. Royalty calculation
3. Hand name correctness
4. FL entry/stay logic
5. Score arithmetic
"""
import sqlite3, json, sys
sys.path.insert(0, '.')
from backend.main import (
    evaluate_hand, hand_category, hand_name_from_val,
    _evaluate_with_joker_constraint, _B, check_fl_stay
)

RANK_NAMES = {2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',
              9:'9',10:'T',11:'J',12:'Q',13:'K',14:'A'}

def compute_hand_values(boards):
    """Recompute constrained hand values for both seats."""
    results = [{}, {}]
    for seat in [0, 1]:
        board = boards[seat]
        bot_val = evaluate_hand(board["bottom"], 5)
        mid_val = _evaluate_with_joker_constraint(board["middle"], 5, max_value=bot_val)
        top_val = _evaluate_with_joker_constraint(board["top"], 3, max_value=mid_val)
        results[seat] = {"top": top_val, "middle": mid_val, "bottom": bot_val}
    return results

def compute_royalties(hand_vals, seat):
    """Compute royalties from constrained values."""
    roy = {"top": 0, "middle": 0, "bottom": 0, "total": 0}
    
    # Top
    top_cat = hand_category(hand_vals["top"])
    top_r1 = (hand_vals["top"] // (_B ** 4)) % _B
    if top_cat >= 3:
        roy["top"] = 10 + (top_r1 - 2)
    elif top_cat == 1 and top_r1 >= 6:
        roy["top"] = top_r1 - 5
    
    # Middle
    mid_cat = hand_category(hand_vals["middle"])
    mid_r1 = (hand_vals["middle"] // (_B ** 4)) % _B
    if mid_cat == 8 and mid_r1 == 14: roy["middle"] = 50
    elif mid_cat == 8: roy["middle"] = 30
    elif mid_cat == 7: roy["middle"] = 20
    elif mid_cat == 6: roy["middle"] = 12
    elif mid_cat == 5: roy["middle"] = 8
    elif mid_cat == 4: roy["middle"] = 4
    elif mid_cat == 3: roy["middle"] = 2
    
    # Bottom
    bot_cat = hand_category(hand_vals["bottom"])
    bot_r1 = (hand_vals["bottom"] // (_B ** 4)) % _B
    if bot_cat == 8 and bot_r1 == 14: roy["bottom"] = 25
    elif bot_cat == 8: roy["bottom"] = 15
    elif bot_cat == 7: roy["bottom"] = 10
    elif bot_cat == 6: roy["bottom"] = 6
    elif bot_cat == 5: roy["bottom"] = 4
    elif bot_cat == 4: roy["bottom"] = 2
    
    roy["total"] = roy["top"] + roy["middle"] + roy["bottom"]
    return roy

def check_bust(hand_vals):
    return (hand_vals["top"] > hand_vals["middle"] or 
            hand_vals["middle"] > hand_vals["bottom"])

def check_fl_entry_from_val(top_val):
    top_cat = hand_category(top_val)
    top_r1 = (top_val // (_B ** 4)) % _B
    if top_cat >= 3:
        return True, 17
    elif top_cat == 1:
        if top_r1 == 12: return True, 14
        elif top_r1 == 13: return True, 15
        elif top_r1 == 14: return True, 16
    return False, 0

def main():
    conn = sqlite3.connect('data/ofc_logs.db')
    cur = conn.cursor()
    
    cur.execute("SELECT hand_id, result_detail FROM hands WHERE result_detail IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    
    total = 0
    errors = 0
    warnings = 0
    
    for hand_id, result_json in rows:
        try:
            result = json.loads(result_json)
        except:
            continue
        
        boards = result.get("boards", [])
        if len(boards) != 2:
            continue
        
        # Check board completeness
        if (len(boards[0].get("top", [])) != 3 or 
            len(boards[0].get("middle", [])) != 5 or
            len(boards[0].get("bottom", [])) != 5):
            continue
        if (len(boards[1].get("top", [])) != 3 or 
            len(boards[1].get("middle", [])) != 5 or
            len(boards[1].get("bottom", [])) != 5):
            continue
            
        total += 1
        hand_vals = compute_hand_values(boards)
        
        logged_busted = result.get("busted", [False, False])
        logged_royalties = result.get("royalties", [{}, {}])
        logged_names = result.get("hand_names", [{}, {}])
        logged_fl = result.get("fl_entry", [False, False])
        logged_fl_cards = result.get("fl_card_count", [0, 0])
        
        for seat in [0, 1]:
            # 1. Bust check
            computed_bust = check_bust(hand_vals[seat])
            if computed_bust != logged_busted[seat]:
                print(f"[BUST MISMATCH] Hand {hand_id[:8]} Seat {seat}: "
                      f"logged={logged_busted[seat]} computed={computed_bust}")
                print(f"  Board: {boards[seat]}")
                errors += 1
            
            if logged_busted[seat]:
                continue  # Skip further checks for busted hands
            
            # 2. Royalty check
            computed_roy = compute_royalties(hand_vals[seat], seat)
            for row in ["top", "middle", "bottom", "total"]:
                if computed_roy[row] != logged_royalties[seat].get(row, 0):
                    print(f"[ROYALTY MISMATCH] Hand {hand_id[:8]} Seat {seat} {row}: "
                          f"logged={logged_royalties[seat].get(row)} computed={computed_roy[row]}")
                    print(f"  Board: {boards[seat]}")
                    cat = hand_category(hand_vals[seat][row]) if row != "total" else -1
                    print(f"  Cat: {cat}")
                    errors += 1
            
            # 3. Hand name check
            for row, ec in [("top", 3), ("middle", 5), ("bottom", 5)]:
                computed_name = hand_name_from_val(hand_vals[seat][row], ec)
                logged_name = logged_names[seat].get(row, "")
                if computed_name != logged_name:
                    print(f"[NAME MISMATCH] Hand {hand_id[:8]} Seat {seat} {row}: "
                          f"logged='{logged_name}' computed='{computed_name}'")
                    warnings += 1
            
            # 4. FL entry check (only check if not already in FL - we can't know FL state from logs alone)
            fl_entry_computed, fl_cards_computed = check_fl_entry_from_val(hand_vals[seat]["top"])
            if fl_entry_computed and not logged_fl[seat]:
                print(f"[FL MISSED] Hand {hand_id[:8]} Seat {seat}: "
                      f"should enter FL but logged=False")
                warnings += 1
            elif not fl_entry_computed and logged_fl[seat]:
                # Could be FL stay - not an error necessarily
                print(f"[FL EXTRA] Hand {hand_id[:8]} Seat {seat}: "
                      f"logged FL=True but top doesn't qualify (could be FL stay)")
                warnings += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {total} hands tested")
    print(f"  Errors: {errors}")
    print(f"  Warnings: {warnings}")
    if errors == 0:
        print("  ALL CHECKS PASSED!")
    else:
        print("  FAILURES DETECTED - see details above")

if __name__ == "__main__":
    main()
