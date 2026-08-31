"""
OFC Pineapple: Calculate placement combinations per turn.

Board: Top(3) + Mid(5) + Bot(5) = 13 cards
T0: Deal 5, place all 5
T1-T4: Deal 3, discard 1, place 2
"""
from itertools import product
from collections import Counter

def count_t0_placements():
    """T0: Place 5 cards into Top(≤3)/Mid(≤5)/Bot(≤5)."""
    count = 0
    # Each card assigned to T(0), M(1), or B(2)
    for assignment in product(range(3), repeat=5):
        t = assignment.count(0)
        m = assignment.count(1)
        b = assignment.count(2)
        if t <= 3 and m <= 5 and b <= 5:
            count += 1
    return count

def count_placements_2cards(cap_t, cap_m, cap_b):
    """Count valid placements for 2 distinct cards given remaining row capacities."""
    count = 0
    rows = [(0, cap_t), (1, cap_m), (2, cap_b)]
    for r1, _ in rows:
        for r2, _ in rows:
            caps = {0: cap_t, 1: cap_m, 2: cap_b}
            # Place card A in r1, card B in r2
            if r1 == r2:
                if caps[r1] >= 2:
                    count += 1
            else:
                if caps[r1] >= 1 and caps[r2] >= 1:
                    count += 1
    return count

def enumerate_all_states():
    """Enumerate all possible capacity states at each turn."""
    
    # T0: Start with (3, 5, 5), place 5 cards
    print("=" * 60)
    t0 = count_t0_placements()
    print(f"T0: 5 cards dealt, place all 5")
    print(f"  Placements: {t0}")
    print()
    
    # After T0: all possible (ct, cm, cb) where ct+cm+cb = 8, ct≤3, cm≤5, cb≤5
    # These are the remaining capacities = (3-t, 5-m, 5-b) for valid T0 placements
    
    turns = [
        ("T0", 5, 13),  # cards_to_place, total_remaining_after
        ("T1", 2, 8),
        ("T2", 2, 6),
        ("T3", 2, 4),
        ("T4", 2, 2),
    ]
    
    # Build all possible remaining-capacity vectors after each turn
    # Start: (3, 5, 5) = 13 slots
    
    # After T0: remaining = 8 slots
    # Valid: (ct, cm, cb) where ct+cm+cb=8, 0≤ct≤3, 0≤cm≤5, 0≤cb≤5
    def valid_caps(total_remaining, max_t=3, max_m=5, max_b=5):
        """All valid capacity vectors with given total remaining slots."""
        states = []
        for ct in range(min(max_t, total_remaining) + 1):
            for cm in range(min(max_m, total_remaining - ct) + 1):
                cb = total_remaining - ct - cm
                if 0 <= cb <= max_b:
                    states.append((ct, cm, cb))
        return states
    
    # For T1-T4: compute placements for each possible capacity state
    remaining_per_turn = [8, 6, 4, 2]
    
    for turn_idx, remaining in enumerate(remaining_per_turn):
        turn_name = f"T{turn_idx + 1}"
        states = valid_caps(remaining)
        
        placement_counts = []
        for ct, cm, cb in states:
            p = count_placements_2cards(ct, cm, cb)
            placement_counts.append((ct, cm, cb, p))
        
        # Total actions = 3 (discard choices) × placements
        actions = [(ct, cm, cb, 3 * p) for ct, cm, cb, p in placement_counts]
        
        all_p = [a[3] for a in actions]
        min_a = min(all_p)
        max_a = max(all_p)
        avg_a = sum(all_p) / len(all_p)
        
        print(f"{turn_name}: 3 cards dealt, discard 1, place 2 (remaining slots: {remaining})")
        print(f"  Possible board states: {len(states)}")
        print(f"  Actions (discard×placement): min={min_a}, max={max_a}, avg={avg_a:.1f}")
        
        # Show distribution
        dist = Counter(a[3] for a in actions)
        print(f"  Distribution: {dict(sorted(dist.items()))}")
        
        # Show a few representative states
        print(f"  Examples:")
        for ct, cm, cb, total_a in sorted(actions, key=lambda x: -x[3])[:3]:
            p2 = total_a // 3
            print(f"    Cap({ct},{cm},{cb}) → {p2} placements × 3 discard = {total_a} actions")
        if min_a != max_a:
            worst = min(actions, key=lambda x: x[3])
            print(f"    Cap({worst[0]},{worst[1]},{worst[2]}) → {worst[3]//3} placements × 3 discard = {worst[3]} actions")
        print()
    
    # Summary table
    print("=" * 60)
    print("Summary: OFC Pineapple Action Space per Turn")
    print("=" * 60)
    print(f"{'Turn':<6} {'Cards':<8} {'Action':<20} {'Combinations':>14}")
    print("-" * 60)
    print(f"{'T0':<6} {'5dealt':<8} {'place all 5':<20} {t0:>14}")
    
    for turn_idx, remaining in enumerate(remaining_per_turn):
        turn_name = f"T{turn_idx + 1}"
        states = valid_caps(remaining)
        all_a = [3 * count_placements_2cards(ct, cm, cb) for ct, cm, cb in states]
        min_a, max_a = min(all_a), max(all_a)
        print(f"{turn_name:<6} {'3dealt':<8} {'discard 1, place 2':<20} {min_a:>4} ~ {max_a:>4} (avg {sum(all_a)/len(all_a):.0f})")
    
    print("-" * 60)
    
    # Total game tree size (max branching)
    max_tree = t0 * 27 * 27 * 27 * 9
    typical_tree = t0 * 27 * 24 * 18 * 6
    print(f"\nGame tree branching (max):     {t0} × 27 × 27 × 27 × 9 = {max_tree:,}")
    print(f"Game tree branching (typical): {t0} × 27 × 24 × 18 × 6 = {typical_tree:,}")
    print(f"\nNote: This is per-player. Opponent has same branching.")
    print(f"Full 2-player tree: ({typical_tree:,})^2 × card_deals = astronomical")


if __name__ == '__main__':
    enumerate_all_states()
