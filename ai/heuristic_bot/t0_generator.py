import random
from collections import Counter
from itertools import combinations

# Card representation: Rank (2-9, T, J, Q, K, A, X1, X2), Suit (s, h, d, c)
RANKS = "23456789TJQKA"
SUITS = "shdc"

def rank_value(r):
    if r.startswith('X'): return 15 # Joker
    return RANKS.index(r) + 2

def is_straight_draw(cards):
    """Check if 4 cards form a straight draw (open or gutshot)."""
    if len(cards) < 4: return False
    vals = sorted([rank_value(c[0]) for c in cards])
    # Very simple check: gap between max and min is <= 4, and all unique
    # Or 4 cards of a straight
    if len(set(vals)) < len(vals): return False
    return vals[-1] - vals[0] <= 4

def generate_t0_placements(hand):
    """
    Given a list of 5 cards, returns a list of possible valid dicts:
    {'top': [...], 'mid': [...], 'bot': [...]}
    """
    # Parse hand
    ranks = [c[0] for c in hand]
    suits = [c[1] for c in hand if not c.startswith('X')]
    
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    
    jokers = [c for c in hand if c.startswith('X')]
    aces = [c for c in hand if c[0] == 'A']
    kings = [c for c in hand if c[0] == 'K']
    queens = [c for c in hand if c[0] == 'Q']
    
    # Identify pairs, trips, quads
    pairs = [r for r, count in rank_counts.items() if count == 2]
    trips = [r for r, count in rank_counts.items() if count == 3]
    quads = [r for r, count in rank_counts.items() if count == 4]
    
    # Sort pairs by rank value
    pairs = sorted(pairs, key=rank_value, reverse=True)
    
    placements = []
    
    # Helper to generate placement
    def make_placement(top, mid, bot):
        # Fill remaining cards by General High Card Rule
        placed = set(top + mid + bot)
        remaining = [c for c in hand if c not in placed]
        
        # Sort remaining by rank
        remaining_sorted = sorted(remaining, key=lambda c: rank_value(c[0]), reverse=True)
        
        tmp_top = list(top)
        tmp_mid = list(mid)
        tmp_bot = list(bot)
        
        for c in remaining_sorted:
            r = c[0]
            val = rank_value(r)
            
            # General rule
            if val == 14 or val == 13: # A or K
                if len(tmp_top) < 3 and len(aces) == 0: # If A is 0, Top max 1 (Wait, K is allowed to Top)
                    # User: "KやAはトップ" -> We can put K in top
                    if len(tmp_top) < 1 or (r == 'K' and 'K' in [x[0] for x in tmp_top]):
                        tmp_top.append(c)
                        continue
            
            if val >= 8:
                if len(tmp_bot) < 5:
                    tmp_bot.append(c)
                    continue
            
            if len(tmp_mid) < 5:
                tmp_mid.append(c)
            else:
                tmp_bot.append(c) # Fallback
                
        # Helper to check pair strength
        def get_pairs_count(cards):
            ranks = [c[0] for c in cards if not c.startswith('X')]
            counts = Counter(ranks)
            return sum(1 for v in counts.values() if v >= 2)

        # 1. Bot Overload (5 cards, only One Pair or High Card)
        if len(tmp_bot) == 5:
            s_counts = Counter([c[1] for c in tmp_bot if not c.startswith('X')])
            has_flush_draw = any(count >= 4 for count in s_counts.values())
            has_straight_draw = is_straight_draw(tmp_bot)
            pairs = get_pairs_count(tmp_bot)
            
            if not has_flush_draw and not has_straight_draw and pairs <= 1:
                # Move the LOWEST card to Mid
                smallest = min(tmp_bot, key=lambda c: rank_value(c[0]))
                tmp_bot.remove(smallest)
                tmp_mid.append(smallest)
                
        # 2. Bot Overload (4 cards, no pair, no draw)
        elif len(tmp_bot) == 4:
            s_counts = Counter([c[1] for c in tmp_bot if not c.startswith('X')])
            has_flush_draw = any(count >= 4 for count in s_counts.values())
            has_straight_draw = is_straight_draw(tmp_bot)
            pairs = get_pairs_count(tmp_bot)
            
            if not has_flush_draw and not has_straight_draw and pairs == 0:
                smallest = min(tmp_bot, key=lambda c: rank_value(c[0]))
                tmp_bot.remove(smallest)
                tmp_mid.append(smallest)

        # 3. Mid Overload (5 cards, only One Pair or High Card)
        if len(tmp_mid) == 5:
            pairs = get_pairs_count(tmp_mid)
            if pairs <= 1:
                # Move the HIGHEST card to Bot
                largest = max(tmp_mid, key=lambda c: rank_value(c[0]))
                tmp_mid.remove(largest)
                tmp_bot.append(largest)
                
        # 4. Mid Overload (4 cards, no pair)
        elif len(tmp_mid) == 4:
            pairs = get_pairs_count(tmp_mid)
            if pairs == 0:
                # Move the HIGHEST card to Bot
                largest = max(tmp_mid, key=lambda c: rank_value(c[0]))
                tmp_mid.remove(largest)
                tmp_bot.append(largest)
                    
        return {'top': tmp_top, 'mid': tmp_mid, 'bot': tmp_bot}

    # RULE 1: Quads
    if quads:
        r = quads[0]
        q_cards = [c for c in hand if c[0] == r]
        rest = [c for c in hand if c[0] != r]
        # Quads to Bottom
        placements.append({'top': rest, 'mid': [], 'bot': q_cards})
        return placements # Stop here if quads
        
    # RULE 2: Flush Draw (No A)
    has_flush_rule_applied = False
    if len(aces) == 0:
        for s, count in suit_counts.items():
            if count == 4:
                f_cards = [c for c in hand if c[1] == s]
                other_card = [c for c in hand if c[1] != s][0]
                if other_card[0] == 'K':
                    placements.append({'top': [other_card], 'mid': [], 'bot': f_cards})
                else:
                    placements.append({'top': [], 'mid': [other_card], 'bot': f_cards})
                has_flush_rule_applied = True
                
            elif count == 3:
                # 3 flush + Pair (<= QQ) without overlap
                f_cards = [c for c in hand if c[1] == s]
                for p_rank in pairs:
                    if rank_value(p_rank) <= 12: # <= QQ
                        p_cards = [c for c in hand if c[0] == p_rank]
                        # Check overlap
                        if len(set(f_cards).intersection(set(p_cards))) == 0:
                            # Option A: Bot = Flush, Mid = Pair
                            placements.append(make_placement(top=[], mid=p_cards, bot=f_cards))
                            # Option B: Bot = Pair, Mid = Flush
                            placements.append(make_placement(top=[], mid=f_cards, bot=p_cards))
                            has_flush_rule_applied = True
                            
    if has_flush_rule_applied:
        return placements

    # RULE 3 & 4: Aces and Pairs
    # Base states to start with
    base_states = [{'top': [], 'mid': [], 'bot': []}]
    
    if len(aces) >= 2:
        base_states = [{'top': aces[:2], 'mid': [], 'bot': []}]
    elif len(aces) == 1:
        # If we have 1 A and at least 1 K (and K is not paired)
        if len(kings) == 1:
            base_states = [
                {'top': [aces[0], kings[0]], 'mid': [], 'bot': []},            # AK in Top
                {'top': [kings[0]], 'mid': [aces[0]], 'bot': []},              # K in Top, A in Mid
                {'top': [aces[0]], 'mid': [], 'bot': [kings[0]]},              # A in Top, K in Bot
            ]
        else:
            base_states = [{'top': [aces[0]], 'mid': [], 'bot': []}]
            
    # Apply Pairs logic on top of base_states
    new_base_states = []
    for state in base_states:
        tmp_top = list(state['top'])
        tmp_mid = list(state['mid'])
        tmp_bot = list(state['bot'])
        
        # Two pair logic
        if len(pairs) == 2:
            p1, p2 = pairs # p1 is bigger
            c1 = [c for c in hand if c[0] == p1]
            c2 = [c for c in hand if c[0] == p2]
            if p1 not in ['A', 'K']:
                tmp_bot.extend(c1)
                tmp_mid.extend(c2)
                placements.append(make_placement(tmp_top, tmp_mid, tmp_bot))
                continue
                
        # Single pair logic
        if len(pairs) == 1:
            p1 = pairs[0]
            c1 = [c for c in hand if c[0] == p1]
            if p1 == 'K':
                # If we already placed K in base_states, skip to avoid duplicates/errors
                if not any(c[0] == 'K' for c in tmp_top + tmp_mid + tmp_bot):
                    placements.append(make_placement(tmp_top + c1, tmp_mid, tmp_bot)) # KK Top
                    placements.append(make_placement(tmp_top, tmp_mid, tmp_bot + c1)) # KK Bot
                else:
                    placements.append(make_placement(tmp_top, tmp_mid, tmp_bot))
                continue
            elif p1 == 'Q':
                tmp_bot.extend(c1)
                
        placements.append(make_placement(tmp_top, tmp_mid, tmp_bot))
        
    return placements


def print_placement(hand, p):
    t = " ".join(p['top']).ljust(8)
    m = " ".join(p['mid']).ljust(15)
    b = " ".join(p['bot']).ljust(15)
    print(f"Hand: {' '.join(hand)}")
    print(f"  Top: {t} | Mid: {m} | Bot: {b}")


if __name__ == "__main__":
    # Test cases defined by user's edge cases
    test_hands = [
        ["Ac", "Ah", "8c", "5d", "2s"], # 2 Aces
        ["As", "Kd", "Qs", "Jh", "9c"], # 1 Ace, High cards
        ["Ks", "Kh", "8c", "4d", "2s"], # KK pair
        ["Qs", "Qh", "8c", "4d", "2s"], # QQ pair
        ["Js", "Jh", "8c", "8d", "2s"], # Two pair (JJ, 88)
        ["8s", "8h", "3c", "3d", "2s"], # Two pair (88, 33)
        ["7s", "7h", "7d", "7c", "2s"], # Quads
        ["Ks", "8s", "5s", "2s", "Qd"], # 4 Flush with K
        ["Js", "8s", "5s", "2s", "Qd"], # 4 Flush without K
        ["8s", "5s", "2s", "Qh", "Qd"], # 3 Flush + Pair (QQ) No overlap
        ["Ks", "Qd", "Jc", "9h", "8s"], # High cards (Straight Draw)
        ["Kd", "Jc", "9h", "5s", "3c"], # High cards (No draw)
        ["Qs", "Jh", "Jc", "Td", "8h"], # Bot overload (QJJT8)
        ["7h", "4s", "3h", "3c", "2h"], # Mid overload (74332)
        ["As", "7h", "4c", "3d", "2s"], # A + 4 low cards
    ]
    
    for hand in test_hands:
        ps = generate_t0_placements(hand)
        for p in ps:
            print_placement(hand, p)
        print("-" * 50)
