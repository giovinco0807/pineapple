import random
from collections import Counter

RANKS = "23456789TJQKA"
SUITS = "shdc"

def rank_value(r):
    if r.startswith('X'): return 15
    return RANKS.index(r) + 2

def evaluate_row(cards):
    """Simple evaluator to check pairs/two pairs in a row."""
    if not cards: return "Empty"
    ranks = [c[0] for c in cards if not c.startswith('X')]
    jokers = len([c for c in cards if c.startswith('X')])
    counts = Counter(ranks)
    pairs = sum(1 for c in counts.values() if c >= 2)
    
    if not counts:
        return "OnePair" if jokers >= 2 else "HighCard"
        
    # Very basic evaluation for heuristic branching
    if pairs >= 2 or (pairs == 1 and jokers >= 1) or counts.most_common(1)[0][1] + jokers >= 3:
        return "TwoPairOrBetter"
    if pairs == 1 or jokers == 1:
        return "OnePair"
    return "HighCard"

def has_flush_draw(cards):
    if len(cards) < 3: return False, None
    suits = [c[1] for c in cards if not c.startswith('X')]
    if not suits: return False, None
    counts = Counter(suits)
    most_common_suit, count = counts.most_common(1)[0]
    jokers = len([c for c in cards if c.startswith('X')])
    if count + jokers >= 3:
        return True, most_common_suit
    return False, None

def generate_t1_actions(board, drawn_cards):
    """
    board: {'top': [...], 'mid': [...], 'bot': [...]}
    drawn_cards: list of 3 cards
    Returns list of valid actions: [{'place': {'top': [], 'mid': [], 'bot': []}, 'discard': [...]}]
    Note: The returned 'place' dict contains ONLY the newly placed cards.
    """
    actions = []
    
    drawn_ranks = [c[0] for c in drawn_cards]
    jokers = [c for c in drawn_cards if c.startswith('X')]
    aces = [c for c in drawn_cards if c[0] == 'A']
    
    board_top = board.get('top', [])
    board_mid = board.get('mid', [])
    board_bot = board.get('bot', [])
    
    # State flags
    top_has_AA = board_top.count('A') >= 2 or (board_top.count('A') == 1 and any(c.startswith('X') for c in board_top))
    top_has_A = board_top.count('A') == 1
    
    bot_eval = evaluate_row(board_bot)
    mid_eval = evaluate_row(board_mid)
    
    bot_has_fd, bot_flush_suit = has_flush_draw(board_bot)
    
    # We will build placements as a list of (card, location)
    # Then select 2 to place, 1 to discard
    
    # Let's generate branches instead of strict single path, as requested by user
    branches = [] # List of list of (card, location)
    
    def add_branch(placement_list):
        # Ensure we place exactly 2 cards
        if len(placement_list) == 2:
            branches.append(placement_list)
            
    # For each card in drawn_cards, where can it go?
    possible_locations = {c: [] for c in drawn_cards}
    
    # Rule 1: Joker
    for joker in jokers:
        if not top_has_AA:
            possible_locations[joker].append('top')
        elif bot_eval == "TwoPairOrBetter":
            possible_locations[joker].append('mid')
        elif bot_eval == "OnePair" and mid_eval != "OnePair" and mid_eval != "TwoPairOrBetter":
            possible_locations[joker].append('mid')
        elif bot_has_fd and len(board_bot) == 4:
            possible_locations[joker].append('mid')
        else:
            possible_locations[joker].append('bot')
            
    # Rule 2: Aces
    for a in aces:
        # User: "topでAのワンペアがなくてAを引いたらtopにA"
        if not top_has_AA:
            possible_locations[a].append('top')
            
        # User: "フラッシュドローでAが同スートならトップとボトムに置くパターンを生成"
        if bot_has_fd and a[1] == bot_flush_suit:
            possible_locations[a].append('bot')
            
    # Rule 3: Pairs with Board (Hit)
    for c in drawn_cards:
        if c.startswith('X') or c[0] == 'A': continue # Handled above
        r = c[0]
        if r in [x[0] for x in board_mid]:
            possible_locations[c].append('mid')
        if r in [x[0] for x in board_bot]:
            possible_locations[c].append('bot')
        if r in [x[0] for x in board_top]:
            possible_locations[c].append('top')
            
    # Rule 4: Drawn Pairs (e.g. 77)
    drawn_counts = Counter(drawn_ranks)
    drawn_pairs = [r for r, cnt in drawn_counts.items() if cnt >= 2 and r != 'A' and not r.startswith('X')]
    
    mid_has_pair = evaluate_row(board_mid) in ["OnePair", "TwoPairOrBetter"]
    bot_has_pair = evaluate_row(board_bot) in ["OnePair", "TwoPairOrBetter"]
    
    for r in drawn_pairs:
        pair_cards = [c for c in drawn_cards if c[0] == r]
        
        mid_empty = 5 - len(board_mid)
        bot_empty = 5 - len(board_bot)
        
        can_place_mid = mid_empty > 2 or (mid_empty == 2 and mid_has_pair)
        can_place_bot = bot_empty > 2 or (bot_empty == 2 and bot_has_pair)
        
        for c in pair_cards:
            locs = []
            if can_place_mid: locs.append('mid')
            if can_place_bot: locs.append('bot')
            
            if locs:
                possible_locations[c] = locs

    # Rule 5: Kings to Top (low priority)
    for c in drawn_cards:
        if c[0] == 'K' and 'top' not in possible_locations[c]:
            possible_locations[c].append('top')

    # Fallback for cards that have no rules applied
    for c in drawn_cards:
        if not possible_locations[c]:
            # General fallback: High to Bot, Low to Mid
            if rank_value(c[0]) >= 8:
                possible_locations[c].append('bot')
            else:
                possible_locations[c].append('mid')

    # --- Discard Logic ---
    # Assign a penalty to discarding each card (higher penalty = keep this card)
    discard_penalties = {}
    for c in drawn_cards:
        penalty = rank_value(c[0]) * 5  # Base penalty based on rank (2=10, K=65)
        if c.startswith('X'):
            penalty += 10000
        if c[0] == 'A':
            penalty += 5000
            
        # Hit on board
        if c[0] in [x[0] for x in board_mid + board_bot + board_top]:
            penalty += 1000
            
        # Drawn pair
        if c[0] in drawn_pairs:
            penalty += 800
            
        # Flush draw connection
        if bot_has_fd and c[1] == bot_flush_suit:
            penalty += 600
            
        # Straight draw connection (very basic check)
        bot_ranks = [rank_value(x[0]) for x in board_bot if not x.startswith('X')]
        if bot_ranks and min(bot_ranks) - 2 <= rank_value(c[0]) <= max(bot_ranks) + 2:
            penalty += 100
            
        discard_penalties[c] = penalty

    # Find the card with the LOWEST penalty to discard
    # If tie, random choice or just pick the first minimum
    worst_card = min(drawn_cards, key=lambda c: discard_penalties[c])
    
    # We only place the other two cards
    cards_to_place = [c for c in drawn_cards if c != worst_card]
    c1, c2 = cards_to_place[0], cards_to_place[1]
    
    valid_placements = []
    
    locs1 = possible_locations[c1]
    locs2 = possible_locations[c2]
    
    from itertools import product
    for l1, l2 in product(locs1, locs2):
        # Check slot limits
        sim_top = len(board_top) + (1 if l1=='top' else 0) + (1 if l2=='top' else 0)
        sim_mid = len(board_mid) + (1 if l1=='mid' else 0) + (1 if l2=='mid' else 0)
        sim_bot = len(board_bot) + (1 if l1=='bot' else 0) + (1 if l2=='bot' else 0)
        
        if sim_top <= 3 and sim_mid <= 5 and sim_bot <= 5:
            place_dict = {'top': [], 'mid': [], 'bot': []}
            place_dict[l1].append(c1)
            place_dict[l2].append(c2)
            
            # Check for Drawn Pair rule violation
            if c1[0] == c2[0] and l1 != l2:
                # Only force them together if there is a safe row for them
                mid_empty = 5 - len(board_mid)
                bot_empty = 5 - len(board_bot)
                mid_has_pair = evaluate_row(board_mid) in ["OnePair", "TwoPairOrBetter"]
                bot_has_pair = evaluate_row(board_bot) in ["OnePair", "TwoPairOrBetter"]
                
                safe_mid = mid_empty > 2 or (mid_empty == 2 and mid_has_pair)
                safe_bot = bot_empty > 2 or (bot_empty == 2 and bot_has_pair)
                
                if safe_mid or safe_bot:
                    continue # Disallow splitting!
            
            valid_placements.append({
                'place': place_dict,
                'discard': [worst_card]
            })

    # Deduplicate
    unique_placements = []
    seen = set()
    for p in valid_placements:
        # Create a signature
        sig = str(sorted(p['place']['top'])) + str(sorted(p['place']['mid'])) + str(sorted(p['place']['bot']))
        if sig not in seen:
            seen.add(sig)
            unique_placements.append(p)
            
    return unique_placements

def print_t1_action(board, drawn, action):
    p = action['place']
    d = action['discard']
    print(f"Drawn: {' '.join(drawn)}")
    print(f"  Place -> Top: {' '.join(p['top']).ljust(5)} | Mid: {' '.join(p['mid']).ljust(5)} | Bot: {' '.join(p['bot']).ljust(5)}  [Discard: {d[0]}]")

if __name__ == "__main__":
    # Test cases based on user rules
    
    # Case 1: Drawn Joker. Top has no AA.
    b1 = {'top': ['2s'], 'mid': ['5d', '3c'], 'bot': ['9h', '8d']}
    a1 = generate_t1_actions(b1, ['X1', 'Qc', '4s'])
    print("--- Case 1: Joker drawn, Top has no AA ---")
    for a in a1: print_t1_action(b1, ['X1', 'Qc', '4s'], a)

    # Case 2: Drawn A matching Bottom flush draw
    b2 = {'top': ['Kd'], 'mid': ['3h'], 'bot': ['8s', '5s', '2s']}
    a2 = generate_t1_actions(b2, ['As', 'Qh', '4c'])
    print("\n--- Case 2: Drawn A matching Bottom flush draw ---")
    for a in a2: print_t1_action(b2, ['As', 'Qh', '4c'], a)

    # Case 3: Drawn pair (77)
    b3 = {'top': ['Ac'], 'mid': ['2h', '3h', '4h'], 'bot': ['Kc', 'Qc']}
    a3 = generate_t1_actions(b3, ['7s', '7d', '9c'])
    print("\n--- Case 3: Drawn pair (77). Mid has 3 cards (2 empty). Bot has 2 cards (3 empty) ---")
    # Should not place in Mid! Must place in Bot.
    for a in a3: print_t1_action(b3, ['7s', '7d', '9c'], a)
    
    # Case 4: Drawn K (low priority Top)
    b4 = {'top': ['Ac'], 'mid': ['2h', '3h'], 'bot': ['Qc', 'Jc']}
    a4 = generate_t1_actions(b4, ['Ks', '9d', '4c'])
    print("\n--- Case 4: Drawn K ---")
    for a in a4: print_t1_action(b4, ['Ks', '9d', '4c'], a)

