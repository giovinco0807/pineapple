"""
Large-scale PolicyNet v2 accuracy test.
1. Test against Phase 1-2 training data (336 hands) - in-sample accuracy
2. Generate 200 random hands and check strategic patterns
"""
import json
import sys
import glob
import random
from pathlib import Path
from collections import Counter, defaultdict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state, STATE_DIM, ALL_CARDS
from ai.engine.action_space import get_initial_actions, MAX_ACTIONS
from ai.models.networks import PolicyNetworkV2

PYTHON_RANKS = "23456789TJQKA"
PYTHON_SUITS = "hdcs"


def rust_to_python_card(card):
    if card in ("JK", "Jo"):
        return "X1"
    return card


def load_policy_net(model_path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    first_key = "input_proj.0.weight"
    input_dim = checkpoint[first_key].shape[1] if first_key in checkpoint else STATE_DIM
    model = PolicyNetworkV2(input_dim=input_dim, max_actions=MAX_ACTIONS)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model


def action_to_placement_str(action, dealt_cards):
    card_to_pos = {}
    for card, pos in action.placements:
        card_to_pos[card] = pos
    by_pos = {"top": [], "middle": [], "bottom": []}
    for card in dealt_cards:
        pos = card_to_pos.get(card)
        if pos:
            by_pos[pos].append(card)
    top_str = " ".join(by_pos["top"])
    mid_str = " ".join(by_pos["middle"])
    bot_str = " ".join(by_pos["bottom"])
    return "Top[{}] Mid[{}] Bot[{}]".format(top_str, mid_str, bot_str)


def predict_for_hand(model, dealt_cards_python, device="cpu"):
    board = Board()
    all_actions = get_initial_actions(dealt_cards_python, board)
    n_actions = len(all_actions)
    if n_actions == 0:
        return []

    obs = Observation(
        board_self=Board(),
        board_opponent=Board(),
        dealt_cards=dealt_cards_python,
        known_discards_self=[],
        turn=0,
        is_btn=True,
    )
    state = encode_state(obs)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    valid_mask = torch.zeros(1, MAX_ACTIONS, dtype=torch.bool, device=device)
    valid_mask[0, :n_actions] = True

    with torch.no_grad():
        probs = model(state_tensor, valid_mask)
    probs = probs[0].cpu().numpy()

    sorted_indices = np.argsort(probs)[::-1]
    results = []
    for idx in sorted_indices:
        if idx < n_actions:
            action = all_actions[idx]
            pstr = action_to_placement_str(action, dealt_cards_python)
            results.append((pstr, float(probs[idx])))
    return results


def parse_placement(p_str):
    """Parse placement string into top/mid/bot card lists."""
    import re
    top_m = re.search(r'Top\[([^\]]*)\]', p_str)
    mid_m = re.search(r'Mid\[([^\]]*)\]', p_str)
    bot_m = re.search(r'Bot\[([^\]]*)\]', p_str)
    top = [c for c in top_m.group(1).split() if c] if top_m else []
    mid = [c for c in mid_m.group(1).split() if c] if mid_m else []
    bot = [c for c in bot_m.group(1).split() if c] if bot_m else []
    return top, mid, bot


def classify_hand(cards):
    """Classify a 5-card hand type."""
    ranks = [c[0] for c in cards if len(c) >= 2]
    suits = [c[-1] for c in cards if len(c) >= 2]
    rc = Counter(ranks)
    sc = Counter(suits)
    
    has_ace = 'A' in ranks
    has_pair = any(v >= 2 for v in rc.values())
    has_two_pair = sum(1 for v in rc.values() if v >= 2) >= 2
    has_trips = any(v >= 3 for v in rc.values())
    has_suited_3 = any(v >= 3 for v in sc.values())
    
    if has_trips:
        return "Trips"
    elif has_two_pair:
        return "TwoPair"
    elif has_pair:
        return "OnePair"
    elif has_ace:
        return "Ace-High"
    else:
        return "NoPair-NoAce"


def main():
    device = "cpu"
    model_path = r"c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\ai\models\t0_policynet_v2\bc_policy_best.pt"

    print("Loading PolicyNet v2...")
    model = load_policy_net(model_path, device)
    print("Loaded.\n")

    # ========================================
    # Part 1: Test against Phase 1-2 data (336 hands)
    # ========================================
    print("=" * 80)
    print("PART 1: In-Sample Test (Phase 1-2 Data, 336 hands)")
    print("=" * 80)
    
    top1_match = 0
    top3_match = 0
    top5_match = 0
    top10_match = 0
    total = 0
    ev_losses = []
    misses_by_type = defaultdict(list)
    
    for f in sorted(glob.glob(r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\ai\data\t0_gcs\*.jsonl')):
        for line in open(f, encoding='utf-8'):
            rec = json.loads(line.strip())
            hand_rust = rec['hand']
            hand_cards_rust = hand_rust.split()
            hand_cards_python = [rust_to_python_card(c) for c in hand_cards_rust]
            
            cfr_sorted = sorted(rec['placements'], key=lambda x: x['ev'], reverse=True)
            cfr_best_p = cfr_sorted[0]['p']
            cfr_best_ev = cfr_sorted[0]['ev']
            cfr_ev_map = {p['p']: p['ev'] for p in cfr_sorted}
            
            pn_results = predict_for_hand(model, hand_cards_python, device)
            if not pn_results:
                continue
            
            total += 1
            pn_top_ps = [p for p, prob in pn_results]
            
            if cfr_best_p == pn_top_ps[0]:
                top1_match += 1
            if cfr_best_p in pn_top_ps[:3]:
                top3_match += 1
            if cfr_best_p in pn_top_ps[:5]:
                top5_match += 1
            if cfr_best_p in pn_top_ps[:10]:
                top10_match += 1
            
            pn_top1_ev = cfr_ev_map.get(pn_top_ps[0], 0)
            loss = cfr_best_ev - pn_top1_ev
            ev_losses.append(loss)
            
            htype = classify_hand(hand_cards_python)
            if cfr_best_p != pn_top_ps[0]:
                misses_by_type[htype].append(loss)
    
    print("\n  Total hands tested: {}".format(total))
    print("  Top-1  Accuracy: {}/{} ({:.1%})".format(top1_match, total, top1_match/total))
    print("  Top-3  Accuracy: {}/{} ({:.1%})".format(top3_match, total, top3_match/total))
    print("  Top-5  Accuracy: {}/{} ({:.1%})".format(top5_match, total, top5_match/total))
    print("  Top-10 Accuracy: {}/{} ({:.1%})".format(top10_match, total, top10_match/total))
    
    ev_arr = np.array(ev_losses)
    print("\n  EV Loss Stats:")
    print("    Mean:   {:.3f}".format(ev_arr.mean()))
    print("    Median: {:.3f}".format(np.median(ev_arr)))
    print("    Std:    {:.3f}".format(ev_arr.std()))
    print("    Min:    {:.3f}".format(ev_arr.min()))
    print("    Max:    {:.3f}".format(ev_arr.max()))
    print("    P90:    {:.3f}".format(np.percentile(ev_arr, 90)))
    
    # EV loss distribution
    bins = [0, 0.5, 1, 2, 5, 10, 20, 100]
    print("\n  EV Loss Distribution:")
    for i in range(len(bins)-1):
        cnt = sum(1 for x in ev_losses if bins[i] <= x < bins[i+1])
        pct = cnt / total * 100
        bar = "#" * int(pct)
        print("    [{:>5.1f}, {:>5.1f}): {:>4} ({:>5.1f}%) {}".format(
            bins[i], bins[i+1], cnt, pct, bar))
    
    print("\n  Miss Rate by Hand Type:")
    for htype in sorted(misses_by_type.keys()):
        losses = misses_by_type[htype]
        # Count total of this type
        print("    {:<15}: {} misses, avg loss {:.3f}".format(
            htype, len(losses), np.mean(losses)))

    # ========================================
    # Part 2: Strategic Pattern Check (200 random hands)
    # ========================================
    print("\n" + "=" * 80)
    print("PART 2: Strategic Pattern Analysis (200 random hands)")
    print("=" * 80)
    
    rng = random.Random(42)
    deck = []
    for s in PYTHON_SUITS:
        for r in PYTHON_RANKS:
            deck.append("{}{}".format(r, s))
    deck.append("X1")
    deck.append("X2")
    
    ace_hands = 0
    ace_top_count = 0
    pair_hands = 0
    pair_same_row = 0
    two_pair_hands = 0
    two_pair_separated = 0
    
    ace_top_details = []
    pair_details = []
    
    for _ in range(200):
        d = deck[:]
        rng.shuffle(d)
        hand = d[:5]
        
        pn_results = predict_for_hand(model, hand, device)
        if not pn_results:
            continue
        
        top1_p = pn_results[0][0]
        top, mid, bot = parse_placement(top1_p)
        
        ranks = [c[0] for c in hand if len(c) >= 2]
        rc = Counter(ranks)
        
        # Check Ace handling
        has_ace = 'A' in ranks
        if has_ace:
            ace_hands += 1
            ace_in_top = any(c[0] == 'A' for c in top)
            if ace_in_top:
                ace_top_count += 1
            ace_top_details.append((
                " ".join(hand), top1_p, ace_in_top, pn_results[0][1]
            ))
        
        # Check pair handling
        pairs = [r for r, cnt in rc.items() if cnt >= 2]
        if len(pairs) >= 1:
            pair_hands += 1
            for p in pairs:
                # Check if both cards of pair are in same row
                in_top = sum(1 for c in top if c[0] == p)
                in_mid = sum(1 for c in mid if c[0] == p)
                in_bot = sum(1 for c in bot if c[0] == p)
                if in_top >= 2 or in_mid >= 2 or in_bot >= 2:
                    pair_same_row += 1
                    break
        
        if len(pairs) >= 2:
            two_pair_hands += 1
            # Check if pairs are in different rows
            pair_rows = []
            for p in pairs:
                in_top = sum(1 for c in top if c[0] == p)
                in_mid = sum(1 for c in mid if c[0] == p)
                in_bot = sum(1 for c in bot if c[0] == p)
                if in_mid >= 2:
                    pair_rows.append("M")
                elif in_bot >= 2:
                    pair_rows.append("B")
                elif in_top >= 2:
                    pair_rows.append("T")
                else:
                    pair_rows.append("split")
            if len(set(pair_rows)) >= 2 and "split" not in pair_rows:
                two_pair_separated += 1
            pair_details.append((" ".join(hand), top1_p, pair_rows))
    
    print("\n  Ace-High Hands: {} / 200 random hands".format(ace_hands))
    print("  Ace placed in Top: {}/{} ({:.1%})".format(
        ace_top_count, ace_hands, ace_top_count/max(1,ace_hands)))
    print("  (CFR optimal: A-Top is almost always best for FL entry)")
    
    print("\n  First 10 Ace hands (PN prediction):")
    for hand, p, is_top, prob in ace_top_details[:10]:
        status = "A-Top OK" if is_top else "A-NOT-Top !!"
        print("    {} -> {} [{}, p={:.3f}]".format(hand, p, status, prob))
    
    print("\n  Pair Hands: {} / 200 random hands".format(pair_hands))
    print("  Pair in same row: {}/{} ({:.1%})".format(
        pair_same_row, pair_hands, pair_same_row/max(1,pair_hands)))
    
    print("\n  Two-Pair Hands: {} / 200 random hands".format(two_pair_hands))
    print("  Pairs separated (Mid/Bot): {}/{} ({:.1%})".format(
        two_pair_separated, two_pair_hands, two_pair_separated/max(1,two_pair_hands)))
    print("  (CFR optimal: separate pairs into Mid+Bot)")
    
    print("\n  First 10 Two-Pair hands:")
    tp_shown = 0
    for hand, p, rows in pair_details:
        if len(rows) >= 2:
            tp_shown += 1
            print("    {} -> {} [rows: {}]".format(hand, p, rows))
            if tp_shown >= 10:
                break


if __name__ == "__main__":
    main()
