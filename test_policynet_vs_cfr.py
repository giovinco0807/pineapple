"""
Detailed comparison of PolicyNet v2 predictions vs high-fidelity CFR results.
Shows full ranking comparison, probability distribution, and strategic analysis.
"""
import json
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state, STATE_DIM, ALL_CARDS
from ai.engine.action_space import get_initial_actions, MAX_ACTIONS
from ai.models.networks import PolicyNetworkV2


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
            results.append((pstr, float(probs[idx]), int(idx)))
    
    return results


def analyze_placement(p_str):
    """Analyze strategic properties of a placement."""
    import re
    top_m = re.search(r'Top\[([^\]]*)\]', p_str)
    mid_m = re.search(r'Mid\[([^\]]*)\]', p_str)
    bot_m = re.search(r'Bot\[([^\]]*)\]', p_str)
    
    top = [c for c in top_m.group(1).split() if c] if top_m else []
    mid = [c for c in mid_m.group(1).split() if c] if mid_m else []
    bot = [c for c in bot_m.group(1).split() if c] if bot_m else []
    
    props = []
    
    # Check for Ace in top (FL entry strategy)
    rank_map = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
    
    for c in top:
        if c[0] == 'A':
            props.append("A-Top(FL)")
    
    # Check for pairs
    all_cards = top + mid + bot
    ranks = [c[0] for c in all_cards if len(c) >= 2]
    from collections import Counter
    rank_counts = Counter(ranks)
    pairs = [r for r, cnt in rank_counts.items() if cnt >= 2]
    
    for p in pairs:
        # Where are the pair cards?
        pair_locations = []
        for c in top:
            if c[0] == p: pair_locations.append("T")
        for c in mid:
            if c[0] == p: pair_locations.append("M")
        for c in bot:
            if c[0] == p: pair_locations.append("B")
        loc_str = "+".join(pair_locations)
        props.append("Pair{}@{}".format(p, loc_str))
    
    # Check suited cards per row
    for row_name, row_cards in [("T", top), ("M", mid), ("B", bot)]:
        suits = [c[-1] for c in row_cards if len(c) >= 2]
        suit_counts = Counter(suits)
        for s, cnt in suit_counts.items():
            if cnt >= 2:
                props.append("Suited{}x{}@{}".format(s, cnt, row_name))
    
    return props


def main():
    device = "cpu"
    model_path = r"c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\ai\models\t0_policynet_v2\bc_policy_best.pt"
    cfr_path = r"c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\ai\data\filtered_test5_out.jsonl"

    print("Loading PolicyNet v2...")
    model = load_policy_net(model_path, device)
    print("Loaded.\n")

    total_hands = 0
    top1_match = 0
    top3_match = 0
    top5_match = 0
    total_ev_loss = 0

    for line in open(cfr_path, encoding="utf-8"):
        cfr = json.loads(line.strip())
        hand_rust = cfr["hand"]
        hand_cards_rust = hand_rust.split()
        hand_cards_python = [rust_to_python_card(c) for c in hand_cards_rust]

        cfr_placements = cfr["placements"]
        cfr_best_p = cfr_placements[0]["p"]
        cfr_best_ev = cfr_placements[0]["ev"]
        cfr_ev_map = {p["p"]: p["ev"] for p in cfr_placements}
        cfr_rank_map = {p["p"]: i+1 for i, p in enumerate(cfr_placements)}

        pn_results = predict_for_hand(model, hand_cards_python, device)
        if not pn_results:
            continue
        
        total_hands += 1
        pn_top1_p = pn_results[0][0]
        pn_rank_map = {p: i+1 for i, (p, prob, idx) in enumerate(pn_results)}

        # Check top-N accuracy
        pn_top_placements = [p for p, prob, idx in pn_results[:5]]
        if cfr_best_p == pn_top1_p:
            top1_match += 1
        if cfr_best_p in pn_top_placements[:3]:
            top3_match += 1
        if cfr_best_p in pn_top_placements[:5]:
            top5_match += 1

        # EV loss
        pn_top1_ev = cfr_ev_map.get(pn_top1_p, 0)
        ev_loss = cfr_best_ev - pn_top1_ev
        total_ev_loss += ev_loss

        # Print detailed analysis
        match_str = "MATCH" if cfr_best_p == pn_top1_p else "MISS"
        print("=" * 100)
        print("Hand {}: {} ({})  [{}]  EV Loss: {:.3f}".format(
            cfr["hand_idx"], hand_rust, cfr["type"], match_str, ev_loss))
        print("=" * 100)
        
        # Side-by-side: CFR ranking vs PolicyNet ranking
        print("")
        print("  {:>4} | {:>8} {:>6} | {:>8} {:>6} | {}".format(
            "", "CFR_EV", "PN#", "PN_prob", "CFR#", "Placement"))
        print("  " + "-" * 95)
        
        # Show CFR top-20
        for i, p in enumerate(cfr_placements[:20]):
            pn_r = pn_rank_map.get(p["p"], "?")
            pn_prob = 0
            for pp, prob, idx in pn_results:
                if pp == p["p"]:
                    pn_prob = prob
                    break
            
            # Strategic analysis
            props = analyze_placement(p["p"])
            props_str = " | ".join(props) if props else ""
            
            marker = ""
            if i == 0:
                marker = " <-- CFR BEST"
            if p["p"] == pn_top1_p:
                marker = " <-- PN TOP-1"
            if i == 0 and p["p"] == pn_top1_p:
                marker = " <-- BOTH BEST"
            
            print("  {:>4} | {:>8.3f} {:>6} | {:>8.4f} {:>6} | {}{}  [{}]".format(
                i+1, p["ev"], pn_r, pn_prob, "", p["p"], marker, props_str))
        
        # Show PolicyNet top-10 (with CFR EV)
        print("")
        print("  PolicyNet Top-10:")
        print("  {:>4} | {:>8} {:>6} | {:>8} | {}".format(
            "PN#", "PN_prob", "CFR#", "CFR_EV", "Placement"))
        print("  " + "-" * 85)
        for i, (p, prob, idx) in enumerate(pn_results[:10]):
            cfr_r = cfr_rank_map.get(p, "?")
            cfr_ev = cfr_ev_map.get(p, 0)
            props = analyze_placement(p)
            props_str = " | ".join(props) if props else ""
            marker = " <-- CFR BEST" if p == cfr_best_p else ""
            print("  {:>4} | {:>8.4f} {:>6} | {:>8.3f} | {}{}  [{}]".format(
                i+1, prob, cfr_r, cfr_ev, p, marker, props_str))
        
        # Probability distribution analysis
        print("")
        top5_prob = sum(prob for _, prob, _ in pn_results[:5])
        top10_prob = sum(prob for _, prob, _ in pn_results[:10])
        print("  Prob distribution: Top1={:.1%} Top5={:.1%} Top10={:.1%} Entropy={:.2f}".format(
            pn_results[0][1], top5_prob, top10_prob,
            -sum(prob * np.log(prob + 1e-10) for _, prob, _ in pn_results if prob > 0)))
        print("")

    # Summary
    print("=" * 100)
    print("SUMMARY ({} hands)".format(total_hands))
    print("=" * 100)
    print("  Top-1 Accuracy: {}/{} ({:.1%})".format(top1_match, total_hands, top1_match/max(1,total_hands)))
    print("  Top-3 Accuracy: {}/{} ({:.1%})".format(top3_match, total_hands, top3_match/max(1,total_hands)))
    print("  Top-5 Accuracy: {}/{} ({:.1%})".format(top5_match, total_hands, top5_match/max(1,total_hands)))
    print("  Avg EV Loss:    {:.3f}".format(total_ev_loss / max(1, total_hands)))


if __name__ == "__main__":
    main()
