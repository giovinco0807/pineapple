"""
Check two things:
1. Training data sample size analysis
2. Does PolicyNet top-50 contain CFR optimal placement? (336 hands)
"""
import json
import sys
import glob
from pathlib import Path
from collections import Counter

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


def get_all_placements_ranked(model, dealt_cards_python, device="cpu"):
    board = Board()
    all_actions = get_initial_actions(dealt_cards_python, board)
    n_actions = len(all_actions)
    if n_actions == 0:
        return [], 0

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
    return results, n_actions


def main():
    device = "cpu"
    model_path = r"c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\ai\models\t0_policynet_v2\bc_policy_best.pt"

    print("Loading PolicyNet v2...")
    model = load_policy_net(model_path, device)
    print("Loaded.\n")

    # ========================================
    # Top-50 filter check against Phase 1-2 data
    # ========================================
    print("=" * 80)
    print("Top-50 Filter: Does it contain CFR optimal? (336 hands)")
    print("=" * 80)

    contained_in_50 = 0
    contained_in_30 = 0
    contained_in_20 = 0
    contained_in_10 = 0
    total = 0
    missed_hands = []
    cfr_best_pn_ranks = []

    for f in sorted(glob.glob(r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\ai\data\t0_gcs\*.jsonl')):
        for line in open(f, encoding='utf-8'):
            rec = json.loads(line.strip())
            hand_rust = rec['hand']
            hand_cards_rust = hand_rust.split()
            hand_cards_python = [rust_to_python_card(c) for c in hand_cards_rust]

            cfr_sorted = sorted(rec['placements'], key=lambda x: x['ev'], reverse=True)
            cfr_best_p = cfr_sorted[0]['p']
            cfr_best_ev = cfr_sorted[0]['ev']

            pn_results, n_total = get_all_placements_ranked(model, hand_cards_python, device)
            if not pn_results:
                continue

            total += 1
            pn_placements = [p for p, prob in pn_results]

            # Find CFR best's rank in PolicyNet
            pn_rank = None
            for i, (p, prob) in enumerate(pn_results):
                if p == cfr_best_p:
                    pn_rank = i + 1
                    break

            if pn_rank is not None:
                cfr_best_pn_ranks.append(pn_rank)
                if pn_rank <= 10:
                    contained_in_10 += 1
                if pn_rank <= 20:
                    contained_in_20 += 1
                if pn_rank <= 30:
                    contained_in_30 += 1
                if pn_rank <= 50:
                    contained_in_50 += 1
                else:
                    missed_hands.append((hand_rust, pn_rank, n_total, cfr_best_ev, cfr_best_p))
            else:
                cfr_best_pn_ranks.append(999)
                missed_hands.append((hand_rust, "NOT FOUND", n_total, cfr_best_ev, cfr_best_p))

    print("\n  CFR optimal placement in PolicyNet top-N:")
    print("    Top-10:  {}/{} ({:.1%})".format(contained_in_10, total, contained_in_10/total))
    print("    Top-20:  {}/{} ({:.1%})".format(contained_in_20, total, contained_in_20/total))
    print("    Top-30:  {}/{} ({:.1%})".format(contained_in_30, total, contained_in_30/total))
    print("    Top-50:  {}/{} ({:.1%})".format(contained_in_50, total, contained_in_50/total))

    ranks = np.array(cfr_best_pn_ranks)
    print("\n  CFR optimal's PN rank stats:")
    print("    Mean:   {:.1f}".format(ranks.mean()))
    print("    Median: {:.1f}".format(np.median(ranks)))
    print("    P75:    {:.1f}".format(np.percentile(ranks, 75)))
    print("    P90:    {:.1f}".format(np.percentile(ranks, 90)))
    print("    P95:    {:.1f}".format(np.percentile(ranks, 95)))
    print("    Max:    {}".format(ranks.max()))

    # Distribution
    print("\n  PN rank distribution:")
    bins = [(1,1), (2,3), (4,5), (6,10), (11,20), (21,30), (31,50), (51,100), (101,999)]
    for lo, hi in bins:
        cnt = sum(1 for r in cfr_best_pn_ranks if lo <= r <= hi)
        pct = cnt / total * 100
        bar = "#" * int(pct / 2)
        label = "#{}" if lo == hi else "#{}-{}"
        print("    {:>8}: {:>4} ({:>5.1f}%) {}".format(
            label.format(lo, hi) if lo != hi else label.format(lo),
            cnt, pct, bar))

    if missed_hands:
        print("\n  Hands where CFR optimal is OUTSIDE top-50 ({} hands):".format(len(missed_hands)))
        for hand, rank, n_total, ev, p in missed_hands[:20]:
            print("    {} -> PN rank {} / {} total, EV={:.2f}".format(hand, rank, n_total, ev))

    # ========================================
    # Training Data Size Analysis
    # ========================================
    print("\n" + "=" * 80)
    print("Training Data Size Analysis")
    print("=" * 80)
    print()
    print("  Model: PolicyNetworkV2")
    print("  Parameters: 2,835,962 (~2.8M)")
    print()
    print("  Training Data:")
    print("    Unique hands:     329 (after skipping)")
    print("    x24 augmentation: 7,896 samples")
    print("    But effective unique decisions: 329")
    print()
    print("  Action Space:")
    print("    Per hand: 60 - 232 valid placements")
    print("    MAX_ACTIONS: 250")
    print()
    print("  Ratio Analysis:")
    print("    Parameters / Unique samples:  {:.0f}x overparameterized".format(2835962 / 329))
    print("    Parameters / Aug samples:     {:.0f}x overparameterized".format(2835962 / 7896))
    print("    Typical rule: params < 10x samples for generalization")
    print()
    print("  CONCLUSION:")
    print("    2.8M params vs 329 unique hands = 8,620x overparameterized")
    print("    The model has memorized the 329 training hands")
    print("    but cannot generalize to unseen hands.")
    print("    Even with 24x suit augmentation, effective diversity is very low.")
    print()
    print("  Recommended minimum samples: 10,000 - 50,000 unique hands")
    print("  Current: 329 -> need 30x - 150x more data")


if __name__ == "__main__":
    main()
