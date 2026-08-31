import json
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai.engine.encoding import Board, Observation, encode_state, STATE_DIM, ALL_CARDS
from ai.engine.action_space import get_initial_actions, MAX_ACTIONS
from ai.models.networks import PolicyNetworkV2, PolicyNetwork

import torch.nn as nn
from collections import Counter

CARD_DIM_BASE = 18
HAND_FEAT_DIM = 6
CARD_DIM = 24
NUM_ROWS = 3
MAX_CARDS_V4 = 5

def encode_card_v4(card_str: str) -> np.ndarray:
    features = np.zeros(CARD_DIM_BASE, dtype=np.float32)
    if card_str in ('JK', 'X1', 'X2', 'Jo'):
        features[17] = 1.0
    else:
        rank = card_str[0]
        suit_char = card_str[1]
        rank_map = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,
                   '8':6,'9':7,'T':8,'J':9,'Q':10,'K':11,'A':12}
        r = rank_map.get(rank, 0)
        features[r] = 1.0
        suit_map = {'s':0,'h':1,'d':2,'c':3}
        s = suit_map.get(suit_char, 0)
        features[13 + s] = 1.0
    return features

def compute_hand_features_v4(cards: list) -> np.ndarray:
    rank_map = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,
                '8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
    ranks = [rank_map.get(c[0], 0) if c not in ('JK','X1','X2','Jo') else 15 for c in cards]
    suits = [c[1] if c not in ('JK','X1','X2','Jo') else 'joker' for c in cards]
    
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    
    extras = np.zeros((5, HAND_FEAT_DIM), dtype=np.float32)
    for i, (r, s) in enumerate(zip(ranks, suits)):
        if r == 15:
            continue
        extras[i, 0] = 1.0 if rank_counts[r] >= 2 else 0.0
        extras[i, 1] = 1.0 if rank_counts[r] >= 3 else 0.0
        extras[i, 2] = 1.0 if suit_counts[s] >= 3 else 0.0
        extras[i, 3] = 1.0 if any(abs(r - r2) == 1 for j, r2 in enumerate(ranks) if j != i and r2 != 15) else 0.0
        extras[i, 4] = 1.0 if r >= 10 else 0.0
        extras[i, 5] = 1.0 if r >= 12 and rank_counts[r] >= 2 else 0.0
    return extras

def encode_hand_v4(cards: list) -> torch.Tensor:
    base_features = np.stack([encode_card_v4(c) for c in cards])
    hand_features = compute_hand_features_v4(cards)
    features = np.concatenate([base_features, hand_features], axis=1)
    return torch.from_numpy(features).unsqueeze(0)

class T0PlacementNet(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_ff=256, dropout=0.2):
        super().__init__()
        self.card_embed = nn.Sequential(
            nn.Linear(CARD_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_CARDS_V4, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.row_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_ROWS),
        )
        self.ev_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, cards):
        x = self.card_embed(cards) + self.pos_embed[:, :cards.size(1)]
        x = self.encoder(x)
        row_logits = self.row_head(x)
        ev_pred = self.ev_head(x.mean(dim=1))
        return row_logits, ev_pred

def rust_to_python_card(card):
    if card in ("JK", "Jo"):
        return "X1"
    return card

def load_policy_net(model_path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        config = {}
        
    d_model = config.get("d_model", 128)
    num_layers = config.get("num_layers", 4)
    dropout = config.get("dropout", 0.2)
    
    model = T0PlacementNet(d_model=d_model, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(state_dict)
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

    features = encode_hand_v4(dealt_cards_python).to(device)
    with torch.no_grad():
        logits, ev_pred = model(features)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    log_probs_5x3 = log_probs[0].cpu().numpy()
    
    # Handle duplicate cards by building a mapping that pops indices
    from collections import defaultdict
    card_to_indices = defaultdict(list)
    for i, c in enumerate(dealt_cards_python):
        card_to_indices[c].append(i)
        
    row_map = {"top": 0, "middle": 1, "bottom": 2}

    action_log_probs = []
    for action in all_actions:
        lp = 0.0
        # Clone indices list so we can assign duplicate cards to distinct positions properly
        available_indices = {k: list(v) for k, v in card_to_indices.items()}
        for card, pos in action.placements:
            c_idx = available_indices[card].pop(0)
            r_idx = row_map[pos]
            lp += log_probs_5x3[c_idx, r_idx]
        action_log_probs.append(lp)
    
    action_log_probs = np.array(action_log_probs)
    max_lp = np.max(action_log_probs)
    exp_lp = np.exp(action_log_probs - max_lp)
    probs = exp_lp / np.sum(exp_lp)

    sorted_indices = np.argsort(probs)[::-1]
    
    results = []
    for idx in sorted_indices:
        action = all_actions[idx]
        pstr = action_to_placement_str(action, dealt_cards_python)
        results.append((pstr, float(probs[idx]), int(idx)))
    
    return results

def analyze_placement(p_str):
    import re
    top_m = re.search(r'Top\[([^\]]*)\]', p_str)
    mid_m = re.search(r'Mid\[([^\]]*)\]', p_str)
    bot_m = re.search(r'Bot\[([^\]]*)\]', p_str)
    
    top = [c for c in top_m.group(1).split() if c] if top_m else []
    mid = [c for c in mid_m.group(1).split() if c] if mid_m else []
    bot = [c for c in bot_m.group(1).split() if c] if bot_m else []
    
    props = []
    
    for c in top:
        if c[0] == 'A':
            props.append("A-Top(FL)")
    
    all_cards = top + mid + bot
    ranks = [c[0] for c in all_cards if len(c) >= 2]
    from collections import Counter
    rank_counts = Counter(ranks)
    pairs = [r for r, cnt in rank_counts.items() if cnt >= 2]
    
    for p in pairs:
        pair_locations = []
        for c in top:
            if c[0] == p: pair_locations.append("T")
        for c in mid:
            if c[0] == p: pair_locations.append("M")
        for c in bot:
            if c[0] == p: pair_locations.append("B")
        loc_str = "+".join(pair_locations)
        props.append("Pair{}@{}".format(p, loc_str))
    
    return props

def main():
    device = "cpu"
    model_path = r"c:\Users\Owner\.gemini\antigravity\worktrees\ofc-pineapple\verify-gcp-phase-one-20260501\ai\models\t0_placement_net_v4.pt"
    cfr_path = r"c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\ai\data\filtered_test5_out.jsonl"

    print("Loading PolicyNet v4...")
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

        pn_top_placements = [p for p, prob, idx in pn_results[:5]]
        if cfr_best_p == pn_top1_p:
            top1_match += 1
        if cfr_best_p in pn_top_placements[:3]:
            top3_match += 1
        if cfr_best_p in pn_top_placements[:5]:
            top5_match += 1

        pn_top1_ev = cfr_ev_map.get(pn_top1_p, 0)
        ev_loss = cfr_best_ev - pn_top1_ev
        total_ev_loss += ev_loss

        match_str = "MATCH" if cfr_best_p == pn_top1_p else "MISS"
        print("=" * 100)
        print("Hand {}: {} ({})  [{}]  EV Loss: {:.3f}".format(
            cfr.get("hand_idx", total_hands), hand_rust, cfr.get("type", ""), match_str, ev_loss))
        print("=" * 100)
        
        print("")
        print("  {:>4} | {:>8} {:>6} | {:>8} {:>6} | {}".format(
            "", "CFR_EV", "PN#", "PN_prob", "CFR#", "Placement"))
        print("  " + "-" * 95)
        
        for i, p in enumerate(cfr_placements[:10]):
            pn_r = pn_rank_map.get(p["p"], "?")
            pn_prob = 0
            for pp, prob, idx in pn_results:
                if pp == p["p"]:
                    pn_prob = prob
                    break
            
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
        
        print("")

    print("=" * 100)
    print("SUMMARY ({} hands)".format(total_hands))
    print("=" * 100)
    print("  Top-1 Accuracy: {}/{} ({:.1%})".format(top1_match, total_hands, top1_match/max(1,total_hands)))
    print("  Top-3 Accuracy: {}/{} ({:.1%})".format(top3_match, total_hands, top3_match/max(1,total_hands)))
    print("  Top-5 Accuracy: {}/{} ({:.1%})".format(top5_match, total_hands, top5_match/max(1,total_hands)))
    print("  Avg EV Loss:    {:.3f}".format(total_ev_loss / max(1, total_hands)))

if __name__ == "__main__":
    main()
