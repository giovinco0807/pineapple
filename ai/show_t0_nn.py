#!/usr/bin/env python3
"""T0 NNの配置を確認するスクリプト。10ハンドの上位5配置を表示。"""
import random, sys, itertools
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_t0_placement import T0PlacementNet, CARD_DIM, encode_card, compute_hand_features

RANKS_STR = "23456789TJQKA"
SUITS_STR = "shdc"
ALL_CARDS = [f"{r}{s}" for r in RANKS_STR for s in SUITS_STR] + ["X1", "X2"]
SUIT_NAMES = {'s':'spades','h':'hearts','d':'diamonds','c':'clubs'}
ROW = {0:'Top', 1:'Mid', 2:'Bot'}

ALL_T0 = [p for p in itertools.product([0,1,2], repeat=5) if p.count(0) <= 3]
T0_TENSOR = torch.tensor(ALL_T0, dtype=torch.long)

def card_dict(s):
    if s in ('X1','X2','JK'): return {'rank':'Joker','suit':'joker'}
    return {'rank':s[0], 'suit':SUIT_NAMES.get(s[1],s[1])}

def load_model():
    path = Path(__file__).parent / 'models' / 't0_placement_net_v4.pt'
    dev = torch.device('cpu')
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    cfg = ckpt.get('config', {})
    model = T0PlacementNet(
        d_model=cfg.get('d_model',128), nhead=4,
        num_layers=cfg.get('num_layers',4),
        dim_ff=cfg.get('d_model',128)*2,
        dropout=cfg.get('dropout',0.2)
    ).to(dev)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, dev

def get_top_placements(model, dev, hand, top_k=5):
    dicts = [card_dict(c) for c in hand]
    base = np.stack([encode_card(c) for c in dicts])
    hf = compute_hand_features(dicts)
    feat = torch.from_numpy(np.concatenate([base, hf], axis=1)).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits, _ = model(feat)
        log_p = F.log_softmax(logits, dim=-1)
    vt = T0_TENSOR.to(dev)
    scores = sum(log_p[0:1, i, vt[:, i]] for i in range(5))
    probs = torch.softmax(scores[0], dim=0)
    vals, idx = torch.topk(probs, top_k)
    results = []
    for rank, (i, prob) in enumerate(zip(idx.tolist(), vals.tolist())):
        p = ALL_T0[i]
        top = [hand[j] for j in range(5) if p[j]==0]
        mid = [hand[j] for j in range(5) if p[j]==1]
        bot = [hand[j] for j in range(5) if p[j]==2]
        results.append((rank+1, prob, top, mid, bot))
    return results

def main():
    model, dev = load_model()
    rng = random.Random(12345)
    
    for h in range(10):
        deck = list(ALL_CARDS)
        rng.shuffle(deck)
        hand = deck[:5]
        
        print(f"\n{'='*60}")
        print(f"  Hand #{h+1}:  {' '.join(hand)}")
        print(f"{'='*60}")
        
        placements = get_top_placements(model, dev, hand, top_k=5)
        for rank, prob, top, mid, bot in placements:
            print(f"  #{rank} ({prob*100:5.1f}%)  Top[{' '.join(top):>10s}]  "
                  f"Mid[{' '.join(mid):>16s}]  Bot[{' '.join(bot):>16s}]")

if __name__ == '__main__':
    main()
