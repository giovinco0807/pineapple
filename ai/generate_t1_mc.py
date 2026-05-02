#!/usr/bin/env python3
"""
T1 Training Data Generator using Nested Expectimax.

Pipeline:
1. Deal 5 random cards (T0)
2. Use T0 NN to prune 232 placements -> Top 50, sample one
3. Deal 3 cards (T1 hand)
4. Enumerate ALL T1 actions -> for each, nested Expectimax:
   - T2: n2 random deals x all actions -> max
   - T3: n3 random deals x all actions -> max
   - T4: ALL possible deals x all actions -> max -> terminal eval
5. Output T1 training record in JSONL format

Usage:
    python ai/generate_t1_mc.py --n-hands 50000 --n2 3 --n3 3 --workers 30
"""
import argparse, json, random, time, sys, os, itertools
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from train_t0_placement import T0PlacementNet, CARD_DIM, NUM_ROWS, MAX_CARDS, encode_card, compute_hand_features

RANKS_STR = "23456789TJQKA"
SUITS_STR = "shdc"
ALL_CARDS_STR = [f"{r}{s}" for r in RANKS_STR for s in SUITS_STR] + ["X1", "X2"]
RANK_VALUES = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
SUIT_NAMES = {'s':'spades','h':'hearts','d':'diamonds','c':'clubs'}
TOP_MAX, MID_MAX, BOT_MAX = 3, 5, 5
ROW_NAMES = {0:'Top',1:'Middle',2:'Bottom'}

ALL_T0_PLACEMENTS = [p for p in itertools.product([0,1,2], repeat=5) if p.count(0)<=3]
T0_PLACEMENT_TENSOR = torch.tensor(ALL_T0_PLACEMENTS, dtype=torch.long)

_t0_model = None
_t0_device = None

def _load_t0_model():
    global _t0_model, _t0_device
    if _t0_model is not None: return
    model_path = Path(__file__).parent / 'models' / 't0_placement_net_v4.pt'
    _t0_device = torch.device('cpu')
    ckpt = torch.load(model_path, map_location=_t0_device, weights_only=False)
    cfg = ckpt.get('config', {})
    _t0_model = T0PlacementNet(d_model=cfg.get('d_model',128), nhead=4,
        num_layers=cfg.get('num_layers',4), dim_ff=cfg.get('d_model',128)*2,
        dropout=cfg.get('dropout',0.2)).to(_t0_device)
    _t0_model.load_state_dict(ckpt['model_state_dict'])
    _t0_model.eval()

def card_str_to_dict(s):
    if s in ('X1','X2','JK'): return {'rank':'Joker','suit':'joker'}
    return {'rank':s[0],'suit':SUIT_NAMES.get(s[1],s[1])}

def t0_nn_top50(hand_5):
    _load_t0_model()
    dicts = [card_str_to_dict(c) for c in hand_5]
    base = np.stack([encode_card(c) for c in dicts])
    hf = compute_hand_features(dicts)
    feat = torch.from_numpy(np.concatenate([base,hf],axis=1)).unsqueeze(0).to(_t0_device)
    with torch.no_grad():
        logits, _ = _t0_model(feat)
        log_p = F.log_softmax(logits, dim=-1)
    vt = T0_PLACEMENT_TENSOR.to(_t0_device)
    scores = sum(log_p[0:1, i, vt[:, i]] for i in range(5))
    _, top_idx = torch.topk(scores[0], min(5, len(ALL_T0_PLACEMENTS)))
    return [ALL_T0_PLACEMENTS[i] for i in top_idx.tolist()]

# ── Board State ──
class Board:
    __slots__ = ['top','mid','bot']
    def __init__(self, t=None, m=None, b=None):
        self.top=list(t or []); self.mid=list(m or []); self.bot=list(b or [])
    def copy(self): return Board(self.top, self.mid, self.bot)
    def place(self, card, row):
        [self.top, self.mid, self.bot][row].append(card)
    def is_complete(self): return len(self.top)==3 and len(self.mid)==5 and len(self.bot)==5
    def slots(self): return (TOP_MAX-len(self.top), MID_MAX-len(self.mid), BOT_MAX-len(self.bot))
    def all_cards_set(self): return set(self.top+self.mid+self.bot)
    def fmt(self):
        return f"Top[{' '.join(self.top)}] Mid[{' '.join(self.mid)}] Bot[{' '.join(self.bot)}]"

# ── Hand Evaluation ──
_B=15; _B5=_B**5
def _enc(cat,*r):
    v=cat
    for i in range(5): v=v*_B+(r[i] if i<len(r) else 0)
    return v
def hand_cat(v): return v//_B5

def _chk_str(sr, j=0):
    if len(sr)+j<5: return False
    u=sorted(set(sr),reverse=True)
    for h in range(14,4,-1):
        if len(set(range(h,h-5,-1))-set(u))<=j: return True
    if len({14,2,3,4,5}-set(u))<=j: return True
    return False

def _str_hi(sr, j=0):
    if j==0:
        if sr==[14,5,4,3,2]: return 5
        return sr[0]
    u=sorted(set(sr),reverse=True)
    for h in range(14,4,-1):
        nd=set(range(h,h-5,-1))
        if len(nd-set(u))<=j and len(set(u)-nd)==0: return h
    return sr[0] if sr else 0

def eval_hand(cards, n):
    if len(cards)!=n: return 0
    ranks,suits,jk=[],[],0
    for c in cards:
        if c in("X1","X2","JK"): jk+=1
        else: ranks.append(RANK_VALUES.get(c[0],0)); suits.append(c[1])
    rc=Counter(ranks); sc=Counter(suits); sr=sorted(ranks,reverse=True)
    fl=len(sc)==1 and(len(suits)+jk)==n; st=_chk_str(sr,jk)
    if n==3:
        b=max(rc.values()) if rc else 0
        if b+jk>=3:
            r=max((r for r,c in rc.items() if c>=max(2,b)),default=sr[0] if sr else 14)
            return _enc(3,r)
        if b+jk>=2:
            if b>=2: pr=max(r for r,c in rc.items() if c>=2); k=sorted([r for r in ranks if r!=pr],reverse=True)
            else: pr=sr[0]; k=sr[1:]
            return _enc(1,pr,k[0] if k else 0)
        return _enc(0,*sr)
    b=max(rc.values()) if rc else 0
    prs=sorted([r for r,c in rc.items() if c>=2],reverse=True)
    if fl and st: return _enc(8,_str_hi(sr,jk))
    if b+jk>=4:
        if b>=4: qr=max(r for r,c in rc.items() if c>=4)
        elif b>=3: qr=max(r for r,c in rc.items() if c>=3)
        else: qr=prs[0] if prs else(sr[0] if sr else 0)
        return _enc(7,qr,max((r for r in ranks if r!=qr),default=0))
    if b>=3:
        tr=max(r for r,c in rc.items() if c>=3)
        pc=[r for r,c in rc.items() if c>=2 and r!=tr]
        if pc: return _enc(6,tr,max(pc))
    if jk>=1 and len(prs)>=2: return _enc(6,prs[0],prs[1])
    if fl: return _enc(5,*sr[:5])
    if st: return _enc(4,_str_hi(sr,jk))
    if b+jk>=3:
        if b>=3: tr=max(r for r,c in rc.items() if c>=3)
        elif b>=2: tr=max(r for r,c in rc.items() if c>=2)
        else: tr=sr[0] if sr else 0
        k=sorted([r for r in ranks if r!=tr],reverse=True)
        return _enc(3,tr,k[0] if len(k)>0 else 0,k[1] if len(k)>1 else 0)
    if len(prs)>=2:
        return _enc(2,prs[0],prs[1],max((r for r in ranks if r not in prs[:2]),default=0))
    if b>=2 or jk>=1:
        if prs: pr=prs[0]; k=sorted([r for r in ranks if r!=pr],reverse=True)
        else: pr=sr[0] if sr else 0; k=sr[1:]
        return _enc(1,pr,k[0] if len(k)>0 else 0,k[1] if len(k)>1 else 0,k[2] if len(k)>2 else 0)
    return _enc(0,*sr[:5])

# ── Royalties ──
def top_roy(cards):
    ranks,jk=[],0
    for c in cards:
        if c in("X1","X2"): jk+=1
        else: ranks.append(RANK_VALUES.get(c[0],0))
    rc=Counter(ranks); best=0
    for r in sorted(rc.keys(),reverse=True):
        if rc[r]+jk>=3: return 10+(r-2)
        if rc[r]+jk>=2 and r>=6: best=max(best,r-5)
    return best

def mid_roy(cards):
    v=eval_hand(cards,5); c=hand_cat(v); r1=(v//(_B**4))%_B
    if c==8 and r1==14: return 50
    if c==8: return 30
    if c==7: return 20
    if c==6: return 12
    if c==5: return 8
    if c==4: return 4
    if c==3: return 2
    return 0

def bot_roy(cards):
    v=eval_hand(cards,5); c=hand_cat(v); r1=(v//(_B**4))%_B
    if c==8 and r1==14: return 25
    if c==8: return 15
    if c==7: return 10
    if c==6: return 6
    if c==5: return 4
    if c==4: return 2
    return 0

FL_EV={14:15.8,15:22.7,16:28.6,17:35.1}
def chk_fl(top):
    ranks,jk=[],0
    for c in top:
        if c in("X1","X2"): jk+=1
        else: ranks.append(RANK_VALUES.get(c[0],0))
    rc=Counter(ranks)
    for r in sorted(rc.keys(),reverse=True):
        if rc[r]+jk>=3: return True,17
        if rc[r]+jk>=2:
            if r==14: return True,16
            elif r==13: return True,15
            elif r==12: return True,14
    return False,0

def eval_terminal(b):
    tv=eval_hand(b.top,3); mv=eval_hand(b.mid,5); bv=eval_hand(b.bot,5)
    if tv>mv or mv>bv: return -6.0
    t=float(top_roy(b.top)+mid_roy(b.mid)+bot_roy(b.bot))
    ok,fc=chk_fl(b.top)
    if ok: t+=FL_EV.get(fc,0.0)
    return t

# ── Action Generation ──
def gen_actions(hand, board):
    assert len(hand)==3
    lim={0:TOP_MAX,1:MID_MAX,2:BOT_MAX}
    cur={0:len(board.top),1:len(board.mid),2:len(board.bot)}
    acts=[]; seen=set()
    for di in range(3):
        d=hand[di]
        if d in("X1","X2"): continue
        rem=[hand[i] for i in range(3) if i!=di]
        for r0 in range(3):
            for r1 in range(3):
                caps=dict(cur)
                if caps[r0]>=lim[r0]: continue
                caps[r0]+=1
                if caps[r1]>=lim[r1]: continue
                p=sorted([(rem[0],r0),(rem[1],r1)],key=lambda x:(x[1],x[0]))
                key=(d,tuple(p))
                if key in seen: continue
                seen.add(key)
                acts.append({'discard':d,'placements':[(rem[0],r0),(rem[1],r1)]})
    return acts

def fmt_action(a):
    d=a['discard']
    ps=", ".join(f"{c}\u2192{ROW_NAMES[r]}" for c,r in a['placements'])
    return d, ps

def apply_action(board, action):
    b=board.copy()
    for c,r in action['placements']: b.place(c,r)
    return b

# ── Nested Expectimax ──
def expectimax_t4(board, deck_list, n4, rng):
    """T4: sample n4 deals, all actions -> max, terminal eval."""
    if board.is_complete():
        return eval_terminal(board)
    deck=list(deck_list)
    total=0.0
    for _ in range(n4):
        rng.shuffle(deck)
        hand=deck[:3]
        actions=gen_actions(hand, board)
        if not actions:
            total+=eval_terminal(board); continue
        best=float('-inf')
        for a in actions:
            b2=apply_action(board,a)
            v=eval_terminal(b2) if b2.is_complete() else -6.0
            if v>best: best=v
        total+=best
    return total/n4

def expectimax_t3(board, deck_list, n3, n4, rng):
    """T3: sample n3 deals, all actions -> max, then T4."""
    if board.is_complete():
        return eval_terminal(board)
    deck=list(deck_list)
    total=0.0
    for _ in range(n3):
        rng.shuffle(deck)
        hand=deck[:3]
        rest=deck[3:]
        actions=gen_actions(hand, board)
        if not actions:
            total+=eval_terminal(board); continue
        best=float('-inf')
        for a in actions:
            b2=apply_action(board,a)
            v=expectimax_t4(b2, rest, n4, rng)
            if v>best: best=v
        total+=best
    return total/n3

def expectimax_t2(board, deck_list, n2, n3, n4, rng):
    """T2: sample n2 deals, all actions -> max, then T3."""
    if board.is_complete():
        return eval_terminal(board)
    deck=list(deck_list)
    total=0.0
    for _ in range(n2):
        rng.shuffle(deck)
        hand=deck[:3]
        rest=deck[3:]
        actions=gen_actions(hand, board)
        if not actions:
            total+=eval_terminal(board); continue
        best=float('-inf')
        for a in actions:
            b2=apply_action(board,a)
            v=expectimax_t3(b2, rest, n3, n4, rng)
            if v>best: best=v
        total+=best
    return total/n2

# ── Worker ──
def generate_one_hand(args):
    hand_idx, n2, n3, n4, seed = args
    rng = random.Random(seed)
    # T0: deal 5, NN top50
    deck=list(ALL_CARDS_STR); rng.shuffle(deck)
    t0_hand=deck[:5]; remaining=deck[5:]
    top50=t0_nn_top50(t0_hand)
    if not top50: return None
    
    # T1: deal 3
    rng.shuffle(remaining)
    t1_hand=remaining[:3]; deck_after=remaining[3:]
    
    results=[]
    for ci, t0_placement in enumerate(top50):
        board=Board()
        for i,row in enumerate(t0_placement): board.place(t0_hand[i],row)
        
        actions=gen_actions(t1_hand, board)
        if not actions: continue
        
        for a in actions:
            b2=apply_action(board,a)
            d2=[c for c in deck_after if c!=a['discard']]
            ev=expectimax_t2(b2, d2, n2, n3, n4, rng)
            d,p=fmt_action(a)
            t0_fmt = ", ".join(f"{t0_hand[i]}->{ROW_NAMES[t0_placement[i]]}" for i in range(5))
            results.append({
                't0_idx': ci,
                't0_p': t0_fmt,
                'd': d,
                'p': p,
                'ev': round(ev,3)
            })
            
    if not results: return None
    results.sort(key=lambda x:x['ev'],reverse=True)
    return {'hand_idx':hand_idx,'turn':1,
            't0_hand':" ".join(t0_hand),
            't1_hand':" ".join(t1_hand),
            'n_placements':len(results),
            'nesting':[n2,n3,n4],'placements':results}

# ── Main ──
def main():
    ap=argparse.ArgumentParser(description="T1 Expectimax data generator")
    ap.add_argument('--n-hands',type=int,default=50000)
    ap.add_argument('--n2',type=int,default=3,help="T2 deal samples")
    ap.add_argument('--n3',type=int,default=3,help="T3 deal samples")
    ap.add_argument('--n4',type=int,default=3,help="T4 deal samples")
    ap.add_argument('--workers',type=int,default=0)
    ap.add_argument('--output',type=str,default='t1_mc_50k.jsonl')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--batch-size',type=int,default=500)
    args=ap.parse_args()
    nw=args.workers if args.workers>0 else max(1,cpu_count()-2)
    print(f"=== T1 Expectimax Generator ===")
    print(f"  Hands:    {args.n_hands:,}")
    print(f"  Nesting:  T2={args.n2}, T3={args.n3}, T4={args.n4}")
    print(f"  Workers:  {nw}")
    print(f"  Output:   {args.output}")
    print(flush=True)
    items=[(i,args.n2,args.n3,args.n4,args.seed+i*1000) for i in range(args.n_hands)]
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    t0=time.time(); done=0; written=0
    with open(args.output,'w',encoding='utf-8') as f:
        with Pool(nw) as pool:
            for res in pool.imap_unordered(generate_one_hand, items, chunksize=10):
                done+=1
                if res:
                    f.write(json.dumps(res,ensure_ascii=False)+'\n')
                    written+=1
                if done%args.batch_size==0:
                    el=time.time()-t0; rate=done/el
                    eta=(args.n_hands-done)/rate if rate>0 else 0
                    print(f"  [{done:>6}/{args.n_hands}] written={written} "
                          f"rate={rate:.2f}/s ETA={eta:.0f}s ({eta/3600:.1f}h)",flush=True)
                    f.flush()
    el=time.time()-t0
    print(f"\n=== Done === {el:.0f}s ({el/3600:.1f}h) | {written:,} hands | {args.output}")

if __name__=='__main__':
    main()
