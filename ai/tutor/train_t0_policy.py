"""T0-BB policy net: five dealt cards -> a distribution over all 232 openings.

Two-stage training:
  A (pretrain)  the dense-but-biased corrected-chain labels (t0r1, 23k roots
                x 232 values) provide the broad shape of the space;
  B (correct)   the referee material (deep-replay verdicts, 870 roots,
                ~6 candidates each) pulls the ordering toward full-game truth.

The hand is suit-canonicalised (T0-BB values are exactly invariant under a
global suit permutation: the board is empty and nothing else is visible), so
every training hand teaches 24 permutations at once.  Actions are encoded as
a row assignment per canonical card position: index = sum(row_i * 3^i),
243 slots of which 11 (four-plus cards on top) are masked illegal.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SUITS = "cdhs"
RANKS = "23456789TJQKA"
RV = {c: i for i, c in enumerate(RANKS, 2)}
D = Path("D:/ofc_data/hu")


def canonical(cards):
    """Suit-canonicalise and order a 5-card hand.

    Suits are relabelled by (count desc, rank-multiset desc) so any suit
    permutation of the same hand lands on identical bytes; cards are then
    sorted (rank desc, suit index) with jokers last.  A policy action index
    refers to positions of THIS order.
    """
    plain = [c for c in cards if not c.startswith("X")]
    jokers = sorted(c for c in cards if c.startswith("X"))
    by_suit = {}
    for c in plain:
        by_suit.setdefault(c[1], []).append(RV[c[0]])
    order = sorted(by_suit, key=lambda s: (len(by_suit[s]), sorted(by_suit[s])), reverse=True)
    relabel = {s: SUITS[i] for i, s in enumerate(order)}
    out = sorted((c[0] + relabel[c[1]] for c in plain),
                 key=lambda c: (-RV[c[0]], SUITS.index(c[1])))
    out += [f"X{i+1}" for i in range(len(jokers))]
    mapping = {}
    for orig in plain:
        mapping[orig] = orig[0] + relabel[orig[1]]
    for i, j in enumerate(jokers):
        mapping[j] = f"X{i+1}"
    return out, mapping


def feats(canon):
    x = np.zeros(54, np.float32)
    for c in canon:
        if c.startswith("X"):
            x[52 + int(c[1]) - 1] = 1.0
        else:
            x[(RV[c[0]] - 2) * 4 + SUITS.index(c[1])] = 1.0
    return x


def action_index(key, canon, mapping):
    parts = key.split("|")
    row_of = {}
    for row, part in enumerate(parts[:3]):
        for c in part.split(","):
            if c:
                row_of[mapping[c]] = row
    if len(row_of) != 5:
        return None
    idx = 0
    for i, c in enumerate(canon):
        idx += row_of[c] * (3 ** i)
    return idx


def legal_mask():
    m = np.zeros(243, bool)
    for idx in range(243):
        rows = [(idx // (3 ** i)) % 3 for i in range(5)]
        if rows.count(0) <= 3:
            m[idx] = True
    return m


MASK = legal_mask()


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(54, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 243))

    def forward(self, x):
        out = self.net(x)
        return out.masked_fill(~torch.as_tensor(MASK, device=out.device), -1e9)


def load_stage_a(tau):
    # Lap1 and lap2 root ids collide (each lap counts from its own base), so
    # every label file joins only its own lap's request file.
    pairs = (("t0r1_labels/t0_bb_l1.jsonl", "onpol_requests/t0_bb_onpol.jsonl"),
             ("t0r1_labels/t0_bb_l2.jsonl", "onpol2_requests/t0_bb_lap2.jsonl"))
    xs, ts = [], []
    for f, reqf in pairs:
        reqs = {}
        for l in open(D / reqf, encoding="utf-8"):
            r = json.loads(l)
            reqs[r["id"]] = r["draw"]
        for l in open(D / f, encoding="utf-8"):
            row = json.loads(l)
            draw = reqs.get(row["id"])
            if draw is None:
                continue
            canon, mp = canonical(draw)
            vals = np.full(243, -np.inf, np.float32)
            ok = True
            for a in row["actions"]:
                idx = action_index(a["action_key"], canon, mp)
                if idx is None:
                    ok = False
                    break
                vals[idx] = max(vals[idx], a["value"])
            if not ok:
                continue
            t = np.zeros(243, np.float32)
            fin = np.isfinite(vals)
            e = np.exp((vals[fin] - vals[fin].max()) / tau)
            t[fin] = e / e.sum()
            xs.append(feats(canon))
            ts.append(t)
    return np.stack(xs), np.stack(ts)


def dense_targets(tau):
    """Per-root dense soft targets from the t0r1 labels, keyed by root id.

    Stage B anchors on these: correcting six candidates while leaving the
    other 226 logits unconstrained let hallucinated argmaxes float to the
    top (v1: novel picks confirmed at -14 vs serving).  The dense term pins
    the whole landscape while the referee term reorders the top.
    """
    pairs = (("t0r1_labels/t0_bb_l1.jsonl", "onpol_requests/t0_bb_onpol.jsonl"),
             ("t0r1_labels/t0_bb_l2.jsonl", "onpol2_requests/t0_bb_lap2.jsonl"))
    out = {}
    for f, reqf in pairs:
        reqs = {}
        for l in open(D / reqf, encoding="utf-8"):
            r = json.loads(l)
            reqs[r["id"]] = r["draw"]
        for l in open(D / f, encoding="utf-8"):
            row = json.loads(l)
            draw = reqs.get(row["id"])
            if draw is None:
                continue
            canon, mp = canonical(draw)
            vals = np.full(243, -np.inf, np.float32)
            ok = True
            for a in row["actions"]:
                idx = action_index(a["action_key"], canon, mp)
                if idx is None:
                    ok = False
                    break
                vals[idx] = max(vals[idx], a["value"])
            if not ok:
                continue
            t = np.zeros(243, np.float32)
            fin = np.isfinite(vals)
            e = np.exp((vals[fin] - vals[fin].max()) / tau)
            t[fin] = e / e.sum()
            out[(f, row["id"])] = t
    by_id = {}
    for (f, rid), t in out.items():
        by_id.setdefault(rid, t)
    return by_id


def load_stage_b():
    rows = [json.loads(l) for l in open(D / "t0_mine1/material.jsonl", encoding="utf-8")]
    out = []
    for r in rows:
        canon, mp = canonical(r["cards"].split(","))
        idxs, means = [], []
        skip = False
        for k, v in r["sel_means"].items():
            idx = action_index(k, canon, mp)
            if idx is None:
                skip = True
                break
            idxs.append(idx)
            means.append(v)
        if skip:
            continue
        # Depth weighting (owner's directive): rows measured with more
        # particles carry more training weight -- escalated re-audits (2,400)
        # over scored pairs (600) over select-only agreements (192).
        depth_w = 3.0 if r.get("escalated") else (2.0 if "ci" in r else 1.0)
        out.append(dict(id=r["id"], x=feats(canon), idxs=np.array(idxs),
                        means=np.array(means, np.float32),
                        verdict=r["verdict"], margin=r.get("margin", 0.0),
                        weight=depth_w,
                        serving=action_index(r["model_pick"], canon, mp),
                        ref=action_index(r["ref_pick"], canon, mp)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-a", type=float, default=2.0)
    ap.add_argument("--tau-b", type=float, default=1.5)
    ap.add_argument("--epochs-a", type=int, default=30)
    ap.add_argument("--epochs-b", type=int, default=200)
    ap.add_argument("--holdout", type=int, default=100)
    ap.add_argument("--no-depth-weights", action="store_true")
    ap.add_argument("--no-margin-hinge", action="store_true")
    ap.add_argument("--torch-seed", type=int, default=20260829)
    ap.add_argument("--out", type=Path, default=D / "t0_policy_v2")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Weight init and batch order were unseeded through v5b: identical recipes
    # landed anywhere in 21-52% holdout recovery.  Recipe comparisons need
    # this pinned (and multiple --torch-seed values to average over).
    torch.manual_seed(args.torch_seed)
    rng = random.Random(20260828)

    print("loading stage A ...", flush=True)
    xa, ta = load_stage_a(args.tau_a)
    print(f"A: {len(xa)} roots", flush=True)
    mat = load_stage_b()
    rng.shuffle(mat)
    hold, fit_b = mat[:args.holdout], mat[args.holdout:]
    # Normalize depth weights to mean 1 over the fit set: the directive is
    # RELATIVE authority (escalated > scored > select-only), not a global
    # learning-rate inflation -- unnormalized x2/x3 weights on half the rows
    # doubled the effective LR and collapsed v5 (recovery 51.9% -> 26.7%).
    if args.no_depth_weights:
        for h in fit_b:
            h["weight"] = 1.0
    wm = sum(h["weight"] for h in fit_b) / len(fit_b)
    for h in fit_b:
        h["weight"] /= wm
    print(f"B: {len(fit_b)} fit + {len(hold)} holdout (weight mean {wm:.2f} normalized)", flush=True)

    net = Policy().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    XA = torch.tensor(xa, device=dev)
    TA = torch.tensor(ta, device=dev)
    for ep in range(args.epochs_a):
        perm = torch.randperm(len(XA), device=dev)
        tot = 0.0
        for s in range(0, len(XA), 4096):
            b = perm[s:s + 4096]
            loss = -(TA[b] * torch.log_softmax(net(XA[b]), -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep % 10 == 9:
            print(f"A ep{ep+1} loss {tot/len(XA):.4f}", flush=True)

    def eval_hold():
        agree_ref = agree_serve = in_cands = 0
        pick_val = serve_val = best_val = 0.0
        with torch.no_grad():
            for h in hold:
                # Rows whose serving pick fell out of the referee's recorded
                # ordering (the mine4 halving truncation) can't anchor the
                # recovery denominator; score everything else on them.
                if h["serving"] not in h["idxs"].tolist():
                    continue
                logits = net(torch.tensor(h["x"], device=dev).unsqueeze(0))[0]
                sub = logits[torch.tensor(h["idxs"], device=dev)]
                pick_i = int(sub.argmax())
                g = int(logits.argmax())
                in_cands += g in set(h["idxs"].tolist())
                agree_ref += h["idxs"][pick_i] == h["ref"]
                agree_serve += h["idxs"][pick_i] == h["serving"]
                pick_val += h["means"][pick_i]
                serve_val += h["means"][h["idxs"].tolist().index(h["serving"])]
                best_val += h["means"].max()
        n = sum(1 for h in hold if h["serving"] in h["idxs"].tolist())
        print(f"  holdout: 審判一致 {agree_ref/n:.1%}  配信一致 {agree_serve/n:.1%}  "
              f"global-argmaxが候補内 {in_cands/n:.1%}", flush=True)
        print(f"  候補内価値: policy {pick_val/n:+.3f}  serving {serve_val/n:+.3f}  "
              f"best {best_val/n:+.3f}  -> 回収率 {(pick_val-serve_val)/(best_val-serve_val+1e-9):.1%}",
              flush=True)

    print("after stage A:", flush=True)
    eval_hold()

    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    XB = torch.tensor(np.stack([h["x"] for h in fit_b]), device=dev)
    dense = dense_targets(args.tau_a)
    TB = torch.tensor(np.stack([dense[h["id"]] for h in fit_b]), device=dev)
    replay = torch.randperm(len(XA))[:len(XA) // 10]
    for ep in range(args.epochs_b):
        order = list(range(len(fit_b)))
        rng.shuffle(order)
        tot = 0.0
        opt.zero_grad()
        for step, i in enumerate(order):
            h = fit_b[i]
            logits = net(XB[i].unsqueeze(0))[0]
            # anchor: the full 232-action landscape from the dense labels
            loss = -(TB[i] * torch.log_softmax(logits, -1)).sum()
            # correction: the referee's ordering over its candidates
            idxs = torch.tensor(h["idxs"], device=dev)
            sub = logits[idxs]
            m = torch.tensor(h["means"], device=dev)
            target = torch.softmax(m / args.tau_b, -1)
            loss = loss + 2.0 * -(target * torch.log_softmax(sub, -1)).sum()
            if h["verdict"] == "error":
                li = logits[h["ref"]] - logits[h["serving"]]
                # margin-scaled hinge: a +34 double-joker disaster pushes
                # harder than a +1.6 near-miss (capped at 3x)
                hinge_w = 1.0 if args.no_margin_hinge else min(1.0 + h["margin"] / 4.0, 3.0)
                loss = loss + hinge_w * torch.relu(1.0 - li)
            # depth weight: deeper-audited rows teach with more authority
            loss = loss * h.get("weight", 1.0)
            tot += loss.item()
            loss.backward()
            if step % 64 == 63:
                opt.step()
                opt.zero_grad()
        b = replay[torch.randperm(len(replay))[:2048]].to(dev)
        loss = -(TA[b] * torch.log_softmax(net(XA[b]), -1)).sum(-1).mean() * 3.0
        loss.backward()
        opt.step()
        opt.zero_grad()
        if ep % 50 == 49:
            print(f"B ep{ep+1} loss {tot/len(fit_b):.4f}", flush=True)
            eval_hold()

    args.out.mkdir(exist_ok=True)
    torch.save(dict(model_state_dict=net.state_dict()), args.out / "policy_best.pt")
    print("saved ->", args.out / "policy_best.pt", flush=True)
    print("final:", flush=True)
    eval_hold()


if __name__ == "__main__":
    main()
