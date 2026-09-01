"""Distil a champion street-seat decision into a fast CANDIDATE EVALUATOR.

The first attempt asked one forward pass over the pre-move state to emit a
distribution over 27 actions, and reached 24-32% top-1.  That form makes the
net imagine all 27 resulting boards; the champion never does that -- it
encodes each candidate board and scores it.  So this scores candidates too:
the same trace label (the move the champion chose) trains a listwise softmax
over the 27 candidate boards, and serving takes the argmax.  Same labels,
same microsecond cost class, a far easier function to learn.

Features are per CANDIDATE (the board the move reaches), all lookups:
  own board after the move   3 x 54   162
  opponent board             3 x 54   162
  seen (own+opp+draw+discards)   54
  discarded card                 54
  per-row stats, own and opp: count, jokers, max rank, rank sum,
      best-suit count, pair count, straight-window fill, flush outs
                              2 x 3 x 8 = 48
  room after the move, own       3
  street one-hot                 4
                              = 487

Usage:
    python -m ai.tutor.train_hu_fast_eval --slot t1_bb \
        --material D:/ofc_data/hu/fast/material --out D:/ofc_data/hu/fast/nets
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

RANKS = "23456789TJQKA"
SUITS = "shdc"
CAP = (3, 5, 5)
FEATURES = 487
ACTIONS = 27
WINDOWS = [(14, 2, 3, 4, 5)] + [tuple(range(low, low + 5)) for low in range(2, 11)]


def card_index(name: str) -> int:
    if name.startswith("X"):
        return 52 + (0 if name == "X1" else 1)
    return SUITS.index(name[1]) * 13 + RANKS.index(name[0])


def rank_of(name: str) -> int:
    return 15 if name.startswith("X") else RANKS.index(name[0]) + 2


def row_stats(cards, unseen_rank, unseen_suit) -> list[float]:
    plain = [c for c in cards if not c.startswith("X")]
    ranks = [rank_of(c) for c in plain]
    jokers = len(cards) - len(plain)
    suits = [c[1] for c in plain]
    best_suit_n, best_suit = 0, None
    for s in set(suits):
        if suits.count(s) > best_suit_n:
            best_suit_n, best_suit = suits.count(s), s
    pairs = sum(1 for r in set(ranks) if ranks.count(r) > 1)
    fill = 0.0
    for window in WINDOWS:
        filled = set()
        ok = True
        for r in ranks:
            if r not in window or r in filled:
                ok = False
                break
            filled.add(r)
        if ok:
            fill = max(fill, min((len(filled) + jokers) / 5.0, 1.0))
    flush_outs = (unseen_suit.get(best_suit, 0) / 13.0) if best_suit else 0.0
    return [len(cards) / 5.0, jokers / 2.0,
            (max(ranks) if ranks else 0) / 15.0, sum(ranks) / 70.0,
            best_suit_n / 5.0, pairs / 2.0, fill,
            sum(unseen_rank.get(r, 0) for r in ranks) / 9.0]


def candidates(own, draw, room):
    """(action index, board after, discarded card) for every legal action."""
    order = sorted(draw)
    out = []
    for discard_pos in range(3):
        kept = [c for i, c in enumerate(order) if i != discard_pos]
        for r1 in range(3):
            for r2 in range(3):
                need = [0, 0, 0]
                need[r1] += 1
                need[r2] += 1
                if any(need[i] > room[i] for i in range(3)):
                    continue
                after = [list(r) for r in own]
                after[r1].append(kept[0])
                after[r2].append(kept[1])
                out.append((discard_pos * 9 + r1 * 3 + r2, after, order[discard_pos]))
    return out


def featurise(after, opp, seen, discard, street, room_after,
              unseen_rank, unseen_suit) -> np.ndarray:
    x = np.zeros(FEATURES, np.float32)
    at = 0
    for rows in (after, opp):
        for row in rows:
            for card in row:
                x[at + card_index(card)] = 1.0
            at += 54
    for card in seen:
        x[at + card_index(card)] = 1.0
    at += 54
    x[at + card_index(discard)] = 1.0
    at += 54
    for rows in (after, opp):
        for row in rows:
            x[at:at + 8] = row_stats(row, unseen_rank, unseen_suit)
            at += 8
    x[at:at + 3] = [r / 5.0 for r in room_after]
    at += 3
    x[at + min(street, 3)] = 1.0
    at += 4
    assert at == FEATURES, at
    return x


def sample_matrix(sample: dict, street: int):
    own, opp, draw = sample["own"], sample["opp"], list(sample["draw"])
    room = tuple(CAP[i] - len(own[i]) for i in range(3))
    seen = ([c for r in own for c in r] + [c for r in opp for c in r]
            + draw + list(sample["own_discards"]))
    seen_set = set(seen)
    unseen_rank, unseen_suit = {}, {}
    for suit in SUITS:
        for rank in RANKS:
            name = rank + suit
            if name not in seen_set:
                unseen_rank[rank_of(name)] = unseen_rank.get(rank_of(name), 0) + 1
                unseen_suit[suit] = unseen_suit.get(suit, 0) + 1
    rows, index = [], []
    for action, after, discard in candidates(own, draw, room):
        room_after = tuple(CAP[i] - len(after[i]) for i in range(3))
        rows.append(featurise(after, opp, seen, discard, street, room_after,
                              unseen_rank, unseen_suit))
        index.append(action)
    return np.stack(rows), np.asarray(index, np.int64)


class FastEval(nn.Module):
    def __init__(self, hidden=(256, 128, 64)):
        super().__init__()
        layers, previous = [], FEATURES
        for size in hidden:
            layers += [nn.Linear(previous, size), nn.ReLU()]
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def split_of(index: int, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}/{index}".encode()).digest()
    return "fit" if int.from_bytes(digest[:4], "big") % 100 < 90 else "dev"


def load(path: Path, slot: str, limit: int | None):
    street = int(slot[1])
    blocks, targets, which, widths = [], [], [], []
    for i, line in enumerate(path.open(encoding="utf-8")):
        if not line.strip():
            continue
        if limit and i >= limit:
            break
        sample = json.loads(line)
        rows, index = sample_matrix(sample, street)
        target = int(np.flatnonzero(index == sample["action"])[0])
        blocks.append(rows)
        targets.append(target)
        widths.append(len(index))
        which.append(split_of(i, slot))
    return blocks, np.asarray(targets), np.asarray(widths), np.asarray(which)


def pack(blocks, targets, widths):
    """Ragged candidate sets padded to the widest, with a validity mask.

    Held as float16: the features are one-hots and ratios in [0,1], all of
    which float16 represents exactly enough for a forward pass, and float32
    over 214k decisions x 27 candidates x 487 features is 9.4 GB -- which is
    how the first full run died.  Batches are cast back on the way to the GPU.
    """
    n, w = len(blocks), int(widths.max())
    x = np.zeros((n, w, FEATURES), np.float16)
    mask = np.zeros((n, w), np.bool_)
    for i, block in enumerate(blocks):
        x[i, :len(block)] = block.astype(np.float16)
        mask[i, :len(block)] = True
    return x, mask, targets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slot", required=True)
    ap.add_argument("--material", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blocks, targets, widths, which = load(
        args.material / f"{args.slot}.jsonl", args.slot, args.limit or None)
    x, mask, y = pack(blocks, targets, widths)
    fit, dev = which == "fit", which == "dev"
    print(f"{args.slot}: {len(y)} decisions, fit {fit.sum()} dev {dev.sum()}, "
          f"candidates/decision {widths.mean():.1f}")
    # Feature blocks stay in host RAM as float16 and are cast per batch; the
    # dev split is small enough to sit on the device for the whole run.
    xf, mf = x[fit], mask[fit]
    yf = torch.from_numpy(y[fit]).to(device)
    xd = torch.from_numpy(x[dev]).float().to(device)
    md = torch.from_numpy(mask[dev]).to(device)
    yd = torch.from_numpy(y[dev]).to(device)

    model = FastEval().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    best = {"top1": -1.0, "epoch": -1}
    args.out.mkdir(parents=True, exist_ok=True)

    def scored(xb, mb):
        n, w, _ = xb.shape
        s = model(xb.reshape(n * w, FEATURES)).reshape(n, w)
        return s.masked_fill(~mb, float("-inf"))

    for epoch in range(args.epochs):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(yf))
        for start in range(0, len(order), args.batch):
            idx = np.sort(order[start:start + args.batch])
            xb = torch.from_numpy(xf[idx]).float().to(device)
            mb = torch.from_numpy(mf[idx]).to(device)
            yb = yf[torch.from_numpy(idx).to(device)]
            optim.zero_grad(set_to_none=True)
            loss_fn(scored(xb, mb), yb).backward()
            optim.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            s = scored(xd, md)
            top1 = (s.argmax(1) == yd).float().mean().item()
            top3 = (s.topk(3, dim=1).indices == yd[:, None]).any(1).float().mean().item()
        if top1 > best["top1"]:
            best = {"top1": top1, "top3": top3, "epoch": epoch}
            torch.save({"model_state_dict": model.state_dict(),
                        "features": FEATURES, "slot": args.slot,
                        "schema": "hu_fast_eval/v1"},
                       args.out / f"{args.slot}.pt")
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch}: top1 {top1:.4f} top3 {top3:.4f}", flush=True)
    print(f"{args.slot} best: top1 {best['top1']:.4f} top3 {best['top3']:.4f} "
          f"(epoch {best['epoch']})")
    (args.out / f"{args.slot}.json").write_text(json.dumps(best, indent=2),
                                                encoding="utf-8")


if __name__ == "__main__":
    main()
