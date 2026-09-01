"""Distil one champion street-seat decision into a fast policy net.

The champion needs a 207-dim pair vector per candidate, and two of its blocks
are 200-completion samples -- which is why a rollout costs 130 ms and why a
referee cannot afford the particle counts the regular track uses.  This net
reads the state ONCE and emits a distribution over the street's 27 actions,
so a decision costs one forward pass over ~600 deterministic features.

The label is the champion's move.  Nothing here learns a value: a distilled
value would reorder candidates in a referee, while a distilled move mostly
cancels across them (measured 66x on the regular track).

Features, all lookups or counts, no sampling:
  own board      3 rows x 54          162
  opponent board 3 rows x 54          162
  own draw       3 sorted cards x 54  162
  seen           54 (own board + own discards + opp board + draw)
  cheap          per row, own and opp: count, jokers, max rank, rank sum,
                 best-suit count, pair count                 2 x 3 x 6 = 36
  room           3 own + 3 opp                                 6
  street         one-hot 4                                     4
                                                            = 586

Usage:
    python -m ai.tutor.train_hu_fast --slot t1_bb \
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
FEATURES = 586
ACTIONS = 27


def card_index(name: str) -> int:
    if name.startswith("X"):
        return 52 + (0 if name == "X1" else 1)
    return SUITS.index(name[1]) * 13 + RANKS.index(name[0])


def rank_of(name: str) -> int:
    return 15 if name.startswith("X") else RANKS.index(name[0]) + 2


def row_stats(cards: list[str]) -> list[float]:
    ranks = [rank_of(c) for c in cards if not c.startswith("X")]
    jokers = sum(1 for c in cards if c.startswith("X"))
    suits = [c[1] for c in cards if not c.startswith("X")]
    best_suit = max((suits.count(s) for s in set(suits)), default=0)
    pairs = sum(1 for r in set(ranks) if ranks.count(r) > 1)
    return [len(cards) / 5.0, jokers / 2.0, (max(ranks) if ranks else 0) / 15.0,
            sum(ranks) / 70.0, best_suit / 5.0, pairs / 2.0]


def featurise(sample: dict) -> np.ndarray:
    x = np.zeros(FEATURES, np.float32)
    at = 0
    for rows in (sample["own"], sample["opp"]):
        for row in rows:
            for card in row:
                x[at + card_index(card)] = 1.0
            at += 54
    for card in sorted(sample["draw"]):
        x[at + card_index(card)] = 1.0
        at += 54
    seen = ([c for r in sample["own"] for c in r]
            + [c for r in sample["opp"] for c in r]
            + list(sample["draw"]) + list(sample["own_discards"]))
    for card in seen:
        x[at + card_index(card)] = 1.0
    at += 54
    for rows in (sample["own"], sample["opp"]):
        for row in rows:
            stats = row_stats(row)
            x[at:at + 6] = stats
            at += 6
    for rows in (sample["own"], sample["opp"]):
        for i, row in enumerate(rows):
            x[at] = (CAP[i] - len(row)) / 5.0
            at += 1
    street = int(sample.get("street", 0))
    x[at + min(street, 3)] = 1.0
    at += 4
    assert at == FEATURES, at
    return x


class FastPolicy(nn.Module):
    def __init__(self, hidden=(512, 256, 128)):
        super().__init__()
        layers, previous = [], FEATURES
        for size in hidden:
            layers += [nn.Linear(previous, size), nn.ReLU()]
            previous = size
        layers.append(nn.Linear(previous, ACTIONS))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def split_of(index: int, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}/{index}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "fit" if bucket < 90 else "dev"


def load(path: Path, slot: str):
    street = int(slot[1])
    xs, ys, ms, which = [], [], [], []
    for i, line in enumerate(path.open(encoding="utf-8")):
        if not line.strip():
            continue
        sample = json.loads(line)
        sample["street"] = street
        xs.append(featurise(sample))
        ys.append(sample["action"])
        ms.append(sample["mask"])
        which.append(split_of(i, slot))
    return (np.stack(xs), np.asarray(ys, np.int64),
            np.asarray(ms, np.bool_), np.asarray(which))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slot", required=True)
    ap.add_argument("--material", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y, mask, which = load(args.material / f"{args.slot}.jsonl", args.slot)
    fit, dev = which == "fit", which == "dev"
    print(f"{args.slot}: {len(y)} samples, fit {fit.sum()} dev {dev.sum()}, "
          f"legal actions/sample {mask.sum(1).mean():.1f}")
    xf = torch.from_numpy(x[fit]).to(device)
    yf = torch.from_numpy(y[fit]).to(device)
    xd = torch.from_numpy(x[dev]).to(device)
    yd = torch.from_numpy(y[dev]).to(device)
    md = torch.from_numpy(mask[dev]).to(device)

    model = FastPolicy().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    best = {"top1": -1.0, "epoch": -1}
    args.out.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(yf), device=device)
        for start in range(0, len(order), args.batch):
            idx = order[start:start + args.batch]
            optim.zero_grad(set_to_none=True)
            loss_fn(model(xf[idx]), yf[idx]).backward()
            optim.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(xd)
            logits = logits.masked_fill(~md, float("-inf"))
            top1 = (logits.argmax(1) == yd).float().mean().item()
            top3 = (logits.topk(3, dim=1).indices == yd[:, None]).any(1).float().mean().item()
        if top1 > best["top1"]:
            best = {"top1": top1, "top3": top3, "epoch": epoch}
            torch.save({"model_state_dict": model.state_dict(),
                        "features": FEATURES, "actions": ACTIONS,
                        "slot": args.slot, "schema": "hu_fast_policy/v1"},
                       args.out / f"{args.slot}.pt")
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch}: top1 {top1:.4f} top3 {top3:.4f}", flush=True)
    print(f"{args.slot} best: top1 {best['top1']:.4f} top3 {best['top3']:.4f} "
          f"(epoch {best['epoch']})")
    (args.out / f"{args.slot}.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
