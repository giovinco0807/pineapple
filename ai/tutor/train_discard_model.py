"""Predict which unseen cards sit in the opponent's discards.

The HU traces carry every seat's visible board and its actual discards, so
this is plain supervised learning: for each hand and each seat, at the
T4-street boundary the seat shows eleven cards and hides three discards
among the twenty-nine cards its opponent cannot see.  One training row per
hidden card: did this card get discarded?

The consumer is the weighted V4 sweep: `p_c = P(c in opponent dead | board)`
turns the uniform draw expectation into a weighted one, for the opponent's
draws and hero's own alike.  What matters there is calibration -- the
weights multiply values -- so the report bins predictions against realised
frequencies, and the model is kept small enough to re-implement in Rust in
twenty lines (linear in hand-written features, one hidden layer at most).

Features are relations between the candidate card and the visible board
(flush pull, straight pull, rank pairing), not rules: the trace decides
their weights.
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


def parse(card: str) -> tuple[int, int]:
    """(rank 2..14, suit 0..3); jokers are (0, 4)."""
    if card.startswith("X"):
        return 0, 4
    return RANKS.index(card[0]) + 2, SUITS.index(card[1])


FEATURE_NAMES = [
    "rank",              # (rank-2)/12
    "is_joker",
    "same_rank_top",     # pairs a top-row rank (their FL material)
    "same_rank_mid",
    "same_rank_bot",
    "same_suit_mid",     # flush pull toward their middle
    "same_suit_bot",
    "near_rank_mid",     # straight pull: |dr| <= 2 in their middle
    "near_rank_bot",
    "board_suit_total",  # how much of this suit they hold anywhere
    "low_card",          # rank <= 6
    "high_card",         # rank >= 11
]


def features(card: str, board: list[list[str]]) -> list[float]:
    rank, suit = parse(card)
    if rank == 0:
        # A joker's relations are meaningless; the flag carries it.
        return [0.5, 1.0] + [0.0] * (len(FEATURE_NAMES) - 2)
    rows = [[parse(c) for c in row if not c.startswith("X")] for row in board]
    same_rank = [sum(1 for r, _ in row if r == rank) for row in rows]
    same_suit = [sum(1 for _, s in row if s == suit) for row in rows]
    near_rank = [sum(1 for r, _ in row if 0 < abs(r - rank) <= 2) for row in rows]
    return [
        (rank - 2) / 12.0,
        0.0,
        float(same_rank[0]),
        float(same_rank[1]),
        float(same_rank[2]),
        float(same_suit[1]),
        float(same_suit[2]),
        float(near_rank[1]),
        float(near_rank[2]),
        float(sum(same_suit)),
        1.0 if rank <= 6 else 0.0,
        1.0 if rank >= 11 else 0.0,
    ]


def extract(traces: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per (hand, seat, hidden card) at the T4-street boundary."""
    xs: list[list[float]] = []
    ys: list[float] = []
    hands: list[int] = []
    # Collect both seats' street-4 rows per hand, then cross them.
    per_hand: dict[int, dict[str, dict]] = {}
    with traces.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("street") == 4 and "seat" in row:
                per_hand.setdefault(row["hand"], {})[row["seat"]] = row
    deck = [r + s for r in RANKS for s in SUITS]
    for hand, seats in per_hand.items():
        if len(seats) != 2:
            continue
        for shown, observer in (("bb", "btn"), ("btn", "bb")):
            visible = seats[shown]["board"]          # 11 cards
            dead = set(seats[shown]["dead"])          # 3 hidden discards
            observer_seen: set[str] = set(seats[observer]["dead"])
            for row in seats[observer]["board"]:
                observer_seen.update(row)
            shown_cards = {c for row in visible for c in row}
            # The observer's 29-card pool, jokers by count.
            jokers_visible = sum(
                1 for c in list(shown_cards) + list(observer_seen) if c.startswith("X")
            )
            pool = [
                c for c in deck if c not in shown_cards and c not in observer_seen
            ]
            for _ in range(2 - jokers_visible):
                pool.append("X?")
            dead_jokers = sum(1 for c in dead if c.startswith("X"))
            for card in pool:
                xs.append(features(card, visible))
                if card == "X?":
                    ys.append(1.0 if dead_jokers > 0 else 0.0)
                    if dead_jokers > 0:
                        dead_jokers -= 1
                else:
                    ys.append(1.0 if card in dead else 0.0)
                hands.append(hand)
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(hands, dtype=np.int64),
    )


def split_of(hand: int) -> str:
    bucket = int.from_bytes(
        hashlib.sha256(f"discard-model/{hand}".encode()).digest()[:4], "big"
    ) % 100
    return "fit" if bucket < 80 else ("dev" if bucket < 90 else "test")


class Tiny(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.net = (
            nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            if hidden
            else nn.Linear(width, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(hidden: int, data: dict, epochs: int = 30) -> tuple[Tiny, float]:
    torch.manual_seed(20260816)
    model = Tiny(data["fit_x"].shape[1], hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.tensor(data["fit_x"])
    y = torch.tensor(data["fit_y"])
    dev_x = torch.tensor(data["dev_x"])
    dev_y = torch.tensor(data["dev_y"])
    loss_fn = nn.BCEWithLogitsLoss()
    best = float("inf")
    best_state = None
    for _ in range(epochs):
        model.train()
        for start in range(0, len(x), 65536):
            batch = slice(start, start + 65536)
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_loss = loss_fn(model(dev_x), dev_y).item()
        if dev_loss < best:
            best = dev_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    x, y, hands = extract(args.traces)
    splits = np.array([split_of(int(h)) for h in hands])
    data = {}
    for name in ("fit", "dev", "test"):
        mask = splits == name
        data[f"{name}_x"] = x[mask]
        data[f"{name}_y"] = y[mask]
    base = float(data["fit_y"].mean())
    base_loss = -(
        base * np.log(base) + (1 - base) * np.log(1 - base)
    )
    print(f"rows: fit {len(data['fit_y'])}, dev {len(data['dev_y'])}, test {len(data['test_y'])}")
    print(f"base rate {base:.4f}, base log-loss {base_loss:.4f}")

    report = {"base_rate": base, "base_log_loss": float(base_loss), "models": {}}
    winner = None
    for hidden in (0, 16):
        model, dev_loss = train_one(hidden, data)
        label = "linear" if hidden == 0 else f"mlp{hidden}"
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor(data["test_x"]))).numpy()
        t = data["test_y"]
        eps = 1e-7
        test_loss = float(-np.mean(t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps)))
        bins = []
        for low in np.arange(0.0, 0.5, 0.05):
            mask = (p >= low) & (p < low + 0.05)
            if mask.sum() > 100:
                bins.append(
                    {
                        "bin": f"{low:.2f}-{low+0.05:.2f}",
                        "predicted": float(p[mask].mean()),
                        "realised": float(t[mask].mean()),
                        "n": int(mask.sum()),
                    }
                )
        report["models"][label] = {
            "dev_log_loss": dev_loss,
            "test_log_loss": test_loss,
            "calibration": bins,
        }
        print(f"{label}: dev {dev_loss:.4f}  test {test_loss:.4f}")
        if winner is None or dev_loss < winner[1]:
            winner = (label, dev_loss, model)

    label, _, model = winner
    weights = {
        "schema": "ofc_hu_discard_model/v1",
        "kind": label,
        "features": FEATURE_NAMES,
        "state": {k: v.tolist() for k, v in model.state_dict().items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "discard_model.json").write_text(
        json.dumps(weights), encoding="utf-8"
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"winner: {label} -> {args.out_dir / 'discard_model.json'}")


if __name__ == "__main__":
    main()
