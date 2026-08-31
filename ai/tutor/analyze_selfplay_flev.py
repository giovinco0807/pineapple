"""Fantasyland EV from mirror self-play, decomposed by opponent state.

The owner's instruction (2026-08-15): the measure is each hand's realized
settlement, and the opponent matters -- Fantasyland's value is what it moves
across the table, which depends on what the other player was doing.  So
nothing here is pooled blindly:

    fl_vs_normal[n]   the value: my FL hand of width n against a normal hand
    fl_vs_fl[n][m]    the convolution; symmetric widths should sit near zero
    normal_vs_normal  the baseline; a mirror match should sit near zero

The entry value is **episodic**: the owner's definition is the hands from the
next one onward, so each entry is followed through its consecutive
Fantasyland hands (a stay re-enters at the same width -- frozen rule) and the
chain's settlements are summed.  No baseline is subtracted.  The displaced
normal hands carry their own entry options, and in a mirror match the two
cancel exactly --

    E[settle | normal] + sum_m entry_rate(m) * FL_EV(m) = 0

-- which doubles as a consistency check this report prints.  Subtracting the
normal-hand mean (this file's first version did) double-counts those options.
The geometric form E[settle | FL_n] / (1 - stay_n) is reported alongside; it
should agree when stays are memoryless, and disagreement is worth seeing.

Everything is reported twice: `raw` (unclamped points) and `paid` (what
actually crossed the table under the 200-point stacks and the zero floor).
The owner picked `paid` as the label's measure; `raw` rides along because the
difference is the stack effect and seeing it costs nothing.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def mean_se(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "se": None}
    mean = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mean, "se": None}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return {"n": n, "mean": mean, "se": math.sqrt(var / n)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    # Per-player-hand samples.  Each hand line yields two samples, one from
    # each player's perspective; `paid` flips sign exactly (zero-sum).
    fl_vs_normal: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"raw": [], "paid": []})
    fl_vs_fl: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: {"raw": [], "paid": []})
    normal_vs_normal: dict[str, list[float]] = {"raw": [], "paid": []}
    normal_vs_fl: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"raw": [], "paid": []})
    me_fl: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"raw": [], "paid": []})
    me_normal: dict[str, list[float]] = {"raw": [], "paid": []}
    stays: dict[int, list[int]] = defaultdict(list)
    entries: dict[int, int] = defaultdict(int)
    normal_hands = 0
    fouls = 0
    player_hands = 0
    sessions = defaultdict(int)
    hands = 0

    def width_of(state: str) -> int:
        return int(state[2:]) if state.startswith("fl") else 0

    # Entry episodes: consecutive FL hands per (worker, player), summed.  A
    # session reset always returns both players to normal, so a chain never
    # spans sessions; a zero-ended session truncates the chain, which is the
    # game truncating it and belongs in the number.
    open_chain: dict[tuple[int, str], dict] = {}
    chains: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"raw": [], "paid": [], "length": []})

    def close_chain(key: tuple[int, str]) -> None:
        chain = open_chain.pop(key, None)
        if chain is not None:
            chains[chain["width"]]["raw"].append(chain["raw"])
            chains[chain["width"]]["paid"].append(chain["paid"])
            chains[chain["width"]]["length"].append(chain["length"])

    with args.hands.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            hand = json.loads(line)
            hands += 1
            if hand["end"]:
                sessions[hand["end"]] += 1
            for me, other, sign_ in (("a", "b", 1.0), ("b", "a", -1.0)):
                raw = sign_ * hand["settle_raw"]
                paid = sign_ * hand["settle_paid"]
                my_width = width_of(hand[f"state_{me}"])
                their_width = width_of(hand[f"state_{other}"])
                player_hands += 1
                fouls += hand[f"foul_{me}"]
                key = (hand["worker"], me)
                if my_width:
                    chain = open_chain.setdefault(
                        key, {"width": my_width, "raw": 0.0, "paid": 0.0, "length": 0}
                    )
                    chain["raw"] += raw
                    chain["paid"] += paid
                    chain["length"] += 1
                    if not hand[f"stay_{me}"] or hand["end"]:
                        close_chain(key)
                else:
                    # Defensive: a normal hand closes any chain bookkeeping
                    # left open (it should already be closed).
                    close_chain(key)
                if my_width:
                    me_fl[my_width]["raw"].append(raw)
                    me_fl[my_width]["paid"].append(paid)
                    stays[my_width].append(1 if hand[f"stay_{me}"] else 0)
                    if their_width:
                        fl_vs_fl[(my_width, their_width)]["raw"].append(raw)
                        fl_vs_fl[(my_width, their_width)]["paid"].append(paid)
                    else:
                        fl_vs_normal[my_width]["raw"].append(raw)
                        fl_vs_normal[my_width]["paid"].append(paid)
                else:
                    normal_hands += 1
                    me_normal["raw"].append(raw)
                    me_normal["paid"].append(paid)
                    if hand[f"entry_{me}"] >= 14:
                        entries[hand[f"entry_{me}"]] += 1
                    if their_width:
                        normal_vs_fl[their_width]["raw"].append(raw)
                        normal_vs_fl[their_width]["paid"].append(paid)
                    else:
                        normal_vs_normal["raw"].append(raw)
                        normal_vs_normal["paid"].append(paid)

    report = {
        "schema": "ofc_selfplay_flev/v1",
        "hands": hands,
        "sessions_ended": dict(sessions),
        "foul_rate_per_player_hand": fouls / max(player_hands, 1),
        "entry_rate_per_normal_hand": {
            str(width): entries[width] / max(normal_hands, 1) for width in sorted(entries)
        },
        "stay_rate": {
            str(width): sum(values) / len(values) for width, values in sorted(stays.items())
        },
        "normal_vs_normal": {kind: mean_se(v) for kind, v in normal_vs_normal.items()},
        "fl_vs_normal": {
            str(width): {kind: mean_se(v) for kind, v in bucket.items()}
            for width, bucket in sorted(fl_vs_normal.items())
        },
        "normal_vs_fl": {
            str(width): {kind: mean_se(v) for kind, v in bucket.items()}
            for width, bucket in sorted(normal_vs_fl.items())
        },
        "fl_vs_fl": {
            f"{a}v{b}": {kind: mean_se(v) for kind, v in bucket.items()}
            for (a, b), bucket in sorted(fl_vs_fl.items())
        },
    }

    # Flush chains left open by the end of each worker's stream.
    for key in list(open_chain):
        close_chain(key)

    # The entry value: mean total settlement of an entry's whole chain.
    table = {}
    for width in sorted(chains):
        stay = sum(stays[width]) / max(len(stays[width]), 1)
        row = {
            "stay_rate": stay,
            "chains": len(chains[width]["length"]),
            "mean_chain_length": sum(chains[width]["length"])
            / max(len(chains[width]["length"]), 1),
        }
        for kind in ("raw", "paid"):
            episodic = mean_se(chains[width][kind])
            per_hand = mean_se(me_fl[width][kind])
            row[kind] = {
                "fl_ev": episodic["mean"],
                "se": episodic["se"],
                "geometric_check": (
                    per_hand["mean"] / (1.0 - stay)
                    if per_hand["mean"] is not None and stay < 1.0
                    else None
                ),
            }
        table[str(width)] = row
    report["fl_ev_table"] = table

    # Mirror-equilibrium identity: a normal hand's mean settlement should
    # offset its entry options.  Printed, not asserted -- a gap is signal.
    base = mean_se(me_normal["paid"])
    option = sum(
        (entries[width] / max(normal_hands, 1))
        * (table[str(width)]["paid"]["fl_ev"] or 0.0)
        for width in sorted(chains)
    )
    report["consistency_identity"] = {
        "normal_hand_mean_paid": base["mean"],
        "minus_entry_option_value": -option,
        "note": "these two should roughly agree in a mirror match",
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
