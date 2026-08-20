"""Openings whose T0 first-seat answer has been measured, answered instantly.

Solving one opening costs about thirteen minutes of local compute for a single
1,024-particle batch over thirty-two candidates, and rather more to reach a
resolution that actually decides the ranking.  Twenty-four have been paid for.
This looks the answer up instead of paying again.

## Suit isomorphism, and why the table is bigger than it looks

At T0 acting first both boards are empty and nothing has been discarded, so the
position is the five cards and nothing else.  Permuting the suits therefore
maps a position to an equally-valued position, and one solved opening answers
every one of its suit-permutations: up to twenty-four raw deals per entry, and
2,598,960 raw openings collapse to 134,459 distinct ones.

So entries are stored in canonical suiting -- the lexicographically smallest of
the twenty-four relabellings -- and a query is canonicalised the same way. The
stored placement comes back through the *inverse* of the query's relabelling, so
the answer names the querent's own cards.  `As Ks Qs 7s 3s` and `Ah Kh Qh 7h 3h`
are one entry; the second is told about hearts.

## Every entry carries how well it is known

The measurements behind this table disagree with each other by more than their
own error bars once claimed, which is the reason the `status` field exists
rather than a bare best action:

  settled     the gap from first place to second excludes zero at 95%
  unresolved  it does not -- first place is the best guess and nothing more

The interval uses the batch-to-batch scatter measured on 2026-08-20 (SD 0.46 on
a top-two gap at 1,024 particles, halving as particles quadruple) rather than a
per-hand estimate, because most entries rest on one or two batches and a
standard deviation from two samples is worth less than a measured constant.
Batches are pooled by particle count: a 4,096-particle batch counts four times a
1,024-particle one, in the mean and in the variance alike.

**Most entries are `unresolved`, and that is the honest state of them.**  A
single 1,024-particle batch separates two actions only when they are about 0.9
apart, and on the twenty-four openings solved so far only nine are.  An
`unresolved` row is still the best action known; it is not a ranking anyone
should defend.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

TABLE_SCHEMA = "hu_t0_solved_openings_v1"
DEFAULT_TABLE = pathlib.Path("D:/ofc_data/t0_rung/t0_solved_openings.json")

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_VALUE = {rank: index + 2 for index, rank in enumerate(RANKS)}

# Top-two gap scatter of a single batch, measured on `Ah Kc Qs 8h 7c` over
# twelve 4,096-particle batches: SD 0.231 there, so 0.462 at 1,024. See
# docs/t0_particle_noise_20260820.md.
GAP_SD_AT_1024 = 0.462
REFERENCE_SAMPLES = 1024


def _card(card: str) -> tuple[int, str]:
    return RANK_VALUE[card[0]], card[1]


def canonical_form(hand: Sequence[str]) -> tuple[tuple[str, ...], dict[str, str]]:
    """The hand's canonical suiting, and the relabelling that produced it.

    Returns the smallest of the twenty-four relabellings under a fixed card
    order, together with the map from the hand's own suits to canonical ones.
    Ties between relabellings are broken by the map itself, so the choice is
    deterministic for hands whose suit pattern has a symmetry -- a pair of
    identically-shaped suits must not canonicalise two ways on two calls.
    """

    best: tuple[tuple[str, ...], dict[str, str]] | None = None
    for permuted in itertools.permutations(SUITS):
        mapping = dict(zip(SUITS, permuted))
        relabelled = sorted(
            (f"{card[0]}{mapping[card[1]]}" for card in hand),
            key=lambda c: (-RANK_VALUE[c[0]], SUITS.index(c[1])),
        )
        candidate = (tuple(relabelled), mapping)
        if best is None or (candidate[0], sorted(mapping.items())) < (
            best[0], sorted(best[1].items())
        ):
            best = candidate
    assert best is not None
    return best


def _apply(mapping: Mapping[str, str], card: str) -> str:
    return f"{card[0]}{mapping[card[1]]}"


def _invert(mapping: Mapping[str, str]) -> dict[str, str]:
    return {value: key for key, value in mapping.items()}


def pooled(batches: Iterable[tuple[int, float]]) -> tuple[float, int]:
    """Particle-weighted mean of one action's scores, and the particles behind it.

    A batch's variance goes as one over its particle count, so weighting by
    particles is the inverse-variance weighting and the total is what the
    interval is computed from.
    """

    rows = list(batches)
    total = sum(samples for samples, _ in rows)
    return sum(samples * value for samples, value in rows) / total, total


def _interval(gap: float, particles: int) -> tuple[float, float, str]:
    sem = GAP_SD_AT_1024 / (particles / REFERENCE_SAMPLES) ** 0.5
    low, high = gap - 1.96 * sem, gap + 1.96 * sem
    return low, high, ("settled" if low > 0 else "unresolved")


# --------------------------------------------------------------------------
# Building


def _row_labels(placements: Sequence[Sequence[str]],
                mapping: Mapping[str, str] | None = None) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {"top": [], "middle": [], "bottom": []}
    for card, row in placements:
        rows[row].append(_apply(mapping, card) if mapping else card)
    for row in rows.values():
        row.sort(key=lambda c: (-RANK_VALUE[c[0]], SUITS.index(c[1])))
    return rows


def build(sources: Sequence[pathlib.Path], deep: Sequence[pathlib.Path],
          extra: Sequence[pathlib.Path]) -> dict[str, Any]:
    """Fold every measurement of every opening into one entry apiece."""

    # hand -> action_key -> [(samples, score)], plus the placement for labelling.
    scores: dict[tuple[str, ...], dict[str, list[tuple[int, float]]]] = {}
    places: dict[tuple[str, ...], dict[str, list[list[str]]]] = {}
    hands: dict[tuple[str, ...], list[str]] = {}

    for path in sources:
        solved = json.loads(path.read_text(encoding="utf-8"))
        hand = tuple(solved["hand"])
        hands[hand] = list(solved["hand"])
        for action in solved["kept"]:
            scores.setdefault(hand, {}).setdefault(action["key"], []).append(
                (int(solved.get("samples", 1024)), float(action["score"]))
            )
            places.setdefault(hand, {})[action["key"]] = action["placements"]

    # Extra 1,024-particle batches from the ladder, and deep 4,096 batches.
    for path in extra:
        record = json.loads(path.read_text(encoding="utf-8"))
        hand = tuple(record["observation"]["dealt_cards"])
        if hand not in scores:
            continue
        for key, value in record["runs"][0]["scores"].items():
            if key in scores[hand]:
                scores[hand][key].append((int(record["samples"]), float(value)))

    for path in deep:
        payload = json.loads(path.read_text(encoding="utf-8"))
        hand = tuple(payload["hand"])
        if hand not in scores:
            continue
        samples = int(payload.get("samples", 4096))
        runs = ([run for arm in payload["arms"].values() for run in arm]
                if "arms" in payload else [payload])
        for run in runs:
            for key, value in run["scores"].items():
                if key in scores[hand]:
                    scores[hand][key].append((samples, float(value)))

    entries = {}
    for hand, actions in sorted(scores.items()):
        canon, mapping = canonical_form(hands[hand])
        summarised = []
        for key, batches in actions.items():
            mean, particles = pooled(batches)
            summarised.append({
                "action_key": key, "ev": round(mean, 4),
                "particles": particles, "batches": len(batches),
                "rows": _row_labels(places[hand][key], mapping),
            })
        summarised.sort(key=lambda a: -a["ev"])
        best, second = summarised[0], summarised[1]
        gap = best["ev"] - second["ev"]
        # The gap is only as well measured as its worse-measured half.
        particles = min(best["particles"], second["particles"])
        low, high, status = _interval(gap, particles)

        entries[" ".join(canon)] = {
            "canonical_hand": list(canon),
            "measured_as": hands[hand],
            "best": best["rows"],
            "best_ev": best["ev"],
            "gap_to_second": round(gap, 4),
            "gap_ci95": [round(low, 4), round(high, 4)],
            "status": status,
            "particles_behind_gap": particles,
            "batches_behind_best": best["batches"],
            "runner_up": second["rows"],
            "candidates": summarised[:10],
        }

    settled = sum(1 for e in entries.values() if e["status"] == "settled")
    return {
        "schema": TABLE_SCHEMA,
        "entries": len(entries),
        "settled": settled,
        "unresolved": len(entries) - settled,
        "gap_sd_at_1024": GAP_SD_AT_1024,
        "note": (
            "Keyed by canonical suiting; a query is canonicalised and the "
            "stored placement returned through the inverse relabelling, so one "
            "entry answers up to twenty-four raw deals. `unresolved` means the "
            "first-to-second gap does not exclude zero at 95% -- the action is "
            "the best known, not an established ranking."
        ),
        "openings": entries,
    }


# --------------------------------------------------------------------------
# Lookup


def lookup(hand: Sequence[str], table: Mapping[str, Any]) -> dict[str, Any] | None:
    """The stored answer for this opening, named in the querent's own suits."""

    canon, mapping = canonical_form(hand)
    entry = table["openings"].get(" ".join(canon))
    if entry is None:
        return None
    back = _invert(mapping)
    return {
        "hand": list(hand),
        "canonical_hand": entry["canonical_hand"],
        "best": {row: [_apply(back, card) for card in cards]
                 for row, cards in entry["best"].items()},
        "runner_up": {row: [_apply(back, card) for card in cards]
                      for row, cards in entry["runner_up"].items()},
        "best_ev": entry["best_ev"],
        "gap_to_second": entry["gap_to_second"],
        "gap_ci95": entry["gap_ci95"],
        "status": entry["status"],
        "particles_behind_gap": entry["particles_behind_gap"],
        "suits_relabelled": mapping != {s: s for s in SUITS},
    }


def _render(answer: Mapping[str, Any]) -> str:
    lines = [f"{' '.join(answer['hand'])}"]
    if answer["suits_relabelled"]:
        lines.append(f"  (solved as {' '.join(answer['canonical_hand'])}, "
                     "suit-isomorphic)")
    for name in ("best", "runner_up"):
        rows = answer[name]
        body = "  ".join(
            f"{label}: {' '.join(rows[key]) or '-'}"
            for label, key in (("top", "top"), ("mid", "middle"), ("bot", "bottom")))
        lines.append(f"  {name:9s} {body}")
    low, high = answer["gap_ci95"]
    lines.append(f"  EV {answer['best_ev']:+.3f}   gap to second "
                 f"{answer['gap_to_second']:+.3f}  95% CI [{low:+.3f}, {high:+.3f}]")
    lines.append(f"  status {answer['status']}  "
                 f"({answer['particles_behind_gap']:,} particles behind the gap)")
    if answer["status"] == "unresolved":
        lines.append("  -- the gap does not exclude zero: best known, not decided")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=str(DEFAULT_TABLE))
    parser.add_argument("--hand", default="", help="five cards, e.g. 'Ah Kc Qs 8h 7c'")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    table_path = pathlib.Path(args.table)
    if args.build:
        root = pathlib.Path("D:/ofc_data/t0_rung")
        table = build(
            sources=sorted((root / "akq_local").glob("hand_*.json"))
                    + sorted((root / "special").glob("special_*.json")),
            deep=([p for p in ((root / "local4096" / "seed_independence.json"),)
                   if p.is_file()]
                  + sorted((root / "local4096" / "hand0_settled").glob("b*.json"))),
            extra=(sorted((root / "ladder20_local").glob("h*_b*.json"))
                   + sorted((root / "ladder20_elim").glob("h*_e*.json"))),
        )
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(json.dumps(table, indent=1), encoding="utf-8")
        print(f"{table_path}: {table['entries']} openings, "
              f"{table['settled']} settled, {table['unresolved']} unresolved")
        return 0

    table = json.loads(table_path.read_text(encoding="utf-8"))
    if args.list:
        print(f"{table['entries']} openings "
              f"({table['settled']} settled, {table['unresolved']} unresolved)\n")
        rows = sorted(table["openings"].values(),
                      key=lambda e: -e["gap_to_second"])
        print(f"  {'canonical opening':20s} {'gap':>7} {'particles':>10}  status")
        for entry in rows:
            print(f"  {' '.join(entry['canonical_hand']):20s} "
                  f"{entry['gap_to_second']:7.3f} "
                  f"{entry['particles_behind_gap']:10,}  {entry['status']}")
        return 0

    if not args.hand:
        parser.error("pass --hand, --list or --build")
    cards = args.hand.replace(",", " ").split()
    if len(cards) != 5:
        parser.error(f"a T0 opening is five cards, got {len(cards)}")
    answer = lookup(cards, table)
    if answer is None:
        canon, _ = canonical_form(cards)
        print(f"{' '.join(cards)}: not solved "
              f"(canonical {' '.join(canon)})")
        return 1
    print(_render(answer))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
