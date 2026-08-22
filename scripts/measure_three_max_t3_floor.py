"""Measure the T3-BTN teacher's own noise floor.

    python scripts/measure_three_max_t3_floor.py --corpus corpus.jsonl --roots 300

A model's regret is meaningless on its own.  The teacher itself is a sampled
estimate, so relabelling the same root from a disjoint draw stream picks a
different action some of the time -- and the value that second labelling gives
up against the first is regret the model can never beat.  Heads-up calls this
the referee floor and reports ``gap = model regret - floor``; a model at the
floor is done, and a "bad" regret above a high floor is a labelling problem,
not a modelling one.

Both directions are measured and averaged, because the two labellings have
equal standing and a one-directional comparison would silently anoint the
first one as truth.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ofc_regular.state import Board  # noqa: E402
from ofc_regular.three_max.exact import evaluate_t3  # noqa: E402
from ofc_regular.three_max.world import ThreeMaxObservation  # noqa: E402

SEED_OFFSET = 777_000_000  # disjoint from any corpus seed block


def _board(payload: dict) -> Board:
    return Board.from_rows(payload["top"], payload["middle"], payload["bottom"])


def observation_of(record: dict) -> ThreeMaxObservation:
    return ThreeMaxObservation(
        hero_board=_board(record["hero_board"]),
        opponent_boards=tuple(_board(p) for p in record["opponent_boards"]),
        dealt_cards=tuple(record["dealt"]),
        hero_private_discards=tuple(record["hero_private_discards"]),
        seat=record["seat"],
        street=record["street"],
    )


def measure(
    *, corpus: Path, roots: int, samples: int, fl_ev_14: float
) -> dict:
    records = []
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
            if len(records) >= roots:
                break

    started = time.time()
    forward: list[float] = []
    backward: list[float] = []
    agree = 0
    for record in records:
        observation = observation_of(record)
        original = {
            tuple(map(tuple, action["placements"])): action["ev"]
            for action in record["actions"]
        }
        relabelled = evaluate_t3(
            observation,
            samples=samples,
            seed=SEED_OFFSET + record["seed"],
            fl_ev_per_pair={14: fl_ev_14},
        )
        second = {
            tuple(map(tuple, candidate.action.placements)): candidate.ev
            for candidate in relabelled
        }

        first_pick = max(original, key=lambda key: original[key])
        second_pick = max(second, key=lambda key: second[key])
        agree += first_pick == second_pick
        # Each labelling judges the other's pick on its own scale.
        forward.append(original[first_pick] - original[second_pick])
        backward.append(second[second_pick] - second[first_pick])

    symmetric = [(a + b) / 2 for a, b in zip(forward, backward)]
    return {
        "roots": len(records),
        "samples": samples,
        "relabel_seed_offset": SEED_OFFSET,
        "argmax_agreement": agree / max(len(records), 1),
        "floor_forward": statistics.fmean(forward),
        "floor_backward": statistics.fmean(backward),
        "floor": statistics.fmean(symmetric),
        "floor_p95": sorted(symmetric)[int(0.95 * (len(symmetric) - 1))],
        "wall_seconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=300)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--fl-ev-14", type=float, default=9.6)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = measure(
        corpus=args.corpus,
        roots=args.roots,
        samples=args.samples,
        fl_ev_14=args.fl_ev_14,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
