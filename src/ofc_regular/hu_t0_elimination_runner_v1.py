"""Run sequential elimination over a set of T0 openings.

The operational half of `hu_t0_sequential_elimination_v1`, which holds the
statistics. That module decides who survives; this one decides what to measure
next, calls the engine, and writes a batch per file so a stopped run resumes
exactly where it left off.

## Narrowing: the model, not a sieve

The field starts as the model's top ``--keep`` actions. The sieve it replaces
spends 64 rollouts on each of 232 actions before anything is scored; the model
spends one forward pass each. It is also the better narrowing stage: across the
seventeen openings settled on 2026-08-20 the model placed the measured best
first fifteen times, second once and third once -- worst rank three of 232 --
so a keep of fifteen sits five times deeper than any true best has been found.

Every candidate's model rank is written into batch zero. If a hand's eventual
winner ever turns up near the keep boundary, the cut was too tight and the files
say so rather than the run hiding it. That check exists because the sieve-based
top-ten this replaced *did* hide exactly that: on one opening eighteen
candidates were still within four sigma after a batch and the cut had discarded
eight of them.

## Ordering: one batch per open hand per pass

Never one hand to completion. An earlier version did that and stalled -- two
workers sat on openings needing millions of particles while hands behind them
that needed only a convergence check went unvisited, and a third worker that had
finished its list idled. Round-robin costs an unfinishable hand one batch a pass
instead of a queue.

## Workers

``--worker i --workers n`` strides the hand list. Workers coordinate only
through the filesystem: a batch file that already exists is never recomputed,
and files are written to a sibling ``.partial`` and renamed, so a worker killed
mid-write leaves nothing a resume would mistake for complete.

Set ``RAYON_NUM_THREADS`` so that ``workers * threads`` matches the machine. The
engine parallelises by splitting the candidate list, so thread counts that
divide the candidate count use them all and others do not: ten candidates over
five threads is five full chunks, over eight threads it is five chunks on eight
threads and three idle.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from collections.abc import Sequence
from typing import Any

from ofc_regular.hu_t0_sequential_elimination_v1 import (
    DEFAULT_BUDGET,
    Batch,
    read_batches,
    survivors,
    verdict,
)

BATCH_SCHEMA = "regular_ofc_elim_batch_v1"
DEFAULT_SAMPLES = 1024
DEFAULT_KEEP = 15
RANK_ORDER = {"A": 14, "K": 13, "Q": 12, "J": 11, "T": 10}


def rank_of(card: str) -> int:
    return RANK_ORDER.get(card[0], int(card[0]) if card[0].isdigit() else 0)


def placement_label(placements: Sequence[Sequence[str]]) -> str:
    rows: dict[str, list[str]] = {"top": [], "middle": [], "bottom": []}
    for card, row in placements:
        rows[row].append(card)
    return " / ".join(" ".join(sorted(rows[name], key=lambda c: -rank_of(c))) or "-"
                      for name in ("top", "middle", "bottom"))


def observation_dict(hand: Sequence[str], fl_ev: float = 9.6) -> dict[str, Any]:
    """The T0 first-seat information set: five cards, both boards empty."""

    return {
        "schema": "regular_ofc_actor_observation_v1",
        "street": "T0", "seat": "first", "to_act_order": "first",
        "dealt_cards": list(hand),
        "hero_board": {"top": [], "middle": [], "bottom": []},
        "hero_private_discards": [], "hero_in_fantasyland": False,
        "opponent_public_board": {"top": [], "middle": [], "bottom": []},
        "opponent_discard_count": 0, "opponent_in_fantasyland": False,
        "scoring": {"schema": "regular_ofc_scoring_context_v1",
                    "fl_ev": {"14": fl_ev}, "fantasyland_cards": 14,
                    "foul_enabled": True, "hu_line_points": True,
                    "middle_trips_royalty": 2, "scoop_bonus": 3},
    }


def batch_path(out: pathlib.Path, index: int, number: int) -> pathlib.Path:
    return out / f"h{index:03d}_b{number:02d}.json"


def existing(out: pathlib.Path, index: int) -> list[Batch]:
    return read_batches([(out, f"h{index:03d}_b*.json")])


def next_step(batches: Sequence[Batch], budget: float) -> tuple[str, list[str]]:
    """What to do with this opening now: finish it, or measure these actions."""

    if not batches:
        return "seed", []
    state = verdict(batches, budget=budget)
    if state["state"] != "open":
        return state["state"], state["alive"]
    alive, _, _ = survivors(batches)
    return "open", alive


def write_batch(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Write through a sibling temp so a kill cannot leave a half file behind."""

    staging = path.with_suffix(".partial")
    staging.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    staging.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", required=True,
                        help="JSON with a 'hands' list of five-card openings")
    parser.add_argument("--out", required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                        help="reference particles a hand, in 1,024 units")
    parser.add_argument("--seed-base", type=int, default=200_000_000)
    parser.add_argument("--engine", default="",
                        help="engine library path; blank uses the pinned default")
    args = parser.parse_args(argv)

    # Imported here so `--help` and the tests do not need the native engine.
    from trainer import engine_eval
    import ofc_regular.hu_m3_rust as m3
    from ofc_regular.hu_infoset import ActorObservation
    from ofc_regular.hu_turn3_joint_exact_teacher import JointExactConfig

    library = (m3.load_native_engine(path=args.engine) if args.engine
               else m3.load_native_engine())
    _, weights = engine_eval._ensure_loaded()
    pins: dict[str, str] = {}
    for stem, (path, digest) in weights.items():
        prefix, name = (("fast_", stem[5:]) if stem.startswith("fast_")
                        else ("learned_", stem))
        pins[f"{prefix}{name}_model_path"] = path
        pins[f"{prefix}{name}_model_sha256"] = digest
    t0_path, t0_sha = weights["t0_first"]

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = json.loads(pathlib.Path(args.hands).read_text(encoding="utf-8"))
    hands = [{"index": n, "hand": list(cards),
              "observation": ActorObservation.from_dict(observation_dict(cards))}
             for n, cards in enumerate(payload["hands"])]
    mine = hands[args.worker::args.workers]
    print(f"worker {args.worker}/{args.workers}: {len(mine)} hands, "
          f"keep {args.keep}, budget {args.budget:.0f}k", flush=True)

    places: dict[int, dict[str, list]] = {}
    open_hands = list(mine)
    while open_hands:
        still = []
        for entry in open_hands:
            index = entry["index"]
            batches = existing(out, index)
            state, alive = next_step(batches, args.budget)

            if state in {"converged", "budget"}:
                final = verdict(batches, budget=args.budget)
                best = final["best"]
                print(f"DONE hand {index} {' '.join(entry['hand'])}: "
                      f"{len(final['alive'])} left after "
                      f"{final['particles_spent']:.0f}k ({state})   "
                      f"EV {final['means'][best]:+.3f}  gap {final['gap']:+.3f}   "
                      f"{placement_label(places.get(index, {}).get(best, []))}",
                      flush=True)
                continue

            target = batch_path(out, index, len(batches))
            if target.is_file():
                still.append(entry)      # a sibling worker took it; re-read next pass
                continue

            model_rank: dict[str, int] = {}
            if state == "seed":
                scored = m3.evaluate_request({
                    "schema": m3.HU_M3_REQUEST_SCHEMA, "kind": "model_scores",
                    "observation": entry["observation"].to_dict(),
                    "observation_fingerprint": entry["observation"].fingerprint(),
                    "config": {"learned_t0_first_model_path": t0_path,
                               "learned_t0_first_model_sha256": t0_sha},
                }, library=library)
                if scored.get("status") != "ok":
                    raise SystemExit(f"hand {index} model_scores: {str(scored)[:200]}")
                order = [row["action_key"] for row in
                         sorted(scored["actions"], key=lambda r: -float(r["score"]))]
                model_rank = {key: n + 1 for n, key in enumerate(order)}
                alive = order[:args.keep]

            seed = args.seed_base + index * 100_000 + len(batches) * 1_000
            config = JointExactConfig(
                evaluation_samples=args.samples,
                seed=seed, candidate_seed=seed, evaluation_seed=seed,
                seat="first", to_act_order="first",
                run_id=f"elim-h{index}-b{len(batches)}",
                restrict_action_keys=tuple(alive), **pins)

            began, wall = os.times(), time.time()
            result = m3.evaluate_t0(entry["observation"], config=config,
                                    library=library)
            used = os.times()
            if result.get("status") != "ok":
                raise SystemExit(f"hand {index} b{len(batches)}: {str(result)[:250]}")
            rows = [r for r in result["actions"] if r.get("score") is not None]
            places.setdefault(index, {}).update(
                {r["action_key"]: r.get("placements", []) for r in rows})

            write_batch(target, {
                "schema": BATCH_SCHEMA,
                "hand_index": index, "batch": len(batches), "seed": seed,
                "observation": {"dealt_cards": list(entry["hand"])},
                "samples": args.samples, "candidates": len(rows),
                "runs": [{"seed_trial": len(batches),
                          "scores": {r["action_key"]: float(r["score"])
                                     for r in rows}}],
                "placements": {r["action_key"]: r.get("placements", [])
                               for r in rows},
                "model_rank": model_rank,
                "wall_s": round(time.time() - wall, 1),
                "core_s": round((used.user + used.system)
                                - (began.user + began.system), 1),
            })
            print(f"  h{index:03d} b{len(batches):02d}: {len(rows)} candidates, "
                  f"{time.time() - wall:.0f}s", flush=True)
            still.append(entry)
        open_hands = still

    print(f"worker {args.worker}: finished", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
