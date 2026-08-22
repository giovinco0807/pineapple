"""Gate ii phase B: price each (T0, second) position's full fan by elimination.

Worktree environment: ofc_regular comes from C:/TMP/opt-wt/src (the copy whose
JointExactConfig carries restrict_action_keys), the engine DLL from
D:/ofc-build-cache/opt-experiment/release.  trainer comes from the main repo,
whose engine_eval inserts its own src at sys.path[0] -- so every ofc_regular
module is imported and pinned in sys.modules BEFORE trainer is touched, and the
origin is asserted afterwards.

Shape is hu_t0_elimination_runner_v1 with three changes: positions carry an
opponent board, the seat is second, and the narrowing model is t0_second.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

os.environ.setdefault("RAYON_NUM_THREADS", "5")

WT_SRC = "C:/TMP/opt-wt/src"
REPO = "C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple/regular-ofc-pineapple"
ENGINE = "D:/ofc-build-cache/opt-experiment/release/ofc_hu_m3_engine.dll"

sys.path.insert(0, REPO)      # trainer
sys.path.insert(0, WT_SRC)    # ofc_regular, and it must stay first until pinned

import ofc_regular.hu_m3_rust as m3  # noqa: E402
from ofc_regular.hu_infoset import ActorObservation  # noqa: E402
from ofc_regular.hu_turn3_joint_exact_teacher import JointExactConfig  # noqa: E402
from ofc_regular.hu_t0_sequential_elimination_v1 import (  # noqa: E402
    read_batches,
    survivors,
    verdict,
)

import dataclasses  # noqa: E402

_FIELDS = {f.name for f in dataclasses.fields(JointExactConfig)}
if "restrict_action_keys" not in _FIELDS:
    raise SystemExit("unpatched JointExactConfig: restrict_action_keys missing "
                     f"(loaded from {sys.modules['ofc_regular.hu_turn3_joint_exact_teacher'].__file__})")

from trainer import engine_eval  # noqa: E402  (inserts repo src at position 0)

_OFC_ORIGIN = pathlib.Path(sys.modules["ofc_regular"].__file__ or "").resolve()
if not str(_OFC_ORIGIN).lower().startswith(str(pathlib.Path(WT_SRC).resolve()).lower()):
    raise SystemExit(f"ofc_regular resolved outside the worktree: {_OFC_ORIGIN}")

BATCH_SCHEMA = "regular_ofc_elim_batch_v1"


def observation_dict(record: dict[str, Any], fl_ev: float = 9.6) -> dict[str, Any]:
    """The T0 second-seat information set: five cards facing a placed board."""
    return {
        "schema": "regular_ofc_actor_observation_v1",
        "street": "T0", "seat": "second", "to_act_order": "second",
        "dealt_cards": list(record["hand"]),
        "hero_board": {"top": [], "middle": [], "bottom": []},
        "hero_private_discards": [], "hero_in_fantasyland": False,
        "opponent_public_board": {row: list(cards) for row, cards
                                  in record["opp_board"].items()},
        "opponent_discard_count": 0, "opponent_in_fantasyland": False,
        "scoring": {"schema": "regular_ofc_scoring_context_v1",
                    "fl_ev": {"14": fl_ev}, "fantasyland_cards": 14,
                    "foul_enabled": True, "hu_line_points": True,
                    "middle_trips_royalty": 2, "scoop_bonus": 3},
    }


def batch_path(out: pathlib.Path, index: int, number: int) -> pathlib.Path:
    return out / f"p{index:03d}_b{number:02d}.json"


def existing(out: pathlib.Path, index: int):
    return read_batches([(out, f"p{index:03d}_b*.json")])


def next_step(batches, budget: float) -> tuple[str, list[str]]:
    if not batches:
        return "seed", []
    state = verdict(batches, budget=budget)
    if state["state"] != "open":
        return state["state"], state["alive"]
    alive, _, _ = survivors(batches)
    return "open", alive


def write_batch(path: pathlib.Path, payload: dict[str, Any]) -> None:
    staging = path.with_suffix(".partial")
    staging.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    staging.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keep", type=int, default=32)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--budget", type=float, default=8.0,
                        help="reference particles a position, in 1,024 units")
    parser.add_argument("--seed-base", type=int, default=300_000_000)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many positions (0 = all)")
    args = parser.parse_args(argv)

    library = m3.load_native_engine(path=ENGINE)
    print(f"engine: {ENGINE}", flush=True)

    _, weights = engine_eval._ensure_loaded()
    pins: dict[str, str] = {}
    for stem, (path, digest) in weights.items():
        prefix, name = (("fast_", stem[5:]) if stem.startswith("fast_")
                        else ("learned_", stem))
        pins[f"{prefix}{name}_model_path"] = path
        pins[f"{prefix}{name}_model_sha256"] = digest
    t0s_path, t0s_sha = weights["t0_second"]

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in
               pathlib.Path(args.positions).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if args.limit:
        records = records[:args.limit]
    entries = [{"index": r["index"], "record": r,
                "observation": ActorObservation.from_dict(observation_dict(r))}
               for r in records]
    mine = entries[args.worker::args.workers]
    print(f"worker {args.worker}/{args.workers}: {len(mine)} positions, "
          f"keep {args.keep}, samples {args.samples}, "
          f"budget {args.budget:.0f}k", flush=True)

    open_positions = list(mine)
    while open_positions:
        still = []
        for entry in open_positions:
            index = entry["index"]
            batches = existing(out, index)
            state, alive = next_step(batches, args.budget)

            if state in {"converged", "budget"}:
                final = verdict(batches, budget=args.budget)
                best = final["best"]
                print(f"DONE p{index:02d}: {len(final['alive'])} alive after "
                      f"{final['particles_spent']:.0f}k ({state})  "
                      f"EV {final['means'][best]:+.3f}  gap {final['gap']:+.3f}",
                      flush=True)
                continue

            target = batch_path(out, index, len(batches))
            if target.is_file():
                still.append(entry)
                continue

            model_rank: dict[str, int] = {}
            if state == "seed":
                scored = m3.evaluate_request({
                    "schema": m3.HU_M3_REQUEST_SCHEMA, "kind": "model_scores",
                    "observation": entry["observation"].to_dict(),
                    "observation_fingerprint": entry["observation"].fingerprint(),
                    "config": {"learned_t0_second_model_path": t0s_path,
                               "learned_t0_second_model_sha256": t0s_sha},
                }, library=library)
                if scored.get("status") != "ok":
                    raise SystemExit(f"p{index} model_scores: {str(scored)[:200]}")
                order = [row["action_key"] for row in
                         sorted(scored["actions"], key=lambda r: -float(r["score"]))]
                model_rank = {key: n + 1 for n, key in enumerate(order)}
                alive = order[:args.keep]

            seed = args.seed_base + index * 100_000 + len(batches) * 1_000
            config = JointExactConfig(
                evaluation_samples=args.samples,
                seed=seed, candidate_seed=seed, evaluation_seed=seed,
                seat="second", to_act_order="second",
                run_id=f"gate2-p{index}-b{len(batches)}",
                restrict_action_keys=tuple(alive), **pins)

            wall = time.time()
            result = m3.evaluate_t0(entry["observation"], config=config,
                                    library=library)
            if result.get("status") != "ok":
                raise SystemExit(f"p{index} b{len(batches)}: {str(result)[:250]}")
            rows = [r for r in result["actions"] if r.get("score") is not None]
            if len(rows) != len(alive):
                print(f"  NOTE p{index} b{len(batches)}: asked {len(alive)} "
                      f"got {len(rows)} scored rows", flush=True)

            write_batch(target, {
                "schema": BATCH_SCHEMA,
                "hand_index": index, "batch": len(batches), "seed": seed,
                "observation": {"dealt_cards": list(entry["record"]["hand"]),
                                "opp_board": entry["record"]["opp_board"]},
                "samples": args.samples, "candidates": len(rows),
                "runs": [{"seed_trial": len(batches),
                          "scores": {r["action_key"]: float(r["score"])
                                     for r in rows}}],
                "placements": {r["action_key"]: r.get("placements", [])
                               for r in rows},
                "model_rank": model_rank,
                "wall_s": round(time.time() - wall, 1),
            })
            print(f"p{index:02d} b{len(batches)}: {len(rows)} cands x "
                  f"{args.samples}p in {time.time() - wall:.0f}s", flush=True)
            still.append(entry)
        open_positions = still
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
