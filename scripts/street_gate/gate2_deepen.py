"""Deepen one gate-ii position: keep adding shared-CRN batches on a fixed
candidate set until every candidate carries the target particles.

Continues the existing batch numbering, so phase C and the archive stay valid.
Same worktree discipline as phase B; same seed scheme, so batches added later
are the batches the runner would have drawn next.
"""

from __future__ import annotations

import argparse
import dataclasses
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

sys.path.insert(0, REPO)
sys.path.insert(0, WT_SRC)

import ofc_regular.hu_m3_rust as m3  # noqa: E402
from ofc_regular.hu_infoset import ActorObservation  # noqa: E402
from ofc_regular.hu_turn3_joint_exact_teacher import JointExactConfig  # noqa: E402
from ofc_regular.hu_t0_sequential_elimination_v1 import read_batches  # noqa: E402

if "restrict_action_keys" not in {f.name for f in dataclasses.fields(JointExactConfig)}:
    raise SystemExit("unpatched JointExactConfig")

from trainer import engine_eval  # noqa: E402

_ORIGIN = str(pathlib.Path(sys.modules["ofc_regular"].__file__ or "").resolve()).lower()
if not _ORIGIN.startswith(str(pathlib.Path(WT_SRC).resolve()).lower()):
    raise SystemExit(f"ofc_regular resolved outside the worktree: {_ORIGIN}")

BATCH_SCHEMA = "regular_ofc_elim_batch_v1"


def observation_dict(record: dict[str, Any], fl_ev: float = 9.6) -> dict[str, Any]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--keys-mode", choices=["top2", "alive"], required=True,
                        help="top2: best two by pooled mean over every key ever "
                             "measured; alive: the final surviving set")
    parser.add_argument("--target-k", type=float, default=16.0,
                        help="cumulative particles per candidate, in 1,024 units")
    parser.add_argument("--positions", default="D:/ofc_data/t0_rung/gate2_t0second/positions.jsonl")
    parser.add_argument("--batches", default="D:/ofc_data/t0_rung/gate2_t0second/batches")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seed-base", type=int, default=300_000_000)
    args = parser.parse_args()

    index = args.position
    out = pathlib.Path(args.batches)
    record = next(json.loads(l) for l in
                  pathlib.Path(args.positions).read_text(encoding="utf-8").splitlines()
                  if l.strip() and json.loads(l)["index"] == index)
    observation = ActorObservation.from_dict(observation_dict(record))

    batches = read_batches([(out, f"p{index:03d}_b*.json")])
    if not batches:
        raise SystemExit(f"p{index} has no batches; run phase B first")

    tally: dict[str, dict[str, float]] = {}
    for b in batches:
        for k, s in b["scores"].items():
            e = tally.setdefault(k, {"particles": 0, "wsum": 0.0})
            e["particles"] += b["samples"]
            e["wsum"] += b["samples"] * s

    if args.keys_mode == "top2":
        keys = sorted(tally, key=lambda k: -tally[k]["wsum"] / tally[k]["particles"])[:2]
    else:
        # The alive set is whoever the last batch still measured: read_batches
        # returns files in name order and elimination only ever shrinks the set.
        keys = sorted(batches[-1]["scores"].keys())
    print(f"p{index}: deepening {len(keys)} keys to {args.target_k:.0f}k each "
          f"({args.keys_mode})", flush=True)
    for k in keys:
        print(f"  {k[:60]}  now {tally[k]['particles']:,}p "
              f"EV {tally[k]['wsum']/tally[k]['particles']:+.3f}", flush=True)

    library = m3.load_native_engine(path=ENGINE)
    _, weights = engine_eval._ensure_loaded()
    pins: dict[str, str] = {}
    for stem, (path, digest) in weights.items():
        prefix, name = (("fast_", stem[5:]) if stem.startswith("fast_")
                        else ("learned_", stem))
        pins[f"{prefix}{name}_model_path"] = path
        pins[f"{prefix}{name}_model_sha256"] = digest

    target = args.target_k * 1024
    number = len(batches)
    while min(tally[k]["particles"] for k in keys) < target:
        path = out / f"p{index:03d}_b{number:02d}.json"
        if path.is_file():
            number += 1
            continue
        seed = args.seed_base + index * 100_000 + number * 1_000
        config = JointExactConfig(
            evaluation_samples=args.samples,
            seed=seed, candidate_seed=seed, evaluation_seed=seed,
            seat="second", to_act_order="second",
            run_id=f"gate2deep-p{index}-b{number}",
            restrict_action_keys=tuple(keys), **pins)
        wall = time.time()
        result = m3.evaluate_t0(observation, config=config, library=library)
        if result.get("status") != "ok":
            raise SystemExit(f"b{number}: {str(result)[:250]}")
        rows = [r for r in result["actions"] if r.get("score") is not None]
        payload = {
            "schema": BATCH_SCHEMA,
            "hand_index": index, "batch": number, "seed": seed,
            "observation": {"dealt_cards": list(record["hand"]),
                            "opp_board": record["opp_board"]},
            "samples": args.samples, "candidates": len(rows),
            "runs": [{"seed_trial": number,
                      "scores": {r["action_key"]: float(r["score"]) for r in rows}}],
            "placements": {r["action_key"]: r.get("placements", []) for r in rows},
            "model_rank": {},
            "wall_s": round(time.time() - wall, 1),
        }
        staging = path.with_suffix(".partial")
        staging.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        staging.replace(path)
        for r in rows:
            e = tally[r["action_key"]]
            e["particles"] += args.samples
            e["wsum"] += args.samples * float(r["score"])
        done = min(tally[k]["particles"] for k in keys)
        print(f"p{index} b{number}: {len(rows)} cands x {args.samples}p "
              f"in {time.time()-wall:.0f}s  ({done:,}/{target:,.0f}p)", flush=True)
        number += 1

    print(f"p{index}: DONE", flush=True)
    for k in keys:
        print(f"  {k[:60]}  {tally[k]['particles']:,}p "
              f"EV {tally[k]['wsum']/tally[k]['particles']:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
