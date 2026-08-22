"""T2 gate phase B: the T1 version one street deeper.

Deltas from t1_gate_b.py: street T2, evaluate_t2, the t2 models for the model
rank, the hero's one prior discard in the observation, and a fresh seed base.
Stop rule and worktree discipline unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import sys
import time
from typing import Any

os.environ.setdefault("RAYON_NUM_THREADS", "3")

WT_SRC = "C:/TMP/opt-wt/src"
REPO = "C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple/regular-ofc-pineapple"
ENGINE = "D:/ofc-build-cache/opt-experiment/release/ofc_hu_m3_engine.dll"

sys.path.insert(0, REPO)
sys.path.insert(0, WT_SRC)

import ofc_regular.hu_m3_rust as m3  # noqa: E402
from ofc_regular.hu_infoset import ActorObservation  # noqa: E402
from ofc_regular.hu_turn3_joint_exact_teacher import JointExactConfig  # noqa: E402

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
        "street": "T2", "seat": record["seat"], "to_act_order": record["seat"],
        "dealt_cards": list(record["hand"]),
        "hero_board": {r: list(c) for r, c in record["hero_board"].items()},
        "hero_private_discards": list(record["hero_discards"]),
        "hero_in_fantasyland": False,
        "opponent_public_board": {r: list(c) for r, c in record["opp_board"].items()},
        "opponent_discard_count": record["opp_discard_count"],
        "opponent_in_fantasyland": False,
        "scoring": {"schema": "regular_ofc_scoring_context_v1",
                    "fl_ev": {"14": fl_ev}, "fantasyland_cards": 14,
                    "foul_enabled": True, "hu_line_points": True,
                    "middle_trips_royalty": 2, "scoop_bonus": 3},
    }


def champ_key_of(record: dict, placements: dict[str, Any]) -> str | None:
    want = (sorted(tuple(p) for p in record["champ_placements"]),
            record.get("champ_discard"))
    for key, meta in placements.items():
        if (sorted(tuple(p) for p in meta["placements"]) == want[0]
                and meta.get("discard") == want[1]):
            return key
    return None


def fate(batch_scores: list[dict[str, float]], champ: str) -> dict[str, Any]:
    pooled: dict[str, float] = {}
    for scores in batch_scores:
        for key, value in scores.items():
            pooled[key] = pooled.get(key, 0.0) + value
    order = sorted(pooled, key=lambda k: -pooled[k])
    rival = order[1] if order[0] == champ else order[0]
    gaps = [s[rival] - s[champ] for s in batch_scores
            if rival in s and champ in s]
    n = len(gaps)
    mean = sum(gaps) / n
    if n < 2:
        return {"decided": False, "gap": mean, "se": None, "t": None,
                "champ_is_leader": order[0] == champ}
    sd = math.sqrt(sum((g - mean) ** 2 for g in gaps) / (n - 1))
    se = sd / math.sqrt(n) if sd > 0 else 1e-9
    return {"decided": abs(mean / se) >= 4.0 and n >= 3,
            "gap": mean, "se": se, "t": mean / se,
            "champ_is_leader": order[0] == champ}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--min-batches", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=500_000_000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    base = pathlib.Path(args.dir)
    out = base / "batches"
    out.mkdir(parents=True, exist_ok=True)

    library = m3.load_native_engine(path=ENGINE)
    _, weights = engine_eval._ensure_loaded()
    pins: dict[str, str] = {}
    for stem, (path, digest) in weights.items():
        prefix, name = (("fast_", stem[5:]) if stem.startswith("fast_")
                        else ("learned_", stem))
        pins[f"{prefix}{name}_model_path"] = path
        pins[f"{prefix}{name}_model_sha256"] = digest

    records = [json.loads(line) for line in
               (base / "positions.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if args.limit:
        records = records[:args.limit]
    seat = records[0]["seat"]
    seat_off = 0 if seat == "first" else 50_000_000
    mine = records[args.worker::args.workers]
    print(f"T2-{seat} worker {args.worker}/{args.workers}: {len(mine)} positions, "
          f"{args.samples}p x [{args.min_batches},{args.max_batches}] batches",
          flush=True)

    for record in mine:
        index = record["index"]
        done_mark = out / f"p{index:03d}_done.json"
        if done_mark.is_file():
            continue
        observation = ActorObservation.from_dict(observation_dict(record))

        batch_scores: list[dict[str, float]] = []
        placements: dict[str, Any] = {}
        model_rank: dict[str, int] = {}
        number = 0
        while True:
            existing_path = out / f"p{index:03d}_b{number:02d}.json"
            if existing_path.is_file():
                payload = json.loads(existing_path.read_text(encoding="utf-8"))
                batch_scores.append(payload["runs"][0]["scores"])
                for key, meta in payload["placements"].items():
                    placements[key] = meta
                model_rank = payload.get("model_rank") or model_rank
                number += 1
                continue
            break

        while True:
            if len(batch_scores) >= args.min_batches:
                champ = champ_key_of(record, placements)
                if champ is None:
                    print(f"p{index:03d}: champion action not in fan (!)",
                          flush=True)
                    break
                verdict = fate(batch_scores, champ)
                if verdict["decided"] or len(batch_scores) >= args.max_batches:
                    break

            if number == 0 and not model_rank:
                scored = m3.evaluate_request({
                    "schema": m3.HU_M3_REQUEST_SCHEMA, "kind": "model_scores",
                    "observation": observation.to_dict(),
                    "observation_fingerprint": observation.fingerprint(),
                    "config": {f"learned_t2_{seat}_model_path": weights[f"t2_{seat}"][0],
                               f"learned_t2_{seat}_model_sha256": weights[f"t2_{seat}"][1]},
                }, library=library)
                if scored.get("status") == "ok":
                    order = [row["action_key"] for row in
                             sorted(scored["actions"],
                                    key=lambda r: -float(r["score"]))]
                    model_rank = {key: n + 1 for n, key in enumerate(order)}

            seed = args.seed_base + seat_off + index * 100_000 + number * 1_000
            config = JointExactConfig(
                evaluation_samples=args.samples,
                seed=seed, candidate_seed=seed, evaluation_seed=seed,
                seat=seat, to_act_order=seat,
                run_id=f"t2gate-{seat}-p{index}-b{number}",
                **pins)
            wall = time.time()
            result = m3.evaluate_t2(observation, config=config, library=library)
            if result.get("status") != "ok":
                raise SystemExit(f"p{index} b{number}: {str(result)[:250]}")
            rows = [r for r in result["actions"] if r.get("score") is not None]
            scores = {r["action_key"]: float(r["score"]) for r in rows}
            batch_scores.append(scores)
            for r in rows:
                placements[r["action_key"]] = {
                    "placements": r.get("placements", []),
                    "discard": (r.get("discards") or [None])[0],
                }

            payload = {
                "schema": BATCH_SCHEMA,
                "hand_index": index, "batch": number, "seed": seed,
                "samples": args.samples, "candidates": len(rows),
                "runs": [{"seed_trial": number, "scores": scores}],
                "placements": placements if number == 0 else
                    {k: placements[k] for k in scores},
                "model_rank": model_rank if number == 0 else {},
                "wall_s": round(time.time() - wall, 1),
            }
            staging = existing_path.with_suffix(".partial")
            staging.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            staging.replace(existing_path)
            print(f"p{index:03d} b{number}: {len(rows)} cands x {args.samples}p "
                  f"in {time.time()-wall:.0f}s", flush=True)
            number += 1
            existing_path = out / f"p{index:03d}_b{number:02d}.json"

        champ = champ_key_of(record, placements)
        verdict = fate(batch_scores, champ) if champ else None
        done_mark.write_text(json.dumps({
            "index": index, "seat": seat, "street": "T2",
            "batches": len(batch_scores),
            "particles": len(batch_scores) * args.samples,
            "champ_key": champ,
            "champ_model_rank": model_rank.get(champ) if champ else None,
            "fan": len(placements),
            "verdict": verdict,
        }, indent=1), encoding="utf-8")
        if verdict:
            state = ("winner" if verdict["champ_is_leader"] else "behind")
            firm = "decided" if verdict["decided"] else "open"
            print(f"DONE p{index:03d}: {state}/{firm} gap {verdict['gap']:+.3f} "
                  f"({len(batch_scores)} batches)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
