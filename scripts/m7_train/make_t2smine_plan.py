"""Build an explicit-roots T2-second mining plan from the v7 package base.

The base is the akq500-v7 worker plan: same package (m7v7), same engine and
CURRENT-champion weight digests -- which is the point.  Mining labels must be
priced under the continuations the champion actually plays, so the pins come
from the plan that carries them, not from the m7v5 label-era generator whose
pins reproduce the previous generation.

Overrides: street/seat, explicit T2 observations as roots, 8 seeds x 256
samples per position (the fixed-depth reformulation of the local gate), no
prefilter, fresh eval seed base, and the shard split.

The result is validated by the repo worker's own load_plan before anything is
uploaded: geometry, root schema, and refused-field checks all run locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path("C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple/regular-ofc-pineapple")
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def make_shards(position_count: int, shard_count: int) -> list[dict]:
    small, extra = divmod(position_count, shard_count)
    counts = [small] * (shard_count - extra) + [small + 1] * extra
    width = max(2, len(str(shard_count - 1)))
    shards, start = [], 0
    for index, count in enumerate(counts):
        shards.append({"shard_id": f"{index:0{width}d}", "start": start,
                       "count": count})
        start += count
    return shards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", default="C:/tmp/v7plan.json")
    parser.add_argument("--positions", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--street", default="T2")
    parser.add_argument("--seat", default="second")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seeds-per-position", type=int, default=8)
    parser.add_argument("--eval-seed-base", type=int, default=30_000_000)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = json.loads(pathlib.Path(args.base_plan).read_text(encoding="utf-8"))
    records = [json.loads(line) for line in
               pathlib.Path(args.positions).read_text(encoding="utf-8").splitlines()
               if line.strip()]

    roots = []
    for record in records:
        roots.append({
            "dealt_cards": list(record["hand"]),
            "hero_board": {row: list(cards) for row, cards
                           in record["hero_board"].items()},
            "opponent_public_board": {row: list(cards) for row, cards
                                      in record["opp_board"].items()},
            "hero_private_discards": list(record.get("hero_discards", [])),
            "opponent_discard_count": record["opp_discard_count"],
            "note": f"onpolicy hand_seed {record['hand_seed']}",
        })

    plan = dict(base)
    plan.update({
        "job_id": args.job_id,
        "street": args.street,
        "seat": args.seat,
        "samples": args.samples,
        "seeds_per_position": args.seeds_per_position,
        "eval_seed_base": args.eval_seed_base,
        "root_source": "explicit",
        "roots": roots,
        "shards": make_shards(len(roots), args.shards),
    })
    for field in ("hand_seed_base", "behavior_seed_offset",
                  "prefilter_samples", "prefilter_keep", "prefilter_margin"):
        plan.pop(field, None)
    plan["provenance"] = {
        "schema": "t2smine_plan_provenance_v1",
        "purpose": ("on-policy T2-second mining: fixed-depth 8x256p explicit "
                    "roots, current-champion pins inherited from the v7 base"),
        "base_plan_sha256": hashlib.sha256(
            pathlib.Path(args.base_plan).read_bytes()).hexdigest(),
        "positions_source": str(args.positions),
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(plan, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True).encode("ascii")
    out.write_bytes(raw)
    print(f"plan: {out}  ({len(roots)} roots, {args.shards} shards)")
    print(f"sha256: {hashlib.sha256(raw).hexdigest()}")

    from ofc_regular import hu_m31_label_gen_worker_v1 as worker
    loaded = worker.load_plan(out)
    print(f"load_plan: OK  street={loaded['street']} seat={loaded['seat']} "
          f"samples={loaded['samples']} seeds={loaded['seeds_per_position']} "
          f"roots={len(loaded['roots'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
