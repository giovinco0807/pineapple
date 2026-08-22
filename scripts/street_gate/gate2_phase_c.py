"""Gate ii phase C: where does the champion's pick land in each priced fan?

Per position: the champion's action key (matched by placements), its model
rank, and its fate under elimination -- winner, in the surviving tie set, or
eliminated with a paired gap to the leader.  The summary is the divergence
picture the UPGRADE sizing needs.

Reads only batch files; runs anywhere (uses the worktree stats module for the
paired arithmetic so the two phases share one implementation).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

WT_SRC = "C:/TMP/opt-wt/src"
sys.path.insert(0, WT_SRC)

from ofc_regular.hu_t0_sequential_elimination_v1 import (  # noqa: E402
    paired_gap,
    read_batches,
    standard_error,
    verdict,
)


def champion_key(record: dict, batch0: dict) -> str | None:
    want = sorted(tuple(p) for p in record["champ_placements"])
    for key, placements in batch0["placements"].items():
        if sorted(tuple(p) for p in placements) == want:
            return key
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", required=True)
    parser.add_argument("--batches", required=True)
    parser.add_argument("--budget", type=float, default=8.0)
    args = parser.parse_args()

    batches_dir = pathlib.Path(args.batches)
    records = [json.loads(line) for line in
               pathlib.Path(args.positions).read_text(encoding="utf-8").splitlines()
               if line.strip()]

    rows = []
    for record in records:
        index = record["index"]
        b0_path = batches_dir / f"p{index:03d}_b00.json"
        if not b0_path.is_file():
            rows.append({"index": index, "state": "pending"})
            continue
        b0 = json.loads(b0_path.read_text(encoding="utf-8"))
        batches = read_batches([(batches_dir, f"p{index:03d}_b*.json")])
        final = verdict(batches, budget=args.budget)
        key = champion_key(record, b0)
        model_rank = b0["model_rank"].get(key) if key else None

        if key is None:
            fate = "not-in-keep"
            gap = se = None
        elif key == final["best"]:
            fate = "winner"
            gap, mass = paired_gap(batches, final["best"], final["alive"][1]) \
                if len(final["alive"]) > 1 else (final["gap"], 0)
            se = standard_error(mass) if mass else None
        elif key in final["alive"]:
            fate = "tied"
            gap, mass = paired_gap(batches, final["best"], key)
            se = standard_error(mass) if mass else None
        else:
            fate = "eliminated"
            gap, mass = paired_gap(batches, final["best"], key)
            se = standard_error(mass) if mass else None

        rows.append({
            "index": index,
            "state": final["state"],
            "n_batches": len(batches),
            "alive": len(final["alive"]),
            "spent_k": round(final["particles_spent"], 1),
            "champ_model_rank": model_rank,
            "fate": fate,
            "champ_gap": None if gap is None else round(gap, 3),
            "gap_se": None if se is None else round(se, 3),
            "sigmas": None if not gap or not se else round(gap / se, 1),
        })

    done = [r for r in rows if r["state"] not in ("pending",)]
    print(f"{'idx':>3} {'state':>9} {'b':>2} {'alive':>5} {'spent':>6} "
          f"{'mrank':>5} {'fate':>11} {'gap':>7} {'se':>6} {'sig':>5}")
    for r in rows:
        if r["state"] == "pending":
            print(f"{r['index']:>3}   pending")
            continue
        print(f"{r['index']:>3} {r['state']:>9} {r['n_batches']:>2} "
              f"{r['alive']:>5} {r['spent_k']:>6} "
              f"{str(r['champ_model_rank']):>5} {r['fate']:>11} "
              f"{str(r['champ_gap']):>7} {str(r['gap_se']):>6} "
              f"{str(r['sigmas']):>5}")

    if done:
        n = len(done)
        winners = sum(1 for r in done if r["fate"] == "winner")
        tied = sum(1 for r in done if r["fate"] == "tied")
        elim = [r for r in done if r["fate"] == "eliminated"]
        missing = sum(1 for r in done if r["fate"] == "not-in-keep")
        print(f"\npositions analysed  : {n}")
        print(f"champion = winner   : {winners}  ({winners/n:.0%})")
        print(f"champion in tie set : {tied}  ({tied/n:.0%})")
        print(f"champion eliminated : {len(elim)}  ({len(elim)/n:.0%})")
        print(f"champion not in keep: {missing}")
        if elim:
            gaps = sorted((r["champ_gap"] for r in elim), reverse=True)
            print(f"eliminated gaps     : max {gaps[0]:+.3f}  "
                  f"median {gaps[len(gaps)//2]:+.3f}  min {gaps[-1]:+.3f}")
            firm = [r for r in elim if r["sigmas"] and r["sigmas"] >= 3]
            print(f"eliminated at >=3sig: {len(firm)}")
        ranks = [r["champ_model_rank"] for r in done if r["champ_model_rank"]]
        if ranks:
            print(f"champ model rank    : all==1 {all(x == 1 for x in ranks)} "
                  f"(max {max(ranks)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
