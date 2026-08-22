"""Extract confirmed misses from a fleet mining run.

Fleet position files carry 8 shared-world runs of rak1-keyed scores.  The
champion's pick comes from the positions file (phase A), translated to its
canonical key by the same action_key module the engine mirrors.  The verdict
is the street gate's: champion vs pooled leader, empirical SE over the 8 runs,
4 sigma to call it.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path("C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple/regular-ofc-pineapple")
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ofc_regular.action_key import action_key_from_payload  # noqa: E402


def fate(runs: list[dict[str, float]], champ: str) -> dict:
    pooled: dict[str, float] = {}
    for scores in runs:
        for key, value in scores.items():
            pooled[key] = pooled.get(key, 0.0) + value
    order = sorted(pooled, key=lambda k: -pooled[k])
    rival = order[1] if order[0] == champ else order[0]
    gaps = [s[rival] - s[champ] for s in runs if rival in s and champ in s]
    n = len(gaps)
    mean = sum(gaps) / n
    sd = math.sqrt(sum((g - mean) ** 2 for g in gaps) / (n - 1))
    se = sd / math.sqrt(n) if sd > 0 else 1e-9
    return {"champ_is_leader": order[0] == champ,
            "decided": abs(mean / se) >= 4.0 and n >= 3,
            "gap": mean, "se": se, "t": mean / se,
            "winner": order[0], "pooled_mean_winner": pooled[order[0]] / n}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--received", required=True,
                        help="directory holding <shard>/files/position_*.json")
    parser.add_argument("--positions", required=True,
                        help="phase-A jsonl in plan-roots order")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in
               pathlib.Path(args.positions).read_text(encoding="utf-8").splitlines()
               if line.strip()]

    files = sorted(pathlib.Path(args.received).glob("*/files/position_*.json"))
    fates = {"winner_decided": 0, "winner_open": 0,
             "behind_decided": 0, "behind_open": 0}
    misses, mismatched = [], 0
    for path in files:
        d = json.loads(path.read_text(encoding="utf-8"))
        offset = d["offset"]
        record = records[offset]
        champ_key = action_key_from_payload({
            "placements": [tuple(p) for p in record["champ_placements"]],
            "discards": [record["champ_discard"]] if record.get("champ_discard") else [],
        }).to_token()
        runs = [r["scores"] for r in d["runs"]]
        if champ_key not in runs[0]:
            mismatched += 1
            continue
        v = fate(runs, champ_key)
        key = (("winner" if v["champ_is_leader"] else "behind")
               + ("_decided" if v["decided"] else "_open"))
        fates[key] += 1
        if not v["champ_is_leader"] and v["decided"]:
            misses.append({
                "offset": offset,
                "hand_seed": record["hand_seed"],
                "gap": round(v["gap"], 3), "t": round(v["t"], 1),
                "champ_key": champ_key, "winner_key": v["winner"],
                "position_file": str(path),
            })

    n = sum(fates.values())
    summary = {"positions": n, "fates": fates, "mismatched": mismatched,
               "miss_rate": len(misses) / n if n else None,
               "gap_mean": (sum(m["gap"] for m in misses) / len(misses)
                            if misses else None)}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for miss in misses:
            handle.write(json.dumps(miss) + "\n")
        handle.write(json.dumps({"summary": summary}) + "\n")

    print(f"n={n}  mismatched={mismatched}")
    for k, c in fates.items():
        print(f"  {k:16s} {c:4d}  ({c/n:.1%})")
    if misses:
        gaps = sorted(m["gap"] for m in misses)
        print(f"確定誤り {len(misses)}件 ({len(misses)/n:.1%})  "
              f"gap中央値 {gaps[len(gaps)//2]:+.3f}  平均 {summary['gap_mean']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
