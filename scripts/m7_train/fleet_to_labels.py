"""Convert fleet mining output into m7v5-shaped label records.

Two facts force the shape: labelgen_feature_dump reads only ``runs[0]``, so
the eight 256-particle runs are pooled into one 2,048-particle-equivalent run
(the same depth as the original corpus); and the train split buckets by a
stable hash of the position, so oversampling copies of the miss positions is
emitted as extra records with distinct offsets -- the copies share the hash
bucket and cannot straddle the holdout.

Outputs: labels_mine.jsonl (every position once), boost_xK.jsonl (the miss
positions repeated K-1 more times) for each requested factor.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--received", required=True)
    parser.add_argument("--misses", required=True,
                        help="misses.jsonl from fleet_mine_analyze")
    parser.add_argument("--offset-base", type=int, default=1_000_000)
    parser.add_argument("--boost", type=int, nargs="*", default=[8, 16])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    miss_offsets = set()
    for line in pathlib.Path(args.misses).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "offset" in record:
            miss_offsets.add(record["offset"])

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(pathlib.Path(args.received).glob("*/files/position_*.json"))

    mine_path = out_dir / "labels_mine.jsonl"
    boost_handles = {k: (out_dir / f"boost_x{k}.jsonl").open("w", encoding="utf-8")
                     for k in args.boost}
    n_written = n_miss = 0
    with mine_path.open("w", encoding="utf-8") as mine:
        for path in files:
            d = json.loads(path.read_text(encoding="utf-8"))
            obs = d["observation"]
            for required in ("seat", "street", "scoring", "to_act_order"):
                if required not in obs:
                    raise SystemExit(f"{path}: observation lacks {required}")
            keys = d["runs"][0]["scores"].keys()
            pooled = {k: sum(r["scores"][k] for r in d["runs"]) / len(d["runs"])
                      for k in keys}
            base_record = {
                "observation": obs,
                "offset": args.offset_base + d["offset"],
                "plan_sha256": d["plan_sha256"],
                "runs": [{"scores": pooled}],
                "skeleton": d["skeleton"],
            }
            mine.write(json.dumps(base_record, sort_keys=True) + "\n")
            n_written += 1
            if d["offset"] in miss_offsets:
                n_miss += 1
                for factor, handle in boost_handles.items():
                    for copy in range(1, factor):
                        rec = dict(base_record)
                        rec["offset"] = (args.offset_base + d["offset"]
                                         + 2_000_000 * copy)
                        handle.write(json.dumps(rec, sort_keys=True) + "\n")
    for handle in boost_handles.values():
        handle.close()

    print(f"labels_mine: {n_written} records ({n_miss} misses) -> {mine_path}")
    for k in args.boost:
        path = out_dir / f"boost_x{k}.jsonl"
        lines = sum(1 for _ in path.open(encoding="utf-8"))
        print(f"boost_x{k}: {lines} copy records -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
