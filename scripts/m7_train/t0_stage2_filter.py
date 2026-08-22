"""Filter T0-second label records down to their stage-2 (refined) rows.

The v1 objective admitted stage2-vs-stage2 pairs unconditionally, excluded
stage1-vs-stage1 entirely, and let cross pairs in only past a 2.36 margin at
0.74 weight (model_v1/metadata.json two_stage block).  train_ship cannot
express pair rules, so the clean subset that the frozen protocol CAN train on
is the stage-2 rows alone -- exactly the fully-trusted class, and the region
the leak lives in.

Reads labels.NNN.jsonl + labels.NNN.stages.jsonl side by side; writes records
whose runs[0].scores keep only stage-2 keys.  Offsets are preserved.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-rows", type=int, default=2,
                        help="drop positions with fewer stage-2 rows than this")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = dropped = 0
    with out.open("w", encoding="utf-8") as handle:
        for labels_path in args.labels:
            labels_path = pathlib.Path(labels_path)
            stages_path = labels_path.with_suffix("").with_suffix("")
            stages_path = labels_path.parent / (
                labels_path.name.replace(".jsonl", ".stages.jsonl"))
            with labels_path.open(encoding="utf-8") as lab, \
                 stages_path.open(encoding="utf-8") as stg:
                for lab_line, stg_line in zip(lab, stg):
                    if not lab_line.strip():
                        continue
                    record = json.loads(lab_line)
                    stages = json.loads(stg_line)
                    if record["offset"] != stages["offset"]:
                        raise SystemExit(
                            f"offset mismatch {record['offset']} vs "
                            f"{stages['offset']} in {labels_path.name}")
                    keep = {key for key, (stage, _score)
                            in stages["rows"].items() if stage == 2}
                    scores = record["runs"][0]["scores"]
                    filtered = {k: v for k, v in scores.items() if k in keep}
                    if len(filtered) < args.min_rows:
                        dropped += 1
                        continue
                    record["runs"] = [{"scores": filtered}]
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    written += 1
    print(f"wrote {written} positions (dropped {dropped} with <{args.min_rows} "
          f"stage-2 rows) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
