"""Split a vs-FL teacher into per-opp_count datasets.

One model per Fantasyland count instead of a shared model with a one-hot:
the four opponents are different games (a 17-card FL is a different beast
from a 14-card one), and the shared net was visibly worst on the rare
counts.  Rows do not store opp_count, but roots are seed-deterministic, so
the same alignment replay used by every re-encoder recovers it per row.

Usage:
    python -m ai.tutor.split_teacher_by_count --street t2 \
        --in-dir D:/ofc_data/t2_vs_fl_teacher_v3
Writes <in-dir>_c14 .. _c17 with identical schema.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ai.tutor.reencode_vs_fl_teacher import SAMPLERS, chunk_rows

SPLITS = ("fit", "dev", "test")
COUNTS = (14, 15, 16, 17)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--street", choices=["t0", "t1", "t2"], required=True)
    parser.add_argument("--in-dir", type=Path, required=True)
    args = parser.parse_args()
    sampler = SAMPLERS[args.street]

    from ai.tutor.generate_t0_vs_fl_teacher import split_of as split_t0
    from ai.tutor.generate_t1_vs_fl_teacher import split_of as split_t1
    from ai.tutor.generate_t2_vs_fl_teacher import split_of as split_t2
    split_of = {"t0": split_t0, "t1": split_t1, "t2": split_t2}[args.street]

    manifest = json.loads((args.in_dir / "manifest.json").read_text(encoding="utf-8"))
    seed, roots_total = manifest["seed"], manifest["roots"]
    feature_size = manifest["feature_size"]

    data = {name: np.load(args.in_dir / f"{name}.npz") for name in SPLITS}
    cursors = {name: 0 for name in SPLITS}
    buckets = {
        count: {name: {"x": [], "y": [], "j": []} for name in SPLITS}
        for count in COUNTS
    }
    for index in range(seed, seed + roots_total):
        root = sampler(index)
        bucket = split_of(index)
        count = root["opp_count"]
        n_actions = len(chunk_rows(args.street, root))
        cursor = cursors[bucket]
        for key in ("x", "y"):
            buckets[count][bucket][key].append(
                data[bucket][key][cursor:cursor + n_actions]
            )
        buckets[count][bucket]["j"].append(
            data[bucket]["jokers"][cursor:cursor + n_actions]
        )
        cursors[bucket] = cursor + n_actions
    for name in SPLITS:
        if cursors[name] != data[name]["y"].shape[0]:
            raise SystemExit(
                f"{name}: consumed {cursors[name]} of {data[name]['y'].shape[0]} "
                "rows -- alignment bug"
            )

    for count in COUNTS:
        out_dir = Path(str(args.in_dir) + f"_c{count}")
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = {}
        for name in SPLITS:
            parts = buckets[count][name]
            x = (np.concatenate(parts["x"]) if parts["x"]
                 else np.zeros((0, feature_size), dtype=np.float32))
            y = (np.concatenate(parts["y"]) if parts["y"]
                 else np.zeros((0,), dtype=np.float32))
            j = (np.concatenate(parts["j"]) if parts["j"]
                 else np.zeros((0,), dtype=np.int8))
            np.savez_compressed(out_dir / f"{name}.npz", x=x, y=y, jokers=j)
            rows[name] = int(y.shape[0])
        sub_manifest = dict(manifest)
        sub_manifest["schema"] = manifest["schema"] + f"+count_{count}_only"
        sub_manifest["opp_count"] = count
        sub_manifest["splits"] = {name: {"rows": rows[name]} for name in SPLITS}
        (out_dir / "manifest.json").write_text(
            json.dumps(sub_manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(count, rows)


if __name__ == "__main__":
    main()
