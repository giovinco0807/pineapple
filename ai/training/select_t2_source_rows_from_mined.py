"""Select original T2 source rows referenced by mined rows.

Use this after cheap source-model mining to build a smaller JSONL for
``run_t2_exact_oracle``.  ``group_id`` is treated as the source row index in the
normalized source dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def select(args: argparse.Namespace) -> dict:
    source_rows = list(iter_jsonl(Path(args.source)))
    mined_rows = list(iter_jsonl(Path(args.rows)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    selected: dict[int, dict] = {}
    for mined in mined_rows:
        if args.dataset and str(mined.get("dataset")) != args.dataset:
            continue
        if args.min_ev_loss is not None and float(mined.get("ev_loss") or 0.0) < float(args.min_ev_loss):
            continue
        group_id = int(mined["group_id"])
        if group_id < 0 or group_id >= len(source_rows):
            continue
        previous = selected.get(group_id)
        if previous is None or float(mined.get("ev_loss") or 0.0) > float(previous.get("ev_loss") or 0.0):
            selected[group_id] = mined
        if args.limit and len(selected) >= args.limit:
            break

    order = sorted(selected, key=lambda idx: (-float(selected[idx].get("ev_loss") or 0.0), idx))
    with output.open("w", encoding="utf-8") as out:
        for out_index, source_idx in enumerate(order):
            row = dict(source_rows[source_idx])
            mined = selected[source_idx]
            row["original_source_index"] = int(source_idx)
            row["mined_rank"] = int(out_index)
            row["mined_ev_loss"] = float(mined.get("ev_loss") or 0.0)
            row["mined_reasons"] = list(mined.get("reasons") or [])
            row["mined_choices"] = mined.get("choices")
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "source": str(args.source),
        "rows": str(args.rows),
        "output": str(output),
        "dataset": args.dataset,
        "min_ev_loss": args.min_ev_loss,
        "limit": args.limit,
        "source_rows": len(source_rows),
        "mined_rows": len(mined_rows),
        "selected_rows": len(order),
        "selected_source_indices": order[: min(len(order), 50)],
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--min-ev-loss", type=float)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(select(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
