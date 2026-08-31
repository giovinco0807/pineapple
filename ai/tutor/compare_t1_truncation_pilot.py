"""Prepare and report the matched FL14 T1 truncation pilot.

Pass A and pass B label the same positions with different request ids.  The id
controls both the shared continuation streams and the Fantasyland pool draw.
Each pass therefore selects on one stream and is evaluated on the other.  The
lap-two truncated choice is frozen before either pass and is evaluated on both.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


B_SUFFIX = "::lap3b"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def prepare(source: Path, out_a: Path, out_b: Path, roots: int) -> None:
    records = read_jsonl(source)
    records.sort(key=lambda row: hashlib.sha256(f"lap3-t1-pilot/{row['id']}".encode()).digest())
    selected = records[:roots]
    if len(selected) != roots:
        raise ValueError(f"wanted {roots} roots, found {len(selected)}")

    pass_a: list[dict[str, Any]] = []
    pass_b: list[dict[str, Any]] = []
    for source_record in selected:
        record = dict(source_record)
        record.update(
            t2_samples=32,
            t3_samples=10,
            t4_draw_sample=60,
            pool_opponents=400,
            truncate_depth=None,
        )
        pass_a.append(record)
        second = dict(record)
        second["id"] = f"{record['id']}{B_SUFFIX}"
        pass_b.append(second)
    write_jsonl(out_a, pass_a)
    write_jsonl(out_b, pass_b)
    print(f"prepared {roots} matched roots")
    print(out_a)
    print(out_b)


def argmax(actions: list[dict[str, Any]]) -> str:
    # Label writers sort by key; strict max preserves that deterministic tie break.
    best = actions[0]
    for action in actions[1:]:
        if float(action["value"]) > float(best["value"]):
            best = action
    return str(best["action_key"])


def values(record: dict[str, Any]) -> dict[str, float]:
    return {str(action["action_key"]): float(action["value"]) for action in record["actions"]}


def mean_ci(values_: list[float]) -> dict[str, float]:
    n = len(values_)
    mean = sum(values_) / n
    variance = sum((value - mean) ** 2 for value in values_) / (n - 1)
    stderr = math.sqrt(variance / n)
    half = 1.959963984540054 * stderr
    return {"mean": mean, "stderr": stderr, "low": mean - half, "high": mean + half}


def report(truncated_path: Path, pass_a_path: Path, pass_b_path: Path, out: Path) -> None:
    pass_a = {str(row["id"]): row for row in read_jsonl(pass_a_path)}
    pass_b = {
        str(row["id"])[: -len(B_SUFFIX)]: row
        for row in read_jsonl(pass_b_path)
        if str(row["id"]).endswith(B_SUFFIX)
    }
    ids = sorted(set(pass_a) & set(pass_b))
    wanted = set(ids)
    truncated: dict[str, dict[str, Any]] = {}
    with truncated_path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            ident = str(record["id"])
            if ident in wanted:
                truncated[ident] = record
                if len(truncated) == len(wanted):
                    break
    ids = [ident for ident in ids if ident in truncated]
    if not ids:
        raise ValueError("pilot labels do not join to the truncated control")

    a_on_b: list[float] = []
    b_on_a: list[float] = []
    control_b: list[float] = []
    control_a: list[float] = []
    changed_a = changed_b = 0
    for ident in ids:
        control = argmax(truncated[ident]["actions"])
        choose_a = argmax(pass_a[ident]["actions"])
        choose_b = argmax(pass_b[ident]["actions"])
        va, vb = values(pass_a[ident]), values(pass_b[ident])
        if set(va) != set(vb) or control not in va:
            raise ValueError(f"{ident}: action sets differ")
        changed_a += choose_a != control
        changed_b += choose_b != control
        a_on_b.append(vb[choose_a] - vb[control])
        b_on_a.append(va[choose_b] - va[control])
        control_b.append(vb[control])
        control_a.append(va[control])

    crossfit = [(a + b) / 2.0 for a, b in zip(a_on_b, b_on_a)]
    result = {
        "schema": "ofc_fl14_t1_truncation_crossfit/v1",
        "roots": len(ids),
        "budgets": {"t2_samples": 32, "t3_samples": 10, "t4_draw_sample": 60, "pool_opponents": 400},
        "changed_from_lap2": {"pass_a": changed_a, "pass_b": changed_b},
        "full_a_choice_evaluated_on_b": mean_ci(a_on_b),
        "full_b_choice_evaluated_on_a": mean_ci(b_on_a),
        "symmetric_crossfit_delta_vs_lap2_choice": mean_ci(crossfit),
        "lap2_choice_value": mean_ci([(a + b) / 2.0 for a, b in zip(control_a, control_b)]),
        "interpretation": (
            "positive means a full-playout-selected T1 action beats the frozen lap2 "
            "truncated action on an independent stream; this is a teacher-method pilot, "
            "not the final whole-hand promotion gate"
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--out-a", type=Path, required=True)
    prep.add_argument("--out-b", type=Path, required=True)
    prep.add_argument("--roots", type=int, default=60)
    rep = sub.add_parser("report")
    rep.add_argument("--truncated", type=Path, required=True)
    rep.add_argument("--pass-a", type=Path, required=True)
    rep.add_argument("--pass-b", type=Path, required=True)
    rep.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.out_a, args.out_b, args.roots)
    else:
        report(args.truncated, args.pass_a, args.pass_b, args.out)


if __name__ == "__main__":
    main()
