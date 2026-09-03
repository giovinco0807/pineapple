"""Sharp T0-BUTTON labels -> fit/dev arrays for the T4F1 evaluator trainer.

Downstream half of `t0_btn_label.py`.  A label record is one root: the
Button's five cards, BB's placed board, and `field` -- the referee's mean
(champion continuations, 128 CRN rollouts) for every opening in the labelled
field, evaluator top-32 U ranker top-16 U one per top-row stratum U the
served move, about 40 of the 232.  Dev roots carry a second independent
pass in `field_pass2`.

What this writes is what `train_t4_first_evaluator` reads: fit.npz / dev.npz
with x (the 207-dim serving vector, float32), y (referee mean; on dev the
mean of both passes), roots (the root's line index in roots_all.jsonl,
int64) and jokers; dev also carries y1/y2 so the dev regret can report a
label-noise standard error (docs/t2_width128_20260831.md §6.5, the
instrument that decided the T2 and T1 campaigns).  Rows are the LABELLED
openings only: the ~190 openings outside the field are unreachable through
the ranker fence, and anchoring them to the stale scores they came from
would teach the staleness.  Whether a retrained net's argmax escapes the
field is `t0_btn_sharp_audit.py`'s question, asked over all 232 encodings.

Encoding is `encode_t0_material.encode` -- the serving encoder at the serve
spec (joint 200 x 32 under the node seed model-rank/t0) -- never a second
implementation.  Each root is first ranked with the binary's own
`--hu-t0-deep --rollouts 0` (the audit's and the labeler's enumeration),
which supplies the key spelling, the ranker rank and the served score
column; every encoded row is then checked against that column in float32,
and nothing is written if the largest gap exceeds --parity-tol.

Split: a record with `field_pass2` is dev (the labeler's --passes 2), the
rest fit; --split hash is the fallback for single-pass material.  Keys are
matched joker-blind (X1/X2 are one card; the audit renumbers by position).
Error records ({"id","index","error"}) are skipped and counted.

Usage:
    python -m ai.tutor.build_t0_btn_sharp --labels runs/btnlabel1/mine \\
        --out-dir D:/ofc_data/hu/t0_btn_sharp
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ai.tutor.encode_fl_material import forward, load_bin, rows_of
from ai.tutor.encode_t0_material import (
    D, FEATURE_SIZE, REPO, SERVE_JOINT_SAMPLES, SERVE_NODE_SEED, encode,
    opening_key_rows, rank_root)
from ai.tutor.t0_btn_label import joker_blind

OPENINGS = 232
SEAT_BTN = 1
SHIP_BIN = "hu/t0_btn.bin"


def label_files(spec: Path) -> list[Path]:
    """One jsonl, a directory of them, or a glob -- fleet shards land as
    `runs/<job>/mine/*.jsonl`."""
    if spec.is_dir():
        return sorted(spec.glob("*.jsonl"))
    if spec.exists():
        return [spec]
    return [Path(p) for p in sorted(glob.glob(str(spec)))]


def load_labels(spec: Path) -> tuple[list[dict], Counter]:
    """Every usable label record (first copy of an id wins); what was skipped
    and why is counted, not hidden."""
    files = label_files(spec)
    if not files:
        raise SystemExit(f"no label files under {spec}")
    records, skipped, seen = [], Counter(), set()
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped["truncated_line"] += 1  # a preempted worker's last line
                continue
            if "error" in rec:
                skipped["error_record"] += 1
                continue
            if not rec.get("field") or "index" not in rec or not rec.get("opp_board") \
                    or not rec.get("served"):
                skipped["malformed_record"] += 1
                continue
            if rec["id"] in seen:
                skipped["duplicate_id"] += 1
                continue
            seen.add(rec["id"])
            records.append(rec)
    print(f"labels: {len(records)} roots from {len(files)} file(s); "
          f"skipped {dict(skipped) or 'none'}", flush=True)
    return records, skipped


def is_dev(rec: dict, split: str, salt: str, dev_pct: int) -> bool:
    if split == "two-pass":
        return bool(rec.get("field_pass2"))
    digest = hashlib.sha256(f"{salt}{rec['id']}".encode()).hexdigest()
    return int(digest[:8], 16) % 100 < dev_pct


def match_field(rec: dict, entries: list[dict]) -> list[tuple[str, dict]]:
    """(label key, ranked entry) for every labelled opening, joker-blind."""
    exact = {e["key"]: e for e in entries}
    blind: dict[str, list[dict]] = {}
    for entry in entries:
        blind.setdefault(joker_blind(entry["key"]), []).append(entry)
    matched = []
    for key in rec["field"]:
        entry = exact.get(key)
        if entry is None:
            candidates = blind.get(joker_blind(key), [])
            if not candidates:
                raise AssertionError(
                    f"{rec['id']}: labelled opening {key} is not one the serve ranked")
            entry = candidates[0]
        matched.append((key, entry))
    return matched


def pass2_values(rec: dict, keys: list[str]) -> list[float] | None:
    """The second pass's mean for each labelled key, or None on a fit root."""
    second = rec.get("field_pass2")
    if not second:
        return None
    blind = {joker_blind(k): v for k, v in second.items()}
    values = []
    for key in keys:
        if key in second:
            values.append(float(second[key]))
        elif joker_blind(key) in blind:
            values.append(float(blind[joker_blind(key)]))
        else:
            raise AssertionError(f"{rec['id']}: pass 2 lacks the opening {key}")
    return values


def count_jokers(cards: str, opp_board: str) -> int:
    visible = cards.split(",") + [c for row in rows_of(opp_board) for c in row]
    return sum(c.startswith("X") for c in visible)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True,
                    help="a label jsonl, a directory of them, or a glob")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, default=D / "models_ship_20260903",
                    help="the bundle the labels were enumerated under "
                         "(its hu/t0_btn.bin is the parity oracle)")
    ap.add_argument("--binary", type=Path,
                    default=REPO / "ai/rust_solver/target/release/t4_first_exact.exe")
    ap.add_argument("--fl-ev-config", type=Path, default=None,
                    help="defaults to <models-dir>/fl_ev.json")
    ap.add_argument("--ranks-dir", type=Path, default=None,
                    help="cache of per-root rankings (default <out-dir>/ranks); "
                         "share it only across runs of the same bundle")
    ap.add_argument("--topk", type=int, default=4, help="ranker fence the serve uses at Button")
    ap.add_argument("--split", choices=("two-pass", "hash"), default="two-pass",
                    help="two-pass: records with field_pass2 are dev; hash: --dev-pct by id hash")
    ap.add_argument("--dev-pct", type=int, default=10)
    ap.add_argument("--split-salt", default="t0-btn-sharp-v1/")
    ap.add_argument("--parity-tol", type=float, default=1e-4,
                    help="max |py - rust| allowed over every row (observed 1.1e-5 on 110k rows)")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--rank-workers", type=int, default=4)
    args = ap.parse_args()
    fl_ev = args.fl_ev_config or args.models_dir / "fl_ev.json"
    ranks_dir = args.ranks_dir or args.out_dir / "ranks"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranks_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    records, skipped = load_labels(args.labels)
    if not records:
        raise SystemExit("no usable label records")

    # Ranking is one short process per root; the binary threads the 232
    # encodings, so a few side by side fill the box.
    def ranked(rec):
        wid = rec["id"].replace("/", "_")
        return rec["id"], rank_root(args.binary, args.models_dir, fl_ev, rec["cards"], SEAT_BTN,
                                    rec["opp_board"], SERVE_JOINT_SAMPLES, args.topk,
                                    ranks_dir / f"{wid}.jsonl")

    ranks: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=args.rank_workers) as pool:
        for done, (rid, entries) in enumerate(pool.map(ranked, records), 1):
            ranks[rid] = entries
            if done % 100 == 0:
                print(f"  ranked {done}/{len(records)} roots ({time.time() - started:.0f}s)",
                      flush=True)
    print(f"ranked {len(ranks)} roots in {time.time() - started:.0f}s", flush=True)

    pairs, rows = [], []
    field_sizes: Counter = Counter()
    roots_per_split: Counter = Counter()
    served_missing = 0
    for rec in records:
        entries = ranks[rec["id"]]
        if len(entries) != OPENINGS:
            raise AssertionError(f"{rec['id']}: {len(entries)} openings ranked, expected {OPENINGS}")
        matched = match_field(rec, entries)
        keys = [key for key, _ in matched]
        first = [float(rec["field"][key]) for key in keys]
        second = pass2_values(rec, keys)
        split = "dev" if is_dev(rec, args.split, args.split_salt, args.dev_pct) else "fit"
        roots_per_split[split] += 1
        field_sizes[len(keys)] += 1
        opp_rows = rows_of(rec["opp_board"])
        served_blind = joker_blind(rec["served"])
        jokers = count_jokers(rec["cards"], rec["opp_board"])
        sources = rec.get("sources", {})
        served_seen = False
        for (label_key, entry), v1, v2 in zip(matched, first, second or first):
            pairs.append((opening_key_rows(entry["key"]), opp_rows))
            served = joker_blind(entry["key"]) == served_blind
            served_seen |= served
            rows.append(dict(
                split=split, id=rec["id"], index=int(rec["index"]), key=entry["key"],
                label_key=label_key, y=(v1 + v2) / 2.0 if second else v1, y1=v1,
                y2=v2 if second else None, passes=2 if second else 1, served=served,
                ranker_rank=entry.get("ranker_rank") or 0, own_rank=int(entry["rank"]),
                own_score=float(entry["score"]), jokers=jokers,
                sources=sources.get(label_key, [])))
        served_missing += not served_seen
    print(f"rows: {len(rows)} labelled openings over {len(records)} roots "
          f"(fit {roots_per_split['fit']} roots, dev {roots_per_split['dev']} roots); "
          f"field sizes {dict(sorted(field_sizes.items()))}", flush=True)
    if served_missing:
        print(f"WARNING {served_missing} roots have no served opening in their field", flush=True)

    x = encode(args.binary, fl_ev, pairs, SERVE_JOINT_SAMPLES, SERVE_NODE_SEED, args.batch)
    print(f"encoded {len(x)} rows in {time.time() - started:.0f}s "
          f"(joint {SERVE_JOINT_SAMPLES}, seed {SERVE_NODE_SEED})", flush=True)

    # Parity on every row: the vector must reproduce the score the serving
    # path published for the same placement, or this file is not the
    # serving encoder and nothing downstream may be trusted.
    bin_path = args.models_dir / SHIP_BIN
    mean, std, mats = load_bin(bin_path)
    if len(mean) != FEATURE_SIZE:
        raise SystemExit(f"{bin_path} reads {len(mean)} dims, this encoder writes {FEATURE_SIZE}")
    py = forward(mats, (x - mean) / std)
    rust = np.asarray([row["own_score"] for row in rows], np.float32)
    delta = np.abs(py - rust)
    worst = int(delta.argmax())
    print(f"PARITY max|py-rust| {delta.max():.3e}  mean {delta.mean():.3e}  "
          f"(worst row {rows[worst]['id']} {rows[worst]['key']}: "
          f"py {py[worst]:.5f} rust {rust[worst]:.5f})", flush=True)
    if delta.max() > args.parity_tol:
        order = np.argsort(-delta)[:10]
        (args.out_dir / "parity_failed.json").write_text(json.dumps(
            [dict(id=rows[i]["id"], key=rows[i]["key"], py=float(py[i]), rust=float(rust[i]))
             for i in order], indent=2), encoding="utf-8")
        raise SystemExit(f"PARITY FAILED (tol {args.parity_tol:g}) -- the Python encoder is not "
                         f"the serving encoder; nothing written but parity_failed.json")

    manifest = dict(
        labels=str(args.labels), models_dir=str(args.models_dir), bin=str(bin_path),
        feature_size=FEATURE_SIZE, joint_samples=SERVE_JOINT_SAMPLES, joint_seed=SERVE_NODE_SEED,
        fence=["ranker_rank", args.topk], split=args.split,
        roots=len(records), rows=len(rows), skipped=dict(skipped),
        field_size_histogram={str(k): v for k, v in sorted(field_sizes.items())},
        parity_max_abs=float(delta.max()), parity_mean_abs=float(delta.mean()),
        parity_tol=args.parity_tol, splits={})
    for split in ("fit", "dev"):
        chosen = [i for i, row in enumerate(rows) if row["split"] == split]
        if not chosen:
            print(f"WARNING no {split} roots; {split}.npz not written", flush=True)
            continue
        sub = [rows[i] for i in chosen]
        arrays = dict(
            x=x[chosen],
            y=np.asarray([r["y"] for r in sub], np.float32),
            roots=np.asarray([r["index"] for r in sub], np.int64),
            jokers=np.asarray([r["jokers"] for r in sub], np.int8),
            passes=np.asarray([r["passes"] for r in sub], np.int8),
            ranker_rank=np.asarray([r["ranker_rank"] for r in sub], np.int16),
            own_rank=np.asarray([r["own_rank"] for r in sub], np.int16),
            own_score=np.asarray([r["own_score"] for r in sub], np.float32),
            served=np.asarray([r["served"] for r in sub], bool))
        two_pass = all(r["passes"] == 2 for r in sub)
        if split == "dev":
            if two_pass:
                arrays["y1"] = np.asarray([r["y1"] for r in sub], np.float32)
                arrays["y2"] = np.asarray([r["y2"] for r in sub], np.float32)
            else:
                print("WARNING dev holds single-pass roots: y1/y2 not written, "
                      "so no label-noise SE will be available", flush=True)
        np.savez_compressed(args.out_dir / f"{split}.npz", **arrays)
        with open(args.out_dir / f"{split}_rows.jsonl", "w", encoding="utf-8") as handle:
            for position, row in enumerate(sub):
                handle.write(json.dumps(dict(row=position, **{k: v for k, v in row.items()
                                                              if k != "split"}),
                                        ensure_ascii=False) + "\n")
        roots_here = len({r["index"] for r in sub})
        manifest["splits"][split] = dict(roots=roots_here, rows=len(sub), two_pass=two_pass)
        print(f"wrote {split}.npz x{arrays['x'].shape} ({roots_here} roots"
              f"{', y1/y2' if 'y1' in arrays else ''}) and {split}_rows.jsonl", flush=True)
    manifest["elapsed_seconds"] = time.time() - started
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"BUILD_DONE roots={len(records)} rows={len(rows)} "
          f"errors_skipped={skipped['error_record']} parity_max={delta.max():.3e} "
          f"({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
