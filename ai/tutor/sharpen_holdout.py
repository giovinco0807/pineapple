"""Re-price the audit holdout at high particle counts, to separate model error
from label error.

The correction work is graded against `sel_means`, which come from the mining
race at 16/48/128 cumulative particles.  Those means carry a standard error
around 2.3 points while the median root's best-vs-second gap is 1.2, so on
most holdout roots the label "the referee's best" is not actually resolved --
a net can be marked wrong for preferring a candidate the referee cannot
distinguish from its own favourite.

This re-duels a short list per root (the current top six plus each net's
served pick, so every compared pick is priced) at 4 CRN batches x 256 = 1,024
rollouts, which should bring the per-candidate SE near 0.5 and the paired
difference SE well below it.  It is the pre-registered falsifier for the claim
that the 120-dim feature set is no longer the binding constraint: sharpen the
labels, re-score the SAME checkpoints without retraining, and see whether the
misses were noise.

Seed band 890,000,000 + holdout_index*10 + batch -- unused by the miner
(850/860 select and score, 870/880 the race) and by the fleet, so nothing here
re-draws a future the graded runs already saw.

Checkpointed per root: a killed run resumes from the shards, which matters
because this is a four-hour job on a box that is also busy.

Usage:
    python -m ai.tutor.sharpen_holdout --plan C:/tmp/sharpen_plan.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = Path("D:/ofc_data/hu")


def duel(binary: Path, own: str, fl_ev: Path, cards: str, keys_file: Path,
         rollouts: int, seed: int, out: Path):
    done = subprocess.run(
        [str(binary), "--fl-t0-deep", "--t0-cards", cards,
         "--rollouts", str(rollouts), "--self-play-seed", str(seed),
         "--arm-a-own", own, "--fl-ev-config", str(fl_ev),
         "--t0-keys", str(keys_file), "--output", str(out)],
        capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"seed {seed}: {done.stderr[-500:]}")
    return {r["key"]: r["scores"]
            for r in (json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, default=Path("C:/tmp/sharpen_plan.json"))
    ap.add_argument("--models-dir", type=Path, default=D / "models_ship_20260830/own_lap4")
    ap.add_argument("--binary", type=Path,
                    default=REPO / "ai/rust_solver/target_gate/release/t4_first_exact.exe")
    ap.add_argument("--fl-ev-config", type=Path, default=REPO / "ai/config/fl_ev.json")
    ap.add_argument("--out-dir", type=Path, default=D / "fl_evalfix")
    ap.add_argument("--work", type=Path, default=Path("C:/tmp/sharpen_work"))
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--rollouts", type=int, default=256)
    ap.add_argument("--seed-base", type=int, default=890_000_000)
    # Slice for fleet sharding.  The seed is keyed on the plan's own `index`,
    # which is the position in the canonical holdout ordering and travels with
    # the plan file -- so a shard computes byte-identical futures to the same
    # roots run locally, and the two merge without a reconciliation step.
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--shard-jsonl", type=Path, default=None,
                    help="rewrite this JSONL (one line per finished root) after "
                         "each root, for a fleet worker to publish")
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    shards = args.out_dir / "_sharp"
    shards.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    own = ",".join(str(args.models_dir / n) for n in ("t0.bin", "t1.bin", "t2.bin"))
    started = time.time()
    order = sorted(plan.items(), key=lambda kv: kv[1]["index"])
    stop = len(order) if args.count is None else min(args.start + args.count, len(order))
    order = order[args.start:stop]
    done_already = sum(1 for rid, _ in order if (shards / f"{rid}.json").exists())
    print(f"{len(order)} roots [{args.start},{stop}), {done_already} already sharpened",
          flush=True)

    def publish() -> None:
        if args.shard_jsonl is None:
            return
        with open(args.shard_jsonl, "w", encoding="utf-8") as handle:
            for rid, _ in order:
                shard = shards / f"{rid}.json"
                if shard.exists():
                    payload = {"root": rid,
                               "rows": json.loads(shard.read_text(encoding="utf-8"))}
                    handle.write(json.dumps(payload) + "\n")

    for position, (rid, spec) in enumerate(order):
        shard = shards / f"{rid}.json"
        if shard.exists():
            continue
        keys_file = args.work / f"{rid}_keys.txt"
        keys_file.write_text("\n".join(spec["keys"]) + "\n", encoding="utf-8")
        pooled: dict[str, list[float]] = {k: [] for k in spec["keys"]}
        for batch in range(args.batches):
            seed = args.seed_base + spec["index"] * 10 + batch
            got = duel(args.binary, own, args.fl_ev_config, spec["cards"], keys_file,
                       args.rollouts, seed, args.work / f"{rid}_b{batch}.jsonl")
            for key in spec["keys"]:
                pooled[key].extend(got[key])
        rows = []
        for key, scores in pooled.items():
            n = len(scores)
            mean = sum(scores) / n
            var = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
            rows.append(dict(root=rid, key=key, mean=mean,
                             se=math.sqrt(var / n), n=n, scores=scores))
        shard.write_text(json.dumps(rows), encoding="utf-8")
        publish()
        elapsed = time.time() - started
        remaining = len(order) - position - 1
        rate = (position + 1 - done_already) / max(elapsed, 1e-9)
        print(f"[{position + 1}/{len(order)}] {rid} {len(rows)} candidates "
              f"se {min(r['se'] for r in rows):.2f}-{max(r['se'] for r in rows):.2f} "
              f"| eta {remaining / max(rate, 1e-9) / 60:.0f} min", flush=True)

    # Assemble once every shard is present, so a partial run leaves no
    # half-written jsonl for the re-scoring step to read as complete.
    publish()
    if args.count is None and args.start == 0 and all(
            (shards / f"{rid}.json").exists() for rid, _ in order):
        target = args.out_dir / "holdout_sharp.jsonl"
        with open(target, "w", encoding="utf-8") as handle:
            for rid, _ in order:
                for row in json.loads((shards / f"{rid}.json").read_text(encoding="utf-8")):
                    handle.write(json.dumps({k: row[k] for k in ("root", "key", "mean", "se", "n")}) + "\n")
        print(f"wrote {target} ({time.time() - started:.0f}s)", flush=True)
    else:
        print("incomplete; rerun to resume from the shards", flush=True)


if __name__ == "__main__":
    main()
