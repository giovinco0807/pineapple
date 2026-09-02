"""The per-street, per-seat leak map for the joker HU chain.

For every decision the champion actually makes in a hand, this asks one
question: how much is the move it served worth against the best move a deep
referee can find from the same state?  The answer per square is a rate (how
often the served move is confirmably beaten) and a size (by how much), and
their product is the square's leak in points per hand -- which is what says
where mining should go next.

Squares are (street, seat) where a MODEL decides: T0/T1/T2 for both seats and
T3-BB.  T3-BTN is closed-form and T4 is exact, so their leak is zero by
construction and they are not measured.

Three stages per root, and the seeds that drive them are kept apart:

  enumerate  `--rollouts 0` lists the legal placements with the champion's
             own scores.  The served move comes from the trace, not from a
             re-derivation, so what is graded is what was actually played.

  race       the candidates are played out with the DISTILLED continuations
             (2.65x cheaper, verified to preserve the ordering) and cut down
             over rounds.  This stage only nominates a challenger.

  verdict    the served move and the challenger are re-priced with the
             CHAMPION's own continuations on a DISJOINT seed band.  Selecting
             and scoring on the same rollouts keeps whichever move those
             particular futures flattered -- measured at 55% shrinkage on
             this codebase -- so the margin reported is earned on futures the
             race never saw.

A root where the race nominates the served move itself costs nothing further:
its gap is zero and no verdict is needed.

The verdict is paired: every batch prices both moves on the same rollout
seeds, so a batch's difference of means IS the paired difference, and the
standard error is taken across batches of that difference -- never across the
two moves independently, which would throw away the common-random-numbers
correlation that makes the comparison sharp.

Usage (one square, a slice of roots):
    python -m ai.tutor.hu_leak_map --square t1_bb --start 0 --count 50 \
        --traces D:/ofc_data/hu/fast/all_traces --out leak_t1_bb.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SQUARES = {
    "t0_bb": (0, 0), "t0_btn": (0, 1),
    "t1_bb": (1, 0), "t1_btn": (1, 1),
    "t2_bb": (2, 0), "t2_btn": (2, 1),
    "t3_bb": (3, 0),
}
# Seed bands.  Race and verdict must not share one, and every root gets its
# own seeds within a band: an audit found the first draft priced every root
# in every square on the same few deck permutations, which correlates roots
# and understates the across-root error.  ROOT_STRIDE leaves room for the
# rounds/batches of one root; the bands are a billion apart.
SELECT_BAND = 2_100_000_001
SCORE_BAND = 3_300_000_001
NULL_BAND = 4_400_000_001
ROOT_STRIDE = 100


def model_args(models: Path, fast):
    names = ("t0_bb.bin", "t0_btn.bin", "t1_bb.bin", "t1_btn.bin",
             "t2_bb.bin", "t2_btn.bin", "t3_bb.bin", "t3_btn.bin")
    hu = ",".join(str(models / "hu" / n) for n in names)
    rk = ",".join(str(models / "rankers" / n) for n in names)
    own = ",".join(str(models / "own_lap4" / n) for n in ("t0.bin", "t1.bin", "t2.bin"))
    out = ["--hu-a-models", hu, "--hu-b-models", hu,
           "--hu-a-rankers", rk, "--hu-b-rankers", rk, "--hu-topk", "4",
           "--hu-a-t0-policy", str(models / "policy.bin"),
           "--hu-b-t0-policy", str(models / "policy.bin"),
           "--hu-t0-policy-topk", "8",
           "--serve-joint-samples", "200", "--serve-joint-samples-b", "200",
           "--arm-a-own", own, "--arm-b-own", own,
           "--fl-ev-config", str(REPO / "ai/config/fl_ev.json")]
    if fast is not None:
        out += ["--hu-fast-nets", str(fast)]
    return out


def board_key(rows) -> str:
    """A board in the enumeration's spelling: rows sorted by card name."""
    return "|".join(",".join(sorted(r)) for r in rows)


class Referee:
    def __init__(self, binary: Path, models: Path, fast: Path, work: Path):
        self.binary, self.models, self.fast, self.work = binary, models, fast, work
        self.calls = 0
        self.seconds = 0.0

    def run(self, hand_file: Path, street: int, seat: int, rollouts: int,
            seed: int, keys, use_fast: bool, tag: str):
        out = self.work / f"{tag}.jsonl"
        cmd = [str(self.binary), "--hu-match", "--hu-deep-replay"]
        cmd += model_args(self.models, self.fast if use_fast else None)
        cmd += ["--input", str(hand_file), "--replay-hand", "0",
                "--replay-street", str(street), "--replay-seat", str(seat),
                "--rollouts", str(rollouts), "--self-play-seed", str(seed),
                "--output", str(out)]
        if keys is not None:
            # deep-replay reads only the inline form of the key list.
            cmd += ["--t0-keys-inline", ";".join(keys)]
        started = time.time()
        done = subprocess.run(cmd, capture_output=True, text=True)
        self.calls += 1
        self.seconds += time.time() - started
        if done.returncode != 0:
            raise RuntimeError(f"{tag}: {done.stderr[-600:]}")
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        return {r["key"]: r["mean"] for r in rows}


def grade_root(ref: Referee, hand: dict, square: str, index: int, args) -> dict:
    street, seat = SQUARES[square]
    step = next((s for s in hand["steps"]
                 if s["street"] == street and s["seat"] == seat), None)
    if step is None:
        return {"index": index, "square": square, "skipped": "no such decision"}
    served = board_key(step["board"])

    # The hand is written on its own so `--replay-hand 0` is unambiguous: the
    # traces are sharded and hand numbers repeat across shards.
    hand_file = ref.work / f"h_{square}_{index}.jsonl"
    solo = dict(hand)
    solo["hand"] = 0
    hand_file.write_text(json.dumps(solo) + "\n", encoding="utf-8")

    scores = ref.run(hand_file, street, seat, 0, SELECT_BAND, None, True,
                     f"enum_{square}_{index}")
    if args.null_calibration:
        # The background: the served move priced against ITSELF on two
        # disjoint seed bands.  Whatever `confirmed` rate this produces is
        # the instrument's false-positive rate, to be printed beside the map.
        diffs = []
        for b in range(args.batches):
            a = ref.run(hand_file, street, seat, args.verdict_rollouts,
                        SCORE_BAND + index * ROOT_STRIDE + b, [served], False,
                        f"nullA{b}_{square}_{index}")
            z = ref.run(hand_file, street, seat, args.verdict_rollouts,
                        NULL_BAND + index * ROOT_STRIDE + b, [served], False,
                        f"nullB{b}_{square}_{index}")
            diffs.append(z[served] - a[served])
        record = {"index": index, "square": square, "served": served,
                  "null_calibration": True, "challenger": served}
        record.update(finish_verdict(np.asarray(diffs, float), args))
        return record
    if served not in scores:
        return {"index": index, "square": square,
                "skipped": f"served key absent: {served}"}
    ranked = sorted(scores, key=lambda k: -scores[k])
    served_model_rank = ranked.index(served) + 1

    raced = ranked[:args.t0_width] if street == 0 else ranked
    capped = street == 0 and len(ranked) > args.t0_width
    alive = list(raced)
    if served not in alive:
        alive = alive[:-1] + [served]

    for r, (keep, rollouts) in enumerate([(args.keep1, args.round1),
                                          (2, args.round2)]):
        race = ref.run(hand_file, street, seat, rollouts,
                       SELECT_BAND + index * ROOT_STRIDE + (r + 1), alive, True,
                       f"race{r}_{square}_{index}")
        alive = sorted(race, key=lambda k: -race[k])[:keep]
    challenger = alive[0]

    record = {
        "index": index, "square": square, "served": served,
        "served_model_rank": served_model_rank, "candidates": len(scores),
        "raced": len(raced), "t0_capped": capped, "challenger": challenger,
    }
    if challenger == served:
        record.update(margin=0.0, se=0.0, confirmed=False, verdict_batches=0)
        return record

    pair = [served, challenger]
    diffs = []
    for b in range(args.batches):
        got = ref.run(hand_file, street, seat, args.verdict_rollouts,
                      SCORE_BAND + index * ROOT_STRIDE + b, pair, False,
                      f"verdict{b}_{square}_{index}")
        diffs.append(got[challenger] - got[served])
    record.update(finish_verdict(np.asarray(diffs, float), args))
    return record


def finish_verdict(d: np.ndarray, args) -> dict:
    """Per-root margin, its paired SE, and a *descriptive* confirmation flag.

    The headline leak is the mean margin over ALL graded roots (computed by
    the summary, zeros and negatives included); that estimator is unbiased
    and does not need per-root precision.  `confirmed` is reported for
    interpretability only, and it uses the t quantile for the actual number
    of batches: with 4 batches the empirical SE has 3 degrees of freedom and
    `margin > 3*se` passes 2.9% of true-zero roots, not 0.13% -- the same
    fat-tail failure the other track corrected on 8/22.
    """
    from scipy.stats import t as student_t
    margin = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    t_crit = float(student_t.ppf(1 - args.alpha, len(d) - 1)) if len(d) > 1 else float("inf")
    return {"margin": margin, "se": se, "verdict_batches": int(len(d)),
            "t_crit": t_crit,
            "confirmed": bool(len(d) > 1 and margin > t_crit * se and margin > args.floor)}


def load_hands(traces: Path, wanted: set[int]):
    hands, seen = {}, 0
    for shard in sorted(traces.glob("*.jsonl")):
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                if seen in wanted:
                    hands[seen] = json.loads(line)
                seen += 1
        if len(hands) == len(wanted):
            break
    return hands, seen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--square", required=True, choices=sorted(SQUARES))
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--stride", type=int, default=97,
                    help="hand slots are stride apart so shards of one square "
                         "never draw the same hand and roots stay spread")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("leakwork"))
    ap.add_argument("--models", type=Path,
                    default=Path("D:/ofc_data/hu/models_ship_20260903"))
    ap.add_argument("--fast", type=Path,
                    default=Path("D:/ofc_data/hu/fast/bins_full"))
    ap.add_argument("--binary", type=Path,
                    default=REPO / "ai/rust_solver/target/release/t4_first_exact.exe")
    ap.add_argument("--t0-width", type=int, default=40,
                    help="street 0 races the model's top N of 232; logged, not silent")
    ap.add_argument("--keep1", type=int, default=6)
    ap.add_argument("--round1", type=int, default=24)
    ap.add_argument("--round2", type=int, default=96)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--verdict-rollouts", type=int, default=192)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.001,
                    help="one-sided level for the descriptive `confirmed` flag; "
                         "applied through the t quantile for --batches")
    ap.add_argument("--null-calibration", action="store_true",
                    help="price the served move against itself on two disjoint "
                         "bands instead of racing; measures the false-positive rate")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.out.parent != Path(""):
        args.out.parent.mkdir(parents=True, exist_ok=True)

    wanted = [(args.start + i) * args.stride for i in range(args.count)]
    hands, seen = load_hands(args.traces, set(wanted))
    missing = [w for w in wanted if w not in hands]
    if missing:
        print(f"WARNING: {len(missing)} of {len(wanted)} hand slots are past the "
              f"corpus end ({seen} hands); grading {len(hands)}", flush=True)

    ref = Referee(args.binary, args.models, args.fast, args.work)
    errors = 0
    done = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for slot in wanted:
            if slot not in hands:
                continue
            try:
                record = grade_root(ref, hands[slot], args.square,
                                    slot, args)
            except Exception as exc:
                # One unusable root must not cost the shard, but it must not
                # read as "no leak here" either: it is recorded as an error
                # and counted apart from the graded roots.
                record = {"index": slot, "square": args.square,
                          "error": str(exc)[:400]}
                errors += 1
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            done += 1
            if record.get("error"):
                print(f"  [{args.square} {slot}] ERROR "
                      f"{record['error'][:120]}", flush=True)
            if record.get("confirmed"):
                print(f"  [{args.square} {slot}] CONFIRMED margin "
                      f"{record['margin']:+.2f} +-{record['se']:.2f} "
                      f"(served was model rank {record['served_model_rank']})",
                      flush=True)
            if done % 10 == 0:
                print(f"  {args.square}: {done}/{len(hands)} roots, "
                      f"{ref.seconds / 60:.1f} min", flush=True)
    print(f"SQUARE_DONE {args.square} {done} roots ({errors} errored), "
          f"{ref.seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
