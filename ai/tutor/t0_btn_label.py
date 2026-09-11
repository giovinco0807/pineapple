"""Sharp T0-BUTTON labels: the referee as teacher, over the openings that matter.

The T0 labels the served Button evaluator learned from are three hops from
exact ground and, measured today, carry no opponent-conditioned signal above
the noise: a policy trained on them ranks the referee's best opening worse
than the ranker fence does.  The deep referee is the one instrument here that
beats serving (it found a +0.87/decision leak), so this makes labels with it.

Per root the labelled field is
    evaluator top-32  U  ranker top-16  U  best-ranked opening of each
    top-row-count stratum (0..3 cards on top)  U  the served move
which always contains the ranker's top-4, so every opening serving could
actually choose is labelled; the ~190 openings outside the field are never
reachable through the fence and are left out rather than anchored to the
stale scores they came from.

Values come from `--hu-t0-deep` with the CHAMPION's continuations (never the
distilled nets: they inflate ambitious two-card tops, and the strata put
exactly those in the field), all candidates of a root sharing the same
rollout futures (common random numbers), so the ranking within a root is as
sharp as the count allows.  Dev roots get a second independent pass so the
dev regret carries a label-noise standard error -- the instrument that
decided the T2/T1 campaigns.

Record per root:
  {"id", "cards", "opp_board", "served", "field": {key: mean}, "field_size",
   "rollouts", "pass", "seed", "sources": {key: [why it is in the field]}}

Usage (a slice of roots; sharded by the fleet the same way t0_mine is):
    python -m ai.tutor.t0_btn_label --roots D:/ofc_data/hu/t0btn_mine/roots_all.jsonl \
        --start 100000 --count 20 --rollouts 128 --out labels.jsonl
    add --passes 2 for dev roots.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Fresh band, disjoint from every band in the seed registry
# (220M/310M/600M/700M/810M/850M/860M/880M/900M/115-135M/2.1e9/3.3e9/4.4e9/5.1e9/5.2e9/6.1e9/6.6e9/7.7e9/8.9e9/
#  9.1e9 = tr3 local champion trace 20260904, --self-play-seed 9_100_000_001;
#  9.3e9 = fence duel; 9.4e9 = t0_bb_roots deal sampling; 9.5e9 = hu_street_roots sampling).
LABEL_BAND = 6_100_000_001
ROOT_STRIDE = 10
# Far enough apart that a staged root's seeds never meet another root's.
STAGE_STRIDE = 1_000_000_007


CONTRACTS = {
    # Serving contract v2 (models_ship_20260911, 2026-09-11): fences wide open
    # at the streets (27), Button opening ranker top-16, BB opening policy top-16.
    "v2": ["--hu-topk", "27", "--hu-t0-btn-topk", "16", "--hu-t0-policy-topk", "16"],
    # The 20260829..20260904 contract: ranker top-4 everywhere, BB opening policy
    # top-8, no Button width.  Every fixed raced set (t0btn_race480, the T0-BB
    # eval 480, the five street slots) was priced under this continuation, and it
    # is 2.3x cheaper per rollout (2 keys x 512 rollouts: 254 s against v2's
    # 576 s, 2026-09-11) -- so a corpus that has to sit on the same instrument
    # as those sets, or be bought cheaply, is labelled here.
    "old": ["--hu-topk", "4", "--hu-t0-policy-topk", "8"],
}


def model_args(models: Path, contract: str = "v2"):
    """The champion as the referee's continuation: the bundle's nets under one
    serving contract.  Labels made under different contracts are separate
    instruments and must not be pooled; every row stamps its contract."""
    names = ("t0_bb.bin", "t0_btn.bin", "t1_bb.bin", "t1_btn.bin",
             "t2_bb.bin", "t2_btn.bin", "t3_bb.bin", "t3_btn.bin")
    hu = ",".join(str(models / "hu" / n) for n in names)
    rk = ",".join(str(models / "rankers" / n) for n in names)
    own = ",".join(str(models / "own_lap4" / n) for n in ("t0.bin", "t1.bin", "t2.bin"))
    if contract not in CONTRACTS:
        raise ValueError(f"unknown serving contract {contract!r}; one of {sorted(CONTRACTS)}")
    return ["--hu-a-models", hu, "--hu-b-models", hu,
            "--hu-a-rankers", rk, "--hu-b-rankers", rk,
            "--hu-a-t0-policy", str(models / "policy.bin"),
            "--hu-b-t0-policy", str(models / "policy.bin"),
            *CONTRACTS[contract],
            "--serve-joint-samples", "200", "--serve-joint-samples-b", "200",
            "--arm-a-own", own, "--arm-b-own", own]


def joker_blind(key: str) -> str:
    return key.replace("X1", "X").replace("X2", "X")


def top_count(key: str) -> int:
    return len([c for c in key.split("|")[0].split(",") if c])


class Teacher:
    def __init__(self, binary: Path, models: Path, fl_ev: Path, work: Path, seat: int = 1,
                 street: int = 0, contract: str = "v2"):
        self.binary, self.models, self.fl_ev, self.work = binary, models, fl_ev, work
        self.contract = contract
        # 1 = Button (hero's five + BB's placed board); 0 = Big Blind, whose
        # opening is decided by hero's five alone (docs/t0_bb_opponent_block_20260824.md).
        self.seat = seat
        # Street 0 prices openings with --hu-t0-deep; streets 1..3 replay a
        # traced hand up to the decision with --hu-deep-replay, so the root is
        # the trace line itself (`self.hand_file`, set per root by label_root).
        self.street = street
        self.hand_file: Path | None = None
        self.calls, self.seconds = 0, 0.0

    def run(self, cards: str, opp: str, rollouts: int, seed: int, keys, tag: str):
        out = self.work / f"{tag}.jsonl"
        if self.street == 0:
            cmd = [str(self.binary), "--hu-match", "--hu-t0-deep"] + model_args(self.models, self.contract)
            cmd += ["--fl-ev-config", str(self.fl_ev), "--t0-cards", cards,
                    "--rollouts", str(rollouts), "--self-play-seed", str(seed),
                    "--output", str(out)]
            if self.seat == 1:
                cmd += ["--hu-t0-seat", "1", "--t0-opp-board", opp]
        else:
            cmd = [str(self.binary), "--hu-match", "--hu-deep-replay"] + model_args(self.models, self.contract)
            cmd += ["--fl-ev-config", str(self.fl_ev), "--input", str(self.hand_file),
                    "--replay-hand", "0", "--replay-street", str(self.street),
                    "--replay-seat", str(self.seat),
                    "--rollouts", str(rollouts), "--self-play-seed", str(seed),
                    "--output", str(out)]
        if keys is not None:
            cmd += ["--t0-keys-inline", ";".join(keys)]
        started = time.time()
        done = subprocess.run(cmd, capture_output=True, text=True)
        self.calls += 1
        self.seconds += time.time() - started
        if done.returncode != 0:
            raise RuntimeError(f"{tag}: {done.stderr[-500:]}")
        return [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]


def choose_field(scored, served, eval_top: int, ranker_top: int, field: str,
                 serve_top: int = 8):
    """The openings worth a sharp label, and why each is there.

    `wide` is lap 1's field: the evaluator's top 32, the ranker's top 16, the
    best of each top-row-count stratum, and the served move -- about 40
    openings at 128 rollouts each.

    `fence` is the ranker's top-K and the served move, nothing else.  It exists
    because the ranker fence shipped at K=16 on 2026-09-04: serving cannot
    choose an opening outside the ranker's top 16, so labelling the rest buys
    nothing a trained model can act on, and the budget it frees buys depth.
    Measured over 20 dev roots at equal cost (16 openings x 316 rollouts vs
    39.5 x 128), the middle/bottom orderings that lap 1 could not resolve
    reproduce on an independent pass 75.1% -> 86.0% of the time
    (docs/t0btn_campaign_20260902.md 12.11).  Widening the fence again would
    mean widening this field with it.
    """
    sources: dict[str, list[str]] = {}
    # The fence is whichever shortlist the serve consulted: at BB (and any
    # bundle carrying policy.bin) the T0 policy's order, else the ranker's.
    fence_key = "policy_rank" if any(r.get("policy_rank") for r in scored) else "ranker_rank"
    ranked = [r for r in scored if r.get(fence_key)]
    if field == "wide":
        by_eval = sorted(scored, key=lambda r: r["rank"])
        for r in by_eval[:eval_top]:
            sources.setdefault(r["key"], []).append("eval_top")
    for r in sorted(ranked, key=lambda r: r[fence_key])[:ranker_top]:
        sources.setdefault(r["key"], []).append("ranker_top")
    if field == "wide":
        seen = set()
        for r in sorted(ranked, key=lambda r: r[fence_key]):
            t = top_count(r["key"])
            if t not in seen:
                seen.add(t)
                sources.setdefault(r["key"], []).append(f"stratum_top{t}")
    if served is None:
        # No trace behind this root (BB roots are bare deals): the served move
        # is the serve's own pick, the evaluator's best inside the fence.
        inside = [r for r in ranked if r[fence_key] <= serve_top]
        hit = min(inside, key=lambda r: r["rank"])["key"]
    else:
        hit = next((r["key"] for r in scored if joker_blind(r["key"]) == joker_blind(served)), None)
        if hit is None:
            raise RuntimeError(f"served move {served} not among the audit's openings")
    sources.setdefault(hit, []).append("served")
    return hit, sources, fence_key


def priced(row: dict, score: str) -> float:
    """One opening's label under the chosen scoring rule.

    `full` is what the referee reports: the zero-sum settlement plus hero's
    Fantasyland credit minus the opponent's.  `own_fl` drops the opponent's
    entry term.  The opponent's board is pinned before hero opens and its
    play barely answers hero's placement, so that term carries almost no
    signal about which opening is better -- and about half the variance: a
    Fantasyland seat is a coin flip worth 38.76 at width 16.  Dropping it
    halved the per-opening variance at T0 with no measurable cost to the
    objective (docs/t0btn_campaign_20260902.md 12.7, 12.10).
    """
    if score == "full":
        return float(row["mean"])
    parts = (row.get("settle"), row.get("own_entry"))
    if not parts[0] or not parts[1]:
        raise RuntimeError(
            "--score own_fl needs the referee's per-rollout components; this "
            "binary predates hu_match::Parts and reports only the total"
        )
    settle, own = parts
    return sum(s + o for s, o in zip(settle, own)) / len(settle)


def per_rollout(row: dict, score: str):
    """Every rollout's value for one opening, under the scoring rule."""
    if score == "full":
        return [float(x) for x in row["scores"]]
    settle, own = row.get("settle"), row.get("own_entry")
    if not settle or not own:
        raise RuntimeError("--score own_fl needs a binary that emits components")
    return [float(s) + float(o) for s, o in zip(settle, own)]


def race_field(teacher, cards, opp, keys, index, pass_index, args):
    """Price a field by racing: keep spending only where the answer is open.

    Every candidate gets a batch; after each batch a candidate is dropped once
    it trails the leader by more than `z` combined standard errors (and has
    had its floor), and the race stops when the two leaders are resolved to
    `target_se`, or every survivor has hit the cap.  A position whose best is
    obvious finishes in two batches; one where three openings sit within a
    point of each other gets the whole budget, which is where a validation
    set needs it.  Each batch draws its own seeds, so a leader's early luck
    does not survive into its final value, and every reported number carries
    the standard error it was actually measured to.
    """
    alive = list(keys)
    n = {k: 0 for k in keys}
    sums = {k: 0.0 for k in keys}
    sq = {k: 0.0 for k in keys}
    batches = 0
    stop = ""
    while True:
        seed = LABEL_BAND + index * ROOT_STRIDE + pass_index + STAGE_STRIDE * (batches + 1)
        rows = teacher.run(cards, opp, args.race_batch, seed, alive,
                           f"lab_{index}_p{pass_index}_b{batches}")
        got = {r["key"]: per_rollout(r, args.score) for r in rows}
        missing = [k for k in alive if k not in got]
        if missing:
            raise RuntimeError(f"batch {batches} lost {len(missing)} openings: {missing[:3]}")
        for k, vals in got.items():
            n[k] += len(vals)
            sums[k] += sum(vals)
            sq[k] += sum(v * v for v in vals)
        batches += 1
        mean = {k: sums[k] / n[k] for k in alive}
        se = {k: (max(sq[k] / n[k] - mean[k] ** 2, 1e-9) / n[k]) ** 0.5 for k in alive}
        lead = max(alive, key=lambda k: mean[k])
        alive = [k for k in alive
                 if k == lead or n[k] < args.race_floor
                 or mean[lead] - mean[k] <= args.race_z * (se[lead] ** 2 + se[k] ** 2) ** 0.5]
        if len(alive) <= 1:
            stop = "decided"
            break
        top = sorted(alive, key=lambda k: -mean[k])[:2]
        gap_se = (se[top[0]] ** 2 + se[top[1]] ** 2) ** 0.5
        if gap_se < args.race_target_se:
            stop = "resolved"
            break
        if all(n[k] >= args.race_cap for k in alive):
            stop = "capped"
            break
    values = {k: sums[k] / n[k] for k in keys}
    errors = {k: (max(sq[k] / n[k] - values[k] ** 2, 1e-9) / n[k]) ** 0.5 for k in keys}
    return values, {"se": errors, "n": dict(n), "batches": batches, "stop": stop,
                    "survivors": alive}


def price_field(teacher, cards, opp, keys, index, pass_index, args):
    """One pass over the field, under whichever budget the schedule asks for.

    Flat spends the same rollouts on all sixteen openings.  Staged spends a
    little on all of them, keeps the leaders, and spends the rest on those --
    which is where the budget belongs: a 316-rollout pass already names the
    true best 16 times in 20 and its misses are the runner-up, so what a label
    gets wrong is almost always the top two or three, and the twelve openings
    that were never in contention are being measured to a precision nobody
    reads.  Simulated over the 20 dev roots, at the same ~5,000 rollouts, a
    staged budget cuts the label's own regret 1.083 -> 0.884, and is worth
    about twice the rollouts spent flat (docs/t0btn_campaign_20260902.md 12.15).

    Each stage draws its own seeds, so the value a survivor is finally
    reported at was not the one that selected it -- otherwise the leader of a
    shallow stage carries its own good luck into the label (the winner's curse
    the miner spends a whole scoring stage avoiding).  An opening eliminated
    early keeps the shallow value it was eliminated on: it is far from the top
    by construction, and its exact depth is what nobody reads.
    """
    if not args.schedule:
        seed = LABEL_BAND + index * ROOT_STRIDE + pass_index
        rows = teacher.run(cards, opp, args.rollouts, seed, keys, f"lab_{index}_p{pass_index}")
        return {r["key"]: priced(r, args.score) for r in rows}
    values: dict[str, float] = {}
    alive = list(keys)
    for stage, (rollouts, keep) in enumerate(args.schedule):
        seed = LABEL_BAND + index * ROOT_STRIDE + pass_index + STAGE_STRIDE * (stage + 1)
        rows = teacher.run(cards, opp, rollouts, seed, alive,
                           f"lab_{index}_p{pass_index}_s{stage}")
        got = {r["key"]: priced(r, args.score) for r in rows}
        missing = [k for k in alive if k not in got]
        if missing:
            raise RuntimeError(f"stage {stage} lost {len(missing)} openings: {missing[:3]}")
        values.update(got)
        alive = sorted(alive, key=lambda k: -got[k])[:keep]
    return values


def canonical_jokers(cards: list[str]) -> list[str]:
    """Spell hero's jokers by position: the first one X1, the second X2.

    The two jokers are the same card.  `--rollouts 0` enumerates openings in
    the deal's spelling, but the deep referee renumbers hero's jokers by
    position, so a lone joker dealt as X2 makes every key the enumeration
    produced "unknown" to the pricing call (bbdev shard 001800, 2026-09-07).
    Renaming on the way in makes both paths spell the same card the same way.
    """
    out, seen = [], 0
    for c in cards:
        if c.startswith("X"):
            seen += 1
            out.append(f"X{seen}")
        else:
            out.append(c)
    return out


def street_field(scored, served: str):
    """Streets 1..3: the referee enumerates every legal placement (27 at T1/T2,
    12 at T3), few enough to label whole, so the field is all of them and the
    served move is the board the trace recorded."""
    sources = {r["key"]: ["all"] for r in scored}
    hit = next((k for k in sources if joker_blind(k) == joker_blind(served)), None)
    if hit is None:
        raise RuntimeError(f"served board {served} not among the {len(sources)} enumerated placements")
    sources[hit].append("served")
    return hit, sources, "all"


def label_root(teacher: Teacher, index: int, root: dict, args) -> dict:
    if args.street:
        # A traced decision: hand line -> one-hand trace file for deep-replay.
        hand = root["trace"]
        teacher.hand_file = teacher.work / f"hand_{index}.jsonl"
        hand = dict(hand, hand=0)  # deep-replay selects --replay-hand 0
        teacher.hand_file.write_text(json.dumps(hand) + "\n", encoding="utf-8")
        cards = ",".join(root.get("draw") or [])
        opp = ""
        scored = teacher.run(cards, opp, 0, 0, None, f"enum_{index}")
        served_key, sources, fence_key = street_field(scored, root["served"])
    else:
        cards = ",".join(canonical_jokers(list(root.get("btn_cards") or root["draw"])))
        opp = root.get("bb_board") or "||"
        scored = teacher.run(cards, opp, 0, 0, None, f"enum_{index}")
        served_key, sources, fence_key = choose_field(
            scored, root.get("served"), args.eval_top, args.ranker_top, args.field, args.serve_top)
    keys = list(sources)
    passes = []
    race_info = None
    for p in range(args.passes):
        if args.race:
            values, info = race_field(teacher, cards, opp, keys, index, p, args)
            passes.append(values)
            if p == 0:
                race_info = info
        else:
            passes.append(price_field(teacher, cards, opp, keys, index, p, args))
    record = {"id": root["id"], "index": index, "cards": cards, "opp_board": opp,
              "served": served_key, "field_size": len(keys), "rollouts": args.rollouts,
              "passes": args.passes, "sources": sources,
              # Stamped on every row: a corpus that mixes scoring rules or
              # field widths is two corpora, and nothing downstream could tell.
              "field_kind": args.field, "score": args.score,
              "schedule": args.schedule,
              "seat": args.seat, "street": args.street, "fence_by": fence_key,
              "contract": args.contract,
              "stratum": root.get("stratum"),
              "field": passes[0]}
    if race_info is not None:
        record["race"] = {"batch": args.race_batch, "z": args.race_z, "floor": args.race_floor,
                          "cap": args.race_cap, "target_se": args.race_target_se}
        record["field_se"] = race_info["se"]
        record["field_n"] = race_info["n"]
        record["race_batches"] = race_info["batches"]
        record["race_stop"] = race_info["stop"]
        record["race_survivors"] = race_info["survivors"]
    if args.passes > 1:
        record["field_pass2"] = passes[1]
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", type=Path, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("labelwork"))
    ap.add_argument("--models", type=Path, default=Path("D:/ofc_data/hu/models_ship_20260903"))
    ap.add_argument("--binary", type=Path,
                    default=REPO / "ai/rust_solver/target/release/t4_first_exact.exe")
    ap.add_argument("--fl-ev-config", type=Path, default=REPO / "ai/config/fl_ev.json")
    ap.add_argument("--rollouts", type=int, default=128)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--eval-top", type=int, default=32,
                    help="wide field only: how many of the evaluator's own top openings to label")
    ap.add_argument("--ranker-top", type=int, default=16,
                    help="how much of the ranker's order to label; must cover the served fence")
    ap.add_argument("--field", choices=("wide", "fence"), default="wide",
                    help="wide: lap 1's ~40 openings. fence: the ranker's top-K and the "
                         "served move, which is everything serving can choose at K=16")
    ap.add_argument("--schedule", default="",
                    help="staged budget as rollouts:keep,... e.g. 64:8,250:3,700:1 . "
                         "Empty spends --rollouts flat on the whole field")
    ap.add_argument("--race", action="store_true",
                    help="adaptive racing instead of a fixed budget; see race_field")
    ap.add_argument("--race-batch", type=int, default=64, help="rollouts per candidate per round")
    ap.add_argument("--race-z", type=float, default=3.0,
                    help="drop a candidate trailing the leader by more than this many combined SEs")
    ap.add_argument("--race-floor", type=int, default=128,
                    help="rollouts every candidate gets before it can be dropped")
    ap.add_argument("--race-cap", type=int, default=8192, help="rollouts per survivor at most")
    ap.add_argument("--race-target-se", type=float, default=0.35,
                    help="stop once the leaders' gap is measured to this SE")
    ap.add_argument("--seat", choices=("bb", "btn"), default="btn",
                    help="btn: roots carry btn_cards/bb_board/served. bb: roots carry "
                         "draw (five cards); the served move is the serve's own fenced pick")
    ap.add_argument("--street", type=int, default=0, choices=(0, 1, 2, 3),
                    help="0: openings (--hu-t0-deep). 1..3: traced street decisions replayed "
                         "with --hu-deep-replay; roots carry trace/street/seat/served "
                         "(ai/tutor/hu_street_roots.py) and the field is every placement")
    ap.add_argument("--contract", choices=tuple(CONTRACTS), default="v2",
                    help="serving contract the referee's continuation plays under (see CONTRACTS); "
                         "old = the 4/(4)/8 fences every fixed raced set was priced with, 2.3x cheaper")
    ap.add_argument("--serve-top", type=int, default=8,
                    help="serving fence width used to name the served move on roots "
                         "without a trace (BB serves the policy's top 8)")
    ap.add_argument("--score", choices=("full", "own_fl"), default="full",
                    help="full: the referee's own number. own_fl: drop the opponent's "
                         "Fantasyland credit (needs a binary that emits components)")
    args = ap.parse_args()
    if args.schedule:
        args.schedule = [tuple(int(x) for x in part.split(":"))
                         for part in args.schedule.split(",")]
        if any(len(s) != 2 for s in args.schedule):
            raise SystemExit("--schedule wants rollouts:keep pairs")
        if args.schedule[-1][1] != 1:
            raise SystemExit("--schedule must end keeping exactly one opening")
    else:
        args.schedule = []
    args.work.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if args.out.exists():
        for line in args.out.open(encoding="utf-8"):
            if line.strip():
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    roots = []
    with args.roots.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < args.start:
                continue
            if i >= args.start + args.count:
                break
            if line.strip():
                roots.append((i, json.loads(line)))

    teacher = Teacher(args.binary, args.models, args.fl_ev_config, args.work,
                      seat=1 if args.seat == "btn" else 0, street=args.street,
                      contract=args.contract)
    written = errors = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for index, root in roots:
            if root["id"] in done_ids:
                continue
            try:
                record = label_root(teacher, index, root, args)
            except Exception as exc:
                record = {"id": root["id"], "index": index, "error": str(exc)[:400]}
                errors += 1
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            written += 1
            if written % 5 == 0 or "error" in record:
                print(f"  {written}/{len(roots)} roots, {teacher.seconds / 60:.1f} min"
                      + (f"  ERROR {record['error'][:80]}" if "error" in record else ""),
                      flush=True)
    print(f"LABEL_DONE start={args.start} count={args.count} written={written} "
          f"errors={errors} calls={teacher.calls} minutes={teacher.seconds / 60:.1f}",
          flush=True)


if __name__ == "__main__":
    main()
