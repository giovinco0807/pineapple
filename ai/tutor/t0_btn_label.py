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
# (220M/310M/600M/700M/810M/850M/860M/880M/900M/115-130M/2.1e9/3.3e9/4.4e9/5.1e9/5.2e9/6.6e9/7.7e9/8.9e9).
LABEL_BAND = 6_100_000_001
ROOT_STRIDE = 10


def model_args(models: Path):
    names = ("t0_bb.bin", "t0_btn.bin", "t1_bb.bin", "t1_btn.bin",
             "t2_bb.bin", "t2_btn.bin", "t3_bb.bin", "t3_btn.bin")
    hu = ",".join(str(models / "hu" / n) for n in names)
    rk = ",".join(str(models / "rankers" / n) for n in names)
    own = ",".join(str(models / "own_lap4" / n) for n in ("t0.bin", "t1.bin", "t2.bin"))
    return ["--hu-a-models", hu, "--hu-b-models", hu,
            "--hu-a-rankers", rk, "--hu-b-rankers", rk, "--hu-topk", "4",
            "--hu-a-t0-policy", str(models / "policy.bin"),
            "--hu-b-t0-policy", str(models / "policy.bin"),
            "--hu-t0-policy-topk", "8",
            "--serve-joint-samples", "200", "--serve-joint-samples-b", "200",
            "--arm-a-own", own, "--arm-b-own", own]


def joker_blind(key: str) -> str:
    return key.replace("X1", "X").replace("X2", "X")


def top_count(key: str) -> int:
    return len([c for c in key.split("|")[0].split(",") if c])


class Teacher:
    def __init__(self, binary: Path, models: Path, fl_ev: Path, work: Path):
        self.binary, self.models, self.fl_ev, self.work = binary, models, fl_ev, work
        self.calls, self.seconds = 0, 0.0

    def run(self, cards: str, opp: str, rollouts: int, seed: int, keys, tag: str):
        out = self.work / f"{tag}.jsonl"
        cmd = [str(self.binary), "--hu-match", "--hu-t0-deep"] + model_args(self.models)
        cmd += ["--fl-ev-config", str(self.fl_ev), "--hu-t0-seat", "1",
                "--t0-cards", cards, "--t0-opp-board", opp,
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


def choose_field(scored, served: str, eval_top: int, ranker_top: int):
    """The openings worth a sharp label, and why each is there."""
    sources: dict[str, list[str]] = {}
    by_eval = sorted(scored, key=lambda r: r["rank"])
    for r in by_eval[:eval_top]:
        sources.setdefault(r["key"], []).append("eval_top")
    ranked = [r for r in scored if r.get("ranker_rank")]
    for r in sorted(ranked, key=lambda r: r["ranker_rank"])[:ranker_top]:
        sources.setdefault(r["key"], []).append("ranker_top")
    seen = set()
    for r in sorted(ranked, key=lambda r: r["ranker_rank"]):
        t = top_count(r["key"])
        if t not in seen:
            seen.add(t)
            sources.setdefault(r["key"], []).append(f"stratum_top{t}")
    hit = next((r["key"] for r in scored if joker_blind(r["key"]) == joker_blind(served)), None)
    if hit is None:
        raise RuntimeError(f"served move {served} not among the audit's openings")
    sources.setdefault(hit, []).append("served")
    return hit, sources


def label_root(teacher: Teacher, index: int, root: dict, args) -> dict:
    cards = ",".join(root["btn_cards"])
    opp = root["bb_board"]
    scored = teacher.run(cards, opp, 0, 0, None, f"enum_{index}")
    served_key, sources = choose_field(scored, root["served"], args.eval_top, args.ranker_top)
    keys = list(sources)
    passes = []
    for p in range(args.passes):
        seed = LABEL_BAND + index * ROOT_STRIDE + p
        rows = teacher.run(cards, opp, args.rollouts, seed, keys, f"lab_{index}_p{p}")
        passes.append({r["key"]: r["mean"] for r in rows})
    record = {"id": root["id"], "index": index, "cards": cards, "opp_board": opp,
              "served": served_key, "field_size": len(keys), "rollouts": args.rollouts,
              "passes": args.passes, "sources": sources,
              "field": passes[0]}
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
    ap.add_argument("--eval-top", type=int, default=32)
    ap.add_argument("--ranker-top", type=int, default=16)
    args = ap.parse_args()
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

    teacher = Teacher(args.binary, args.models, args.fl_ev_config, args.work)
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
