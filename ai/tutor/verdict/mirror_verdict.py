"""Adjudicate a mirror match: paired per-deal edge, pooled across seed sets.

WHY THIS EXISTS.  `hu_match::mirror_hands` writes the settlement in real points
only.  Fantasyland is *not played* in a mirror -- entry is recorded as a width
and the arm's credit for it comes from the fl_ev table, exactly the way
`hu_match.rs`'s `entry_credit` closure does it for the session path:

    credit(f) = table[width - 14]   if the board stands and entered
                0                   otherwise

Score a mirror without that term and you have measured a different quantity: a
world where reaching Fantasyland is worth nothing.  That is a legitimate
robustness reading and this module reports it too, under its own name -- but it
is not the verdict.  The two differ by more than a factor of two on the run this
module was written against, and the mistake has been made once already.

WHAT A DEAL IS.  A mirror deal is played twice with the seats traded, so the
pair cancels the deal's luck:

    edge = ((settle[0] + credit_a[0] - credit_b[0])
          + (settle[1] + credit_a[1] - credit_b[1])) / 2

Identical arms give exactly zero on every deal -- that is the acceptance test
(`launch_verdict.sh accept`), and `--acceptance` asserts it here.

WHAT THE VERDICT IS.  The pool of every seed set, bootstrapped over deals.
Never a single set: on m7/m7b two sets landed with non-overlapping intervals and
opposite signs, and either one alone read as significant.  Per-set numbers are
printed for the record and are not the finding.

    python -m ai.tutor.verdict.mirror_verdict --runs v1=C:/tmp/verdict/v1 \
        v1b=C:/tmp/verdict/v1b --label "union vs gen2"
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

# The table these mirrors were played and scored under.  It is an argument, not
# a constant of the game: fl_ev has moved once already (14: 5.28 -> 6.57 on
# 2026-08-21) and a simulator has since proposed another table on a *different
# scale* (stackless raw, where ai/config/fl_ev.json is in-game paid).  Whatever
# is passed here must be the table the match was played under, not today's.
SHIPPED_TABLE = {14: 5.28, 15: 16.61, 16: 38.76, 17: 70.07}
WIDTHS = (14, 15, 16, 17)


def load_table(path: Path | None) -> dict[int, float]:
    if path is None:
        return dict(SHIPPED_TABLE)
    config = json.loads(path.read_text(encoding="utf-8"))
    return {int(w): float(v) for w, v in config["fl_ev"].items()}


def deal_rows(run_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(run_dir / "*.jsonl"))):
        shard = Path(path).stem
        for line in open(path, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                row["_shard"] = shard
                rows.append(row)
    if not rows:
        raise SystemExit(f"no deals under {run_dir}")
    return rows


def edges(rows: list[dict], table: dict[int, float], credit: bool):
    """A's per-deal edge, and the same rows' foul and entry rates."""
    out = np.empty(len(rows), dtype=np.float64)
    fouls_a = fouls_b = entries_a = entries_b = 0
    for index, row in enumerate(rows):
        total = 0.0
        for orientation in range(2):
            points = float(row["settle"][orientation])
            if credit:
                points += (table.get(row["a_entry"][orientation], 0.0)
                           - table.get(row["b_entry"][orientation], 0.0))
            total += points
        out[index] = total / 2.0
        fouls_a += sum(bool(x) for x in row["a_foul"])
        fouls_b += sum(bool(x) for x in row["b_foul"])
        entries_a += sum(1 for w in row["a_entry"] if w >= 14)
        entries_b += sum(1 for w in row["b_entry"] if w >= 14)
    hands = 2 * len(rows)
    return out, (fouls_a / hands, fouls_b / hands,
                 entries_a / hands, entries_b / hands)


def interval(sample: np.ndarray, resamples: int, seed: int):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sample), size=(resamples, len(sample)))
    means = sample[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def call(low: float, high: float, name_a: str, name_b: str) -> str:
    if low > 0:
        return f"{name_a}優位"
    if high < 0:
        return f"{name_b}優位"
    return "差なし"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="one or more seed sets as name=directory")
    parser.add_argument("--label", default="A vs B")
    parser.add_argument("--arms", nargs=2, default=("A", "B"), metavar=("ARM_A", "ARM_B"))
    parser.add_argument("--fl-ev-config", type=Path, default=None,
                        help="the table the match was PLAYED under "
                             f"(default: the shipped {SHIPPED_TABLE})")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--acceptance", action="store_true",
                        help="same-arm run: assert every deal settles to exactly zero")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    # The 公式 / FLゼロ labels must survive a cp932 console: on 2026-09-04 a
    # garbled heading let the FL-zero row be read as the verdict for a week.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    table = load_table(args.fl_ev_config)
    arm_a, arm_b = args.arms
    report = {"schema": "ofc_mirror_verdict/v1", "label": args.label,
              "arms": [arm_a, arm_b], "fl_ev": {str(w): table[w] for w in WIDTHS},
              "sets": {}, "pool": {}}

    print(f"=== {args.label} ===")
    print(f"fl_ev: {[table[w] for w in WIDTHS]}"
          f"{'  (default -- pass --fl-ev-config if the match used another)' if args.fl_ev_config is None else ''}")

    pooled_full, pooled_zero = [], []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"--runs wants name=directory, got {spec!r}")
        name, directory = spec.split("=", 1)
        rows = deal_rows(Path(directory))
        full, rates = edges(rows, table, credit=True)
        zero, _ = edges(rows, table, credit=False)

        if args.acceptance:
            offenders = int((full != 0).sum())
            print(f"  {name}: acceptance {len(full)} deals, "
                  f"{offenders} non-zero -> {'PASS' if offenders == 0 else 'FAIL'}")
            report["sets"][name] = {"n_deals": len(full), "non_zero": offenders}
            if offenders:
                raise SystemExit(
                    f"acceptance FAILED on {name}: {offenders} deals settled non-zero. "
                    "Identical arms must cancel exactly; the bundle is mis-assembled."
                )
            continue

        low, high = interval(full, args.resamples, args.seed)
        print(f"  {name}: {len(full)} deals  公式 {full.mean():+.4f} [{low:+.4f}, {high:+.4f}]"
              f"   FLゼロ採点 {zero.mean():+.4f}"
              f"   foul {rates[0]:.1%}/{rates[1]:.1%}  FL {rates[2]:.1%}/{rates[3]:.1%}"
              f"   一致 {(full == 0).mean():.1%}")
        report["sets"][name] = {
            "n_deals": len(full), "mean": float(full.mean()), "ci95": [low, high],
            "mean_fl_zero": float(zero.mean()),
            "foul_a": rates[0], "foul_b": rates[1],
            "fl_a": rates[2], "fl_b": rates[3],
            "identical_share": float((full == 0).mean()),
        }
        pooled_full.append(full)
        pooled_zero.append(zero)

    if args.acceptance:
        print("acceptance PASS -- every deal settled to exactly zero")
        report["pool"] = {"acceptance": "PASS"}
    else:
        every = np.concatenate(pooled_full)
        every_zero = np.concatenate(pooled_zero)
        low, high = interval(every, args.resamples, args.seed)
        zlow, zhigh = interval(every_zero, args.resamples, args.seed)
        verdict = call(low, high, arm_a, arm_b)
        print(f"  POOL ({len(args.runs)} sets): {len(every)} deals  "
              f"公式 {every.mean():+.4f} [{low:+.4f}, {high:+.4f}]  -> {verdict}  <- 判定はこの行")
        print(f"  POOL FLゼロ採点: {every_zero.mean():+.4f} [{zlow:+.4f}, {zhigh:+.4f}]"
              f"  -> {call(zlow, zhigh, arm_a, arm_b)}")
        if len(args.runs) < 2:
            print("  ! 1セットのみ。規律は2セット以上のプール判定 "
                  "(m7/m7b は符号が逆で、どちらも単体なら有意に読めた)")
        report["pool"] = {
            "n_sets": len(args.runs), "n_deals": len(every),
            "mean": float(every.mean()), "ci95": [low, high], "verdict": verdict,
            "mean_fl_zero": float(every_zero.mean()), "ci95_fl_zero": [zlow, zhigh],
        }

    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                            encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
