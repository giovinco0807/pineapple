"""What a 1,024-particle T0 batch can and cannot resolve.

Reads the ladder corpus -- twelve independent 1,024-particle batches of each of
twenty openings, written by ``hu_t0_ladder20_explicit_plan_v1`` -- and answers
the question those batches were bought to answer: when a single batch says one
action beats another by 0.3, is that a fact about the game or about the draw?

## What is computed, and why each one

**Scatter.** The standard deviation of an action's score across the twelve
batches, and separately the standard deviation of the top-two *gap*.  These are
different numbers and the second is the one that matters.  Every batch scores
all its candidates against the same sampled worlds, so the luck of the draw
moves the whole slate together and cancels in differences; a scatter of 0.5 in
the levels can sit on top of a scatter of 0.05 in the gaps.  Quoting the level
scatter as if it bounded the ranking -- which this project did -- overstates the
doubt by whatever that cancellation is worth, and only the gap column says by
how much.

**Winner agreement, as a ladder.**  Averaging *k* independent batches has
exactly the variance of one batch of 1,024*k*, so subsets of the twelve stand in
for the rungs: pairs are 2,048, fours are 4,096, all twelve are 12,288.  For
each *k* this reports how often a *k*-batch decision picks the same action as
the full twelve.  That is the honest form of "can 1,024 particles be trusted" --
not whether the number moves, but whether the decision does.

**Regret of a k-batch decision.**  Agreement alone cannot separate a coin-flip
between two actions worth the same from a real mistake, and only the second
costs anything.  So each disagreement is priced against a fixed six-batch
pricing pool that took no part in choosing.  The pool is fixed rather than
"whatever is left over" on purpose: a leftover pool shrinks as k grows and its
own argmax inflates, which would make regret climb with k for a reason that has
nothing to do with the decision.

**The model's regret, bracketed.**  The 0.107 points/hand quoted from the
single-batch runs is inflated and has to be.  "Best" there was the maximum of
about thirty noisy estimates, and the maximum of noisy estimates is biased
upward -- the luckiest draw wins the argmax -- so the gap to anything else is
overstated.  Choosing the best on one half of the batches and scoring it on the
other removes that, but overshoots the other way: a best chosen from noise is
sometimes the wrong action, so its held-out score understates the true best and
the estimate can even come out negative.

Neither is the answer alone, and together they are:

    split-sample  <=  true regret  <=  naive

Both are reported.  The naive column is the arithmetic that produced 0.107, so
the width of the bracket is exactly how much that number was overstating.

**Sieve stability.**  Each batch re-runs its own 64-particle sieve, so the
surviving set can differ between batches of one hand.  Everything above is
computed on the actions common to all twelve; this reports how much was
discarded to get there, because a hand whose candidate set churns is a hand
where stage one, not stage two, is the thing under suspicion.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# Subset sizes standing in for particle counts: k batches of 1,024 carry the
# precision of one batch of 1,024k.
LADDER = (1, 2, 3, 4, 6, 8)
FULL = 12
# Cap on the k-subsets enumerated per rung; C(12,6)=924 is the widest and
# fits, so this only bites if the batch count grows. Seeded, so a rerun
# reproduces the same subsets.
SUBSET_SAMPLES = 1000


def load_positions(root: pathlib.Path) -> list[dict[str, Any]]:
    """Every position record under ``root``, in offset order."""

    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {"SHARD_DONE.json"} or path.name.startswith("plan"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, Mapping) or "runs" not in payload:
            continue
        records.append(dict(payload))
    records.sort(key=lambda record: int(record.get("offset", -1)))
    return records


def batch_scores(record: Mapping[str, Any]) -> dict[str, float]:
    """One batch's measured score per action.

    Stage-one survivors that never reached stage two carry no ``score`` and are
    dropped: they were sieved out, not measured, and treating a missing score
    as a low one would invent a ranking the run never produced.
    """

    out: dict[str, float] = {}
    for run in record.get("runs", ()):
        for key, row in run.get("scores", {}).items():
            value = row.get("score") if isinstance(row, Mapping) else row
            if value is None:
                continue
            out[key] = float(value)
    return out


def group_by_hand(
    records: Iterable[Mapping[str, Any]]
) -> dict[tuple[str, ...], list[dict[str, float]]]:
    """Batches per opening, keyed by the dealt cards."""

    grouped: dict[tuple[str, ...], list[dict[str, float]]] = {}
    for record in records:
        dealt = record.get("observation", {}).get("dealt_cards")
        if not dealt:
            continue
        scores = batch_scores(record)
        if scores:
            grouped.setdefault(tuple(dealt), []).append(scores)
    return grouped


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _argmax(scores: Mapping[str, float]) -> str:
    return max(scores, key=lambda key: scores[key])


def _averaged(batches: Sequence[Mapping[str, float]], keys: Sequence[str],
              picks: Iterable[int]) -> dict[str, float]:
    chosen = [batches[index] for index in picks]
    return {key: _mean([batch[key] for batch in chosen]) for key in keys}


def analyse_hand(batches: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """Scatter, ladder agreement and priced regret for one opening."""

    common = sorted(set.intersection(*(set(batch) for batch in batches)))
    seen = set().union(*(set(batch) for batch in batches))
    n = len(batches)

    # Scatter of the levels, and of the gap that actually decides the ranking.
    level_sd = [
        statistics.stdev([batch[key] for batch in batches]) for key in common
    ] if n > 1 else []
    gaps: list[float] = []
    for batch in batches:
        ordered = sorted((batch[key] for key in common), reverse=True)
        if len(ordered) > 1:
            gaps.append(ordered[0] - ordered[1])
    gap_sd = statistics.stdev(gaps) if len(gaps) > 1 else float("nan")

    reference = _averaged(batches, common, range(n))
    truth = _argmax(reference)

    # The ladder.  Both columns price against the twelve-batch reference, and
    # NOT against a held-out half, which is the trap this replaced.
    #
    # Complementary halves of a fixed twelve are not independent: conditioned on
    # the realised total, whatever the choosing half likes the pricing half must
    # dislike by exactly as much.  So as k grows and the choice tracks its half
    # more tightly, it tracks the pricing half's disagreement more tightly too,
    # and measured regret CLIMBS with k -- the opposite of the truth.  A
    # synthetic hand with a 0.037 edge and a near-tied realised sample produced
    # exactly that: 0.021 at k=1 rising to 0.032 at k=6.
    #
    # Pricing on all twelve is correlated the other way (the reference contains
    # the choosing subset), so these are mildly optimistic and reach exactly
    # zero at k=12 by construction. They are read as a shape -- how fast does
    # the decision settle -- and the absolute level of the model's own regret is
    # bracketed separately, where the bias direction is known on both sides.
    reference_best_value = reference[truth]
    ladder: dict[int, dict[str, float]] = {}
    for k in LADDER:
        if k > n:
            continue
        agree = 0
        regrets: list[float] = []
        subsets = list(itertools.combinations(range(n), k))
        if len(subsets) > SUBSET_SAMPLES:
            subsets = random.Random(20260819).sample(subsets, SUBSET_SAMPLES)
        for picks in subsets:
            chosen = _argmax(_averaged(batches, common, picks))
            regrets.append(reference_best_value - reference[chosen])
            agree += chosen == truth
        ladder[k] = {
            "equivalent_particles": k * 1024,
            "trials": len(subsets),
            "agreement": agree / len(subsets),
            "mean_regret": _mean(regrets),
            "max_regret": max(regrets),
        }

    return {
        "batches": n,
        "actions_common": len(common),
        "actions_seen": len(seen),
        "sieve_churn": 1.0 - len(common) / len(seen) if seen else 0.0,
        "level_sd_median": statistics.median(level_sd) if level_sd else float("nan"),
        "level_sd_max": max(level_sd) if level_sd else float("nan"),
        "gap_sd": gap_sd,
        "gap_mean": _mean(gaps) if gaps else float("nan"),
        "reference_best": truth,
        "reference_gap": (
            sorted(reference.values(), reverse=True)[0]
            - sorted(reference.values(), reverse=True)[1]
            if len(reference) > 1 else float("nan")
        ),
        "reference": reference,
        "ladder": ladder,
    }


def split_sample_regret(
    batches: Sequence[Mapping[str, float]], pick: str | None
) -> dict[str, float]:
    """Price an action against a best chosen on batches that did not score it.

    ``pick`` is the shipped model's action, which depends on nothing measured
    here and so needs no split of its own.  The split exists for the *best*:
    choosing and scoring it on the same batches is what inflated the 0.107.
    """

    common = sorted(set.intersection(*(set(batch) for batch in batches)))
    if pick not in common:
        return {"regret_split": float("nan"), "regret_naive": float("nan"),
                "splits": 0}
    n = len(batches)
    half = n // 2
    honest: list[float] = []
    naive: list[float] = []
    for picks in itertools.combinations(range(n), half):
        rest = [index for index in range(n) if index not in picks]
        chooser = _averaged(batches, common, picks)
        scorer = _averaged(batches, common, rest)
        best = _argmax(chooser)
        # Honest: best chosen on one half, both priced on the other.
        honest.append(scorer[best] - scorer[pick])
        # Naive: chosen and priced on the same half, which is the arithmetic
        # that produced 0.107. Carried so the inflation can be read off rather
        # than asserted.
        naive.append(chooser[best] - chooser[pick])
    return {
        "regret_split": _mean(honest),
        "regret_naive": _mean(naive),
        "splits": len(honest),
    }


def report(grouped: Mapping[tuple[str, ...], list[dict[str, float]]],
           model_picks: Mapping[tuple[str, ...], str] | None = None) -> dict[str, Any]:
    hands = {}
    for dealt, batches in sorted(grouped.items()):
        entry = analyse_hand(batches)
        if model_picks and dealt in model_picks:
            entry["model_pick"] = model_picks[dealt]
            entry.update(split_sample_regret(batches, model_picks[dealt]))
            entry["model_is_reference_best"] = (
                model_picks[dealt] == entry["reference_best"]
            )
        entry.pop("reference", None)
        hands[" ".join(dealt)] = entry

    complete = [e for e in hands.values() if e["batches"] >= FULL]
    pooled: dict[str, Any] = {
        "hands": len(hands),
        "hands_with_all_batches": len(complete),
    }
    if complete:
        pooled["level_sd_median"] = statistics.median(
            e["level_sd_median"] for e in complete
        )
        pooled["gap_sd_median"] = statistics.median(e["gap_sd"] for e in complete)
        pooled["gap_sd_max"] = max(e["gap_sd"] for e in complete)
        pooled["sieve_churn_median"] = statistics.median(
            e["sieve_churn"] for e in complete
        )
        pooled["ladder"] = {
            k: {
                "equivalent_particles": k * 1024,
                "agreement": _mean([e["ladder"][k]["agreement"] for e in complete]),
                "mean_regret": _mean([e["ladder"][k]["mean_regret"] for e in complete]),
                "max_regret": max(e["ladder"][k]["max_regret"] for e in complete),
            }
            for k in LADDER if all(k in e["ladder"] for e in complete)
        }
        priced = [e["regret_split"] for e in complete
                  if not math.isnan(e.get("regret_split", float("nan")))]
        if priced:
            pooled["model_regret_split_sample"] = _mean(priced)
            pooled["model_regret_worst"] = max(priced)
            pooled["model_regret_naive"] = _mean(
                [e["regret_naive"] for e in complete
                 if not math.isnan(e.get("regret_naive", float("nan")))]
            )
            pooled["model_matches_reference_best"] = sum(
                1 for e in complete if e.get("model_is_reference_best")
            )
    return {"pooled": pooled, "hands": hands}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True,
                        help="directory of received position records")
    parser.add_argument("--model-picks", default="",
                        help="JSON mapping 'hand string' -> action_key, to "
                             "price the shipped policy against a split-sample "
                             "best")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    records = load_positions(pathlib.Path(args.corpus))
    grouped = group_by_hand(records)
    picks = None
    if args.model_picks:
        raw = json.loads(pathlib.Path(args.model_picks).read_text(encoding="utf-8"))
        picks = {tuple(key.split()): value for key, value in raw.items()}

    result = report(grouped, picks)
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).write_text(text, encoding="utf-8")

    pooled = result["pooled"]
    print(f"{len(records)} position records, {pooled['hands']} hands, "
          f"{pooled['hands_with_all_batches']} with all {FULL} batches\n")
    if "gap_sd_median" in pooled:
        print("scatter across batches at 1,024 particles:")
        print(f"  per-action level   median SD {pooled['level_sd_median']:.3f}")
        print(f"  top-two gap        median SD {pooled['gap_sd_median']:.3f}"
              f"   worst {pooled['gap_sd_max']:.3f}")
        print(f"  sieve churn        median {pooled['sieve_churn_median']:.1%} "
              "of actions not measured in every batch\n")
        print("does the DECISION move? (k batches = 1,024k particles)")
        print(f"  {'k':>2} {'particles':>10} {'agrees w/ 12':>13} "
              f"{'mean regret':>12} {'worst':>8}")
        for k, row in sorted(pooled.get("ladder", {}).items()):
            print(f"  {k:2d} {row['equivalent_particles']:10,} "
                  f"{row['agreement']:12.1%} {row['mean_regret']:12.4f} "
                  f"{row['max_regret']:8.3f}")
        if "model_regret_split_sample" in pooled:
            print(f"\nshipped policy, priced on held-out batches:")
            print(f"  mean regret {pooled['model_regret_split_sample']:.4f} "
                  f"points/hand   worst {pooled['model_regret_worst']:.3f}")
            print(f"  matches the 12-batch best on "
                  f"{pooled['model_matches_reference_best']}/"
                  f"{pooled['hands_with_all_batches']}")
    if args.out:
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
