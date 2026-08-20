"""Spend particles where the ranking is still open, and nowhere else.

A fixed schedule -- N batches of the top K -- is wrong at both ends. On
`Ah Kh Ts 6s 4h` one batch of thirty-two candidates leaves a single survivor and
the other eleven batches buy nothing; on `As Kd Kh Ts 7s` twelve batches leave
two candidates 0.038 apart and a hundred more would not separate them. Measured
across twenty openings the requirement spans a factor of forty, and no constant
fits that.

This replaces the constant with a test. Measure a batch, drop every candidate
more than four standard errors behind the leader, measure the survivors again,
stop when one remains or the budget runs out.

## Four sigma, and why it does not grow

The bar is four standard errors throughout; what shrinks is how many points that
is, because the standard error falls as the square root of the particles spent:

    1 batch    1.90 points        8 batches   0.67
    2 batches  1.34              12 batches   0.55
    4 batches  0.95              22 batches   0.40

Sequential testing usually needs a threshold that grows with the number of looks,
to stop repeated chances at a false rejection from accumulating. Not at this
bar: a one-sided four-sigma test errs about 3e-5 of the time, so twelve looks
over thirty candidates expects 0.01 false eliminations. The error budget is
spent a hundred times over elsewhere.

## Two things this gets right that the fixed schedule did not

**The candidate set is measured, not assumed.** The top-K it replaced was
chosen by a single 1,024-particle batch -- the measurement this project
concluded cannot rank anything -- so a true best sitting at K+1 was invisible by
construction. On `Ad Kd Qs 9c 7h` eighteen candidates were still within four
sigma after one batch; a top-ten cut had already thrown eight of them away.

**Candidates are compared over the batches they share.** A candidate that
survives into round five has five batches; one eliminated in round two has two.
Pooling all of a candidate's batches against all of the leader's would judge the
newcomer at the leader's precision, on evidence never gathered about it. Only
batches measuring both count, and they are weighted by particle count so a
4,096-particle batch is worth four 1,024-particle ones -- in the estimate and in
its variance alike.

## Ordering

One batch per open hand per pass, never one hand to completion. The first
version of this ran each hand to its end and stalled: a worker that reached an
opening needing two and a half million particles stayed there forever while
every hand queued behind it waited -- two of them already past four sigma and
needing nothing but a convergence check -- and a third worker that had finished
its list sat idle. Round-robin costs an unfinishable hand one batch a pass
instead of a queue.

## Scale of the saving

Simulated against the twenty openings, elimination costs 40-55% of twelve
batches of the top ten, and three of the twenty finish on the first batch where
the fixed schedule paid twelve. The projection for all 134,459 canonical T0
openings is roughly $31,000 of GCP Spot against $67,000 -- see
`docs/t0_particle_noise_20260820.md` for where the particle counts come from and
`docs/gpu_playout_investigation_20260820.md` for what else the money could buy.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# Top-two gap scatter of one 1,024-particle batch, measured over twelve openings
# on 2026-08-20; the 4,096-particle figure was 0.231, and the two agree with the
# square-root law to within 3%.
GAP_SD_AT_REFERENCE = 0.475
REFERENCE_SAMPLES = 1024
STOP_SIGMA = 4.0
# In reference-particle units, not batches, so an opening carrying 4,096-particle
# work is charged for what it cost rather than for how it was filed.
DEFAULT_BUDGET = 120.0


class Batch(dict):
    """One measurement pass: ``{"samples": int, "scores": {action: float}}``."""


def load_batch(record: Mapping[str, Any], default_samples: int = REFERENCE_SAMPLES
               ) -> Batch:
    """Read a stored batch, accepting either the worker or the duel file shape."""

    scores = (record["runs"][0]["scores"] if "runs" in record
              else record["scores"])
    return Batch(samples=int(record.get("samples", default_samples)),
                 scores={key: float(value) for key, value in scores.items()})


def pooled_means(batches: Sequence[Batch]) -> dict[str, float]:
    """Each action's particle-weighted mean over every batch that measured it."""

    weighted: dict[str, list[float]] = {}
    for batch in batches:
        for key, value in batch["scores"].items():
            row = weighted.setdefault(key, [0.0, 0.0])
            row[0] += batch["samples"] * value
            row[1] += batch["samples"]
    return {key: total / mass for key, (total, mass) in weighted.items()}


def paired_gap(batches: Sequence[Batch], leader: str, key: str
               ) -> tuple[float, int]:
    """Leader-minus-candidate over the batches measuring both, and their particles.

    Paired rather than pooled because every batch scores its candidates against
    the same sampled worlds: the shared draw cancels in the difference, and a
    difference taken across batches would put it back.
    """

    shared = [b for b in batches
              if key in b["scores"] and leader in b["scores"]]
    if not shared:
        return float("nan"), 0
    mass = sum(b["samples"] for b in shared)
    gap = sum(b["samples"] * (b["scores"][leader] - b["scores"][key])
              for b in shared) / mass
    return gap, mass


def standard_error(particles: int) -> float:
    """The scatter of a gap measured with this many particles."""

    return GAP_SD_AT_REFERENCE / (particles / REFERENCE_SAMPLES) ** 0.5


def survivors(batches: Sequence[Batch], sigma: float = STOP_SIGMA
              ) -> tuple[list[str], str, dict[str, float]]:
    """Actions still within ``sigma`` standard errors of the leader."""

    means = pooled_means(batches)
    if not means:
        return [], "", {}
    leader = max(means, key=lambda key: means[key])
    alive = [leader]
    for key in means:
        if key == leader:
            continue
        gap, particles = paired_gap(batches, leader, key)
        if particles and gap < sigma * standard_error(particles):
            alive.append(key)
    return alive, leader, means


def particles_spent(batches: Sequence[Batch]) -> float:
    """The most any single action has had spent on it, in reference units."""

    per_key: dict[str, int] = {}
    for batch in batches:
        for key in batch["scores"]:
            per_key[key] = per_key.get(key, 0) + batch["samples"]
    return max(per_key.values(), default=0) / REFERENCE_SAMPLES


def particles_needed(gap: float, sigma: float = STOP_SIGMA) -> float:
    """Reference particles a gap of this size needs before it clears the bar.

    Quadratic in the reciprocal of the gap, which is why the twenty openings
    spanned a factor of forty: 3k particles for a 1.19-point gap, 2,529k for
    0.038.  Worth computing before committing to finish a hand.
    """

    if gap <= 0:
        return float("inf")
    return (sigma * GAP_SD_AT_REFERENCE / gap) ** 2


def verdict(batches: Sequence[Batch], budget: float = DEFAULT_BUDGET
            ) -> dict[str, Any]:
    """Whether this opening is finished, and on what evidence."""

    alive, leader, means = survivors(batches)
    spent = particles_spent(batches)
    ordered = sorted(means, key=lambda key: -means[key])
    gap = (means[ordered[0]] - means[ordered[1]]) if len(ordered) > 1 else float("inf")
    if len(alive) <= 1:
        state = "converged"
    elif spent >= budget:
        state = "budget"
    else:
        state = "open"
    return {
        "state": state,
        "best": leader,
        "alive": alive,
        "gap": gap,
        "particles_spent": spent,
        "particles_needed": particles_needed(gap),
        "means": means,
    }


def read_batches(sources: Iterable[tuple[pathlib.Path, str]],
                 default_samples: int = REFERENCE_SAMPLES) -> list[Batch]:
    """Every batch under the given (directory, glob) pairs, unreadable ones skipped.

    A worker killed mid-write can leave a partial file; those are skipped rather
    than raised on, because a run that dies because one of two hundred files is
    truncated is worse than one that measures a hand twice.
    """

    batches: list[Batch] = []
    for folder, pattern in sources:
        if folder is None or not pathlib.Path(folder).is_dir():
            continue
        for path in sorted(pathlib.Path(folder).glob(pattern)):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, KeyError):
                continue
            batches.append(load_batch(record, default_samples))
    return batches
