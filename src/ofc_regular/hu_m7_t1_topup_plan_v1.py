"""Build the immutable M7v6 T1 second-seat +7,000-root top-up plan.

This does not supersede `hu_m7_t1_256_plan_v1`; it *extends* one of its two
corpora.  The 18,000-root second-seat corpus that generator produced is
labelled, verified and on disk.  What follows it is the measurement that sent
this plan back to the fleet:

* Retrained on the relabelled (fl_ev 9.6) corpus the second seat's held-out
  regret improved by +0.0156 (0.1964 -> 0.1808) and it then **lost** the mirror
  match, -0.0705 over 1,500 deals (-0.1240 at the frozen learning rate).
* Isolating with the protocol fixed and only the root count varied -- on the
  OLD (fl_ev 10.227) corpus, so no relabel effect can leak in -- the second
  seat reads 0.1707 at 18,000 roots against 0.1547 at 25,000.  That **0.0160
  is the relabel's entire gain, spent on the reduction to 18,000**.  The
  2026-08-16 sizing said the data-scaling curve flattens below 25,000, which is
  a claim about label quality; it did not price the fact that the incumbent
  this corpus has to beat was itself trained on 25,000 roots.
* The first seat is insensitive to the same change (0.2042 vs 0.2019, 0.0023),
  so it stays at 18,000 and **this plan is second-seat only**.  There is no
  pair here; a first-seat top-up would buy a difference nothing measured.

So this run buys back exactly the root count, at exactly the 18k corpus's
settings, so that the two corpora merge into one 25,000-root corpus.  Every
field that a merge requires to be identical is not merely written to the same
value -- it is taken from the 18k generator's own plan body and then checked
back against a second copy of it built from the same package (see
`validate_merges_with_extended_corpus`).  A drifted pin would produce a corpus
whose rows were labelled by two different teachers under one name.

The one thing that MUST differ is the seeds, and non-overlap is the property
this plan exists to guarantee: the worker derives a position's hand from
`hand_seed_base + offset`, so a seed shared with the 18k corpus produces a
duplicate position that is invisible in the merged output -- identical rows
that read as two independent observations and quietly double their own weight.

Seed allocation, checked at build time rather than asserted here:

* hand `[952,100,000, 952,107,000)`, behavior `[952,600,000, 952,607,000)`.
  Deliberately inside the same 952,000,000 family as the 18k corpus, because
  the two runs are one corpus; what a plan may never share is a seed, and the
  four intervals the merged corpus draws from are pairwise disjoint:
  18k hand `[952,000,000, 952,018,000)`, top-up hand `[952,100,000,
  952,107,000)`, 18k behavior `[952,500,000, 952,518,000)`, top-up behavior
  `[952,600,000, 952,607,000)`.
* eval base `12,100,000`.  Note what the base alone does not tell you: the
  worker's evaluation seed is `eval_seed_base + offset * 7`, so an eval block
  is **seven times wider than its position count** and the 18k corpus's block
  reaches 12,125,993 -- past this base.  The two seed SETS are still disjoint,
  because they are arithmetic progressions of the same stride with different
  residues (12,000,000 is 5 mod 7, 12,100,000 is 3 mod 7).  This module proves
  that exactly instead of inferring it from the bases, and a future top-up
  wanting clear *ranges* needs seven times its position count of headroom, not
  one times.

Everything else is the 18k second-seat plan: 256 particles, all 27 actions
scored, one label run a position, the coarse T2 pair answering the nested
replies with the full-precision pair pinned beside it, fl_ev 9.6 at 14 cards.

Output is write-once.  Existing plans are evidence and are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from ofc_regular import hu_m7_t1_256_plan_v1 as corpus
from ofc_regular import hu_m7_t1_1024_plan_v1 as superseded
from ofc_regular import hu_m7_t2_2048_plan_v1 as t2_corpus

# The consumer's own names for the field group it holds all-or-nothing and the
# ones it refuses outside T0.  Imported rather than copied so this generator
# cannot drift from the worker that has to accept what it writes.
from ofc_regular.hu_m31_label_gen_worker_v1 import (
    FAST_T2_FIELDS,
    PRUNING_SAFETY_FIELDS,
    RACE_FIELDS,
)
from ofc_regular.hu_m7_t1_256_plan_v1 import (  # noqa: F401
    DEFAULT_PACKAGE_IDENTITY,
    EXPECTED_LEDGER_SHA256,
    EXPECTED_RUNTIME_SHA256,
    EXPECTED_WHEELHOUSE_SHA256,
    PLAN_SCHEMA,
    ExpectedPackageIdentity,
    PackageAudit,
    PlanValidationError,
    _require,
    audit_package,
    make_shards,
    rendered_json_bytes,
    sha256_bytes,
)

PROVENANCE_SCHEMA = "hu_m7_t1_second_topup_plan_provenance_v1"
PLAN_SET_SCHEMA = "hu_m7_t1_second_topup_plan_set_v1"

SEAT = "second"
POSITION_COUNT = 7_000
SHARD_COUNT = 46

# Taken from the corpus this extends rather than restated: these are the fields
# whose equality is what makes two label runs one corpus.
SAMPLES = corpus.SAMPLES
SEEDS_PER_POSITION = corpus.SEEDS_PER_POSITION
BEHAVIOR_SEED_OFFSET = corpus.BEHAVIOR_SEED_OFFSET
FL_EV_CARDS = corpus.FL_EV_CARDS
FL_EV_VALUE = corpus.FL_EV_VALUE

EXTENDS_JOB_ID = corpus.SECOND_JOB_ID
EXTENDS_POSITION_COUNT = corpus.POSITION_COUNT
MERGED_POSITION_COUNT = EXTENDS_POSITION_COUNT + POSITION_COUNT

HAND_SEED_BASE = 952_100_000
EVAL_SEED_BASE = 12_100_000

# `hu_m31_label_gen_worker_v1` computes a position's evaluation seed as
# `eval_seed_base + offset * 7 + trial * 3_000_017`.  With one label run a
# position the trial term is always zero, which is what makes the exact
# progression check below sound; `check_seed_allocation` refuses to run if that
# stops being true.
EVAL_SEED_STRIDE = 7
EVAL_SEED_TRIAL_STRIDE = 3_000_017

JOB_ID = "m7v6-t1second-topup7k-256p"
PLAN_FILENAME = "worker_plan_m7v6_t1second_topup7k_256p.json"
MANIFEST_FILENAME = "plan_set_m7v6_t1second_topup7k_256p.json"

# A base the ledger records without a count is reserved this wide, so a block
# nobody sized cannot be walked into by arithmetic.  Every such base is at
# least 4,000,000 from the one this plan takes, so the reservation costs
# nothing here; it is written down so a LATER allocation gets the same refusal.
_UNRECORDED_EXTENT = 1_000_000

# Every hand-seed allocation this project has recorded, with its source.  The
# live ones are read from the generators that own them -- a copied number can
# go stale without anything failing -- and the rest come from the seed ledger in
# docs/hu_m7_cascade_20260806.md.  The fourth element says whether the block
# follows the +500,000 behavior-offset convention, which decides whether a
# second interval is reserved beside it.
_RECORDED_ALLOCATIONS: tuple[tuple[str, int, int, bool], ...] = (
    (
        "m7v6 T1 256p corpora, both seats (the corpus this extends)",
        corpus.HAND_SEED_BASE,
        corpus.POSITION_COUNT,
        True,
    ),
    (
        "m7v6 T1 1024p plans (written 2026-08-15, never run)",
        superseded.HAND_SEED_BASE,
        superseded.POSITION_COUNT,
        True,
    ),
    (
        "m7v5 T2 2048p corpora",
        t2_corpus.HAND_SEED_BASE,
        t2_corpus.POSITION_COUNT,
        True,
    ),
    ("worker_plan_t3first_512p", 940_000_000, 50_000, True),
    ("m7v4 T3 corpora, both seats", 942_000_000, 18_000, True),
    ("T3 v7 match-gate deals, first seat", 945_000_000, 20_004, True),
    ("T3 v7 match-gate deals, second seat", 946_000_000, 20_004, True),
    ("generation-1 fleet, all eight street-seats", 960_000_000, 25_000, True),
    ("the reference plans", 970_000_000, 904, True),
    ("worker_plan_t0miniref_1024p", 971_000_000, 100, True),
    ("T4 v6 deals (extent not recorded)", 930_000_000, _UNRECORDED_EXTENT, False),
    ("doc-recorded base (extent not recorded)", 956_000_000, _UNRECORDED_EXTENT, False),
)

# Evaluation-seed progressions already spent, as (label, base, positions).
_RECORDED_EVAL_PROGRESSIONS: tuple[tuple[str, int, int], ...] = (
    ("m7v6 T1 256p corpora", corpus.EVAL_SEED_BASE, corpus.POSITION_COUNT),
    ("m7v6 T1 1024p plans", superseded.EVAL_SEED_BASE, superseded.POSITION_COUNT),
    ("m7v5 T2 2048p corpora", t2_corpus.EVAL_SEED_BASE, t2_corpus.POSITION_COUNT),
)

# The older eval bases the cascade ledger records, none of which has a position
# count written down.  A new base has to sit above all of them, which at this
# distance it does by five million.
_DOC_RECORDED_EVAL_BASES: tuple[int, ...] = (5_000_000, 6_000_000, 8_000_000, 9_000_000)

# The fields a top-up is allowed to change.  Everything else is compared for
# equality against the extended corpus's plan, and the key SETS are compared
# too, so a field added to the 18k plan later has to be answered here rather
# than silently dropped from the run that continues it.
_MUST_DIFFER: tuple[str, ...] = (
    "job_id",
    "hand_seed_base",
    "eval_seed_base",
    "shards",
    "provenance",
)


def reserved_seed_intervals() -> tuple[tuple[str, int, int], ...]:
    """Every hand/behavior interval already spoken for, as (label, lo, hi)."""

    rows: list[tuple[str, int, int]] = []
    for label, base, count, paired_behavior in _RECORDED_ALLOCATIONS:
        rows.append((f"{label}: hand", base, base + count))
        if paired_behavior:
            behavior = base + BEHAVIOR_SEED_OFFSET
            rows.append((f"{label}: behavior", behavior, behavior + count))
    return tuple(rows)


def _intervals_disjoint(a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> bool:
    return a_hi <= b_lo or b_hi <= a_lo


def eval_seed_span(base: int, positions: int) -> tuple[int, int]:
    """Half-open span an evaluation-seed progression touches, stride included.

    The span is not the position count: the worker strides by seven, so a
    7,000-position block reaches 48,993 past its base.
    """

    _require(positions > 0, "an eval progression needs at least one position")
    return base, base + EVAL_SEED_STRIDE * (positions - 1) + 1


def eval_progressions_disjoint(
    base_a: int, positions_a: int, base_b: int, positions_b: int
) -> bool:
    """Exact disjointness of two same-stride evaluation-seed progressions.

    Same stride, so they either never collide (different residue) or collide
    wherever their spans overlap.  Deciding it on the spans alone would reject
    allocations that are in fact clean -- which is exactly the pair this plan
    forms with the corpus it extends.
    """

    if (base_a - base_b) % EVAL_SEED_STRIDE != 0:
        return True
    lo_a, hi_a = eval_seed_span(base_a, positions_a)
    lo_b, hi_b = eval_seed_span(base_b, positions_b)
    return _intervals_disjoint(lo_a, hi_a, lo_b, hi_b)


def check_seed_allocation() -> dict[str, Any]:
    """Refuse to render at all unless this block is free.  Returns the audit."""

    _require(
        SEEDS_PER_POSITION == 1,
        "the evaluation-seed check assumes one label run a position; at "
        f"{SEEDS_PER_POSITION} the trial stride {EVAL_SEED_TRIAL_STRIDE} joins "
        "the progression and this guard no longer decides it",
    )

    mine = {
        "hand": (HAND_SEED_BASE, HAND_SEED_BASE + POSITION_COUNT),
        "behavior": (
            HAND_SEED_BASE + BEHAVIOR_SEED_OFFSET,
            HAND_SEED_BASE + BEHAVIOR_SEED_OFFSET + POSITION_COUNT,
        ),
    }
    _require(
        _intervals_disjoint(*mine["hand"], *mine["behavior"]),
        "this plan's own hand and behavior intervals overlap",
    )

    checked: list[dict[str, Any]] = []
    for label, lo, hi in reserved_seed_intervals():
        for which, (my_lo, my_hi) in mine.items():
            _require(
                _intervals_disjoint(my_lo, my_hi, lo, hi),
                f"this plan's {which} interval [{my_lo},{my_hi}) overlaps "
                f"{label} [{lo},{hi}); a shared seed makes duplicate positions "
                "that are invisible in the merged output",
            )
        checked.append({"block": label, "interval": [lo, hi]})

    eval_lo, eval_hi = eval_seed_span(EVAL_SEED_BASE, POSITION_COUNT)
    eval_checked: list[dict[str, Any]] = []
    for label, base, positions in _RECORDED_EVAL_PROGRESSIONS:
        other_lo, other_hi = eval_seed_span(base, positions)
        _require(
            eval_progressions_disjoint(
                EVAL_SEED_BASE, POSITION_COUNT, base, positions
            ),
            f"this plan's eval progression from {EVAL_SEED_BASE} collides with "
            f"{label} from {base}",
        )
        eval_checked.append(
            {
                "block": label,
                "base": base,
                "span": [other_lo, other_hi],
                "spans_overlap": not _intervals_disjoint(
                    eval_lo, eval_hi, other_lo, other_hi
                ),
                "residue_mod_stride": base % EVAL_SEED_STRIDE,
            }
        )
    for base in _DOC_RECORDED_EVAL_BASES:
        _require(
            EVAL_SEED_BASE > base,
            f"eval base {EVAL_SEED_BASE} is not above doc-recorded base {base}, "
            "whose extent nobody wrote down",
        )

    return {
        "hand_seed_interval": list(mine["hand"]),
        "behavior_seed_interval": list(mine["behavior"]),
        "eval_seed_base": EVAL_SEED_BASE,
        "eval_seed_stride": EVAL_SEED_STRIDE,
        "eval_seed_span": [eval_lo, eval_hi],
        "eval_residue_mod_stride": EVAL_SEED_BASE % EVAL_SEED_STRIDE,
        "partial_seed_block_reuse": False,
        "checked_against_hand_blocks": checked,
        "checked_against_eval_progressions": eval_checked,
        "note": (
            "The 952,000,000 family is shared with the 18k corpus on purpose -- "
            "the two runs are one corpus -- but no seed is. All four intervals "
            "the merged corpus draws from are pairwise disjoint. The eval "
            "blocks' SPANS do overlap, because the worker strides evaluation "
            "seeds by 7 and the 18k block therefore reaches 12,125,993; the "
            "seed sets are still disjoint by residue (5 vs 3 mod 7) and this "
            "generator proves that rather than assuming it from the bases."
        ),
    }


def _topup_plan(audit: PackageAudit) -> dict[str, Any]:
    """The 18k second-seat plan body, reseeded and resharded for the top-up.

    Built from the 18k generator's own `_base_plan` rather than from a second
    copy of its field list.  That is the point: a top-up whose pins were
    retyped could be one digest away from labelling its rows with a different
    teacher than the corpus it joins, and nothing downstream would see it.
    """

    plan = corpus._base_plan(audit, seat=SEAT)
    inherited = plan["provenance"]

    plan["job_id"] = JOB_ID
    plan["hand_seed_base"] = HAND_SEED_BASE
    plan["eval_seed_base"] = EVAL_SEED_BASE
    plan["shards"] = make_shards(
        position_count=POSITION_COUNT, shard_count=SHARD_COUNT
    )
    plan["provenance"] = {
        "schema": PROVENANCE_SCHEMA,
        "generation": "m7v6",
        "extends": {
            "generator": "ofc_regular.hu_m7_t1_256_plan_v1",
            "job_id": EXTENDS_JOB_ID,
            "positions": EXTENDS_POSITION_COUNT,
            "merged_positions": MERGED_POSITION_COUNT,
            "relationship": (
                "top-up, not supersession: these rows are labelled under the "
                "18k plan's exact settings and merge with it into one corpus."
            ),
        },
        "package_audit": audit.provenance(),
        "purchase": {
            "position_count": POSITION_COUNT,
            "teacher_particles": SAMPLES,
            "reason": (
                "The second seat's held-out regret improved +0.0156 on the "
                "relabelled corpus and the mirror match still went -0.0705 "
                "over 1,500 deals. Isolating on the OLD corpus with only the "
                "root count varied, the second seat reads 0.1707 at 18,000 "
                "against 0.1547 at 25,000: the 0.0160 reduction cost is the "
                "whole relabel gain. This buys the count back so the 25,000 "
                "the incumbent was trained on is matched. Owner decision "
                "2026-08-18; docs/hu_m7_roadmap_20260818.md."
            ),
            "declined": (
                "A first-seat top-up. The same isolation reads 0.2042 at "
                "18,000 against 0.2019 at 25,000 (0.0023), so the first seat "
                "is insensitive to root count and stays where it is. Buying "
                "it anyway would spend fleet time on a difference nothing "
                "measured, and would put a second unpaired corpus in flight."
            ),
            "unchanged_from_the_extended_corpus": (
                "particles, seeds per position, the full 27-action fan, the "
                "coarse T2 pair with the full-precision pair beside it, the "
                "T3/T4 pins, and fl_ev 9.6 at 14 cards. Checked field by "
                "field against a plan rebuilt from the same package, not "
                "merely written to the same values."
            ),
        },
        "seed_audit": check_seed_allocation(),
        # Identical by construction: the continuation IS the 18k corpus's, so
        # these blocks are carried over rather than restated in words that
        # could drift from it.
        "continuation": inherited["continuation"],
        "fantasyland": inherited["fantasyland"],
        "sharding": {
            "shards": SHARD_COUNT,
            "concurrency_is_not_authorized_by_this_plan": True,
            "recommended_max_live_c4_standard_8": 58,
            "intended_workers_per_vm": 6,
            "waves_at_recommended_max_live": 1,
            "sizing_reason": (
                "A second-seat root at 256 particles with coarse T2 replies is "
                "about 151 core-s (measured under 10-way contention and "
                "recorded by the 18k plan). A 152-root shard at six workers is "
                "about 1.1 hours, the same shard size the 18k run proved "
                "against the six-hour watchdog, and 46 shards fit in a single "
                "wave under the same 58-VM ceiling that run used in two."
            ),
        },
    }
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    """Everything a lone second-seat T1 top-up plan has to be, on its own."""

    _require(plan.get("schema") == PLAN_SCHEMA, "plan schema drifted")
    _require(plan.get("job_id") == JOB_ID, "plan job id drifted")
    _require(plan.get("street") == "T1", "plan is not T1")
    _require(plan.get("seat") == SEAT, "plan is not the second seat")
    _require(plan.get("samples") == SAMPLES, "plan is not 256p")
    _require(
        plan.get("seeds_per_position") == SEEDS_PER_POSITION,
        "plan does not have exactly one label run",
    )
    _require(plan.get("hand_seed_base") == HAND_SEED_BASE, "hand seed base drifted")
    _require(
        plan.get("behavior_seed_offset") == BEHAVIOR_SEED_OFFSET,
        "behavior seed offset drifted",
    )
    _require(plan.get("eval_seed_base") == EVAL_SEED_BASE, "eval seed base drifted")

    shards = plan.get("shards")
    _require(
        isinstance(shards, list) and len(shards) == SHARD_COUNT,
        f"plan does not have {SHARD_COUNT} shards",
    )
    expected_start = 0
    seen: set[str] = set()
    for shard in shards:
        shard_id = shard.get("shard_id")
        _require(
            isinstance(shard_id, str) and shard_id not in seen,
            "shard id is missing or duplicated",
        )
        seen.add(shard_id)
        _require(
            shard.get("start") == expected_start,
            f"shard {shard_id} is not contiguous",
        )
        count = shard.get("count")
        _require(isinstance(count, int) and count > 0, "bad shard count")
        expected_start += count
    _require(expected_start == POSITION_COUNT, f"plan does not cover {POSITION_COUNT}")

    _require("probe" not in plan, "plan carries a probe block")
    # The second seat never has a T1 reply ahead of it; the worker refuses a
    # plan that pins one, because a pinned-but-unreachable model reads as a
    # dependency the label does not have.
    for field in ("t1_second_model", "t1_second_model_sha256"):
        _require(field not in plan, f"second-seat plan pins {field}")
    # All 27 actions are scored. Narrowing was declined for the 18k corpus and
    # a top-up that narrowed would not be the same corpus; the worker gates
    # these fields to T0 in any case.
    for field in ("prefilter_samples", "prefilter_keep", *PRUNING_SAFETY_FIELDS,
                  *RACE_FIELDS):
        _require(field not in plan, f"plan carries the T0-only field {field}")
    # The coarse pair is all-or-nothing in the worker; a plan naming half of it
    # would run one T2 seat coarse and the other exact.
    for field in FAST_T2_FIELDS:
        _require(field in plan, f"plan is missing {field}")
    # The full-precision pair must survive alongside it: the engine loads both,
    # and the worker refuses a T1 plan without the full pins.
    for field in ("t2_first_model", "t2_second_model", "t3_first_model",
                  "t3_second_model", "t4_model"):
        _require(field in plan, f"plan is missing {field}")
    _require(
        plan.get("fl_ev_cards") == FL_EV_CARDS
        and plan.get("fl_ev_value") == FL_EV_VALUE,
        "plan FL contract drifted",
    )
    _require(
        plan.get("provenance", {}).get("schema") == PROVENANCE_SCHEMA,
        "plan provenance schema drifted",
    )


def validate_merges_with_extended_corpus(
    plan: dict[str, Any], extended: dict[str, Any]
) -> None:
    """The top-up and the corpus it joins differ in seeds and nothing else.

    `extended` is the 18k second-seat plan built from the SAME package audit,
    so every model digest on both sides came from the same bytes.  Comparing
    key sets as well as values is what makes this hold in the future: a field
    added to the 18k plan later fails here instead of quietly going missing
    from the run that continues it.
    """

    _require(
        set(plan) == set(extended),
        "top-up and extended corpus plans do not carry the same fields: "
        f"only in top-up={sorted(set(plan) - set(extended))}, "
        f"only in corpus={sorted(set(extended) - set(plan))}",
    )
    for field in sorted(set(plan) - set(_MUST_DIFFER)):
        _require(
            plan[field] == extended[field],
            f"{field} differs from the corpus this top-up joins; the merged "
            "rows would not share a teacher",
        )
    for field in ("job_id", "hand_seed_base", "eval_seed_base"):
        _require(
            plan[field] != extended[field],
            f"{field} is identical to the corpus this top-up joins; a shared "
            "seed block makes duplicate positions that are invisible in the "
            "merged output",
        )
    _require(
        plan["shards"] != extended["shards"],
        "top-up shards are the extended corpus's shards",
    )

    # The seed evidence, computed from the two plans' own fields rather than
    # from this module's constants.
    intervals = {}
    for name, side in (("topup", plan), ("corpus", extended)):
        positions = sum(row["count"] for row in side["shards"])
        base = side["hand_seed_base"]
        intervals[f"{name}_hand"] = (base, base + positions)
        intervals[f"{name}_behavior"] = (
            base + side["behavior_seed_offset"],
            base + side["behavior_seed_offset"] + positions,
        )
    names = sorted(intervals)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            _require(
                _intervals_disjoint(*intervals[left], *intervals[right]),
                f"{left} {list(intervals[left])} overlaps {right} "
                f"{list(intervals[right])}",
            )
    _require(
        eval_progressions_disjoint(
            plan["eval_seed_base"],
            sum(row["count"] for row in plan["shards"]),
            extended["eval_seed_base"],
            sum(row["count"] for row in extended["shards"]),
        ),
        "the two eval-seed progressions share a seed",
    )


def build_plan(
    package_dir: pathlib.Path,
    *,
    expected_identity: ExpectedPackageIdentity = DEFAULT_PACKAGE_IDENTITY,
) -> tuple[dict[str, Any], PackageAudit]:
    audit = audit_package(package_dir, expected_identity=expected_identity)
    plan = _topup_plan(audit)
    validate_plan(plan)
    validate_merges_with_extended_corpus(plan, corpus._base_plan(audit, seat=SEAT))
    return plan, audit


def write_plan_once(
    package_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    expected_identity: ExpectedPackageIdentity = DEFAULT_PACKAGE_IDENTITY,
) -> dict[str, Any]:
    plan, audit = build_plan(package_dir, expected_identity=expected_identity)
    rendered = {PLAN_FILENAME: rendered_json_bytes(plan)}
    manifest = {
        "schema": PLAN_SET_SCHEMA,
        "generator": "ofc_regular.hu_m7_t1_topup_plan_v1",
        "package_audit": audit.provenance(),
        "plans": {
            "second": {
                "filename": PLAN_FILENAME,
                "job_id": JOB_ID,
                "sha256": sha256_bytes(rendered[PLAN_FILENAME]),
            }
        },
        "topup_contract": {
            "seat": SEAT,
            "positions": POSITION_COUNT,
            "samples": SAMPLES,
            "shards": SHARD_COUNT,
            "extends_job_id": EXTENDS_JOB_ID,
            "extends_positions": EXTENDS_POSITION_COUNT,
            "merged_positions": MERGED_POSITION_COUNT,
            "hand_seed_interval": [HAND_SEED_BASE, HAND_SEED_BASE + POSITION_COUNT],
            "behavior_seed_interval": [
                HAND_SEED_BASE + BEHAVIOR_SEED_OFFSET,
                HAND_SEED_BASE + BEHAVIOR_SEED_OFFSET + POSITION_COUNT,
            ],
            "eval_seed_base": EVAL_SEED_BASE,
            "eval_seed_span": list(eval_seed_span(EVAL_SEED_BASE, POSITION_COUNT)),
            "seed_blocks_disjoint_from_extended_corpus": True,
            "model_pins_identical_to_extended_corpus": True,
            "coarse_t2_replies": True,
            "first_seat_topup": False,
            "current_profile_changed": False,
        },
    }
    rendered[MANIFEST_FILENAME] = rendered_json_bytes(manifest)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: output_dir / name for name in rendered}
    collisions = sorted(str(path) for path in targets.values() if path.exists())
    _require(not collisions, f"write-once output already exists: {collisions}")
    for name in (PLAN_FILENAME, MANIFEST_FILENAME):
        try:
            with targets[name].open("xb") as stream:
                stream.write(rendered[name])
                stream.flush()
        except FileExistsError as error:
            raise PlanValidationError(
                f"write-once race: output appeared while writing {targets[name]}"
            ) from error
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "write"), required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args(argv)

    package_dir = pathlib.Path(args.package_dir)
    if args.mode == "audit":
        if args.out_dir:
            raise SystemExit("--out-dir is only valid in write mode")
        plan, audit = build_plan(package_dir)
        print(json.dumps({
            "schema": "hu_m7_t1_second_topup_plan_audit_summary_v1",
            "package_audit": audit.provenance(),
            "planned": {
                "seat": SEAT,
                "positions": POSITION_COUNT,
                "samples": SAMPLES,
                "shards": SHARD_COUNT,
                "hand_seed_base": HAND_SEED_BASE,
                "eval_seed_base": EVAL_SEED_BASE,
                "extends_job_id": EXTENDS_JOB_ID,
                "merged_positions": MERGED_POSITION_COUNT,
                "coarse_t2_replies": True,
            },
            "seed_audit": plan["provenance"]["seed_audit"],
            "writes_performed": False,
        }, indent=2, sort_keys=True))
        return 0

    if not args.out_dir:
        raise SystemExit("write mode requires --out-dir")
    print(json.dumps(write_plan_once(package_dir, pathlib.Path(args.out_dir)),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PACKAGE_IDENTITY",
    "EXPECTED_LEDGER_SHA256",
    "EXPECTED_RUNTIME_SHA256",
    "EXPECTED_WHEELHOUSE_SHA256",
    "EVAL_SEED_BASE",
    "EVAL_SEED_STRIDE",
    "EXTENDS_JOB_ID",
    "ExpectedPackageIdentity",
    "FL_EV_VALUE",
    "HAND_SEED_BASE",
    "JOB_ID",
    "MANIFEST_FILENAME",
    "MERGED_POSITION_COUNT",
    "PLAN_FILENAME",
    "POSITION_COUNT",
    "PackageAudit",
    "PlanValidationError",
    "SAMPLES",
    "SEAT",
    "SHARD_COUNT",
    "audit_package",
    "build_plan",
    "check_seed_allocation",
    "eval_progressions_disjoint",
    "eval_seed_span",
    "make_shards",
    "reserved_seed_intervals",
    "validate_merges_with_extended_corpus",
    "validate_plan",
    "write_plan_once",
]
