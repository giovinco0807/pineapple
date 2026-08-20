"""Twelve independent 1,024-particle observations of each of twenty openings.

## The doubt this answers

Twenty openings were solved locally on 2026-08-19 -- a 64-particle sieve keeping
sixteen with margin 2.4, then every survivor re-scored uniformly at 1,024
particles, no race -- and the ranking those runs produced is now carrying two
claims: that the shipped policy picks the measured best on 20 of 24 hands, and
that its mean regret is 0.107 points.  Both are read off a single batch a hand.

The gaps those batches report are small.  Nine of twenty-four separate first
from second by less than 0.5 and two by less than 0.2, the narrowest being 0.037
on ``Ad Ts 9c 7d 6d``.  A 1,024-particle batch is not obviously able to resolve
0.037, and nobody has measured whether it can.  Until someone does, "the model
picked the best action" is a statement about one draw of the particle stream.

## Why twelve at 1,024 rather than one each at 2,048 and 4,096

The obvious design re-scores at higher particle counts and watches the answer
move.  Averaging independent batches gets there by a shorter road: the estimate
is a mean over particles, so the mean of *k* independent 1,024-particle batches
has exactly the variance of one batch of 1,024*k*.  Twelve batches therefore
carry the whole ladder -- pair them for 2,048, take fours for 4,096, take all
twelve for a 12,288-particle reference -- and, unlike three fixed rungs, they
also measure the 1,024 scatter directly, which is the quantity actually in
doubt.

Two practical arguments point the same way.  A root at 4,096 particles over
thirty-two survivors is about eight core-hours, and the fleet's engine evaluates
a root on one core -- the T0 parallel patch of 2026-08-19 lives in a worktree
and is not in this package -- so a 4,096 rung would put single shards eight
hours deep into Spot preemption territory.  At 1,024 a root is under three
hours, which is the size the m7 fleet is built around.  And the same total
outlay buys more here: 240 roots at 1,024 is 640 core-hours where three rungs of
three seeds would have been 920.

The one thing this design does NOT reproduce is a single sieve pass per rung.
Each batch re-runs its own 64-particle sieve, so the surviving set can differ
between batches of the same hand, where a true 2,048-particle run would have
sieved once.  That is a difference in the conservative direction -- it adds
variation rather than hiding it -- and it turns the sieve's own stability into
something this run measures rather than assumes.  The analysis pairs batches on
the actions they have in common and records, per hand, how often the surviving
set moved at all.

## The shape of the plan

Twelve repeats of twenty hands are written as 240 explicit roots with
``seeds_per_position: 1``, not as twenty roots asked for twelve labels each.
The worker strides its eight per-VM workers over ROOTS, so twenty roots would
leave seven of eight idle while one worker ground through twelve batches in
series.  As 240 roots the fleet's own sizing rule applies unchanged: eight roots
a shard, one root a worker, a shard's wall time is one root's.

The repeats are interleaved rather than blocked -- root *i* is hand *i* mod 20 --
so a shard holds eight different hands.  A shard lost to preemption then costs
one batch from each of eight hands instead of eight batches from one, which is
the difference between a thinner estimate everywhere and a hand that cannot be
adjudicated at all.

Sieve and evaluation are identical to the local runs being checked: 64 / 16 /
2.4 and 1,024 uniform with no race.  Anything else would measure a different
schedule and answer a question nobody asked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

from ofc_regular.hu_m31_label_gen_worker_v1 import ROOT_SOURCE_EXPLICIT
from ofc_regular.hu_m7_t1_256_plan_v1 import (  # noqa: F401
    DEFAULT_PACKAGE_IDENTITY,
    PLAN_SCHEMA,
    ExpectedPackageIdentity,
    PackageAudit,
    PlanValidationError,
    _package_pin,
    _require,
    audit_package,
    make_shards,
    rendered_json_bytes,
    sha256_bytes,
)
from ofc_regular.hu_t0_akq500_explicit_plan_v1 import (
    _DOC_RECORDED_EVAL_BASES,
    _MODEL_PINS,
    _RECORDED_EVAL_PROGRESSIONS,
    _UNRECORDED_EXTENT,
    EVAL_SEED_STRIDE,
    eval_progressions_disjoint,
    eval_seed_span,
)

PROVENANCE_SCHEMA = "hu_t0_ladder20_explicit_plan_provenance_v1"
PLAN_SET_SCHEMA = "hu_t0_ladder20_explicit_plan_set_v1"

STREET = "T0"
SEAT = "first"

# The twenty openings under review are the first twenty of the akq500 hand
# file, which is the slice the local runs took.  Read from that file rather
# than copied here so the two cannot drift.
HAND_COUNT = 20
REPEATS = 12
POSITION_COUNT = HAND_COUNT * REPEATS
SEEDS_PER_POSITION = 1
SHARD_COUNT = 30

SAMPLES = 1_024
PREFILTER_SAMPLES = 64
PREFILTER_KEEP = 16
PREFILTER_MARGIN = 2.4

FL_EV_CARDS = 14
FL_EV_VALUE = 9.6

# Stride seven over 240 positions with one label run apiece touches
# 25,000,000..25,001,673 -- 1,674 wide, because the trial term that makes these
# blocks three million wide is not spent here.  Clear of everything: the label
# corpora reach 12,148,993, the akq500 plan holds 20,000,000..20,003,493, and
# the T0 restricted-evaluation experiments of 2026-08-19 spent 61M through 97M.
EVAL_SEED_BASE = 25_000_000

JOB_ID = "t0first-ladder20-1024p-x12"
PLAN_FILENAME = "worker_plan_t0first_ladder20_1024p_x12.json"
MANIFEST_FILENAME = "plan_set_t0first_ladder20_1024p_x12.json"

DEFAULT_HANDS_FILE = pathlib.Path("D:/ofc_data/t0_rung/akq500/hands.json")


def load_hands(path: pathlib.Path) -> tuple[list[list[str]], dict[str, Any]]:
    """Take the first ``HAND_COUNT`` hands of the akq500 file, validated.

    The slice is the point: these are the hands already solved locally, and a
    run that measured a different twenty would be checking numbers nobody
    quoted.  The file's digest is carried into the plan so a later reader can
    prove which twenty these were.
    """

    _require(path.is_file(), f"hand file is missing: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PlanValidationError(f"hand file is not UTF-8 JSON: {path}") from error
    _require(isinstance(payload, Mapping), "hand file must be a JSON object")
    _require("hands" in payload, "hand file is missing 'hands'")
    hands = payload["hands"]
    _require(isinstance(hands, list), "hands must be a list")
    _require(
        len(hands) >= HAND_COUNT,
        f"hand file holds {len(hands)} hands; this plan needs at least {HAND_COUNT}",
    )
    taken = hands[:HAND_COUNT]
    seen: set[tuple[str, ...]] = set()
    for index, hand in enumerate(taken):
        _require(
            isinstance(hand, list) and len(hand) == 5,
            f"hand {index} is not a list of five cards: {hand!r}",
        )
        for card in hand:
            _require(
                isinstance(card, str),
                f"hand {index} holds {card!r}, which is not a card string",
            )
        _require(len(set(hand)) == 5, f"hand {index} repeats a card: {hand!r}")
        key = tuple(hand)
        _require(key not in seen, f"hand {index} duplicates an earlier hand")
        seen.add(key)
    source = {
        "path": str(path),
        "sha256": digest,
        "bytes": len(raw),
        "count": len(hands),
        "taken": HAND_COUNT,
        "slice": "hands[:20], the twenty solved locally on 2026-08-19",
    }
    for key in ("seed", "filter", "dedup", "attempts"):
        if key in payload:
            source[key] = payload[key]
    return [list(hand) for hand in taken], source


def build_roots(hands: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    """``REPEATS`` interleaved passes over the hand list.

    Root *i* is hand *i* mod ``HAND_COUNT``, so consecutive roots -- and
    therefore the eight roots of a shard -- are eight different hands.  See the
    module docstring on why interleaved beats blocked when shards can be lost.
    """

    return [
        {"dealt_cards": list(hands[index % HAND_COUNT])}
        for index in range(POSITION_COUNT)
    ]


def _plan(audit: PackageAudit, roots: Sequence[Mapping[str, Any]],
          source: Mapping[str, Any]) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "job_id": JOB_ID,
        "street": STREET,
        "seat": SEAT,
        "samples": SAMPLES,
        "seeds_per_position": SEEDS_PER_POSITION,
        "eval_seed_base": EVAL_SEED_BASE,
        "root_source": ROOT_SOURCE_EXPLICIT,
        "roots": [dict(root) for root in roots],
        "engine_library": "native/libofc_hu_m3_engine.so",
        "engine_library_sha256": _package_pin(
            audit, "native/libofc_hu_m3_engine.so"
        ),
        "feature_encoder_library": "native/libofc_stage3_feature_encoder.so",
        "feature_encoder_library_sha256": _package_pin(
            audit, "native/libofc_stage3_feature_encoder.so"
        ),
        "prefilter_samples": PREFILTER_SAMPLES,
        "prefilter_keep": PREFILTER_KEEP,
        "prefilter_margin": PREFILTER_MARGIN,
        "fl_ev_cards": FL_EV_CARDS,
        "fl_ev_value": FL_EV_VALUE,
        "shards": make_shards(
            position_count=POSITION_COUNT, shard_count=SHARD_COUNT
        ),
    }
    for field, relative in _MODEL_PINS:
        plan[field] = relative
        plan[f"{field}_sha256"] = _package_pin(audit, relative)

    plan["provenance"] = {
        "schema": PROVENANCE_SCHEMA,
        "generation": "m7v7",
        "package_audit": audit.provenance(),
        "hand_source": dict(source),
        "purpose": (
            "Measure the scatter of a 1,024-particle T0 root batch, on the "
            "twenty openings whose single-batch rankings are carrying the "
            "claims 'the shipped policy picks the measured best on 20 of 24' "
            "and 'its mean regret is 0.107'. Nine of those twenty-four hands "
            "separate first from second by less than 0.5 and two by less than "
            "0.2, the narrowest 0.037, and no measurement says whether 1,024 "
            "particles resolve a gap that size."
        ),
        "design": {
            "repeats": REPEATS,
            "why_repeats_not_rungs": (
                "The mean of k independent 1,024-particle batches has exactly "
                "the variance of one batch of 1,024k, so twelve batches carry "
                "the whole ladder -- pairs for 2,048, fours for 4,096, all "
                "twelve for a 12,288-particle reference -- and additionally "
                "measure the 1,024 scatter directly, which fixed rungs cannot. "
                "They are also cheaper (640 core-hours against 920 for three "
                "rungs of three seeds) and they keep a root under three hours, "
                "where a 4,096 root is about eight and the package's engine "
                "scores a root on one core."
            ),
            "known_difference_from_true_rungs": (
                "Each batch re-runs its own 64-particle sieve, so the "
                "surviving set can differ between batches of one hand where a "
                "true 2,048-particle run would have sieved once. This adds "
                "variation rather than hiding it, and makes the sieve's own "
                "stability measurable; the analysis pairs batches on their "
                "common actions and records how often the set moved."
            ),
            "interleaved_roots": (
                "Root i is hand i mod 20, so a shard holds eight different "
                "hands and a shard lost to preemption costs one batch from "
                "each of eight rather than eight batches from one."
            ),
        },
        "root_schedule": {
            "prefilter_samples": PREFILTER_SAMPLES,
            "prefilter_keep": PREFILTER_KEEP,
            "prefilter_margin": PREFILTER_MARGIN,
            "evaluation_samples": SAMPLES,
            "race": None,
            "why_this_schedule": (
                "Identical to the local runs under review -- 64/16/2.4 then "
                "1,024 uniform, no race. A different schedule would measure a "
                "different thing and leave the quoted numbers unchecked."
            ),
        },
        "continuation": {field: relative for field, relative in _MODEL_PINS},
    }
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    """Refuse anything that is not the measurement this module describes."""

    _require(plan.get("schema") == PLAN_SCHEMA, "plan schema drifted")
    _require(plan.get("job_id") == JOB_ID, "job id drifted")
    _require(plan.get("street") == STREET, "street drifted")
    _require(plan.get("seat") == SEAT, "seat drifted")
    _require(plan.get("samples") == SAMPLES, "evaluation width drifted")
    _require(
        plan.get("seeds_per_position") == SEEDS_PER_POSITION,
        "seeds per position drifted; repeats are roots here, not trials",
    )
    _require(plan.get("root_source") == ROOT_SOURCE_EXPLICIT, "root source drifted")
    _require(plan.get("prefilter_samples") == PREFILTER_SAMPLES, "sieve width drifted")
    _require(plan.get("prefilter_keep") == PREFILTER_KEEP, "sieve keep drifted")
    _require(plan.get("prefilter_margin") == PREFILTER_MARGIN, "sieve margin drifted")
    _require("race_schedule" not in plan, "a raced plan cannot measure the race")
    _require(plan.get("fl_ev_cards") == FL_EV_CARDS, "fl_ev cards drifted")
    _require(plan.get("fl_ev_value") == FL_EV_VALUE, "fl_ev value drifted")
    _require(plan.get("eval_seed_base") == EVAL_SEED_BASE, "eval seed base drifted")

    roots = plan.get("roots")
    _require(isinstance(roots, list), "roots must be a list")
    _require(
        len(roots) == POSITION_COUNT,
        f"plan holds {len(roots)} roots; expected {POSITION_COUNT}",
    )
    for index, root in enumerate(roots):
        _require(isinstance(root, Mapping), f"root {index} is not an object")
        _require(
            set(root) == {"dealt_cards"},
            f"root {index} carries fields beyond dealt_cards: {sorted(root)}",
        )
        _require(len(root["dealt_cards"]) == 5, f"root {index} is not five cards")

    # Every hand appears exactly REPEATS times, and the repeats are interleaved.
    counts: dict[tuple[str, ...], int] = {}
    for root in roots:
        key = tuple(root["dealt_cards"])
        counts[key] = counts.get(key, 0) + 1
    _require(
        len(counts) == HAND_COUNT,
        f"plan covers {len(counts)} distinct hands; expected {HAND_COUNT}",
    )
    _require(
        set(counts.values()) == {REPEATS},
        f"repeat counts are uneven: {sorted(set(counts.values()))}",
    )
    for index in range(POSITION_COUNT):
        _require(
            roots[index]["dealt_cards"] == roots[index % HAND_COUNT]["dealt_cards"],
            f"root {index} breaks the interleave",
        )

    shards = plan.get("shards")
    _require(isinstance(shards, list), "shards must be a list")
    _require(
        len(shards) == SHARD_COUNT,
        f"plan holds {len(shards)} shards; expected {SHARD_COUNT}",
    )
    # Contiguous, gapless, and covering every root exactly once: a shard set
    # that merely sums to 240 could still skip a position and double another,
    # and the roots it skipped would be missing batches nobody counted.
    expected_start = 0
    shard_ids: set[str] = set()
    for shard in shards:
        _require(isinstance(shard, Mapping), "a shard is not an object")
        shard_id = shard.get("shard_id")
        _require(isinstance(shard_id, str) and shard_id, "a shard has no id")
        _require(shard_id not in shard_ids, f"shard id {shard_id!r} repeats")
        shard_ids.add(shard_id)
        _require(
            shard.get("start") == expected_start,
            f"shard {shard_id} starts at {shard.get('start')}, "
            f"expected {expected_start}",
        )
        count = shard.get("count")
        _require(
            isinstance(count, int) and count > 0,
            f"shard {shard_id} has a non-positive count",
        )
        expected_start += count
    _require(
        expected_start == POSITION_COUNT,
        f"shards cover {expected_start} positions; expected {POSITION_COUNT}",
    )

    for field, relative in _MODEL_PINS:
        _require(plan.get(field) == relative, f"{field} drifted")
        _require(plan.get(f"{field}_sha256"), f"{field} is unpinned")


# Everything the akq500 module knows about, plus the akq500 plan itself: that
# plan reserved 20,000,000 for five hundred positions when it was written, and
# a later plan that ignored it would be the second half of the same mistake.
_SPENT_PROGRESSIONS: tuple[tuple[str, int, int], ...] = (
    *_RECORDED_EVAL_PROGRESSIONS,
    ("t0first-akq500-1024p plan", 20_000_000, 500),
)


def check_seed_allocation() -> dict[str, Any]:
    """Prove this plan's evaluation seeds are untouched, at build time."""

    conflicts: list[str] = []
    for label, base, positions in _SPENT_PROGRESSIONS:
        if not eval_progressions_disjoint(
            EVAL_SEED_BASE, POSITION_COUNT, base, positions
        ):
            conflicts.append(f"{label} (base {base:,})")
    low, high = eval_seed_span(EVAL_SEED_BASE, POSITION_COUNT)
    for base in _DOC_RECORDED_EVAL_BASES:
        if base <= high and low <= base + _UNRECORDED_EXTENT:
            conflicts.append(f"unsized base {base:,}")
    _require(
        not conflicts,
        "evaluation seed block collides with " + ", ".join(conflicts),
    )
    return {
        "eval_seed_base": EVAL_SEED_BASE,
        "eval_seed_span": [low, high],
        "eval_seed_stride": EVAL_SEED_STRIDE,
        "compared_against": [label for label, _, _ in _SPENT_PROGRESSIONS],
        "unsized_bases_reserved": list(_DOC_RECORDED_EVAL_BASES),
        "note": (
            "Twelve repeats are roots, not trials, so the trial term that "
            "makes these blocks three million wide is not spent: 240 positions "
            "at stride seven touch 1,674 seeds and nothing else."
        ),
    }


def build_plan(
    package_dir: pathlib.Path,
    *,
    hands_file: pathlib.Path = DEFAULT_HANDS_FILE,
    expected_identity: ExpectedPackageIdentity = DEFAULT_PACKAGE_IDENTITY,
) -> tuple[dict[str, Any], PackageAudit]:
    audit = audit_package(package_dir, expected_identity=expected_identity)
    hands, source = load_hands(pathlib.Path(hands_file))
    plan = _plan(audit, build_roots(hands), source)
    validate_plan(plan)
    return plan, audit


def write_plan_once(
    package_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    hands_file: pathlib.Path = DEFAULT_HANDS_FILE,
    expected_identity: ExpectedPackageIdentity = DEFAULT_PACKAGE_IDENTITY,
) -> dict[str, Any]:
    plan, audit = build_plan(
        package_dir, hands_file=hands_file, expected_identity=expected_identity
    )
    rendered = {PLAN_FILENAME: rendered_json_bytes(plan)}
    manifest = {
        "schema": PLAN_SET_SCHEMA,
        "generator": "ofc_regular.hu_t0_ladder20_explicit_plan_v1",
        "package_audit": audit.provenance(),
        "plans": {
            "first": {
                "filename": PLAN_FILENAME,
                "job_id": JOB_ID,
                "sha256": sha256_bytes(rendered[PLAN_FILENAME]),
            }
        },
        "measurement_contract": {
            "street": STREET,
            "seat": SEAT,
            "root_source": ROOT_SOURCE_EXPLICIT,
            "hands": HAND_COUNT,
            "repeats": REPEATS,
            "positions": POSITION_COUNT,
            "samples": SAMPLES,
            "seeds_per_position": SEEDS_PER_POSITION,
            "shards": SHARD_COUNT,
            "prefilter_samples": PREFILTER_SAMPLES,
            "prefilter_keep": PREFILTER_KEEP,
            "prefilter_margin": PREFILTER_MARGIN,
            "raced": False,
            "hand_source": plan["provenance"]["hand_source"],
            "eval_seed_base": EVAL_SEED_BASE,
            "eval_seed_span": list(eval_seed_span(EVAL_SEED_BASE, POSITION_COUNT)),
            "hand_seed_block_spent": False,
            "distilled_replies": False,
        },
    }
    rendered[MANIFEST_FILENAME] = rendered_json_bytes(manifest)

    output_dir = pathlib.Path(output_dir).resolve()
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
    parser.add_argument("--hands-file", default=str(DEFAULT_HANDS_FILE))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--expect-ledger-sha256", default="")
    parser.add_argument("--expect-runtime-sha256", default="")
    parser.add_argument("--expect-wheelhouse-sha256", default="")
    args = parser.parse_args(argv)

    package_dir = pathlib.Path(args.package_dir)
    hands_file = pathlib.Path(args.hands_file)
    overrides = (args.expect_ledger_sha256, args.expect_runtime_sha256,
                 args.expect_wheelhouse_sha256)
    if any(overrides) and not all(overrides):
        parser.error("pass all three --expect-*-sha256 digests or none")
    identity = (
        ExpectedPackageIdentity(*overrides) if all(overrides)
        else DEFAULT_PACKAGE_IDENTITY
    )

    if args.mode == "audit":
        plan, audit = build_plan(
            package_dir, hands_file=hands_file, expected_identity=identity
        )
        seeds = check_seed_allocation()
        print(json.dumps({
            "job_id": plan["job_id"],
            "hands": HAND_COUNT,
            "repeats": REPEATS,
            "positions": len(plan["roots"]),
            "samples": plan["samples"],
            "shards": len(plan["shards"]),
            "sieve": [PREFILTER_SAMPLES, PREFILTER_KEEP, PREFILTER_MARGIN],
            "eval_seeds": seeds,
            "plan_sha256": sha256_bytes(rendered_json_bytes(plan)),
            "package": audit.provenance(),
        }, indent=1))
        return 0

    if not args.out_dir:
        parser.error("--mode write needs --out-dir")
    manifest = write_plan_once(
        package_dir, pathlib.Path(args.out_dir),
        hands_file=hands_file, expected_identity=identity,
    )
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
