"""Build the immutable M7v6 T1 256-particle label-plan pair.

This supersedes `hu_m7_t1_1024_plan_v1`, which is kept: its plans are evidence
of what was written on 2026-08-15 and a plan is never rewritten in place.  The
difference is three cuts, each made because a measurement said the thing cut
does not matter, and one addition:

* **256 particles, not 1,024.**  The T1 probe's own paired verdict (30 roots a
  seat against a 4,096 reference) put 256-vs-1024 across zero at both seats,
  and 1,024 was recorded as a purchase rather than a resolution.  Since then T3
  measured what label precision is worth to a model at all: adding sigma = 0.4
  of noise to its labels moved held-out regret by +0.00017.  A model that
  cannot see 0.4 of noise cannot see the gap between these two rungs.  Cost is
  superlinear in particles here -- 383/309 core-s a root at 256 against
  3,300/1,660 at 1,024 -- so this is a 7.2x cut, not 4x.
* **18,000 roots a seat, not 25,000.**  The generation-1 scaling curve flattens
  well below 25,000, which is why the T3 street was sized at 18,000.
* **The distilled T2 pair answers the nested replies.**  The chain A/B moved a
  T1 label no further than reseeding the same chain did (first
  -0.045 [-0.134, +0.044], second -0.036 [-0.144, +0.072]), and it was run **at
  256 particles**, which is exactly the rung this plan uses -- so the caveat
  that held the clones back on 2026-08-15 does not apply here.
* **All 27 actions are still scored.**  Narrowing was measured (top-5 through
  top-10 indistinguishable from the full fan on the T1 corpus itself, top-3
  breaking at +26 %) and declined: it saves about $20 and buys back a risk of
  dropping the true best move, which is the very defect this relabel exists to
  remove.  The worker cannot narrow a T1 plan today in any case.

What has NOT changed: the roots, the continuation generation, the FL constant,
and the write-once discipline.  Before a plan is rendered every archive digest
and every hash the M7v6 ledger declares is recomputed from the package.

Output is write-once.  Existing plans are evidence and are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from ofc_regular.hu_m7_t1_1024_plan_v1 import (  # noqa: F401
    DEFAULT_PACKAGE_IDENTITY,
    EXPECTED_LEDGER_SHA256,
    EXPECTED_RUNTIME_SHA256,
    EXPECTED_WHEELHOUSE_SHA256,
    ExpectedPackageIdentity,
    PLAN_SCHEMA,
    PackageAudit,
    PlanValidationError,
    _package_pin,
    _require,
    audit_package,
    rendered_json_bytes,
    sha256_bytes,
)

PROVENANCE_SCHEMA = "hu_m7_t1_256_plan_provenance_v1"
PLAN_SET_SCHEMA = "hu_m7_t1_256_plan_set_v1"

POSITION_COUNT = 18_000
SHARD_COUNT = 116
SAMPLES = 256
SEEDS_PER_POSITION = 1
# A fresh block.  The 2026-08-15 plans took [948,000,000, 948,025,000); this
# generation does not reuse any part of it, because a partial overlap between
# two label plans produces duplicate positions that are invisible in the output.
HAND_SEED_BASE = 952_000_000
BEHAVIOR_SEED_OFFSET = 500_000
EVAL_SEED_BASE = 12_000_000
FL_EV_CARDS = 14
FL_EV_VALUE = 9.6

FIRST_JOB_ID = "m7v6-t1first-18k-256p"
SECOND_JOB_ID = "m7v6-t1second-18k-256p"
FIRST_FILENAME = "worker_plan_m7v6_t1first_18k_256p.json"
SECOND_FILENAME = "worker_plan_m7v6_t1second_18k_256p.json"
MANIFEST_FILENAME = "plan_set_m7v6_t1_18k_256p.json"


def make_shards(
    *, position_count: int = POSITION_COUNT, shard_count: int = SHARD_COUNT
) -> list[dict[str, Any]]:
    _require(position_count > 0, "position_count must be positive")
    _require(0 < shard_count <= position_count, "invalid shard_count")
    small, extra = divmod(position_count, shard_count)
    counts = [small] * (shard_count - extra) + [small + 1] * extra
    width = max(2, len(str(shard_count - 1)))
    shards: list[dict[str, Any]] = []
    start = 0
    for index, count in enumerate(counts):
        shards.append(
            {"shard_id": f"{index:0{width}d}", "start": start, "count": count}
        )
        start += count
    _require(start == position_count, "internal shard construction error")
    return shards


def _base_plan(audit: PackageAudit, *, seat: str) -> dict[str, Any]:
    _require(seat in ("first", "second"), f"unsupported seat {seat!r}")
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "job_id": FIRST_JOB_ID if seat == "first" else SECOND_JOB_ID,
        "street": "T1",
        "seat": seat,
        "samples": SAMPLES,
        "seeds_per_position": SEEDS_PER_POSITION,
        "hand_seed_base": HAND_SEED_BASE,
        "behavior_seed_offset": BEHAVIOR_SEED_OFFSET,
        "eval_seed_base": EVAL_SEED_BASE,
        "engine_library": "native/libofc_hu_m3_engine.so",
        "engine_library_sha256": _package_pin(
            audit, "native/libofc_hu_m3_engine.so"
        ),
        "feature_encoder_library": "native/libofc_stage3_feature_encoder.so",
        "feature_encoder_library_sha256": _package_pin(
            audit, "native/libofc_stage3_feature_encoder.so"
        ),
        "t4_model": "weights/t4_model_v6.bin",
        "t4_model_sha256": _package_pin(audit, "weights/t4_model_v6.bin"),
        "t3_second_model": "weights/t3_model_v3.bin",
        "t3_second_model_sha256": _package_pin(audit, "weights/t3_model_v3.bin"),
        "t3_first_model": "weights/t3first_model_v2.bin",
        "t3_first_model_sha256": _package_pin(
            audit, "weights/t3first_model_v2.bin"
        ),
        # The full-precision T2 pair stays pinned even though the coarse pair
        # answers the replies: the engine loads both and reports the evaluator
        # as `learned_fast` only because it reached it through the coarse one.
        # Dropping the full pair would make the plan unreadable to the worker.
        "t2_second_model": "weights/t2_model_v2.bin",
        "t2_second_model_sha256": _package_pin(audit, "weights/t2_model_v2.bin"),
        "t2_first_model": "weights/t2first_model_v2.bin",
        "t2_first_model_sha256": _package_pin(
            audit, "weights/t2first_model_v2.bin"
        ),
        "fast_t2_second_model": "weights/fast_t2_second_v1.bin",
        "fast_t2_second_model_sha256": _package_pin(
            audit, "weights/fast_t2_second_v1.bin"
        ),
        "fast_t2_first_model": "weights/fast_t2_first_v1.bin",
        "fast_t2_first_model_sha256": _package_pin(
            audit, "weights/fast_t2_first_v1.bin"
        ),
        "fl_ev_cards": FL_EV_CARDS,
        "fl_ev_value": FL_EV_VALUE,
        "shards": make_shards(),
    }
    if seat == "first":
        plan.update(
            {
                "t1_second_model": "weights/t1_model_v1.bin",
                "t1_second_model_sha256": _package_pin(
                    audit, "weights/t1_model_v1.bin"
                ),
            }
        )

    plan["provenance"] = {
        "schema": PROVENANCE_SCHEMA,
        "generation": "m7v6",
        "supersedes": "hu_m7_t1_1024_plan_v1 (written 2026-08-15, never run)",
        "package_audit": audit.provenance(),
        "purchase": {
            "position_count_per_seat": POSITION_COUNT,
            "teacher_particles": SAMPLES,
            "reason": (
                "256 rather than 1024: the T1 particle probe's paired "
                "256-vs-1024 verdict crosses zero at both seats, and T3 "
                "measured a model's held-out regret unmoved (+0.00017) by "
                "adding sigma=0.4 of label noise, which is far more than the "
                "gap between these rungs. 18,000 rather than 25,000: the "
                "generation-1 scaling curve flattens well below 25,000. "
                "Owner decision 2026-08-16; full sizing in "
                "docs/t1_relabel_sizing_20260816.md."
            ),
            "declined": (
                "Narrowing to top-K. Measured on the T1 corpus itself, top-5 "
                "to top-10 cannot be told apart from the full fan and top-3 "
                "costs +26%; it was declined anyway because it would drop the "
                "true best action on some roots, which is the defect this "
                "relabel exists to remove. The worker also gates its prefilter "
                "fields to T0."
            ),
        },
        "seed_audit": {
            "hand_seed_interval": [HAND_SEED_BASE, HAND_SEED_BASE + POSITION_COUNT],
            "behavior_seed_interval": [
                HAND_SEED_BASE + BEHAVIOR_SEED_OFFSET,
                HAND_SEED_BASE + BEHAVIOR_SEED_OFFSET + POSITION_COUNT,
            ],
            "eval_seed_base": EVAL_SEED_BASE,
            "whole_block_shared_by_both_seats": True,
            "partial_seed_block_reuse": False,
            "note": (
                "[952000000,952018000) allocated fresh. The superseded 1024p "
                "plans hold [948000000,948025000) and no part of it is reused: "
                "a partial overlap between two label plans makes duplicate "
                "positions that are invisible in the output."
            ),
        },
        "continuation": {
            "t3_first": "v7 / weights/t3first_model_v2.bin",
            "t3_second": "v7 / weights/t3_model_v3.bin",
            "t4": "v6 / weights/t4_model_v6.bin",
            "t2_first": "v2 / weights/t2first_model_v2.bin",
            "t2_second": "v2 / weights/t2_model_v2.bin",
            "t2_replies_answered_by": (
                "the distilled pair (fast_t2_*_v1). The chain A/B moved a T1 "
                "label no further than reseeding the same chain: first "
                "-0.045 [-0.134,+0.044], second -0.036 [-0.144,+0.072]. That "
                "A/B ran at 256 particles, which is this plan's rung, so its "
                "null applies directly rather than by extrapolation."
            ),
            "t1_second_reply": (
                "generation-1 / weights/t1_model_v1.bin (the incumbent this "
                "relabel replaces)"
                if seat == "first"
                else "not ahead of a T1 second-seat root"
            ),
        },
        "fantasyland": {
            "cards": FL_EV_CARDS,
            "value": FL_EV_VALUE,
            "config": "configs/fl_ev_regular_v4_selfplay.json",
            "config_sha256": _package_pin(
                audit, "configs/fl_ev_regular_v4_selfplay.json"
            ),
            "why_this_relabel_exists": (
                "The shipped T1 corpus bakes in 10.227. Measured on its own "
                "roots with common random numbers at 1024 particles, moving "
                "the constant to 9.6 changes the best move on 5 of 24 roots "
                "(Kendall tau 0.8919 +/- 0.0178, 5.53% of pairs reordered). "
                "The shift is 0.627 x P(FL entry), so only its SPREAD across "
                "candidates can reorder anything -- and at T1 that spread "
                "averages 1.1253. T4 saw 0 of 36 decisions move because entry "
                "is nearly settled there and every candidate shifts together."
            ),
        },
        "sharding": {
            "shards": SHARD_COUNT,
            "concurrency_is_not_authorized_by_this_plan": True,
            "recommended_max_live_c4_standard_8": 58,
            "intended_workers_per_vm": 6,
            "waves_at_recommended_max_live": 2,
            "sizing_reason": (
                "At 256 particles with coarse T2 replies a first-seat root is "
                "about 250 core-s and a second-seat one about 151 (measured "
                "under 10-way contention). A 156-root shard at six workers is "
                "about 1.8 hours at the first seat, well inside the six-hour "
                "watchdog, so 116 shards suffice where the 1024p plan needed "
                "300."
            ),
        },
    }
    return plan


def validate_plan_pair(plans: dict[str, dict[str, Any]]) -> None:
    _require(set(plans) == {"first", "second"}, "plan pair must contain both seats")
    first, second = plans["first"], plans["second"]
    for seat, plan in plans.items():
        _require(plan.get("schema") == PLAN_SCHEMA, f"{seat} plan schema drifted")
        _require(plan.get("street") == "T1", f"{seat} plan is not T1")
        _require(plan.get("seat") == seat, f"{seat} plan seat drifted")
        _require(plan.get("samples") == SAMPLES, f"{seat} plan is not 256p")
        _require(
            plan.get("seeds_per_position") == SEEDS_PER_POSITION,
            f"{seat} plan does not have exactly one label run",
        )
        shards = plan.get("shards")
        _require(isinstance(shards, list) and len(shards) == SHARD_COUNT,
                 f"{seat} does not have {SHARD_COUNT} shards")
        expected_start = 0
        seen: set[str] = set()
        for shard in shards:
            shard_id = shard.get("shard_id")
            _require(isinstance(shard_id, str) and shard_id not in seen,
                     f"{seat} shard id is missing or duplicated")
            seen.add(shard_id)
            _require(shard.get("start") == expected_start,
                     f"{seat} shard {shard_id} is not contiguous")
            count = shard.get("count")
            _require(isinstance(count, int) and count > 0,
                     f"{seat} bad shard count")
            expected_start += count
        _require(expected_start == POSITION_COUNT,
                 f"{seat} does not cover {POSITION_COUNT}")
        _require("probe" not in plan, f"{seat} plan carries a probe block")
        # The coarse pair is all-or-nothing in the worker; a plan that names
        # half of it would run one T2 seat coarse and the other exact.
        for field in ("fast_t2_first_model", "fast_t2_first_model_sha256",
                      "fast_t2_second_model", "fast_t2_second_model_sha256"):
            _require(field in plan, f"{seat} plan is missing {field}")
        # The full-precision pair must survive alongside it: the engine loads
        # both, and the worker refuses a T1 plan without the full pins.
        for field in ("t2_first_model", "t2_second_model",
                      "t3_first_model", "t3_second_model", "t4_model"):
            _require(field in plan, f"{seat} plan is missing {field}")
        _require(
            plan.get("fl_ev_cards") == FL_EV_CARDS
            and plan.get("fl_ev_value") == FL_EV_VALUE,
            f"{seat} plan FL contract drifted",
        )

    for field in ("hand_seed_base", "behavior_seed_offset", "eval_seed_base",
                  "samples", "seeds_per_position", "engine_library_sha256",
                  "feature_encoder_library_sha256", "t3_first_model_sha256",
                  "t3_second_model_sha256", "t2_first_model_sha256",
                  "t2_second_model_sha256", "fast_t2_first_model_sha256",
                  "fast_t2_second_model_sha256", "t4_model_sha256",
                  "fl_ev_cards", "fl_ev_value", "shards"):
        _require(first.get(field) == second.get(field),
                 f"paired field {field} drifted")
    _require("t1_second_model" in first,
             "T1 first plan is missing the second-seat reply model")
    _require("t1_second_model" not in second,
             "T1 second plan pins an unreachable T1 reply model")
    _require(first.get("job_id") != second.get("job_id"), "plan job ids collide")
    # The superseded generation's block must not be touched.
    _require(
        HAND_SEED_BASE + POSITION_COUNT <= 948_000_000
        or HAND_SEED_BASE >= 948_025_000,
        "hand block overlaps the superseded 1024p plans",
    )


def build_plan_pair(
    package_dir: pathlib.Path,
    *,
    expected_identity: ExpectedPackageIdentity = DEFAULT_PACKAGE_IDENTITY,
) -> tuple[dict[str, dict[str, Any]], PackageAudit]:
    audit = audit_package(package_dir, expected_identity=expected_identity)
    plans = {"first": _base_plan(audit, seat="first"),
             "second": _base_plan(audit, seat="second")}
    validate_plan_pair(plans)
    return plans, audit


def write_plan_pair_once(
    package_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    expected_identity: ExpectedPackageIdentity = DEFAULT_PACKAGE_IDENTITY,
) -> dict[str, Any]:
    plans, audit = build_plan_pair(package_dir, expected_identity=expected_identity)
    rendered = {
        FIRST_FILENAME: rendered_json_bytes(plans["first"]),
        SECOND_FILENAME: rendered_json_bytes(plans["second"]),
    }
    manifest = {
        "schema": PLAN_SET_SCHEMA,
        "generator": "ofc_regular.hu_m7_t1_256_plan_v1",
        "package_audit": audit.provenance(),
        "plans": {
            "first": {"filename": FIRST_FILENAME, "job_id": FIRST_JOB_ID,
                      "sha256": sha256_bytes(rendered[FIRST_FILENAME])},
            "second": {"filename": SECOND_FILENAME, "job_id": SECOND_JOB_ID,
                       "sha256": sha256_bytes(rendered[SECOND_FILENAME])},
        },
        "paired_contract": {
            "positions_per_seat": POSITION_COUNT,
            "samples": SAMPLES,
            "shards_per_seat": SHARD_COUNT,
            "hand_seed_interval": [HAND_SEED_BASE, HAND_SEED_BASE + POSITION_COUNT],
            "eval_seed_base": EVAL_SEED_BASE,
            "coarse_t2_replies": True,
            "first_only_t1_second_pin": True,
            "current_profile_changed": False,
        },
    }
    rendered[MANIFEST_FILENAME] = rendered_json_bytes(manifest)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: output_dir / name for name in rendered}
    collisions = sorted(str(p) for p in targets.values() if p.exists())
    _require(not collisions, f"write-once output already exists: {collisions}")
    for name in (FIRST_FILENAME, SECOND_FILENAME, MANIFEST_FILENAME):
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
        plans, audit = build_plan_pair(package_dir)
        print(json.dumps({
            "schema": "hu_m7_t1_256_plan_audit_summary_v1",
            "package_audit": audit.provenance(),
            "planned": {
                "positions_per_seat": POSITION_COUNT, "samples": SAMPLES,
                "shards_per_seat": SHARD_COUNT,
                "hand_seed_base": HAND_SEED_BASE,
                "eval_seed_base": EVAL_SEED_BASE,
                "coarse_t2_replies": True,
            },
            "writes_performed": False,
        }, indent=2, sort_keys=True))
        return 0

    if not args.out_dir:
        raise SystemExit("write mode requires --out-dir")
    print(json.dumps(write_plan_pair_once(package_dir, pathlib.Path(args.out_dir)),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
