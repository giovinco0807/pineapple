"""Self-healing supervisor for immutable label-generation Spot runs.

The v1 executor's ``--only-missing`` option only checks whether an instance
exists.  A completed Spot worker normally disappears, so repeatedly invoking
that option can recreate completed shards forever.  This supervisor makes the
durable GCS ``SHARD_DONE.json`` marker authoritative, validates it against the
worker plan digest, and fills only incomplete, instance-free shards up to a
fixed live-instance cap.

Every cloud write still requires ``--allow-writes``.  A launch intent is
written locally *before* the create request and the result is written to a
separate immutable receipt.  This leaves enough evidence to recover safely if
the controller itself is interrupted between the API request and its result.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.parse
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .hu_m31_label_gen_gcp_execute_v1 import (
    DEFAULT_IMAGE,
    RECEIPT_ROOT,
    _instance_name,
    _instance_spec,
    _placement,
    _write_once,
)
from .hu_m31_label_gen_gcp_plan_v1 import PLAN_RECEIPT_SCHEMA
from .hu_m31_label_gen_resume_v1 import (
    COMPLETE_CHECKPOINT_SCHEMA,
    complete_checkpoint_object_name,
)
from .hu_m31_label_gen_worker_v1 import DONE_SCHEMA, canonical_bytes

DURABLE_SCHEMA = "hu_m31_label_gen_supervisor_generation_durable_v1"
SNAPSHOT_SCHEMA = "hu_m31_label_gen_supervisor_snapshot_v1"
INTENT_SCHEMA = "hu_m31_label_gen_supervisor_launch_intent_v1"
RESULT_SCHEMA = "hu_m31_label_gen_supervisor_launch_result_v1"
DELETE_INTENT_SCHEMA = "hu_m31_label_gen_supervisor_delete_intent_v1"
DELETE_RESULT_SCHEMA = "hu_m31_label_gen_supervisor_delete_result_v1"
DEFAULT_BOOT_GRACE_SECONDS = 60 * 60
DEFAULT_HEARTBEAT_STALE_SECONDS = 45 * 60
PINNED_IMAGE = (
    "https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/"
    "debian-12-bookworm-v20260804"
)
# The startup script carries its own copy of the position-schema table, so
# teaching the fleet a new street changes its bytes. Runs already completed were
# produced by the script as it was then and must keep validating against it, so
# the digest is a contract field like the partition and the particle count --
# not one constant every run has to agree on.
STARTUP_SHA256_M7 = "5d4658ea56fc09853acc56bf4a25cec6ddbf61c413a10dc2c6a71ba2d797e472"
STARTUP_SHA256_T0 = "c1e0cd1e73a75e11ebac319a8caba8361af1e44fa0ab6ed8a6fef65fb5903ae6"
PINNED_STARTUP_SHA256 = STARTUP_SHA256_M7
PROJECT = "ofc-solver-485418"
# A run's identity is its worker plan, its particle count, and the partition of
# positions it may cover. Those three used to be constants in the contract
# check, which made the check exact for the M7 T2 2048 relabel and unusable for
# anything else. They are contract fields now: the same assertions run, on the
# values the named run is allowed to carry, so a plan still cannot invent a
# partition, a sample count, or a runtime of its own.
EXPECTED_RUN_CONTRACTS = {
    # The immutable worker plans retain their original job_id and digest.  The
    # cloud run names are deliberately fresh because the old staging object
    # names are create-only and already exist.
    "m7v5-t2first-25k-2048p-r2": {
        "worker_plan_job_id": "m7v5-t2first-25k-2048p",
        "worker_plan_sha256": (
            "358445e90543819977479365650b49b144458731f4a224f56330696171adfa91"
        ),
        "partition": "m7_t2_2048",
        "samples": 2048,
        "startup_sha256": STARTUP_SHA256_M7,
    },
    "m7v5-t2second-25k-2048p-r2": {
        "worker_plan_job_id": "m7v5-t2second-25k-2048p",
        "worker_plan_sha256": (
            "8800ab83b976e117c6a5652a7a8d58fa4dd93b5ed104da05c5b1e59bb7f5573b"
        ),
        "partition": "m7_t2_2048",
        "samples": 2048,
        "startup_sha256": STARTUP_SHA256_M7,
    },
    # The first seat is cut into contiguous slices of the SAME canonical
    # partition so it can run beside the second seat instead of behind it: one
    # region cannot hold both (PREEMPTIBLE_CPUS is 468 per region and the second
    # seat's 58 c4-standard-8 already spend 464 of asia-northeast1's). Both
    # slices carry the identical immutable worker plan, so a position's label is
    # the same object whichever slice produces it; only the bucket prefix and
    # the zone differ.
    "m7v5-t2first-25k-2048p-r2b": {
        "worker_plan_job_id": "m7v5-t2first-25k-2048p",
        "worker_plan_sha256": (
            "358445e90543819977479365650b49b144458731f4a224f56330696171adfa91"
        ),
        "partition": "m7_t2_2048",
        "samples": 2048,
        "startup_sha256": STARTUP_SHA256_M7,
    },
    # Tail windows cut on 2026-08-07 onto the slots the second seat freed when
    # it finished. Same immutable worker plan again; only the placement and the
    # bucket prefix differ.
    "m7v5-t2first-25k-2048p-r2e": {
        "worker_plan_job_id": "m7v5-t2first-25k-2048p",
        "worker_plan_sha256": (
            "358445e90543819977479365650b49b144458731f4a224f56330696171adfa91"
        ),
        "partition": "m7_t2_2048",
        "samples": 2048,
        "startup_sha256": STARTUP_SHA256_M7,
    },
    "m7v5-t2first-25k-2048p-r2f": {
        "worker_plan_job_id": "m7v5-t2first-25k-2048p",
        "worker_plan_sha256": (
            "358445e90543819977479365650b49b144458731f4a224f56330696171adfa91"
        ),
        "partition": "m7_t2_2048",
        "samples": 2048,
        "startup_sha256": STARTUP_SHA256_M7,
    },
    # The T3-vs-Fantasyland relabel of 2026-08-09. 100,000 positions at 100
    # opponent samples under fl_ev 9.6, replacing a 20,000-position corpus
    # labelled at 400 samples under 9.109. Both changes were measured: T3's
    # scaling curve is n^-0.53 and still descending at 18,000, while the cheap
    # label costs 0.0009 of regret against the rich one, 1.2% of the model error
    # it feeds -- so the samples pay for positions.
    # The T2-vs-Fantasyland relabel of 2026-08-09, cut into the two zones that
    # are free while T3 holds asia-northeast1. Only t3_draws moves, 32 -> 512:
    # it is the one axis that pays here, cutting the label standard error by
    # exactly 1/sqrt2 per doubling all the way to 1024, because ~8.4 of the 9.95
    # core-seconds per root is fixed Fantasyland work every extra draw reuses.
    # Opponent samples and t4_draws stay put -- re-asked on top of t3_draws=128
    # they bought 7.5% for 4.4x and 5.1% for 1.4x respectively.
    "t2vsfl-50k-t3d512-flev96-r1a": {
        "worker_plan_job_id": "t2vsfl-50k-t3d512-flev96",
        "worker_plan_sha256": (
            "99af0b735b05cca34193a1624cb32f3deb61927081bf7b5e56cd83c86999680d"
        ),
        "partition": "t2vsfl_50k",
        "samples": 200,
        "startup_sha256": STARTUP_SHA256_M7,
    },
    "t2vsfl-50k-t3d512-flev96-r1b": {
        "worker_plan_job_id": "t2vsfl-50k-t3d512-flev96",
        "worker_plan_sha256": (
            "99af0b735b05cca34193a1624cb32f3deb61927081bf7b5e56cd83c86999680d"
        ),
        "partition": "t2vsfl_50k",
        "samples": 200,
        "startup_sha256": STARTUP_SHA256_M7,
    },
    # The T0-first-vs-Fantasyland bootstrap of 2026-08-09. 3,480 positions with
    # the fan unnarrowed, which is the point: the only ranker that could cut 232
    # openings today is the normal-table T0 model, and cutting with it would
    # prejudge whether a Fantasyland opponent shifts hero's opening at all. This
    # run buys the labels a vs-Fantasyland ranker is distilled from, and that
    # ranker cuts the full 134,459-position pass.
    #
    # 60 positions a shard, not 87: a T0 position costs ~1,531 core-seconds, so
    # 87 of them is 6.2 hours against a 6-hour watchdog.
    # -r1 booted 58 VMs that died inside the startup script's own copy of the
    # position-schema table and idled for forty minutes. -r2 is the same
    # immutable worker plan against the script that knows the street; a fresh
    # run name because the staging objects are create-only.
    # -r2 ran on the script whose default chunk was 32: with ten positions to a
    # stride and 1,531 core-seconds to a position, every worker computed for
    # four hours and wrote nothing until the end -- no progress to watch and
    # nothing salvaged from a preemption. -r3 chunks by one.
    # 20,000 more T0 positions, disjoint from the bootstrap's 3,480, because the
    # positions curve measured on that bootstrap is still falling steeply --
    # 1,390 to 2,780 positions cut regret from 1.71 to 0.64 -- and nothing about
    # where it flattens can be read from two points that far up the slope.
    #
    # The fan is not narrowed. Measured on three roots, cutting 232 openings to
    # 20 saves 1.42x, not the 11.6x proportionality assumed: two thirds of a T0
    # position is Fantasyland solving that every opening shares, because the
    # opponent's hand depends on which cards hero holds and not on which rows
    # they went in. At that saving the risk of dropping the answer is not worth
    # taking.
    #
    # 100 positions a shard, 200 shards: a shard is ~4.9 hours against a 6-hour
    # watchdog, and the supervisor refills the 58 live slots as they finish.
    # 30,000 more, taking the corpus to 50,000. The curve was still falling at
    # n^-0.56 when the last run finished -- 0.446 at 10,000 positions, 0.308 at
    # 19,280 -- and the mid-run reading of "saturated" turned out to be the
    # index bias, not the model.
    #
    # Two defects from the 20k run are fixed in its roots file rather than here:
    # indices start at zero, so a shard's start offset resolves; and the file is
    # shuffled, so a partial collection is a sample of the space instead of its
    # low-rank prefix.
    # A yardstick, not training data. Every T0 number measured so far is scored
    # against labels whose own top two openings disagree by 0.33 between seeds,
    # while the model's error is 0.25 -- the ruler is coarser than what it is
    # measuring, and three extrapolations and one "saturated" reading have
    # already been wrong inside that noise. 200 positions at draws=64 cost
    # sixteen times a corpus position each and should put the teacher's own
    # disagreement near 0.09, which is finally a margin.
    #
    # The positions are drawn from the bootstrap holdout, so they are the ones
    # every model here is judged on; a reference over different positions would
    # measure nothing about the reported numbers.
    # The reference, rebuilt to draw its opponents from the pre-solved pool.
    # Verified against solving at every leaf on six roots: same opening 6/6,
    # zero regret, values shifted -0.0096, and self-agreement 0.4263 against
    # 0.4277 -- the reuse the pool trades for speed cost nothing measurable.
    # 2.3x faster, so the yardstick is 2.5 hours instead of ten.
    "t0first-vsfl-ref200-d64-pool-r1": {
        "worker_plan_job_id": "t0first-vsfl-ref200-d64-pool",
        "worker_plan_sha256": (
            "dacdf2e8d73081ee4ce39afc0d7332894bbd1b4e7efee8179804f538f1877c8e"
        ),
        "partition": "t0first_ref200",
        "samples": 16,
        "startup_sha256": STARTUP_SHA256_T0,
    },
    "t0first-vsfl-ref200-d64-flev96-r1": {
        "worker_plan_job_id": "t0first-vsfl-ref200-d64-flev96",
        "worker_plan_sha256": (
            "bb3fd3b6f6fe32f91b8f9a460d78023c269e7d3e8dff232c0bc5cb0add1fd645"
        ),
        "partition": "t0first_ref200",
        "samples": 16,
        "startup_sha256": STARTUP_SHA256_T0,
    },
    "t0first-vsfl-30k-d16-flev96-r1": {
        "worker_plan_job_id": "t0first-vsfl-30k-d16-flev96",
        "worker_plan_sha256": (
            "9d71a68eda9b2ec9cd3d28a03b6f35bb3c4ab62d69ec1290c59752a9699bcccd"
        ),
        "partition": "t0first_30k",
        "samples": 16,
        "startup_sha256": STARTUP_SHA256_T0,
    },
    "t0first-vsfl-20k-d16-flev96-r1": {
        "worker_plan_job_id": "t0first-vsfl-20k-d16-flev96",
        "worker_plan_sha256": (
            "ff6802dc8cd9690f26c098068a5832c886eb4216c4dbc9d1db9ff78664f79dad"
        ),
        "partition": "t0first_20k",
        "samples": 16,
        "startup_sha256": STARTUP_SHA256_T0,
    },
    "t0first-vsfl-3480-d16-flev96-r3": {
        "worker_plan_job_id": "t0first-vsfl-3480-d16-flev96",
        "worker_plan_sha256": (
            "dff4eaf4ddecdad9d7ea80e9e2b2f2057f057982d941caf66dd148a508972652"
        ),
        "partition": "t0first_3480",
        "samples": 16,
        "startup_sha256": STARTUP_SHA256_T0,
    },
    "t0first-vsfl-3480-d16-flev96-r2": {
        "worker_plan_job_id": "t0first-vsfl-3480-d16-flev96",
        "worker_plan_sha256": (
            "dff4eaf4ddecdad9d7ea80e9e2b2f2057f057982d941caf66dd148a508972652"
        ),
        "partition": "t0first_3480",
        "samples": 16,
        "startup_sha256": STARTUP_SHA256_T0,
    },
    "t0first-vsfl-3480-d16-flev96-r1": {
        "worker_plan_job_id": "t0first-vsfl-3480-d16-flev96",
        "worker_plan_sha256": (
            "dff4eaf4ddecdad9d7ea80e9e2b2f2057f057982d941caf66dd148a508972652"
        ),
        "partition": "t0first_3480",
        "samples": 16,
        "startup_sha256": STARTUP_SHA256_T0,
    },
    "t3vsfl-100k-s100-flev96-r1": {
        "worker_plan_job_id": "t3vsfl-100k-s100-flev96",
        "worker_plan_sha256": (
            "07396f3006cb90b5e9584648dce0f035f72c4d3f39817bba7134cd3ef8d3fd10"
        ),
        "partition": "t3vsfl_100k",
        "samples": 100,
        "startup_sha256": STARTUP_SHA256_M7,
    },
}


def _partition(counts: list[int], *, id_width: int, total: int):
    """A canonical cut of a corpus, with the starts implied by the counts."""
    starts: list[int] = []
    running = 0
    for count in counts:
        starts.append(running)
        running += count
    if running != total:
        raise SystemExit(f"partition covers {running} positions, expected {total}")
    return {"counts": counts, "starts": starts, "id_width": id_width}


# Every run is cut from one of these. A plan may carry a contiguous slice of its
# partition, never a partition of its own.
CANONICAL_PARTITIONS = {
    "m7_t2_2048": _partition([143] * 56 + [144] * 118, id_width=3, total=25_000),
    "t3vsfl_100k": _partition([1725] * 8 + [1724] * 50, id_width=2, total=100_000),
    "t2vsfl_50k": _partition([848] * 27 + [847] * 32, id_width=2, total=50_000),
    "t0first_3480": _partition([60] * 58, id_width=2, total=3_480),
    "t0first_20k": _partition([100] * 200, id_width=3, total=20_000),
    "t0first_30k": _partition([100] * 300, id_width=3, total=30_000),
    "t0first_ref200": _partition([5] * 40, id_width=2, total=200),
}

# Retained under their original names: the M7 T2 relabel's records, its
# rebalance arithmetic and its postflight all refer to these.
CANONICAL_T2_SHARD_COUNTS = CANONICAL_PARTITIONS["m7_t2_2048"]["counts"]
CANONICAL_T2_SHARD_STARTS = CANONICAL_PARTITIONS["m7_t2_2048"]["starts"]

# Placements the supervisor will launch into. Every entry is already in the
# lifecycle layer's APPROVED_ZONES -- this set narrows that list to the zones
# where c4-standard-8 exists AND the project holds PREEMPTIBLE_CPUS quota, and
# it may never widen it.
ALLOWED_PLACEMENTS = {
    ("asia-northeast1", "asia-northeast1-b"),  # 468 vCPU -> 58 instances
    ("us-west1", "us-west1-a"),  # 252 vCPU -> 31 instances
    ("us-east1", "us-east1-b"),  # 229 vCPU -> 28 instances
}

# How many c4-standard-8 Spot instances a zone may hold, from that REGION's
# PREEMPTIBLE_CPUS quota divided by 8 vCPU. The budget is per zone because the
# quota is per region: a fleet in us-west1 does not consume asia-northeast1's
# capacity, so counting instances project-wide would make two concurrent runs
# starve each other for no reason.
ZONE_FLEET_BUDGET = {
    "asia-northeast1-b": 58,
    "us-west1-a": 31,
    "us-east1-b": 28,
}


class Adapter(Protocol):
    def get_object_bytes(
        self, *, bucket: str, object_name: str, generation: str | None = None
    ) -> bytes | None: ...

    def get_instance(self, *, instance_name: str) -> Mapping[str, Any] | None: ...

    def create_instance(
        self, *, instance_spec: Mapping[str, Any], request_id: str
    ) -> Mapping[str, Any]: ...

    def delete_instance(
        self, *, instance_name: str, request_id: str
    ) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True)
class ShardState:
    shard_id: str
    done: bool
    instance_status: str | None
    instance_created_at: str | None = None
    latest_heartbeat_at: str | None = None
    attempt_id: str | None = None


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_run_plan(path: pathlib.Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_RECEIPT_SCHEMA:
        raise SystemExit(f"unsupported run-plan schema: {plan.get('schema')!r}")
    stated = plan.get("plan_receipt_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_receipt_sha256", None)
    actual = _canonical_sha(unsigned)
    if stated != actual:
        raise SystemExit(
            f"run-plan digest mismatch: stated {stated!r}, actual {actual}"
        )
    shard_ids = [entry.get("shard_id") for entry in plan.get("shards", [])]
    if not shard_ids or len(shard_ids) != len(set(shard_ids)):
        raise SystemExit("run plan must contain unique non-empty shards")
    return plan


def load_stage_receipt(run_dir: pathlib.Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    path = run_dir / "stage_receipt.json"
    if not path.is_file():
        raise SystemExit(f"stage receipt missing: {path}")
    stage = json.loads(path.read_text(encoding="utf-8"))
    if stage.get("schema") != "hu_m31_label_gen_stage_receipt_v1":
        raise SystemExit(f"unsupported stage receipt: {stage.get('schema')!r}")
    if stage.get("run_name") != plan["run_name"] or stage.get("bucket") != plan["bucket"]:
        raise SystemExit("stage receipt belongs to a different run or bucket")
    expected = {
        row["relative"]: (row["object"], row["sha256"], row["bytes"])
        for row in plan["content_bindings_without_generations"]
    }
    observed: dict[str, tuple[Any, Any, Any]] = {}
    for row in stage.get("content_bindings", []):
        if not isinstance(row.get("generation"), str) or not row["generation"]:
            raise SystemExit("stage binding has no observed generation")
        observed[row["relative"]] = (row["object"], row["sha256"], row["bytes"])
    if observed != expected:
        raise SystemExit("stage receipt content bindings do not match the run plan")
    return stage


def validate_m7_t2_2048_run_contract(
    plan: Mapping[str, Any],
    *,
    image: str,
    max_live: int,
    max_run_seconds: int,
) -> None:
    run_name = plan.get("run_name")
    contract = EXPECTED_RUN_CONTRACTS.get(run_name)
    if contract is None:
        raise SystemExit(f"supervisor contract refuses run {run_name!r}")
    partition = CANONICAL_PARTITIONS[contract["partition"]]
    shard_counts = partition["counts"]
    shard_starts = partition["starts"]
    id_width = partition["id_width"]
    shard_total = len(shard_counts)
    if (
        plan.get("worker_plan_job_id") != contract["worker_plan_job_id"]
        or plan.get("worker_plan_sha256") != contract["worker_plan_sha256"]
        or plan.get("samples") != contract["samples"]
        or plan.get("machine_type") != "c4-standard-8"
        or (plan.get("region"), plan.get("zone")) not in ALLOWED_PLACEMENTS
        or plan.get("bucket") != "pokerhu-ofc-solver-485418-training"
        or plan.get("startup_script_relative")
        != "scripts/startup_hu_m31_label_gen_v1.sh"
        or plan.get("startup_script_sha256") != contract["startup_sha256"]
    ):
        raise SystemExit("M7 T2 2048 run identity drifted")
    if image != PINNED_IMAGE:
        raise SystemExit("M7 T2 2048 run must use the reviewed concrete image")
    if not 0 < max_live <= 58:
        raise SystemExit("M7 T2 2048 max_live must be in 1..58")
    if max_run_seconds != 7 * 3600:
        raise SystemExit("M7 T2 2048 max_run_seconds must be exactly 25200")
    shards = plan.get("shards")
    if not isinstance(shards, list) or not 0 < len(shards) <= shard_total:
        raise SystemExit(f"run must carry 1..{shard_total} shards")
    # A plan carries a contiguous WINDOW of the canonical partition. The window
    # offset is read from the first shard's id and every shard is then checked
    # against the canonical start/count for its own canonical index, so a plan
    # can never invent a partition of its own -- only decline to cover part of
    # the one partition. shard_id stays canonical because the worker resolves
    # its work by looking that id up in the immutable worker plan.
    try:
        offset = int(str(shards[0].get("shard_id")), 10)
    except (TypeError, ValueError):
        raise SystemExit("M7 T2 2048 first shard id is not a canonical index")
    if not 0 <= offset or offset + len(shards) > shard_total:
        raise SystemExit("M7 T2 2048 shard window leaves the canonical partition")
    counts: list[int] = []
    for index, shard in enumerate(shards):
        canonical_index = offset + index
        shard_id = f"{canonical_index:0{id_width}d}"
        object_prefix = f"labelgen/{run_name}/shards/{shard_id}"
        if (
            shard.get("shard_id") != shard_id
            or shard.get("start") != shard_starts[canonical_index]
            or shard.get("count") != shard_counts[canonical_index]
            or shard.get("object_prefix") != object_prefix
        ):
            raise SystemExit(f"M7 T2 shard {canonical_index} partition drifted")
        metadata = shard.get("metadata_values", {})
        if (
            metadata.get("lg-worker-count") != "6"
            or metadata.get("lg-watchdog-seconds") != "21600"
            or metadata.get("lg-plan-sha256") != plan["worker_plan_sha256"]
            or metadata.get("lg-shard-id") != shard_id
            or metadata.get("lg-bucket") != plan["bucket"]
            or metadata.get("lg-object-prefix") != object_prefix
        ):
            raise SystemExit(f"M7 T2 shard {index} metadata drifted")
        counts.append(shard["count"])
    window = shard_counts[offset:offset + len(shards)]
    if counts != window:
        raise SystemExit("M7 T2 2048 shard coverage drifted")
    bindings = plan.get("content_bindings_without_generations", [])
    expected_objects = {
        "static/runtime_archive/runtime.tar.gz": (
            f"labelgen/{run_name}/staging/runtime.tar.gz"
        ),
        "static/wheelhouse_archive/wheelhouse.zip": (
            f"labelgen/{run_name}/staging/wheelhouse.zip"
        ),
        "static/plan/plan.json": f"labelgen/{run_name}/staging/plan.json",
    }
    observed_objects = {
        row.get("relative"): row.get("object")
        for row in bindings
        if isinstance(row, Mapping)
    }
    if len(bindings) != 3 or observed_objects != expected_objects:
        raise SystemExit("M7 T2 2048 content bindings drifted")


@contextmanager
def controller_lock(run_dir: pathlib.Path):
    if os.name != "posix":
        raise SystemExit("continuous supervisor must run inside the pinned WSL host")
    import fcntl

    path = run_dir / "supervisor.lock"
    stream = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(f"another supervisor holds {path}") from error
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def validate_done_marker(
    raw: bytes | None,
    *,
    plan: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> bool:
    if raw is None:
        return False
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"shard {shard['shard_id']} has an unreadable SHARD_DONE marker"
        ) from error
    expected = {
        "schema": DONE_SCHEMA,
        "plan_sha256": plan["worker_plan_sha256"],
        "shard_id": shard["shard_id"],
        "positions": shard["count"],
    }
    if payload != expected or raw != canonical_bytes(expected):
        raise SystemExit(
            f"shard {shard['shard_id']} has a foreign or non-canonical "
            f"SHARD_DONE marker: {payload!r}"
        )
    return True


def validate_complete_checkpoint(
    raw: bytes | None,
    *,
    plan: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> bool:
    """Require the after-all-files checkpoint, not only SHARD_DONE.

    The startup publisher writes this checkpoint only after every file upload
    has returned successfully.  Requiring its exact file inventory closes the
    race where a DONE object becomes visible while final position objects are
    still in flight.
    """

    if raw is None:
        return False
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"shard {shard['shard_id']} has an unreadable complete checkpoint"
        ) from error
    if raw != canonical_bytes(payload):
        raise SystemExit(
            f"shard {shard['shard_id']} complete checkpoint is non-canonical"
        )
    stated_digest = payload.get("checkpoint_sha256")
    unsigned = dict(payload)
    unsigned.pop("checkpoint_sha256", None)
    if stated_digest != hashlib.sha256(canonical_bytes(unsigned)).hexdigest():
        raise SystemExit(
            f"shard {shard['shard_id']} complete checkpoint digest mismatch"
        )
    expected_keys = {
        "schema", "checkpoint_kind", "plan_sha256", "shard_id", "attempt_id",
        "completed_position_count", "complete", "files",
        "checkpoint_published_after_files", "create_only", "checkpoint_sha256",
    }
    if set(payload) != expected_keys:
        raise SystemExit(
            f"shard {shard['shard_id']} complete checkpoint fields drifted"
        )
    count = int(shard["count"])
    if (
        payload["schema"] != COMPLETE_CHECKPOINT_SCHEMA
        or payload["checkpoint_kind"] != "complete"
        or payload["plan_sha256"] != plan["worker_plan_sha256"]
        or payload["shard_id"] != shard["shard_id"]
        or not isinstance(payload["attempt_id"], str)
        or not payload["attempt_id"]
        or payload["completed_position_count"] != count
        or payload["complete"] is not True
        or payload["checkpoint_published_after_files"] is not True
        or payload["create_only"] is not True
    ):
        raise SystemExit(
            f"shard {shard['shard_id']} complete checkpoint provenance drifted"
        )
    files = payload["files"]
    if not isinstance(files, list):
        raise SystemExit(f"shard {shard['shard_id']} checkpoint files are not a list")
    expected_relatives = {"SHARD_DONE.json"} | {
        f"position_{offset:08d}.json"
        for offset in range(int(shard["start"]), int(shard["start"]) + count)
    }
    observed_relatives: set[str] = set()
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path", "object_name", "sha256", "bytes"
        }:
            raise SystemExit(
                f"shard {shard['shard_id']} checkpoint file row drifted"
            )
        relative = row["relative_path"]
        if relative in observed_relatives:
            raise SystemExit(
                f"shard {shard['shard_id']} checkpoint duplicates {relative!r}"
            )
        observed_relatives.add(relative)
        if row["object_name"] != f"{shard['object_prefix']}/files/{relative}":
            raise SystemExit(
                f"shard {shard['shard_id']} checkpoint object path drifted"
            )
        digest = row["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
            or not isinstance(row["bytes"], int)
            or row["bytes"] <= 0
        ):
            raise SystemExit(
                f"shard {shard['shard_id']} checkpoint file metadata is invalid"
            )
    if observed_relatives != expected_relatives:
        missing = sorted(expected_relatives - observed_relatives)[:3]
        extra = sorted(observed_relatives - expected_relatives)[:3]
        raise SystemExit(
            f"shard {shard['shard_id']} checkpoint inventory mismatch; "
            f"missing={missing}, extra={extra}"
        )
    return True


def _list_prefix(
    adapter: Adapter, *, bucket: str, prefix: str
) -> list[Mapping[str, Any]]:
    """List a small per-shard prefix through the adapter's authenticated GET.

    The production adapter intentionally exposes point reads only.  A heartbeat
    is an append-only sequence, so finding the newest one requires the narrow
    JSON-API list call here.  Test adapters may provide ``list_prefix``
    directly; the fallback uses the same private, retrying GET primitive as all
    other adapter reads and never accesses its token.
    """

    direct = getattr(adapter, "list_prefix", None)
    if direct is not None:
        return list(direct(bucket=bucket, prefix=prefix))
    call = getattr(adapter, "_call", None)
    decode = getattr(adapter, "_json", None)
    if call is None or decode is None:
        raise SystemExit("adapter cannot list heartbeat objects")
    items: list[Mapping[str, Any]] = []
    page_token = ""
    while True:
        query = {
            "prefix": prefix,
            "fields": "items(name,updated),nextPageToken",
            "maxResults": "1000",
        }
        if page_token:
            query["pageToken"] = page_token
        url = (
            "https://storage.googleapis.com/storage/v1/b/"
            + urllib.parse.quote(bucket, safe="")
            + "/o?"
            + urllib.parse.urlencode(query)
        )
        payload = decode(call("GET", url), "GCS heartbeat listing")
        page_items = payload.get("items", [])
        if not isinstance(page_items, list):
            raise SystemExit("GCS heartbeat listing has non-list items")
        items.extend(row for row in page_items if isinstance(row, Mapping))
        page_token = payload.get("nextPageToken", "")
        if not page_token:
            return items


def _instance_metadata_value(instance: Mapping[str, Any], key: str) -> str | None:
    metadata = instance.get("metadata", {})
    rows = metadata.get("items", []) if isinstance(metadata, Mapping) else []
    values = [
        row.get("value")
        for row in rows
        if isinstance(row, Mapping) and row.get("key") == key
    ]
    if len(values) > 1:
        raise SystemExit(f"instance metadata duplicates {key!r}")
    if not values:
        return None
    value = values[0]
    if not isinstance(value, str) or not value:
        raise SystemExit(f"instance metadata {key!r} is not non-empty text")
    return value


def _latest_heartbeat(
    adapter: Adapter,
    *,
    plan: Mapping[str, Any],
    shard: Mapping[str, Any],
    attempt_id: str,
) -> str | None:
    prefix = f"{shard['object_prefix']}/heartbeats/{attempt_id}-"
    rows = _list_prefix(
        adapter, bucket=plan["bucket"], prefix=prefix
    )
    if not rows:
        return None
    candidates: list[tuple[str, str]] = []
    for row in rows:
        name = row.get("name")
        updated = row.get("updated")
        if (
            not isinstance(name, str)
            or not name.startswith(prefix)
            or not name.endswith(".json")
            or not name[len(prefix):-5].isdigit()
            or not isinstance(updated, str)
            or not updated
        ):
            raise SystemExit(
                f"shard {shard['shard_id']} has malformed heartbeat metadata"
            )
        _parse_utc(updated)
        candidates.append((updated, name))
    updated, name = max(candidates, key=lambda row: _parse_utc(row[0]))
    raw = adapter.get_object_bytes(bucket=plan["bucket"], object_name=name)
    if raw is None:
        raise SystemExit(f"heartbeat disappeared after listing: {name}")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"heartbeat is unreadable: {name}") from error
    if raw != canonical_bytes(payload):
        raise SystemExit(f"heartbeat is non-canonical: {name}")
    stated = payload.get("heartbeat_sha256")
    unsigned = dict(payload)
    unsigned.pop("heartbeat_sha256", None)
    if stated != hashlib.sha256(canonical_bytes(unsigned)).hexdigest():
        raise SystemExit(f"heartbeat digest mismatch: {name}")
    expected_keys = {
        "schema", "plan_sha256", "shard_id", "attempt_id", "sequence",
        "completed_position_count", "create_only", "observed_at_utc",
        "heartbeat_sha256",
    }
    sequence_text = name[len(prefix):-5]
    if (
        set(payload) != expected_keys
        or payload["schema"] != "hu_m31_label_gen_heartbeat_v1"
        or payload["plan_sha256"] != plan["worker_plan_sha256"]
        or payload["shard_id"] != shard["shard_id"]
        or payload["attempt_id"] != attempt_id
        or not isinstance(payload["sequence"], int)
        or payload["sequence"] < 0
        or int(sequence_text) != payload["sequence"]
        or not isinstance(payload["completed_position_count"], int)
        or not 0 <= payload["completed_position_count"] <= int(shard["count"])
        or payload["create_only"] is not True
        or not isinstance(payload["observed_at_utc"], str)
    ):
        raise SystemExit(f"heartbeat provenance mismatch: {name}")
    _parse_utc(payload["observed_at_utc"])
    return updated


def inspect_shards(adapter: Adapter, plan: Mapping[str, Any]) -> list[ShardState]:
    states: list[ShardState] = []
    for shard in plan["shards"]:
        marker_name = f"{shard['object_prefix']}/files/SHARD_DONE.json"
        marker_valid = validate_done_marker(
            adapter.get_object_bytes(bucket=plan["bucket"], object_name=marker_name),
            plan=plan,
            shard=shard,
        )
        checkpoint_name = complete_checkpoint_object_name(shard["object_prefix"])
        checkpoint_valid = validate_complete_checkpoint(
            adapter.get_object_bytes(
                bucket=plan["bucket"], object_name=checkpoint_name
            ),
            plan=plan,
            shard=shard,
        )
        if checkpoint_valid and not marker_valid:
            raise SystemExit(
                f"shard {shard['shard_id']} has a complete checkpoint but its "
                "SHARD_DONE object is absent"
            )
        done = marker_valid and checkpoint_valid
        instance = adapter.get_instance(
            instance_name=_instance_name(plan["run_name"], shard["shard_id"])
        )
        heartbeat = None
        attempt_id = None
        if not done and instance is not None:
            attempt_id = _instance_metadata_value(instance, "lg-attempt-id")
            if attempt_id is None:
                raise SystemExit(
                    f"owned instance for shard {shard['shard_id']} has no "
                    "lg-attempt-id metadata"
                )
            heartbeat = _latest_heartbeat(
                adapter,
                plan=plan,
                shard=shard,
                attempt_id=attempt_id,
            )
        states.append(
            ShardState(
                shard_id=shard["shard_id"],
                done=done,
                instance_status=(str(instance.get("status")) if instance else None),
                instance_created_at=(
                    str(instance.get("creationTimestamp"))
                    if instance and instance.get("creationTimestamp")
                    else None
                ),
                latest_heartbeat_at=heartbeat,
                attempt_id=attempt_id,
            )
        )
    return states


def completion_is_durable(
    adapter: Adapter, plan: Mapping[str, Any], shard: Mapping[str, Any]
) -> bool:
    marker_name = f"{shard['object_prefix']}/files/SHARD_DONE.json"
    marker = validate_done_marker(
        adapter.get_object_bytes(bucket=plan["bucket"], object_name=marker_name),
        plan=plan,
        shard=shard,
    )
    checkpoint_name = complete_checkpoint_object_name(shard["object_prefix"])
    checkpoint = validate_complete_checkpoint(
        adapter.get_object_bytes(
            bucket=plan["bucket"], object_name=checkpoint_name
        ),
        plan=plan,
        shard=shard,
    )
    if checkpoint and not marker:
        raise SystemExit(
            f"shard {shard['shard_id']} has a complete checkpoint but its "
            "SHARD_DONE object is absent"
        )
    return marker and checkpoint


def _parse_utc(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SystemExit(f"invalid cloud timestamp {value!r}") from error
    if parsed.tzinfo is None:
        raise SystemExit(f"cloud timestamp has no timezone: {value!r}")
    return parsed.astimezone(dt.timezone.utc)


def stale_incomplete_states(
    states: list[ShardState],
    *,
    now: dt.datetime,
    boot_grace_seconds: int,
    heartbeat_stale_seconds: int,
) -> list[ShardState]:
    if now.tzinfo is None:
        raise SystemExit("supervisor clock must be timezone-aware")
    stale: list[ShardState] = []
    for state in states:
        if state.done or state.instance_status is None:
            continue
        if state.instance_status in {"STOPPED", "SUSPENDED", "TERMINATED"}:
            continue
        if state.latest_heartbeat_at:
            age = (now - _parse_utc(state.latest_heartbeat_at)).total_seconds()
            if age > heartbeat_stale_seconds:
                stale.append(state)
        elif state.instance_created_at:
            age = (now - _parse_utc(state.instance_created_at)).total_seconds()
            if age > boot_grace_seconds:
                stale.append(state)
    return stale


def _refresh_gcloud_token() -> str:
    completed = subprocess.run(
        ["gcloud", "auth", "print-access-token", "--quiet"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    token = completed.stdout.strip()
    if not token:
        raise SystemExit("gcloud returned an empty access token")
    return token


def _project_instance_names(zone: str) -> set[str]:
    """Instances in ONE zone.

    Scoped to a zone because the capacity it guards -- PREEMPTIBLE_CPUS -- is a
    regional quota. A project-wide count would make a fleet in one region read
    another region's fleet as competition for its own slots.
    """
    completed = subprocess.run(
        [
            "gcloud", "compute", "instances", "list",
            f"--project={PROJECT}", f"--zones={zone}",
            "--format=value(name)", "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def make_adapter(*, refresh_gcloud_token: bool, zone: str):
    if refresh_gcloud_token:
        os.environ["GOOGLE_OAUTH_ACCESS_TOKEN"] = _refresh_gcloud_token()
    token = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN", "")
    if not token:
        raise SystemExit(
            "GOOGLE_OAUTH_ACCESS_TOKEN is required, or pass "
            "--refresh-gcloud-token"
        )
    from .hu_m31_t3_step6d_fresh_quality_gcp_provider_v1 import GcpQualityRestAdapter

    return GcpQualityRestAdapter(access_token=token, zone=zone)


def _startup_source(plan: Mapping[str, Any]) -> str:
    path = pathlib.Path(__file__).resolve().parents[2] / plan["startup_script_relative"]
    source = path.read_text(encoding="utf-8")
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual != plan["startup_script_sha256"]:
        raise SystemExit(
            "startup script changed since planning; generate a fresh run plan"
        )
    return source


def _receipt_name(kind: str, shard_id: str, attempt_id: str) -> str:
    return f"supervisor_{kind}_{time.time_ns()}_{shard_id}_{attempt_id}.json"


def _submit_delete(
    *,
    adapter: Adapter,
    plan: Mapping[str, Any],
    run_dir: pathlib.Path,
    shard_id: str,
    reason: str,
) -> None:
    request_id = str(uuid.uuid4())
    operation_id = str(uuid.uuid4())
    name = _instance_name(plan["run_name"], shard_id)
    intent = {
        "schema": DELETE_INTENT_SCHEMA,
        "run_name": plan["run_name"],
        "shard_id": shard_id,
        "instance_name": name,
        "request_id": request_id,
        "operation_id": operation_id,
        "plan_sha256": plan["worker_plan_sha256"],
        "reason": reason,
        "written_before_delete": True,
    }
    _write_once(
        run_dir / _receipt_name("delete_intent", shard_id, operation_id), intent
    )
    try:
        operation = adapter.delete_instance(instance_name=name, request_id=request_id)
    except Exception as error:  # noqa: BLE001 - persisted audit result
        result = {
            "schema": DELETE_RESULT_SCHEMA,
            **{key: intent[key] for key in (
                "run_name", "shard_id", "instance_name", "request_id",
                "operation_id", "plan_sha256", "reason",
            )},
            "status": "delete_failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_once(
            run_dir / _receipt_name("delete_result", shard_id, operation_id),
            result,
        )
        raise
    result = {
        "schema": DELETE_RESULT_SCHEMA,
        **{key: intent[key] for key in (
            "run_name", "shard_id", "instance_name", "request_id",
            "operation_id", "plan_sha256", "reason",
        )},
        "status": "absent" if operation is None else "delete_submitted",
        "operation": None if operation is None else operation.get("name"),
    }
    _write_once(
        run_dir / _receipt_name("delete_result", shard_id, operation_id), result
    )


def supervise_once(
    *,
    adapter: Adapter,
    plan: Mapping[str, Any],
    run_dir: pathlib.Path,
    service_account: str,
    image: str,
    max_live: int,
    max_run_seconds: int,
    allow_writes: bool,
    boot_grace_seconds: int = DEFAULT_BOOT_GRACE_SECONDS,
    heartbeat_stale_seconds: int = DEFAULT_HEARTBEAT_STALE_SECONDS,
    max_attempts_per_shard: int = 5,
    project_instance_names: set[str] | None = None,
) -> dict[str, Any]:
    if not 0 < max_live <= 58:
        raise SystemExit("max_live must be in 1..58")
    if max_run_seconds <= 0:
        raise SystemExit("max_run_seconds must be positive")
    stage = load_stage_receipt(run_dir, plan)
    states = inspect_shards(adapter, plan)
    by_id = {state.shard_id: state for state in states}
    shard_by_id = {shard["shard_id"]: shard for shard in plan["shards"]}
    complete = [state for state in states if state.done]
    occupied = [state for state in states if state.instance_status is not None]
    dead_statuses = {"STOPPED", "SUSPENDED", "TERMINATED"}
    dead_incomplete = [
        state
        for state in states
        if not state.done and state.instance_status in dead_statuses
    ]
    stale_incomplete = stale_incomplete_states(
        states,
        now=dt.datetime.now(dt.timezone.utc),
        boot_grace_seconds=boot_grace_seconds,
        heartbeat_stale_seconds=heartbeat_stale_seconds,
    )

    # A valid done marker is durable only after every position file was
    # published.  Deleting its lingering VM is therefore safe and immediately
    # makes quota available to the next shard.  A later cycle observes absence
    # before trying to use that slot.
    done_deletes: list[str] = []
    dead_deletes: list[str] = []
    stale_deletes: list[str] = []
    if allow_writes:
        for state in complete:
            if state.instance_status is None:
                continue
            # Re-read both durable completion witnesses and the instance just
            # before deletion.  A worker may finish between the initial scan
            # and this mutation; deletion remains safe only after this check.
            shard = shard_by_id[state.shard_id]
            if not completion_is_durable(adapter, plan, shard):
                continue
            name = _instance_name(plan["run_name"], state.shard_id)
            if adapter.get_instance(instance_name=name) is None:
                continue
            _submit_delete(
                adapter=adapter, plan=plan, run_dir=run_dir,
                shard_id=state.shard_id, reason="durable_complete",
            )
            done_deletes.append(state.shard_id)
        for state in dead_incomplete:
            name = _instance_name(plan["run_name"], state.shard_id)
            current = adapter.get_instance(instance_name=name)
            if current is None:
                continue
            if (
                _instance_metadata_value(current, "lg-attempt-id") != state.attempt_id
                or str(current.get("status")) not in dead_statuses
            ):
                continue
            if completion_is_durable(adapter, plan, shard_by_id[state.shard_id]):
                continue
            _submit_delete(
                adapter=adapter, plan=plan, run_dir=run_dir,
                shard_id=state.shard_id, reason="dead_incomplete",
            )
            dead_deletes.append(state.shard_id)
        for state in stale_incomplete:
            name = _instance_name(plan["run_name"], state.shard_id)
            current = adapter.get_instance(instance_name=name)
            if current is None:
                continue
            if _instance_metadata_value(current, "lg-attempt-id") != state.attempt_id:
                continue
            latest = _latest_heartbeat(
                adapter,
                plan=plan,
                shard=shard_by_id[state.shard_id],
                attempt_id=state.attempt_id,
            )
            now = dt.datetime.now(dt.timezone.utc)
            if latest is not None:
                if (
                    now - _parse_utc(latest)
                ).total_seconds() <= heartbeat_stale_seconds:
                    continue
            else:
                created = current.get("creationTimestamp")
                if (
                    not isinstance(created, str)
                    or (now - _parse_utc(created)).total_seconds()
                    <= boot_grace_seconds
                ):
                    continue
            if completion_is_durable(adapter, plan, shard_by_id[state.shard_id]):
                continue
            _submit_delete(
                adapter=adapter, plan=plan, run_dir=run_dir,
                shard_id=state.shard_id, reason="stale_heartbeat",
            )
            stale_deletes.append(state.shard_id)

    available = max(0, max_live - len(occupied))
    foreign_instances: set[str] = set()
    if project_instance_names is not None:
        owned_names = {
            _instance_name(plan["run_name"], shard["shard_id"])
            for shard in plan["shards"]
        }
        foreign_instances = set(project_instance_names) - owned_names
        # project_instance_names is scoped to this run's zone, and the budget is
        # that zone's own share of the regional Spot quota. Any instance in the
        # zone that is not ours -- another run, a stray -- is still spending the
        # same quota, so it conservatively occupies a slot.
        zone_budget = ZONE_FLEET_BUDGET[_placement(plan)[0]]
        available = min(
            available,
            max(0, zone_budget - len(project_instance_names)),
        )
    candidates = [
        shard
        for shard in plan["shards"]
        if not by_id[shard["shard_id"]].done
        and by_id[shard["shard_id"]].instance_status is None
    ]
    launches: list[dict[str, Any]] = []
    if available and candidates:
        if not service_account:
            raise SystemExit("service_account is required when shards need launch")
        startup = _startup_source(plan)
        bindings_b64 = base64.b64encode(
            canonical_bytes(stage["content_bindings"])
        ).decode("ascii")
        for shard in candidates[:available]:
            # TOCTOU guard immediately before create: a previous attempt can
            # publish its final checkpoint or become visible after the scan.
            if completion_is_durable(adapter, plan, shard):
                continue
            if adapter.get_instance(
                instance_name=_instance_name(plan["run_name"], shard["shard_id"])
            ) is not None:
                continue
            # The write-once intent precedes the API call.  Counting intents is
            # conservative across the ambiguous case where Compute accepted a
            # create but the controller died before it could persist a result.
            attempts = len(list(run_dir.glob(
                f"supervisor_intent_*_{shard['shard_id']}_*.json"
            )))
            if attempts >= max_attempts_per_shard:
                raise SystemExit(
                    f"shard {shard['shard_id']} reached the fail-closed attempt "
                    f"limit {max_attempts_per_shard}"
                )
            attempt_id = str(uuid.uuid4())
            request_id = str(uuid.uuid4())
            spec = _instance_spec(
                plan,
                shard,
                bindings_b64=bindings_b64,
                attempt_id=attempt_id,
                startup_script=startup,
                service_account=service_account,
                image=image,
                max_run_seconds=max_run_seconds,
            )
            intent = {
                "schema": INTENT_SCHEMA,
                "run_name": plan["run_name"],
                "shard_id": shard["shard_id"],
                "instance_name": spec["name"],
                "attempt_id": attempt_id,
                "request_id": request_id,
                "plan_sha256": plan["worker_plan_sha256"],
                "spec_sha256": hashlib.sha256(canonical_bytes(spec)).hexdigest(),
                "image": image,
                "service_account": service_account,
                "max_run_seconds": max_run_seconds,
                "written_before_create": True,
            }
            if not allow_writes:
                launches.append({**intent, "render_only": True})
                continue
            _write_once(
                run_dir / _receipt_name("intent", shard["shard_id"], attempt_id),
                intent,
            )
            try:
                operation = adapter.create_instance(
                    instance_spec=spec, request_id=request_id
                )
            except Exception as error:  # noqa: BLE001 - persisted audit result
                result = {
                    "schema": RESULT_SCHEMA,
                    **{key: intent[key] for key in (
                    "run_name", "shard_id", "instance_name", "attempt_id",
                        "request_id", "plan_sha256", "spec_sha256", "image",
                        "service_account", "max_run_seconds",
                    )},
                    "status": "create_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                _write_once(
                    run_dir / _receipt_name("result", shard["shard_id"], attempt_id),
                    result,
                )
                launches.append(result)
                # Quota and transient capacity failures make the rest of this
                # cycle likely to fail too.  Preserve completed creates and let
                # the next monitored cycle retry with a fresh token/request ID.
                break
            result = {
                "schema": RESULT_SCHEMA,
                **{key: intent[key] for key in (
                    "run_name", "shard_id", "instance_name", "attempt_id",
                    "request_id", "plan_sha256", "spec_sha256", "image",
                    "service_account", "max_run_seconds",
                )},
                "status": "create_submitted",
                "operation": operation.get("name"),
            }
            _write_once(
                run_dir / _receipt_name("result", shard["shard_id"], attempt_id),
                result,
            )
            launches.append(result)

    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "run_name": plan["run_name"],
        "total_shards": len(states),
        "complete_shards": len(complete),
        "occupied_instances": len(occupied),
        "launch_slots": available,
        "launches": launches,
        "done_instances_delete_submitted": done_deletes,
        "dead_incomplete_delete_submitted": dead_deletes,
        "stale_incomplete_delete_submitted": stale_deletes,
        "foreign_project_instances": sorted(foreign_instances),
        "project_instances": (
            sorted(project_instance_names)
            if project_instance_names is not None
            else None
        ),
        "observed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(
        f"{plan['run_name']}: {len(complete)}/{len(states)} complete, "
        f"{len(occupied)} instances present, {len(launches)} launch attempts",
        flush=True,
    )
    return snapshot


def _durable_receipt(run_dir: pathlib.Path, plan: Mapping[str, Any]) -> None:
    path = run_dir / "supervisor_generation_durable.json"
    payload = {
        "schema": DURABLE_SCHEMA,
        "run_name": plan["run_name"],
        "worker_plan_sha256": plan["worker_plan_sha256"],
        "shards": len(plan["shards"]),
        "positions": sum(int(shard["count"]) for shard in plan["shards"]),
        "meaning": (
            "all canonical DONE markers and after-all-files checkpoints are "
            "durable and no owned instance remains; this is not the final "
            "download/postflight validation receipt"
        ),
    }
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise SystemExit("existing supervisor durable receipt disagrees")
        return
    _write_once(path, payload)


def generation_is_durable(snapshot: Mapping[str, Any]) -> bool:
    """All run-owned work is durable even when unrelated project VMs exist."""

    return (
        snapshot["complete_shards"] == snapshot["total_shards"]
        and snapshot["occupied_instances"] == 0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract", required=True, choices=("m7-t2-2048-25k-v1",)
    )
    parser.add_argument("--plan-receipt", required=True)
    parser.add_argument("--service-account", default="")
    parser.add_argument("--image", default=PINNED_IMAGE)
    parser.add_argument("--max-live", type=int, default=58)
    parser.add_argument("--max-run-seconds", type=int, default=7 * 3600)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument(
        "--boot-grace-seconds", type=int, default=DEFAULT_BOOT_GRACE_SECONDS
    )
    parser.add_argument(
        "--heartbeat-stale-seconds",
        type=int,
        default=DEFAULT_HEARTBEAT_STALE_SECONDS,
    )
    parser.add_argument("--max-attempts-per-shard", type=int, default=5)
    parser.add_argument("--max-cycle-errors", type=int, default=20)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--allow-writes", action="store_true")
    parser.add_argument("--refresh-gcloud-token", action="store_true")
    args = parser.parse_args()

    plan = load_run_plan(pathlib.Path(args.plan_receipt))
    validate_m7_t2_2048_run_contract(
        plan,
        image=args.image,
        max_live=args.max_live,
        max_run_seconds=args.max_run_seconds,
    )
    if not args.once and args.allow_writes and not args.refresh_gcloud_token:
        raise SystemExit(
            "continuous write supervision requires --refresh-gcloud-token"
        )
    run_dir = RECEIPT_ROOT / plan["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.poll_seconds < 30:
        raise SystemExit("poll_seconds must be at least 30")
    if args.max_cycle_errors <= 0:
        raise SystemExit("max_cycle_errors must be positive")
    with controller_lock(run_dir):
        cycle_errors = 0
        while True:
            try:
                adapter = make_adapter(
                    refresh_gcloud_token=args.refresh_gcloud_token,
                    zone=_placement(plan)[0],
                )
                project_names = (
                    _project_instance_names(_placement(plan)[0])
                    if args.refresh_gcloud_token
                    else None
                )
                snapshot = supervise_once(
                    adapter=adapter,
                    plan=plan,
                    run_dir=run_dir,
                    service_account=args.service_account,
                    image=args.image,
                    max_live=args.max_live,
                    max_run_seconds=args.max_run_seconds,
                    allow_writes=args.allow_writes,
                    boot_grace_seconds=args.boot_grace_seconds,
                    heartbeat_stale_seconds=args.heartbeat_stale_seconds,
                    max_attempts_per_shard=args.max_attempts_per_shard,
                    project_instance_names=project_names,
                )
                _write_once(
                    run_dir / f"supervisor_snapshot_{time.time_ns()}.json", snapshot
                )
                cycle_errors = 0
            except Exception as error:  # transient token/API/controller IO
                cycle_errors += 1
                failure = {
                    "schema": "hu_m31_label_gen_supervisor_cycle_error_v1",
                    "run_name": plan["run_name"],
                    "sequence": cycle_errors,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "observed_at_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                }
                _write_once(
                    run_dir / f"supervisor_cycle_error_{time.time_ns()}.json",
                    failure,
                )
                if args.once or cycle_errors >= args.max_cycle_errors:
                    raise
                time.sleep(min(300, 15 * cycle_errors))
                continue
            if generation_is_durable(snapshot):
                _durable_receipt(run_dir, plan)
                return 0
            if args.once:
                return 0
            if not args.allow_writes:
                raise SystemExit(
                    "continuous supervision without --allow-writes cannot make progress"
                )
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
