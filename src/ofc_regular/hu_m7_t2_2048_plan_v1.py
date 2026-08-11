"""Build the immutable M7v5 T2 2,048-particle label-plan pair.

This module is intentionally narrower than the generic label planner.  It
names one purchased experiment: 25,000 paired behavior roots for each T2 seat,
evaluated with 2,048 particles against the accepted M7 continuation.  Before a
plan can be rendered it re-hashes the package archives and every digest the
M7v5 ledger declares.  A package with a self-consistent but different ledger
is refused as well; this generator is for the reviewed M7v5 image, not for the
next image that happens to have the same directory layout.

The two seats deliberately share the complete hand/behavior seed block.  They
do not share label files or plan digests.  Acting first at T2 still has the
opponent's T2 second-seat action ahead of it, so only that plan pins the
generation-1 T2-second model.

Output is write-once.  Existing plans are evidence and are never overwritten.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import pathlib
import re
import tarfile
import zipfile
from collections.abc import Mapping
from typing import Any


PLAN_SCHEMA = "hu_m31_label_gen_plan_v1"
LEDGER_SCHEMA = "hu_m31_label_gen_package_ledger_v1"
PROVENANCE_SCHEMA = "hu_m7_t2_2048_plan_provenance_v1"
PLAN_SET_SCHEMA = "hu_m7_t2_2048_plan_set_v1"

POSITION_COUNT = 25_000
SHARD_COUNT = 174
SAMPLES = 2_048
SEEDS_PER_POSITION = 1
HAND_SEED_BASE = 947_000_000
BEHAVIOR_SEED_OFFSET = 500_000
EVAL_SEED_BASE = 10_000_000
FL_EV_CARDS = 14
FL_EV_VALUE = 9.6

FIRST_JOB_ID = "m7v5-t2first-25k-2048p"
SECOND_JOB_ID = "m7v5-t2second-25k-2048p"
FIRST_FILENAME = "worker_plan_m7v5_t2first_25k_2048p.json"
SECOND_FILENAME = "worker_plan_m7v5_t2second_25k_2048p.json"
MANIFEST_FILENAME = "plan_set_m7v5_t2_25k_2048p.json"

# Exact reviewed package identity.  Verifying the ledger digest first prevents
# a later, internally consistent package from silently taking this run name.
EXPECTED_LEDGER_SHA256 = (
    "d124e98617a34d7767f0da9e04b5ceea489c97a52009b7779ffd150b20ee5dcf"
)
EXPECTED_RUNTIME_SHA256 = (
    "ae8be6cd2aa333654141368c1ad713a1c6ab3ff1dedea752edcdeab2551d1988"
)
EXPECTED_WHEELHOUSE_SHA256 = (
    "338ca072984775dc886e7c9c88d1d7e10b5363ae4aff5a7ba936eebf57d25053"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")

# Every non-archive digest declared by ledger.json has exactly one member whose
# bytes it identifies.  Keeping the table exhaustive is deliberate: if a later
# ledger adds another digest this version fails instead of claiming an all-hash
# audit while ignoring the new field.
_DECLARED_MEMBER_BY_LEDGER_PATH = {
    "engine_library_sha256": "runtime/native/libofc_hu_m3_engine.so",
    "feature_encoder_library_sha256": (
        "runtime/native/libofc_stage3_feature_encoder.so"
    ),
    "fast_t0_second_model_sha256": "runtime/weights/fast_t0_second_v1.bin",
    "fast_t1_first_model_sha256": "runtime/weights/fast_t1_first_v1.bin",
    "fast_t1_second_model_sha256": "runtime/weights/fast_t1_second_v1.bin",
    "fast_t2_first_model_sha256": "runtime/weights/fast_t2_first_v1.bin",
    "fast_t2_second_model_sha256": "runtime/weights/fast_t2_second_v1.bin",
    "t0_first_model_sha256": "runtime/weights/t0first_model_v1.bin",
    "t0_second_model_sha256": "runtime/weights/t0_model_v1.bin",
    "t1_first_model_sha256": "runtime/weights/t1first_model_v1.bin",
    "t1_second_model_sha256": "runtime/weights/t1_model_v1.bin",
    "t2_first_model_sha256": "runtime/weights/t2first_model_v1.bin",
    "t2_second_model_sha256": "runtime/weights/t2_model_v1.bin",
    "t3_first_model_sha256": "runtime/weights/t3first_model_v1.bin",
    "t3_first_model_v2_sha256": "runtime/weights/t3first_model_v2.bin",
    "t3_second_model_sha256": "runtime/weights/t3_model_v2.bin",
    "t3_second_model_v3_sha256": "runtime/weights/t3_model_v3.bin",
    "t4_model_sha256": "runtime/weights/t4_model_v5.bin",
    "t4_model_v6_sha256": "runtime/weights/t4_model_v6.bin",
    "fl_ev.config_sha256": "runtime/configs/fl_ev_regular_v4_selfplay.json",
}

_ARCHIVE_LEDGER_PATHS = {
    "runtime_archive.sha256",
    "wheelhouse_archive.sha256",
}


class PlanValidationError(ValueError):
    """The immutable input contract does not describe the reviewed run."""


@dataclasses.dataclass(frozen=True)
class ExpectedPackageIdentity:
    ledger_sha256: str
    runtime_sha256: str
    wheelhouse_sha256: str


DEFAULT_PACKAGE_IDENTITY = ExpectedPackageIdentity(
    ledger_sha256=EXPECTED_LEDGER_SHA256,
    runtime_sha256=EXPECTED_RUNTIME_SHA256,
    wheelhouse_sha256=EXPECTED_WHEELHOUSE_SHA256,
)


@dataclasses.dataclass(frozen=True)
class PackageAudit:
    package_dir: pathlib.Path
    ledger_sha256: str
    runtime_sha256: str
    runtime_bytes: int
    wheelhouse_sha256: str
    wheelhouse_bytes: int
    declared_hashes: Mapping[str, str]
    member_hashes: Mapping[str, str]
    python_files: int
    model_files: int
    weight_files: int
    wheel_files: int

    def provenance(self) -> dict[str, Any]:
        return {
            "schema": "hu_m7v5_package_hash_audit_v1",
            "ledger_sha256": self.ledger_sha256,
            "runtime_archive": {
                "sha256": self.runtime_sha256,
                "bytes": self.runtime_bytes,
            },
            "wheelhouse_archive": {
                "sha256": self.wheelhouse_sha256,
                "bytes": self.wheelhouse_bytes,
            },
            "declared_hashes": dict(sorted(self.declared_hashes.items())),
            "verified_archive_members": dict(sorted(self.member_hashes.items())),
            "verified_counts": {
                "python_files": self.python_files,
                "model_files": self.model_files,
                "weight_files": self.weight_files,
                "wheel_files": self.wheel_files,
            },
            "all_declared_hashes_recomputed": True,
        }


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def rendered_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanValidationError(message)


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase 64-character sha256, got {value!r}",
    )
    return value


def _mapping_value(root: Mapping[str, Any], dotted: str) -> Any:
    value: Any = root
    for component in dotted.split("."):
        _require(isinstance(value, Mapping), f"{dotted} crosses a non-object")
        _require(component in value, f"ledger is missing {dotted}")
        value = value[component]
    return value


def _declared_sha256_values(value: Any, prefix: str = "") -> dict[str, str]:
    found: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if str(key) == "sha256" or str(key).endswith("_sha256"):
                found[dotted] = _require_sha256(child, f"ledger {dotted}")
            else:
                found.update(_declared_sha256_values(child, dotted))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(_declared_sha256_values(child, f"{prefix}[{index}]"))
    return found


def _safe_archive_name(name: str, label: str) -> None:
    path = pathlib.PurePosixPath(name)
    _require(not path.is_absolute(), f"{label} contains absolute path {name!r}")
    _require(".." not in path.parts, f"{label} contains traversal path {name!r}")


def _read_unique_tar_member(
    archive: tarfile.TarFile, members_by_name: Mapping[str, list[tarfile.TarInfo]], name: str
) -> bytes:
    members = members_by_name.get(name, [])
    _require(len(members) == 1, f"runtime archive has {len(members)} copies of {name}")
    member = members[0]
    _require(member.isfile(), f"runtime archive member is not a file: {name}")
    stream = archive.extractfile(member)
    _require(stream is not None, f"runtime archive cannot read {name}")
    return stream.read()


def audit_package(
    package_dir: pathlib.Path,
    *,
    expected_identity: ExpectedPackageIdentity = DEFAULT_PACKAGE_IDENTITY,
) -> PackageAudit:
    """Recompute the archive identity and every hash declared by its ledger."""

    package_dir = package_dir.resolve()
    ledger_path = package_dir / "ledger.json"
    runtime_path = package_dir / "runtime.tar.gz"
    wheelhouse_path = package_dir / "wheelhouse.zip"
    for path in (ledger_path, runtime_path, wheelhouse_path):
        _require(path.is_file(), f"M7v5 package artifact is missing: {path}")

    ledger_raw = ledger_path.read_bytes()
    ledger_sha = sha256_bytes(ledger_raw)
    _require(
        ledger_sha == expected_identity.ledger_sha256,
        f"ledger digest {ledger_sha} does not match reviewed "
        f"{expected_identity.ledger_sha256}",
    )
    try:
        ledger = json.loads(ledger_raw)
    except json.JSONDecodeError as error:
        raise PlanValidationError(f"ledger is not valid JSON: {error}") from error
    _require(isinstance(ledger, Mapping), "ledger root must be an object")
    _require(
        ledger.get("schema") == LEDGER_SCHEMA,
        f"unsupported ledger schema {ledger.get('schema')!r}",
    )

    runtime_sha = sha256_file(runtime_path)
    wheelhouse_sha = sha256_file(wheelhouse_path)
    _require(
        runtime_sha == expected_identity.runtime_sha256,
        f"runtime digest {runtime_sha} does not match reviewed "
        f"{expected_identity.runtime_sha256}",
    )
    _require(
        wheelhouse_sha == expected_identity.wheelhouse_sha256,
        f"wheelhouse digest {wheelhouse_sha} does not match reviewed "
        f"{expected_identity.wheelhouse_sha256}",
    )

    runtime_record = ledger.get("runtime_archive")
    wheelhouse_record = ledger.get("wheelhouse_archive")
    _require(isinstance(runtime_record, Mapping), "ledger runtime_archive is missing")
    _require(
        isinstance(wheelhouse_record, Mapping), "ledger wheelhouse_archive is missing"
    )
    _require(runtime_record.get("file") == "runtime.tar.gz", "wrong runtime filename")
    _require(
        wheelhouse_record.get("file") == "wheelhouse.zip", "wrong wheelhouse filename"
    )
    _require(
        runtime_record.get("sha256") == runtime_sha,
        "ledger runtime archive digest does not match its bytes",
    )
    _require(
        wheelhouse_record.get("sha256") == wheelhouse_sha,
        "ledger wheelhouse archive digest does not match its bytes",
    )
    _require(
        runtime_record.get("bytes") == runtime_path.stat().st_size,
        "ledger runtime archive byte count does not match its bytes",
    )
    _require(
        wheelhouse_record.get("bytes") == wheelhouse_path.stat().st_size,
        "ledger wheelhouse archive byte count does not match its bytes",
    )

    declared_hashes = _declared_sha256_values(ledger)
    expected_declared_paths = set(_DECLARED_MEMBER_BY_LEDGER_PATH) | _ARCHIVE_LEDGER_PATHS
    _require(
        set(declared_hashes) == expected_declared_paths,
        "ledger sha256 field set drifted: missing="
        f"{sorted(expected_declared_paths - set(declared_hashes))}, extra="
        f"{sorted(set(declared_hashes) - expected_declared_paths)}",
    )

    member_hashes: dict[str, str] = {}
    with tarfile.open(runtime_path, mode="r:gz") as archive:
        members = archive.getmembers()
        members_by_name: dict[str, list[tarfile.TarInfo]] = {}
        for member in members:
            _safe_archive_name(member.name, "runtime archive")
            members_by_name.setdefault(member.name, []).append(member)

        for dotted, member_name in _DECLARED_MEMBER_BY_LEDGER_PATH.items():
            data = _read_unique_tar_member(archive, members_by_name, member_name)
            actual = sha256_bytes(data)
            expected = _require_sha256(
                _mapping_value(ledger, dotted), f"ledger {dotted}"
            )
            _require(
                actual == expected,
                f"{member_name} digest {actual} does not match ledger {expected}",
            )
            member_hashes[member_name.removeprefix("runtime/")] = actual

        python_files = sum(
            member.isfile() and member.name.endswith(".py") for member in members
        )
        model_files = sum(
            member.isfile()
            and pathlib.PurePosixPath(member.name).parent
            == pathlib.PurePosixPath("runtime/models")
            for member in members
        )
        weight_files = sum(
            member.isfile()
            and pathlib.PurePosixPath(member.name).parent
            == pathlib.PurePosixPath("runtime/weights")
            for member in members
        )

        fl_raw = _read_unique_tar_member(
            archive,
            members_by_name,
            "runtime/configs/fl_ev_regular_v4_selfplay.json",
        )

    try:
        fl_config = json.loads(fl_raw)
    except json.JSONDecodeError as error:
        raise PlanValidationError(f"FL config is not valid JSON: {error}") from error
    fl_ledger = ledger.get("fl_ev")
    _require(isinstance(fl_ledger, Mapping), "ledger fl_ev object is missing")
    _require(
        fl_ledger.get("config") == "configs/fl_ev_regular_v4_selfplay.json",
        "ledger does not select the v4 self-play FL config",
    )
    _require(fl_ledger.get("cards") == FL_EV_CARDS, "ledger FL card count is not 14")
    _require(fl_ledger.get("value") == FL_EV_VALUE, "ledger FL EV is not 9.6")
    _require(fl_config.get("rule_set") == "regular", "FL config is not regular OFC")
    _require(fl_config.get("include_jokers") is False, "FL config enables jokers")
    _require(fl_config.get("deck_cards") == 52, "FL config deck is not 52 cards")
    _require(fl_config.get("fl_stay_cards") == 14, "FL stay is not 14 cards")
    entry = fl_config.get("fl_entry_cards")
    _require(
        isinstance(entry, Mapping)
        and {str(key): value for key, value in entry.items()}
        == {"qq": 14, "kk": 14, "aa": 14, "trips": 14},
        "FL entry contract is not the 14-card regular rule",
    )
    values = fl_config.get("fl_ev")
    _require(
        isinstance(values, Mapping) and values.get("14") == FL_EV_VALUE,
        "FL config does not set fl_ev[14] to 9.6",
    )

    for label, actual in (
        ("python_files", python_files),
        ("model_files", model_files),
        ("weight_files", weight_files),
    ):
        _require(ledger.get(label) == actual, f"ledger {label} count is not {actual}")

    try:
        with zipfile.ZipFile(wheelhouse_path, mode="r") as wheelhouse:
            infos = wheelhouse.infolist()
            for info in infos:
                _safe_archive_name(info.filename, "wheelhouse archive")
            names = [info.filename for info in infos]
            _require(len(names) == len(set(names)), "wheelhouse has duplicate names")
            wheel_files = sum(not info.is_dir() for info in infos)
    except zipfile.BadZipFile as error:
        raise PlanValidationError(f"wheelhouse is not a valid zip: {error}") from error
    _require(
        ledger.get("wheel_files") == wheel_files,
        f"ledger wheel_files count is not {wheel_files}",
    )

    return PackageAudit(
        package_dir=package_dir,
        ledger_sha256=ledger_sha,
        runtime_sha256=runtime_sha,
        runtime_bytes=runtime_path.stat().st_size,
        wheelhouse_sha256=wheelhouse_sha,
        wheelhouse_bytes=wheelhouse_path.stat().st_size,
        declared_hashes=dict(sorted(declared_hashes.items())),
        member_hashes=dict(sorted(member_hashes.items())),
        python_files=python_files,
        model_files=model_files,
        weight_files=weight_files,
        wheel_files=wheel_files,
    )


def make_shards(
    *, position_count: int = POSITION_COUNT, shard_count: int = SHARD_COUNT
) -> list[dict[str, Any]]:
    _require(position_count > 0, "position_count must be positive")
    _require(0 < shard_count <= position_count, "invalid shard_count")
    small, extra = divmod(position_count, shard_count)
    # Match the established label plans: the larger shards are at the end.
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


def _package_pin(audit: PackageAudit, relative: str) -> str:
    try:
        return audit.member_hashes[relative]
    except KeyError as error:
        raise PlanValidationError(f"package audit did not verify {relative}") from error


def _base_plan(audit: PackageAudit, *, seat: str) -> dict[str, Any]:
    _require(seat in ("first", "second"), f"unsupported seat {seat!r}")
    job_id = FIRST_JOB_ID if seat == "first" else SECOND_JOB_ID
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "job_id": job_id,
        "street": "T2",
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
        "fl_ev_cards": FL_EV_CARDS,
        "fl_ev_value": FL_EV_VALUE,
        "shards": make_shards(),
    }
    if seat == "first":
        plan.update(
            {
                "t2_second_model": "weights/t2_model_v1.bin",
                "t2_second_model_sha256": _package_pin(
                    audit, "weights/t2_model_v1.bin"
                ),
            }
        )

    plan["provenance"] = {
        "schema": PROVENANCE_SCHEMA,
        "generation": "m7v5",
        "package_audit": audit.provenance(),
        "purchase": {
            "position_count_per_seat": POSITION_COUNT,
            "teacher_particles": SAMPLES,
            "reason": (
                "Owner selected 2048-particle T2 labels; this run buys label "
                "precision and does not claim that 2048 is a mathematical "
                "exact solution."
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
                "[947000000,947025000) is allocated as one fresh whole block; "
                "the two seats share it intentionally to pair their behavior "
                "roots."
            ),
        },
        "continuation": {
            "t3_first": "v7 / weights/t3first_model_v2.bin",
            "t3_second": "v7 / weights/t3_model_v3.bin",
            "t4": "v6 / weights/t4_model_v6.bin",
            "t2_second_reply": (
                "generation-1 / weights/t2_model_v1.bin"
                if seat == "first"
                else "not ahead of a T2 second-seat root"
            ),
        },
        "fantasyland": {
            "cards": FL_EV_CARDS,
            "value": FL_EV_VALUE,
            "config": "configs/fl_ev_regular_v4_selfplay.json",
            "config_sha256": _package_pin(
                audit, "configs/fl_ev_regular_v4_selfplay.json"
            ),
        },
        "sharding": {
            "shards": SHARD_COUNT,
            "concurrency_is_not_authorized_by_this_plan": True,
            "recommended_max_live_c4_standard_8": 58,
            "intended_workers_per_vm": 6,
            "waves_at_recommended_max_live": 3,
            "sizing_reason": (
                "The 116-shard first-seat cloud-tail estimate had p95 about "
                "6.8 hours at six workers, beyond the six-hour watchdog. "
                "174 shards reduce each wave to 143 or 144 roots while the "
                "58-VM live cap remains unchanged."
            ),
        },
    }
    return plan


def validate_plan_pair(plans: Mapping[str, Mapping[str, Any]]) -> None:
    _require(set(plans) == {"first", "second"}, "plan pair must contain both seats")
    first = plans["first"]
    second = plans["second"]
    for seat, plan in plans.items():
        _require(plan.get("schema") == PLAN_SCHEMA, f"{seat} plan schema drifted")
        _require(plan.get("street") == "T2", f"{seat} plan is not T2")
        _require(plan.get("seat") == seat, f"{seat} plan seat drifted")
        _require(plan.get("samples") == SAMPLES, f"{seat} plan is not 2048p")
        _require(
            plan.get("seeds_per_position") == SEEDS_PER_POSITION,
            f"{seat} plan does not have exactly one label run",
        )
        shards = plan.get("shards")
        _require(isinstance(shards, list), f"{seat} shards are missing")
        _require(
            len(shards) == SHARD_COUNT,
            f"{seat} does not have {SHARD_COUNT} shards",
        )
        expected_start = 0
        shard_ids: set[str] = set()
        for shard in shards:
            _require(isinstance(shard, Mapping), f"{seat} shard is not an object")
            shard_id = shard.get("shard_id")
            _require(isinstance(shard_id, str), f"{seat} shard id is not text")
            _require(shard_id not in shard_ids, f"{seat} duplicate shard {shard_id}")
            shard_ids.add(shard_id)
            _require(
                shard.get("start") == expected_start,
                f"{seat} shard {shard_id} is not contiguous",
            )
            count = shard.get("count")
            _require(isinstance(count, int) and count > 0, f"{seat} bad shard count")
            expected_start += count
        _require(expected_start == POSITION_COUNT, f"{seat} does not cover 25,000")
        _require("probe" not in plan, f"{seat} plan carries a probe block")
        _require(
            plan.get("fl_ev_cards") == FL_EV_CARDS
            and plan.get("fl_ev_value") == FL_EV_VALUE,
            f"{seat} plan FL contract drifted",
        )

    paired_fields = (
        "hand_seed_base",
        "behavior_seed_offset",
        "eval_seed_base",
        "samples",
        "seeds_per_position",
        "engine_library_sha256",
        "feature_encoder_library_sha256",
        "t3_first_model_sha256",
        "t3_second_model_sha256",
        "t4_model_sha256",
        "fl_ev_cards",
        "fl_ev_value",
        "shards",
    )
    for field in paired_fields:
        _require(first.get(field) == second.get(field), f"paired field {field} drifted")
    _require(
        "t2_second_model" in first and "t2_second_model_sha256" in first,
        "T2 first plan is missing the second-seat reply model",
    )
    _require(
        "t2_second_model" not in second and "t2_second_model_sha256" not in second,
        "T2 second plan pins an unreachable T2 reply model",
    )
    _require(first.get("job_id") != second.get("job_id"), "plan job ids collide")


def build_plan_pair(
    package_dir: pathlib.Path,
    *,
    expected_identity: ExpectedPackageIdentity = DEFAULT_PACKAGE_IDENTITY,
) -> tuple[dict[str, dict[str, Any]], PackageAudit]:
    audit = audit_package(package_dir, expected_identity=expected_identity)
    plans = {
        "first": _base_plan(audit, seat="first"),
        "second": _base_plan(audit, seat="second"),
    }
    validate_plan_pair(plans)
    return plans, audit


def _render_plan_set(
    plans: Mapping[str, Mapping[str, Any]], audit: PackageAudit
) -> tuple[dict[str, bytes], dict[str, Any]]:
    rendered = {
        FIRST_FILENAME: rendered_json_bytes(plans["first"]),
        SECOND_FILENAME: rendered_json_bytes(plans["second"]),
    }
    manifest = {
        "schema": PLAN_SET_SCHEMA,
        "generator": "ofc_regular.hu_m7_t2_2048_plan_v1",
        "package_audit": audit.provenance(),
        "plans": {
            "first": {
                "filename": FIRST_FILENAME,
                "job_id": FIRST_JOB_ID,
                "sha256": sha256_bytes(rendered[FIRST_FILENAME]),
            },
            "second": {
                "filename": SECOND_FILENAME,
                "job_id": SECOND_JOB_ID,
                "sha256": sha256_bytes(rendered[SECOND_FILENAME]),
            },
        },
        "paired_contract": {
            "positions_per_seat": POSITION_COUNT,
            "samples": SAMPLES,
            "shards_per_seat": SHARD_COUNT,
            "hand_seed_interval": [HAND_SEED_BASE, HAND_SEED_BASE + POSITION_COUNT],
            "eval_seed_base": EVAL_SEED_BASE,
            "first_only_t2_second_pin": True,
            "current_profile_changed": False,
        },
    }
    rendered[MANIFEST_FILENAME] = rendered_json_bytes(manifest)
    return rendered, manifest


def write_plan_pair_once(
    package_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    expected_identity: ExpectedPackageIdentity = DEFAULT_PACKAGE_IDENTITY,
) -> dict[str, Any]:
    plans, audit = build_plan_pair(package_dir, expected_identity=expected_identity)
    rendered, manifest = _render_plan_set(plans, audit)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: output_dir / name for name in rendered}
    collisions = sorted(str(path) for path in targets.values() if path.exists())
    _require(not collisions, f"write-once output already exists: {collisions}")

    # Every target is preflighted before the first byte is written.  ``xb`` is
    # still used so a concurrent writer cannot turn that check into overwrite.
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


def _audit_summary(audit: PackageAudit) -> dict[str, Any]:
    return {
        "schema": "hu_m7_t2_2048_plan_audit_summary_v1",
        "package_audit": audit.provenance(),
        "planned": {
            "positions_per_seat": POSITION_COUNT,
            "samples": SAMPLES,
            "shards_per_seat": SHARD_COUNT,
            "hand_seed_base": HAND_SEED_BASE,
            "eval_seed_base": EVAL_SEED_BASE,
        },
        "writes_performed": False,
    }


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
        _, audit = build_plan_pair(package_dir)
        print(json.dumps(_audit_summary(audit), indent=2, sort_keys=True))
        return 0

    if not args.out_dir:
        raise SystemExit("write mode requires --out-dir")
    manifest = write_plan_pair_once(package_dir, pathlib.Path(args.out_dir))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PACKAGE_IDENTITY",
    "EVAL_SEED_BASE",
    "ExpectedPackageIdentity",
    "FIRST_FILENAME",
    "FL_EV_VALUE",
    "HAND_SEED_BASE",
    "MANIFEST_FILENAME",
    "POSITION_COUNT",
    "PackageAudit",
    "PlanValidationError",
    "SAMPLES",
    "SECOND_FILENAME",
    "SHARD_COUNT",
    "audit_package",
    "build_plan_pair",
    "make_shards",
    "validate_plan_pair",
    "write_plan_pair_once",
]
