"""Two-phase, local-first preparation for a fresh full-100 wave run.

Phase A creates only a fresh wave plan, a newly bound immutable outer package,
and its content-stage plan.  It cannot create a ledger or resume plan.

Phase B accepts a fresh, exact, GET-only absence receipt covering *every*
planned a00/a01 instance and same-named boot disk across all three waves.  Only
then may it derive the empty initial transition, attempt ledger, and wave-0
resume plan.

The frozen scientific payload and offline wheelhouse may be reused.  A prior
run's claim, controller state, random material, IAM plan/receipts, content-stage
handoff, execution identity, and content prefix are never inputs.  No function
changes an AI profile.  Only ``collect_all_owned_absence`` can contact GCP, and
it exposes GET alone with an environment-only bearer token.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.parse
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import hu_m31_t3_step6d_full100_spot_v1 as scientific
from . import hu_m31_t3_step6d_full100_wave_content_stage_v2 as content_v2
from . import hu_m31_t3_step6d_full100_wave_package_v2 as package_v2
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_gcp_adapter_v2 as gcp_v2
from . import hu_m31_t3_step6d_full100_wave_science_registry_v2 as science_registry
from .hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    HttpResponse,
    _stdlib_http_request,
)


PHASE_A_SCHEMA = "hu_m31_t3_step6d_full100_wave_run_prepare_phase_a_v2"
ALL_OWNED_ABSENCE_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_all_owned_initial_absence_v2"
)
PHASE_B_SCHEMA = "hu_m31_t3_step6d_full100_wave_run_prepare_phase_b_v2"

WAVE_PLAN_NAME = "wave_plan.json"
OUTER_PACKAGE_NAME = "outer_package"
CONTENT_STAGE_PLAN_NAME = "content_stage_plan.json"
PHASE_A_RECEIPT_NAME = "phase_a_receipt.json"
ABSENCE_RECEIPT_NAME = "all_owned_absence_receipt.json"
INITIAL_TRANSITION_NAME = "initial_transition.json"
ATTEMPT_LEDGER_NAME = "attempt_ledger_v0.json"
RESUME_PLAN_NAME = "resume_plan_wave0.json"
PHASE_B_RECEIPT_NAME = "phase_b_receipt.json"

PROJECT = gcp_v2.PROJECT
ZONE = gcp_v2.ZONE
BUCKET = gcp_v2.BUCKET
TOKEN_ENV = gcp_v2.TOKEN_ENV
IDENTITY_SALT_ENV = "OFC_FULL100_IDENTITY_SALT"
MAX_ABSENCE_AGE_SECONDS = 300
EXPECTED_STARTUP_SHA256 = (
    "204a12c40b56dda643b7228e1687818a618bf1cb4a1f50359187b113c82a7d87"
)

RUN005_RUN_NAME = "regular-hu-m31-c02-f100wv2-20260722-005"
RUN005_EXECUTION_IDENTITY_SHA256 = (
    "d81c12713084aaffbdeaeedeea65c7d263d923fd3cb5b6009698d3fa79365de6"
)
RUN005_CONTENT_PAYLOAD_SHA256 = (
    "0c7bf6f1d7a77ae0999f71b3382982fd1d8420e0cde9b808798236c862f45f67"
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_PROJECT = re.compile(r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")
_ZONE = re.compile(r"^[a-z]+-[a-z0-9]+[0-9]-[a-z]$")

_PHASE_A_KEYS = frozenset(
    {
        "schema", "status", "run_name", "execution_identity_sha256",
        "wave_plan_sha256", "outer_manifest_sha256",
        "content_payload_sha256", "content_stage_plan_sha256",
        "scientific_source_sha256", "wheelhouse_archive_sha256",
        "wheelhouse_manifest_sha256", "startup_sha256", "entry_count",
        "new_run_identity_required", "identity_salt_persisted",
        "scientific_payload_reuse_allowed", "wheelhouse_reuse_allowed",
        "prior_claim_reused", "prior_control_state_reused",
        "prior_random_material_reused", "prior_iam_handoff_reused",
        "prior_content_handoff_reused", "initial_transition_created",
        "attempt_ledger_created", "resume_plan_created", "cloud_mutated",
        "cloud_launch_authorized", "current_profile_changed",
        "receipt_sha256",
    }
)
_ABSENCE_KEYS = frozenset(
    {
        "schema", "status", "run_name", "execution_identity_sha256",
        "wave_plan_sha256", "project", "zone", "observed_at_utc",
        "expires_at_utc", "provider_source", "credential_source",
        "planned_attempt_count", "instance_name_count",
        "boot_disk_name_count", "http_get_count", "rows",
        "all_instances_absent", "all_boot_disks_absent", "read_only",
        "cloud_mutated", "current_profile_changed", "receipt_sha256",
    }
)
_ABSENCE_ROW_KEYS = frozenset(
    {
        "wave_index", "job_id", "source_role", "attempt_id",
        "instance_name", "boot_disk_name", "instance_http_status",
        "boot_disk_http_status", "instance_absent", "boot_disk_absent",
    }
)
_PHASE_B_KEYS = frozenset(
    {
        "schema", "status", "run_name", "execution_identity_sha256",
        "wave_plan_sha256", "phase_a_receipt_sha256",
        "absence_receipt_sha256", "initial_transition_sha256",
        "attempt_ledger_sha256", "resume_plan_sha256",
        "finalized_at_utc", "resume_wave_index", "selected_attempt_count",
        "all_planned_attempt_names_absent", "prior_claim_reused",
        "prior_control_state_reused", "prior_random_material_reused",
        "prior_iam_handoff_reused", "prior_content_handoff_reused",
        "cloud_mutated", "cloud_launch_authorized",
        "current_profile_changed", "receipt_sha256",
    }
)

HttpRequester = Callable[
    [str, str, Mapping[str, str], bytes | None, int], HttpResponse
]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def _validate_seal(
    value: Mapping[str, Any], *, keys: frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} fields changed")
    result = deepcopy(dict(value))
    digest = result.pop("receipt_sha256", None)
    if not isinstance(digest, str) or digest != canonical_sha256(result):
        raise ValueError(f"{label} digest changed")
    result["receipt_sha256"] = digest
    return result


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical UTC seconds")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{label} must be UTC")
    return parsed


def _render_utc(value: datetime) -> str:
    if value.tzinfo != timezone.utc or value.microsecond != 0:
        raise ValueError("timestamp must be UTC whole seconds")
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_directory(path: str | Path, label: str) -> Path:
    return package_v2._safe_existing_directory(path, label)


def _safe_file(path: str | Path, label: str) -> Path:
    return package_v2._safe_existing_file(path, label)


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    target = _safe_file(path, label)
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    package_v2._write_once(path, canonical_bytes(value))


def _assert_exact_root(root: Path, expected: set[str], label: str) -> None:
    safe = _safe_directory(root, label)
    observed = {child.name for child in safe.iterdir()}
    if observed != expected:
        raise ValueError(f"{label} has extra or missing entries")
    for child in safe.iterdir():
        if package_v2._is_link_or_junction(child):
            raise ValueError(f"{label} contains a symlink or junction")


def _phase_a_core(
    *, plan: Mapping[str, Any], outer: Mapping[str, Any],
    stage_plan: Mapping[str, Any], wheelhouse_archive_sha256: str,
    wheelhouse_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": PHASE_A_SCHEMA,
        "status": (
            "fresh_startup_canary_plan_outer_package_and_stage_plan_only"
            if plan["scope"] == wave_v2.STARTUP_CANARY_SCOPE
            else "fresh_plan_outer_package_and_stage_plan_only"
        ),
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "outer_manifest_sha256": outer["manifest_sha256"],
        "content_payload_sha256": outer["content_payload_sha256"],
        "content_stage_plan_sha256": stage_plan["plan_sha256"],
        "scientific_source_sha256": outer["scientific_lineage"]["source_sha256"],
        "wheelhouse_archive_sha256": wheelhouse_archive_sha256,
        "wheelhouse_manifest_sha256": wheelhouse_manifest_sha256,
        "startup_sha256": outer["expected_startup_sha256"],
        "entry_count": outer["entry_count"],
        "new_run_identity_required": True,
        "identity_salt_persisted": False,
        "scientific_payload_reuse_allowed": True,
        "wheelhouse_reuse_allowed": True,
        "prior_claim_reused": False,
        "prior_control_state_reused": False,
        "prior_random_material_reused": False,
        "prior_iam_handoff_reused": False,
        "prior_content_handoff_reused": False,
        "initial_transition_created": False,
        "attempt_ledger_created": False,
        "resume_plan_created": False,
        "cloud_mutated": False,
        "cloud_launch_authorized": False,
        "current_profile_changed": False,
    }


def prepare_phase_a(
    *, output_dir: str | Path, run_name: str, identity_salt: str,
    scientific_package_dir: str | Path, startup_script: str | Path,
    wheelhouse_archive: str | Path, wheelhouse_manifest: str | Path,
    image_digest: str, bucket: str = BUCKET,
    expected_startup_sha256: str | None = None,
    full100_plan_path: str | Path = wave_v2.DEFAULT_FULL100_PLAN_PATH,
    execution_scope: str | None = None,
) -> dict[str, Any]:
    """Create an immutable Phase-A directory without ledger/resume state."""

    destination = Path(os.path.abspath(os.fspath(output_dir)))
    if destination.exists() or package_v2._is_link_or_junction(destination):
        raise FileExistsError("Phase-A destination is immutable")
    frozen = wave_v2._read_frozen_plan(full100_plan_path)
    descriptor = science_registry.descriptor_for_plan(frozen)
    science = descriptor.validate_package(scientific_package_dir)
    source_sha = _require_sha(science.get("source_sha256"), "scientific source")
    selected_scope = (
        descriptor.execution_scope
        if execution_scope is None
        else execution_scope
    )
    plan = wave_v2.build_wave_plan(
        run_name=run_name,
        identity_salt=identity_salt,
        package_sha256=source_sha,
        image_digest=image_digest,
        full100_plan=frozen,
        execution_scope=selected_scope,
    )
    startup_sha256 = science_registry.resolve_startup_sha256(
        plan, expected_startup_sha256
    )
    if selected_scope == wave_v2.STARTUP_CANARY_SCOPE:
        wave_v2.validate_startup_canary_plan(plan)
    if (
        plan["run_name"] == RUN005_RUN_NAME
        or plan["execution_identity_sha256"] == RUN005_EXECUTION_IDENTITY_SHA256
    ):
        raise ValueError("run005 execution identity reuse is forbidden")
    wheel = _safe_file(wheelhouse_archive, "offline wheelhouse archive")
    wheel_manifest_path = _safe_file(
        wheelhouse_manifest, "offline wheelhouse manifest"
    )
    parent = destination.parent
    package_v2._ensure_directory(parent)
    stage = parent / f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    published = False
    try:
        stage.mkdir()
        package_v2._fsync_directory(stage)
        package_v2._fsync_directory(parent)
        _write_once(stage / WAVE_PLAN_NAME, plan)
        outer = package_v2.materialize_outer_package(
            output_dir=stage / OUTER_PACKAGE_NAME,
            scientific_package_dir=scientific_package_dir,
            startup_script=startup_script,
            wheelhouse_archive=wheel,
            wheelhouse_manifest=wheel_manifest_path,
            expected_startup_sha256=startup_sha256,
            wave_plan=plan,
        )
        if outer["content_payload_sha256"] == RUN005_CONTENT_PAYLOAD_SHA256:
            raise ValueError("run005 content handoff reuse is forbidden")
        stage_plan = content_v2.build_content_stage_plan(
            package_dir=stage / OUTER_PACKAGE_NAME,
            wave_plan=plan,
            expected_startup_sha256=startup_sha256,
            bucket=bucket,
        )
        _write_once(stage / CONTENT_STAGE_PLAN_NAME, stage_plan)
        receipt = _seal(
            _phase_a_core(
                plan=plan,
                outer=outer,
                stage_plan=stage_plan,
                wheelhouse_archive_sha256=sha256_file(wheel),
                wheelhouse_manifest_sha256=sha256_file(wheel_manifest_path),
            )
        )
        _write_once(stage / PHASE_A_RECEIPT_NAME, receipt)
        validate_phase_a(
            stage, expected_startup_sha256=startup_sha256
        )
        package_v2._publish_no_replace(stage, destination)
        published = True
        package_v2._fsync_directory(destination)
        package_v2._fsync_directory(parent)
        return validate_phase_a(
            destination, expected_startup_sha256=startup_sha256
        )["receipt"]
    except BaseException:
        if stage.exists() and not package_v2._is_link_or_junction(stage):
            shutil.rmtree(stage)
            package_v2._fsync_directory(parent)
        if published and destination.exists() and not package_v2._is_link_or_junction(destination):
            shutil.rmtree(destination)
            package_v2._fsync_directory(parent)
        raise


def validate_phase_a(
    phase_a_dir: str | Path, *,
    expected_startup_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    root = _safe_directory(phase_a_dir, "Phase-A root")
    _assert_exact_root(
        root,
        {
            WAVE_PLAN_NAME, OUTER_PACKAGE_NAME, CONTENT_STAGE_PLAN_NAME,
            PHASE_A_RECEIPT_NAME,
        },
        "Phase-A root",
    )
    plan = wave_v2.validate_wave_plan(
        _read_canonical(root / WAVE_PLAN_NAME, "Phase-A wave plan")
    )
    startup_sha256 = science_registry.resolve_startup_sha256(
        plan, expected_startup_sha256
    )
    outer = package_v2.validate_outer_package(
        root / OUTER_PACKAGE_NAME,
        plan,
        expected_startup_sha256=startup_sha256,
    )
    stage_plan = content_v2.validate_content_stage_plan(
        outer,
        _read_canonical(
            root / CONTENT_STAGE_PLAN_NAME, "Phase-A content stage plan"
        ),
    )
    receipt = _validate_seal(
        _read_canonical(root / PHASE_A_RECEIPT_NAME, "Phase-A receipt"),
        keys=_PHASE_A_KEYS,
        label="Phase-A receipt",
    )
    expected = _phase_a_core(
        plan=plan,
        outer=outer,
        stage_plan=stage_plan,
        wheelhouse_archive_sha256=outer["wheelhouse_binding"]["archive_sha256"],
        wheelhouse_manifest_sha256=outer["wheelhouse_binding"]["manifest_sha256"],
    )
    if receipt != {**expected, "receipt_sha256": canonical_sha256(expected)}:
        raise ValueError("Phase-A receipt binding changed")
    if (
        plan["run_name"] == RUN005_RUN_NAME
        or plan["execution_identity_sha256"] == RUN005_EXECUTION_IDENTITY_SHA256
        or outer["content_payload_sha256"] == RUN005_CONTENT_PAYLOAD_SHA256
        or outer["scientific_lineage"]["launcher_reuse_forbidden"] is not True
        or stage_plan["cloud_launch_authorized"] is not False
    ):
        raise ValueError("Phase-A reused a forbidden prior-run handoff")
    return {
        "plan": plan, "outer_manifest": outer,
        "content_stage_plan": stage_plan, "receipt": receipt,
    }


def _planned_attempt_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    validated = wave_v2.validate_wave_plan(plan)
    rows: list[dict[str, Any]] = []
    for wave in validated["waves"]:
        for pair in wave["candidate_reference_pairs"]:
            for role in wave_v2.SOURCE_ROLES:
                job_id = pair[f"{role}_job_id"]
                mapping = pair[f"{role}_attempt_instance_ids"]
                for attempt_id in wave_v2.ATTEMPT_IDS:
                    name = mapping[attempt_id]
                    rows.append(
                        {
                            "wave_index": wave["wave_index"],
                            "job_id": job_id,
                            "source_role": role,
                            "attempt_id": attempt_id,
                            "instance_name": name,
                            "boot_disk_name": name,
                        }
                    )
    if len(rows) != 40 or len({row["instance_name"] for row in rows}) != 40:
        raise ValueError("planned all-attempt owned-name coverage changed")
    return rows


def _absence_core(
    *, plan: Mapping[str, Any], project: str, zone: str,
    observed_at_utc: str, rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observed = _parse_utc(observed_at_utc, "absence observation time")
    return {
        "schema": ALL_OWNED_ABSENCE_SCHEMA,
        "status": "all_planned_a00_a01_instance_and_disk_names_absent",
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "project": project,
        "zone": zone,
        "observed_at_utc": observed_at_utc,
        "expires_at_utc": _render_utc(
            observed + timedelta(seconds=MAX_ABSENCE_AGE_SECONDS)
        ),
        "provider_source": "compute.googleapis.com/v1 exact-name GET-only",
        "credential_source": f"environment-only:{TOKEN_ENV}",
        "planned_attempt_count": len(rows),
        "instance_name_count": len(rows),
        "boot_disk_name_count": len(rows),
        "http_get_count": 2 * len(rows),
        "rows": deepcopy(list(rows)),
        "all_instances_absent": True,
        "all_boot_disks_absent": True,
        "read_only": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }


def validate_all_owned_absence_receipt(
    wave_plan: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    plan = wave_v2.validate_wave_plan(wave_plan)
    receipt = _validate_seal(
        value, keys=_ABSENCE_KEYS, label="all-owned absence receipt"
    )
    expected_rows = _planned_attempt_rows(plan)
    rows = receipt.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_rows):
        raise ValueError("all-owned absence rows do not exactly cover the plan")
    clean: list[dict[str, Any]] = []
    for expected, raw in zip(expected_rows, rows, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != _ABSENCE_ROW_KEYS:
            raise ValueError("all-owned absence row fields changed")
        row = deepcopy(dict(raw))
        if (
            {key: row[key] for key in expected} != expected
            or row["instance_http_status"] != 404
            or row["boot_disk_http_status"] != 404
            or row["instance_absent"] is not True
            or row["boot_disk_absent"] is not True
        ):
            raise ValueError("planned instance or disk absence is incomplete")
        clean.append(row)
    observed = _parse_utc(receipt.get("observed_at_utc"), "absence observation")
    expires = _parse_utc(receipt.get("expires_at_utc"), "absence expiry")
    if (
        receipt.get("schema") != ALL_OWNED_ABSENCE_SCHEMA
        or receipt.get("status")
        != "all_planned_a00_a01_instance_and_disk_names_absent"
        or receipt.get("run_name") != plan["run_name"]
        or receipt.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or receipt.get("wave_plan_sha256") != plan["schedule_sha256"]
        or receipt.get("project") != PROJECT
        or receipt.get("zone") != ZONE
        or receipt.get("provider_source")
        != "compute.googleapis.com/v1 exact-name GET-only"
        or receipt.get("credential_source") != f"environment-only:{TOKEN_ENV}"
        or expires - observed != timedelta(seconds=MAX_ABSENCE_AGE_SECONDS)
        or receipt.get("planned_attempt_count") != 40
        or receipt.get("instance_name_count") != 40
        or receipt.get("boot_disk_name_count") != 40
        or receipt.get("http_get_count") != 80
        or receipt.get("all_instances_absent") is not True
        or receipt.get("all_boot_disks_absent") is not True
        or receipt.get("read_only") is not True
        or receipt.get("cloud_mutated") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("all-owned absence receipt contract changed")
    receipt["rows"] = clean
    return receipt


def collect_all_owned_absence(
    *, wave_plan: Mapping[str, Any], project: str = PROJECT,
    zone: str = ZONE, requester: HttpRequester | None = None,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """GET every planned a00/a01 instance and boot disk; never list or mutate."""

    plan = wave_v2.validate_wave_plan(wave_plan)
    if project != PROJECT or zone != ZONE:
        raise ValueError("all-owned absence target changed")
    send = _stdlib_http_request if requester is None else requester
    now = (
        (lambda: datetime.now(timezone.utc).replace(microsecond=0))
        if clock is None else clock
    )
    rows: list[dict[str, Any]] = []
    collisions: list[str] = []
    for expected in _planned_attempt_rows(plan):
        statuses: list[int] = []
        for collection, name in (
            ("instances", expected["instance_name"]),
            ("disks", expected["boot_disk_name"]),
        ):
            token = os.environ.get(TOKEN_ENV)
            if (
                not isinstance(token, str) or len(token) < 20
                or any(character.isspace() for character in token)
            ):
                raise PermissionError(
                    f"Bearer token must be supplied only through {TOKEN_ENV}"
                )
            url = (
                "https://compute.googleapis.com/compute/v1/projects/"
                f"{urllib.parse.quote(project, safe='')}/zones/"
                f"{urllib.parse.quote(zone, safe='')}/{collection}/"
                f"{urllib.parse.quote(name, safe='')}"
            )
            try:
                response = send(
                    "GET", url,
                    {"Authorization": f"Bearer {token}", "Accept": "application/json"},
                    None, 60,
                )
            except Exception as exc:
                raise RuntimeError(
                    "all-owned absence GET transport failed without provider body"
                ) from exc
            if not isinstance(response, HttpResponse) or response.status not in {200, 404}:
                status = response.status if isinstance(response, HttpResponse) else "invalid"
                raise RuntimeError(
                    f"all-owned absence GET failed with status {status}"
                )
            statuses.append(response.status)
            if response.status == 200:
                collisions.append(f"{collection}:{name}")
        rows.append(
            {
                **expected,
                "instance_http_status": statuses[0],
                "boot_disk_http_status": statuses[1],
                "instance_absent": statuses[0] == 404,
                "boot_disk_absent": statuses[1] == 404,
            }
        )
    if collisions:
        raise FileExistsError(
            "planned GCE instance/disk names are not all absent: "
            + ",".join(collisions)
        )
    observed = now()
    if observed.tzinfo != timezone.utc:
        raise ValueError("absence collector clock must return UTC")
    observed = observed.replace(microsecond=0)
    receipt = _seal(
        _absence_core(
            plan=plan, project=project, zone=zone,
            observed_at_utc=_render_utc(observed), rows=rows,
        )
    )
    return validate_all_owned_absence_receipt(plan, receipt)


def _phase_b_core(
    *, phase_a: Mapping[str, Any], absence: Mapping[str, Any],
    transition: Mapping[str, Any], ledger: Mapping[str, Any],
    resume: Mapping[str, Any], finalized_at_utc: str,
) -> dict[str, Any]:
    plan = phase_a["plan"]
    return {
        "schema": PHASE_B_SCHEMA,
        "status": (
            "fresh_all_owned_absence_bound_single_attempt_startup_canary"
            if plan["scope"] == wave_v2.STARTUP_CANARY_SCOPE
            else "fresh_all_owned_absence_bound_initial_ledger_resume"
        ),
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "phase_a_receipt_sha256": phase_a["receipt"]["receipt_sha256"],
        "absence_receipt_sha256": absence["receipt_sha256"],
        "initial_transition_sha256": transition["transition_digest"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "finalized_at_utc": finalized_at_utc,
        "resume_wave_index": resume["resume_wave_index"],
        "selected_attempt_count": len(resume["selected_attempts"]),
        "all_planned_attempt_names_absent": True,
        "prior_claim_reused": False,
        "prior_control_state_reused": False,
        "prior_random_material_reused": False,
        "prior_iam_handoff_reused": False,
        "prior_content_handoff_reused": False,
        "cloud_mutated": False,
        "cloud_launch_authorized": False,
        "current_profile_changed": False,
    }


def finalize_phase_b(
    *, phase_a_dir: str | Path, output_dir: str | Path,
    absence_receipt: Mapping[str, Any], current_time_utc: str,
    expected_startup_sha256: str | None = None,
) -> dict[str, Any]:
    """Create initial ledger/resume only from fresh all-owned absence proof."""

    phase_a = validate_phase_a(
        phase_a_dir, expected_startup_sha256=expected_startup_sha256
    )
    plan = phase_a["plan"]
    absence = validate_all_owned_absence_receipt(plan, absence_receipt)
    current = _parse_utc(current_time_utc, "Phase-B current time")
    observed = _parse_utc(absence["observed_at_utc"], "absence observation")
    expires = _parse_utc(absence["expires_at_utc"], "absence expiry")
    if not observed <= current <= expires:
        raise PermissionError("all-owned absence receipt is stale or future-dated")
    transition = wave_v2.build_observed_transition(
        plan,
        project_id=absence["project"],
        zone=absence["zone"],
        observed_at_utc=absence["observed_at_utc"],
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(plan),
        owned_vms=(), done_objects=(), acceptance_records=(),
        readback_source="gcloud_readback",
    )
    ledger = wave_v2.build_attempt_ledger(plan, transitions=[transition])
    resume = wave_v2.build_resume_plan(plan, attempt_ledger=ledger)
    expected_selected_count = (
        1 if plan["scope"] == wave_v2.STARTUP_CANARY_SCOPE else 8
    )
    if (
        resume["resume_wave_index"] != 0
        or len(resume["selected_attempts"]) != expected_selected_count
    ):
        raise ValueError("fresh initial resume does not match its execution scope")
    if plan["scope"] == wave_v2.STARTUP_CANARY_SCOPE:
        wave_v2.validate_startup_canary_plan(plan)
        if resume["selected_attempts"] != [
            {
                "job_id": wave_v2.STARTUP_CANARY_JOB_ID,
                "source_role": wave_v2.STARTUP_CANARY_SOURCE_ROLE,
                "attempt_id": wave_v2.STARTUP_CANARY_ATTEMPT_ID,
                "instance_id": plan["waves"][0]["candidate_reference_pairs"][0][
                    "candidate_attempt_instance_ids"
                ][wave_v2.STARTUP_CANARY_ATTEMPT_ID],
                "artifact_prefix": plan["artifact_contract"][
                    "attempt_path_template"
                ].format(
                    job_id=wave_v2.STARTUP_CANARY_JOB_ID,
                    attempt_id=wave_v2.STARTUP_CANARY_ATTEMPT_ID,
                ),
            }
        ]:
            raise ValueError("startup canary selection escaped candidate-shard-00/a00")
    destination = Path(os.path.abspath(os.fspath(output_dir)))
    if destination.exists() or package_v2._is_link_or_junction(destination):
        raise FileExistsError("Phase-B destination is immutable")
    parent = destination.parent
    package_v2._ensure_directory(parent)
    stage = parent / f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    published = False
    try:
        stage.mkdir()
        package_v2._fsync_directory(stage)
        package_v2._fsync_directory(parent)
        _write_once(stage / ABSENCE_RECEIPT_NAME, absence)
        _write_once(stage / INITIAL_TRANSITION_NAME, transition)
        _write_once(stage / ATTEMPT_LEDGER_NAME, ledger)
        _write_once(stage / RESUME_PLAN_NAME, resume)
        core = _phase_b_core(
            phase_a=phase_a, absence=absence, transition=transition,
            ledger=ledger, resume=resume, finalized_at_utc=current_time_utc,
        )
        receipt = _seal(core)
        _write_once(stage / PHASE_B_RECEIPT_NAME, receipt)
        validate_phase_b(
            phase_a_dir=phase_a_dir, phase_b_dir=stage,
            expected_startup_sha256=expected_startup_sha256,
        )
        package_v2._publish_no_replace(stage, destination)
        published = True
        package_v2._fsync_directory(destination)
        package_v2._fsync_directory(parent)
        return validate_phase_b(
            phase_a_dir=phase_a_dir, phase_b_dir=destination,
            expected_startup_sha256=expected_startup_sha256,
        )["receipt"]
    except BaseException:
        if stage.exists() and not package_v2._is_link_or_junction(stage):
            shutil.rmtree(stage)
            package_v2._fsync_directory(parent)
        if published and destination.exists() and not package_v2._is_link_or_junction(destination):
            shutil.rmtree(destination)
            package_v2._fsync_directory(parent)
        raise


def validate_phase_b(
    *, phase_a_dir: str | Path, phase_b_dir: str | Path,
    expected_startup_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    phase_a = validate_phase_a(
        phase_a_dir, expected_startup_sha256=expected_startup_sha256
    )
    plan = phase_a["plan"]
    root = _safe_directory(phase_b_dir, "Phase-B root")
    _assert_exact_root(
        root,
        {
            ABSENCE_RECEIPT_NAME, INITIAL_TRANSITION_NAME,
            ATTEMPT_LEDGER_NAME, RESUME_PLAN_NAME, PHASE_B_RECEIPT_NAME,
        },
        "Phase-B root",
    )
    absence = validate_all_owned_absence_receipt(
        plan,
        _read_canonical(root / ABSENCE_RECEIPT_NAME, "Phase-B absence receipt"),
    )
    transition = wave_v2.validate_observed_transition(
        plan,
        _read_canonical(root / INITIAL_TRANSITION_NAME, "initial transition"),
    )
    ledger = wave_v2.validate_attempt_ledger(
        plan,
        _read_canonical(root / ATTEMPT_LEDGER_NAME, "initial attempt ledger"),
    )
    resume = wave_v2.validate_resume_plan(
        plan, ledger,
        _read_canonical(root / RESUME_PLAN_NAME, "wave-0 resume plan"),
    )
    receipt = _validate_seal(
        _read_canonical(root / PHASE_B_RECEIPT_NAME, "Phase-B receipt"),
        keys=_PHASE_B_KEYS, label="Phase-B receipt",
    )
    finalized = _parse_utc(receipt["finalized_at_utc"], "Phase-B finalization")
    if not (
        _parse_utc(absence["observed_at_utc"], "absence observation")
        <= finalized
        <= _parse_utc(absence["expires_at_utc"], "absence expiry")
    ):
        raise ValueError("Phase-B was not finalized inside the absence window")
    expected = _phase_b_core(
        phase_a=phase_a, absence=absence, transition=transition,
        ledger=ledger, resume=resume,
        finalized_at_utc=receipt["finalized_at_utc"],
    )
    if (
        transition != ledger["transitions"][0]
        or receipt != {**expected, "receipt_sha256": canonical_sha256(expected)}
    ):
        raise ValueError("Phase-B evidence binding changed")
    return {
        "absence_receipt": absence, "initial_transition": transition,
        "attempt_ledger": ledger, "resume_plan": resume, "receipt": receipt,
    }


def write_canonical_once(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(os.path.abspath(os.fspath(path)))
    _write_once(target, value)


def read_canonical_file(path: str | Path, label: str = "JSON input") -> dict[str, Any]:
    return _read_canonical(path, label)


__all__ = [
    "ALL_OWNED_ABSENCE_SCHEMA", "ATTEMPT_LEDGER_NAME",
    "EXPECTED_STARTUP_SHA256", "IDENTITY_SALT_ENV", "PHASE_A_SCHEMA",
    "PHASE_B_SCHEMA", "collect_all_owned_absence", "finalize_phase_b",
    "prepare_phase_a", "read_canonical_file", "validate_all_owned_absence_receipt",
    "validate_phase_a", "validate_phase_b", "write_canonical_once",
]
