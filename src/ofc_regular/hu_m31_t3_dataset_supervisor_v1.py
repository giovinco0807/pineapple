"""Single-wave-at-a-time production supervisor for M3.1 dataset v1.

The supervisor does not change the v1 scientific or cloud contracts.  It
adds an append-only operational boundary around the existing 8-VM controller:

fresh redacted OAuth TTL preflight -> execute once -> bounded poll -> exact
cleanup -> receive -> accept -> checkpoint.

Only one supervisor process may own a controller at a time.  Any exception
stops the run.  If launch fails after a partial create, exact selected
instances/disks and temporary IAM are reconciled and removed before the
exception is re-raised; no next wave is opened automatically.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from . import hu_m31_t3_dataset_gcp_controller_v1 as controller
from . import hu_m31_t3_dataset_gcp_provider_v1 as provider
from . import hu_m31_t3_dataset_gcp_transport_v1 as transport
from . import hu_m31_t3_dataset_source_binding_v1 as source_binding
from . import hu_m31_t3_step6d_fresh_quality_gcp_provider_v1 as quality_provider
from .hu_m31_t3_dataset_gcp_provider_v2 import GcpParallelDatasetRestAdapter


SUPERVISOR_CHECKPOINT_SCHEMA = "hu_m31_t3_dataset_supervisor_checkpoint_v1"
CHECKPOINT_INTENT_SCHEMA = "hu_m31_t3_dataset_supervisor_checkpoint_intent_v1"
OAUTH_PREFLIGHT_SCHEMA = "hu_m31_t3_dataset_supervisor_oauth_preflight_v1"
ABORT_CLEANUP_SCHEMA = "hu_m31_t3_dataset_supervisor_abort_cleanup_v1"
SOURCE_RECEIPT_SCHEMA = "hu_m31_t3_dataset_supervisor_source_receipt_v1"
COST_RESERVATION_SCHEMA = "hu_m31_t3_dataset_cost_reservation_v1"
COST_SETTLEMENT_SCHEMA = "hu_m31_t3_dataset_cost_settlement_v1"
COST_STATUS_SCHEMA = "hu_m31_t3_dataset_cost_status_v1"
MIN_OAUTH_TTL_SECONDS = 2700
MAX_POLL_ATTEMPTS = 720
DEFAULT_POLL_INTERVAL_SECONDS = 30
MIN_PHASE_OAUTH_TTL_SECONDS = 900
MAX_SUPERVISED_LIFECYCLES = transport.WAVE_COUNT * transport.MAX_ATTEMPTS_PER_SHARD
LOCK_FILENAME = "ACTIVE_SUPERVISOR.lock"

# Cost values use integer micro-USD throughout.  The $0.50 VM-hour guard is
# deliberately above the preregistered July 2026 c4-standard-16 Spot estimate
# ($0.45638), and the separate $25 reserve covers disks/GCS/control-plane
# overhead.  Before launch the operator must recheck that the live Spot price
# remains at or below the guard.
TOTAL_CLOUD_COST_CAP_MICRO_USD = 500_000_000
NON_COMPUTE_RESERVE_MICRO_USD = 25_000_000
COMPUTE_COST_CAP_MICRO_USD = (
    TOTAL_CLOUD_COST_CAP_MICRO_USD - NON_COMPUTE_RESERVE_MICRO_USD
)
SPOT_RATE_GUARD_MICRO_USD_PER_VM_HOUR = 500_000
CONFIRM_TOTAL_COST_CAP_USD = "500.000000"
CONFIRM_SPOT_RATE_GUARD_USD_PER_VM_HOUR = "0.500000"
MINIMUM_BILLABLE_SECONDS = 60

PINNED_SOURCE_HASHES = {
    "src/ofc_regular/ai_profiles.py": (
        "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
    ),
    "src/ofc_regular/hu_m31_t3_dataset_contract_v1.py": (
        "143ad36c841ddf99a58696a8d2b6ad7780e58bf9b9e6a0e19b61a141b45a45f1"
    ),
    "src/ofc_regular/hu_m31_t3_dataset_executor_v1.py": (
        "eb361097eef6b7ac72383a372914d1067888abe59ad3b863e15ef4fff71c3ba9"
    ),
    "src/ofc_regular/hu_m31_t3_dataset_portable_worker_v1.py": (
        "43b2ed4d6e91a9d3554c11d5f79e751531a2576aeca14aa89b63fc4cbc8fca3b"
    ),
    "src/ofc_regular/hu_m31_t3_dataset_gcp_transport_v1.py": (
        "613acacf76129619e425849127cba653b54c6b2028d94d4499eace541709f390"
    ),
    "src/ofc_regular/hu_m31_t3_dataset_gcp_provider_v1.py": (
        "ee12fdb412d77b39f32ae94e33518471a8ff2a125bb8d29ce1627c4b2aa46111"
    ),
    "src/ofc_regular/hu_m31_t3_dataset_gcp_controller_v1.py": (
        "64f6b0c32c2756e53ec1e41dde051fe0d4277162c3d37da5160f9e5d92a8b498"
    ),
    "src/ofc_regular/hu_m31_t3_dataset_source_binding_v1.py": (
        "06a66834a596a1ac26f2468aa1884060282ab1823d7fa6f00a609d182d64ae34"
    ),
}


class TtlCloud(Protocol):
    def get_oauth_token_ttl_seconds(self) -> int: ...


class _ExpiryBoundGcpCloud(GcpParallelDatasetRestAdapter):
    """REST adapter whose gcloud expiry stays memory-only."""

    def __init__(
        self,
        *,
        access_token: str,
        expiry_unix_seconds: int,
        now: Callable[[], int],
    ) -> None:
        super().__init__(access_token=access_token)
        self._expiry_unix_seconds = expiry_unix_seconds
        self._now = now

    def get_oauth_token_ttl_seconds(self) -> int:
        return max(0, self._expiry_unix_seconds - self._now())


class _GcloudCredentialLease:
    """Force refresh once per wave, then refresh phases only when needed."""

    def __init__(self, *, now: Callable[[], int] | None = None) -> None:
        self._now = now or (lambda: int(time.time()))
        self._cloud: _ExpiryBoundGcpCloud | None = None

    def get(self, *, force_refresh: bool) -> _ExpiryBoundGcpCloud:
        if (
            force_refresh
            or self._cloud is None
            or self._cloud.get_oauth_token_ttl_seconds() < MIN_PHASE_OAUTH_TTL_SECONDS
        ):
            self._cloud = _force_refreshed_cloud_from_gcloud(now=self._now)
        return self._cloud


def canonical_bytes(value: Any) -> bytes:
    return controller.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return controller.canonical_sha256(value)


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    copied = deepcopy(dict(value))
    copied.pop(field, None)
    return canonical_sha256(copied)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_receipt() -> dict[str, Any]:
    root = _repo_root()
    rows = []
    for relative, expected in sorted(PINNED_SOURCE_HASHES.items()):
        path = root / relative
        observed = (
            _file_sha256(path) if path.is_file() and not path.is_symlink() else None
        )
        rows.append(
            {
                "relative_path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "match": observed == expected,
            }
        )
    if not all(row["match"] for row in rows):
        raise PermissionError("M3.1 v1 supervisor source hash changed")
    core = {
        "schema": SOURCE_RECEIPT_SCHEMA,
        "status": "exact_v1_sources_and_current_hash_match",
        "files": rows,
        "file_count": len(rows),
        "all_match": True,
        "cost_contract": {
            "total_cloud_cost_cap_micro_usd": TOTAL_CLOUD_COST_CAP_MICRO_USD,
            "non_compute_reserve_micro_usd": NON_COMPUTE_RESERVE_MICRO_USD,
            "compute_cost_cap_micro_usd": COMPUTE_COST_CAP_MICRO_USD,
            "spot_rate_guard_micro_usd_per_vm_hour": (
                SPOT_RATE_GUARD_MICRO_USD_PER_VM_HOUR
            ),
            "live_spot_price_recheck_required_before_launch": True,
            "reservation_before_cloud_factory": True,
            "settlement_after_exact_cleanup": True,
        },
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _validate_checkpoint_intent(
    value: Mapping[str, Any],
    *,
    confirm_run_name: str,
    source_receipt_sha256: str,
) -> dict[str, Any]:
    intent = deepcopy(dict(value))
    required = {
        "schema",
        "status",
        "run_name",
        "wave_index",
        "request_sha256",
        "source_receipt_sha256",
        "cost_reservation_receipt_sha256",
        "cost_settlement_receipt_sha256",
        "oauth_preflight_receipt_sha256",
        "launch_receipt_sha256",
        "poll_receipt_sha256",
        "cleanup_incomplete",
        "lifecycle_receipt_sha256",
        "receive_receipt_sha256",
        "one_wave_only",
        "max_concurrent_wave_count",
        "cloud_mutated",
        "current_profile_changed",
        "intent_sha256",
    }
    if (
        set(intent) != required
        or intent.get("intent_sha256") != _self_digest(intent, "intent_sha256")
        or intent.get("schema") != CHECKPOINT_INTENT_SCHEMA
        or intent.get("status") != "wave_phases_complete_ready_to_accept"
        or intent.get("run_name") != confirm_run_name
        or intent.get("source_receipt_sha256") != source_receipt_sha256
        or not isinstance(intent.get("wave_index"), int)
        or isinstance(intent.get("wave_index"), bool)
        or not 0 <= intent["wave_index"] < transport.WAVE_COUNT
        or any(
            not isinstance(intent.get(field), str) or len(intent[field]) != 64
            for field in (
                "request_sha256",
                "source_receipt_sha256",
                "cost_reservation_receipt_sha256",
                "cost_settlement_receipt_sha256",
                "oauth_preflight_receipt_sha256",
                "launch_receipt_sha256",
                "poll_receipt_sha256",
                "lifecycle_receipt_sha256",
                "receive_receipt_sha256",
            )
        )
        or not isinstance(intent.get("cleanup_incomplete"), bool)
        or intent.get("one_wave_only") is not True
        or intent.get("max_concurrent_wave_count") != 1
        or intent.get("cloud_mutated") is not True
        or intent.get("current_profile_changed") is not False
    ):
        raise ValueError("M3.1 supervisor checkpoint intent changed")
    return intent


def _accepted_with_next(
    *,
    base: Mapping[str, Any],
    controller_status: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = deepcopy(dict(base))
    if (
        accepted.get("acceptance_sha256") != _self_digest(accepted, "acceptance_sha256")
        or accepted.get("schema") != controller.ACCEPTANCE_SCHEMA
        or accepted.get("cloud_mutated") is not False
        or accepted.get("current_profile_changed") is not False
    ):
        raise ValueError("M3.1 recovered acceptance receipt changed")
    next_status = controller_status.get("resume_status")
    if next_status not in {
        "wave_ready",
        "complete",
        "no_go_attempts_exhausted",
    }:
        raise ValueError("M3.1 recovered controller status changed")
    return {
        **accepted,
        "next_status": next_status,
        "next_wave_index": (
            controller_status.get("resume_wave_index")
            if next_status == "wave_ready"
            else None
        ),
        "next_selected_count": (
            controller_status.get("selected_count")
            if next_status == "wave_ready"
            else 0
        ),
    }


def _build_checkpoint(
    *,
    intent: Mapping[str, Any],
    accepted: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        accepted.get("acceptance_sha256") is None
        or accepted.get("request_sha256") != intent["request_sha256"]
        or accepted.get("wave_index") != intent["wave_index"]
        or accepted.get("lifecycle_receipt_sha256")
        != intent["lifecycle_receipt_sha256"]
        or accepted.get("receive_receipt_sha256") != intent["receive_receipt_sha256"]
        or accepted.get("next_status")
        not in {"wave_ready", "complete", "no_go_attempts_exhausted"}
    ):
        raise ValueError("M3.1 checkpoint acceptance differs from intent")
    acceptance_core = {
        key: value
        for key, value in accepted.items()
        if key not in {"next_status", "next_wave_index", "next_selected_count"}
    }
    if acceptance_core.get("acceptance_sha256") != _self_digest(
        acceptance_core, "acceptance_sha256"
    ):
        raise ValueError("M3.1 checkpoint acceptance digest changed")
    core = {
        "schema": SUPERVISOR_CHECKPOINT_SCHEMA,
        "status": "wave_checkpointed_continue",
        "run_name": intent["run_name"],
        "wave_index": intent["wave_index"],
        "request_sha256": intent["request_sha256"],
        "source_receipt_sha256": intent["source_receipt_sha256"],
        "cost_reservation_receipt_sha256": intent["cost_reservation_receipt_sha256"],
        "cost_settlement_receipt_sha256": intent["cost_settlement_receipt_sha256"],
        "oauth_preflight_receipt_sha256": intent["oauth_preflight_receipt_sha256"],
        "launch_receipt_sha256": intent["launch_receipt_sha256"],
        "poll_receipt_sha256": intent["poll_receipt_sha256"],
        "cleanup_incomplete": intent["cleanup_incomplete"],
        "lifecycle_receipt_sha256": intent["lifecycle_receipt_sha256"],
        "receive_receipt_sha256": intent["receive_receipt_sha256"],
        "acceptance_receipt_sha256": accepted["acceptance_sha256"],
        "next_status": accepted["next_status"],
        "one_wave_only": True,
        "max_concurrent_wave_count": 1,
        "cloud_mutated": True,
        "current_profile_changed": False,
    }
    return {**core, "checkpoint_sha256": canonical_sha256(core)}


def _validate_checkpoint(
    value: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = deepcopy(dict(value))
    required = {
        "schema",
        "status",
        "run_name",
        "wave_index",
        "request_sha256",
        "source_receipt_sha256",
        "cost_reservation_receipt_sha256",
        "cost_settlement_receipt_sha256",
        "oauth_preflight_receipt_sha256",
        "launch_receipt_sha256",
        "poll_receipt_sha256",
        "cleanup_incomplete",
        "lifecycle_receipt_sha256",
        "receive_receipt_sha256",
        "acceptance_receipt_sha256",
        "next_status",
        "one_wave_only",
        "max_concurrent_wave_count",
        "cloud_mutated",
        "current_profile_changed",
        "checkpoint_sha256",
    }
    if (
        set(checkpoint) != required
        or checkpoint.get("checkpoint_sha256")
        != _self_digest(checkpoint, "checkpoint_sha256")
        or checkpoint.get("schema") != SUPERVISOR_CHECKPOINT_SCHEMA
        or checkpoint.get("status") != "wave_checkpointed_continue"
        or checkpoint.get("run_name") != intent["run_name"]
        or checkpoint.get("wave_index") != intent["wave_index"]
        or checkpoint.get("request_sha256") != intent["request_sha256"]
        or checkpoint.get("source_receipt_sha256") != intent["source_receipt_sha256"]
        or checkpoint.get("cost_reservation_receipt_sha256")
        != intent["cost_reservation_receipt_sha256"]
        or checkpoint.get("cost_settlement_receipt_sha256")
        != intent["cost_settlement_receipt_sha256"]
        or checkpoint.get("oauth_preflight_receipt_sha256")
        != intent["oauth_preflight_receipt_sha256"]
        or checkpoint.get("launch_receipt_sha256") != intent["launch_receipt_sha256"]
        or checkpoint.get("poll_receipt_sha256") != intent["poll_receipt_sha256"]
        or checkpoint.get("cleanup_incomplete") is not intent["cleanup_incomplete"]
        or checkpoint.get("lifecycle_receipt_sha256")
        != intent["lifecycle_receipt_sha256"]
        or checkpoint.get("receive_receipt_sha256") != intent["receive_receipt_sha256"]
        or not isinstance(checkpoint.get("acceptance_receipt_sha256"), str)
        or len(checkpoint["acceptance_receipt_sha256"]) != 64
        or checkpoint.get("next_status")
        not in {"wave_ready", "complete", "no_go_attempts_exhausted"}
        or checkpoint.get("one_wave_only") is not True
        or checkpoint.get("max_concurrent_wave_count") != 1
        or checkpoint.get("cloud_mutated") is not True
        or checkpoint.get("current_profile_changed") is not False
    ):
        raise ValueError("M3.1 supervisor checkpoint changed")
    return checkpoint


def _validate_receipt_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    request_sha256: str | None,
) -> dict[str, Any]:
    value = _read(path, label)
    if (
        value.get("receipt_sha256") != expected_sha256
        or value.get("receipt_sha256") != _self_digest(value, "receipt_sha256")
        or (
            request_sha256 is not None and value.get("request_sha256") != request_sha256
        )
    ):
        raise ValueError(f"{label} differs from checkpoint intent")
    return value


def _validate_intent_artifacts(
    wave_root: Path,
    *,
    intent: Mapping[str, Any],
) -> None:
    _validate_receipt_file(
        wave_root / "source-receipt.json",
        label="supervisor source receipt",
        expected_sha256=intent["source_receipt_sha256"],
        request_sha256=None,
    )
    reservation = _validate_cost_reservation(
        _read(wave_root / "cost-reservation.json", "dataset cost reservation")
    )
    if (
        reservation["receipt_sha256"] != intent["cost_reservation_receipt_sha256"]
        or reservation["request_sha256"] != intent["request_sha256"]
    ):
        raise ValueError("dataset cost reservation differs from checkpoint intent")
    _validate_receipt_file(
        wave_root
        / "oauth-preflight"
        / f"{intent['oauth_preflight_receipt_sha256']}.json",
        label="supervisor OAuth preflight",
        expected_sha256=intent["oauth_preflight_receipt_sha256"],
        request_sha256=intent["request_sha256"],
    )
    _validate_receipt_file(
        wave_root / "launch-receipt.json",
        label="supervisor launch receipt",
        expected_sha256=intent["launch_receipt_sha256"],
        request_sha256=None,
    )
    poll_root = wave_root / "poll"
    if poll_root.is_symlink() or not poll_root.is_dir():
        raise ValueError("supervisor poll receipt directory is unsafe")
    poll_paths = [
        path
        for path in poll_root.iterdir()
        if path.name.endswith(f"-{intent['poll_receipt_sha256']}.json")
    ]
    if not poll_paths:
        raise ValueError("supervisor checkpoint poll receipt is absent")
    for poll_path in poll_paths:
        _validate_receipt_file(
            poll_path,
            label="supervisor poll receipt",
            expected_sha256=intent["poll_receipt_sha256"],
            request_sha256=None,
        )
    _validate_receipt_file(
        wave_root / "lifecycle-receipt.json",
        label="supervisor lifecycle receipt",
        expected_sha256=intent["lifecycle_receipt_sha256"],
        request_sha256=intent["request_sha256"],
    )
    lifecycle = _read(
        wave_root / "lifecycle-receipt.json",
        "dataset cost lifecycle receipt",
    )
    settlement = _validate_cost_settlement(
        _read(wave_root / "cost-settlement.json", "dataset cost settlement"),
        reservation=reservation,
        lifecycle=lifecycle,
    )
    if settlement["receipt_sha256"] != intent["cost_settlement_receipt_sha256"]:
        raise ValueError("dataset cost settlement differs from checkpoint intent")
    _validate_receipt_file(
        wave_root / "receive-receipt.json",
        label="supervisor receive receipt",
        expected_sha256=intent["receive_receipt_sha256"],
        request_sha256=None,
    )


def _read(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_bytes(value)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise FileExistsError(f"immutable supervisor artifact conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _guarded_compute_cost_micro_usd(
    *, instance_count: int, billable_seconds: int
) -> int:
    if (
        not isinstance(instance_count, int)
        or isinstance(instance_count, bool)
        or not 1 <= instance_count <= transport.MAX_CONCURRENT_VMS
        or not isinstance(billable_seconds, int)
        or isinstance(billable_seconds, bool)
        or not MINIMUM_BILLABLE_SECONDS <= billable_seconds <= provider.MAX_RUN_SECONDS
    ):
        raise ValueError("M3.1 dataset cost inputs are outside the frozen guard")
    return _ceil_div(
        instance_count * billable_seconds * SPOT_RATE_GUARD_MICRO_USD_PER_VM_HOUR,
        3600,
    )


def _cost_reservation(
    *,
    context: controller.WaveContext,
    reserved_at_unix_seconds: int,
) -> dict[str, Any]:
    if (
        not isinstance(reserved_at_unix_seconds, int)
        or isinstance(reserved_at_unix_seconds, bool)
        or reserved_at_unix_seconds <= 0
    ):
        raise ValueError("M3.1 dataset cost reservation time is invalid")
    selected_count = context.wave_request["selected_count"]
    reserved_compute = _guarded_compute_cost_micro_usd(
        instance_count=selected_count,
        billable_seconds=provider.MAX_RUN_SECONDS,
    )
    core = {
        "schema": COST_RESERVATION_SCHEMA,
        "status": "compute_cost_reserved_before_cloud_open",
        "run_name": context.contract["run_name"],
        "wave_index": context.wave_request["wave_index"],
        "request_sha256": context.wave_request["request_sha256"],
        "selected_instance_count": selected_count,
        "reserved_at_unix_seconds": reserved_at_unix_seconds,
        "max_run_seconds": provider.MAX_RUN_SECONDS,
        "minimum_billable_seconds": MINIMUM_BILLABLE_SECONDS,
        "spot_rate_guard_micro_usd_per_vm_hour": (
            SPOT_RATE_GUARD_MICRO_USD_PER_VM_HOUR
        ),
        "reserved_compute_cost_micro_usd": reserved_compute,
        "compute_cost_cap_micro_usd": COMPUTE_COST_CAP_MICRO_USD,
        "non_compute_reserve_micro_usd": NON_COMPUTE_RESERVE_MICRO_USD,
        "total_cloud_cost_cap_micro_usd": TOTAL_CLOUD_COST_CAP_MICRO_USD,
        "live_spot_price_recheck_required_before_launch": True,
        "cloud_opened": False,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _validate_cost_reservation(
    value: Mapping[str, Any],
    *,
    context: controller.WaveContext | None = None,
) -> dict[str, Any]:
    reservation = deepcopy(dict(value))
    required = {
        "schema",
        "status",
        "run_name",
        "wave_index",
        "request_sha256",
        "selected_instance_count",
        "reserved_at_unix_seconds",
        "max_run_seconds",
        "minimum_billable_seconds",
        "spot_rate_guard_micro_usd_per_vm_hour",
        "reserved_compute_cost_micro_usd",
        "compute_cost_cap_micro_usd",
        "non_compute_reserve_micro_usd",
        "total_cloud_cost_cap_micro_usd",
        "live_spot_price_recheck_required_before_launch",
        "cloud_opened",
        "cloud_mutated",
        "current_profile_changed",
        "receipt_sha256",
    }
    selected_count = reservation.get("selected_instance_count")
    reserved_at = reservation.get("reserved_at_unix_seconds")
    valid_inputs = (
        isinstance(selected_count, int)
        and not isinstance(selected_count, bool)
        and 1 <= selected_count <= transport.MAX_CONCURRENT_VMS
        and isinstance(reserved_at, int)
        and not isinstance(reserved_at, bool)
        and reserved_at > 0
    )
    expected_cost = (
        _guarded_compute_cost_micro_usd(
            instance_count=selected_count,
            billable_seconds=provider.MAX_RUN_SECONDS,
        )
        if valid_inputs
        else None
    )
    if (
        set(reservation) != required
        or reservation.get("receipt_sha256")
        != _self_digest(reservation, "receipt_sha256")
        or reservation.get("schema") != COST_RESERVATION_SCHEMA
        or reservation.get("status") != "compute_cost_reserved_before_cloud_open"
        or not isinstance(reservation.get("run_name"), str)
        or not isinstance(reservation.get("wave_index"), int)
        or isinstance(reservation.get("wave_index"), bool)
        or not 0 <= reservation["wave_index"] < transport.WAVE_COUNT
        or not isinstance(reservation.get("request_sha256"), str)
        or len(reservation["request_sha256"]) != 64
        or not valid_inputs
        or reservation.get("max_run_seconds") != provider.MAX_RUN_SECONDS
        or reservation.get("minimum_billable_seconds") != MINIMUM_BILLABLE_SECONDS
        or reservation.get("spot_rate_guard_micro_usd_per_vm_hour")
        != SPOT_RATE_GUARD_MICRO_USD_PER_VM_HOUR
        or reservation.get("reserved_compute_cost_micro_usd") != expected_cost
        or reservation.get("compute_cost_cap_micro_usd") != COMPUTE_COST_CAP_MICRO_USD
        or reservation.get("non_compute_reserve_micro_usd")
        != NON_COMPUTE_RESERVE_MICRO_USD
        or reservation.get("total_cloud_cost_cap_micro_usd")
        != TOTAL_CLOUD_COST_CAP_MICRO_USD
        or reservation.get("live_spot_price_recheck_required_before_launch") is not True
        or reservation.get("cloud_opened") is not False
        or reservation.get("cloud_mutated") is not False
        or reservation.get("current_profile_changed") is not False
    ):
        raise ValueError("M3.1 dataset cost reservation changed")
    if context is not None and (
        reservation["run_name"] != context.contract["run_name"]
        or reservation["wave_index"] != context.wave_request["wave_index"]
        or reservation["request_sha256"] != context.wave_request["request_sha256"]
        or reservation["selected_instance_count"]
        != context.wave_request["selected_count"]
    ):
        raise ValueError("M3.1 dataset cost reservation differs from current wave")
    return reservation


def _cost_settlement(
    *,
    reservation: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    settled_at_unix_seconds: int,
) -> dict[str, Any]:
    reserved = _validate_cost_reservation(reservation)
    if (
        not isinstance(settled_at_unix_seconds, int)
        or isinstance(settled_at_unix_seconds, bool)
        or settled_at_unix_seconds < reserved["reserved_at_unix_seconds"]
        or lifecycle.get("request_sha256") != reserved["request_sha256"]
        or lifecycle.get("receipt_sha256") != _self_digest(lifecycle, "receipt_sha256")
        or lifecycle.get("owned_vm_disk_absent") is not True
        or lifecycle.get("worker_iam_removed_before_receive") is not True
    ):
        raise ValueError("M3.1 dataset cost settlement boundary is invalid")
    elapsed = settled_at_unix_seconds - reserved["reserved_at_unix_seconds"]
    billable_seconds = min(
        provider.MAX_RUN_SECONDS,
        max(MINIMUM_BILLABLE_SECONDS, elapsed),
    )
    guarded_cost = _guarded_compute_cost_micro_usd(
        instance_count=reserved["selected_instance_count"],
        billable_seconds=billable_seconds,
    )
    core = {
        "schema": COST_SETTLEMENT_SCHEMA,
        "status": "guarded_compute_cost_settled_after_exact_cleanup",
        "run_name": reserved["run_name"],
        "wave_index": reserved["wave_index"],
        "request_sha256": reserved["request_sha256"],
        "reservation_receipt_sha256": reserved["receipt_sha256"],
        "lifecycle_receipt_sha256": lifecycle["receipt_sha256"],
        "selected_instance_count": reserved["selected_instance_count"],
        "reserved_at_unix_seconds": reserved["reserved_at_unix_seconds"],
        "settled_at_unix_seconds": settled_at_unix_seconds,
        "elapsed_seconds": elapsed,
        "guarded_billable_seconds": billable_seconds,
        "spot_rate_guard_micro_usd_per_vm_hour": (
            SPOT_RATE_GUARD_MICRO_USD_PER_VM_HOUR
        ),
        "reserved_compute_cost_micro_usd": reserved["reserved_compute_cost_micro_usd"],
        "settled_compute_cost_micro_usd": guarded_cost,
        "settled_not_above_reservation": (
            guarded_cost <= reserved["reserved_compute_cost_micro_usd"]
        ),
        "exact_owned_cleanup_complete": True,
        "cost_is_conservative_upper_bound_if_live_spot_price_within_guard": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _validate_cost_settlement(
    value: Mapping[str, Any],
    *,
    reservation: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
) -> dict[str, Any]:
    settlement = deepcopy(dict(value))
    reserved = _validate_cost_reservation(reservation)
    required = {
        "schema",
        "status",
        "run_name",
        "wave_index",
        "request_sha256",
        "reservation_receipt_sha256",
        "lifecycle_receipt_sha256",
        "selected_instance_count",
        "reserved_at_unix_seconds",
        "settled_at_unix_seconds",
        "elapsed_seconds",
        "guarded_billable_seconds",
        "spot_rate_guard_micro_usd_per_vm_hour",
        "reserved_compute_cost_micro_usd",
        "settled_compute_cost_micro_usd",
        "settled_not_above_reservation",
        "exact_owned_cleanup_complete",
        "cost_is_conservative_upper_bound_if_live_spot_price_within_guard",
        "cloud_mutated",
        "current_profile_changed",
        "receipt_sha256",
    }
    settled_at = settlement.get("settled_at_unix_seconds")
    elapsed = (
        settled_at - reserved["reserved_at_unix_seconds"]
        if isinstance(settled_at, int) and not isinstance(settled_at, bool)
        else None
    )
    billable = (
        min(
            provider.MAX_RUN_SECONDS,
            max(MINIMUM_BILLABLE_SECONDS, elapsed),
        )
        if elapsed is not None and elapsed >= 0
        else None
    )
    expected_cost = (
        _guarded_compute_cost_micro_usd(
            instance_count=reserved["selected_instance_count"],
            billable_seconds=billable,
        )
        if billable is not None
        else None
    )
    if (
        set(settlement) != required
        or settlement.get("receipt_sha256")
        != _self_digest(settlement, "receipt_sha256")
        or settlement.get("schema") != COST_SETTLEMENT_SCHEMA
        or settlement.get("status")
        != "guarded_compute_cost_settled_after_exact_cleanup"
        or settlement.get("run_name") != reserved["run_name"]
        or settlement.get("wave_index") != reserved["wave_index"]
        or settlement.get("request_sha256") != reserved["request_sha256"]
        or settlement.get("reservation_receipt_sha256") != reserved["receipt_sha256"]
        or settlement.get("lifecycle_receipt_sha256") != lifecycle.get("receipt_sha256")
        or settlement.get("selected_instance_count")
        != reserved["selected_instance_count"]
        or settlement.get("reserved_at_unix_seconds")
        != reserved["reserved_at_unix_seconds"]
        or elapsed is None
        or elapsed < 0
        or settlement.get("elapsed_seconds") != elapsed
        or settlement.get("guarded_billable_seconds") != billable
        or settlement.get("spot_rate_guard_micro_usd_per_vm_hour")
        != SPOT_RATE_GUARD_MICRO_USD_PER_VM_HOUR
        or settlement.get("reserved_compute_cost_micro_usd")
        != reserved["reserved_compute_cost_micro_usd"]
        or settlement.get("settled_compute_cost_micro_usd") != expected_cost
        or settlement.get("settled_not_above_reservation") is not True
        or settlement.get("exact_owned_cleanup_complete") is not True
        or settlement.get(
            "cost_is_conservative_upper_bound_if_live_spot_price_within_guard"
        )
        is not True
        or settlement.get("cloud_mutated") is not False
        or settlement.get("current_profile_changed") is not False
        or lifecycle.get("request_sha256") != reserved["request_sha256"]
        or lifecycle.get("receipt_sha256") != _self_digest(lifecycle, "receipt_sha256")
        or lifecycle.get("owned_vm_disk_absent") is not True
        or lifecycle.get("worker_iam_removed_before_receive") is not True
    ):
        raise ValueError("M3.1 dataset cost settlement changed")
    return settlement


def _cost_status(
    supervisor_root: Path,
    *,
    allowed_unreserved_wave_root: Path | None = None,
) -> dict[str, Any]:
    if supervisor_root.is_symlink() or not supervisor_root.is_dir():
        raise ValueError("M3.1 dataset cost root is unsafe")
    records = []
    request_ids: set[str] = set()
    committed_compute = 0
    for path in sorted(supervisor_root.iterdir()):
        if path.name == LOCK_FILENAME:
            continue
        if path.is_symlink() or not path.is_dir() or not path.name.startswith("wave-"):
            raise ValueError("M3.1 dataset cost root contains an unsafe entry")
        reservation_path = path / "cost-reservation.json"
        if (
            allowed_unreserved_wave_root is not None
            and path == allowed_unreserved_wave_root
            and not reservation_path.exists()
            and not reservation_path.is_symlink()
        ):
            continue
        reservation = _validate_cost_reservation(
            _read(reservation_path, "dataset cost reservation")
        )
        if reservation["request_sha256"] in request_ids:
            raise ValueError("M3.1 dataset cost reservation is duplicated")
        request_ids.add(reservation["request_sha256"])
        settlement_path = path / "cost-settlement.json"
        if settlement_path.exists() or settlement_path.is_symlink():
            lifecycle = _read(
                path / "lifecycle-receipt.json",
                "dataset cost lifecycle receipt",
            )
            settlement = _validate_cost_settlement(
                _read(settlement_path, "dataset cost settlement"),
                reservation=reservation,
                lifecycle=lifecycle,
            )
            committed = settlement["settled_compute_cost_micro_usd"]
            state = "settled"
            settlement_sha = settlement["receipt_sha256"]
        else:
            committed = reservation["reserved_compute_cost_micro_usd"]
            state = "reserved_unsettled"
            settlement_sha = None
        committed_compute += committed
        records.append(
            {
                "wave_index": reservation["wave_index"],
                "request_sha256": reservation["request_sha256"],
                "state": state,
                "reservation_receipt_sha256": reservation["receipt_sha256"],
                "settlement_receipt_sha256": settlement_sha,
                "committed_compute_cost_micro_usd": committed,
            }
        )
    total_guarded = NON_COMPUTE_RESERVE_MICRO_USD + committed_compute
    if (
        committed_compute > COMPUTE_COST_CAP_MICRO_USD
        or total_guarded > TOTAL_CLOUD_COST_CAP_MICRO_USD
    ):
        raise PermissionError("M3.1 dataset cloud cost cap is exhausted")
    core = {
        "schema": COST_STATUS_SCHEMA,
        "status": "within_frozen_500_usd_total_cloud_guard",
        "reservation_count": len(records),
        "records": records,
        "record_aggregate_sha256": canonical_sha256(records),
        "committed_compute_cost_micro_usd": committed_compute,
        "remaining_compute_cost_micro_usd": (
            COMPUTE_COST_CAP_MICRO_USD - committed_compute
        ),
        "non_compute_reserve_micro_usd": NON_COMPUTE_RESERVE_MICRO_USD,
        "guarded_total_cost_micro_usd": total_guarded,
        "total_cloud_cost_cap_micro_usd": TOTAL_CLOUD_COST_CAP_MICRO_USD,
        "cost_cap_not_exceeded": True,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def build_cost_status(supervisor_root: str | Path) -> dict[str, Any]:
    root = Path(supervisor_root).resolve()
    if not root.exists():
        root.mkdir(parents=True)
    return _cost_status(root)


def _authorize_cost_reservation(
    *,
    supervisor_root: Path,
    wave_root: Path,
    context: controller.WaveContext,
    reserved_at_unix_seconds: int,
) -> dict[str, Any]:
    reservation_path = wave_root / "cost-reservation.json"
    if reservation_path.exists() or reservation_path.is_symlink():
        return _validate_cost_reservation(
            _read(reservation_path, "dataset cost reservation"),
            context=context,
        )
    status = _cost_status(
        supervisor_root,
        allowed_unreserved_wave_root=wave_root,
    )
    reservation = _cost_reservation(
        context=context,
        reserved_at_unix_seconds=reserved_at_unix_seconds,
    )
    if (
        status["committed_compute_cost_micro_usd"]
        + reservation["reserved_compute_cost_micro_usd"]
        > COMPUTE_COST_CAP_MICRO_USD
    ):
        raise PermissionError(
            "M3.1 dataset next wave exceeds the frozen $500 cloud cost cap"
        )
    wave_root.mkdir(parents=True, exist_ok=True)
    _write_once(reservation_path, reservation)
    return reservation


def _settle_cost(
    *,
    wave_root: Path,
    reservation: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    settled_at_unix_seconds: int,
) -> dict[str, Any]:
    path = wave_root / "cost-settlement.json"
    if path.exists() or path.is_symlink():
        return _validate_cost_settlement(
            _read(path, "dataset cost settlement"),
            reservation=reservation,
            lifecycle=lifecycle,
        )
    settlement = _cost_settlement(
        reservation=reservation,
        lifecycle=lifecycle,
        settled_at_unix_seconds=settled_at_unix_seconds,
    )
    _write_once(path, settlement)
    return settlement


def _require_cost_confirmation(
    *,
    total_cost_cap_usd: str,
    spot_rate_guard_usd_per_vm_hour: str,
) -> None:
    if (
        total_cost_cap_usd != CONFIRM_TOTAL_COST_CAP_USD
        or spot_rate_guard_usd_per_vm_hour != CONFIRM_SPOT_RATE_GUARD_USD_PER_VM_HOUR
    ):
        raise PermissionError(
            "M3.1 dataset cost cap/live Spot guard confirmation differs"
        )


@contextmanager
def _exclusive_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock = root / LOCK_FILENAME
    payload = canonical_bytes(
        {
            "schema": "hu_m31_t3_dataset_supervisor_lock_v1",
            "pid": os.getpid(),
            "create_only": True,
        }
    )
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PermissionError("another M3.1 dataset supervisor is active") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        if lock.is_file() and not lock.is_symlink() and lock.read_bytes() == payload:
            lock.unlink()


@contextmanager
def _phase_sentinel(context: controller.WaveContext, phase: str) -> Iterator[None]:
    previous = os.environ.get(controller.PHASE_SENTINEL_ENV)
    os.environ[controller.PHASE_SENTINEL_ENV] = controller.expected_phase_sentinel(
        context, phase
    )
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(controller.PHASE_SENTINEL_ENV, None)
        else:
            os.environ[controller.PHASE_SENTINEL_ENV] = previous


def build_oauth_preflight(
    *,
    context: controller.WaveContext,
    cloud: TtlCloud,
    observed_at_utc: str,
) -> dict[str, Any]:
    ttl = cloud.get_oauth_token_ttl_seconds()
    if not isinstance(ttl, int) or isinstance(ttl, bool) or ttl < MIN_OAUTH_TTL_SECONDS:
        raise PermissionError("M3.1 dataset OAuth TTL is below 2700 seconds")
    core = {
        "schema": OAUTH_PREFLIGHT_SCHEMA,
        "status": "fresh_redacted_oauth_ttl_pass",
        "run_name": context.contract["run_name"],
        "request_sha256": context.wave_request["request_sha256"],
        "wave_index": context.wave_request["wave_index"],
        "observed_at_utc": observed_at_utc,
        "ttl_seconds": ttl,
        "minimum_ttl_seconds": MIN_OAUTH_TTL_SECONDS,
        "access_token_persisted": False,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _load_optional(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    return _read(path, label)


def _recover_accepted_checkpoint(
    *,
    controller_root: str | Path,
    supervisor_root: Path,
    confirm_run_name: str,
    source_receipt: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not supervisor_root.exists():
        return None
    if supervisor_root.is_symlink() or not supervisor_root.is_dir():
        raise ValueError("M3.1 supervisor root is unsafe")
    candidates = []
    for path in sorted(supervisor_root.iterdir()):
        if path.is_symlink() or not path.is_dir():
            if path.name == LOCK_FILENAME:
                continue
            raise ValueError("M3.1 supervisor root contains an unsafe entry")
        intent_path = path / "checkpoint-intent.json"
        checkpoint_path = path / "SUPERVISOR_CHECKPOINT.json"
        if checkpoint_path.is_symlink():
            raise ValueError("M3.1 supervisor checkpoint is unsafe")
        if intent_path.exists() or intent_path.is_symlink():
            intent = _validate_checkpoint_intent(
                _read(intent_path, "supervisor checkpoint intent"),
                confirm_run_name=confirm_run_name,
                source_receipt_sha256=source_receipt["receipt_sha256"],
            )
            if checkpoint_path.exists():
                checkpoint = _validate_checkpoint(
                    _read(checkpoint_path, "supervisor checkpoint"),
                    intent=intent,
                )
                accepted = _read(
                    path / "acceptance-receipt.json",
                    "supervisor acceptance receipt",
                )
                if _build_checkpoint(intent=intent, accepted=accepted) != checkpoint:
                    raise ValueError("supervisor acceptance differs from checkpoint")
            else:
                _validate_intent_artifacts(path, intent=intent)
                candidates.append((path, intent))
        elif checkpoint_path.exists():
            raise ValueError("M3.1 supervisor checkpoint has no intent")
    if len(candidates) > 1:
        raise PermissionError("multiple incomplete supervisor checkpoints exist")
    if not candidates:
        return None
    wave_root, intent = candidates[0]
    controller_wave = (
        Path(controller_root).resolve()
        / "waves"
        / f"w{intent['wave_index']:02d}-{intent['request_sha256'][:16]}"
    )
    controller_acceptance_path = controller_wave / "acceptance_receipt.json"
    if (
        not controller_acceptance_path.exists()
        and not controller_acceptance_path.is_symlink()
    ):
        return None
    acceptance_base = _read(controller_acceptance_path, "controller acceptance receipt")
    status = controller.controller_status(controller_root)
    accepted_ordinal = acceptance_base.get("accepted_ordinal")
    if not isinstance(accepted_ordinal, int) or isinstance(accepted_ordinal, bool):
        raise ValueError("controller acceptance ordinal changed")
    accepted_count = status["accepted_lifecycle_count"]
    if accepted_count == accepted_ordinal:
        # The controller may have written its acceptance receipt but crashed
        # before publishing the accepted lifecycle marker.  The current-wave
        # path below will replay accept_next_wave and complete that commit.
        return None
    if accepted_count != accepted_ordinal + 1:
        raise PermissionError(
            "controller advanced beyond an incomplete supervisor checkpoint"
        )
    accepted = _accepted_with_next(
        base=acceptance_base,
        controller_status=status,
    )
    _write_once(wave_root / "acceptance-receipt.json", accepted)
    checkpoint = _build_checkpoint(intent=intent, accepted=accepted)
    _write_once(wave_root / "SUPERVISOR_CHECKPOINT.json", checkpoint)
    return checkpoint


def _reconcile_unwritten_iam(
    *,
    context: controller.WaveContext,
    cloud: Any,
    now_unix_seconds: int,
) -> dict[str, Any] | None:
    """Adopt only an exact complete wave binding set for immediate removal."""

    policy = dict(cloud.get_bucket_iam_policy(bucket=context.provider_plan["bucket"]))
    bindings, _etag, _version = provider._policy_parts(  # type: ignore[attr-defined]
        policy
    )
    titles = set(context.provider_plan["iam_contract"]["condition_titles"])
    matching = [
        row
        for row in bindings
        if provider._condition_title(row) in titles  # type: ignore[attr-defined]
    ]
    if not matching:
        return None
    expiry = provider._existing_iam_expiry(  # type: ignore[attr-defined]
        context.provider_plan,
        bindings,
        now_unix_seconds=now_unix_seconds,
    )
    if expiry is None:
        raise PermissionError("unwritten dataset IAM could not be reconciled")
    core = {
        "schema": provider.IAM_RECEIPT_SCHEMA,
        "status": "exact_temporary_worker_bindings_installed",
        "provider_plan_sha256": context.provider_plan["provider_plan_sha256"],
        "bucket": context.provider_plan["bucket"],
        "condition_titles": list(
            context.provider_plan["iam_contract"]["condition_titles"]
        ),
        "members": context.provider_plan["iam_contract"]["members"],
        "expires_at_utc": expiry,
        "bindings_installed": len(context.provider_plan["workers"]) + 1,
        "readback_exact": True,
        "response_reconciled": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return provider.validate_iam_receipt(
        {**core, "receipt_sha256": canonical_sha256(core)},
        provider_plan=context.provider_plan,
    )


def _require_live_iam_matches_receipt(
    *,
    context: controller.WaveContext,
    iam_receipt: Mapping[str, Any],
    cloud: Any,
    now_unix_seconds: int,
) -> None:
    stored = provider.validate_iam_receipt(
        iam_receipt, provider_plan=context.provider_plan
    )
    live = _reconcile_unwritten_iam(
        context=context,
        cloud=cloud,
        now_unix_seconds=now_unix_seconds,
    )
    if live is None or live["expires_at_utc"] != stored["expires_at_utc"]:
        raise PermissionError(
            "stored dataset IAM receipt has no exact live binding set"
        )


def _abort_partial_launch(
    *,
    context: controller.WaveContext,
    cloud: Any,
    sleep: Callable[[float], None],
    now_unix_seconds: int,
) -> dict[str, Any]:
    """Delete only exact selected compute and remove exact temporary IAM."""

    stage_path = context.directory / "stage_receipt.json"
    iam_path = context.directory / "iam_receipt.json"
    stage = _load_optional(stage_path, "dataset stage receipt")
    iam = _load_optional(iam_path, "dataset IAM receipt")
    if stage is None:
        stage_valid = None
    else:
        stage_valid = provider.validate_stage_receipt(
            stage, provider_plan=context.provider_plan
        )
    if iam is None:
        iam_valid = _reconcile_unwritten_iam(
            context=context,
            cloud=cloud,
            now_unix_seconds=now_unix_seconds,
        )
    else:
        iam_valid = provider.validate_iam_receipt(
            iam, provider_plan=context.provider_plan
        )
    workers = {row["shard_id"]: row for row in context.provider_plan["workers"]}
    rows = []
    for selected in context.provider_plan["selected_attempts"]:
        instance = cloud.get_instance(instance_name=selected["instance_id"])
        disk = cloud.get_disk_optional(disk_name=selected["instance_id"])
        if instance is None and disk is None:
            rows.append(
                {
                    "shard_id": selected["shard_id"],
                    "instance_id": selected["instance_id"],
                    "instance_present": False,
                    "disk_present": False,
                    "instance_deleted": False,
                    "disk_deleted": False,
                }
            )
            continue
        if stage_valid is None:
            raise PermissionError("partial compute exists without staged content")
        spec = provider._instance_spec(  # type: ignore[attr-defined]
            provider=context.provider_plan,
            worker=workers[selected["shard_id"]],
            selected=selected,
            stage_receipt=stage_valid,
        )
        if instance is not None:
            provider._validate_owned_instance(  # type: ignore[attr-defined]
                instance, expected_spec=spec
            )
        if disk is not None:
            provider._validate_owned_disk(  # type: ignore[attr-defined]
                disk, expected_spec=spec
            )
        if instance is not None:
            request_id = provider._uuid_for(  # type: ignore[attr-defined]
                context.provider_plan["provider_plan_sha256"],
                selected["shard_id"],
                selected["attempt_id"],
                "delete",
            )
            operation = cloud.delete_instance(
                instance_name=selected["instance_id"], request_id=request_id
            )
            if operation is not None:
                quality_provider._wait_operation(  # type: ignore[arg-type]
                    cloud,
                    initial=operation,
                    operation_type="delete",
                    target_name=selected["instance_id"],
                    sleep=sleep,
                )
        disk_after_instance_delete = cloud.get_disk_optional(
            disk_name=selected["instance_id"]
        )
        disk_deleted = False
        if disk_after_instance_delete is not None:
            provider._validate_owned_disk(  # type: ignore[attr-defined]
                disk_after_instance_delete, expected_spec=spec
            )
            request_id = provider._uuid_for(  # type: ignore[attr-defined]
                context.provider_plan["provider_plan_sha256"],
                selected["shard_id"],
                selected["attempt_id"],
                "delete-disk",
            )
            operation = cloud.delete_disk(
                disk_name=selected["instance_id"], request_id=request_id
            )
            if operation is not None:
                quality_provider._wait_operation(  # type: ignore[arg-type]
                    cloud,
                    initial=operation,
                    operation_type="delete",
                    target_name=selected["instance_id"],
                    sleep=sleep,
                )
            disk_deleted = True
        rows.append(
            {
                "shard_id": selected["shard_id"],
                "instance_id": selected["instance_id"],
                "instance_present": instance is not None,
                "disk_present": disk is not None,
                "instance_deleted": instance is not None,
                "disk_deleted": disk_deleted,
            }
        )
    remaining = [
        selected["instance_id"]
        for selected in context.provider_plan["selected_attempts"]
        if cloud.get_instance(instance_name=selected["instance_id"]) is not None
        or cloud.get_disk_optional(disk_name=selected["instance_id"]) is not None
    ]
    if remaining:
        raise PermissionError("partial launch exact compute absence not proven")
    iam_cleanup_sha = None
    if iam_valid is not None:
        iam_cleanup = provider.remove_worker_iam(
            provider_plan=context.provider_plan,
            transport_plan=context.transport_plan,
            ledger=context.ledger,
            resume=context.resume,
            wave_request=context.wave_request,
            iam_receipt=iam_valid,
            transport=cloud,
        )
        iam_cleanup_sha = iam_cleanup["receipt_sha256"]
    core = {
        "schema": ABORT_CLEANUP_SCHEMA,
        "status": "partial_launch_exact_cleanup_complete_stop",
        "request_sha256": context.wave_request["request_sha256"],
        "rows": rows,
        "owned_vm_disk_absent": True,
        "worker_iam_absent": iam_valid is None or iam_cleanup_sha is not None,
        "iam_cleanup_receipt_sha256": iam_cleanup_sha,
        "next_wave_authorized": False,
        "wildcard_delete_used": False,
        "unrelated_resource_touched": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _parse_gcloud_expiry(value: Any) -> int:
    if not isinstance(value, str) or not value:
        raise PermissionError("gcloud credential expiry is absent")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PermissionError("gcloud credential expiry is invalid") from exc
    if parsed.tzinfo is None:
        raise PermissionError("gcloud credential expiry has no timezone")
    return int(parsed.timestamp())


def _force_refreshed_cloud_from_gcloud(
    *, now: Callable[[], int] | None = None
) -> _ExpiryBoundGcpCloud:
    clock = now or (lambda: int(time.time()))
    result = subprocess.run(
        [
            "gcloud",
            "config",
            "config-helper",
            "--force-auth-refresh",
            "--format=json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PermissionError("gcloud config-helper returned invalid JSON") from exc
    credential = value.get("credential") if isinstance(value, Mapping) else None
    if not isinstance(credential, Mapping):
        raise PermissionError("gcloud config-helper omitted credential data")
    token = credential.get("access_token")
    expiry = _parse_gcloud_expiry(credential.get("token_expiry"))
    if (
        not isinstance(token, str)
        or len(token) < 20
        or any(character.isspace() for character in token)
    ):
        raise PermissionError("gcloud returned an invalid refreshed OAuth token")
    if expiry <= clock():
        raise PermissionError("gcloud returned an expired refreshed OAuth token")
    return _ExpiryBoundGcpCloud(
        access_token=token,
        expiry_unix_seconds=expiry,
        now=clock,
    )


def _fresh_cloud_from_gcloud() -> GcpParallelDatasetRestAdapter:
    """Backward-compatible one-shot force-refresh helper."""

    return _force_refreshed_cloud_from_gcloud()


def run_next_wave(
    *,
    controller_root: str | Path,
    supervisor_root: str | Path,
    confirm_run_name: str,
    allow_cloud_mutation: bool,
    cloud_factory: Callable[[], Any] | None = None,
    max_poll_attempts: int = MAX_POLL_ATTEMPTS,
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], int] = lambda: int(time.time()),
) -> dict[str, Any]:
    if (
        allow_cloud_mutation is not True
        or not isinstance(max_poll_attempts, int)
        or isinstance(max_poll_attempts, bool)
        or not 1 <= max_poll_attempts <= MAX_POLL_ATTEMPTS
        or not isinstance(poll_interval_seconds, int)
        or isinstance(poll_interval_seconds, bool)
        or not 0 <= poll_interval_seconds <= 60
    ):
        raise PermissionError("M3.1 supervisor execution boundary is invalid")
    source_receipt = build_source_receipt()
    root = Path(supervisor_root).resolve()
    with _exclusive_lock(root):
        recovered = _recover_accepted_checkpoint(
            controller_root=controller_root,
            supervisor_root=root,
            confirm_run_name=confirm_run_name,
            source_receipt=source_receipt,
        )
        if recovered is not None:
            return recovered
        planned = controller.plan_next_wave(controller_root)
        if not isinstance(planned, controller.WaveContext):
            return {
                "status": planned["status"],
                "source_receipt_sha256": source_receipt["receipt_sha256"],
                "cost_status": _cost_status(root),
                "cloud_mutated": False,
                "current_profile_changed": False,
            }
        context = planned
        if context.contract["run_name"] != confirm_run_name:
            raise PermissionError("M3.1 supervisor confirmed run name differs")
        wave_root = (
            root / f"wave-{context.wave_request['wave_index']:02d}-"
            f"{context.wave_request['request_sha256'][:16]}"
        )
        if (wave_root / "abort-cleanup.json").exists() or (
            wave_root / "abort-cleanup.json"
        ).is_symlink():
            raise PermissionError(
                "previous partial launch was cleaned; explicit new controller "
                "recovery is required"
            )
        reservation = _authorize_cost_reservation(
            supervisor_root=root,
            wave_root=wave_root,
            context=context,
            reserved_at_unix_seconds=now(),
        )
        _write_once(wave_root / "source-receipt.json", source_receipt)
        pending_intent = _load_optional(
            wave_root / "checkpoint-intent.json",
            "supervisor checkpoint intent",
        )
        if pending_intent is not None:
            intent = _validate_checkpoint_intent(
                pending_intent,
                confirm_run_name=confirm_run_name,
                source_receipt_sha256=source_receipt["receipt_sha256"],
            )
            if (
                intent["request_sha256"] != context.wave_request["request_sha256"]
                or intent["wave_index"] != context.wave_request["wave_index"]
            ):
                raise PermissionError(
                    "pending checkpoint intent differs from current wave"
                )
            _validate_intent_artifacts(wave_root, intent=intent)
            accepted = controller.accept_next_wave(controller_root)
            _write_once(wave_root / "acceptance-receipt.json", accepted)
            checkpoint = _build_checkpoint(intent=intent, accepted=accepted)
            _write_once(wave_root / "SUPERVISOR_CHECKPOINT.json", checkpoint)
            return checkpoint

        if cloud_factory is None:
            lease = _GcloudCredentialLease()
            cloud = lease.get(force_refresh=True)
            phase_cloud_factory = lambda: lease.get(force_refresh=False)
        else:
            cloud = cloud_factory()
            phase_cloud_factory = cloud_factory
        observed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now()))
        preflight = build_oauth_preflight(
            context=context, cloud=cloud, observed_at_utc=observed_at
        )
        _write_once(
            wave_root / "oauth-preflight" / f"{preflight['receipt_sha256']}.json",
            preflight,
        )

        launch_path = wave_root / "launch-receipt.json"
        launch = _load_optional(launch_path, "supervisor launch receipt")
        if launch is None:
            controller_launch = _load_optional(
                context.directory / "launch_receipt.json",
                "controller launch receipt",
            )
            if controller_launch is None:
                stored_iam = _load_optional(
                    context.directory / "iam_receipt.json",
                    "dataset IAM receipt",
                )
                if stored_iam is not None:
                    _require_live_iam_matches_receipt(
                        context=context,
                        iam_receipt=stored_iam,
                        cloud=cloud,
                        now_unix_seconds=now(),
                    )
            try:
                with _phase_sentinel(context, "execute"):
                    launch = controller.execute_next_wave(
                        controller_root,
                        confirm_run_name=confirm_run_name,
                        allow_cloud_mutation=True,
                        cloud=cloud,
                        now_unix_seconds=now(),
                        sleep=sleep,
                    )
                _write_once(launch_path, launch)
            except Exception:
                abort = _abort_partial_launch(
                    context=context,
                    cloud=cloud,
                    sleep=sleep,
                    now_unix_seconds=now(),
                )
                _write_once(wave_root / "abort-cleanup.json", abort)
                raise
        else:
            launch = provider.validate_launch_receipt(
                launch, provider_plan=context.provider_plan
            )

        latest = None
        cleanup_incomplete = False
        for index in range(max_poll_attempts):
            poll_cloud = phase_cloud_factory()
            with _phase_sentinel(context, "poll"):
                latest = controller.poll_next_wave(
                    controller_root,
                    confirm_run_name=confirm_run_name,
                    allow_cloud_read=True,
                    cloud=poll_cloud,
                )
            _write_once(
                wave_root / "poll" / f"{index:06d}-{latest['receipt_sha256']}.json",
                latest,
            )
            if latest["complete_count"] == latest["selected_shard_count"]:
                break
            if any(
                row["instance_present"] is False and row["complete"] is False
                for row in latest["rows"]
            ):
                cleanup_incomplete = True
                break
            if index + 1 == max_poll_attempts:
                cleanup_incomplete = True
                break
            sleep(poll_interval_seconds)
        assert latest is not None

        cleanup_cloud = phase_cloud_factory()
        cleanup_phase = "cleanup-incomplete" if cleanup_incomplete else "cleanup"
        with _phase_sentinel(context, cleanup_phase):
            lifecycle = controller.cleanup_next_wave(
                controller_root,
                confirm_run_name=confirm_run_name,
                allow_cloud_mutation=True,
                allow_incomplete_cleanup=cleanup_incomplete,
                cloud=cleanup_cloud,
                sleep=sleep,
            )
        _write_once(wave_root / "lifecycle-receipt.json", lifecycle)
        settlement = _settle_cost(
            wave_root=wave_root,
            reservation=reservation,
            lifecycle=lifecycle,
            settled_at_unix_seconds=now(),
        )

        receive_cloud = phase_cloud_factory()
        with _phase_sentinel(context, "receive"):
            received = controller.receive_next_wave(
                controller_root,
                confirm_run_name=confirm_run_name,
                allow_cloud_read=True,
                cloud=receive_cloud,
            )
        _write_once(wave_root / "receive-receipt.json", received)
        intent_core = {
            "schema": CHECKPOINT_INTENT_SCHEMA,
            "status": "wave_phases_complete_ready_to_accept",
            "run_name": confirm_run_name,
            "wave_index": context.wave_request["wave_index"],
            "request_sha256": context.wave_request["request_sha256"],
            "source_receipt_sha256": source_receipt["receipt_sha256"],
            "cost_reservation_receipt_sha256": reservation["receipt_sha256"],
            "cost_settlement_receipt_sha256": settlement["receipt_sha256"],
            "oauth_preflight_receipt_sha256": preflight["receipt_sha256"],
            "launch_receipt_sha256": launch["receipt_sha256"],
            "poll_receipt_sha256": latest["receipt_sha256"],
            "cleanup_incomplete": cleanup_incomplete,
            "lifecycle_receipt_sha256": lifecycle["receipt_sha256"],
            "receive_receipt_sha256": received["receipt_sha256"],
            "one_wave_only": True,
            "max_concurrent_wave_count": 1,
            "cloud_mutated": True,
            "current_profile_changed": False,
        }
        intent = {
            **intent_core,
            "intent_sha256": canonical_sha256(intent_core),
        }
        _write_once(wave_root / "checkpoint-intent.json", intent)
        _validate_intent_artifacts(wave_root, intent=intent)
        accepted = controller.accept_next_wave(controller_root)
        _write_once(wave_root / "acceptance-receipt.json", accepted)
        checkpoint = _build_checkpoint(intent=intent, accepted=accepted)
        _write_once(wave_root / "SUPERVISOR_CHECKPOINT.json", checkpoint)
        return checkpoint


def run_until_terminal(
    *,
    controller_root: str | Path,
    supervisor_root: str | Path,
    confirm_run_name: str,
    allow_cloud_mutation: bool,
    maximum_waves: int = MAX_SUPERVISED_LIFECYCLES,
    cloud_factory: Callable[[], Any] | None = None,
    max_poll_attempts: int = MAX_POLL_ATTEMPTS,
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if maximum_waves != MAX_SUPERVISED_LIFECYCLES:
        raise PermissionError("M3.1 supervisor lifecycle safety cap changed")
    checkpoints = []
    for _index in range(maximum_waves):
        result = run_next_wave(
            controller_root=controller_root,
            supervisor_root=supervisor_root,
            confirm_run_name=confirm_run_name,
            allow_cloud_mutation=allow_cloud_mutation,
            cloud_factory=cloud_factory,
            max_poll_attempts=max_poll_attempts,
            poll_interval_seconds=poll_interval_seconds,
            sleep=sleep,
        )
        if result.get("schema") == SUPERVISOR_CHECKPOINT_SCHEMA:
            checkpoints.append(result["checkpoint_sha256"])
            if result["next_status"] == "complete":
                break
        else:
            break
    status = controller.controller_status(controller_root)
    return {
        "schema": "hu_m31_t3_dataset_supervisor_run_status_v1",
        "status": status["resume_status"],
        "checkpoint_sha256": checkpoints,
        "checkpoint_count": len(checkpoints),
        "controller_status": status,
        "cost_status": build_cost_status(supervisor_root),
        "max_concurrent_wave_count": 1,
        "current_profile_changed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M3.1 v1 8-VM single-wave dataset supervisor"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    source = sub.add_parser("source-check")
    source.add_argument("--output")
    for name in ("run-next", "run-all"):
        command = sub.add_parser(name)
        command.add_argument("--controller-root", required=True)
        command.add_argument("--supervisor-root", required=True)
        command.add_argument("--confirm-run-name", required=True)
        command.add_argument("--source-binding", required=True)
        command.add_argument("--allow-cloud-mutation", action="store_true")
        command.add_argument(
            "--confirm-total-cost-cap-usd",
            required=True,
        )
        command.add_argument(
            "--confirm-spot-rate-guard-usd-per-vm-hour",
            required=True,
        )
        command.add_argument("--max-poll-attempts", type=int, default=MAX_POLL_ATTEMPTS)
        command.add_argument(
            "--poll-interval-seconds",
            type=int,
            default=DEFAULT_POLL_INTERVAL_SECONDS,
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "source-check":
            result = build_source_receipt()
            if args.output:
                _write_once(Path(args.output), result)
        elif args.command == "run-next":
            _require_cost_confirmation(
                total_cost_cap_usd=args.confirm_total_cost_cap_usd,
                spot_rate_guard_usd_per_vm_hour=(
                    args.confirm_spot_rate_guard_usd_per_vm_hour
                ),
            )
            source_binding.validate_source_binding_file(
                args.source_binding,
                expected_dataset_run_name=args.confirm_run_name,
                expected_controller_root=args.controller_root,
            )
            result = run_next_wave(
                controller_root=args.controller_root,
                supervisor_root=args.supervisor_root,
                confirm_run_name=args.confirm_run_name,
                allow_cloud_mutation=args.allow_cloud_mutation,
                max_poll_attempts=args.max_poll_attempts,
                poll_interval_seconds=args.poll_interval_seconds,
            )
        elif args.command == "run-all":
            _require_cost_confirmation(
                total_cost_cap_usd=args.confirm_total_cost_cap_usd,
                spot_rate_guard_usd_per_vm_hour=(
                    args.confirm_spot_rate_guard_usd_per_vm_hour
                ),
            )
            source_binding.validate_source_binding_file(
                args.source_binding,
                expected_dataset_run_name=args.confirm_run_name,
                expected_controller_root=args.controller_root,
            )
            result = run_until_terminal(
                controller_root=args.controller_root,
                supervisor_root=args.supervisor_root,
                confirm_run_name=args.confirm_run_name,
                allow_cloud_mutation=args.allow_cloud_mutation,
                max_poll_attempts=args.max_poll_attempts,
                poll_interval_seconds=args.poll_interval_seconds,
            )
        else:  # pragma: no cover
            raise RuntimeError("unknown supervisor command")
        print(canonical_bytes(result).decode("ascii"))
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ABORT_CLEANUP_SCHEMA",
    "CHECKPOINT_INTENT_SCHEMA",
    "COMPUTE_COST_CAP_MICRO_USD",
    "CONFIRM_SPOT_RATE_GUARD_USD_PER_VM_HOUR",
    "CONFIRM_TOTAL_COST_CAP_USD",
    "COST_RESERVATION_SCHEMA",
    "COST_SETTLEMENT_SCHEMA",
    "COST_STATUS_SCHEMA",
    "MAX_POLL_ATTEMPTS",
    "MAX_SUPERVISED_LIFECYCLES",
    "MIN_OAUTH_TTL_SECONDS",
    "NON_COMPUTE_RESERVE_MICRO_USD",
    "OAUTH_PREFLIGHT_SCHEMA",
    "PINNED_SOURCE_HASHES",
    "SOURCE_RECEIPT_SCHEMA",
    "SPOT_RATE_GUARD_MICRO_USD_PER_VM_HOUR",
    "SUPERVISOR_CHECKPOINT_SCHEMA",
    "TOTAL_CLOUD_COST_CAP_MICRO_USD",
    "build_oauth_preflight",
    "build_cost_status",
    "build_source_receipt",
    "main",
    "run_next_wave",
    "run_until_terminal",
]
