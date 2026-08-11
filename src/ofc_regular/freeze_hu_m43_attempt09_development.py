"""Create the strictly conditional Attempt09 post-development Go freeze.

The freeze can be written only after reopening a canonical Go selector decision,
its single-evaluation receipt, the development receive receipt, and the frozen
Spot package.  It authorizes preparation/authorization of the reserved
future-audit package only.  It never starts audit, fit, threshold selection,
runtime activation, or a ``current`` profile change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m43_attempt09_spot as spot
from .hu_m43_attempt09_contract import M43_ATTEMPT09_PLAN_SHA256, M43_ATTEMPT09_PROFILES
from .select_hu_m43_attempt09_development import (
    ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA,
    ATTEMPT09_DEVELOPMENT_RECEIPT_SCHEMA,
)


ATTEMPT09_DEVELOPMENT_GO_FREEZE_SCHEMA = "hu_m43_attempt09_development_go_freeze_v1"
DEVELOPMENT_GO_FREEZE_STATUS = "go_freeze_attempt09_development"
SELECTOR_SOURCE_NAME = "select_hu_m43_attempt09_development.py"
_HEX = frozenset("0123456789abcdef")
_DECISION_KEYS = {
    "schema", "status", "decision", "search_freeze_authorized", "selected_arm",
    "selected_threshold", "source", "development_population", "metrics", "gates",
    "decision_contract", "integrity", "science_boundary",
}
_RECEIPT_KEYS = {
    "schema", "status", "run_name", "decision_sha256", "decision",
    "search_freeze_authorized", "gate_evaluation_count", "selector_executed",
    "future_audit_authorized", "fit_performed", "threshold_selected",
    "runtime_policy_activated", "current_profile_mutated",
}
_RECEIVE_KEYS = {
    "schema", "status", "run_name", "mode", "roots", "root_indices", "profiles",
    "manifest_sha256", "schedule_sha256", "source_sha256", "authorization_sha256",
    "merged_sha256", "audit_sha256", "batch_boundary_validation_count",
    "per_shard_boundary_revalidation_count", "selector_executed",
    "current_profile_mutated", "runtime_policy_activated",
}
_SOURCE_KEYS = {
    "input_jsonl_sha256", "plan_sha256", "authorization_sha256",
    "source_package_sha256", "run_name", "selector_source_sha256",
    "root_identity_sha256",
}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid UTF-8 JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Attempt09 development freeze already exists: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(
                f"Attempt09 development freeze already exists: {path}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _validate_selector(decision_path: Path, receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    decision = _load_canonical(decision_path, "Attempt09 selector decision")
    receipt = _load_canonical(receipt_path, "Attempt09 selector receipt")
    if (
        set(decision) != _DECISION_KEYS
        or decision.get("schema") != ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA
        or decision.get("status") != "go_write_separate_search_freeze_only"
        or decision.get("decision") != "go"
        or decision.get("search_freeze_authorized") is not True
        or decision.get("selected_arm") is not None
        or decision.get("selected_threshold") is not None
    ):
        raise ValueError("Attempt09 selector decision is not the frozen Go boundary")
    gates = decision.get("gates")
    if (
        not isinstance(gates, list)
        or not gates
        or any(not isinstance(gate, Mapping) or gate.get("passed") is not True for gate in gates)
    ):
        raise ValueError("Attempt09 selector Go does not have every gate passed")
    contract = _mapping(decision.get("decision_contract"), "decision contract")
    if (
        contract.get("gate_evaluation_count") != 1
        or contract.get("single_frozen_search_architecture") is not True
        or contract.get("arm_selection_performed") is not False
        or contract.get("threshold_selection_performed") is not False
        or contract.get("all_gates_required") is not True
    ):
        raise ValueError("Attempt09 selector decision contract changed")
    science = _mapping(decision.get("science_boundary"), "science boundary")
    for key in (
        "teacher_values_are_realized_match_ev", "future_audit_authorized",
        "fit_performed", "threshold_selected", "runtime_policy_activated",
        "current_profile_mutated", "full_replacement_enabled",
    ):
        if science.get(key) is not False:
            raise ValueError(f"Attempt09 selector science boundary opened {key}")
    if (
        set(receipt) != _RECEIPT_KEYS
        or receipt.get("schema") != ATTEMPT09_DEVELOPMENT_RECEIPT_SCHEMA
        or receipt.get("status") != "single_frozen_gate_evaluation_complete"
        or receipt.get("decision_sha256") != _sha256(decision_path)
        or receipt.get("decision") != "go"
        or receipt.get("search_freeze_authorized") is not True
        or receipt.get("gate_evaluation_count") != 1
        or receipt.get("selector_executed") is not True
    ):
        raise ValueError("Attempt09 selector receipt is not a single Go evaluation")
    for key in (
        "future_audit_authorized", "fit_performed", "threshold_selected",
        "runtime_policy_activated", "current_profile_mutated",
    ):
        if receipt.get(key) is not False:
            raise ValueError(f"Attempt09 selector receipt opened {key}")
    return decision, receipt


def build_attempt09_development_go_freeze(
    *,
    run_dir: str | Path,
    received_dir: str | Path,
    decision_path: str | Path,
    decision_receipt_path: str | Path,
) -> dict[str, Any]:
    """Reopen every Go dependency and return an unwritten canonical payload."""

    run = Path(run_dir).resolve()
    received = Path(received_dir).resolve()
    decision_file = Path(decision_path).resolve()
    selector_receipt_file = Path(decision_receipt_path).resolve()
    decision, selector_receipt = _validate_selector(decision_file, selector_receipt_file)

    manifest, _launch = spot.validate_launch(run)
    if manifest.get("mode") != "development" or manifest.get("total_shards") != 200:
        raise ValueError("Attempt09 Go freeze requires the development200 package")
    manifest_file = run / "manifest.json"
    execution_file = run / "execution_authorization.json"
    if not execution_file.is_file():
        raise ValueError("Attempt09 development execution authorization is missing")
    manifest_sha = _sha256(manifest_file)
    execution_sha = _sha256(execution_file)

    receive_file = received / "merged" / "receive_receipt.json"
    merged_file = received / "merged" / "teacher.jsonl"
    receive = _load_canonical(receive_file, "Attempt09 development receive receipt")
    expected_indices = list(range(200))
    expected_profiles = [
        M43_ATTEMPT09_PROFILES[index % len(M43_ATTEMPT09_PROFILES)]
        for index in expected_indices
    ]
    if (
        set(receive) != _RECEIVE_KEYS
        or receive.get("schema") != spot.RECEIVE_SCHEMA
        or receive.get("status") != "complete"
        or receive.get("run_name") != manifest.get("run_name")
        or receive.get("mode") != "development"
        or receive.get("roots") != 200
        or receive.get("root_indices") != expected_indices
        or receive.get("profiles") != expected_profiles
        or receive.get("manifest_sha256") != manifest_sha
        or receive.get("schedule_sha256") != manifest.get("schedule_sha256")
        or receive.get("source_sha256") != manifest.get("source_sha256")
        or receive.get("authorization_sha256") != execution_sha
        or receive.get("batch_boundary_validation_count") != 1
        or receive.get("per_shard_boundary_revalidation_count") != 0
        or receive.get("selector_executed") is not False
        or receive.get("current_profile_mutated") is not False
        or receive.get("runtime_policy_activated") is not False
        or not _is_sha256(receive.get("audit_sha256"))
    ):
        raise ValueError("Attempt09 development receive boundary changed")
    if not merged_file.is_file() or receive.get("merged_sha256") != _sha256(merged_file):
        raise ValueError("Attempt09 development merged input hash changed")

    source = _mapping(decision.get("source"), "selector source bindings")
    selector_source = Path(__file__).with_name(SELECTOR_SOURCE_NAME)
    if (
        set(source) != _SOURCE_KEYS
        or source.get("run_name") != manifest.get("run_name")
        or selector_receipt.get("run_name") != manifest.get("run_name")
        or source.get("plan_sha256") != M43_ATTEMPT09_PLAN_SHA256
        or source.get("plan_sha256") != manifest.get("plan_sha256")
        or source.get("authorization_sha256") != execution_sha
        or source.get("source_package_sha256") != manifest.get("source_sha256")
        or source.get("input_jsonl_sha256") != receive.get("merged_sha256")
        or source.get("selector_source_sha256") != _sha256(selector_source)
        or not _is_sha256(source.get("root_identity_sha256"))
    ):
        raise ValueError("Attempt09 selector/source hash closure changed")

    bindings = {
        "plan_sha256": M43_ATTEMPT09_PLAN_SHA256,
        "manifest_sha256": manifest_sha,
        "execution_authorization_sha256": execution_sha,
        "source_package_sha256": manifest["source_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "merged_input_sha256": receive["merged_sha256"],
        "receive_receipt_sha256": _sha256(receive_file),
        "selector_decision_sha256": _sha256(decision_file),
        "selector_receipt_sha256": _sha256(selector_receipt_file),
        "selector_source_sha256": source["selector_source_sha256"],
        "root_identity_sha256": source["root_identity_sha256"],
    }
    if not all(_is_sha256(value) for value in bindings.values()):
        raise ValueError("Attempt09 Go freeze contains an invalid source hash")
    return {
        "schema": ATTEMPT09_DEVELOPMENT_GO_FREEZE_SCHEMA,
        "status": DEVELOPMENT_GO_FREEZE_STATUS,
        "decision": "go",
        "run_name": manifest["run_name"],
        "bindings": bindings,
        "authorization_scope": {
            "future_audit_package_authorized": True,
            "future_audit_authorization_artifact_authorized": True,
            "future_audit_launch_authorized": False,
            "future_audit_started": False,
            "fit_authorized": False,
            "fit_started": False,
            "threshold_selection_authorized": False,
            "threshold_selection_started": False,
            "runtime_policy_activated": False,
            "current_profile_mutated": False,
            "full_replacement_enabled": False,
        },
        "future_audit_authorized": False,
        "fit_performed": False,
        "threshold_selected": False,
        "runtime_policy_activated": False,
        "current_profile_mutated": False,
        "full_replacement_enabled": False,
    }


def freeze_attempt09_development(
    *,
    run_dir: str | Path,
    received_dir: str | Path,
    decision_path: str | Path,
    decision_receipt_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Validate the Go closure and immutably write one canonical freeze."""

    payload = build_attempt09_development_go_freeze(
        run_dir=run_dir,
        received_dir=received_dir,
        decision_path=decision_path,
        decision_receipt_path=decision_receipt_path,
    )
    _write_once(Path(output), _canonical_bytes(payload))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the conditional Attempt09 development Go freeze"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--received-dir", required=True, type=Path)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--decision-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = freeze_attempt09_development(
        run_dir=args.run_dir,
        received_dir=args.received_dir,
        decision_path=args.decision,
        decision_receipt_path=args.decision_receipt,
        output=args.output,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
