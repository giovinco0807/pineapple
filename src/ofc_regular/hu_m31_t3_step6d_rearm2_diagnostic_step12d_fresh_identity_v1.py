"""Fresh-identity boundary for the post-Step12c exact-pair canary.

Step12b's and Step12c's consumed cloud identities and audit roots are both
terminal.  This module does not create a new transport protocol; it proves
that a newly prepared direct-v2 contract is disjoint from BOTH terminal
executions before a live backend can be constructed.  Pre-registered in
docs/hu_joint_policy_m31_t3_step12d_entrypoint_decision_20260721.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12c_fresh_identity_v1
    as step12c_identity,
)


SCHEMA = "hu_m31_t3_step6d_step12d_fresh_identity_v1"
STATUS = "fresh_step12d_identity_disjoint_from_terminal_step12b_and_step12c"
TERMINAL_STEP12C_CLOSEOUT_RECEIPT_SHA256 = (
    "31e675593ca168fc899a6dc34083a6f54c8258aabbb62dcd71906d56d77161ef"
)
TERMINAL_STEP12C_DEPLOYMENT_CONTRACT_SHA256 = (
    "37c23ffb1dbeca76b026ab580bb51702f952dbecc7959fcf7ef16fd1b65488be"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PARENT = REPO_ROOT / "outputs" / "hu_joint_policy" / "m31_t3_step6d"
TERMINAL_STEP12B_OUTPUT_ROOT = step12c_identity.TERMINAL_OUTPUT_ROOT
TERMINAL_STEP12C_OUTPUT_ROOT = ARTIFACT_PARENT / "step12c_pair_v1_actual"
EXPECTED_OUTPUT_ROOT = ARTIFACT_PARENT / "step12d_pair_v1_actual"

canonical_bytes = step12c_identity.canonical_bytes
canonical_sha256 = step12c_identity.canonical_sha256
seal = step12c_identity.seal


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"terminal artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"terminal JSON changed: {path}")
    return value


def _validated_sealed_receipt(
    value: Mapping[str, Any], *, expected_sha256: str, label: str
) -> dict[str, Any]:
    checked = dict(value)
    supplied = checked.pop("receipt_sha256", None)
    if (
        supplied != expected_sha256
        or canonical_sha256(checked) != expected_sha256
    ):
        raise ValueError(f"{label} receipt changed")
    return {**checked, "receipt_sha256": supplied}


def exact_output_path(path: str | Path) -> Path:
    root = Path(path).resolve()
    expected = EXPECTED_OUTPUT_ROOT.resolve()
    for terminal in (
        TERMINAL_STEP12B_OUTPUT_ROOT.resolve(),
        TERMINAL_STEP12C_OUTPUT_ROOT.resolve(),
    ):
        if root == terminal or terminal in root.parents:
            raise PermissionError(
                "only the exact fresh Step12d output root is allowed"
            )
    if root != expected:
        raise PermissionError(
            "only the exact fresh Step12d output root is allowed"
        )
    return root


def exact_output_root(path: str | Path) -> Path:
    root = exact_output_path(path)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Step12d output root is not fresh: {root}")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise FileNotFoundError("Step12d output parent must be a real directory")
    return root


def _tree_snapshot(root: Path, *, label: str) -> dict[str, Any]:
    resolved = root.resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise FileNotFoundError(f"terminal {label} output root changed")
    records: list[dict[str, Any]] = []
    for path in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"terminal {label} output contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"terminal {label} output contains a non-file")
        records.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not records:
        raise ValueError(f"terminal {label} output tree is empty")
    return {
        "file_count": len(records),
        "tree_sha256": canonical_sha256(records),
    }


def terminal_tree_snapshot() -> dict[str, Any]:
    return {
        "step12b": _tree_snapshot(
            TERMINAL_STEP12B_OUTPUT_ROOT, label="Step12b"
        ),
        "step12c": _tree_snapshot(
            TERMINAL_STEP12C_OUTPUT_ROOT, label="Step12c"
        ),
    }


def _terminal_identity(root: Path, *, label: str) -> dict[str, Any]:
    deployment = _read_json(root / "deployment_contract.json")
    public_key = _read_json(root / "controller_public_key.json")
    sources = _read_json(root / "bootstrap_source_provision_receipt.json")
    controller = deployment.get("controller_service_account")
    remote_layout = deployment.get("remote_layout")
    return {
        "label": label,
        "controller_key_id": public_key.get("key_id"),
        "run_nonce": deployment.get("run_nonce"),
        "deployment_contract_sha256": deployment.get(
            "deployment_contract_sha256"
        ),
        "run_identity_sha256": deployment.get("run_identity_sha256"),
        "direct_stage_identity_sha256": deployment.get(
            "direct_stage_identity_sha256"
        ),
        "run_name": deployment.get("run_name"),
        "controller_service_account": (
            controller.get("email") if isinstance(controller, Mapping) else None
        ),
        "stage_prefix": (
            remote_layout.get("stage_prefix")
            if isinstance(remote_layout, Mapping)
            else None
        ),
        "source_prefix": sources.get("source_prefix"),
        "instance_names": {
            row.get("instance_name")
            for row in deployment.get("instances", [])
            if isinstance(row, Mapping)
        },
    }


def _validate_terminal_closeouts() -> None:
    step12b_closeout = _validated_sealed_receipt(
        _read_json(
            TERMINAL_STEP12B_OUTPUT_ROOT
            / "token_barrier_failure_closeout_receipt.json"
        ),
        expected_sha256=step12c_identity.TERMINAL_CLOSEOUT_RECEIPT_SHA256,
        label="terminal Step12b closeout",
    )
    step12c_closeout = _validated_sealed_receipt(
        _read_json(
            TERMINAL_STEP12C_OUTPUT_ROOT
            / "phase2_schema_failure_closeout_receipt.json"
        ),
        expected_sha256=TERMINAL_STEP12C_CLOSEOUT_RECEIPT_SHA256,
        label="terminal Step12c closeout",
    )
    step12b_deployment = _read_json(
        TERMINAL_STEP12B_OUTPUT_ROOT / "deployment_contract.json"
    )
    step12c_deployment = _read_json(
        TERMINAL_STEP12C_OUTPUT_ROOT / "deployment_contract.json"
    )
    if (
        step12b_deployment.get("deployment_contract_sha256")
        != step12c_identity.TERMINAL_DEPLOYMENT_CONTRACT_SHA256
        or step12b_closeout.get("deployment_contract_sha256")
        != step12c_identity.TERMINAL_DEPLOYMENT_CONTRACT_SHA256
    ):
        raise ValueError("terminal Step12b deployment binding changed")
    if (
        step12c_deployment.get("deployment_contract_sha256")
        != TERMINAL_STEP12C_DEPLOYMENT_CONTRACT_SHA256
        or step12c_closeout.get("deployment_contract_sha256")
        != TERMINAL_STEP12C_DEPLOYMENT_CONTRACT_SHA256
    ):
        raise ValueError("terminal Step12c deployment binding changed")


def validate_fresh_identity(
    *,
    deployment_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    output_root: str | Path = EXPECTED_OUTPUT_ROOT,
    require_output_absent: bool = True,
) -> dict[str, Any]:
    if type(require_output_absent) is not bool:
        raise TypeError("require_output_absent must be bool")
    root = (
        exact_output_root(output_root)
        if require_output_absent
        else exact_output_path(output_root)
    )
    _validate_terminal_closeouts()
    terminals = [
        _terminal_identity(TERMINAL_STEP12B_OUTPUT_ROOT, label="step12b"),
        _terminal_identity(TERMINAL_STEP12C_OUTPUT_ROOT, label="step12c"),
    ]

    deployment = dict(deployment_contract)
    public_key = dict(controller_public_key_record)
    source = dict(source_plan)
    instances = deployment.get("instances")
    controller = deployment.get("controller_service_account")
    remote_layout = deployment.get("remote_layout")
    if (
        deployment.get("attempt_index") != 0
        or deployment.get("vm_count") != 2
        or not isinstance(instances, list)
        or len(instances) != 2
        or not isinstance(controller, Mapping)
        or not isinstance(remote_layout, Mapping)
        or source.get("source_prefix") is None
    ):
        raise ValueError("Step12d exact-pair topology changed")

    fresh_values = {
        "controller_key_id": public_key.get("key_id"),
        "run_nonce": deployment.get("run_nonce"),
        "deployment_contract_sha256": deployment.get(
            "deployment_contract_sha256"
        ),
        "run_identity_sha256": deployment.get("run_identity_sha256"),
        "direct_stage_identity_sha256": deployment.get(
            "direct_stage_identity_sha256"
        ),
        "run_name": deployment.get("run_name"),
        "controller_service_account": controller.get("email"),
        "stage_prefix": remote_layout.get("stage_prefix"),
        "source_prefix": source.get("source_prefix"),
    }
    for label, fresh in fresh_values.items():
        if not isinstance(fresh, str) or not fresh:
            raise ValueError(f"Step12d {label} is missing")
        for terminal in terminals:
            if fresh == terminal[label]:
                raise ValueError(
                    f"Step12d reused terminal {terminal['label']} {label}"
                )

    new_instance_names = [row.get("instance_name") for row in instances]
    old_instance_names: set[Any] = set()
    for terminal in terminals:
        old_instance_names |= terminal["instance_names"]
    if len(set(new_instance_names)) != 2 or any(
        not isinstance(name, str) or name in old_instance_names
        for name in new_instance_names
    ):
        raise ValueError("Step12d reused a terminal VM/disk name")

    return seal(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "terminal_step12b_closeout_receipt_sha256": (
                step12c_identity.TERMINAL_CLOSEOUT_RECEIPT_SHA256
            ),
            "terminal_step12b_deployment_contract_sha256": (
                step12c_identity.TERMINAL_DEPLOYMENT_CONTRACT_SHA256
            ),
            "terminal_step12c_closeout_receipt_sha256": (
                TERMINAL_STEP12C_CLOSEOUT_RECEIPT_SHA256
            ),
            "terminal_step12c_deployment_contract_sha256": (
                TERMINAL_STEP12C_DEPLOYMENT_CONTRACT_SHA256
            ),
            "deployment_contract_sha256": deployment[
                "deployment_contract_sha256"
            ],
            "run_identity_sha256": deployment["run_identity_sha256"],
            "direct_stage_identity_sha256": deployment[
                "direct_stage_identity_sha256"
            ],
            "source_prefix": source["source_prefix"],
            "output_root": str(root),
            "controller_service_account": controller["email"],
            "instance_names": new_instance_names,
            "disk_names": new_instance_names,
            "attempt_index": 0,
            "attempt1_authorized": False,
            "third_vm_authorized": False,
            "automatic_retry_authorized": False,
            "terminal_trees": terminal_tree_snapshot(),
            "current_profile_changed": False,
        }
    )


__all__ = [
    "ARTIFACT_PARENT",
    "EXPECTED_OUTPUT_ROOT",
    "SCHEMA",
    "STATUS",
    "TERMINAL_STEP12B_OUTPUT_ROOT",
    "TERMINAL_STEP12C_CLOSEOUT_RECEIPT_SHA256",
    "TERMINAL_STEP12C_DEPLOYMENT_CONTRACT_SHA256",
    "TERMINAL_STEP12C_OUTPUT_ROOT",
    "canonical_sha256",
    "exact_output_path",
    "exact_output_root",
    "seal",
    "terminal_tree_snapshot",
    "validate_fresh_identity",
]
