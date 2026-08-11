"""Fresh Step12n identity and terminal Step12m boundary.

Step12n is the first exact-pair canary that carries the worker host-
prerequisite and serial-diagnostic fixes.  Step12b through Step12m are
terminal and may neither be resumed nor used as an output destination.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12m_fresh_identity_v1
    as step12m_identity,
)


SCHEMA = "hu_m31_t3_step6d_step12n_fresh_identity_v1"
STATUS = (
    "fresh_step12n_identity_disjoint_from_terminal_step12b_through_step12m"
)
TERMINAL_STEP12M_CLOSEOUT_RECEIPT_SHA256 = (
    "5bfbc1543d896ac53c191e4bc70bc6367dce7f6f72d0275ad79f2321f6aa836b"
)
TERMINAL_STEP12M_DEPLOYMENT_CONTRACT_SHA256 = (
    "d981a1715dd24630bb443816b0281d0e85678f98849b7ade401ca514a8469004"
)
TERMINAL_STEP12M_TREE_FILE_COUNT = 17
TERMINAL_STEP12M_TREE_SHA256 = (
    "74ddea87ca837574a8347bc7424d9031c01be98bc20dfebf7029908dddeba92b"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PARENT = REPO_ROOT / "outputs" / "hu_joint_policy" / "m31_t3_step6d"
TERMINAL_STEP12M_OUTPUT_ROOT = step12m_identity.EXPECTED_OUTPUT_ROOT
EXPECTED_OUTPUT_ROOT = ARTIFACT_PARENT / "step12n_pair_v1_actual"

canonical_bytes = step12m_identity.canonical_bytes
canonical_sha256 = step12m_identity.canonical_sha256
seal = step12m_identity.seal

_TERMINAL_ROOTS: tuple[tuple[str, Path], ...] = (
    *step12m_identity._TERMINAL_ROOTS,
    ("step12m", TERMINAL_STEP12M_OUTPUT_ROOT),
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"terminal artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"terminal JSON changed: {path}")
    return value


def exact_output_path(path: str | Path) -> Path:
    root = Path(path).resolve()
    expected = EXPECTED_OUTPUT_ROOT.resolve()
    for _, terminal_root in _TERMINAL_ROOTS:
        terminal = terminal_root.resolve()
        if root == terminal or terminal in root.parents:
            raise PermissionError(
                "only the exact fresh Step12n output root is allowed"
            )
    if root != expected:
        raise PermissionError(
            "only the exact fresh Step12n output root is allowed"
        )
    return root


def validate_prelaunch_freshness(path: str | Path) -> Path:
    root = exact_output_path(path)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Step12n output root is not fresh: {root}")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise FileNotFoundError("Step12n output parent must be a real directory")
    return root


def exact_output_root(path: str | Path) -> Path:
    return validate_prelaunch_freshness(path)


def terminal_tree_snapshot() -> dict[str, Any]:
    return {
        label: step12m_identity._tree_snapshot(root, label=label)
        for label, root in _TERMINAL_ROOTS
    }


def inspect_terminal_step12m() -> dict[str, Any]:
    closeout = step12m_identity._validated_sealed_receipt(
        _read_json(
            TERMINAL_STEP12M_OUTPUT_ROOT
            / "worker_absent_before_done_closeout_receipt.json"
        ),
        expected_sha256=TERMINAL_STEP12M_CLOSEOUT_RECEIPT_SHA256,
        label="terminal Step12m closeout",
    )
    deployment = _read_json(
        TERMINAL_STEP12M_OUTPUT_ROOT / "deployment_contract.json"
    )
    snapshot = step12m_identity._tree_snapshot(
        TERMINAL_STEP12M_OUTPUT_ROOT, label="step12m"
    )
    if (
        closeout.get("step12m_identity_terminal") is not True
        or closeout.get("attempt_index") != 0
        or closeout.get("automatic_retry_performed") is not False
        or closeout.get("third_vm_authorized") is not False
        or closeout.get("current_profile_changed") is not False
        or closeout.get("runner_cleanup_receipt_status")
        != "mandatory_failure_cleanup_verified"
        or closeout.get("deployment_contract_sha256_for_reference")
        != TERMINAL_STEP12M_DEPLOYMENT_CONTRACT_SHA256
        or deployment.get("deployment_contract_sha256")
        != TERMINAL_STEP12M_DEPLOYMENT_CONTRACT_SHA256
        or snapshot
        != {
            "file_count": TERMINAL_STEP12M_TREE_FILE_COUNT,
            "tree_sha256": TERMINAL_STEP12M_TREE_SHA256,
        }
    ):
        raise ValueError("terminal Step12m evidence changed")
    return seal(
        {
            "schema": "hu_m31_t3_step6d_step12m_terminal_inspection_v1",
            "status": "step12m_terminal_failure_and_cleanup_evidence_frozen",
            "closeout_receipt_sha256": closeout["receipt_sha256"],
            "deployment_contract_sha256": (
                TERMINAL_STEP12M_DEPLOYMENT_CONTRACT_SHA256
            ),
            "tree": snapshot,
            "current_profile_changed": False,
        }
    )


def _validate_terminal_closeouts() -> dict[str, Any]:
    step12m_identity._validate_terminal_closeouts()
    return inspect_terminal_step12m()


def _terminal_identity(root: Path, *, label: str) -> dict[str, Any]:
    return step12m_identity._terminal_identity(root, label=label)


def validate_identity_disjointness(
    *,
    deployment_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    output_root: str | Path = EXPECTED_OUTPUT_ROOT,
) -> dict[str, Any]:
    root = exact_output_path(output_root)
    terminal_step12m = _validate_terminal_closeouts()
    terminals = [
        _terminal_identity(terminal_root, label=label)
        for label, terminal_root in _TERMINAL_ROOTS
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
        or not isinstance(source.get("source_prefix"), str)
    ):
        raise ValueError("Step12n exact-pair topology changed")

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
    for field, value in fresh_values.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"Step12n {field} is missing")
        for terminal in terminals:
            if value == terminal[field]:
                raise ValueError(
                    f"Step12n reused terminal {terminal['label']} {field}"
                )

    new_instance_names = [row.get("instance_name") for row in instances]
    old_instance_names: set[Any] = set()
    for terminal in terminals:
        old_instance_names |= terminal["instance_names"]
    if len(set(new_instance_names)) != 2 or any(
        not isinstance(name, str) or name in old_instance_names
        for name in new_instance_names
    ):
        raise ValueError("Step12n reused a terminal VM/disk name")

    return seal(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "terminal_step12m_inspection_receipt_sha256": (
                terminal_step12m["receipt_sha256"]
            ),
            "terminal_step12m_closeout_receipt_sha256": (
                TERMINAL_STEP12M_CLOSEOUT_RECEIPT_SHA256
            ),
            "terminal_step12m_deployment_contract_sha256": (
                TERMINAL_STEP12M_DEPLOYMENT_CONTRACT_SHA256
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
            "worker_diagnostic_contract": {
                "bounded_host_prerequisite_install": True,
                "python3_venv_required": True,
                "sanitized_serial_failure_marker": True,
                "serial_contents_persisted": False,
                "failure_hold_seconds": 600,
                "done_readback_before_success_self_delete": True,
            },
            "terminal_trees": terminal_tree_snapshot(),
            "current_profile_changed": False,
        }
    )


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
    if require_output_absent:
        validate_prelaunch_freshness(output_root)
    else:
        exact_output_path(output_root)
    return validate_identity_disjointness(
        deployment_contract=deployment_contract,
        controller_public_key_record=controller_public_key_record,
        source_plan=source_plan,
        output_root=output_root,
    )


__all__ = [
    "ARTIFACT_PARENT",
    "EXPECTED_OUTPUT_ROOT",
    "SCHEMA",
    "STATUS",
    "TERMINAL_STEP12M_CLOSEOUT_RECEIPT_SHA256",
    "TERMINAL_STEP12M_DEPLOYMENT_CONTRACT_SHA256",
    "TERMINAL_STEP12M_OUTPUT_ROOT",
    "TERMINAL_STEP12M_TREE_FILE_COUNT",
    "TERMINAL_STEP12M_TREE_SHA256",
    "canonical_sha256",
    "exact_output_path",
    "exact_output_root",
    "inspect_terminal_step12m",
    "seal",
    "terminal_tree_snapshot",
    "validate_fresh_identity",
    "validate_identity_disjointness",
    "validate_prelaunch_freshness",
]
