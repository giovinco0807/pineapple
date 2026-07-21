"""Fresh-identity boundary for the post-Step12i exact-pair canary.

Step12b through Step12k consumed cloud identities and audit roots are all
terminal.  This module proves that a newly prepared direct-v2 contract is
disjoint from all eight terminal executions before a live backend can be
constructed.  Step12j runs after the Phase2 install failure was
instrumented to persist the underlying step11 error (code/status/operation)
in a sealed receipt, so its live run is expected to reveal the exact
set/readback cause that Step12i hid.
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
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12d_fresh_identity_v1
    as step12d_identity,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12e_fresh_identity_v1
    as step12e_identity,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12f_fresh_identity_v1
    as step12f_identity,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12g_fresh_identity_v1
    as step12g_identity,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12h_fresh_identity_v1
    as step12h_identity,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12i_fresh_identity_v1
    as step12i_identity,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12j_fresh_identity_v1
    as step12j_identity,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12k_fresh_identity_v1
    as step12k_identity,
)


SCHEMA = "hu_m31_t3_step6d_step12l_fresh_identity_v1"
STATUS = (
    "fresh_step12l_identity_disjoint_from_terminal_step12b_through_step12k"
)
TERMINAL_STEP12K_CLOSEOUT_RECEIPT_SHA256 = (
    "9c9129a4215977960c56fc8bf1e59ce8a5ddc4e8bdd46dd8e56ba0032591cfb3"
)
TERMINAL_STEP12K_DEPLOYMENT_CONTRACT_SHA256 = (
    "f6cc757ef9e9d99cce0458bfc089338dff194064f3b29c915cc3fab3a421a63d"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PARENT = REPO_ROOT / "outputs" / "hu_joint_policy" / "m31_t3_step6d"
TERMINAL_STEP12B_OUTPUT_ROOT = step12c_identity.TERMINAL_OUTPUT_ROOT
TERMINAL_STEP12C_OUTPUT_ROOT = step12d_identity.TERMINAL_STEP12C_OUTPUT_ROOT
TERMINAL_STEP12D_OUTPUT_ROOT = step12e_identity.TERMINAL_STEP12D_OUTPUT_ROOT
TERMINAL_STEP12E_OUTPUT_ROOT = step12f_identity.TERMINAL_STEP12E_OUTPUT_ROOT
TERMINAL_STEP12F_OUTPUT_ROOT = step12g_identity.TERMINAL_STEP12F_OUTPUT_ROOT
TERMINAL_STEP12G_OUTPUT_ROOT = step12h_identity.TERMINAL_STEP12G_OUTPUT_ROOT
TERMINAL_STEP12H_OUTPUT_ROOT = step12i_identity.TERMINAL_STEP12H_OUTPUT_ROOT
TERMINAL_STEP12I_OUTPUT_ROOT = step12j_identity.TERMINAL_STEP12I_OUTPUT_ROOT
TERMINAL_STEP12J_OUTPUT_ROOT = step12k_identity.TERMINAL_STEP12J_OUTPUT_ROOT
TERMINAL_STEP12K_OUTPUT_ROOT = ARTIFACT_PARENT / "step12k_pair_v1_actual"
EXPECTED_OUTPUT_ROOT = ARTIFACT_PARENT / "step12l_pair_v1_actual"

canonical_bytes = step12c_identity.canonical_bytes
canonical_sha256 = step12c_identity.canonical_sha256
seal = step12c_identity.seal

_TERMINAL_ROOTS: tuple[tuple[str, Path], ...] = (
    ("step12b", TERMINAL_STEP12B_OUTPUT_ROOT),
    ("step12c", TERMINAL_STEP12C_OUTPUT_ROOT),
    ("step12d", TERMINAL_STEP12D_OUTPUT_ROOT),
    ("step12e", TERMINAL_STEP12E_OUTPUT_ROOT),
    ("step12f", TERMINAL_STEP12F_OUTPUT_ROOT),
    ("step12g", TERMINAL_STEP12G_OUTPUT_ROOT),
    ("step12h", TERMINAL_STEP12H_OUTPUT_ROOT),
    ("step12i", TERMINAL_STEP12I_OUTPUT_ROOT),
    ("step12j", TERMINAL_STEP12J_OUTPUT_ROOT),
    ("step12k", TERMINAL_STEP12K_OUTPUT_ROOT),
)


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
    for _, terminal_root in _TERMINAL_ROOTS:
        terminal = terminal_root.resolve()
        if root == terminal or terminal in root.parents:
            raise PermissionError(
                "only the exact fresh Step12l output root is allowed"
            )
    if root != expected:
        raise PermissionError(
            "only the exact fresh Step12l output root is allowed"
        )
    return root


def exact_output_root(path: str | Path) -> Path:
    root = exact_output_path(path)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Step12l output root is not fresh: {root}")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise FileNotFoundError("Step12l output parent must be a real directory")
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
        label: _tree_snapshot(root, label=label)
        for label, root in _TERMINAL_ROOTS
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
    # Step12b through Step12j bindings are revalidated through the Step12j
    # boundary module so all ten stay pinned by one call.
    step12k_identity._validate_terminal_closeouts()
    step12k_closeout = _validated_sealed_receipt(
        _read_json(
            TERMINAL_STEP12K_OUTPUT_ROOT
            / "put_method_bug_closeout_receipt.json"
        ),
        expected_sha256=TERMINAL_STEP12K_CLOSEOUT_RECEIPT_SHA256,
        label="terminal Step12k closeout",
    )
    if step12k_closeout.get("step12k_identity_terminal") is not True:
        raise ValueError("terminal Step12k closeout binding changed")
    step12k_deployment = _read_json(
        TERMINAL_STEP12K_OUTPUT_ROOT / "deployment_contract.json"
    )
    if (
        step12k_deployment.get("deployment_contract_sha256")
        != TERMINAL_STEP12K_DEPLOYMENT_CONTRACT_SHA256
    ):
        raise ValueError("terminal Step12k deployment binding changed")


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
        or source.get("source_prefix") is None
    ):
        raise ValueError("Step12l exact-pair topology changed")

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
            raise ValueError(f"Step12l {label} is missing")
        for terminal in terminals:
            if fresh == terminal[label]:
                raise ValueError(
                    f"Step12l reused terminal {terminal['label']} {label}"
                )

    new_instance_names = [row.get("instance_name") for row in instances]
    old_instance_names: set[Any] = set()
    for terminal in terminals:
        old_instance_names |= terminal["instance_names"]
    if len(set(new_instance_names)) != 2 or any(
        not isinstance(name, str) or name in old_instance_names
        for name in new_instance_names
    ):
        raise ValueError("Step12l reused a terminal VM/disk name")

    return seal(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "terminal_step12j_closeout_receipt_sha256": (
                step12k_identity.TERMINAL_STEP12J_CLOSEOUT_RECEIPT_SHA256
            ),
            "terminal_step12k_closeout_receipt_sha256": (
                TERMINAL_STEP12K_CLOSEOUT_RECEIPT_SHA256
            ),
            "terminal_step12k_deployment_contract_sha256": (
                TERMINAL_STEP12K_DEPLOYMENT_CONTRACT_SHA256
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
            "readonly_get_retry_contract": {
                "module": (
                    "hu_m31_t3_step6d_rearm2_diagnostic_"
                    "step12h_readonly_retry_v3"
                ),
                "retryable_code": "iam_https_transport_failed",
                "retry_strategy": "deadline",
                "max_total_retry_seconds": 480,
                "backoff_cap_seconds": 60,
                "mutations_retried": False,
            },
            "sa_create_readback_poll_contract": {
                "module": (
                    "hu_m31_t3_step6d_rearm2_diagnostic_"
                    "step12f_sa_create_poll_v1"
                ),
                "max_readback_attempts": 8,
                "delays_seconds": [2.0, 4.0, 8.0, 8.0, 8.0, 15.0, 15.0],
                "second_create_performed": False,
            },
            "phase2_failure_instrumentation": (
                "underlying_step11_error_code_status_operation_persisted"
            ),
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
    "TERMINAL_STEP12C_OUTPUT_ROOT",
    "TERMINAL_STEP12D_OUTPUT_ROOT",
    "TERMINAL_STEP12E_OUTPUT_ROOT",
    "TERMINAL_STEP12F_OUTPUT_ROOT",
    "TERMINAL_STEP12G_OUTPUT_ROOT",
    "TERMINAL_STEP12H_OUTPUT_ROOT",
    "TERMINAL_STEP12I_OUTPUT_ROOT",
    "TERMINAL_STEP12K_CLOSEOUT_RECEIPT_SHA256",
    "TERMINAL_STEP12K_DEPLOYMENT_CONTRACT_SHA256",
    "TERMINAL_STEP12J_OUTPUT_ROOT",
    "canonical_sha256",
    "exact_output_path",
    "exact_output_root",
    "seal",
    "terminal_tree_snapshot",
    "validate_fresh_identity",
]
