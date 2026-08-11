"""Fresh-identity boundary for the post-Step12b exact-pair canary.

Step12b's consumed cloud identity and audit root are terminal.  This module
does not create a new transport protocol; it proves that a newly prepared
direct-v2 contract is disjoint from that terminal execution before a live
backend can be constructed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "hu_m31_t3_step6d_step12c_fresh_identity_v1"
STATUS = "fresh_step12c_identity_disjoint_from_terminal_step12b"
TERMINAL_CLOSEOUT_RECEIPT_SHA256 = (
    "c66e75f8dfbc877e7602467231d361054bc382161e724ecd3e5acda88696b2b0"
)
TERMINAL_DEPLOYMENT_CONTRACT_SHA256 = (
    "eba36e0f5510dcf51443dfa8219fe0fdff0b4d7429dc755e402cc799f13d9293"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PARENT = (
    REPO_ROOT / "outputs" / "hu_joint_policy" / "m31_t3_step6d"
)
TERMINAL_OUTPUT_ROOT = ARTIFACT_PARENT / "step12b_pair_v2_actual"
EXPECTED_OUTPUT_ROOT = ARTIFACT_PARENT / "step12c_pair_v1_actual"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(body)
    return {**copied, "receipt_sha256": canonical_sha256(copied)}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"terminal Step12b artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"terminal Step12b JSON changed: {path}")
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
    terminal = TERMINAL_OUTPUT_ROOT.resolve()
    if root != expected or root == terminal or terminal in root.parents:
        raise PermissionError("only the exact fresh Step12c output root is allowed")
    return root


def exact_output_root(path: str | Path) -> Path:
    root = exact_output_path(path)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Step12c output root is not fresh: {root}")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise FileNotFoundError("Step12c output parent must be a real directory")
    return root


def terminal_tree_snapshot() -> dict[str, Any]:
    root = TERMINAL_OUTPUT_ROOT.resolve()
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError("terminal Step12b output root changed")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("terminal Step12b output contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("terminal Step12b output contains a non-file")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not records:
        raise ValueError("terminal Step12b output tree is empty")
    return {
        "file_count": len(records),
        "tree_sha256": canonical_sha256(records),
    }


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
    terminal_deployment = _read_json(
        TERMINAL_OUTPUT_ROOT / "deployment_contract.json"
    )
    terminal_public_key = _read_json(
        TERMINAL_OUTPUT_ROOT / "controller_public_key.json"
    )
    terminal_sources = _read_json(
        TERMINAL_OUTPUT_ROOT / "bootstrap_source_provision_receipt.json"
    )
    terminal_closeout = _validated_sealed_receipt(
        _read_json(
            TERMINAL_OUTPUT_ROOT
            / "token_barrier_failure_closeout_receipt.json"
        ),
        expected_sha256=TERMINAL_CLOSEOUT_RECEIPT_SHA256,
        label="terminal Step12b closeout",
    )
    if (
        terminal_deployment.get("deployment_contract_sha256")
        != TERMINAL_DEPLOYMENT_CONTRACT_SHA256
        or terminal_closeout.get("deployment_contract_sha256")
        != TERMINAL_DEPLOYMENT_CONTRACT_SHA256
    ):
        raise ValueError("terminal Step12b deployment binding changed")

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
        raise ValueError("Step12c exact-pair topology changed")

    old_instance_names = {
        row.get("instance_name")
        for row in terminal_deployment.get("instances", [])
        if isinstance(row, Mapping)
    }
    new_instance_names = [row.get("instance_name") for row in instances]
    disjoint_pairs = {
        "controller_key_id": (
            public_key.get("key_id"), terminal_public_key.get("key_id")
        ),
        "run_nonce": (
            deployment.get("run_nonce"), terminal_deployment.get("run_nonce")
        ),
        "deployment_contract_sha256": (
            deployment.get("deployment_contract_sha256"),
            terminal_deployment.get("deployment_contract_sha256"),
        ),
        "run_identity_sha256": (
            deployment.get("run_identity_sha256"),
            terminal_deployment.get("run_identity_sha256"),
        ),
        "direct_stage_identity_sha256": (
            deployment.get("direct_stage_identity_sha256"),
            terminal_deployment.get("direct_stage_identity_sha256"),
        ),
        "run_name": (
            deployment.get("run_name"), terminal_deployment.get("run_name")
        ),
        "controller_service_account": (
            controller.get("email"),
            terminal_deployment.get("controller_service_account", {}).get(
                "email"
            ),
        ),
        "stage_prefix": (
            remote_layout.get("stage_prefix"),
            terminal_deployment.get("remote_layout", {}).get("stage_prefix"),
        ),
        "source_prefix": (
            source.get("source_prefix"), terminal_sources.get("source_prefix")
        ),
    }
    for label, (fresh, terminal) in disjoint_pairs.items():
        if not isinstance(fresh, str) or not fresh or fresh == terminal:
            raise ValueError(f"Step12c reused terminal {label}")
    if (
        len(set(new_instance_names)) != 2
        or any(
            not isinstance(name, str) or name in old_instance_names
            for name in new_instance_names
        )
    ):
        raise ValueError("Step12c reused a terminal VM/disk name")

    return seal(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "terminal_closeout_receipt_sha256": (
                TERMINAL_CLOSEOUT_RECEIPT_SHA256
            ),
            "terminal_deployment_contract_sha256": (
                TERMINAL_DEPLOYMENT_CONTRACT_SHA256
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
            "terminal_tree": terminal_tree_snapshot(),
            "current_profile_changed": False,
        }
    )


__all__ = [
    "EXPECTED_OUTPUT_ROOT",
    "SCHEMA",
    "STATUS",
    "TERMINAL_CLOSEOUT_RECEIPT_SHA256",
    "TERMINAL_DEPLOYMENT_CONTRACT_SHA256",
    "TERMINAL_OUTPUT_ROOT",
    "canonical_sha256",
    "exact_output_path",
    "exact_output_root",
    "seal",
    "terminal_tree_snapshot",
    "validate_fresh_identity",
]
