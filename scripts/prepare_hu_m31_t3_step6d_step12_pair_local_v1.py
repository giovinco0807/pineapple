#!/usr/bin/env python3
"""Build and validate the fresh local Step 12 pair execution inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ofc_regular import (  # noqa: E402
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter
    as adapter,
)
from ofc_regular import (  # noqa: E402
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from ofc_regular import (  # noqa: E402
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan,
)
from ofc_regular import (  # noqa: E402
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1
    as controller,
)
from ofc_regular import (  # noqa: E402
    hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_cloud_controller_v1
    as pair_cloud,
)
from ofc_regular import (  # noqa: E402
    hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_contract_v1
    as pair_contract_module,
)
from ofc_regular import (  # noqa: E402
    hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_iam_capacity_gate_v1
    as pair_gate,
)


STEP11_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step11_one_vm_v12_fix3_actual"
)
PACKAGE_DIR = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "rearm2_diagnostic_cloud_worker_package_v1"
)
STARTUP_PATH = (
    REPO_ROOT
    / "scripts"
    / "bootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.sh"
)
PREBOOTSTRAP_PATH = (
    REPO_ROOT
    / "scripts"
    / "prebootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_v1.py"
)
BASE_PREBOOTSTRAP_PATH = (
    REPO_ROOT
    / "scripts"
    / "prebootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.py"
)
WHEEL_PATH = (
    STEP11_ROOT
    / "local_preflight"
    / "outer"
    / "wheels"
    / "numpy-2.2.6-cp311-cp311-manylinux_2_17_x86_64."
    "manylinux2014_x86_64.whl"
)
POLICY_REGISTRY_PATH = REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"
NAT_ROUTER_RESOURCE = (
    "projects/ofc-solver-485418/regions/asia-northeast1/"
    "routers/ofc-t3-nat-router-asia-northeast1"
)
EXPECTED_PROFILE_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
EXPECTED_BASE_PREBOOTSTRAP_SHA256 = (
    "9e58d367de175f3fef111f6f89a52f5f0dbf57b37a47e57d6e1c8af6324f0b93"
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required input is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required input is not an object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    controller.exclusive_write_json(path, value)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_local_bundle(
    *,
    output_root: Path,
    issued_at_unix_seconds: int,
    expires_at_unix_seconds: int,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Step12 output root is not fresh: {output_root}")
    if _file_sha(POLICY_REGISTRY_PATH) != EXPECTED_PROFILE_SHA256:
        raise RuntimeError("current/profile registry hash changed")
    if (
        _file_sha(BASE_PREBOOTSTRAP_PATH)
        != EXPECTED_BASE_PREBOOTSTRAP_SHA256
    ):
        raise RuntimeError("Step12 frozen prebootstrap base changed")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()

    stage1 = _read_json(STEP11_ROOT / "transport_contract.json")
    stage1_done = _read_json(STEP11_ROOT / "late_done_envelope.json")
    recovery = _read_json(STEP11_ROOT / "late_done_recovery_receipt.json")
    stage1_receive = adapter.build_receive(
        stage1["adapter_preview"], done_records=[stage1_done]
    )
    if controller.canonical_sha256(stage1_receive) != recovery["receive_sha256"]:
        raise RuntimeError("Step11 recovery receive binding changed")
    wheel = next(
        row
        for row in stage1["outer_package_manifest"]["objects"]
        if row["kind"] == "offline_numpy_cp311_manylinux_x86_64_wheel"
    )
    signer = controller.generate_ephemeral_controller_key(key_size=3_072)
    contracts = [
        transport.build_job_contract(
            package_dir=PACKAGE_DIR,
            stage_id=plan.STAGE2_ID,
            job_id=job_id,
            attempt_index=0,
            offline_wheel_record=wheel,
            controller_public_key_record=signer.public_record,
            prerequisite_stage1_preview=stage1["adapter_preview"],
            prerequisite_stage1_receive=stage1_receive,
        )
        for job_id in plan.STAGE2_JOB_IDS
    ]
    candidate, reference = contracts
    pair = pair_contract_module.build_pair_contract(
        candidate,
        reference,
        recovery,
    )
    pair = pair_contract_module.validate_pair_contract(
        pair,
        candidate_transport_contract=candidate,
        reference_transport_contract=reference,
        stage1_recovery_receipt=recovery,
    )
    gate = pair_gate.build_step12_pair_gate_plan(
        pair,
        candidate_transport_contract=candidate,
        reference_transport_contract=reference,
        stage1_recovery_receipt=recovery,
        issued_at_unix_seconds=issued_at_unix_seconds,
        expires_at_unix_seconds=expires_at_unix_seconds,
        nat_router_resource=NAT_ROUTER_RESOURCE,
    )
    smoke_authorizations = [
        controller.build_controller_authorization(
            contract=contract,
            external_preflight_receipt_sha256=recovery["receipt_sha256"],
            issued_unix_seconds=issued_at_unix_seconds,
            expires_unix_seconds=expires_at_unix_seconds,
            signer=signer,
        )
        for contract in contracts
    ]
    insert_bodies = [
        pair_cloud.build_insert_body(
            transport_contract=contract,
            authorization=authorization,
            public_key_record=signer.public_record,
            startup_path=STARTUP_PATH,
            prebootstrap_path=PREBOOTSTRAP_PATH,
            base_prebootstrap_path=BASE_PREBOOTSTRAP_PATH,
        )
        for contract, authorization in zip(
            contracts, smoke_authorizations, strict=True
        )
    ]
    if (
        [row["name"] for row in insert_bodies]
        != [row["instance_name"] for row in pair["instances"]]
        or any(
            not row["machineType"].endswith(
                f"/machineTypes/{pair_cloud.ACTUAL_MACHINE_TYPE}"
            )
            or row["networkInterfaces"][0]["accessConfigs"] != []
            for row in insert_bodies
        )
    ):
        raise RuntimeError("Step12 concrete insert-body smoke changed")

    _write(output_root / "controller_public_key.json", signer.public_record)
    _write(output_root / "candidate_transport_contract.json", candidate)
    _write(output_root / "reference_transport_contract.json", reference)
    _write(output_root / "pair_contract.json", pair)
    _write(output_root / "iam_capacity_gate_plan.json", gate)
    _write(output_root / "stage1_receive.json", stage1_receive)
    _write(output_root / "stage1_recovery_receipt.json", recovery)

    local_preflight_receipts = []
    for job_id, contract in zip(plan.STAGE2_JOB_IDS, contracts, strict=True):
        destination = output_root / f"local_preflight_{job_id}"
        local_preflight_receipts.append(
            transport.local_preflight(
                contract=contract,
                package_mirror=PACKAGE_DIR,
                fresh_work_root=destination,
                offline_wheel_mirror=WHEEL_PATH,
                offline_install_smoke=False,
            )
        )
    body = {
        "schema": (
            "hu_m31_t3_step6d_rearm2_diagnostic_step12_"
            "pair_local_preflight_receipt_v1"
        ),
        "status": "local_step12_pair_ready_cloud_still_unauthorized",
        "created_at_unix_seconds": int(time.time()),
        "pair_contract_sha256": pair["pair_contract_sha256"],
        "gate_plan_sha256": gate["plan_sha256"],
        "transport_contract_sha256s": [
            transport.canonical_sha256(row) for row in contracts
        ],
        "controller_public_key_sha256": transport.canonical_sha256(
            signer.public_record
        ),
        "local_preflight_receipt_sha256s": [
            transport.canonical_sha256(row)
            for row in local_preflight_receipts
        ],
        "insert_body_sha256s": [
            controller.canonical_sha256(row) for row in insert_bodies
        ],
        "insert_body_smoke_passed": True,
        "actual_machine_type": pair_cloud.ACTUAL_MACHINE_TYPE,
        "inner_machine_type": pair_cloud.INNER_MACHINE_TYPE,
        "inner_rayon_threads": pair_cloud.INNER_RAYON_THREADS,
        "vm_count": 2,
        "attempt_index": 0,
        "cloud_launch_authorized": False,
        "vm_created": False,
        "iam_changed": False,
        "network_operation_performed": False,
        "scientific_payload_present": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_sha256": _file_sha(POLICY_REGISTRY_PATH),
        "current_profile_changed": False,
    }
    receipt = {**body, "receipt_sha256": controller.canonical_sha256(body)}
    _write(output_root / "LOCAL_STEP12_PAIR_READY.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--issued-at-unix-seconds", type=int, default=int(time.time())
    )
    parser.add_argument("--window-seconds", type=int, default=7_200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 600 <= args.window_seconds <= 7_200:
        raise ValueError("Step12 authorization window changed")
    receipt = build_local_bundle(
        output_root=args.output_root.resolve(),
        issued_at_unix_seconds=args.issued_at_unix_seconds,
        expires_at_unix_seconds=(
            args.issued_at_unix_seconds + args.window_seconds
        ),
    )
    sys.stdout.write(json.dumps(receipt, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
