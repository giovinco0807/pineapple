#!/usr/bin/env python3
"""Prepare the corrected Step 11 package and launch contracts locally only."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as canary_plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1
    as controller,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_launch_contract_v1
    as launch,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_DIR = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "rearm2_diagnostic_cloud_worker_package_v1"
)
DEFAULT_WHEEL_PATH = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "rearm2_diagnostic_10c2_local_preflight_v4"
    / "outer"
    / "wheels"
    / transport.EXPECTED_NUMPY_WHEEL_FILENAME
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step11_v12_local_preflight_fix3"
)
STARTUP_PATH = (
    REPO_ROOT
    / "scripts"
    / "bootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.sh"
)
PREBOOTSTRAP_PATH = (
    REPO_ROOT
    / "scripts"
    / "prebootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.py"
)
PROFILE_PATH = REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"
EXPECTED_PROFILE_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
RECEIPT_SCHEMA = "hu_m31_t3_step6d_step11_v12_local_preflight_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    controller.exclusive_write_json(path, value)


def _validate_isolated_bootstrap(
    *,
    outer_root: Path,
    contract_path: Path,
) -> dict[str, Any]:
    program = r"""
import json
import pathlib
import sys

outer = pathlib.Path(sys.argv[1]).resolve()
contract_path = pathlib.Path(sys.argv[2]).resolve()
repository = pathlib.Path(sys.argv[3]).resolve()
extracted = pathlib.Path(sys.argv[4]).resolve()
sys.path = [str(outer / "src")] + [
    entry
    for entry in sys.path
    if entry
    and "site-packages" not in entry
    and repository not in pathlib.Path(entry).resolve().parents
    and pathlib.Path(entry).resolve() != repository
]
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
with contract_path.open("rb") as handle:
    checked = transport.validate_job_contract(json.load(handle))
with (outer / "inner" / "manifest.json").open("rb") as handle:
    inner_manifest = json.load(handle)
transport.safe_extract_worker_source(
    source_zip=(
        outer
        / "inner"
        / "hu_m31_t3_step6d_rearm2_diagnostic_worker_v1.zip"
    ),
    inner_manifest=inner_manifest,
    destination=extracted,
)
import ofc_regular
ofc_regular.__path__.append(str(extracted / "src" / "ofc_regular"))
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter
    as adapter,
)
probe = {"canonical": ["runtime", 1, True]}
expected = b'{"canonical":["runtime",1,true]}\n'
if adapter.canonical_bytes(probe) != expected:
    raise SystemExit("adapter canonical parity changed")
runner_path = (
    extracted
    / "src"
    / "ofc_regular"
    / "run_hu_m31_t3_step6d_performance_v2.py"
)
if not runner_path.is_file():
    raise SystemExit("extracted runner is missing")
print(checked["schema"])
print(adapter.ADAPTER_SCHEMA)
print("extracted-runner-present-parent-numpy-not-required")
"""
    with tempfile.TemporaryDirectory(prefix="ofc-step11-v12-") as temporary:
        completed = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                program,
                str(outer_root),
                str(contract_path),
                str(REPO_ROOT),
                str(Path(temporary) / "extracted"),
            ],
            cwd=outer_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
    expected = [
        transport.CONTRACT_SCHEMA,
        (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "cloud_worker_adapter_v1"
        ),
        "extracted-runner-present-parent-numpy-not-required",
    ]
    if completed.returncode != 0 or completed.stdout.splitlines() != expected:
        raise RuntimeError(
            "isolated bootstrap closure validation failed: "
            + completed.stderr[-2_000:]
        )
    return {
        "python_no_site": True,
        "repository_removed_from_sys_path": True,
        "site_packages_removed_from_sys_path": True,
        "transport_contract_validated": True,
        "runtime_adapter_imported": True,
        "worker_source_safely_extracted": True,
        "extracted_runner_module_present": True,
        "parent_interpreter_numpy_required": False,
        "adapter_canonical_bytes_exact": True,
        "stdout_lines": expected,
    }


def prepare(
    *,
    output_root: Path,
    package_dir: Path,
    wheel_path: Path,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    profile_sha = _sha256_file(PROFILE_PATH)
    if profile_sha != EXPECTED_PROFILE_SHA256:
        raise RuntimeError("current profile source changed before local prepare")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()

    signer = controller.generate_ephemeral_controller_key(key_size=3_072)
    wheel_record = transport.build_offline_wheel_record(wheel_path)
    contract = transport.build_job_contract(
        package_dir=package_dir,
        stage_id=canary_plan.STAGE1_ID,
        job_id=canary_plan.STAGE1_JOB_IDS[0],
        offline_wheel_record=wheel_record,
        controller_public_key_record=signer.public_record,
    )
    contract_path = output_root / "transport_contract.json"
    _write(contract_path, contract)
    _write(output_root / "controller_public_key.json", signer.public_record)

    local_root = output_root / "local_preflight"
    local_receipt = transport.local_preflight(
        contract=contract,
        package_mirror=package_dir,
        fresh_work_root=local_root,
        offline_wheel_mirror=wheel_path,
        offline_install_smoke=False,
    )
    _write(output_root / "local_preflight_receipt.json", local_receipt)
    outer_root = local_root / "outer"

    launch_contract = launch.build_launch_contract(
        transport_contract=contract,
        controller_public_key_record=signer.public_record,
        prebootstrap_path=PREBOOTSTRAP_PATH,
        startup_path=STARTUP_PATH,
    )
    _write(output_root / "launch_contract.json", launch_contract)
    package_plan = controller.build_package_provision_plan(
        contract=contract,
        outer_root=outer_root,
    )
    _write(output_root / "package_provision_plan.json", package_plan)
    isolated = _validate_isolated_bootstrap(
        outer_root=outer_root,
        contract_path=contract_path,
    )

    body = {
        "schema": RECEIPT_SCHEMA,
        "status": "local_v12_package_and_launch_contract_ready_no_cloud",
        "outer_package_identity_sha256": contract[
            "outer_package_manifest"
        ]["outer_package_identity_sha256"],
        "direct_stage_identity_sha256": contract[
            "direct_stage_identity_sha256"
        ],
        "instance_name": contract["metadata_binding"]["instance_name"],
        "transport_contract_sha256": transport.canonical_sha256(contract),
        "launch_contract_sha256": transport.canonical_sha256(
            launch_contract
        ),
        "package_provision_plan_sha256": controller.canonical_sha256(
            package_plan
        ),
        "local_preflight_receipt_sha256": transport.canonical_sha256(
            local_receipt
        ),
        "isolated_bootstrap": isolated,
        "profile_sha256": profile_sha,
        "controller_private_key_serialized": False,
        "authorization_created": False,
        "claim_created": False,
        "package_uploaded": False,
        "iam_changed": False,
        "vm_created": False,
        "cloud_mutation_performed": False,
        "current_profile_changed": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }
    receipt = {**body, "receipt_sha256": controller.canonical_sha256(body)}
    _write(output_root / "LOCAL_V12_READY.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR
    )
    parser.add_argument(
        "--wheel", type=Path, default=DEFAULT_WHEEL_PATH
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = prepare(
        output_root=args.output_root.resolve(),
        package_dir=args.package_dir.resolve(),
        wheel_path=args.wheel.resolve(),
    )
    print(
        receipt["status"],
        receipt["outer_package_identity_sha256"],
        receipt["direct_stage_identity_sha256"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
