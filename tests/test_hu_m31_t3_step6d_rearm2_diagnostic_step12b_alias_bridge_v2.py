from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_alias_bridge_v2
    as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_release_v2
    as pair_release_v2,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP11_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step11_one_vm_v12_fix3_actual"
)
STEP12_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step12_pair_v1_actual"
)
OUTER_ROOT = STEP11_ROOT / "local_preflight" / "outer"
RUN_NONCE = "c3" * 32
INPUTS = tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _approval_caps(
    deployment: dict[str, Any], role_index: int
) -> tuple[
    pair_release_v2.ValidatedRoleRuntimeApproval,
    pair_release_v2.PairReleaseApproval,
]:
    job_id = deployment["selected_job_ids"][role_index]
    instance = deployment["instances"][role_index]
    role_body = {
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "authorization_sha256": "11" * 32,
        "claim_sha256": f"{role_index + 2:02x}" * 32,
        "external_job_id": job_id,
        "inner_job_id": instance["inner_job_id"],
        "source_role": instance["source_role"],
        "instance_name": instance["instance_name"],
        "provider_instance_id": f"987654321000{role_index + 1}",
        "controller_preclaim_metadata_fingerprint": (
            f"ControllerPreClaimFingerprint{role_index + 1:04d}"
        ),
        "package_generations_sha256": "44" * 32,
        "vm_identity_receipt_sha256": "55" * 32,
    }
    role = pair_release_v2.ValidatedRoleRuntimeApproval(
        **role_body,
        receipt_sha256=pair_release_v2.canonical_sha256(role_body),
        _seal=pair_release_v2._ROLE_APPROVAL_SEAL,
    )
    pair_body = {
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "pair_release_sha256": "66" * 32,
        "external_job_id": job_id,
        "own_claim_sha256": role.claim_sha256,
        "phase2_iam_plan_sha256": "77" * 32,
        "phase2_zero_receipt_sha256": "88" * 32,
        "controller_create_receipt_sha256": "99" * 32,
        "controller_delete_receipt_sha256": "aa" * 32,
    }
    pair = pair_release_v2.PairReleaseApproval(
        **pair_body,
        receipt_sha256=pair_release_v2.canonical_sha256(pair_body),
        _seal=pair_release_v2._PAIR_RELEASE_SEAL,
    )
    return role, pair


@pytest.fixture(scope="module")
def inputs() -> INPUTS:
    candidate = _read(
        STEP12_ROOT / "candidate_transport_contract.json"
    )
    reference = _read(
        STEP12_ROOT / "reference_transport_contract.json"
    )
    controller_public_key_record = _read(
        STEP12_ROOT / "controller_public_key.json"
    )
    provision = _read(
        STEP11_ROOT / "package_provision_receipt.json"
    )
    deployment = deployment_v2.build_deployment_contract(
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
    )
    return (
        candidate,
        reference,
        controller_public_key_record,
        provision,
        deployment,
    )


class _PackageReader:
    def __init__(self) -> None:
        self.gets: list[tuple[str, int]] = []

    def generation_pinned_get(
        self, *, uri: str, generation: int
    ) -> bytes:
        self.gets.append((uri, generation))
        prefix = (
            "gs://pokerhu-ofc-solver-485418-training/"
            "hu-m31-r2diag-direct-v1/packages/"
            "593a1f1caaa8710811fc3044c32c66d681ccfe484095b9c94b3"
            "c6ba886aa6548/"
        )
        assert uri.startswith(prefix)
        relative = uri.removeprefix(prefix)
        return (OUTER_ROOT / PurePosixPath(relative)).read_bytes()


class _Writer:
    def __init__(self) -> None:
        self.writes: list[tuple[str, bytes]] = []
        self.objects: dict[str, bytes] = {}

    def conditional_create(
        self, *, uri: str, content: bytes
    ) -> dict[str, Any]:
        assert uri not in self.objects
        self.objects[uri] = content
        self.writes.append((uri, content))
        return {
            "uri": uri,
            "generation": len(self.writes),
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "created": True,
        }


class _Runtime:
    def __init__(self, *, add_extra_file: bool = False) -> None:
        self.add_extra_file = add_extra_file
        self.calls: list[str] = []

    def validate_and_run(
        self,
        *,
        payload_contract: dict[str, Any],
        deployment_contract: dict[str, Any],
        execution_job_id: str,
        outer_root: Path,
        work_root: Path,
    ) -> tuple[Path, dict[str, Any]]:
        assert outer_root.is_dir()
        payload_transport.validate_job_contract(payload_contract)
        self.calls.append(payload_contract["metadata_binding"]["job_id"])
        external = next(
            row
            for row in deployment_contract["remote_layout"]["jobs"]
            if row["job_id"] == execution_job_id
        )
        binding = next(
            row
            for row in deployment_contract["payload_binding"][
                "job_bindings"
            ]
            if row["inner_job_id"] == external["inner_job_id"]
        )
        output = work_root / "output"
        output.mkdir(parents=True)
        for relative in binding["tree_paths"]:
            path = output / PurePosixPath(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                subject.canonical_bytes(
                    {
                        "path": relative,
                        "role": binding["source_role"],
                        "fake_isolated_payload": True,
                    }
                )
            )
        if self.add_extra_file:
            (output / "unexpected.json").write_bytes(b"{}\n")
        return output, subject.build_payload_run_receipt(
            deployment_contract=deployment_contract,
            execution_job_id=execution_job_id,
        )


class _SelfDeleter:
    def __init__(self, writer: _Writer) -> None:
        self.writer = writer
        self.calls: list[dict[str, Any]] = []

    def request_exact_self_delete(
        self,
        *,
        project: str,
        zone: str,
        instance_name: str,
        done_readback: dict[str, Any],
    ) -> dict[str, Any]:
        assert len(self.writer.writes) == subject.RESULT_OBJECT_COUNT
        assert self.writer.writes[-1][0].endswith(
            "/DONE.envelope.json"
        )
        value = {
            "project": project,
            "zone": zone,
            "instance_name": instance_name,
            "delete_requested": True,
            "done_readback_sha256": subject.canonical_sha256(
                done_readback
            ),
        }
        self.calls.append(value)
        return value


@pytest.mark.parametrize("role_index", [0, 1])
def test_fake_vm_bridge_e2e_has_exact_direct_v2_counts_and_no_old_writes(
    tmp_path: Path,
    inputs: INPUTS,
    role_index: int,
) -> None:
    candidate, reference, public_key, provision, deployment = inputs
    execution_job_id = deployment["selected_job_ids"][role_index]
    reader = _PackageReader()
    writer = _Writer()
    runtime = _Runtime()
    deleter = _SelfDeleter(writer)
    role_approval, release_approval = _approval_caps(
        deployment, role_index
    )
    receipt = subject.execute_alias_bridge(
        deployment_contract=deployment,
        selected_payload_contract=(
            candidate if role_index == 0 else reference
        ),
        controller_public_key_record=public_key,
        run_nonce=RUN_NONCE,
        package_generations=provision["package_generations"],
        execution_job_id=execution_job_id,
        role_runtime_approval=role_approval,
        pair_release_approval=release_approval,
        package_reader=reader,
        payload_runtime=runtime,
        result_writer=writer,
        self_deleter=deleter,
        fresh_root=tmp_path / f"vm-{role_index}",
    )
    assert len(reader.gets) == 16
    assert len(writer.writes) == subject.RESULT_OBJECT_COUNT == 44
    assert receipt["tree_object_count"] == 23
    assert receipt["upload_count"] == 10
    assert receipt["heartbeat_count"] == 10
    assert receipt["done_count"] == 1
    assert receipt["old_result_write_count"] == 0
    assert receipt["old_execute_authorized_worker_called"] is False
    assert receipt["old_publisher_called"] is False
    assert receipt["external_direct_v2_only"] is True
    assert receipt["pair_release_verified"] is True
    assert receipt["role_runtime_approval_receipt_sha256"] == (
        role_approval.receipt_sha256
    )
    assert receipt["pair_release_approval_receipt_sha256"] == (
        release_approval.receipt_sha256
    )
    assert len(deleter.calls) == 1
    stage_prefix = deployment["remote_layout"]["stage_prefix"] + "/"
    assert all(uri.startswith(stage_prefix) for uri, _ in writer.writes)
    payload = candidate if role_index == 0 else reference
    forbidden = [
        payload["remote_layout"]["stage_prefix"],
        payload["remote_layout"]["result_prefix"],
        "gs://pokerhu-ofc-solver-485418-training/"
        "hu-m31-r2diag-worker-v1/",
    ]
    assert not any(
        uri.startswith(prefix)
        for uri, _ in writer.writes
        for prefix in forbidden
    )
    done = receipt["publish_receipt"]
    assert done["done_published_last"] is True
    assert done["done_readback"] == done["readbacks"][-1]
    assert done["done"]["tree_object_count"] == 23
    assert done["done"]["upload_count"] == 10
    assert done["done"]["heartbeat_count"] == 10


def test_package_generation_tamper_fails_before_payload_or_result_write(
    tmp_path: Path,
    inputs: INPUTS,
) -> None:
    candidate, reference, public_key, provision, deployment = inputs
    changed = copy.deepcopy(provision["package_generations"])
    uri = next(iter(changed))
    changed[uri] = 0
    runtime = _Runtime()
    writer = _Writer()
    role_approval, release_approval = _approval_caps(deployment, 0)
    with pytest.raises(ValueError, match="package generation"):
        subject.execute_alias_bridge(
            deployment_contract=deployment,
            selected_payload_contract=candidate,
            controller_public_key_record=public_key,
            run_nonce=RUN_NONCE,
            package_generations=changed,
            execution_job_id=deployment["selected_job_ids"][0],
            role_runtime_approval=role_approval,
            pair_release_approval=release_approval,
            package_reader=_PackageReader(),
            payload_runtime=runtime,
            result_writer=writer,
            self_deleter=_SelfDeleter(writer),
            fresh_root=tmp_path / "never-created",
        )
    assert runtime.calls == []
    assert writer.writes == []


def test_unexpected_payload_output_fails_before_any_result_or_self_delete(
    tmp_path: Path,
    inputs: INPUTS,
) -> None:
    candidate, reference, public_key, provision, deployment = inputs
    writer = _Writer()
    deleter = _SelfDeleter(writer)
    role_approval, release_approval = _approval_caps(deployment, 0)
    with pytest.raises(ValueError, match="output tree"):
        subject.execute_alias_bridge(
            deployment_contract=deployment,
            selected_payload_contract=candidate,
            controller_public_key_record=public_key,
            run_nonce=RUN_NONCE,
            package_generations=provision["package_generations"],
            execution_job_id=deployment["selected_job_ids"][0],
            role_runtime_approval=role_approval,
            pair_release_approval=release_approval,
            package_reader=_PackageReader(),
            payload_runtime=_Runtime(add_extra_file=True),
            result_writer=writer,
            self_deleter=deleter,
            fresh_root=tmp_path / "bad-output",
        )
    assert writer.writes == []
    assert deleter.calls == []


def test_wrong_or_forged_approvals_fail_before_any_payload_side_effect(
    tmp_path: Path,
    inputs: INPUTS,
) -> None:
    candidate, reference, public_key, provision, deployment = inputs
    role_approval, release_approval = _approval_caps(deployment, 0)
    reader = _PackageReader()
    runtime = _Runtime()
    writer = _Writer()
    deleter = _SelfDeleter(writer)
    root = tmp_path / "approval-rejected"
    with pytest.raises(ValueError, match="role runtime approval"):
        subject.execute_alias_bridge(
            deployment_contract=deployment,
            selected_payload_contract=reference,
            controller_public_key_record=public_key,
            run_nonce=RUN_NONCE,
            package_generations=provision["package_generations"],
            execution_job_id=deployment["selected_job_ids"][1],
            role_runtime_approval=role_approval,
            pair_release_approval=release_approval,
            package_reader=reader,
            payload_runtime=runtime,
            result_writer=writer,
            self_deleter=deleter,
            fresh_root=root,
        )
    assert reader.gets == []
    assert runtime.calls == []
    assert writer.writes == []
    assert deleter.calls == []
    assert not root.exists()

    with pytest.raises(ValueError, match="cannot be forged"):
        pair_release_v2.PairReleaseApproval(
            deployment_contract_sha256=(
                release_approval.deployment_contract_sha256
            ),
            pair_release_sha256=release_approval.pair_release_sha256,
            external_job_id=release_approval.external_job_id,
            own_claim_sha256=release_approval.own_claim_sha256,
            phase2_iam_plan_sha256=(
                release_approval.phase2_iam_plan_sha256
            ),
            phase2_zero_receipt_sha256=(
                release_approval.phase2_zero_receipt_sha256
            ),
            controller_create_receipt_sha256=(
                release_approval.controller_create_receipt_sha256
            ),
            controller_delete_receipt_sha256=(
                release_approval.controller_delete_receipt_sha256
            ),
            receipt_sha256=release_approval.receipt_sha256,
            _seal=object(),
        )


def _materialize_isolated_bridge_bundle(destination: Path) -> Path:
    bundle = destination / "isolated-bridge-bundle"
    outer = bundle / "outer"
    contracts = bundle / "contracts"
    runtime_package = bundle / "runtime" / "ofc_regular"
    shutil.copytree(OUTER_ROOT, outer)
    contracts.mkdir(parents=True)
    runtime_package.mkdir(parents=True)
    for source in (outer / "src" / "ofc_regular").glob("*.py"):
        shutil.copy2(source, runtime_package / source.name)
    for name in (
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_alias_bridge_v2.py",
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_deployment_contract_v2.py",
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_external_authorization_v2.py",
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_phase2_iam_plan_v2.py",
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step11_rest_iam_admin_v1.py",
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_run_scoped_controller_sa_v2.py",
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_vm_metadata_v2.py",
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_bootstrap_source_v2.py",
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_bootstrap_source_content_v2.py",
        "hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_pair_release_v2.py",
    ):
        shutil.copy2(
            REPO_ROOT / "src" / "ofc_regular" / name,
            runtime_package / name,
        )
    (runtime_package / "__init__.py").write_text(
        '"""Minimal isolated Step12b VM bridge namespace."""\n',
        encoding="utf-8",
    )
    for name in (
        "hu_m31_t3_step6d_candidate02_full100_plan.py",
        "hu_m31_t3_step6d_full100_spot_v1.py",
    ):
        (runtime_package / name).write_text(
            '"""Import-only stub for the immutable outer plan module."""\n',
            encoding="utf-8",
        )
    for source, target in (
        (
            STEP12_ROOT / "candidate_transport_contract.json",
            "candidate_transport_contract.json",
        ),
        (
            STEP12_ROOT / "reference_transport_contract.json",
            "reference_transport_contract.json",
        ),
        (
            STEP12_ROOT / "controller_public_key.json",
            "controller_public_key.json",
        ),
        (
            STEP11_ROOT / "package_provision_receipt.json",
            "package_provision_receipt.json",
        ),
    ):
        shutil.copy2(source, contracts / target)
    return bundle


def test_bridge_core_runs_in_outer_only_isolated_python_process(
    tmp_path: Path,
) -> None:
    bundle = _materialize_isolated_bridge_bundle(tmp_path)
    program = r"""
import hashlib, json, pathlib, sys
bundle = pathlib.Path(sys.argv[1]).resolve()
forbidden_repo = pathlib.Path(sys.argv[2]).resolve()
root = pathlib.Path(sys.argv[3]).resolve()
runtime = bundle / "runtime"
sys.path.insert(0, str(runtime))
for entry in sys.path:
    if not entry:
        continue
    assert "site-packages" not in entry.lower(), sys.path
    resolved = pathlib.Path(entry).resolve()
    assert resolved != forbidden_repo and forbidden_repo not in resolved.parents, sys.path
from ofc_regular import hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1 as pt
from ofc_regular import hu_m31_t3_step6d_rearm2_diagnostic_step12b_alias_bridge_v2 as b
from ofc_regular import hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2 as d
from ofc_regular import hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_release_v2 as pr
for module in (pt, b, d, pr):
    assert runtime in pathlib.Path(module.__file__).resolve().parents
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
contracts = bundle / "contracts"
outer = bundle / "outer"
c = read(contracts / "candidate_transport_contract.json")
r = read(contracts / "reference_transport_contract.json")
k = read(contracts / "controller_public_key.json")
p = read(contracts / "package_provision_receipt.json")
nonce = "c3" * 32
dep = d.build_deployment_contract(candidate_payload_contract=c, reference_payload_contract=r, controller_public_key_record=k, run_nonce=nonce)
job = dep["selected_job_ids"][0]
inst = dep["instances"][0]
role_body = {"deployment_contract_sha256":dep["deployment_contract_sha256"],"authorization_sha256":"11"*32,"claim_sha256":"22"*32,"external_job_id":job,"inner_job_id":inst["inner_job_id"],"source_role":inst["source_role"],"instance_name":inst["instance_name"],"provider_instance_id":"9876543210001","controller_preclaim_metadata_fingerprint":"ControllerPreClaimFingerprint0001","package_generations_sha256":"44"*32,"vm_identity_receipt_sha256":"55"*32}
role = pr.ValidatedRoleRuntimeApproval(**role_body, receipt_sha256=pr.canonical_sha256(role_body), _seal=pr._ROLE_APPROVAL_SEAL)
pair_body = {"deployment_contract_sha256":dep["deployment_contract_sha256"],"pair_release_sha256":"66"*32,"external_job_id":job,"own_claim_sha256":role.claim_sha256,"phase2_iam_plan_sha256":"77"*32,"phase2_zero_receipt_sha256":"88"*32,"controller_create_receipt_sha256":"99"*32,"controller_delete_receipt_sha256":"aa"*32}
pair = pr.PairReleaseApproval(**pair_body, receipt_sha256=pr.canonical_sha256(pair_body), _seal=pr._PAIR_RELEASE_SEAL)
prefix = dep["remote_layout"]["package_prefix"] + "/"
class Reader:
    def generation_pinned_get(self, *, uri, generation):
        del generation
        return (outer / uri.removeprefix(prefix)).read_bytes()
class Writer:
    def __init__(self):
        self.rows = []
    def conditional_create(self, *, uri, content):
        self.rows.append(uri)
        return {"uri":uri,"generation":len(self.rows),"sha256":hashlib.sha256(content).hexdigest(),"bytes":len(content),"created":True}
w = Writer()
class Runtime:
    def validate_and_run(self, *, payload_contract, deployment_contract, execution_job_id, outer_root, work_root):
        pt.validate_job_contract(payload_contract)
        ext = next(x for x in deployment_contract["remote_layout"]["jobs"] if x["job_id"] == execution_job_id)
        bind = next(x for x in deployment_contract["payload_binding"]["job_bindings"] if x["inner_job_id"] == ext["inner_job_id"])
        out = work_root / "output"
        out.mkdir(parents=True)
        for rel in bind["tree_paths"]:
            path = out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b.canonical_bytes({"path":rel,"isolated":True}))
        return out, b.build_payload_run_receipt(deployment_contract=deployment_contract, execution_job_id=execution_job_id)
class Delete:
    def request_exact_self_delete(self, *, project, zone, instance_name, done_readback):
        assert len(w.rows) == 44
        return {"project":project,"zone":zone,"instance_name":instance_name,"delete_requested":True,"done_readback_sha256":b.canonical_sha256(done_readback)}
receipt = b.execute_alias_bridge(
    deployment_contract=dep,
    selected_payload_contract=c,
    controller_public_key_record=k,
    run_nonce=nonce,
    package_generations=p["package_generations"],
    execution_job_id=job,
    role_runtime_approval=role,
    pair_release_approval=pair,
    package_reader=Reader(),
    payload_runtime=Runtime(),
    result_writer=w,
    self_deleter=Delete(),
    fresh_root=root / "vm",
)
print(json.dumps({"objects":receipt["result_object_count"],"old":receipt["old_result_write_count"],"self_delete":receipt["self_delete_receipt"]["delete_requested"]}, sort_keys=True))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            program,
            str(bundle),
            str(REPO_ROOT),
            str(tmp_path),
        ],
        cwd=bundle,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "PYTHONHOME",
                "PYTHONPATH",
                "PYTHONUSERBASE",
            }
        },
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "objects": 44,
        "old": 0,
        "self_delete": True,
    }
