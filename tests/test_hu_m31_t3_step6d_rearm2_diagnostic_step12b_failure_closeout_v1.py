from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_failure_closeout_v1
    as subject,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP6D_ROOT = (
    REPO_ROOT / "outputs" / "hu_joint_policy" / "m31_t3_step6d"
)
OUTPUT_ROOT = STEP6D_ROOT / "step12b_pair_v2_actual"
POLICY_PATH = REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / (
        "verify_hu_m31_t3_step6d_rearm2_diagnostic_"
        "step12b_failure_closeout_v1.py"
    )
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


class _FakeReadOnlyObserver:
    def __init__(
        self,
        *,
        output_root: Path = OUTPUT_ROOT,
        compute_present: bool = False,
        controller_present: bool = False,
        residual_binding: bool = False,
        result_objects: Sequence[str] = (),
        omit_source: bool = False,
        source_generation_delta: int = 0,
        root_cause_audit_drift: bool = False,
        unexpected_operation: bool = False,
    ) -> None:
        self.deployment = _read(output_root / "deployment_contract.json")
        self.plan = _read(output_root / "phase2_iam_plan.json")
        self.source = _read(
            output_root / "bootstrap_source_provision_receipt.json"
        )
        self.compute_present = compute_present
        self.controller_present = controller_present
        self.residual_binding = residual_binding
        self.result_objects = list(result_objects)
        self.omit_source = omit_source
        self.source_generation_delta = source_generation_delta
        self.root_cause_audit_drift = root_cause_audit_drift
        self.unexpected_operation = unexpected_operation
        self.log: list[str] = []

    def verify_exact_compute_absent(
        self,
        *,
        instance_names: Sequence[str],
        disk_names: Sequence[str],
    ) -> Mapping[str, Any]:
        self.log.append("compute_exact_four_get")
        assert list(instance_names) == [
            row["instance_name"] for row in self.deployment["instances"]
        ]
        assert list(disk_names) == list(instance_names)
        return {
            "instance_final_statuses": (
                [200, 404] if self.compute_present else [404, 404]
            ),
            "disk_final_statuses": [404, 404],
            "all_four_targets_get404_verified": not self.compute_present,
        }

    def get_controller_service_account(
        self, *, email: str
    ) -> Mapping[str, Any] | None:
        self.log.append("controller_service_account_get")
        assert email == self.deployment["controller_service_account"]["email"]
        return {"email": email} if self.controller_present else None

    def get_iam_policy(
        self, *, target: str
    ) -> Mapping[str, Any] | None:
        self.log.append("iam_policy_get")
        rows = [
            *self.plan["phase2_bindings"]["controller"],
            *self.plan["phase2_bindings"]["worker"],
        ]
        selected = [row for row in rows if row["target"] == target]
        bindings = []
        if self.residual_binding and selected:
            row = selected[0]
            bindings.append(
                {
                    "role": row["role"],
                    "members": [row["member"]],
                    "condition": copy.deepcopy(row["condition"]),
                }
            )
        return {
            "version": 3,
            "etag": f"fake-{target}",
            "bindings": bindings,
        }

    def list_direct_v2_stage_objects(
        self, *, stage_prefix: str
    ) -> Sequence[str]:
        self.log.append("direct_v2_stage_list")
        assert stage_prefix == self.deployment["remote_layout"]["stage_prefix"]
        return list(self.result_objects)

    def list_bootstrap_source_objects(
        self, *, source_prefix: str
    ) -> Sequence[str]:
        self.log.append("bootstrap_source_list")
        assert source_prefix == self.source["source_prefix"]
        uris = [row["uri"] for row in self.source["records"]]
        return uris[1:] if self.omit_source else uris

    def read_bootstrap_source_generation(
        self, *, uri: str, generation: int
    ) -> Mapping[str, Any]:
        self.log.append("bootstrap_source_generation_get")
        row = next(row for row in self.source["records"] if row["uri"] == uri)
        return {
            "uri": uri,
            "generation": generation + self.source_generation_delta,
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "readback_complete": True,
        }

    def read_token_barrier_audit_events(
        self, *, controller_unique_id: str
    ) -> Sequence[Mapping[str, Any]]:
        self.log.append("cloud_audit_log_read")
        rows = subject._expected_root_cause_audit_events(
            controller_unique_id
        )
        if self.root_cause_audit_drift:
            rows[1]["timestamp"] = "2026-07-20T09:43:57Z"
        return rows

    def audit_log(self) -> Sequence[str]:
        return [
            *self.log,
            *(["delete"] if self.unexpected_operation else []),
        ]


def _local_guards(
    tmp_path: Path,
) -> tuple[Path, dict[str, Path], dict[str, str]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    policy = tmp_path / "ai_profiles.py"
    policy.write_bytes(POLICY_PATH.read_bytes())
    roots: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for name in sorted(subject.EXPECTED_OLD_TREE_KEYS):
        root = tmp_path / name
        root.mkdir()
        (root / "evidence.json").write_text(
            json.dumps({"name": name}, sort_keys=True),
            encoding="utf-8",
        )
        roots[name] = root
        digests[name] = subject.build_local_tree_snapshot(root)[
            "tree_sha256"
        ]
    return policy, roots, digests


def _verify(
    tmp_path: Path,
    *,
    output_root: Path = OUTPUT_ROOT,
    observer: _FakeReadOnlyObserver | None = None,
) -> tuple[dict[str, Any], _FakeReadOnlyObserver, dict[str, Path]]:
    policy, roots, digests = _local_guards(tmp_path)
    selected = (
        _FakeReadOnlyObserver(output_root=output_root)
        if observer is None
        else observer
    )
    receipt = subject.verify_token_barrier_failure_closeout(
        output_root=output_root,
        observer=selected,
        policy_registry_path=policy,
        old_tree_roots=roots,
        expected_old_tree_digests=digests,
    )
    return receipt, selected, roots


def test_happy_closeout_is_sealed_read_only_and_complete(
    tmp_path: Path,
) -> None:
    receipt, observer, _ = _verify(tmp_path)
    assert subject.validate_closeout_receipt(receipt) == receipt
    assert receipt["failure_stage"] == "token_barrier"
    assert receipt["controller_service_account_get_status"] == 404
    assert receipt["instance_final_statuses"] == [404, 404]
    assert receipt["disk_final_statuses"] == [404, 404]
    assert receipt["phase2_condition_title_binding_count"] == 8
    assert all(
        row["member_occurrences"] == 0
        for row in receipt["phase2_condition_title_readbacks"]
    )
    assert receipt["direct_v2_result_object_count"] == 0
    assert receipt["bootstrap_source_object_count"] == 3
    assert receipt["source_objects_deleted"] is False
    assert receipt["old_v1_touched"] is False
    assert receipt["cloud_mutation_performed"] is False
    assert receipt["automatic_retry_performed"] is False
    assert receipt["root_cause_classification"] == (
        subject.ROOT_CAUSE_CLASSIFICATION
    )
    assert receipt["root_cause_token_mint_succeeded"] is True
    assert receipt["root_cause_token_creator_revoke_succeeded"] is True
    assert receipt["root_cause_empty_policy_bindings_omitted"] is True
    assert len(receipt["root_cause_cloud_audit_events"]) == 3
    assert len(observer.log) == 11
    assert not any(
        operation in {"create", "insert", "set", "delete", "cleanup"}
        for operation in observer.log
    )


@pytest.mark.parametrize(
    "observer_kwargs,match",
    [
        ({"compute_present": True}, "VM/two disk"),
        ({"controller_present": True}, "still exists"),
        ({"residual_binding": True}, "binding remained"),
        (
            {"result_objects": ["gs://bucket/unexpected.json"]},
            "objects are not zero",
        ),
        ({"omit_source": True}, "inventory changed"),
        ({"source_generation_delta": 1}, "generation identity"),
        (
            {"root_cause_audit_drift": True},
            "Cloud Audit evidence",
        ),
        ({"unexpected_operation": True}, "operation audit"),
    ],
)
def test_cloud_or_audit_drift_fails_closed(
    tmp_path: Path,
    observer_kwargs: dict[str, Any],
    match: str,
) -> None:
    observer = _FakeReadOnlyObserver(**observer_kwargs)
    with pytest.raises(ValueError, match=match):
        _verify(tmp_path, observer=observer)


def test_resealed_failure_retry_claim_is_rejected(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "step12b_pair_v2_actual"
    shutil.copytree(OUTPUT_ROOT, copied)
    failure_path = copied / "FAILURE.json"
    failure = _read(failure_path)
    failure.pop("receipt_sha256")
    failure["automatic_retry_performed"] = True
    failure["receipt_sha256"] = subject.canonical_sha256(failure)
    failure_path.write_bytes(subject.canonical_bytes(failure) + b"\n")
    observer = _FakeReadOnlyObserver(output_root=copied)
    policy, roots, digests = _local_guards(tmp_path / "guards")
    with pytest.raises(ValueError, match="failure/cleanup evidence"):
        subject.verify_token_barrier_failure_closeout(
            output_root=copied,
            observer=observer,
            policy_registry_path=policy,
            old_tree_roots=roots,
            expected_old_tree_digests=digests,
        )
    assert observer.log == []


def test_old_tree_or_current_profile_change_is_rejected(
    tmp_path: Path,
) -> None:
    policy, roots, digests = _local_guards(tmp_path)
    (roots["step12_pair_v1_actual"] / "evidence.json").write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="predecessor tree changed"):
        subject.verify_token_barrier_failure_closeout(
            output_root=OUTPUT_ROOT,
            observer=_FakeReadOnlyObserver(),
            policy_registry_path=policy,
            old_tree_roots=roots,
            expected_old_tree_digests=digests,
        )

    policy.write_text("changed", encoding="utf-8")
    fresh_roots: dict[str, Path] = {}
    fresh_digests: dict[str, str] = {}
    for name in sorted(subject.EXPECTED_OLD_TREE_KEYS):
        root = tmp_path / "fresh" / name
        root.mkdir(parents=True)
        (root / "file").write_text(name, encoding="utf-8")
        fresh_roots[name] = root
        fresh_digests[name] = subject.build_local_tree_snapshot(root)[
            "tree_sha256"
        ]
    with pytest.raises(ValueError, match="profile registry changed"):
        subject.verify_token_barrier_failure_closeout(
            output_root=OUTPUT_ROOT,
            observer=_FakeReadOnlyObserver(),
            policy_registry_path=policy,
            old_tree_roots=fresh_roots,
            expected_old_tree_digests=fresh_digests,
        )


def test_frozen_real_predecessor_tree_digests_match() -> None:
    roots = {
        "step11_one_vm_v12_fix3_actual": (
            STEP6D_ROOT / "step11_one_vm_v12_fix3_actual"
        ),
        "rearm2_diagnostic_cloud_worker_package_v1": (
            STEP6D_ROOT / "rearm2_diagnostic_cloud_worker_package_v1"
        ),
        "step12_pair_v1_actual": (
            STEP6D_ROOT / "step12_pair_v1_actual"
        ),
    }
    assert {
        name: subject.build_local_tree_snapshot(root)["tree_sha256"]
        for name, root in roots.items()
    } == subject.EXPECTED_OLD_TREE_DIGESTS


def test_receipt_writer_is_write_once_and_idempotent(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "step12b_failure_closeout_script", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt, _, _ = _verify(tmp_path / "guards")
    path = tmp_path / "closeout.json"
    first = module.write_once_or_validate_identical(path, receipt)
    raw = path.read_bytes()
    second = module.write_once_or_validate_identical(path, receipt)
    assert first == second == receipt
    assert path.read_bytes() == raw

    changed = copy.deepcopy(receipt)
    changed.pop("receipt_sha256")
    changed["diagnostic_only"] = False
    changed["receipt_sha256"] = subject.canonical_sha256(changed)
    with pytest.raises(FileExistsError, match="different"):
        module.write_once_or_validate_identical(path, changed)
    assert path.read_bytes() == raw


def test_gcloud_inspector_uses_only_read_commands_and_normalizes_audit() -> None:
    spec = importlib.util.spec_from_file_location(
        "step12b_failure_closeout_script_gcloud", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    unique_id = "113492688565195541029"
    resource = f"projects/-/serviceAccounts/{unique_id}"
    calls: list[list[str]] = []

    def runner(arguments: Sequence[str], **kwargs: Any) -> Any:
        calls.append(list(arguments))
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        if list(arguments[1:3]) == [
            "projects",
            "get-iam-policy",
        ]:
            value: Any = {"etag": "project", "bindings": []}
        elif list(arguments[1:4]) == [
            "storage",
            "buckets",
            "get-iam-policy",
        ]:
            value = {"etag": "bucket", "bindings": []}
        elif list(arguments[1:4]) == [
            "iam",
            "service-accounts",
            "get-iam-policy",
        ]:
            value = {"etag": "worker", "bindings": []}
        elif list(arguments[1:3]) == ["logging", "read"]:
            value = [
                {
                    "timestamp": subject.EXPECTED_TOKEN_MINT_TIMESTAMP,
                    "protoPayload": {
                        "serviceName": "iamcredentials.googleapis.com",
                        "methodName": "GenerateAccessToken",
                        "resourceName": resource,
                        "status": {},
                    },
                },
                {
                    "timestamp": (
                        subject.EXPECTED_TOKEN_CREATOR_REVOKE_TIMESTAMP
                    ),
                    "protoPayload": {
                        "serviceName": "iam.googleapis.com",
                        "methodName": (
                            "google.iam.admin.v1.SetIAMPolicy"
                        ),
                        "resourceName": resource,
                        "status": {},
                        "response": {
                            "@type": (
                                "type.googleapis.com/google.iam.v1.Policy"
                            ),
                            "etag": "BwZXB764JeY=",
                            "version": 1,
                        },
                    },
                },
                {
                    "timestamp": (
                        subject.EXPECTED_CONTROLLER_DELETE_TIMESTAMP
                    ),
                    "protoPayload": {
                        "serviceName": "iam.googleapis.com",
                        "methodName": (
                            "google.iam.admin.v1.DeleteServiceAccount"
                        ),
                        "resourceName": resource,
                        "status": {},
                    },
                },
            ]
        else:
            raise AssertionError(f"unexpected gcloud command: {arguments}")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(value),
            stderr="",
        )

    inspector = module.GcloudReadOnlyInspector(
        project="ofc-solver-485418",
        bucket="pokerhu-ofc-solver-485418-training",
        worker_service_account=(
            "ofc-m31-t3-diagnostic@"
            "ofc-solver-485418.iam.gserviceaccount.com"
        ),
        executable="gcloud-test",
        command_runner=runner,
    )
    assert inspector.get_policy(
        module.rest_iam.PolicyTarget.PROJECT
    )["etag"] == "project"
    assert inspector.get_policy(
        module.rest_iam.PolicyTarget.BUCKET
    )["etag"] == "bucket"
    assert inspector.get_policy(
        module.rest_iam.PolicyTarget.WORKER_SERVICE_ACCOUNT
    )["etag"] == "worker"
    assert inspector.read_token_barrier_audit_events(
        controller_unique_id=unique_id
    ) == subject._expected_root_cause_audit_events(unique_id)
    assert [call[1:3] for call in calls] == [
        ["projects", "get-iam-policy"],
        ["storage", "buckets"],
        ["iam", "service-accounts"],
        ["logging", "read"],
    ]
