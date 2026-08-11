from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_package as package_builder,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_cloud_controller_v1
    as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1
    as primitives,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_launch_contract_v1
    as launch,
)


@pytest.fixture(scope="module")
def key() -> primitives.EphemeralControllerKey:
    return primitives.generate_ephemeral_controller_key(key_size=2_048)


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("step11-cloud-controller") / "package"
    package_builder.build_package(output_dir=target)
    return target


@pytest.fixture(scope="module")
def contract(
    package: Path, key: primitives.EphemeralControllerKey
) -> dict[str, Any]:
    wheel = {
        "path": f"wheels/{transport.EXPECTED_NUMPY_WHEEL_FILENAME}",
        "uri_suffix": f"wheels/{transport.EXPECTED_NUMPY_WHEEL_FILENAME}",
        "sha256": transport.EXPECTED_NUMPY_WHEEL_SHA256,
        "bytes": transport.EXPECTED_NUMPY_WHEEL_BYTES,
        "mode": "0644",
        "kind": "offline_numpy_cp311_manylinux_x86_64_wheel",
    }
    return transport.build_job_contract(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        job_id=plan.STAGE1_JOB_IDS[0],
        offline_wheel_record=wheel,
        controller_public_key_record=key.public_record,
    )


@pytest.fixture(scope="module")
def launch_contract(
    contract: dict[str, Any],
    key: primitives.EphemeralControllerKey,
) -> dict[str, Any]:
    return launch.build_launch_contract(
        transport_contract=contract,
        controller_public_key_record=key.public_record,
        prebootstrap_path=(
            Path(__file__).parents[1]
            / "scripts"
            / "prebootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.py"
        ),
    )


@pytest.fixture()
def authorization(
    contract: dict[str, Any],
    key: primitives.EphemeralControllerKey,
) -> dict[str, Any]:
    return primitives.build_controller_authorization(
        contract=contract,
        external_preflight_receipt_sha256="a" * 64,
        issued_unix_seconds=1_000,
        expires_unix_seconds=2_000,
        nonce="b" * 64,
        signer=key,
    )


def _startup() -> Path:
    return Path(__file__).parents[1] / (
        "scripts/"
        "bootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.sh"
    )


def _prebootstrap() -> Path:
    return Path(__file__).parents[1] / (
        "scripts/"
        "prebootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.py"
    )


def _insert_body(
    *,
    contract: dict[str, Any],
    launch_contract: dict[str, Any],
    authorization: dict[str, Any],
    key: primitives.EphemeralControllerKey,
) -> dict[str, Any]:
    return subject.materialize_insert_body(
        transport_contract=contract,
        launch_contract=launch_contract,
        authorization=authorization,
        public_key_record=key.public_record,
        startup_path=_startup(),
        prebootstrap_path=_prebootstrap(),
    )


def test_materializes_real_gce_metadata_items_and_no_abstract_or_extra_fields(
    contract: dict[str, Any],
    launch_contract: dict[str, Any],
    authorization: dict[str, Any],
    key: primitives.EphemeralControllerKey,
) -> None:
    body = _insert_body(
        contract=contract,
        launch_contract=launch_contract,
        authorization=authorization,
        key=key,
    )
    assert set(body["metadata"]) == {"items"}
    assert "scalarItems" not in body["metadata"]
    assert "metadataFromFileKeys" not in body["metadata"]
    assert "forbiddenAtInsert" not in body["metadata"]
    assert "labels" not in body
    assert "deletionProtection" not in body
    assert body["reservationAffinity"] == {
        "consumeReservationType": "NO_RESERVATION"
    }
    assert body["networkInterfaces"][0]["accessConfigs"] == []
    values = {row["key"]: row["value"] for row in body["metadata"]["items"]}
    assert len(values) == 15
    assert set(contract["metadata_values"]).issubset(values)
    assert set(launch.INITIAL_METADATA_FROM_FILE_KEYS).issubset(values)
    assert values["block-project-ssh-keys"] == "true"
    assert values["ofc-step11-claim-release-state"] == (
        "pending-post-create"
    )
    assert launch.POSTCREATE_CLAIM_METADATA_KEY not in values


def test_claim_cas_preserves_every_initial_key_and_adds_only_claim(
    contract: dict[str, Any],
    launch_contract: dict[str, Any],
    authorization: dict[str, Any],
    key: primitives.EphemeralControllerKey,
) -> None:
    insert = _insert_body(
        contract=contract,
        launch_contract=launch_contract,
        authorization=authorization,
        key=key,
    )
    initial = {
        row["key"]: row["value"] for row in insert["metadata"]["items"]
    }
    claim = primitives.build_worker_claim(
        contract=contract,
        authorization=authorization,
        project_number=subject.PROJECT_NUMBER,
        instance_id="9876543210",
        package_generations={
            row["uri"]: index + 1
            for index, row in enumerate(
                contract["remote_layout"]["package_inventory"]["records"]
            )
        },
        nonce="c" * 64,
        signer=key,
    )
    provider = {
        "metadata": {
            "fingerprint": "provider-cas-fingerprint",
            "items": insert["metadata"]["items"],
        }
    }
    cas = subject.build_claim_cas_body(
        provider_instance=provider,
        expected_initial_metadata=initial,
        claim=claim,
    )
    updated = {row["key"]: row["value"] for row in cas["items"]}
    assert cas["fingerprint"] == "provider-cas-fingerprint"
    assert set(updated) == set(initial) | {launch.POSTCREATE_CLAIM_METADATA_KEY}
    assert all(updated[key] == value for key, value in initial.items())
    assert json.loads(updated[launch.POSTCREATE_CLAIM_METADATA_KEY]) == claim

    changed = deepcopy(provider)
    changed["metadata"]["items"].append(
        {"key": "unknown-provider-key", "value": "fatal"}
    )
    with pytest.raises(ValueError, match="provider metadata changed"):
        subject.build_claim_cas_body(
            provider_instance=changed,
            expected_initial_metadata=initial,
            claim=claim,
        )


class _FakeClient:
    def __init__(
        self, responses: list[primitives.HttpResponse] | None = None
    ) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def request(self, **kwargs: Any) -> primitives.HttpResponse:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError(f"unexpected provider call: {kwargs}")
        return self.responses.pop(0)


def _json_response(status: int, value: Mapping[str, Any]) -> primitives.HttpResponse:
    return primitives.HttpResponse(status, {}, subject.canonical_bytes(value))


def _done_operation(
    *, name: str, instance_name: str, operation_type: str
) -> dict[str, Any]:
    return {
        "name": name,
        "id": "12345",
        "status": "DONE",
        "operationType": operation_type,
        "targetLink": (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{transport.PROJECT}/zones/{transport.ZONE}/instances/"
            f"{instance_name}"
        ),
        "zone": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{transport.PROJECT}/zones/{transport.ZONE}"
        ),
    }


def test_wait_operation_is_exact_and_rejects_wrong_target() -> None:
    instance = "r2d-step11-fixture"
    expected = (
        "https://compute.googleapis.com/compute/v1/projects/"
        f"{transport.PROJECT}/zones/{transport.ZONE}/instances/{instance}"
    )
    official_target = _done_operation(
        name="operation-fixture",
        instance_name=instance,
        operation_type="insert",
    )
    official_target["targetLink"] = official_target["targetLink"].replace(
        "compute.googleapis.com", "www.googleapis.com"
    )
    result = subject.wait_zone_operation(
        client=_FakeClient(),
        initial=official_target,
        expected_instance_url=expected,
    )
    assert result["status"] == "DONE"
    changed = _done_operation(
        name="operation-fixture",
        instance_name="other-instance",
        operation_type="insert",
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        subject.wait_zone_operation(
            client=_FakeClient(),
            initial=changed,
            expected_instance_url=expected,
        )


def test_generation_pinned_backend_can_use_separate_list_credential() -> None:
    prefix = "gs://fixture-bucket/exact/tree"
    uri = f"{prefix}/DONE.json"
    raw = b'{"done":true}'
    metadata = {
        "bucket": "fixture-bucket",
        "name": "exact/tree/DONE.json",
        "generation": "7",
        "metageneration": "2",
        "size": str(len(raw)),
        "crc32c": "AAAAAA==",
        "etag": "etag-7",
    }
    list_client = _FakeClient(
        [
            _json_response(
                200,
                {
                    "items": [metadata],
                },
            )
        ]
    )
    read_client = _FakeClient(
        [
            _json_response(200, metadata),
            primitives.HttpResponse(200, {}, raw),
        ]
    )
    backend = subject.GenerationPinnedGcsBackend(
        read_client=read_client,
        list_client=list_client,
        allowed_prefixes=[prefix],
    )
    rows = list(backend.list_prefix(prefix))
    assert rows == [
        {
            "uri": uri,
            "generation": 7,
            "metageneration": 2,
            "bytes": len(raw),
            "sha256": subject._sha256_bytes(raw),
            "crc32c": "AAAAAA==",
            "etag": "etag-7",
        }
    ]
    assert backend.read_bytes(uri, 7) == raw
    assert len(read_client.calls) == 2
    with pytest.raises(ValueError, match="escaped exact tree"):
        backend.list_prefix("gs://fixture-bucket/other")


def test_done_monitor_fails_fast_when_exact_vm_is_terminal(
    contract: dict[str, Any],
) -> None:
    preview = contract["adapter_preview"]
    instance_name = contract["metadata_binding"]["instance_name"]
    result_reader = _FakeClient(
        [
            primitives.HttpResponse(404, {}, b""),
            primitives.HttpResponse(404, {}, b""),
        ]
    )
    compute_reader = _FakeClient(
        [
            _json_response(
                200,
                {
                    "name": instance_name,
                    "status": "TERMINATED",
                },
            )
        ]
    )
    sleeps: list[float] = []
    with pytest.raises(RuntimeError, match="TERMINATED before publishing DONE"):
        subject.wait_for_done(
            client=result_reader,
            preview=preview,
            instance_client=compute_reader,
            instance_name=instance_name,
            timeout_seconds=60,
            now=lambda: 1.0,
            sleep=sleeps.append,
        )
    assert sleeps == []
    assert len(result_reader.calls) == 2
    assert compute_reader.calls == [
        {
            "method": "GET",
            "url": subject._instance_url(instance_name),
        }
    ]


def test_done_monitor_accepts_done_published_as_vm_enters_stopping(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    preview = contract["adapter_preview"]
    instance_name = contract["metadata_binding"]["instance_name"]
    done = {"job_id": "candidate-shard-00", "fixture": True}
    record = {
        "uri": preview["jobs"][0]["done_uri"],
        "generation": 10,
    }
    pinned_observations = iter(
        [
            None,
            (record, subject.canonical_bytes(done)),
        ]
    )
    done_reads: list[dict[str, Any]] = []

    def read_done(**kwargs: Any) -> tuple[dict[str, Any], bytes] | None:
        done_reads.append(kwargs)
        return next(pinned_observations)

    monkeypatch.setattr(subject, "read_generation_pinned_object", read_done)
    monkeypatch.setattr(
        subject.adapter,
        "build_receive",
        lambda value, *, done_records: {
            "preview": value,
            "done_records": done_records,
        },
    )
    compute_reader = _FakeClient(
        [
            _json_response(
                200,
                {
                    "name": instance_name,
                    "status": "STOPPING",
                },
            )
        ]
    )
    sleeps: list[float] = []
    observed_record, receive = subject.wait_for_done(
        client=_FakeClient(),
        preview=preview,
        instance_client=compute_reader,
        instance_name=instance_name,
        timeout_seconds=60,
        now=lambda: 1.0,
        sleep=sleeps.append,
    )
    assert observed_record == record
    assert receive["done_records"] == [done]
    assert len(done_reads) == 2
    assert sleeps == []
    assert compute_reader.calls == [
        {
            "method": "GET",
            "url": subject._instance_url(instance_name),
        }
    ]


def test_done_monitor_accepts_done_before_querying_terminal_instance(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    preview = contract["adapter_preview"]
    done = {"job_id": "candidate-shard-00", "fixture": True}
    record = {
        "uri": preview["jobs"][0]["done_uri"],
        "generation": 9,
    }
    monkeypatch.setattr(
        subject,
        "read_generation_pinned_object",
        lambda **_kwargs: (record, subject.canonical_bytes(done)),
    )
    monkeypatch.setattr(
        subject.adapter,
        "build_receive",
        lambda value, *, done_records: {
            "preview": value,
            "done_records": done_records,
        },
    )
    compute_reader = _FakeClient()
    observed_record, receive = subject.wait_for_done(
        client=_FakeClient(),
        preview=preview,
        instance_client=compute_reader,
        instance_name=contract["metadata_binding"]["instance_name"],
        timeout_seconds=60,
        now=lambda: 1.0,
        sleep=lambda _seconds: None,
    )
    assert observed_record == record
    assert receive["done_records"] == [done]
    assert compute_reader.calls == []


def test_done_monitor_polls_explicit_direct_uri_not_legacy_preview_uri(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    preview = contract["adapter_preview"]
    legacy_done_uri = preview["jobs"][0]["done_uri"]
    direct_done_uri = contract["remote_layout"]["jobs"][0]["done_uri"]
    done = {"job_id": "candidate-shard-00", "fixture": True}
    record = {"uri": direct_done_uri, "generation": 11}
    polled_uris: list[str] = []

    def read_done(**kwargs: Any) -> tuple[dict[str, Any], bytes]:
        polled_uris.append(kwargs["uri"])
        return record, subject.canonical_bytes(done)

    monkeypatch.setattr(subject, "read_generation_pinned_object", read_done)
    monkeypatch.setattr(
        subject.adapter,
        "build_receive",
        lambda value, *, done_records: {
            "preview": value,
            "done_records": done_records,
        },
    )
    observed_record, receive = subject.wait_for_done(
        client=_FakeClient(),
        preview=preview,
        done_uri=direct_done_uri,
        timeout_seconds=60,
        now=lambda: 1.0,
        sleep=lambda _seconds: None,
    )
    assert direct_done_uri != legacy_done_uri
    assert polled_uris == [direct_done_uri]
    assert observed_record == record
    assert receive["preview"] is preview
    assert receive["done_records"] == [done]


def test_production_result_layout_uses_direct_namespace_not_adapter_preview(
    contract: dict[str, Any],
) -> None:
    checked = transport.validate_job_contract(contract)
    jobs = subject._exact_direct_result_jobs(checked)
    assert jobs == tuple(checked["remote_layout"]["jobs"])
    assert jobs[0]["done_uri"] != checked["adapter_preview"]["jobs"][0][
        "done_uri"
    ]
    assert jobs[0]["tree_prefix"] != checked["adapter_preview"]["jobs"][0][
        "tree_prefix"
    ]
    assert "/hu-m31-r2diag-direct-v1/" in jobs[0]["done_uri"]
    assert "/hu-m31-r2diag-worker-v1/" not in jobs[0]["done_uri"]


def test_shared_project_live_evidence_accepts_only_exact_bounded_bindings(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    gate = subject.iam_gate.build_step11_gate_plan(
        contract,
        issued_at_unix_seconds=1_000,
        expires_at_unix_seconds=8_000,
        nat_router_resource=(
            "projects/ofc-solver-485418/regions/asia-northeast1/"
            "routers/ofc-t3-nat-router-asia-northeast1"
        ),
    )
    worker_bindings = gate["iam_contract"]["worker_bindings"]
    controller_bindings = gate["iam_contract"]["controller_bindings"]
    worker = worker_bindings[0]["member"]
    controller_member = controller_bindings[0]["member"]
    initiating = "user:fixture-owner@example.com"

    def policy_binding(
        role: str,
        member: str,
        expression: str,
        title: str,
    ) -> dict[str, Any]:
        return {
            "role": role,
            "members": [member],
            "condition": {"title": title, "expression": expression},
        }

    project_bindings = [
        policy_binding(
            worker_bindings[2]["role"],
            worker,
            worker_bindings[2]["condition"]["expression"],
            "worker-delete",
        ),
        *[
            policy_binding(
                binding["role"],
                controller_member,
                binding["condition"]["expression"],
                f"controller-{index}",
            )
            for index, binding in enumerate(controller_bindings[:3])
        ],
        policy_binding(
            "roles/serviceusage.serviceUsageConsumer",
            controller_member,
            controller_bindings[0]["condition"]["expression"],
            "service-usage",
        ),
        {
            "role": "roles/editor",
            "members": [
                subject.iam_gate.DEFAULT_COMPUTE_SERVICE_ACCOUNT_PRINCIPAL,
                subject.iam_gate.CLOUD_SERVICES_SERVICE_ACCOUNT_PRINCIPAL,
            ],
        },
    ]
    source = gate["source_contract"]
    base = f"projects/_/buckets/{source['bucket']}/objects/"
    package_resource = base + source["package_object_prefix"] + "/"
    result_resource = base + source["result_object_prefix"] + "/"
    time_expression = controller_bindings[0]["condition"]["expression"]
    bucket_bindings = [
        policy_binding(
            binding["role"],
            binding["member"],
            binding["condition"]["expression"],
            binding["condition"]["title"],
        )
        for binding in worker_bindings[:2]
    ] + [
        policy_binding(
            f"projects/{transport.PROJECT}/roles/ofcM31T3ObjectReaderV1",
            controller_member,
            (
                f'(resource.name.startsWith("{package_resource}") || '
                f'resource.name.startsWith("{result_resource}")) && '
                f"{time_expression}"
            ),
            "controller-read",
        )
    ]
    custom_roles = {
        key: {
            "name": expected["name"],
            "includedPermissions": expected["permissions"],
            "stage": expected["stage"],
            "deleted": False,
        }
        for key, expected in gate["iam_contract"]["custom_roles"].items()
    }
    evidence = subject.seal_shared_project_live_evidence(
        {
            "schema": subject.SHARED_PROJECT_LIVE_EVIDENCE_SCHEMA,
            "collected_at_unix_seconds": 1_490,
            "collected_via_get_only": True,
            "cloud_mutation_performed": False,
            "project": transport.PROJECT,
            "bucket": source["bucket"],
            "worker_service_account": transport.WORKER_SERVICE_ACCOUNT,
            "controller_service_account": (
                primitives.CONTROLLER_SERVICE_ACCOUNT
            ),
            "initiating_principal": initiating,
            "project_policy": {"bindings": project_bindings},
            "bucket_policy": {"bindings": bucket_bindings},
            "worker_service_account_policy": {
                "bindings": [
                    policy_binding(
                        "roles/iam.serviceAccountUser",
                        controller_member,
                        controller_bindings[3]["condition"]["expression"],
                        "worker-actas",
                    )
                ]
            },
            "controller_service_account_policy": {
                "bindings": [
                    policy_binding(
                        "roles/iam.serviceAccountTokenCreator",
                        initiating,
                        controller_bindings[0]["condition"]["expression"],
                        "owner-impersonation",
                    )
                ]
            },
            "custom_roles": custom_roles,
            "enabled_services": [
                "compute.googleapis.com",
                "iamcredentials.googleapis.com",
                "serviceusage.googleapis.com",
                "storage.googleapis.com",
            ],
        }
    )
    monkeypatch.setattr(
        subject,
        "validate_authoritative_preflight",
        lambda **_kwargs: {
            "observation_bundle": {
                "facts": {
                    "bucket_and_iam": {
                        "qualifying_conditioned_bindings": [
                            {
                                "role": binding["role"],
                                "member": binding["member"],
                                "condition": binding["condition"],
                            }
                            for binding in worker_bindings[:2]
                        ],
                        "uniform_bucket_level_access_enabled": True,
                    },
                    "expected_instance_absence": [
                        {
                            "instance_name": contract["metadata_binding"][
                                "instance_name"
                            ],
                            "absent": True,
                        }
                    ],
                    "regional_capacity": {"available_vcpu": 24},
                }
            }
        },
    )
    assert subject.validate_shared_project_live_evidence(
        evidence,
        transport_contract=contract,
        actual_preflight_artifact={},
        gate_plan=gate,
        now_unix_seconds=1_500,
    ) == evidence
    tampered = deepcopy(evidence)
    tampered["bucket_policy"]["bindings"].append(
        policy_binding(
            f"projects/{transport.PROJECT}/roles/ofcM31T3ResultCreatorV1",
            controller_member,
            f'resource.name.startsWith("{package_resource}")',
            "unexpected-create",
        )
    )
    unsigned = dict(tampered)
    unsigned.pop("evidence_sha256")
    tampered["evidence_sha256"] = subject.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="targeted bucket IAM"):
        subject.validate_shared_project_live_evidence(
            tampered,
            transport_contract=contract,
            actual_preflight_artifact={},
            gate_plan=gate,
            now_unix_seconds=1_500,
        )


def test_execution_requires_explicit_flag_confirmation_and_injected_client(
    contract: dict[str, Any],
    launch_contract: dict[str, Any],
    key: primitives.EphemeralControllerKey,
    tmp_path: Path,
) -> None:
    common = {
        "execution_confirmation": subject.EXECUTION_CONFIRMATION,
        "client": None,
        "collector_client": None,
        "transport_contract": contract,
        "launch_contract": launch_contract,
        "signer": key,
        "package_provision_receipt": {},
        "outer_root": tmp_path,
        "actual_preflight_artifact": {},
        "gate_plan": {},
        "gate_observation": {},
        "live_iam_evidence": None,
        "post_claim_callback": None,
        "startup_path": _startup(),
        "prebootstrap_path": _prebootstrap(),
        "destination_root": tmp_path / "receive",
    }
    with pytest.raises(PermissionError):
        subject.execute_step11_attempt0(execute=False, **common)
    with pytest.raises(PermissionError):
        subject.execute_step11_attempt0(execute=True, **common)
    assert not (tmp_path / "receive").exists()


def _provider_instance(
    *,
    instance_name: str,
    metadata_items: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "id": "987654321098",
        "name": instance_name,
        "status": "RUNNING",
        "metadata": {
            "fingerprint": "metadata-fingerprint",
            "items": metadata_items,
        },
        "serviceAccounts": [
            {
                "email": transport.WORKER_SERVICE_ACCOUNT,
                "scopes": [transport.REQUIRED_WORKER_OAUTH_SCOPE],
            }
        ],
        "networkInterfaces": [{"accessConfigs": []}],
        "scheduling": {
            "provisioningModel": "SPOT",
            "instanceTerminationAction": "DELETE",
            "automaticRestart": False,
            "onHostMaintenance": "TERMINATE",
        },
    }


def _patch_authoritative_inputs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    generations = {
        row["uri"]: index + 10
        for index, row in enumerate(
            contract["remote_layout"]["package_inventory"]["records"]
        )
    }
    monkeypatch.setattr(
        subject,
        "validate_authoritative_preflight",
        lambda **_kwargs: {"artifact_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        subject,
        "validate_authoritative_gate",
        lambda **_kwargs: (
            {
                "plan_sha256": "b" * 64,
                "authorization_window": {
                    "issued_at_unix_seconds": 1_000,
                    "expires_at_unix_seconds": 10_000,
                },
            },
            {
                "observation_sha256": "c" * 64,
                "result_sha256": "d" * 64,
            },
        ),
    )
    receipt = {
        "receipt_sha256": "e" * 64,
        "package_generations": generations,
        "package_generations_sha256": subject.canonical_sha256(generations),
    }
    monkeypatch.setattr(
        subject,
        "validate_package_provision_receipt",
        lambda **_kwargs: receipt,
    )
    monkeypatch.setattr(
        subject,
        "wait_zone_operation",
        lambda **kwargs: {
            "name": kwargs["initial"]["name"],
            "status": "DONE",
        },
    )
    monkeypatch.setattr(
        subject,
        "wait_for_done",
        lambda **_kwargs: (
            {
                "uri": contract["adapter_preview"]["jobs"][0]["done_uri"],
                "generation": 99,
                "metageneration": 1,
                "bytes": 10,
                "sha256": "f" * 64,
                "crc32c": "AAAAAA==",
                "etag": "etag",
            },
            {"fixture_receive": True},
        ),
    )
    monkeypatch.setattr(
        subject.receiver,
        "materialize_and_validate_received_stage",
        lambda *_args, **_kwargs: {"validated": True},
    )
    monkeypatch.setattr(
        subject,
        "_validate_cloud_materialization",
        lambda **kwargs: dict(kwargs["value"]),
    )
    monkeypatch.setattr(
        subject,
        "wait_for_instance_absence",
        lambda **kwargs: {
            "instance_name": kwargs["instance_name"],
            "instance_url": subject._instance_url(kwargs["instance_name"]),
            "provider_get_status": 404,
            "query_count": 1,
            "all_expected_instances_absent": True,
        },
    )
    monkeypatch.setattr(
        subject,
        "_validate_claimed_provider_instance",
        lambda value, **kwargs: {
            "instance_id": kwargs["expected_instance_id"],
            "name": kwargs["expected_name"],
            "status": value.get("status"),
            "metadata_fingerprint": "claimed-fingerprint",
            "initial_metadata_sha256": "f" * 64,
            "service_account": transport.WORKER_SERVICE_ACCOUNT,
            "oauth_scopes": [transport.REQUIRED_WORKER_OAUTH_SCOPE],
            "external_access_configs": [],
            "provisioning_model": "SPOT",
            "claimed_metadata_sha256": "e" * 64,
            "claim_present": True,
        },
    )
    return receipt


def test_full_fake_lifecycle_sends_valid_insert_then_claim_only_cas(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
    launch_contract: dict[str, Any],
    key: primitives.EphemeralControllerKey,
    tmp_path: Path,
) -> None:
    _patch_authoritative_inputs(monkeypatch, contract=contract)
    done_poll_calls: list[dict[str, Any]] = []
    legacy_done_uri = contract["adapter_preview"]["jobs"][0]["done_uri"]
    direct_job = contract["remote_layout"]["jobs"][0]
    done_uri = direct_job["done_uri"]

    def wait_for_done_with_collector(
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        done_poll_calls.append(kwargs)
        assert kwargs["instance_client"] is fake
        assert kwargs["instance_name"] == instance_name
        assert kwargs["preview"] is contract["adapter_preview"]
        assert kwargs["done_uri"] == done_uri
        return (
            {
                "uri": done_uri,
                "generation": 99,
                "metageneration": 1,
                "bytes": 10,
                "sha256": "f" * 64,
                "crc32c": "AAAAAA==",
                "etag": "etag",
            },
            {"fixture_receive": True},
        )

    monkeypatch.setattr(subject, "wait_for_done", wait_for_done_with_collector)
    materialization_backend_prefixes: list[tuple[str, ...]] = []

    def materialize_with_direct_backend(
        preview: Mapping[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        assert preview is contract["adapter_preview"]
        materialization_backend_prefixes.append(
            kwargs["backend"]._allowed_prefixes
        )
        return {"validated": True}

    monkeypatch.setattr(
        subject.receiver,
        "materialize_and_validate_received_stage",
        materialize_with_direct_backend,
    )
    auth = primitives.build_controller_authorization(
        contract=contract,
        external_preflight_receipt_sha256="a" * 64,
        issued_unix_seconds=1_000,
        expires_unix_seconds=2_000,
        nonce="b" * 64,
        signer=key,
    )
    monkeypatch.setattr(
        subject.controller,
        "build_controller_authorization",
        lambda **_kwargs: auth,
    )
    insert = _insert_body(
        contract=contract,
        launch_contract=launch_contract,
        authorization=auth,
        key=key,
    )
    instance_name = contract["metadata_binding"]["instance_name"]
    fake = _FakeClient(
        [
            _json_response(
                200,
                _done_operation(
                    name="insert-operation",
                    instance_name=instance_name,
                    operation_type="insert",
                ),
            ),
            _json_response(
                200,
                _provider_instance(
                    instance_name=instance_name,
                    metadata_items=insert["metadata"]["items"],
                ),
            ),
            _json_response(
                200,
                _done_operation(
                    name="metadata-operation",
                    instance_name=instance_name,
                    operation_type="setMetadata",
                ),
            ),
            _json_response(
                200,
                _provider_instance(
                    instance_name=instance_name,
                    metadata_items=insert["metadata"]["items"],
                ),
            ),
        ]
    )
    request_ids = [
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
        "33333333-3333-4333-8333-333333333333",
    ]
    callback_inputs: list[dict[str, Any]] = []
    collector = object()

    def post_claim_callback(
        callback_input: Mapping[str, Any],
    ) -> dict[str, Any]:
        callback_inputs.append(dict(callback_input))
        body = {
            "schema": subject.POST_CLAIM_REVOKE_SCHEMA,
            "status": "launch_and_worker_actas_removed_after_claim",
            "instance_name": callback_input["instance_name"],
            "provider_instance_id": callback_input[
                "provider_instance_id"
            ],
            "claim_sha256": callback_input["claim_sha256"],
            "launch_binding_removed": True,
            "worker_actas_binding_removed": True,
            "readback_verified": True,
            "cloud_mutation_performed": True,
        }
        return {
            **body,
            "receipt_sha256": subject.canonical_sha256(body),
        }

    receipt = subject.execute_step11_attempt0(
        execute=True,
        execution_confirmation=subject.EXECUTION_CONFIRMATION,
        client=fake,
        collector_client=collector,
        transport_contract=contract,
        launch_contract=launch_contract,
        signer=key,
        package_provision_receipt={},
        outer_root=tmp_path,
        actual_preflight_artifact={},
        gate_plan={},
        gate_observation={},
        live_iam_evidence=None,
        post_claim_callback=post_claim_callback,
        startup_path=_startup(),
        prebootstrap_path=_prebootstrap(),
        destination_root=tmp_path / "received",
        now_unix_seconds=lambda: 1_500,
        request_ids=request_ids,
    )
    assert receipt["provider_get_404_observed"] is True
    assert receipt["worker_self_delete_observed"] is True
    assert receipt["controller_cleanup_delete_used"] is False
    assert receipt["attempt_index"] == 0
    assert receipt["vm_count"] == 1
    assert receipt["current_profile_changed"] is False
    assert len(callback_inputs) == 1
    assert done_uri != legacy_done_uri
    assert [call["client"] for call in done_poll_calls] == [collector]
    assert materialization_backend_prefixes == [
        (direct_job["tree_prefix"],)
    ]
    assert receipt["post_claim_revoke_receipt_sha256"] is not None
    insert_sent = json.loads(fake.calls[0]["body"])
    assert set(insert_sent["metadata"]) == {"items"}
    assert "labels" not in insert_sent
    cas_sent = json.loads(fake.calls[2]["body"])
    before = {
        row["key"]: row["value"]
        for row in insert_sent["metadata"]["items"]
    }
    after = {row["key"]: row["value"] for row in cas_sent["items"]}
    assert set(after) == set(before) | {launch.POSTCREATE_CLAIM_METADATA_KEY}
    assert all(after[key] == value for key, value in before.items())


def test_failure_after_insert_uses_only_exact_controller_delete(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
    launch_contract: dict[str, Any],
    key: primitives.EphemeralControllerKey,
    tmp_path: Path,
) -> None:
    _patch_authoritative_inputs(monkeypatch, contract=contract)
    auth = primitives.build_controller_authorization(
        contract=contract,
        external_preflight_receipt_sha256="a" * 64,
        issued_unix_seconds=1_000,
        expires_unix_seconds=2_000,
        nonce="b" * 64,
        signer=key,
    )
    monkeypatch.setattr(
        subject.controller,
        "build_controller_authorization",
        lambda **_kwargs: auth,
    )
    insert = _insert_body(
        contract=contract,
        launch_contract=launch_contract,
        authorization=auth,
        key=key,
    )
    instance_name = contract["metadata_binding"]["instance_name"]
    fake = _FakeClient(
        [
            _json_response(
                200,
                _done_operation(
                    name="insert-operation",
                    instance_name=instance_name,
                    operation_type="insert",
                ),
            ),
            _json_response(
                200,
                _provider_instance(
                    instance_name=instance_name,
                    metadata_items=insert["metadata"]["items"],
                ),
            ),
            primitives.HttpResponse(412, {}, b""),
        ]
    )
    deletes: list[str] = []

    def delete(**kwargs: Any) -> dict[str, Any]:
        deletes.append(kwargs["instance_name"])
        return {
            "instance_name": kwargs["instance_name"],
            "delete_requested": True,
            "provider_get_status": 404,
        }

    monkeypatch.setattr(subject, "delete_exact_instance", delete)
    with pytest.raises(subject.Step11ExecutionError) as caught:
        subject.execute_step11_attempt0(
            execute=True,
            execution_confirmation=subject.EXECUTION_CONFIRMATION,
            client=fake,
            collector_client=None,
            transport_contract=contract,
            launch_contract=launch_contract,
            signer=key,
            package_provision_receipt={},
            outer_root=tmp_path,
            actual_preflight_artifact={},
            gate_plan={},
            gate_observation={},
            live_iam_evidence=None,
            post_claim_callback=None,
            startup_path=_startup(),
            prebootstrap_path=_prebootstrap(),
            destination_root=tmp_path / "received",
            now_unix_seconds=lambda: 1_500,
            request_ids=[
                "11111111-1111-4111-8111-111111111111",
                "22222222-2222-4222-8222-222222222222",
                "33333333-3333-4333-8333-333333333333",
            ],
        )
    assert deletes == [instance_name]
    assert caught.value.cleanup == {
        "instance_name": instance_name,
        "delete_requested": True,
        "provider_get_status": 404,
    }


def test_ambiguous_insert_transport_failure_still_deletes_exact_instance(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
    launch_contract: dict[str, Any],
    key: primitives.EphemeralControllerKey,
    tmp_path: Path,
) -> None:
    _patch_authoritative_inputs(monkeypatch, contract=contract)
    auth = primitives.build_controller_authorization(
        contract=contract,
        external_preflight_receipt_sha256="a" * 64,
        issued_unix_seconds=1_000,
        expires_unix_seconds=2_000,
        nonce="b" * 64,
        signer=key,
    )
    monkeypatch.setattr(
        subject.controller,
        "build_controller_authorization",
        lambda **_kwargs: auth,
    )
    fake = _FakeClient()

    def ambiguous_insert(**kwargs: Any) -> primitives.HttpResponse:
        fake.calls.append(kwargs)
        raise TimeoutError("response lost after possible provider acceptance")

    fake.request = ambiguous_insert  # type: ignore[method-assign]
    instance_name = contract["metadata_binding"]["instance_name"]
    deletes: list[str] = []

    def delete(**kwargs: Any) -> dict[str, Any]:
        deletes.append(kwargs["instance_name"])
        return {
            "instance_name": kwargs["instance_name"],
            "delete_requested": False,
            "already_absent": True,
            "provider_get_status": 404,
        }

    monkeypatch.setattr(subject, "delete_exact_instance", delete)
    with pytest.raises(subject.Step11ExecutionError) as caught:
        subject.execute_step11_attempt0(
            execute=True,
            execution_confirmation=subject.EXECUTION_CONFIRMATION,
            client=fake,
            collector_client=None,
            transport_contract=contract,
            launch_contract=launch_contract,
            signer=key,
            package_provision_receipt={},
            outer_root=tmp_path,
            actual_preflight_artifact={},
            gate_plan={},
            gate_observation={},
            live_iam_evidence=None,
            post_claim_callback=None,
            startup_path=_startup(),
            prebootstrap_path=_prebootstrap(),
            destination_root=tmp_path / "received",
            now_unix_seconds=lambda: 1_500,
            request_ids=[
                "11111111-1111-4111-8111-111111111111",
                "22222222-2222-4222-8222-222222222222",
                "33333333-3333-4333-8333-333333333333",
            ],
        )
    assert deletes == [instance_name]
    assert caught.value.cleanup == {
        "instance_name": instance_name,
        "delete_requested": False,
        "already_absent": True,
        "provider_get_status": 404,
    }
