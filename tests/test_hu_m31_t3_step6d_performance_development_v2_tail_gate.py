from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_tail_gate as subject,
)


class FakeReceiveTransport:
    def __init__(self, objects: Mapping[str, bytes], listings: Mapping[str, Sequence[str]]):
        self.objects = dict(objects)
        self.listings = {key: list(value) for key, value in listings.items()}
        self.calls: list[tuple[str, str]] = []

    def get_object(self, *, object_name: str) -> bytes | None:
        self.calls.append(("GET", object_name))
        return self.objects.get(object_name)

    def list_objects(self, *, prefix: str) -> Sequence[str]:
        self.calls.append(("LIST", prefix))
        return list(self.listings.get(prefix, ()))


def _fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any], dict[str, Any], FakeReceiveTransport]:
    plan = {
        "run_name": "regular-hu-m31-c02-perfdev-v2-test-tail",
        "result_prefix": "hu-m31-t3/perfdev-v2/test-tail/",
        "instances": [
            {
                "source_role": "candidate",
                "instance_name": "candidate-test",
                "ownership_label": "pdv2-11111111111111111111",
            },
            {
                "source_role": "reference",
                "instance_name": "reference-test",
                "ownership_label": "pdv2-11111111111111111111",
            },
        ],
        "project": "test-project",
        "zone": "test-zone",
    }
    view = SimpleNamespace(
        plan=plan,
        plan_sha256="a" * 64,
        source_sha256="b" * 64,
        startup_sha256="c" * 64,
    )
    authorization = {"authorization": "test"}
    authorization_sha = subject.cloud.canonical_sha256(authorization)
    objects: dict[str, bytes] = {}
    listings: dict[str, list[str]] = {}
    collection_roles: list[dict[str, Any]] = []

    shared: dict[str, bytes] = {
        "run_contract.json": subject.cloud.canonical_bytes(
            subject.contract_v1.build_tail_run_contract()
        )
    }
    for index in subject.contract_v1.TAIL_HAND_INDICES:
        shared[f"roots/hand_{index:03d}.json"] = f"root-{index}\n".encode("ascii")

    for role in subject.cloud.SOURCE_ROLES:
        prefix = f"{plan['result_prefix']}results/{role}/"
        artifacts: list[dict[str, Any]] = []
        for relative in subject.cloud._required_artifact_paths(role):
            raw = shared.get(relative, f"{role}:{relative}\n".encode("ascii"))
            objects[f"{prefix}{relative}"] = raw
            artifacts.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                }
            )
        manifest = {"artifacts": artifacts}
        raw_manifest = subject.cloud.canonical_bytes(manifest)
        manifest_object = f"{prefix}RESULT_MANIFEST.json"
        objects[manifest_object] = raw_manifest
        listings[prefix] = sorted(
            [f"{prefix}{row['path']}" for row in artifacts] + [manifest_object]
        )
        heartbeat_prefix = f"{plan['result_prefix']}heartbeats/{role}/"
        progress_prefix = f"{plan['result_prefix']}progress/{role}/"
        listings[heartbeat_prefix] = [f"{heartbeat_prefix}000001.json"]
        listings[progress_prefix] = [
            f"{progress_prefix}run_contract.json",
            f"{progress_prefix}shard_manifest.json",
        ]
        collection_roles.append(
            {
                "source_role": role,
                "result_manifest_object": manifest_object,
                "result_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                "artifact_count": len(artifacts),
                "heartbeat_count": 1,
                "checkpoint_object_count": 2,
                "runner_validation_passed": True,
            }
        )
    unsigned_collection = {
        "schema": subject.cloud.COLLECTION_SCHEMA,
        "status": "complete_candidate_reference_pair_validated",
        "run_name": plan["run_name"],
        "execution_plan_sha256": view.plan_sha256,
        "authorization_sha256": authorization_sha,
        "run_contract_digest": subject.contract_v1.TAIL_RUN_CONTRACT_DIGEST,
        "run_contract_schema": subject.contract_v1.TAIL_RUN_CONTRACT_SCHEMA,
        "run_contract_variant": subject.contract_v1.TAIL_RUN_CONTRACT_VARIANT,
        "selection_manifest_sha256": (
            subject.contract_v1.TAIL_SELECTION_MANIFEST_SHA256
        ),
        "tail_hand_indices": list(subject.contract_v1.TAIL_HAND_INDICES),
        "roles": collection_roles,
        "instance_ownership": plan["instances"],
        "portable_pair_complete": True,
        "artifact_validation_passed": True,
        "partial_result": False,
        "cleanup_authorized": True,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    collection = {
        **unsigned_collection,
        "receipt_content_sha256": subject.cloud.canonical_sha256(unsigned_collection),
    }
    transport = FakeReceiveTransport(objects, listings)

    monkeypatch.setattr(subject.cloud, "_load_cloud_package", lambda *args, **kwargs: view)
    monkeypatch.setattr(
        subject.cloud,
        "_validate_collection_receipt",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        subject.cloud,
        "_validate_launch_authorization",
        lambda value, **kwargs: dict(value),
    )

    def strict_manifest(raw: bytes, **_: Any) -> dict[str, Any]:
        return json.loads(raw.decode("ascii"))

    monkeypatch.setattr(subject.cloud, "_strict_role_result_manifest", strict_manifest)
    monkeypatch.setattr(subject.runner, "validate_completed_output", lambda _: {})
    return view, collection, authorization, transport


def test_receive_is_get_list_only_and_offline_validation_rehashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _view, collection, authorization, transport = _fixture(monkeypatch)
    output = tmp_path / "received"
    receipt = subject.receive_pair_local(
        cloud_package_dir=tmp_path / "package",
        collection_receipt=collection,
        authorization=authorization,
        output_dir=output,
        transport=transport,
    )

    assert receipt["cloud_mutation_count"] == 0
    assert receipt["tail_hand_indices"] == list(subject.contract_v1.TAIL_HAND_INDICES)
    assert receipt["run_contract_digest"] == (
        subject.contract_v1.TAIL_RUN_CONTRACT_DIGEST
    )
    assert receipt["selection_manifest_sha256"] == (
        subject.contract_v1.TAIL_SELECTION_MANIFEST_SHA256
    )
    assert receipt["remote_method_surface"] == ["GET", "LIST"]
    assert {method for method, _target in transport.calls} == {"GET", "LIST"}
    candidate, reference, replayed = subject.validate_received_pair(
        receive_dir=output, cloud_package_dir=tmp_path / "package"
    )
    assert candidate == [output.resolve() / "sources/candidate/DONE.json"]
    assert reference == [output.resolve() / "sources/reference/DONE.json"]
    assert replayed == receipt


def test_offline_validation_rejects_artifact_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _view, collection, authorization, transport = _fixture(monkeypatch)
    output = tmp_path / "received"
    subject.receive_pair_local(
        cloud_package_dir=tmp_path / "package",
        collection_receipt=collection,
        authorization=authorization,
        output_dir=output,
        transport=transport,
    )
    (output / "sources/candidate/DONE.json").write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="artifact bytes changed"):
        subject.validate_received_pair(
            receive_dir=output, cloud_package_dir=tmp_path / "package"
        )


def test_receive_rejects_remote_extra_object_and_leaves_no_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    view, collection, authorization, transport = _fixture(monkeypatch)
    prefix = f"{view.plan['result_prefix']}results/candidate/"
    transport.listings[prefix].append(f"{prefix}unexpected.json")
    output = tmp_path / "received"

    with pytest.raises(ValueError, match="object topology changed"):
        subject.receive_pair_local(
            cloud_package_dir=tmp_path / "package",
            collection_receipt=collection,
            authorization=authorization,
            output_dir=output,
            transport=transport,
        )
    assert not output.exists()


def test_receive_rejects_collection_extension_before_remote_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _view, collection, authorization, transport = _fixture(monkeypatch)
    collection["unexpected"] = True

    with pytest.raises(ValueError, match="collection receipt fields changed"):
        subject.receive_pair_local(
            cloud_package_dir=tmp_path / "package",
            collection_receipt=collection,
            authorization=authorization,
            output_dir=tmp_path / "received",
            transport=transport,
        )
    assert transport.calls == []


def test_run_tail_gate_routes_only_to_candidate02_tail_v2_merger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate/DONE.json"
    reference = tmp_path / "reference/DONE.json"
    monkeypatch.setattr(
        subject,
        "validate_received_pair",
        lambda **_: ([candidate], [reference], {"status": subject.RECEIVE_STATUS}),
    )
    captured: dict[str, Any] = {}

    def merge(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        captured.update(kwargs)
        return (
            {
                "status": "no_go",
                "full_performance_development_authorized": False,
                "quality_pilot_authorized": False,
            },
            {"status": "no_go"},
        )

    monkeypatch.setattr(
        subject.merger, "merge_and_validate_candidate02_tail_v2", merge
    )
    summary, _validation = subject.run_tail_gate(
        receive_dir=tmp_path / "received",
        cloud_package_dir=tmp_path / "package",
        summary_output_path=tmp_path / "summary.json",
        validation_output_path=tmp_path / "validation.json",
    )

    assert captured["candidate_done_paths"] == [candidate]
    assert captured["reference_done_paths"] == [reference]
    assert summary["full_performance_development_authorized"] is False
    assert summary["quality_pilot_authorized"] is False


def test_old_mixed_geometry_run_contract_is_rejected_before_merge() -> None:
    plan = json.loads(subject.contract_v1.DEFAULT_FULL100_PLAN_PATH.read_text("utf-8"))
    old_raw = subject.cloud.canonical_bytes(plan["run_contract"])
    with pytest.raises(ValueError):
        subject._validate_tail_run_contract_bytes(old_raw)
