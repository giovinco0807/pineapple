from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_v2
    as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_content_v2
    as source_content_v2,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP12_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step12_pair_v1_actual"
)
RUN_NONCE = "c3" * 32


def _read(name: str) -> dict[str, Any]:
    value = json.loads((STEP12_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def context() -> dict[str, Any]:
    candidate = _read("candidate_transport_contract.json")
    reference = _read("reference_transport_contract.json")
    public_key = _read("controller_public_key.json")
    sources = {
        path: (
            REPO_ROOT / "src" / Path(path)
        ).read_text(encoding="utf-8")
        for path in subject.REQUIRED_RUNTIME_SOURCE_PATHS
    }
    content_binding = (
        source_content_v2.build_bootstrap_source_content_binding(
            runtime_source_files=sources,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
        )
    )
    deployment = deployment_v2.build_deployment_contract(
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=public_key,
        run_nonce=RUN_NONCE,
        bootstrap_source_content_binding=content_binding,
    )
    plan = subject.build_bootstrap_source_plan(
        deployment_contract=deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=public_key,
        run_nonce=RUN_NONCE,
        runtime_source_files=sources,
    )
    return {
        "candidate": candidate,
        "reference": reference,
        "public_key": public_key,
        "content_binding": content_binding,
        "deployment": deployment,
        "sources": sources,
        "plan": plan,
    }


class _Store:
    def __init__(self, *, initially_present: bool = False) -> None:
        self.objects: dict[str, tuple[int, bytes]] = {}
        self.initially_present = initially_present
        self.calls: list[tuple[str, str, int | None]] = []

    def list_objects(self, *, prefix: str) -> list[str]:
        self.calls.append(("list", prefix, None))
        return [f"{prefix}/stale"] if self.initially_present else []

    def conditional_create(
        self,
        *,
        uri: str,
        content: bytes,
        if_generation_match: int,
    ) -> dict[str, Any]:
        self.calls.append(("create", uri, if_generation_match))
        assert if_generation_match == 0
        assert uri not in self.objects
        generation = 1_900_000_000_000_000 + len(self.objects)
        self.objects[uri] = (generation, content)
        import hashlib

        return {
            "uri": uri,
            "generation": generation,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "created": True,
        }

    def generation_pinned_get(
        self, *, uri: str, generation: int
    ) -> bytes:
        self.calls.append(("get", uri, generation))
        stored_generation, content = self.objects[uri]
        assert generation == stored_generation
        return content


def _provision(
    context: dict[str, Any], store: _Store
) -> subject.ValidatedBootstrapSourceProvision:
    return subject.provision_bootstrap_sources(
        source_plan=context["plan"],
        deployment_contract=context["deployment"],
        candidate_payload_contract=context["candidate"],
        reference_payload_contract=context["reference"],
        controller_public_key_record=context["public_key"],
        run_nonce=RUN_NONCE,
        prefix_observer=store,
        writer=store,
        reader=store,
    )


def test_plan_is_transparent_exact_three_object_direct_v2_source(
    context: dict[str, Any],
) -> None:
    plan = context["plan"]
    checked = subject.validate_bootstrap_source_plan(
        plan,
        deployment_contract=context["deployment"],
        candidate_payload_contract=context["candidate"],
        reference_payload_contract=context["reference"],
        controller_public_key_record=context["public_key"],
        run_nonce=RUN_NONCE,
    )
    expected_prefix = (
        "gs://pokerhu-ofc-solver-485418-training/"
        f"{deployment_v2.DIRECT_NAMESPACE}/bootstrap-sources/"
        f"{context['deployment']['deployment_contract_sha256']}"
    )
    assert checked["source_prefix"] == expected_prefix
    assert checked["object_count"] == subject.SOURCE_OBJECT_COUNT == 3
    assert [row["path"] for row in checked["objects"]] == [
        subject.RUNTIME_SOURCE_OBJECT_PATH,
        subject.CANDIDATE_PAYLOAD_OBJECT_PATH,
        subject.REFERENCE_PAYLOAD_OBJECT_PATH,
    ]
    assert checked["old_package_write_authorized"] is False
    assert checked["old_result_write_authorized"] is False
    assert checked["direct_v2_result_write_authorized"] is False
    assert len(
        subject.canonical_bytes(checked["objects"][0]["content"])
    ) > 262_144


def test_runtime_bundle_is_regular_file_only_and_materializes_safely(
    context: dict[str, Any], tmp_path: Path
) -> None:
    bundle = context["plan"]["objects"][0]["content"]
    checked = subject.validate_runtime_source_bundle(bundle)
    receipt = subject.materialize_runtime_source_bundle(
        checked, destination=tmp_path / "overlay"
    )
    assert receipt["file_count"] == len(subject.REQUIRED_RUNTIME_SOURCE_PATHS)
    assert receipt["symlink_count"] == 0
    for relative in subject.REQUIRED_RUNTIME_SOURCE_PATHS:
        path = tmp_path / "overlay" / Path(relative)
        assert path.is_file()
        assert not path.is_symlink()
    with pytest.raises(FileExistsError, match="fresh"):
        subject.materialize_runtime_source_bundle(
            checked, destination=tmp_path / "overlay"
        )


def test_provision_requires_empty_prefix_and_exact_create_readback(
    context: dict[str, Any],
) -> None:
    store = _Store()
    capability = _provision(context, store)
    receipt = capability.receipt()
    assert len(store.objects) == 3
    assert receipt["object_count"] == 3
    assert receipt["prefix_empty_before_upload"] is True
    assert receipt["if_generation_match"] == 0
    assert receipt["all_generation_bound"] is True
    assert receipt["all_bytes_and_sha256_read_back"] is True
    assert receipt["old_package_write_count"] == 0
    assert receipt["old_result_write_count"] == 0
    assert [call[0] for call in store.calls] == [
        "list",
        "create",
        "get",
        "create",
        "get",
        "create",
        "get",
    ]

    blocked = _Store(initially_present=True)
    with pytest.raises(ValueError, match="not empty"):
        _provision(context, blocked)
    assert blocked.objects == {}


@pytest.mark.parametrize("position", [0, 1])
def test_role_manifest_contains_shared_and_only_own_payload(
    context: dict[str, Any], position: int
) -> None:
    capability = _provision(context, _Store())
    external_job_id = context["deployment"]["selected_job_ids"][position]
    manifest = subject.build_role_bootstrap_manifest(
        source_plan=context["plan"],
        validated_provision=capability,
        deployment_contract=context["deployment"],
        candidate_payload_contract=context["candidate"],
        reference_payload_contract=context["reference"],
        controller_public_key_record=context["public_key"],
        run_nonce=RUN_NONCE,
        external_job_id=external_job_id,
    )
    checked = subject.validate_role_bootstrap_manifest(
        manifest,
        source_plan=context["plan"],
        validated_provision=capability,
        deployment_contract=context["deployment"],
        candidate_payload_contract=context["candidate"],
        reference_payload_contract=context["reference"],
        controller_public_key_record=context["public_key"],
        run_nonce=RUN_NONCE,
        external_job_id=external_job_id,
    )
    envelope = subject.validate_role_bootstrap_manifest_envelope(
        checked,
        expected_deployment_contract_sha256=context["deployment"][
            "deployment_contract_sha256"
        ],
        expected_external_job_id=external_job_id,
        expected_source_role=checked["source_role"],
    )
    assert envelope == checked
    assert checked["object_count"] == 2
    assert checked["one_role_payload_only"] is True
    assert checked["opponent_role_payload_present"] is False
    assert {row["kind"] for row in checked["objects"]} == {
        "shared_runtime_source_bundle",
        f"{checked['source_role']}_role_payload_contract",
    }
    assert all(type(row["generation"]) is int for row in checked["objects"])


def test_worker_role_manifest_envelope_tamper_fails_closed(
    context: dict[str, Any],
) -> None:
    capability = _provision(context, _Store())
    external_job_id = context["deployment"]["selected_job_ids"][0]
    manifest = subject.build_role_bootstrap_manifest(
        source_plan=context["plan"],
        validated_provision=capability,
        deployment_contract=context["deployment"],
        candidate_payload_contract=context["candidate"],
        reference_payload_contract=context["reference"],
        controller_public_key_record=context["public_key"],
        run_nonce=RUN_NONCE,
        external_job_id=external_job_id,
    )
    changed = copy.deepcopy(manifest)
    changed["objects"][1]["kind"] = "reference_role_payload_contract"
    body = dict(changed)
    body.pop("role_manifest_sha256")
    changed["objects_sha256"] = subject.canonical_sha256(
        changed["objects"]
    )
    body = dict(changed)
    body.pop("role_manifest_sha256")
    changed["role_manifest_sha256"] = subject.canonical_sha256(body)
    with pytest.raises(ValueError, match="manifest object"):
        subject.validate_role_bootstrap_manifest_envelope(changed)


def test_tampered_source_plan_and_forged_capability_fail_closed(
    context: dict[str, Any],
) -> None:
    changed = copy.deepcopy(context["plan"])
    changed["objects"][0]["content"]["records"][0]["source"] += (
        "\nTAMPER = True\n"
    )
    body = dict(changed)
    body.pop("source_plan_sha256")
    changed["source_plan_sha256"] = subject.canonical_sha256(body)
    with pytest.raises(ValueError, match="runtime source"):
        subject.validate_bootstrap_source_plan(
            changed,
            deployment_contract=context["deployment"],
            candidate_payload_contract=context["candidate"],
            reference_payload_contract=context["reference"],
            controller_public_key_record=context["public_key"],
            run_nonce=RUN_NONCE,
        )

    with pytest.raises(ValueError, match="cannot be forged"):
        subject.ValidatedBootstrapSourceProvision(
            deployment_contract_sha256=context["deployment"][
                "deployment_contract_sha256"
            ],
            source_plan_sha256=context["plan"]["source_plan_sha256"],
            source_prefix=context["plan"]["source_prefix"],
            provision_receipt_sha256="11" * 32,
            records_sha256="22" * 32,
            generations_sha256="33" * 32,
            _receipt_bytes=b"{}",
            _seal=object(),
        )


def test_missing_or_extra_runtime_source_is_rejected(
    context: dict[str, Any],
) -> None:
    missing = dict(context["sources"])
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="closure"):
        subject.build_runtime_source_bundle(missing)
    extra = dict(context["sources"])
    extra["ofc_regular/extra.py"] = "VALUE = 1\n"
    with pytest.raises(ValueError, match="closure"):
        subject.build_runtime_source_bundle(extra)
