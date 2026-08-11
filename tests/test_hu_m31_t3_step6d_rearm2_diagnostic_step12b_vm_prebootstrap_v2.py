from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter
    as adapter,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as payload_plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1
    as step11_controller,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_content_v2
    as source_content,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_v2
    as bootstrap_source,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_external_authorization_v2
    as external_auth,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_startup_loader_v2
    as startup_loader,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_vm_metadata_v2
    as vm_metadata,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_vm_prebootstrap_v2
    as subject,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
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
OUTER_ROOT = STEP11_ROOT / "local_preflight" / "outer"
RUN_NONCE = "a7" * 32
AUTH_NONCE = "b8" * 32


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


class _SourceStore:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[int, bytes]] = {}

    def list_objects(self, *, prefix: str) -> list[str]:
        return []

    def conditional_create(
        self,
        *,
        uri: str,
        content: bytes,
        if_generation_match: int,
    ) -> dict[str, Any]:
        assert if_generation_match == 0
        generation = 1_900_000_000_000_000 + len(self.objects) + 1
        self.objects[uri] = (generation, content)
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
        stored_generation, content = self.objects[uri]
        assert stored_generation == generation
        return content


@pytest.fixture(scope="module")
def live_context() -> dict[str, Any]:
    signer = step11_controller.generate_ephemeral_controller_key(
        key_size=2_048
    )
    public_key = dict(signer.public_record)
    stage1 = _read(STEP11_ROOT / "transport_contract.json")
    stage1_done = _read(STEP11_ROOT / "late_done_envelope.json")
    stage1_receive = adapter.build_receive(
        stage1["adapter_preview"], done_records=[stage1_done]
    )
    wheel = next(
        row
        for row in stage1["outer_package_manifest"]["objects"]
        if row["kind"] == "offline_numpy_cp311_manylinux_x86_64_wheel"
    )
    payloads = [
        payload_transport.build_job_contract(
            package_dir=PACKAGE_DIR,
            stage_id=payload_plan.STAGE2_ID,
            job_id=job_id,
            attempt_index=0,
            offline_wheel_record=wheel,
            controller_public_key_record=public_key,
            prerequisite_stage1_preview=stage1["adapter_preview"],
            prerequisite_stage1_receive=stage1_receive,
        )
        for job_id in payload_plan.STAGE2_JOB_IDS
    ]
    runtime_sources = {
        path: (REPO_ROOT / "src" / Path(path)).read_text(
            encoding="utf-8"
        )
        for path in bootstrap_source.REQUIRED_RUNTIME_SOURCE_PATHS
    }
    content_binding = (
        source_content.build_bootstrap_source_content_binding(
            runtime_source_files=runtime_sources,
            candidate_payload_contract=payloads[0],
            reference_payload_contract=payloads[1],
        )
    )
    deployment = deployment_v2.build_deployment_contract(
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_key,
        run_nonce=RUN_NONCE,
        bootstrap_source_content_binding=content_binding,
    )
    source_plan = bootstrap_source.build_bootstrap_source_plan(
        deployment_contract=deployment,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_key,
        run_nonce=RUN_NONCE,
        runtime_source_files=runtime_sources,
    )
    store = _SourceStore()
    source_provision = bootstrap_source.provision_bootstrap_sources(
        source_plan=source_plan,
        deployment_contract=deployment,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_key,
        run_nonce=RUN_NONCE,
        prefix_observer=store,
        writer=store,
        reader=store,
    )
    manifests = [
        bootstrap_source.build_role_bootstrap_manifest(
            source_plan=source_plan,
            validated_provision=source_provision,
            deployment_contract=deployment,
            candidate_payload_contract=payloads[0],
            reference_payload_contract=payloads[1],
            controller_public_key_record=public_key,
            run_nonce=RUN_NONCE,
            external_job_id=job_id,
        )
        for job_id in deployment["selected_job_ids"]
    ]
    source_receipt = source_provision.receipt()
    role_summaries = [
        {
            "external_job_id": row["external_job_id"],
            "inner_job_id": row["inner_job_id"],
            "source_role": row["source_role"],
            "object_count": row["object_count"],
            "objects": copy.deepcopy(row["objects"]),
            "role_manifest_sha256": row["role_manifest_sha256"],
            "objects_sha256": row["objects_sha256"],
        }
        for row in manifests
    ]
    source_binding = {
        "schema": external_auth.BOOTSTRAP_SOURCE_BINDING_SCHEMA,
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "bootstrap_source_content_binding_sha256": content_binding[
            "bootstrap_source_content_binding_sha256"
        ],
        "source_plan_sha256": source_plan["source_plan_sha256"],
        "source_provision_receipt_sha256": source_receipt[
            "receipt_sha256"
        ],
        "source_prefix": source_plan["source_prefix"],
        "source_generations": source_receipt["source_generations"],
        "source_generations_sha256": source_receipt[
            "source_generations_sha256"
        ],
        "source_object_count": 3,
        "role_manifests": role_summaries,
        "role_manifests_sha256": subject.canonical_sha256(
            role_summaries
        ),
    }
    preflight = external_auth._build_external_preflight_receipt_record(
        deployment_contract=deployment,
        readonly_preflight_receipt_sha256=hashlib.sha256(
            b"readonly"
        ).hexdigest(),
        iam_capacity_gate_receipt_sha256=hashlib.sha256(
            b"iam"
        ).hexdigest(),
        package_provision_receipt_sha256=hashlib.sha256(
            b"package"
        ).hexdigest(),
        deployment_source_sha256=hashlib.sha256(
            b"deployment-source"
        ).hexdigest(),
        alias_bridge_source_sha256=hashlib.sha256(
            b"alias-source"
        ).hexdigest(),
        prebootstrap_source_sha256=hashlib.sha256(
            b"prebootstrap-source"
        ).hexdigest(),
        startup_source_sha256=startup_loader.startup_loader_sha256(),
        controller_source_sha256=hashlib.sha256(
            b"controller-source"
        ).hexdigest(),
        bootstrap_source_binding=source_binding,
        upstream_cloud_mutation_evidence_bound=True,
    )
    validated_preflight = (
        external_auth._mint_validated_external_preflight_after_gate(
            preflight,
            deployment_contract=deployment,
            gate_validation_seal=external_auth._VALIDATED_PREFLIGHT_SEAL,
        )
    )
    package_generations = {
        row["uri"]: 1_800_000_000_000_000 + position
        for position, row in enumerate(
            payloads[0]["remote_layout"]["package_inventory"]["records"],
            start=1,
        )
    }
    now = int(time.time())
    authorization = external_auth.build_external_authorization(
        deployment_contract=deployment,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_key,
        run_nonce=RUN_NONCE,
        external_job_id=deployment["selected_job_ids"][0],
        package_generations=package_generations,
        validated_external_preflight=validated_preflight,
        issued_unix_seconds=now - 60,
        expires_unix_seconds=now + 600,
        signer=signer,
        nonce=AUTH_NONCE,
    )
    claim = external_auth.build_external_worker_claim(
        authorization=authorization,
        deployment_contract=deployment,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_key,
        run_nonce=RUN_NONCE,
        external_job_id=deployment["selected_job_ids"][0],
        package_generations=package_generations,
        external_preflight_receipt=preflight,
        project_number=external_auth.EXPECTED_PROJECT_NUMBER,
        provider_instance_id="987654321012345678",
        metadata_fingerprint="AbCdEfGhIjKlMnOpQrStUvWxYz012345",
        now_unix_seconds=now,
        signer=signer,
        nonce="c9" * 32,
    )
    metadata = subject.build_role_initial_metadata(
        startup_script=startup_loader.build_startup_loader_source(),
        deployment_contract=deployment,
        selected_payload_contract=payloads[0],
        controller_public_key_record=public_key,
        authorization=authorization,
        package_generations=package_generations,
        external_preflight_receipt=preflight,
        run_nonce=RUN_NONCE,
        external_job_id=deployment["selected_job_ids"][0],
        role_bootstrap_manifest=manifests[0],
        verifier=payload_transport.RsaSha256ControllerTrustVerifier(
            public_key
        ),
        now_unix_seconds=now,
    )
    materialization_sha = "9a" * 32
    records = [
        {
            "kind": row["kind"],
            "path": row["path"],
            "uri": row["uri"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "generation": row["generation"],
            "downloaded": True,
            "readback_verified": True,
        }
        for row in manifests[0]["objects"]
    ]
    download_body = {
        "schema": subject.DOWNLOAD_RECEIPT_SCHEMA,
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "source_plan_sha256": source_plan["source_plan_sha256"],
        "source_provision_receipt_sha256": source_receipt[
            "receipt_sha256"
        ],
        "role_manifest_sha256": manifests[0]["role_manifest_sha256"],
        "external_job_id": manifests[0]["external_job_id"],
        "inner_job_id": manifests[0]["inner_job_id"],
        "source_role": manifests[0]["source_role"],
        "object_count": 2,
        "records": records,
        "records_sha256": subject.canonical_sha256(records),
        "runtime_source_bundle_sha256": manifests[0][
            "runtime_source_bundle_sha256"
        ],
        "runtime_source_materialization_receipt_sha256": (
            materialization_sha
        ),
        "payload_contract_sha256": hashlib.sha256(
            subject.canonical_bytes(payloads[0])
        ).hexdigest(),
        "generation_pinned_get_only": True,
        "opponent_role_payload_download_count": 0,
        "downloaded_before_full_prebootstrap_import": True,
        "repo_import_permitted": False,
        "site_packages_import_permitted": False,
    }
    download_receipt = {
        **download_body,
        "receipt_sha256": subject.canonical_sha256(download_body),
    }
    return {
        "payload": payloads[0],
        "payloads": payloads,
        "public_key": public_key,
        "deployment": deployment,
        "manifest": manifests[0],
        "metadata": metadata,
        "download_receipt": download_receipt,
        "authorization": authorization,
        "claim": claim,
        "preflight": preflight,
        "package_generations": package_generations,
        "now": now,
    }


def test_real_role_metadata_and_download_handoff_validate(
    live_context: dict[str, Any],
) -> None:
    checked = subject.validate_bootstrap_download_receipt(
        live_context["download_receipt"],
        role_bootstrap_manifest=live_context["manifest"],
        selected_payload_contract=live_context["payload"],
    )
    assert checked == live_context["download_receipt"]
    budget = vm_metadata.validate_initial_metadata_budget(
        live_context["metadata"]
    )
    assert budget["remaining_after_reserved_claim_and_release_bytes"] >= (
        vm_metadata.MIN_POSTCLAIM_HEADROOM_BYTES
    )
    assert vm_metadata.ROLE_PAYLOAD_CONTRACT_KEY not in (
        live_context["metadata"]
    )
    assert vm_metadata.RUNTIME_SOURCE_BUNDLE_KEY not in (
        live_context["metadata"]
    )


def test_startup_script_is_bound_to_validated_preflight(
    live_context: dict[str, Any],
) -> None:
    changed = dict(live_context["metadata"])
    changed[vm_metadata.STARTUP_KEY] += "\n# tampered after gate\n"
    with pytest.raises(
        ValueError,
        match="startup script escaped validated external preflight",
    ):
        subject.validate_role_initial_metadata(
            changed,
            selected_payload_contract=live_context["payload"],
            verifier=payload_transport.RsaSha256ControllerTrustVerifier(
                live_context["public_key"]
            ),
            now_unix_seconds=live_context["now"],
        )


def test_real_signed_role_metadata_validates_in_exact_isolated_bundle(
    live_context: dict[str, Any],
    tmp_path: Path,
) -> None:
    sources = {
        path: (REPO_ROOT / "src" / Path(path)).read_text(
            encoding="utf-8"
        )
        for path in bootstrap_source.REQUIRED_RUNTIME_SOURCE_PATHS
    }
    bundle = bootstrap_source.build_runtime_source_bundle(sources)
    runtime = tmp_path / "runtime"
    bootstrap_source.materialize_runtime_source_bundle(
        bundle, destination=runtime
    )
    metadata_path = tmp_path / "metadata.json"
    payload_path = tmp_path / "payload.json"
    metadata_path.write_bytes(
        subject.canonical_bytes(live_context["metadata"])
    )
    payload_path.write_bytes(
        subject.canonical_bytes(live_context["payload"])
    )
    program = r"""
import importlib, json, pathlib, sys
runtime = pathlib.Path(sys.argv[1]).resolve()
forbidden = pathlib.Path(sys.argv[2]).resolve()
assert sys.flags.isolated and "site" not in sys.modules
sys.path.insert(0, str(runtime))
for entry in sys.path[1:]:
    if not entry:
        continue
    assert "site-packages" not in entry.lower()
    resolved = pathlib.Path(entry).resolve()
    assert resolved != forbidden and forbidden not in resolved.parents
vm = importlib.import_module(
    "ofc_regular.hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_vm_prebootstrap_v2"
)
pt = importlib.import_module(
    "ofc_regular.hu_m31_t3_step6d_rearm2_diagnostic_"
    "canary_gce_transport_10c2_v1"
)
metadata = json.loads(pathlib.Path(sys.argv[3]).read_bytes())
payload = json.loads(pathlib.Path(sys.argv[4]).read_bytes())
public_key = json.loads(metadata["ofc-step12b-controller-public-key"])
receipt = vm.validate_role_initial_metadata(
    metadata,
    selected_payload_contract=payload,
    verifier=pt.RsaSha256ControllerTrustVerifier(public_key),
    now_unix_seconds=int(sys.argv[5]),
)
print(vm.canonical_bytes(receipt).decode("ascii"))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            program,
            str(runtime),
            str(REPO_ROOT),
            str(metadata_path),
            str(payload_path),
            str(live_context["now"]),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    receipt = json.loads(completed.stdout)
    assert receipt["phase"] == "role_local_initial_metadata_validated"
    assert receipt["one_role_payload_only"] is True


def test_download_handoff_tamper_fails_before_metadata_or_live_adapter(
    live_context: dict[str, Any],
) -> None:
    changed = copy.deepcopy(live_context["download_receipt"])
    changed["records"][1]["generation"] += 1
    body = dict(changed)
    body.pop("receipt_sha256")
    changed["records_sha256"] = subject.canonical_sha256(
        changed["records"]
    )
    body = dict(changed)
    body.pop("receipt_sha256")
    changed["receipt_sha256"] = subject.canonical_sha256(body)
    with pytest.raises(ValueError, match="download receipt record"):
        subject.validate_bootstrap_download_receipt(
            changed,
            role_bootstrap_manifest=live_context["manifest"],
            selected_payload_contract=live_context["payload"],
        )


def test_missing_pair_release_keeps_package_result_and_delete_at_zero(
    live_context: dict[str, Any],
    tmp_path: Path,
) -> None:
    deployment = live_context["deployment"]
    instance = deployment["instances"][0]

    class _Reader:
        release_reads = 0

        def read_initial_metadata_values(self) -> dict[str, str]:
            return dict(live_context["metadata"])

        def read_postcreate_claim(self) -> str:
            return subject.canonical_bytes(
                live_context["claim"]
            ).decode("ascii")

        def read_guest_identity(self) -> dict[str, Any]:
            return {
                "project_id": payload_transport.PROJECT,
                "project_number": external_auth.EXPECTED_PROJECT_NUMBER,
                "zone": payload_transport.ZONE,
                "instance_name": instance["instance_name"],
                "provider_instance_id": "987654321012345678",
                "service_account_email": (
                    payload_transport.WORKER_SERVICE_ACCOUNT
                ),
                "oauth_scopes": [
                    payload_transport.REQUIRED_WORKER_OAUTH_SCOPE
                ],
                "external_job_id": instance["job_id"],
                "source_role": instance["source_role"],
            }

        def read_pair_release(self) -> None:
            self.release_reads += 1
            return None

    class _SideEffects:
        calls = 0

        def generation_pinned_get(self, **_kwargs: Any) -> bytes:
            self.calls += 1
            raise AssertionError("package GET ran before pair release")

        def validate_and_run(self, **_kwargs: Any) -> Any:
            self.calls += 1
            raise AssertionError("payload ran before pair release")

        def conditional_create(self, **_kwargs: Any) -> Any:
            self.calls += 1
            raise AssertionError("result write ran before pair release")

        def request_exact_self_delete(self, **_kwargs: Any) -> Any:
            self.calls += 1
            raise AssertionError("self-delete ran before pair release")

    clock_values = iter([0.0, 1.0, 2.0])
    side_effects = _SideEffects()
    reader = _Reader()
    with pytest.raises(TimeoutError, match="pair release"):
        subject.execute_vm_prebootstrap(
            metadata_reader=reader,
            selected_payload_contract=live_context["payload"],
            verifier=(
                payload_transport.RsaSha256ControllerTrustVerifier(
                    live_context["public_key"]
                )
            ),
            package_reader=side_effects,
            payload_runtime=side_effects,
            result_writer=side_effects,
            self_deleter=side_effects,
            fresh_root=str(tmp_path / "must-not-exist"),
            wall_clock=live_context["now"],
            release_timeout_seconds=1,
            release_poll_seconds=0.1,
            monotonic=lambda: next(clock_values),
            sleep=lambda _seconds: None,
        )
    assert reader.release_reads == 1
    assert side_effects.calls == 0
    assert not (tmp_path / "must-not-exist").exists()


def test_live_entrypoint_constructs_self_contained_adapters(
    live_context: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Reader:
        def read_initial_metadata_values(self) -> dict[str, str]:
            return dict(live_context["metadata"])

    monkeypatch.setattr(subject, "UrllibGceMetadataTextClient", object)
    monkeypatch.setattr(
        subject, "GceGuestMetadataReader", lambda _client: _Reader()
    )
    monkeypatch.setattr(subject, "DEFAULT_LIVE_WORK_ROOT", tmp_path / "live")
    calls: list[dict[str, Any]] = []

    def _execute(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        assert isinstance(
            kwargs["package_reader"],
            subject.LiveGenerationPinnedPackageReader,
        )
        assert isinstance(
            kwargs["payload_runtime"], subject.LiveDownloadedPayloadRuntime
        )
        assert isinstance(
            kwargs["result_writer"], subject.LiveDirectV2ResultWriter
        )
        assert isinstance(
            kwargs["self_deleter"], subject.LiveExactSelfDeleter
        )
        return {"receipt_sha256": "cd" * 32}

    monkeypatch.setattr(subject, "execute_vm_prebootstrap", _execute)
    receipt = subject.run_downloaded_entrypoint(
        live_context["manifest"],
        live_context["payload"],
        live_context["download_receipt"],
    )
    assert len(calls) == 1
    assert receipt["generation_pinned_source_download_complete"] is True
    assert receipt["pair_release_before_package_download"] is True
    assert receipt["old_result_write_count"] == 0
    assert receipt["done_readback_before_self_delete"] is True


def test_live_package_reader_is_exact_generation_pinned(
    live_context: dict[str, Any],
) -> None:
    package_record = live_context["payload"]["remote_layout"][
        "package_inventory"
    ]["records"][0]
    content = (OUTER_ROOT / package_record["path"]).read_bytes()

    class _Metadata:
        calls = 0

        def read_worker_access_token(self) -> str:
            self.calls += 1
            return "worker-token"

    class _Http:
        calls: list[dict[str, Any]] = []

        def request(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return subject._WorkerApiResponse(
                status=200,
                url=kwargs["url"],
                headers={},
                body=content,
            )

    metadata = _Metadata()
    http = _Http()
    reader = subject.LiveGenerationPinnedPackageReader(
        metadata_reader=metadata,
        selected_payload_contract=live_context["payload"],
        http_client=http,
    )
    with pytest.raises(ValueError, match="escaped inventory"):
        reader.generation_pinned_get(
            uri="gs://pokerhu-ofc-solver-485418-training/wrong",
            generation=1,
        )
    assert metadata.calls == 0
    assert http.calls == []
    assert (
        reader.generation_pinned_get(
            uri=package_record["uri"], generation=1_234_567
        )
        == content
    )
    assert metadata.calls == 1
    assert "generation=1234567" in http.calls[0]["url"]


def test_live_result_writer_and_self_delete_are_exact_and_read_back(
    live_context: dict[str, Any],
) -> None:
    deployment = live_context["deployment"]
    job_id = deployment["selected_job_ids"][0]
    job = deployment["remote_layout"]["jobs"][0]

    class _Metadata:
        calls = 0

        def read_worker_access_token(self) -> str:
            self.calls += 1
            return "worker-token"

    class _Http:
        calls: list[dict[str, Any]] = []
        readback = b""

        def request(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            if kwargs["method"] == "POST":
                uri = job["tree_object_uris"][0]
                bucket, name = subject._gs_parts(uri)
                body = subject.canonical_bytes(
                    {
                        "bucket": bucket,
                        "name": name,
                        "generation": "7654321",
                        "size": str(len(kwargs["body"])),
                    }
                )
                self.readback = kwargs["body"]
                status = 200
            elif kwargs["method"] == "GET":
                body = self.readback
                status = 200
            else:
                body = b'{"name":"delete-operation"}'
                status = 202
            return subject._WorkerApiResponse(
                status=status,
                url=kwargs["url"],
                headers={},
                body=body,
            )

    metadata = _Metadata()
    http = _Http()
    writer = subject.LiveDirectV2ResultWriter(
        metadata_reader=metadata,
        deployment_contract=deployment,
        external_job_id=job_id,
        http_client=http,
    )
    with pytest.raises(ValueError, match="result create"):
        writer.conditional_create(
            uri="gs://pokerhu-ofc-solver-485418-training/old-result",
            content=b"blocked",
        )
    assert metadata.calls == 0
    content = b'{"direct_v2":true}'
    created = writer.conditional_create(
        uri=job["tree_object_uris"][0], content=content
    )
    assert created["created"] is True
    assert created["generation"] == 7_654_321
    assert metadata.calls == 2
    assert "ifGenerationMatch=0" in http.calls[0]["url"]

    deleter = subject.LiveExactSelfDeleter(
        metadata_reader=metadata,
        deployment_contract=deployment,
        external_job_id=job_id,
        http_client=http,
    )
    with pytest.raises(ValueError, match="prerequisite"):
        deleter.request_exact_self_delete(
            project=payload_transport.PROJECT,
            zone=payload_transport.ZONE,
            instance_name=deployment["instances"][0]["instance_name"],
            done_readback={"uri": "wrong"},
        )
    assert metadata.calls == 2
    done = {
        "uri": job["done_uri"],
        "generation": 8_765_432,
        "sha256": "ef" * 32,
    }
    deleted = deleter.request_exact_self_delete(
        project=payload_transport.PROJECT,
        zone=payload_transport.ZONE,
        instance_name=deployment["instances"][0]["instance_name"],
        done_readback=done,
    )
    assert deleted["delete_requested"] is True
    assert metadata.calls == 3
    assert http.calls[-1]["method"] == "DELETE"
