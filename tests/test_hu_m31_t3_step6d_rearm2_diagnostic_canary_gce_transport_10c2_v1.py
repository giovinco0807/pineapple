from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_package as package_builder,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter as adapter,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1 as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan,
)

_TEST_RSA_N_HEX = (
    "a3d0294654b6af7e3117defa88604a05f914837ca47ac7265606fc2ece35b3f"
    "89c06dabe879a731df9614b50707eae9c3fc3ad8e02d595b28bc19c9a1fe4"
    "ec48a9b72018dad5f544e3e512836986b34eaa3d1c6f1da67c5a0f088276e"
    "ab85bb5a782b8a46c6e98e1967e860c8c47216b584feff4d82ff73bd8cc0e"
    "a50244bf24d8d799b70d2db589122c3278409ed541994a80641591fb9def1f"
    "6e8746306e15fdc3d9044bee17414fc23c0b8f0a17650fa525d0af7d0558c"
    "405ede44ff12217bda719decb8ffaecbfb1b5f4c3f8f309707180d40f11067"
    "936a3c2bf597445c1f9b7b82594944377fc1e76bb999ed17d14cabc41e0b5"
    "f97b937a3a4166469237"
)
_TEST_RSA_D_HEX = (
    "97649240c599f0a697889032ca46f32282084b5cae462a376bda18c6c91d28"
    "aee2c412f5732d72a6890b3a27a54feedfc8ae777f7f6f1562657711dbff0d"
    "fab14ce84969af157e0fa4eed8254b298a1d7187f8f75857bd251e2fa7236fd"
    "9e228081fefb545e110b9abf452bed6060704c6215a8ad621be4439209d883"
    "288867d74aa54c06c60fc4fcc6c29b0d6b42a19a5cf583ed509f21fa5c30"
    "c415c96bd94cf9500b92444d30e151077c98b0cc146b1b7e98960f9e39000"
    "4941e8b118b0f8747aaf9bd040f7cb92b9a2d73cb6f2bf40530349eee466ea"
    "df2a5da9aa684115bd2dbc4d468cba6e8bedd5ac435ef66ebce67a0446f4b"
    "7d7fac3b473e813b11"
)


def _test_public_key() -> dict[str, Any]:
    return subject.build_rsa_public_key_record(modulus_hex=_TEST_RSA_N_HEX)


def _test_sign(record_type: str, payload: bytes) -> str:
    modulus = int(_TEST_RSA_N_HEX, 16)
    private_exponent = int(_TEST_RSA_D_HEX, 16)
    width = (modulus.bit_length() + 7) // 8
    digest_info = bytes.fromhex(
        "3031300d060960864801650304020105000420"
    ) + hashlib.sha256(record_type.encode() + b"\0" + payload).digest()
    encoded = (
        b"\x00\x01"
        + b"\xff" * (width - len(digest_info) - 3)
        + b"\x00"
        + digest_info
    )
    signature = pow(
        int.from_bytes(encoded, "big"), private_exponent, modulus
    ).to_bytes(width, "big")
    return base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("r2diag-10c2-direct") / "package"
    package_builder.build_package(output_dir=target)
    return target


@pytest.fixture(scope="module")
def contract(package: Path) -> dict[str, Any]:
    return subject.build_job_contract(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        job_id=plan.STAGE1_JOB_IDS[0],
        controller_public_key_record=_test_public_key(),
    )


def test_runtime_stage_constants_match_frozen_plan() -> None:
    assert subject.STAGE1_ID == plan.STAGE1_ID
    assert subject.STAGE2_ID == plan.STAGE2_ID
    assert subject.STAGE1_RUN_NAME == plan.STAGE1_RUN_NAME
    assert subject.STAGE2_RUN_NAME == plan.STAGE2_RUN_NAME
    assert subject.STAGE1_JOB_IDS == plan.STAGE1_JOB_IDS
    assert subject.STAGE2_JOB_IDS == plan.STAGE2_JOB_IDS
    assert subject.STAGE1_HAND_INDICES == plan.STAGE1_HAND_INDICES
    assert subject.STAGE2_HAND_INDICES == plan.STAGE2_HAND_INDICES


def test_declared_bootstrap_closure_validates_contract_without_repo_imports(
    contract: dict[str, Any],
    package: Path,
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1].resolve()
    bootstrap = tmp_path / "bootstrap"
    for relative in (
        subject.BOOTSTRAP_RELATIVE,
        subject.TRANSPORT_RELATIVE,
        *subject.DIRECT_SUPPORT_RELATIVES,
    ):
        source = repository / relative
        destination = bootstrap / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    contract_path = tmp_path / "transport-contract.json"
    contract_path.write_bytes(subject.canonical_bytes(contract))
    program = r"""
import json
import pathlib
import sys

bootstrap = pathlib.Path(sys.argv[1]).resolve()
contract_path = pathlib.Path(sys.argv[2]).resolve()
repository = pathlib.Path(sys.argv[3]).resolve()
package = pathlib.Path(sys.argv[4]).resolve()
extracted = pathlib.Path(sys.argv[5]).resolve()
sys.path = [str(bootstrap / "src")] + [
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
with (package / "manifest.json").open("rb") as handle:
    inner_manifest = json.load(handle)
transport.safe_extract_worker_source(
    source_zip=(
        package
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
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            program,
            str(bootstrap),
            str(contract_path),
            str(repository),
            str(package),
            str(tmp_path / "extracted"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        subject.CONTRACT_SCHEMA,
        adapter.ADAPTER_SCHEMA,
        "extracted-runner-present-parent-numpy-not-required",
    ]


def test_completed_output_validation_uses_worker_venv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    output = work / "output"
    extracted = work / "extracted"
    venv_python = work / "venv" / "bin" / "python"
    output.mkdir(parents=True)
    (extracted / "src").mkdir(parents=True)
    venv_python.parent.mkdir(parents=True)
    venv_python.write_bytes(b"fixture")
    (work / "venv" / "pyvenv.cfg").write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    calls: list[tuple[list[str], dict[str, Any]]] = []
    original_is_symlink = Path.is_symlink

    def reject_python_symlink_probe(path: Path) -> bool:
        if path == venv_python:
            raise AssertionError("venv/bin/python symlinks must be accepted")
        return original_is_symlink(path)

    def fake_run(
        argv: list[str], **kwargs: Any
    ) -> SimpleNamespace:
        calls.append((list(argv), dict(kwargs)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(Path, "is_symlink", reject_python_symlink_probe)
    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    result = subject.validate_completed_output_with_worker_venv(output)
    assert result["runner_validate_completed_output_performed"] is True
    assert result["parent_interpreter_numpy_required"] is False
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[0] == str(venv_python.resolve())
    assert argv[1] == "-c"
    assert "sys.prefix!=sys.base_prefix" in argv[2]
    assert "runner.validate_completed_output" in argv[2]
    assert argv[3] == str(output.resolve())
    assert argv[4] == str((work / "venv").resolve())
    assert kwargs["cwd"] == extracted.resolve()
    assert kwargs["env"]["PYTHONPATH"] == str(
        (extracted / "src").resolve()
    )
    assert kwargs["check"] is False


def test_completed_output_validation_rejects_missing_venv_python(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    (work / "output").mkdir(parents=True)
    (work / "extracted" / "src").mkdir(parents=True)
    (work / "venv" / "bin").mkdir(parents=True)
    (work / "venv" / "pyvenv.cfg").write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation layout changed"):
        subject.validate_completed_output_with_worker_venv(
            work / "output"
        )


class FakeHttp:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses: list[subject.HttpResponse] = []

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None = None,
        timeout_seconds: int = 30,
    ) -> subject.HttpResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": body,
                "timeout_seconds": timeout_seconds,
            }
        )
        if not self.responses:
            raise AssertionError("unexpected fake HTTP request")
        return self.responses.pop(0)


class FakeVerifier:
    def __init__(self, accepted: bool) -> None:
        self.accepted = accepted
        self.calls: list[str] = []
        key = _test_public_key()
        self.key_id = key["key_id"]
        self.public_key_sha256 = subject.canonical_sha256(key)

    def verify(
        self, *, record_type: str, payload: bytes, signature: str
    ) -> bool:
        self.calls.append(record_type)
        return self.accepted


class FakeShutdown:
    def __init__(self) -> None:
        self.deadlines: list[int] = []

    def request_shutdown(self, *, deadline_seconds: int) -> None:
        self.deadlines.append(deadline_seconds)


def _complete_contract(package: Path) -> dict[str, Any]:
    wheel = {
        "path": f"wheels/{subject.EXPECTED_NUMPY_WHEEL_FILENAME}",
        "uri_suffix": f"wheels/{subject.EXPECTED_NUMPY_WHEEL_FILENAME}",
        "sha256": subject.EXPECTED_NUMPY_WHEEL_SHA256,
        "bytes": subject.EXPECTED_NUMPY_WHEEL_BYTES,
        "mode": "0644",
        "kind": "offline_numpy_cp311_manylinux_x86_64_wheel",
    }
    return subject.build_job_contract(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        job_id=plan.STAGE1_JOB_IDS[0],
        offline_wheel_record=wheel,
        controller_public_key_record=_test_public_key(),
    )


def _queue_valid_metadata(
    http: FakeHttp,
    contract: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> None:
    values = [
        subject.PROJECT,
        claim["instance_id"],
        contract["metadata_binding"]["instance_name"],
        f"projects/{claim['project_number']}/zones/{subject.ZONE}",
        subject.WORKER_SERVICE_ACCOUNT,
        subject.REQUIRED_WORKER_OAUTH_SCOPE + "\n",
        *contract["metadata_values"].values(),
    ]
    http.responses.extend(
        subject.HttpResponse(200, {"Metadata-Flavor": "Google"}, value.encode())
        for value in values
    )
    http.responses.append(
        subject.HttpResponse(
            200,
            {"Metadata-Flavor": "Google"},
            json.dumps(
                {
                    "access_token": "fixture-token",
                    "expires_in": 300,
                    "token_type": "Bearer",
                }
            ).encode(),
        )
    )


def _authorization_and_claim(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = contract["metadata_binding"]
    key_id = contract["authorization_contract"]["controller_key_id"]
    authorization = {
        "schema": subject.AUTHORIZATION_SCHEMA,
        "contract_sha256": subject.canonical_sha256(contract),
        "metadata_binding_sha256": contract["metadata_binding_sha256"],
        "direct_stage_identity_sha256": contract[
            "direct_stage_identity_sha256"
        ],
        "job_id": binding["job_id"],
        "attempt_index": binding["attempt_index"],
        "instance_name": binding["instance_name"],
        "controller_key_id": key_id,
        "external_preflight_receipt_sha256": "1" * 64,
        "allowed_operations": [
            "metadata_identity_read",
            "metadata_token_read",
            "generation_pinned_package_download",
            "generation_match_zero_result_upload",
            "result_readback",
            "compute_delete_self_after_done",
            "bounded_safety_shutdown_on_any_worker_failure",
        ],
        "issued_unix_seconds": 100,
        "expires_unix_seconds": 200,
        "nonce": "2" * 64,
        "signature": "",
    }
    unsigned_auth = {
        key: value for key, value in authorization.items() if key != "signature"
    }
    authorization["signature"] = _test_sign(
        "authorization", subject.canonical_bytes(unsigned_auth)
    )
    claim = {
        "schema": subject.CLAIM_SCHEMA,
        "authorization_sha256": subject.canonical_sha256(authorization),
        "contract_sha256": subject.canonical_sha256(contract),
        "project": subject.PROJECT,
        "project_number": "999999999999",
        "zone": subject.ZONE,
        "instance_name": binding["instance_name"],
        "instance_id": "123456789",
        "worker_service_account": subject.WORKER_SERVICE_ACCOUNT,
        "controller_key_id": key_id,
        "stage_id": binding["stage_id"],
        "job_id": binding["job_id"],
        "attempt_index": binding["attempt_index"],
        "package_generations": {
            row["uri"]: index
            for index, row in enumerate(
                contract["remote_layout"]["package_inventory"]["records"], 1
            )
        },
        "nonce": "3" * 64,
        "signature": "",
    }
    unsigned_claim = {
        key: value for key, value in claim.items() if key != "signature"
    }
    claim["signature"] = _test_sign(
        "claim", subject.canonical_bytes(unsigned_claim)
    )
    return authorization, claim


def test_outer_manifest_and_direct_identity_are_collision_separated(
    contract: dict[str, Any],
) -> None:
    checked = subject.validate_job_contract(contract)
    outer = checked["outer_package_manifest"]
    assert outer["complete_for_direct_v1"] is False
    assert len(outer["objects"]) == 10 + len(subject.DIRECT_SUPPORT_RELATIVES)
    layout = checked["remote_layout"]
    assert layout["package_prefix"].startswith(
        f"gs://{subject.BUCKET}/{subject.DIRECT_NAMESPACE}/packages/"
    )
    assert layout["stage_prefix"].startswith(
        f"gs://{subject.BUCKET}/{subject.DIRECT_NAMESPACE}/stages/"
    )
    assert not layout["stage_prefix"].startswith(layout["package_prefix"])
    assert not layout["package_prefix"].startswith(layout["stage_prefix"])
    assert checked["metadata_binding"]["image"] == {
        "project": "debian-cloud",
        "name": "debian-12-bookworm-v20260609",
        "id": "1449487925682397051",
        "self_link": subject.IMAGE_SELF_LINK,
        "family_resolution_permitted": False,
    }
    assert checked["metadata_binding"]["worker_service_account"] == (
        "ofc-m31-t3-diagnostic@ofc-solver-485418.iam.gserviceaccount.com"
    )
    assert checked["metadata_binding"]["zone"] == "asia-northeast1-b"
    assert checked["capabilities"]["cloud_executable"] is False
    assert checked["capabilities"]["launch_ready"] is False


def test_exact_wheel_record_completes_outer_payload_but_not_launch(
    package: Path,
) -> None:
    wheel = {
        "path": f"wheels/{subject.EXPECTED_NUMPY_WHEEL_FILENAME}",
        "uri_suffix": f"wheels/{subject.EXPECTED_NUMPY_WHEEL_FILENAME}",
        "sha256": subject.EXPECTED_NUMPY_WHEEL_SHA256,
        "bytes": subject.EXPECTED_NUMPY_WHEEL_BYTES,
        "mode": "0644",
        "kind": "offline_numpy_cp311_manylinux_x86_64_wheel",
    }
    value = subject.build_job_contract(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        job_id=plan.STAGE1_JOB_IDS[0],
        offline_wheel_record=wheel,
        controller_public_key_record=_test_public_key(),
    )
    assert value["outer_package_manifest"]["complete_for_direct_v1"] is True
    assert value["capabilities"]["cloud_executable"] is False
    assert value["capabilities"]["external_preflight_passed"] is False

    changed = dict(wheel)
    changed["sha256"] = "4" * 64
    with pytest.raises(ValueError, match="pinned"):
        subject.build_job_contract(
            package_dir=package,
            stage_id=plan.STAGE1_ID,
            job_id=plan.STAGE1_JOB_IDS[0],
            offline_wheel_record=changed,
        )


def test_retry_keeps_direct_stage_and_result_identity(
    package: Path,
) -> None:
    attempt0 = adapter.build_preview(
        package_dir=package, stage_id=plan.STAGE1_ID
    )
    snapshot0 = adapter.empty_snapshot(attempt0)
    contract0 = subject.build_job_contract(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        job_id=plan.STAGE1_JOB_IDS[0],
    )
    contract1 = subject.build_job_contract(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        job_id=plan.STAGE1_JOB_IDS[0],
        attempt_index=1,
        prior_preview=attempt0,
        prior_snapshot=snapshot0,
    )
    assert (
        contract0["direct_stage_identity_sha256"]
        == contract1["direct_stage_identity_sha256"]
    )
    assert (
        contract0["remote_layout"]["result_prefix"]
        == contract1["remote_layout"]["result_prefix"]
    )
    assert (
        contract0["remote_layout"]["attempt_control_prefix"]
        != contract1["remote_layout"]["attempt_control_prefix"]
    )
    assert (
        contract0["metadata_binding"]["instance_name"]
        != contract1["metadata_binding"]["instance_name"]
    )


def test_lifecycle_orders_every_heartbeat_before_done_and_self_delete(
    contract: dict[str, Any],
) -> None:
    operations = contract["lifecycle_plan"]["ordered_success_operations"]
    done_tree = next(
        index
        for index, row in enumerate(operations)
        if row["operation"] == "conditional_create_runner_done_tree"
    )
    done_envelope = next(
        index
        for index, row in enumerate(operations)
        if row["operation"] == "conditional_create_done_envelope"
    )
    delete = next(
        index
        for index, row in enumerate(operations)
        if row["operation"] == "request_compute_delete_self"
    )
    heartbeats = [
        index
        for index, row in enumerate(operations)
        if row["operation"] == "conditional_create_heartbeat"
    ]
    assert len(heartbeats) == 10
    assert max(heartbeats) < done_tree < done_envelope < delete
    assert all(
        row.get("if_generation_match") == 0
        for row in operations
        if row["operation"].startswith("conditional_create")
    )
    assert contract["lifecycle_plan"]["failure_policy"] == {
        "publish_done": False,
        "request_self_delete": False,
        "preserve_completed_objects": True,
        "bounded_local_shutdown_required": True,
        "shutdown_deadline_seconds": 120,
    }


def test_invalid_authorization_fails_before_http(
    contract: dict[str, Any], package: Path, tmp_path: Path
) -> None:
    complete_contract = _complete_contract(package)
    authorization, claim = _authorization_and_claim(complete_contract)
    authorization["signature"] = "bad"
    http = FakeHttp()
    shutdown = FakeShutdown()
    with pytest.raises(ValueError, match="trust verification"):
        subject.execute_authorized_worker(
            contract=complete_contract,
            authorization=authorization,
            claim=claim,
            verifier=FakeVerifier(False),
            client=http,
            shutdown_requester=shutdown,
            fresh_root=tmp_path / "must-not-exist",
            now_unix_seconds=150,
        )
    assert http.calls == []
    assert shutdown.deadlines == [120]


def test_valid_asymmetric_approval_binds_every_package_generation(
    contract: dict[str, Any],
) -> None:
    authorization, claim = _authorization_and_claim(contract)
    approval = subject.validate_controller_approval(
        contract=contract,
        authorization=authorization,
        claim=claim,
        verifier=subject.RsaSha256ControllerTrustVerifier(_test_public_key()),
        now_unix_seconds=150,
    )
    assert approval.contract_sha256 == subject.canonical_sha256(contract)
    assert set(approval.package_generations) == {
        row["uri"]
        for row in contract["remote_layout"]["package_inventory"]["records"]
    }
    changed = deepcopy(claim)
    changed["package_generations"].pop(next(iter(changed["package_generations"])))
    with pytest.raises(ValueError, match="generations"):
        subject.validate_controller_approval(
            contract=contract,
            authorization=authorization,
            claim=changed,
            verifier=subject.RsaSha256ControllerTrustVerifier(
                _test_public_key()
            ),
            now_unix_seconds=150,
        )


def test_metadata_identity_and_token_are_exact(
    contract: dict[str, Any],
) -> None:
    _authorization, claim = _authorization_and_claim(contract)
    http = FakeHttp()
    _queue_valid_metadata(http, contract, claim)
    result = subject.read_metadata_identity_and_token(
        client=http, contract=contract, claim=claim
    )
    assert result["access_token"] == "fixture-token"
    assert all(
        call["headers"] == {"Metadata-Flavor": "Google"}
        for call in http.calls
    )
    assert not any("Authorization" in call["headers"] for call in http.calls)
    assert any(
        call["url"].endswith(
            "/instance/service-accounts/default/scopes"
        )
        for call in http.calls
    )
    assert result["identity"][
        "instance/service-accounts/default/scopes"
    ] == subject.REQUIRED_WORKER_OAUTH_SCOPE


def test_metadata_identity_rejects_missing_cloud_platform_scope(
    contract: dict[str, Any],
) -> None:
    _authorization, claim = _authorization_and_claim(contract)
    http = FakeHttp()
    _queue_valid_metadata(http, contract, claim)
    scope_index = contract["rest_plan"]["metadata"]["identity_paths"].index(
        "/instance/service-accounts/default/scopes"
    )
    http.responses[scope_index] = subject.HttpResponse(
        200,
        {"Metadata-Flavor": "Google"},
        b"https://www.googleapis.com/auth/compute.readonly",
    )
    with pytest.raises(RuntimeError, match="metadata identity mismatch"):
        subject.read_metadata_identity_and_token(
            client=http, contract=contract, claim=claim
        )


def test_metadata_identity_rejects_more_than_one_scope(
    contract: dict[str, Any],
) -> None:
    _authorization, claim = _authorization_and_claim(contract)
    http = FakeHttp()
    _queue_valid_metadata(http, contract, claim)
    scope_index = contract["rest_plan"]["metadata"]["identity_paths"].index(
        "/instance/service-accounts/default/scopes"
    )
    http.responses[scope_index] = subject.HttpResponse(
        200,
        {"Metadata-Flavor": "Google"},
        (
            subject.REQUIRED_WORKER_OAUTH_SCOPE
            + "\nhttps://www.googleapis.com/auth/compute.readonly\n"
        ).encode(),
    )
    with pytest.raises(RuntimeError, match="metadata identity mismatch"):
        subject.read_metadata_identity_and_token(
            client=http, contract=contract, claim=claim
        )


def test_generation_pinned_download_is_fresh_and_hashed(
    tmp_path: Path,
) -> None:
    raw = b"immutable-package-object"
    record = {
        "uri": f"gs://{subject.BUCKET}/unit/object",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    http = FakeHttp()
    http.responses.append(subject.HttpResponse(200, {}, raw))
    destination = tmp_path / "fresh" / "object"
    result = subject.generation_pinned_download(
        client=http,
        access_token="token",
        record=record,
        generation=17,
        destination=destination,
    )
    assert destination.read_bytes() == raw
    assert result["generation"] == 17
    assert "generation=17" in http.calls[0]["url"]
    with pytest.raises(FileExistsError, match="fresh"):
        subject.generation_pinned_download(
            client=FakeHttp(),
            access_token="token",
            record=record,
            generation=17,
            destination=destination,
        )


def test_conditional_create_uses_generation_zero_and_reads_back() -> None:
    raw = b"exact-result-object"
    http = FakeHttp()
    http.responses.extend(
        [
            subject.HttpResponse(200, {}, b'{"generation":"23"}'),
            subject.HttpResponse(200, {}, raw),
        ]
    )
    result = subject.conditional_create_and_readback(
        client=http,
        access_token="token",
        uri=f"gs://{subject.BUCKET}/unit/result",
        content=raw,
    )
    assert result["created"] is True
    assert result["generation"] == 23
    assert "ifGenerationMatch=0" in http.calls[0]["url"]
    assert http.calls[0]["method"] == "POST"
    assert http.calls[1]["method"] == "GET"

    collision = FakeHttp()
    collision.responses.extend(
        [
            subject.HttpResponse(412, {}, b""),
            subject.HttpResponse(200, {}, b'{"generation":"7"}'),
            subject.HttpResponse(200, {}, b"different"),
        ]
    )
    with pytest.raises(FileExistsError, match="differs"):
        subject.conditional_create_and_readback(
            client=collision,
            access_token="token",
            uri=f"gs://{subject.BUCKET}/unit/result",
            content=raw,
        )


def test_self_delete_requires_done_readback_and_failure_shutdown_is_bounded(
    contract: dict[str, Any],
) -> None:
    http = FakeHttp()
    with pytest.raises(ValueError, match="DONE"):
        subject.request_compute_self_delete(
            client=http,
            access_token="token",
            contract=contract,
            done_readback={},
        )
    assert http.calls == []

    shutdown = FakeShutdown()
    result = subject.request_bounded_failure_shutdown(
        requester=shutdown,
        deadline_seconds=120,
        done_published=False,
    )
    assert result["self_delete_requested"] is False
    assert shutdown.deadlines == [120]
    with pytest.raises(ValueError, match="bounded"):
        subject.request_bounded_failure_shutdown(
            requester=shutdown,
            deadline_seconds=121,
            done_published=False,
        )
    post_done = subject.request_bounded_failure_shutdown(
        requester=shutdown,
        deadline_seconds=1,
        done_published=True,
    )
    assert post_done["done_published"] is True
    assert post_done["reason"] == "post_done_self_delete_failure"


def test_safe_extractor_uses_exact_allowlist_and_never_extractall(
    package: Path, tmp_path: Path
) -> None:
    manifest = package_builder.validate_package(package)
    result = subject.safe_extract_worker_source(
        source_zip=package / package_builder.SOURCE_NAME,
        inner_manifest=manifest,
        destination=tmp_path / "safe",
    )
    assert result["file_count"] == 60
    assert result["zip_extractall_used"] is False
    assert (
        tmp_path / "safe" / "src" / "ofc_regular" / "__init__.py"
    ).is_file()


def test_contract_tamper_and_path_escape_fail_closed(
    contract: dict[str, Any],
) -> None:
    changed = deepcopy(contract)
    changed["capabilities"]["launch_ready"] = True
    with pytest.raises(ValueError):
        subject.validate_job_contract(changed)

    outer = deepcopy(contract["outer_package_manifest"])
    outer["objects"][0]["path"] = "../escape"
    with pytest.raises(ValueError, match="escaped"):
        subject.validate_outer_package_manifest(outer)


@pytest.mark.parametrize(
    ("mutate", "rehash_field", "error"),
    [
        (
            lambda value: value["rest_plan"]["self_delete"].__setitem__(
                "url", "https://attacker.invalid/delete"
            ),
            "rest_plan_sha256",
            "REST plan",
        ),
        (
            lambda value: value["remote_layout"]["jobs"][0].__setitem__(
                "done_uri", "gs://pokerhu-ofc-solver-485418-training/evil/DONE"
            ),
            "metadata_binding_sha256",
            "remote layout",
        ),
        (
            lambda value: value["metadata_values"].__setitem__(
                "ofc-direct-binding-uri",
                "gs://pokerhu-ofc-solver-485418-training/evil/binding.json",
            ),
            "metadata_values_sha256",
            "metadata values",
        ),
        (
            lambda value: value["lifecycle_plan"][
                "ordered_success_operations"
            ][0].__setitem__(
                "uri", "gs://pokerhu-ofc-solver-485418-training/evil/result"
            ),
            "lifecycle_plan_sha256",
            "lifecycle",
        ),
        (
            lambda value: value["adapter_preview"]["remote_manifest"].__setitem__(
                "prefix", "https://attacker.invalid/inert-looking-preview"
            ),
            "adapter_preview_sha256",
            "network location",
        ),
    ],
)
def test_rehashed_nested_url_tamper_cannot_expand_transport(
    contract: dict[str, Any],
    mutate: Any,
    rehash_field: str | None,
    error: str,
) -> None:
    changed = deepcopy(contract)
    mutate(changed)
    if rehash_field is not None:
        payload_field = rehash_field.removesuffix("_sha256")
        changed[rehash_field] = subject.canonical_sha256(
            changed[payload_field]
        )
    with pytest.raises(ValueError, match=error):
        subject.validate_job_contract(changed)


def test_contract_bound_urllib_rejects_arbitrary_urls_methods_and_headers(
    contract: dict[str, Any],
) -> None:
    client = subject.UrllibHttpClient(contract=contract)
    assert not any(
        isinstance(handler, subject.urllib.request.ProxyHandler)
        for handler in client._metadata_opener.handlers
    )
    assert any(
        isinstance(handler, subject._NoRedirectHandler)
        for handler in client._metadata_opener.handlers
    )
    with pytest.raises(ValueError, match="escaped exact"):
        client.request(
            method="GET",
            url="http://metadata.google.internal.evil/computeMetadata/v1/instance/id",
            headers={"Metadata-Flavor": "Google"},
        )
    metadata_url = (
        contract["rest_plan"]["metadata"]["root"] + "/instance/id"
    )
    with pytest.raises(ValueError, match="escaped exact"):
        client.request(
            method="POST",
            url=metadata_url,
            headers={"Metadata-Flavor": "Google"},
            body=b"",
        )
    with pytest.raises(ValueError, match="headers"):
        client.request(
            method="GET",
            url=metadata_url,
            headers={
                "Metadata-Flavor": "Google",
                "Authorization": "Bearer stolen",
            },
        )
    template = contract["rest_plan"]["package_downloads"][0][
        "generation_pinned_url_template"
    ]
    with pytest.raises(ValueError, match="escaped exact"):
        client.request(
            method="GET",
            url=template.replace("{generation}", "0"),
            headers={"Authorization": "Bearer token"},
        )


def test_metadata_failure_requests_bounded_shutdown_after_valid_approval(
    package: Path, tmp_path: Path
) -> None:
    contract = _complete_contract(package)
    authorization, claim = _authorization_and_claim(contract)
    http = FakeHttp()
    http.responses.append(subject.HttpResponse(500, {}, b"no metadata"))
    shutdown = FakeShutdown()
    with pytest.raises(RuntimeError, match="metadata read"):
        subject.execute_authorized_worker(
            contract=contract,
            authorization=authorization,
            claim=claim,
            verifier=subject.RsaSha256ControllerTrustVerifier(
                _test_public_key()
            ),
            client=http,
            shutdown_requester=shutdown,
            fresh_root=tmp_path / "fresh",
            now_unix_seconds=150,
        )
    assert len(http.calls) == 1
    assert shutdown.deadlines == [120]


def test_nonfresh_root_failure_requests_bounded_shutdown(
    package: Path, tmp_path: Path
) -> None:
    contract = _complete_contract(package)
    authorization, claim = _authorization_and_claim(contract)
    http = FakeHttp()
    _queue_valid_metadata(http, contract, claim)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    shutdown = FakeShutdown()
    with pytest.raises(FileExistsError, match="fresh"):
        subject.execute_authorized_worker(
            contract=contract,
            authorization=authorization,
            claim=claim,
            verifier=subject.RsaSha256ControllerTrustVerifier(
                _test_public_key()
            ),
            client=http,
            shutdown_requester=shutdown,
            fresh_root=occupied,
            now_unix_seconds=150,
        )
    assert not http.responses
    assert shutdown.deadlines == [120]


def test_post_done_self_delete_failure_preserves_done_and_shuts_down(
    package: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _complete_contract(package)
    authorization, claim = _authorization_and_claim(contract)
    done_uri = contract["metadata_binding"]["job_result_layout"]["done_uri"]
    monkeypatch.setattr(
        subject,
        "read_metadata_identity_and_token",
        lambda **_kwargs: {
            "identity": {"instance_name": "fixture"},
            "access_token": "fixture-token",
        },
    )
    monkeypatch.setattr(
        subject,
        "download_outer_package",
        lambda **_kwargs: {"status": "fixture"},
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        subject, "run_downloaded_worker", lambda **_kwargs: output
    )
    monkeypatch.setattr(
        subject,
        "publish_validated_runner_output",
        lambda **_kwargs: {
            "done_readback": {
                "uri": done_uri,
                "sha256": "4" * 64,
                "generation": 9,
            }
        },
    )
    monkeypatch.setattr(
        subject,
        "request_compute_self_delete",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("self-delete unavailable")
        ),
    )
    shutdown = FakeShutdown()
    with pytest.raises(RuntimeError, match="self-delete unavailable") as caught:
        subject.execute_authorized_worker(
            contract=contract,
            authorization=authorization,
            claim=claim,
            verifier=subject.RsaSha256ControllerTrustVerifier(
                _test_public_key()
            ),
            client=FakeHttp(),
            shutdown_requester=shutdown,
            fresh_root=tmp_path / "worker",
            now_unix_seconds=150,
        )
    assert shutdown.deadlines == [120]
    assert any(
        '"done_published":true' in note
        and '"reason":"post_done_self_delete_failure"' in note
        for note in getattr(caught.value, "__notes__", [])
    )


def test_rsa_verifier_has_public_key_only_and_rejects_tampering(
    contract: dict[str, Any],
) -> None:
    authorization, _claim = _authorization_and_claim(contract)
    unsigned = {
        key: value
        for key, value in authorization.items()
        if key != "signature"
    }
    verifier = subject.RsaSha256ControllerTrustVerifier(_test_public_key())
    assert verifier.verify(
        record_type="authorization",
        payload=subject.canonical_bytes(unsigned),
        signature=authorization["signature"],
    )
    unsigned["job_id"] = "tampered"
    assert not verifier.verify(
        record_type="authorization",
        payload=subject.canonical_bytes(unsigned),
        signature=authorization["signature"],
    )
    trust = contract["authorization_contract"]
    assert trust["signature_algorithm"] == subject.RSA_SIGNATURE_ALGORITHM
    assert trust["controller_private_key_embedded"] is False
    assert trust["worker_shared_signing_secret_present"] is False


def test_bootstrap_wrapper_has_only_explicit_local_and_authorized_modes() -> None:
    script = (
        subject._REPO_ROOT
        / "scripts"
        / "bootstrap_hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_v1.sh"
    ).read_text(encoding="utf-8")
    assert "OFC_DIAGNOSTIC_10C2_LOCAL_PREFLIGHT" in script
    assert "gcloud " not in script
    assert "gsutil " not in script
    assert "curl " not in script
    assert "--controller-public-key" in script
    assert "controller-hmac-key" not in script
    assert "/usr/bin/timeout 120 /sbin/shutdown -h now" in script


def test_authorized_cli_preentry_failure_requests_shutdown_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[Any] = []

    class TrackingShutdown:
        def __init__(self) -> None:
            self.shutdown_attempted = False
            self.deadlines: list[int] = []
            instances.append(self)

        def request_shutdown(self, *, deadline_seconds: int) -> None:
            assert self.shutdown_attempted is False
            self.shutdown_attempted = True
            self.deadlines.append(deadline_seconds)

    monkeypatch.setattr(subject, "SubprocessShutdownRequester", TrackingShutdown)
    monkeypatch.setenv("OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER", "1")
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        subject.main(
            [
                "authorized-worker",
                "--contract",
                str(missing),
                "--authorization",
                str(missing),
                "--claim",
                str(missing),
                "--controller-public-key",
                str(missing),
                "--fresh-root",
                str(tmp_path / "worker"),
            ]
        )
    assert len(instances) == 1
    assert instances[0].deadlines == [120]


def test_authorized_cli_does_not_double_shutdown_after_execute_failure(
    contract: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization, claim = _authorization_and_claim(contract)
    instances: list[Any] = []

    class TrackingShutdown:
        def __init__(self) -> None:
            self.shutdown_attempted = False
            self.deadlines: list[int] = []
            instances.append(self)

        def request_shutdown(self, *, deadline_seconds: int) -> None:
            assert self.shutdown_attempted is False
            self.shutdown_attempted = True
            self.deadlines.append(deadline_seconds)

    def load_fixture(_path: Path, label: str) -> dict[str, Any]:
        return {
            "10c2 contract": contract,
            "controller authorization": authorization,
            "worker claim": claim,
            "controller public key": _test_public_key(),
        }[label]

    def fail_inside_execute(**kwargs: Any) -> dict[str, Any]:
        kwargs["shutdown_requester"].request_shutdown(deadline_seconds=120)
        raise RuntimeError("injected post-entry failure")

    public_key_path = tmp_path / "controller-public-key.json"
    public_key_path.write_bytes(b"x" * 512)
    monkeypatch.setattr(subject, "SubprocessShutdownRequester", TrackingShutdown)
    monkeypatch.setattr(subject, "_load_canonical_json", load_fixture)
    monkeypatch.setattr(subject, "execute_authorized_worker", fail_inside_execute)
    monkeypatch.setenv("OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER", "1")
    with pytest.raises(RuntimeError, match="post-entry failure"):
        subject.main(
            [
                "authorized-worker",
                "--contract",
                str(tmp_path / "contract.json"),
                "--authorization",
                str(tmp_path / "authorization.json"),
                "--claim",
                str(tmp_path / "claim.json"),
                "--controller-public-key",
                str(public_key_path),
                "--fresh-root",
                str(tmp_path / "worker"),
            ]
        )
    assert len(instances) == 1
    assert instances[0].deadlines == [120]


def test_module_entrypoint_reports_an_already_attempted_shutdown_to_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, Any]] = []

    class Completed:
        returncode = 0

    def fake_subprocess_run(
        argv: list[str],
        *,
        check: bool,
        capture_output: bool,
        timeout: int,
    ) -> Completed:
        calls.append(
            {
                "argv": argv,
                "check": check,
                "capture_output": capture_output,
                "timeout": timeout,
            }
        )
        return Completed()

    def fail_after_shutdown() -> int:
        requester = subject.SubprocessShutdownRequester(
            marker_path=tmp_path / "shutdown-requested"
        )
        requester.request_shutdown(deadline_seconds=120)
        raise RuntimeError("injected worker failure")

    monkeypatch.setattr(subject, "_SUBPROCESS_SHUTDOWN_ATTEMPTED", False)
    monkeypatch.setattr(subject.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(subject, "main", fail_after_shutdown)
    assert (
        subject._module_entrypoint()
        == subject.AUTHORIZED_WORKER_SHUTDOWN_ATTEMPTED_EXIT_CODE
    )
    assert calls == [
        {
            "argv": ["/sbin/shutdown", "-h", "now"],
            "check": False,
            "capture_output": True,
            "timeout": 120,
        }
    ]
    marker = tmp_path / "shutdown-requested"
    assert marker.read_bytes() == b"shutdown-requested\n"
    second = subject.SubprocessShutdownRequester(marker_path=marker)
    second.request_shutdown(deadline_seconds=120)
    assert len(calls) == 1
    assert "injected worker failure" in capsys.readouterr().err


def test_rejected_shutdown_removes_owned_marker_for_outer_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Rejected:
        returncode = 1

    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda *_args, **_kwargs: Rejected(),
    )
    marker = tmp_path / "shutdown-requested"
    requester = subject.SubprocessShutdownRequester(marker_path=marker)
    with pytest.raises(RuntimeError, match="request was rejected"):
        requester.request_shutdown(deadline_seconds=120)
    assert not marker.exists()
