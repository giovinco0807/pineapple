from __future__ import annotations

import importlib.util
import json
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any

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
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1 as subject,
)


def _load_prebootstrap() -> Any:
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "prebootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "step11_prebootstrap_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("step11-controller") / "package"
    package_builder.build_package(output_dir=target)
    return target


@pytest.fixture(scope="module")
def key() -> subject.EphemeralControllerKey:
    return subject.generate_ephemeral_controller_key(key_size=2_048)


@pytest.fixture(scope="module")
def contract(
    package: Path, key: subject.EphemeralControllerKey
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


@pytest.fixture()
def authorization(
    contract: dict[str, Any], key: subject.EphemeralControllerKey
) -> dict[str, Any]:
    return subject.build_controller_authorization(
        contract=contract,
        external_preflight_receipt_sha256="a" * 64,
        issued_unix_seconds=1_000,
        expires_unix_seconds=2_000,
        nonce="b" * 64,
        signer=key,
    )


def _generations(contract: dict[str, Any]) -> dict[str, int]:
    return {
        row["uri"]: index + 1
        for index, row in enumerate(
            contract["remote_layout"]["package_inventory"]["records"]
        )
    }


def _claim(
    *,
    contract: dict[str, Any],
    authorization: dict[str, Any],
    key: subject.EphemeralControllerKey,
) -> dict[str, Any]:
    return subject.build_worker_claim(
        contract=contract,
        authorization=authorization,
        project_number="123456789012",
        instance_id="987654321098",
        package_generations=_generations(contract),
        nonce="c" * 64,
        signer=key,
    )


def test_ephemeral_private_key_is_not_json_or_pickle_serializable(
    key: subject.EphemeralControllerKey,
) -> None:
    assert key.public_record["private_key_present"] is False
    assert set(key.public_record) == {
        "schema",
        "algorithm",
        "exponent",
        "modulus_hex",
        "key_id",
        "private_key_present",
    }
    with pytest.raises(TypeError):
        json.dumps(key)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(key)


def test_authorization_and_post_create_claim_validate_end_to_end(
    contract: dict[str, Any],
    authorization: dict[str, Any],
    key: subject.EphemeralControllerKey,
) -> None:
    claim = _claim(
        contract=contract, authorization=authorization, key=key
    )
    checked = transport.validate_controller_approval(
        contract=contract,
        authorization=authorization,
        claim=claim,
        verifier=transport.RsaSha256ControllerTrustVerifier(
            key.public_record
        ),
        now_unix_seconds=1_500,
    )
    assert checked.package_generations == _generations(contract)
    assert authorization["allowed_operations"][-1] == (
        "bounded_safety_shutdown_on_any_worker_failure"
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["package_generations"].popitem(), "incomplete"),
        (
            lambda value: value.__setitem__("instance_id", "not-a-number"),
            "project number or instance ID",
        ),
    ],
)
def test_claim_builder_fails_closed_on_incomplete_or_invalid_binding(
    contract: dict[str, Any],
    authorization: dict[str, Any],
    key: subject.EphemeralControllerKey,
    mutation: Any,
    match: str,
) -> None:
    generations: dict[str, Any] = _generations(contract)
    values: dict[str, Any] = {
        "project_number": "123456789012",
        "instance_id": "987654321098",
        "package_generations": generations,
    }
    mutation(values)
    with pytest.raises(ValueError, match=match):
        subject.build_worker_claim(
            contract=contract,
            authorization=authorization,
            signer=key,
            nonce="c" * 64,
            **values,
        )


def test_package_plan_and_receipt_bind_every_generation(
    package: Path,
    contract: dict[str, Any],
    key: subject.EphemeralControllerKey,
    tmp_path: Path,
) -> None:
    # The exact offline wheel is not part of the generated package fixture, so
    # exercise content identity with the otherwise identical package-only form.
    package_contract = transport.build_job_contract(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        job_id=plan.STAGE1_JOB_IDS[0],
        controller_public_key_record=key.public_record,
    )
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "outer-manifest.json").write_bytes(
        transport.canonical_bytes(package_contract["outer_package_manifest"])
    )
    repo_root = Path(__file__).parents[1]
    for row in package_contract["outer_package_manifest"]["objects"]:
        relative = row["path"]
        source = (
            package / relative.removeprefix("inner/")
            if relative.startswith("inner/")
            else repo_root / relative
        )
        target = outer / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    package_plan = subject.build_package_provision_plan(
        contract=package_contract, outer_root=outer
    )
    readbacks = [
        {
            "uri": row["uri"],
            "generation": index + 10,
            "created": True,
            "sha256": row["sha256"],
            "bytes": row["bytes"],
            "crc32c": f"crc-{index}",
            "etag": f"etag-{index}",
        }
        for index, row in enumerate(package_plan["records"])
    ]
    receipt = subject.build_package_provision_receipt(
        plan=package_plan, readbacks=readbacks
    )
    unsigned = dict(receipt)
    receipt_sha = unsigned.pop("receipt_sha256")
    assert receipt_sha == subject.canonical_sha256(unsigned)
    assert receipt["package_generations"] == {
        row["uri"]: index + 10
        for index, row in enumerate(package_plan["records"])
    }


class _FakeGoogleClient:
    def __init__(self, responses: list[subject.HttpResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def request(self, **kwargs: Any) -> subject.HttpResponse:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class _TokenSource:
    controller_principal = subject.CONTROLLER_SERVICE_ACCOUNT

    @staticmethod
    def access_token() -> str:
        return "test-controller-token"


class _FakeHttpResponse:
    status = 201
    headers: dict[str, str] = {}

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    @staticmethod
    def read() -> bytes:
        return b"{}"


class _FakeOpener:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def open(self, request: Any, *, timeout: int) -> _FakeHttpResponse:
        assert timeout == 60
        self.requests.append(request)
        return _FakeHttpResponse()


def _metadata(*, generation: int, size: int, sha256: str) -> bytes:
    return subject.canonical_bytes(
        {
            "generation": str(generation),
            "size": str(size),
            "crc32c": "AAAAAA==",
            "etag": "etag",
            "metadata": {"sha256": sha256},
        }
    )


def test_conditional_create_accepts_201_and_generation_pinned_readback() -> None:
    content = b"immutable package object"
    client = _FakeGoogleClient(
        [
            subject.HttpResponse(
                201,
                {},
                _metadata(
                    generation=7,
                    size=len(content),
                    sha256=subject.sha256_bytes(content),
                ),
            ),
            subject.HttpResponse(200, {}, content),
        ]
    )
    result = subject.conditional_create_and_readback(
        client=client,  # type: ignore[arg-type]
        uri="gs://ofc-solver-artifacts/step11/object",
        content=content,
    )
    assert result["created"] is True
    assert result["generation"] == 7
    assert "uploadType=multipart" in client.calls[0]["url"]
    assert client.calls[0]["content_type"].startswith(
        "multipart/related; boundary=ofc-s11-"
    )
    assert subject.sha256_bytes(content).encode("ascii") in client.calls[0]["body"]
    assert "generation=7" in client.calls[1]["url"]


def test_google_json_client_accepts_only_generated_multipart_shape() -> None:
    content = b"immutable package object"
    body, content_type = subject._storage_multipart_upload(
        uri="gs://ofc-solver-artifacts/step11/object",
        content=content,
        sha256=subject.sha256_bytes(content),
    )
    client = subject.GoogleJsonClient(_TokenSource())
    opener = _FakeOpener()
    client._opener = opener  # type: ignore[assignment]
    response = client.request(
        method="POST",
        url=subject._storage_upload_url(
            "gs://ofc-solver-artifacts/step11/object"
        ),
        body=body,
        content_type=content_type,
    )
    assert response.status == 201
    assert len(opener.requests) == 1
    assert opener.requests[0].get_header("Content-type") == content_type

    with pytest.raises(ValueError, match="request body changed"):
        client.request(
            method="POST",
            url=subject._storage_upload_url(
                "gs://ofc-solver-artifacts/step11/object"
            ),
            body=body + b"tamper",
            content_type=content_type,
        )


def test_conditional_create_rejects_preexisting_different_bytes() -> None:
    content = b"expected"
    client = _FakeGoogleClient(
        [
            subject.HttpResponse(412, {}, b""),
            subject.HttpResponse(
                200,
                {},
                _metadata(
                    generation=9,
                    size=len(content),
                    sha256=subject.sha256_bytes(content),
                ),
            ),
            subject.HttpResponse(200, {}, b"different"),
        ]
    )
    with pytest.raises(FileExistsError, match="differs"):
        subject.conditional_create_and_readback(
            client=client,  # type: ignore[arg-type]
            uri="gs://ofc-solver-artifacts/step11/object",
            content=content,
        )


def test_prebootstrap_validates_signed_records_and_scope_before_token(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
    authorization: dict[str, Any],
    key: subject.EphemeralControllerKey,
) -> None:
    prebootstrap = _load_prebootstrap()
    claim = _claim(
        contract=contract, authorization=authorization, key=key
    )
    observed = {
        "/project/project-id": transport.PROJECT,
        "/instance/id": claim["instance_id"],
        "/instance/name": contract["metadata_binding"]["instance_name"],
        "/instance/zone": (
            f"projects/{claim['project_number']}/zones/{transport.ZONE}"
        ),
        "/instance/service-accounts/default/email": (
            transport.WORKER_SERVICE_ACCOUNT
        ),
        "/instance/service-accounts/default/scopes": (
            transport.REQUIRED_WORKER_OAUTH_SCOPE
        ),
        **{
            f"/instance/attributes/{name}": value
            for name, value in contract["metadata_values"].items()
        },
    }
    calls: list[str] = []

    def metadata_get(path: str) -> bytes:
        calls.append(path)
        return observed[path].encode("utf-8")

    monkeypatch.setattr(prebootstrap, "_metadata_get", metadata_get)
    _, records, generations = prebootstrap.validate_controller_records(
        contract=contract,
        authorization=authorization,
        claim=claim,
        public_key=key.public_record,
        now_unix_seconds=1_500,
    )
    assert generations == _generations(contract)
    assert len(records) == len(generations)
    assert "/instance/service-accounts/default/token" not in calls

    bad_scope = dict(observed)
    bad_scope["/instance/service-accounts/default/scopes"] = (
        "https://www.googleapis.com/auth/devstorage.read_write"
    )
    monkeypatch.setattr(
        prebootstrap,
        "_metadata_get",
        lambda path: bad_scope[path].encode("utf-8"),
    )
    with pytest.raises(RuntimeError, match="OAuth scope"):
        prebootstrap.validate_controller_records(
            contract=contract,
            authorization=authorization,
            claim=claim,
            public_key=key.public_record,
            now_unix_seconds=1_500,
        )


def test_prebootstrap_rejects_signature_tamper_before_metadata(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
    authorization: dict[str, Any],
    key: subject.EphemeralControllerKey,
) -> None:
    prebootstrap = _load_prebootstrap()
    bad = deepcopy(authorization)
    bad["nonce"] = "d" * 64
    monkeypatch.setattr(
        prebootstrap,
        "_metadata_get",
        lambda _path: pytest.fail("metadata must not be read before trust"),
    )
    with pytest.raises(ValueError, match="signature verification"):
        prebootstrap.validate_controller_records(
            contract=contract,
            authorization=bad,
            claim=_claim(
                contract=contract, authorization=authorization, key=key
            ),
            public_key=key.public_record,
            now_unix_seconds=1_500,
        )


def test_prebootstrap_requires_exact_canonical_files(tmp_path: Path) -> None:
    prebootstrap = _load_prebootstrap()
    target = tmp_path / "record.json"
    target.write_bytes(b'{"a":1}\n')
    with pytest.raises(ValueError, match="exact canonical"):
        prebootstrap._read_canonical(target, "fixture")
