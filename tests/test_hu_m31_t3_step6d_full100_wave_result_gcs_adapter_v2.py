from __future__ import annotations

import json
import os
import urllib.parse
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_result_gcs_adapter_v2 as subject
from ofc_regular import hu_m31_t3_step6d_full100_wave_result_receiver_v2 as receiver_v2
from ofc_regular.hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    HttpResponse,
)


PROJECT = "ofc-project-123"
ZONE = "asia-northeast1-b"
TOKEN_A = "token-a-012345678901234567890123456789"
TOKEN_B = "token-b-012345678901234567890123456789"


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def context() -> tuple[dict, dict, dict]:
    plan = wave_v2.build_wave_plan(
        run_name="regular-hu-m31-c02-f100wv2-20260722-005",
        identity_salt="0123456789abcdef0123456789abcdef",
        package_sha256="2" * 64,
        image_digest="sha256:" + "3" * 64,
    )
    baseline = wave_v2.build_observed_transition(
        plan,
        project_id=PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-22T00:00:00Z",
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(plan),
    )
    ledger = wave_v2.build_attempt_ledger(plan, transitions=[baseline])
    resume = wave_v2.build_resume_plan(plan, attempt_ledger=ledger)
    return plan, ledger, resume


def _acceptance_payload(context: tuple[dict, dict, dict], index: int = 0) -> tuple[str, bytes]:
    plan, ledger, resume = context
    selected = resume["selected_attempts"][index]
    path = plan["artifact_contract"]["job_acceptance_path_template"].format(
        job_id=selected["job_id"]
    )
    core = {
        "schema": receiver_v2.ACCEPTANCE_SCHEMA,
        "status": "done_last_generation_pinned_attempt_accepted",
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "observed_transition_digest": resume["observed_transition_digest"],
        "wave_index": resume["resume_wave_index"],
        "job_id": selected["job_id"],
        "source_role": selected["source_role"],
        "attempt_id": selected["attempt_id"],
        "instance_id": selected["instance_id"],
        "launch_receipt_sha256": _sha("launch-receipt"),
        "attempt_prefix": selected["artifact_prefix"],
        "done_path": f"{selected['artifact_prefix']}/DONE.json",
        "done_generation": 1234,
        "done_bytes": 4321,
        "done_sha256": _sha("done"),
        "done_identity_sha256": _sha("done-identity"),
        "artifact_count": 21,
        "artifact_records_sha256": _sha("artifact-records"),
        "done_observed_before_acceptance": True,
        "create_only": True,
        "current_profile_changed": False,
    }
    value = {
        **core,
        "acceptance_identity_sha256": wave_v2.canonical_sha256(core),
    }
    raw = wave_v2.canonical_bytes(value)
    assert receiver_v2.validate_acceptance_payload_context(
        plan, ledger, resume, path=path, payload=raw
    ) == value
    return path, raw


class FakeGcs:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[int, bytes]] = {}
        self.next_generation = 2000
        self.calls: list[dict] = []
        self.rotate_token_after_first = False
        self.raise_after_create = False
        self.return_extra_create_metadata = False
        self.fail_post_status: int | None = None
        self.error_body = b"provider-response-secret"
        self.repeat_page_token = False
        self.truncate_media_path: str | None = None

    def put(self, path: str, raw: bytes) -> int:
        self.next_generation += 1
        self.objects[path] = (self.next_generation, raw)
        return self.next_generation

    @staticmethod
    def _response(status: int, value: object | bytes) -> HttpResponse:
        raw = value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")
        return HttpResponse(status=status, body=raw, headers={})

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout_seconds: int,
    ) -> HttpResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body_sha256": (
                    None
                    if body is None
                    else __import__("hashlib").sha256(body).hexdigest()
                ),
                "timeout_seconds": timeout_seconds,
            }
        )
        if self.rotate_token_after_first and len(self.calls) == 1:
            os.environ[subject.TOKEN_ENV] = TOKEN_B
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path.startswith("/upload/storage/v1/b/"):
            assert method == "POST"
            assert query["fields"] == ["name,generation,size"]
            assert query["uploadType"] == ["media"]
            assert query["ifGenerationMatch"] == ["0"]
            path = query["name"][0]
            assert body is not None
            if path in self.objects:
                return self._response(412, self.error_body)
            generation = self.put(path, body)
            if self.raise_after_create:
                raise TimeoutError("transport-secret-must-not-leak")
            if self.fail_post_status is not None:
                return self._response(self.fail_post_status, self.error_body)
            metadata = {
                "name": path,
                "generation": str(generation),
                "size": str(len(body)),
            }
            if self.return_extra_create_metadata:
                metadata["kind"] = "storage#object"
            return self._response(200, metadata)

        marker = "/storage/v1/b/" + urllib.parse.quote(subject.BUCKET, safe="") + "/o"
        assert parsed.path.startswith(marker)
        suffix = parsed.path[len(marker) :]
        if suffix == "":
            assert method == "GET"
            prefix = query["prefix"][0]
            paths = sorted(path for path in self.objects if path.startswith(prefix))
            page = query.get("pageToken", [None])[0]
            if page is None and len(paths) > 1:
                selected_paths = paths[:1]
                next_token = "page-2"
            else:
                selected_paths = paths[1:] if page == "page-2" else paths
                next_token = "page-2" if self.repeat_page_token and page else None
            value: dict[str, object] = {
                "items": [
                    {
                        "name": path,
                        "generation": str(self.objects[path][0]),
                        "size": str(len(self.objects[path][1])),
                    }
                    for path in selected_paths
                ]
            }
            if next_token is not None:
                value["nextPageToken"] = next_token
            return self._response(200, value)

        path = urllib.parse.unquote(suffix.removeprefix("/"))
        if query.get("alt") == ["media"]:
            assert method == "GET"
            value = self.objects.get(path)
            if value is None:
                return self._response(404, self.error_body)
            generation, raw = value
            if query.get("ifGenerationMatch") != [str(generation)]:
                return self._response(412, self.error_body)
            if path == self.truncate_media_path:
                raw = raw[:-1]
            return self._response(200, raw)
        value = self.objects.get(path)
        if value is None:
            return self._response(404, self.error_body)
        generation, raw = value
        return self._response(
            200,
            {"name": path, "generation": str(generation), "size": str(len(raw))},
        )


def _store(
    context: tuple[dict, dict, dict], *, mode: str, fake: FakeGcs
) -> subject.GcsResultStoreV2:
    plan, ledger, resume = context
    return subject.GcsResultStoreV2(
        mode=mode,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        requester=fake,
    )


def test_read_poll_uses_exact_prefix_pagination_generation_and_rotating_env_token(
    monkeypatch: pytest.MonkeyPatch, context: tuple[dict, dict, dict]
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    plan, _, resume = context
    selected = resume["selected_attempts"][0]
    meta = {
        row["job_id"]: row for row in plan["full100_plan"]["jobs"]
    }[selected["job_id"]]
    hand = meta["work_hand_indices"][0]
    root_path = f"{selected['artifact_prefix']}/roots/hand_{hand:03d}.json"
    done_path = f"{selected['artifact_prefix']}/DONE.json"
    fake = FakeGcs()
    fake.put(root_path, b'{"artifact":"not-in-receipt-root"}')
    fake.put(done_path, b'{"artifact":"not-in-receipt-done"}')
    fake.rotate_token_after_first = True
    store = _store(context, mode="read", fake=fake)
    receipt = store.poll_selected()
    assert receipt["record_count"] == 2
    assert receipt["list_page_count"] == 9
    assert receipt["http_post_count"] == 0
    assert receipt["read_only"] is True
    assert TOKEN_A not in json.dumps(receipt)
    assert TOKEN_B not in json.dumps(receipt)
    assert "not-in-receipt" not in json.dumps(receipt)
    assert fake.calls[0]["headers"]["Authorization"] == f"Bearer {TOKEN_A}"
    assert all(
        call["headers"]["Authorization"] == f"Bearer {TOKEN_B}"
        for call in fake.calls[1:]
    )
    assert all(call["method"] == "GET" for call in fake.calls)
    assert subject.validate_poll_receipt(*context, receipt) == receipt
    assert not any("token" in key.lower() for key in vars(store))


def test_modes_paths_duplicate_pages_and_metadata_media_mismatch_fail_closed(
    monkeypatch: pytest.MonkeyPatch, context: tuple[dict, dict, dict]
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    plan, _, resume = context
    fake = FakeGcs()
    read_store = _store(context, mode="read", fake=fake)
    path, payload = _acceptance_payload(context)
    with pytest.raises(PermissionError, match="accept mode"):
        read_store.create_only(path=path, data=payload)
    with pytest.raises(ValueError, match="selected job prefixes"):
        read_store.list_prefix(prefix=plan["artifact_contract"]["prefix"] + "/")
    bad_hand = (
        resume["selected_attempts"][0]["artifact_prefix"]
        + "/roots/hand_999.json"
    )
    with pytest.raises(ValueError, match="artifact layout"):
        read_store.read_current(path=bad_hand, allow_missing=True)

    selected = resume["selected_attempts"][0]
    meta = {
        row["job_id"]: row for row in plan["full100_plan"]["jobs"]
    }[selected["job_id"]]
    hand = meta["work_hand_indices"][0]
    root_path = f"{selected['artifact_prefix']}/roots/hand_{hand:03d}.json"
    fake.put(root_path, b'{"root":true}')
    fake.truncate_media_path = root_path
    with pytest.raises(RuntimeError, match="size changed"):
        read_store.list_prefix(
            prefix=selected["artifact_prefix"].split("/attempts/", 1)[0] + "/"
        )

    fake = FakeGcs()
    fake.put(root_path, b'{"root":true}')
    fake.put(f"{selected['artifact_prefix']}/DONE.json", b'{"done":true}')
    fake.repeat_page_token = True
    read_store = _store(context, mode="read", fake=fake)
    with pytest.raises(RuntimeError, match="page token"):
        read_store.list_prefix(
            prefix=selected["artifact_prefix"].split("/attempts/", 1)[0] + "/"
        )


def test_missing_env_token_is_a_local_precondition_not_transport_ambiguity(
    monkeypatch: pytest.MonkeyPatch, context: tuple[dict, dict, dict]
) -> None:
    monkeypatch.delenv(subject.TOKEN_ENV, raising=False)
    fake = FakeGcs()
    path, _ = _acceptance_payload(context)
    with pytest.raises(PermissionError, match=subject.TOKEN_ENV):
        _store(context, mode="accept", fake=fake).read_current(
            path=path, allow_missing=True
        )
    assert fake.calls == []


def test_accept_create_is_generation_zero_exact_readback_and_receipt_redacted(
    monkeypatch: pytest.MonkeyPatch, context: tuple[dict, dict, dict]
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    fake = FakeGcs()
    store = _store(context, mode="accept", fake=fake)
    path, payload = _acceptance_payload(context)
    receipt = store.accept_and_readback(path=path, data=payload)
    assert receipt["provider_create_confirmed"] is True
    assert receipt["transport_ambiguity_reconciled"] is False
    assert receipt["payload_sha256"] == __import__("hashlib").sha256(payload).hexdigest()
    assert receipt["payload_bytes"] == len(payload)
    assert receipt["http_post_count"] == 1
    post = next(call for call in fake.calls if call["method"] == "POST")
    query = urllib.parse.parse_qs(urllib.parse.urlparse(post["url"]).query)
    assert query["fields"] == ["name,generation,size"]
    assert query["ifGenerationMatch"] == ["0"]
    assert query["uploadType"] == ["media"]
    assert query["name"] == [path]
    rendered = json.dumps(receipt)
    assert TOKEN_A not in rendered
    assert payload.decode("ascii") not in rendered
    assert subject.validate_accept_receipt(*context, receipt) == receipt

    non_boolean = dict(receipt)
    non_boolean["provider_create_confirmed"] = 1
    non_boolean["transport_ambiguity_reconciled"] = 0
    unsigned = {
        key: value for key, value in non_boolean.items() if key != "receipt_sha256"
    }
    non_boolean["receipt_sha256"] = wave_v2.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="semantics changed"):
        subject.validate_accept_receipt(*context, non_boolean)

    wrong_path = dict(receipt)
    wrong_path["record"] = {
        **wrong_path["record"],
        "path": context[2]["selected_attempts"][0]["artifact_prefix"]
        + "/DONE.json",
    }
    wrong_path["path"] = wrong_path["record"]["path"]
    unsigned = {
        key: value for key, value in wrong_path.items() if key != "receipt_sha256"
    }
    wrong_path["receipt_sha256"] = wave_v2.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="semantics changed"):
        subject.validate_accept_receipt(*context, wrong_path)


@pytest.mark.parametrize("ambiguous", [True, False])
def test_timeout_or_412_reconciles_only_exact_existing_payload(
    monkeypatch: pytest.MonkeyPatch,
    context: tuple[dict, dict, dict],
    ambiguous: bool,
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    fake = FakeGcs()
    path, payload = _acceptance_payload(context)
    if ambiguous:
        fake.raise_after_create = True
    else:
        fake.put(path, payload)
    receipt = _store(context, mode="accept", fake=fake).accept_and_readback(
        path=path, data=payload
    )
    assert receipt["provider_create_confirmed"] is False
    assert receipt["transport_ambiguity_reconciled"] is True
    assert receipt["http_post_count"] == 1

    mismatch = FakeGcs()
    mismatch.put(path, b'{"different":true}\n')
    with pytest.raises(RuntimeError, match="exact payload was not observed") as exc:
        _store(context, mode="accept", fake=mismatch).accept_and_readback(
            path=path, data=payload
        )
    message = str(exc.value)
    assert TOKEN_A not in message
    assert "provider-response-secret" not in message
    assert payload.decode("ascii") not in message


def test_unexpected_create_metadata_fails_closed_then_exact_412_rerun_recovers(
    monkeypatch: pytest.MonkeyPatch, context: tuple[dict, dict, dict]
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    fake = FakeGcs()
    fake.return_extra_create_metadata = True
    path, payload = _acceptance_payload(context)
    store = _store(context, mode="accept", fake=fake)

    with pytest.raises(RuntimeError, match="metadata fields changed"):
        store.create_only(path=path, data=payload)

    fake.return_extra_create_metadata = False
    recovered = store.create_only(path=path, data=payload)
    generation, raw = fake.objects[path]
    assert recovered == {
        "path": path,
        "generation": generation,
        "bytes": len(payload),
        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
    }
    assert raw == payload


def test_failed_post_body_token_and_transport_exception_are_redacted(
    monkeypatch: pytest.MonkeyPatch, context: tuple[dict, dict, dict]
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    fake = FakeGcs()
    fake.fail_post_status = 500
    path, payload = _acceptance_payload(context)
    with pytest.raises(RuntimeError, match="failed with status 500") as exc:
        _store(context, mode="accept", fake=fake).accept_and_readback(
            path=path, data=payload
        )
    rendered = str(exc.value)
    assert TOKEN_A not in rendered
    assert "provider-response-secret" not in rendered
    assert payload.decode("ascii") not in rendered


def test_cli_requires_explicit_accept_flag_and_has_no_token_argument(
    tmp_path: Path,
    context: tuple[dict, dict, dict],
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan, ledger, resume = context
    paths = []
    for name, value in (("plan", plan), ("ledger", ledger), ("resume", resume)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    object_path, payload = _acceptance_payload(context)
    payload_path = tmp_path / "ACCEPTED.json"
    payload_path.write_bytes(payload)
    argv = [
        "accept",
        "--wave-plan",
        str(paths[0]),
        "--attempt-ledger",
        str(paths[1]),
        "--resume-plan",
        str(paths[2]),
        "--object-path",
        object_path,
        "--payload",
        str(payload_path),
    ]
    with pytest.raises(PermissionError, match="explicit"):
        subject.main(argv)
    with pytest.raises(SystemExit):
        subject.main(["accept", "--help"])
    help_text = capsys.readouterr().out
    assert "--authorize-accept-create" in help_text
    assert "--token" not in help_text
