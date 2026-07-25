from __future__ import annotations

import hashlib
import json
import urllib.parse
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_production_receiver_v2 as subject
from ofc_regular.hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    HttpResponse,
)


RUN = "regular-hu-m31-c02-f100wv2-20260723-production-test"
PROJECT = "ofc-solver-485418"
ZONE = "asia-northeast1-b"


def _initial() -> tuple[dict, dict, dict]:
    plan = wave_v2.build_wave_plan(
        run_name=RUN,
        identity_salt="0123456789abcdef0123456789abcdef",
        package_sha256="2" * 64,
        image_digest="sha256:" + "3" * 64,
    )
    transition = wave_v2.build_observed_transition(
        plan,
        project_id=PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-23T00:00:00Z",
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(plan),
    )
    ledger = wave_v2.build_attempt_ledger(plan, transitions=[transition])
    resume = wave_v2.build_resume_plan(plan, attempt_ledger=ledger)
    return plan, ledger, resume


class EmptyGcs:
    def __init__(self, token: str) -> None:
        self.token = token
        self.calls: list[tuple[str, str, dict[str, str], bytes | None]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: int,
    ) -> HttpResponse:
        del timeout
        self.calls.append((method, url, dict(headers), body))
        assert headers["Authorization"] == f"Bearer {self.token}"
        assert method == "GET"
        parsed = urllib.parse.urlparse(url)
        assert parsed.path.endswith("/o")
        assert "prefix" in urllib.parse.parse_qs(parsed.query)
        return HttpResponse(status=200, headers={}, body=b"{}")


def test_poll_is_generation_pinned_read_only_and_redacts_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, ledger, resume = _initial()
    token = "test-only-token-abcdefghijklmnopqrstuvwxyz"
    monkeypatch.setenv("GOOGLE_OAUTH_ACCESS_TOKEN", token)
    remote = EmptyGcs(token)
    receipt = subject.production_poll(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        requester=remote,
    )
    encoded = json.dumps(receipt, sort_keys=True)
    assert receipt["status"] == "selected_wave_generation_pinned_read_only"
    assert receipt["attempt_ledger_sha256"] == ledger["ledger_sha256"]
    assert receipt["resume_plan_sha256"] == resume["resume_sha256"]
    assert receipt["execution_namespace"] == subject.execution_namespace(
        ledger, resume
    )
    assert receipt["acceptance_create_authorized"] is False
    assert receipt["vm_lifecycle_mutation_performed"] is False
    assert receipt["poll_receipt"]["http_post_count"] == 0
    assert receipt["poll_receipt"]["read_only"] is True
    assert len(remote.calls) == len(resume["selected_attempts"])
    assert token not in encoded
    assert "Authorization" not in encoded


def test_poll_progress_writes_immutable_digest_checkpoints_without_latest(
    tmp_path: Path,
) -> None:
    _, ledger, resume = _initial()
    namespace = subject.execution_namespace(ledger, resume)
    paths = []
    for state in ("empty", "partial", "done"):
        body = {
            "schema": subject.PRODUCTION_POLL_SCHEMA,
            "status": "selected_wave_generation_pinned_read_only",
            "run_name": RUN,
            "wave_index": resume["resume_wave_index"],
            "execution_namespace": namespace,
            "attempt_ledger_sha256": ledger["ledger_sha256"],
            "resume_plan_sha256": resume["resume_sha256"],
            "poll_receipt": {"fake_checkpoint_state": state},
            "acceptance_create_authorized": False,
            "vm_lifecycle_mutation_performed": False,
            "current_profile_changed": False,
        }
        receipt = {**body, "receipt_sha256": wave_v2.canonical_sha256(body)}
        paths.append(
            subject._write_poll_checkpoint(
                tmp_path, namespace=namespace, receipt=receipt
            )
        )
    assert len(set(paths)) == 3
    assert all(path.is_file() for path in paths)
    assert not (tmp_path / namespace / "production_poll_receipt.json").exists()
    assert not (tmp_path / namespace / "latest.json").exists()
    assert all(
        json.loads(path.read_text(encoding="utf-8"))[
            "acceptance_create_authorized"
        ]
        is False
        for path in paths
    )
    before = paths[0].read_bytes()
    first = json.loads(before)
    assert subject._write_poll_checkpoint(
        tmp_path, namespace=namespace, receipt=first
    ) == paths[0]
    assert paths[0].read_bytes() == before


def test_terminals_are_derived_after_proof_and_pair_partial_is_not_claimed_done() -> None:
    _, _, resume = _initial()
    selected = resume["selected_attempts"]
    proof_box: dict[str, dict] = {}
    pinned = frozenset(
        f"{row['artifact_prefix']}/DONE.json" for row in selected
    )
    terminals = subject._LifecycleDerivedTerminals(
        resume_plan=resume, proof_box=proof_box, pinned_paths=pinned
    )
    with pytest.raises(RuntimeError, match="preceded lifecycle proof"):
        list(terminals)

    mappings = []
    for index, row in enumerate(selected):
        mappings.append(
            {
                "job_id": row["job_id"],
                "exact_instance_created": index != 1,
            }
        )
    proof_box["proof"] = {"selected_instance_mapping": mappings}
    rows = list(terminals)
    assert rows[0]["terminal_reason"] == "done_observed"
    assert rows[1]["terminal_reason"] == "create_missing_before_done"
    assert all(
        set(row)
        == {"job_id", "source_role", "attempt_id", "instance_id", "terminal_reason"}
        for row in rows
    )


def test_missing_done_after_exact_closeout_is_generic_worker_failure() -> None:
    _, _, resume = _initial()
    selected = resume["selected_attempts"]
    proof = {
        "proof": {
            "selected_instance_mapping": [
                {"job_id": row["job_id"], "exact_instance_created": True}
                for row in selected
            ]
        }
    }
    rows = list(
        subject._LifecycleDerivedTerminals(
            resume_plan=resume,
            proof_box=proof,
            pinned_paths=frozenset(),
        )
    )
    assert {row["terminal_reason"] for row in rows} == {
        "worker_failed_before_done"
    }


def test_raw_lifecycle_proof_bypass_is_not_a_request_field() -> None:
    plan, ledger, resume = _initial()
    request = {key: {} for key in subject._REQUEST_KEYS}
    request["lifecycle_proof"] = {"unsafe": True}
    with pytest.raises(ValueError, match="request fields changed"):
        subject._validated_material(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            request=request,
        )


def test_write_once_outputs_are_idempotent_and_conflicts_fail(tmp_path: Path) -> None:
    path = tmp_path / "execution-001-deadbeefcafe" / "resume_plan.json"
    first = {"resume_sha256": hashlib.sha256(b"first").hexdigest()}
    subject._write_once_json(path, first)
    original = path.read_bytes()
    subject._write_once_json(path, first)
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError, match="different bytes"):
        subject._write_once_json(
            path, {"resume_sha256": hashlib.sha256(b"second").hexdigest()}
        )


def test_execution_namespace_disambiguates_same_base_wave_retry() -> None:
    _, ledger, resume = _initial()
    first = subject.execution_namespace(ledger, resume)
    retry_ledger = {**ledger, "transitions": [*ledger["transitions"], {}]}
    retry_resume = {
        **resume,
        "resume_sha256": hashlib.sha256(b"retry-resume").hexdigest(),
    }
    second = subject.execution_namespace(retry_ledger, retry_resume)
    assert first == f"execution-000-{resume['resume_sha256'][:12]}"
    assert second == f"execution-001-{retry_resume['resume_sha256'][:12]}"
    assert first != second


def test_receiver_idempotency_view_allows_only_create_performed_flip() -> None:
    first = {
        "attempt_results": [
            {
                "job_id": "candidate-00",
                "acceptance_create_performed": True,
                "acceptance_generation": 123,
                "acceptance_sha256": "a" * 64,
            }
        ],
        "receipt_sha256": "b" * 64,
    }
    rerun = json.loads(json.dumps(first))
    rerun["attempt_results"][0]["acceptance_create_performed"] = False
    rerun["receipt_sha256"] = "c" * 64
    assert subject._receiver_idempotency_view(first) == subject._receiver_idempotency_view(
        rerun
    )
    rerun["attempt_results"][0]["acceptance_generation"] = 124
    assert subject._receiver_idempotency_view(first) != subject._receiver_idempotency_view(
        rerun
    )


def test_receive_cli_requires_explicit_accept_and_exact_run_name(
    tmp_path: Path,
) -> None:
    plan, ledger, resume = _initial()
    paths = {}
    for name, value in (("plan", plan), ("ledger", ledger), ("resume", resume)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    common = [
        "--mode",
        "receive",
        "--wave-plan",
        str(paths["plan"]),
        "--attempt-ledger",
        str(paths["ledger"]),
        "--resume-plan",
        str(paths["resume"]),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    with pytest.raises(PermissionError, match="explicit"):
        subject.main(common)
    with pytest.raises(PermissionError, match="exactly match"):
        subject.main(
            [*common, "--allow-accept-create", "--confirm-run-name", "wrong"]
        )


def test_receive_api_cannot_bypass_explicit_accept_confirmation(tmp_path: Path) -> None:
    plan, ledger, resume = _initial()
    common = {
        "journal_dir": tmp_path / "journal",
        "wave_plan": plan,
        "attempt_ledger": ledger,
        "resume_plan": resume,
        "request": {},
        "destination": tmp_path / "data",
        "output_dir": tmp_path / "output",
        "project_id": PROJECT,
        "zone": ZONE,
    }
    with pytest.raises(PermissionError, match="explicit ACCEPTED"):
        subject.production_receive(**common)
    with pytest.raises(PermissionError, match="does not match"):
        subject.production_receive(
            **common, allow_accept_create=True, confirm_run_name="wrong"
        )


def test_sealed_request_binds_exact_nested_controller_journal(
    tmp_path: Path,
) -> None:
    _, ledger, resume = _initial()
    namespace = subject.execution_namespace(ledger, resume)
    execution_root = tmp_path / namespace
    journal = execution_root / "controller-journal"
    journal.mkdir(parents=True)
    payload = {
        "closeout_event_sha256": "a" * 64,
        "launch_bundle": {},
        "launch_validation": {},
        "startup_script_path": str(tmp_path / "startup.sh"),
        "gce_create_receipt": {},
        "gce_delete_receipt": {},
        "worker_iam_cleanup_receipt": {},
        "content_binding": {},
        "observed_at_utc": "2026-07-23T00:00:00Z",
    }
    request = subject.build_production_receive_request(
        execution_namespace=namespace,
        controller_journal_dir=journal,
        payload=payload,
    )
    assert subject.validate_production_receive_request(
        request,
        expected_execution_namespace=namespace,
        expected_controller_journal_dir=journal,
    ) == request
    assert subject.resolve_production_receive_journal_dir(
        request=request,
        journal_dir=journal,
        journal_dir_mode=subject.JOURNAL_DIR_MODE_EXACT,
        expected_execution_namespace=namespace,
    ) == journal.resolve()
    with pytest.raises(ValueError, match="differs from sealed"):
        subject.resolve_production_receive_journal_dir(
            request=request,
            journal_dir=execution_root,
            journal_dir_mode=subject.JOURNAL_DIR_MODE_EXACT,
            expected_execution_namespace=namespace,
        )
    with pytest.raises(PermissionError, match="exact journal-dir mode"):
        subject.resolve_production_receive_journal_dir(
            request=request,
            journal_dir=journal,
            journal_dir_mode="namespaced-root",
            expected_execution_namespace=namespace,
        )


def test_sealed_request_tamper_is_rejected(tmp_path: Path) -> None:
    _, ledger, resume = _initial()
    namespace = subject.execution_namespace(ledger, resume)
    journal = tmp_path / "controller-journal"
    journal.mkdir()
    request = subject.build_production_receive_request(
        execution_namespace=namespace,
        controller_journal_dir=journal,
        payload={key: {} for key in subject._REQUEST_PAYLOAD_KEYS},
    )
    request["observed_at_utc"] = "2026-07-23T00:00:01Z"
    with pytest.raises(ValueError, match="digest changed"):
        subject.validate_production_receive_request(request)
