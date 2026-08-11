from __future__ import annotations

import importlib.util
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts"
    / "run_hu_m31_t3_step6d_rearm2_diagnostic_step11_one_vm_v1.py"
)
FIX3_ROOT = (
    ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step11_v12_local_preflight_fix3"
)


def _runner():
    spec = importlib.util.spec_from_file_location(
        "step11_fix3_runner_under_test",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fix3_runner_is_bound_to_frozen_local_identity() -> None:
    runner = _runner()
    receipt = json.loads(
        (FIX3_ROOT / "LOCAL_V12_READY.json").read_text(encoding="utf-8")
    )
    assert runner.EXPECTED_OUTER_IDENTITY == receipt[
        "outer_package_identity_sha256"
    ]
    assert runner.EXPECTED_DIRECT_IDENTITY == receipt[
        "direct_stage_identity_sha256"
    ]
    assert runner.OUTPUT_ROOT.name == "step11_one_vm_v12_fix3_actual"
    assert not hasattr(runner, "EXISTING_PACKAGE_OUTPUT_ROOT")
    assert not hasattr(runner, "EXISTING_PACKAGE_RECEIPT_PATH")


def test_fix3_runner_condition_files_match_gate_contract() -> None:
    runner = _runner()
    contract = json.loads(
        (FIX3_ROOT / "transport_contract.json").read_text(encoding="utf-8")
    )
    expiry = int(
        datetime.strptime(
            runner.AUTHORIZATION_EXPIRY,
            "%Y-%m-%dT%H:%M:%SZ",
        )
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    plan = runner.iam_gate.build_step11_gate_plan(
        contract,
        issued_at_unix_seconds=expiry - 7_000,
        expires_at_unix_seconds=expiry,
        nat_router_resource=runner.NAT_ROUTER_RESOURCE,
    )
    runner._validate_fix3_condition_files(
        contract=contract,
        gate_plan=plan,
    )


def test_fix3_runner_gates_before_fresh_upload_and_revokes_before_insert() -> None:
    runner = _runner()
    source = inspect.getsource(runner.main)
    gate = source.index("iam_gate.build_step11_gate_plan")
    authorization = source.index("_install_step11_authorization")
    upload = source.index("controller.provision_package")
    freshness = source.index("fix3 package prefix was not exactly fresh")
    revoke = source.index("_revoke_controller_package_create")
    preflight = source.index("run_actual_read_only_preflight")
    insert = source.index("cloud.execute_step11_attempt0")
    assert (
        gate
        < authorization
        < upload
        < freshness
        < revoke
        < preflight
        < insert
    )


def test_fix3_runner_authorization_is_exactly_eleven_bindings() -> None:
    runner = _runner()
    contract = json.loads(
        (FIX3_ROOT / "transport_contract.json").read_text(encoding="utf-8")
    )
    expiry = int(
        datetime.strptime(
            runner.AUTHORIZATION_EXPIRY,
            "%Y-%m-%dT%H:%M:%SZ",
        )
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    plan = runner.iam_gate.build_step11_gate_plan(
        contract,
        issued_at_unix_seconds=expiry - 7_000,
        expires_at_unix_seconds=expiry,
        nat_router_resource=runner.NAT_ROUTER_RESOURCE,
    )
    specs = runner._step11_binding_specs(plan)
    assert len(specs) == 11
    assert len({row["purpose"] for row in specs}) == 11
    counts = {}
    for row in specs:
        counts[row["target"].value] = counts.get(row["target"].value, 0) + 1
        assert row["condition"]["title"]
        assert row["condition"]["expression"]
    assert counts == {
        "project": 5,
        "bucket": 4,
        "worker_service_account": 1,
        "controller_service_account": 1,
    }


def test_fix3_live_iam_paths_do_not_call_gcloud_iam() -> None:
    runner = _runner()
    for function in (
        runner._revoke_controller_package_create,
        runner._post_claim_revoke,
        runner._collect_live_iam_evidence,
        runner._final_cleanup,
    ):
        source = inspect.getsource(function)
        assert "_run_gcloud" not in source
        assert "_gcloud_json" not in source


def test_fix3_active_user_compute_client_is_exact_name_only() -> None:
    runner = _runner()
    base = (
        "https://compute.googleapis.com/compute/v1/projects/"
        f"{runner.transport.PROJECT}/zones/{runner.transport.ZONE}"
    )
    exact = f"{base}/instances/{runner.EXPECTED_INSTANCE_NAME}"
    assert runner.ActiveUserExactCloudClient._allowed("GET", exact)
    assert not runner.ActiveUserExactCloudClient._allowed(
        "GET",
        f"{base}/instances/other",
    )
    assert not runner.ActiveUserExactCloudClient._allowed(
        "DELETE",
        exact,
    )
    assert not runner.ActiveUserExactCloudClient._allowed(
        "GET",
        exact + "?alt=json",
    )


def test_fix3_preflight_does_not_export_user_token() -> None:
    runner = _runner()
    source = inspect.getsource(runner.main)
    assert "GOOGLE_OAUTH_ACCESS_TOKEN" not in source
    assert "os.environ" not in source
    assert "token_source=user_token" in source


def test_fix3_cleanup_revokes_iam_even_when_compute_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    events: list[str] = []

    class FakeAdmin:
        def get_policy(self, target):
            events.append(f"read:{target.value}")
            return {"bindings": []}

    class FailingCompute:
        def request(self, **kwargs):
            events.append("compute")
            raise RuntimeError("injected compute failure")

    def fake_binding_cleanup(admin, gate_plan):
        events.append("iam_cleanup")
        return {"receipt_sha256": "0" * 64}

    (tmp_path / "iam_gate_plan.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_REST_ADMIN", FakeAdmin())
    monkeypatch.setattr(runner, "_ACTIVE_USER_CLOUD_CLIENT", FailingCompute())
    monkeypatch.setattr(
        runner,
        "_clear_stale_step11_bindings",
        fake_binding_cleanup,
    )
    monkeypatch.setattr(
        runner,
        "_assert_policy_registry_unchanged",
        lambda: None,
    )

    with pytest.raises(BaseExceptionGroup):
        runner._final_cleanup()
    assert events[0] == "iam_cleanup"
    assert "compute" in events


def test_fix3_cleanup_attempts_compute_when_iam_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    events: list[str] = []

    class FakeAdmin:
        def get_policy(self, target):
            return {"bindings": []}

    class AbsentCompute:
        def request(self, **kwargs):
            events.append("compute")
            return runner.controller.HttpResponse(
                status=404,
                headers={},
                body=b"",
            )

    def failing_binding_cleanup(admin, gate_plan):
        events.append("iam_cleanup")
        raise RuntimeError("injected IAM failure")

    (tmp_path / "iam_gate_plan.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_REST_ADMIN", FakeAdmin())
    monkeypatch.setattr(runner, "_ACTIVE_USER_CLOUD_CLIENT", AbsentCompute())
    monkeypatch.setattr(
        runner,
        "_clear_stale_step11_bindings",
        failing_binding_cleanup,
    )
    monkeypatch.setattr(
        runner,
        "_assert_policy_registry_unchanged",
        lambda: None,
    )

    with pytest.raises(BaseExceptionGroup):
        runner._final_cleanup()
    assert events[0] == "iam_cleanup"
    assert "compute" in events


def test_fix3_cleanup_continues_after_initial_profile_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    events: list[str] = []

    class FakeAdmin:
        def get_policy(self, target):
            return {"bindings": []}

    class AbsentCompute:
        def request(self, **kwargs):
            events.append("compute")
            return runner.controller.HttpResponse(
                status=404,
                headers={},
                body=b"",
            )

    def fake_binding_cleanup(admin, gate_plan):
        events.append("iam_cleanup")
        return {"receipt_sha256": "0" * 64}

    def failing_profile_check():
        events.append("profile_check")
        raise RuntimeError("injected profile failure")

    (tmp_path / "iam_gate_plan.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_REST_ADMIN", FakeAdmin())
    monkeypatch.setattr(runner, "_ACTIVE_USER_CLOUD_CLIENT", AbsentCompute())
    monkeypatch.setattr(
        runner,
        "_clear_stale_step11_bindings",
        fake_binding_cleanup,
    )
    monkeypatch.setattr(
        runner,
        "_assert_policy_registry_unchanged",
        failing_profile_check,
    )

    with pytest.raises(BaseExceptionGroup):
        runner._final_cleanup()
    assert events[0] == "profile_check"
    assert "iam_cleanup" in events
    assert "compute" in events


def test_fix3_controller_token_retries_only_propagation_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    calls = 0

    class TokenSource:
        def access_token(self):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise runner.rest_iam.RestIamAdminError(
                    "injected",
                    operation="generate",
                    status_code=403,
                )
            return "token"

    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    runner._wait_for_controller_token(TokenSource())
    assert calls == 3


def test_fix3_package_client_retries_identical_gcs_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    calls = []

    class Inner:
        def request(self, **kwargs):
            calls.append(kwargs)
            return runner.controller.HttpResponse(
                status=403 if len(calls) < 3 else 200,
                headers={},
                body=b"{}",
            )

    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    client = runner.PackagePropagationRetryClient(Inner())
    response = client.request(
        method="GET",
        url="https://storage.googleapis.com/storage/v1/b/example/o/object",
    )
    assert response.status == 200
    assert len(calls) == 3
    assert calls[0] == calls[1] == calls[2]
