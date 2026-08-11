from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts"
    / "run_hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_v2.py"
)


@pytest.fixture(scope="module")
def runner():
    name = "step12b_pair_v2_runner_under_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _sealed(runner: Any, **body: Any) -> dict[str, Any]:
    return {
        **body,
        "receipt_sha256": runner._canonical_sha256(body),
    }


def _prepared(runner: Any) -> Any:
    profile_sha = runner._policy_registry_sha256()
    deployment = {
        "deployment_contract_sha256": "11" * 32,
        "run_identity_sha256": "12" * 32,
        "direct_stage_identity_sha256": "13" * 32,
        "selected_job_ids": ["candidate-external", "reference-external"],
        "source_roles": ["candidate", "reference"],
        "instances": [
            {
                "instance_name": "r2d-s2b-c-a0-fixture",
                "source_role": "candidate",
            },
            {
                "instance_name": "r2d-s2b-r-a0-fixture",
                "source_role": "reference",
            },
        ],
    }
    return runner.PreparedRun(
        signer=object(),
        controller_public_key_record={
            "schema": "public",
            "key_id": "fixture",
        },
        run_nonce="ab" * 32,
        candidate_payload_contract={"job": "candidate"},
        reference_payload_contract={"job": "reference"},
        deployment_contract=deployment,
        runtime_source_files={
            "ofc_regular/a.py": "A = 1\n",
            "ofc_regular/b.py": "B = 2\n",
        },
        source_plan={"source_plan_sha256": "14" * 32},
        phase2_iam_plan={"plan_sha256": "15" * 32},
        package_provision_receipt={"receipt_sha256": "16" * 32},
        issued_at_unix_seconds=1_900_000_000,
        expires_at_unix_seconds=1_900_007_200,
        policy_registry_sha256=profile_sha,
    )


class _Provision:
    def __init__(self, receipt: Mapping[str, Any]) -> None:
        self._receipt = dict(receipt)

    def receipt(self) -> dict[str, Any]:
        return dict(self._receipt)


class _FakeBackend:
    def __init__(
        self,
        runner: Any,
        prepared: Any,
        *,
        fail_stage: str | None = None,
        pair_owned_failure: bool = False,
        outer_cleanup_complete: bool = True,
        token_barrier_failure: bool = False,
    ) -> None:
        self.runner = runner
        self.prepared = prepared
        self.fail_stage = fail_stage
        self.pair_owned_failure = pair_owned_failure
        self.outer_cleanup_complete = outer_cleanup_complete
        self.token_barrier_failure = token_barrier_failure
        self.calls: list[str] = []
        self.raw_secret = "must-never-appear-in-artifacts"

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_stage == name:
            raise RuntimeError(self.raw_secret)

    def provision_sources(self):
        self._call("source_provision")
        receipt = _sealed(
            self.runner,
            status="sources",
            access_token_stored=False,
        )
        manifests = {
            job_id: {
                "external_job_id": job_id,
                "source_role": role,
                "role_manifest_sha256": f"{position + 20:02x}" * 32,
            }
            for position, (job_id, role) in enumerate(
                zip(
                    self.prepared.deployment_contract[
                        "selected_job_ids"
                    ],
                    self.prepared.deployment_contract["source_roles"],
                    strict=True,
                )
            )
        }
        return _Provision(receipt), manifests

    def create_controller_service_account(self):
        self._call("controller_service_account_create")
        return SimpleNamespace(
            receipt=_sealed(
                self.runner,
                status="controller-created",
                private_key_stored=False,
            )
        )

    def run_token_barrier(self):
        self.calls.append("token_barrier")
        if self.fail_stage == "token_barrier":
            if self.token_barrier_failure:
                raise self.runner.token_barrier.TokenBarrierFailure(
                    _sealed(
                        self.runner,
                        status=(
                            self.runner.token_barrier.FAILURE_STATUS
                        ),
                        failure_reason=(
                            "token_creator_revoke_zero_readback_timeout"
                        ),
                        token_creator_revoke_zero_readback={
                            "poll_count": 4,
                            "stale_present_count": 4,
                            "zero_observed": False,
                            "second_add_performed": False,
                            "controller_token_reminted": False,
                        },
                        phase2_started=False,
                        vm_insert_attempt_count=0,
                        access_token_stored=False,
                    )
                )
            raise RuntimeError(self.raw_secret)
        return SimpleNamespace(
            receipt=_sealed(
                self.runner,
                status="token-barrier",
                access_token_stored=False,
                authorization_header_stored=False,
            )
        )

    def install_phase2(self):
        self._call("phase2_install")
        return SimpleNamespace(
            install_receipt=_sealed(
                self.runner,
                status="eight-bindings-installed",
                binding_count=8,
            )
        )

    def collect_and_mint_external_preflight(self):
        self._call("postmutation_external_preflight")
        gate = _sealed(
            self.runner,
            status="external-preflight",
            launch_authorized=False,
        )
        auth = _sealed(
            self.runner,
            status="auth-preflight",
            access_token_stored=False,
        )
        return gate, object(), auth

    def prepare_role_launches(
        self,
        *,
        validated_external_preflight: Any,
        external_preflight_receipt: Mapping[str, Any],
    ):
        del validated_external_preflight, external_preflight_receipt
        self._call("role_launch_prepare")
        return [
            SimpleNamespace(
                external_job_id=job_id,
                authorization={
                    "job_id": job_id,
                    "signature": f"sig-{position}",
                },
                initial_metadata_values={
                    "job_id": job_id,
                    "source_role": role,
                },
            )
            for position, (job_id, role) in enumerate(
                zip(
                    self.prepared.deployment_contract[
                        "selected_job_ids"
                    ],
                    self.prepared.deployment_contract["source_roles"],
                    strict=True,
                )
            )
        ]

    def run_pair_controller(self, *, role_launches: Any):
        assert len(role_launches) == 2
        self.calls.append("exact_pair_attempt0")
        if self.fail_stage == "exact_pair_attempt0":
            if self.pair_owned_failure:
                raise self.runner.pair_controller.ExactPairCloudControllerError(
                    _sealed(
                        self.runner,
                        status="pair-controller-failure",
                        cleanup_records=[
                            {
                                "operation": "phase2_iam_revoke",
                                "completed": False,
                                "sanitized_failure": (
                                    "cleanup_callback_failed"
                                ),
                            }
                        ],
                        automatic_retry_performed=False,
                    )
                )
            raise RuntimeError(self.raw_secret)
        return _sealed(
            self.runner,
            status="pair-released",
            insert_request_count=2,
            attempt_index=0,
            automatic_retry_performed=False,
        )

    def receive_pair(self, *, destination_root: Path):
        self._call("pair_result_receive")
        destination_root.mkdir()
        (destination_root / "candidate").mkdir()
        (destination_root / "reference").mkdir()
        return _sealed(
            self.runner,
            status="received-44x2",
            pair_result_object_count=88,
        )

    def remove_worker_bindings_after_done(
        self, *, pair_receive_receipt: Mapping[str, Any]
    ):
        assert pair_receive_receipt["pair_result_object_count"] == 88
        self._call("worker_iam_cleanup")
        return _sealed(
            self.runner,
            status="all-worker-bindings-zero",
            final_targeted_binding_count=0,
        )

    def ensure_compute_absent(self):
        self._call("compute_absence")
        return _sealed(
            self.runner,
            status="exact-pair-absent",
            instance_final_statuses=[404, 404],
            disk_final_statuses=[404, 404],
        )

    def collect_final_zero_readback(self):
        self._call("final_independent_zero_readback")
        return _sealed(
            self.runner,
            status="all-live-surfaces-independently-zero",
            phase2_controller_and_worker_principals_zero=True,
            token_creator_zero=True,
            controller_service_account_get404=True,
            instance_final_statuses=[404, 404],
            disk_final_statuses=[404, 404],
            cloud_mutation_performed=False,
        )

    def cleanup_failure(self, *, failure_evidence_sha256: str):
        assert len(failure_evidence_sha256) == 64
        self.calls.append("failure_cleanup")
        return _sealed(
            self.runner,
            status="failure-cleaned",
            cleanup_order=["phase2", "service-account", "compute"],
            records=[
                {
                    "operation": "phase2",
                    "completed": self.outer_cleanup_complete,
                },
                {
                    "operation": "service-account",
                    "completed": True,
                },
                {"operation": "compute", "completed": True},
            ],
            mandatory_verification_complete=(
                self.outer_cleanup_complete
            ),
        )


def test_default_cli_is_dry_run_and_never_constructs_cloud_backend(
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _prepared(runner)
    monkeypatch.setattr(runner, "prepare_run", lambda: prepared)

    class _Forbidden:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise AssertionError("dry-run constructed cloud adapters")

    monkeypatch.setattr(runner, "LiveExecutionBackend", _Forbidden)
    assert runner.main([]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == (
        "offline_dry_run_only_no_cloud_adapter_constructed"
    )
    assert receipt["cloud_adapter_constructed"] is False
    assert receipt["cloud_mutation_performed"] is False
    assert receipt["output_root_created"] is False


def test_execute_requires_exact_confirmation_before_prepare_or_output(
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def _prepare():
        nonlocal called
        called = True
        raise AssertionError("prepare must not run")

    monkeypatch.setattr(runner, "prepare_run", _prepare)
    with pytest.raises(PermissionError, match="confirmation"):
        runner.main(
            [
                "--execute",
                "--confirm",
                "almost-but-not-exact",
                "--output-root",
                str(tmp_path / "run"),
            ]
        )
    assert called is False
    assert not (tmp_path / "run").exists()


def test_fake_integration_runs_exact_order_and_writes_no_secrets(
    runner: Any,
    tmp_path: Path,
) -> None:
    prepared = _prepared(runner)
    backend = _FakeBackend(runner, prepared)
    root = tmp_path / "success"
    receipt = runner.execute_prepared(
        prepared=prepared,
        backend=backend,
        output_root=root,
    )
    assert backend.calls == [
        "source_provision",
        "controller_service_account_create",
        "token_barrier",
        "phase2_install",
        "postmutation_external_preflight",
        "role_launch_prepare",
        "exact_pair_attempt0",
        "pair_result_receive",
        "worker_iam_cleanup",
        "compute_absence",
        "final_independent_zero_readback",
    ]
    assert receipt["status"] == (
        "candidate_and_reference_attempt0_received_and_cleaned"
    )
    assert receipt["vm_count"] == 2
    assert receipt["attempt_index"] == 0
    assert receipt["attempt1_authorized"] is False
    assert receipt["third_vm_authorized"] is False
    assert receipt["automatic_retry_performed"] is False
    assert receipt["result_object_count"] == 88
    assert receipt["final_independent_zero_readback_verified"] is True
    assert (
        receipt["final_independent_zero_readback_receipt_sha256"]
        == json.loads(
            (
                root
                / "final_independent_zero_readback_receipt.json"
            ).read_text()
        )["receipt_sha256"]
    )
    assert receipt["policy_registry_sha256_before"] == (
        receipt["policy_registry_sha256_after"]
    )
    assert (root / "FINAL.json").is_file()
    all_artifacts = b"".join(
        path.read_bytes() for path in sorted(root.rglob("*.json"))
    )
    assert backend.raw_secret.encode() not in all_artifacts
    assert b"BEGIN PRIVATE KEY" not in all_artifacts
    assert b"Bearer " not in all_artifacts


def test_failure_after_pair_release_cleans_once_and_never_retries(
    runner: Any,
    tmp_path: Path,
) -> None:
    prepared = _prepared(runner)
    backend = _FakeBackend(
        runner, prepared, fail_stage="pair_result_receive"
    )
    root = tmp_path / "receive-failure"
    with pytest.raises(RuntimeError, match=backend.raw_secret):
        runner.execute_prepared(
            prepared=prepared,
            backend=backend,
            output_root=root,
        )
    assert backend.calls[-2:] == [
        "pair_result_receive",
        "failure_cleanup",
    ]
    assert backend.calls.count("exact_pair_attempt0") == 1
    failure = json.loads((root / "FAILURE.json").read_text())
    assert failure["failure_stage"] == "pair_result_receive"
    assert failure["automatic_retry_performed"] is False
    assert failure["attempt1_authorized"] is False
    assert failure["third_vm_authorized"] is False
    assert failure["exception_message_stored"] is False
    assert backend.raw_secret not in (root / "FAILURE.json").read_text()


def test_token_barrier_failure_receipt_is_validated_saved_and_stops_launch(
    runner: Any,
    tmp_path: Path,
) -> None:
    prepared = _prepared(runner)
    backend = _FakeBackend(
        runner,
        prepared,
        fail_stage="token_barrier",
        token_barrier_failure=True,
    )
    root = tmp_path / "token-barrier-failure"
    with pytest.raises(runner.token_barrier.TokenBarrierFailure):
        runner.execute_prepared(
            prepared=prepared,
            backend=backend,
            output_root=root,
        )
    assert backend.calls == [
        "source_provision",
        "controller_service_account_create",
        "token_barrier",
        "failure_cleanup",
    ]
    assert "phase2_install" not in backend.calls
    assert "exact_pair_attempt0" not in backend.calls
    token_failure = json.loads(
        (root / "token_barrier_failure_receipt.json").read_text()
    )
    failure = json.loads((root / "FAILURE.json").read_text())
    assert token_failure["phase2_started"] is False
    assert token_failure["vm_insert_attempt_count"] == 0
    assert token_failure[
        "token_creator_revoke_zero_readback"
    ]["controller_token_reminted"] is False
    assert failure["failure_stage"] == "token_barrier"
    assert failure["token_barrier_failure_receipt_path"] == (
        "token_barrier_failure_receipt.json"
    )
    assert failure["token_barrier_failure_receipt_exists"] is True
    assert failure["token_barrier_failure_receipt_sha256"] == (
        token_failure["receipt_sha256"]
    )


def test_worker_diagnostic_receipt_is_persisted_before_failure_closeout(
    runner: Any, tmp_path: Path
) -> None:
    prepared = _prepared(runner)
    backend = _FakeBackend(runner, prepared)
    diagnostic = _sealed(
        runner,
        schema=(
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "worker_failure_receipt_v1"
        ),
        status="sanitized_worker_failure_recovered_before_done",
        failure_stage="host_prerequisite_install",
        exception_type="CalledProcessError",
        serial_contents_stored=False,
        access_token_stored=False,
    )

    def fail_receive(*, destination_root: Path) -> dict[str, Any]:
        del destination_root
        backend.calls.append("pair_result_receive")
        raise runner.result_receiver.WorkerDiagnosticFailure(diagnostic)

    backend.receive_pair = fail_receive
    root = tmp_path / "worker-diagnostic-failure"
    with pytest.raises(runner.result_receiver.WorkerDiagnosticFailure):
        runner.execute_prepared(
            prepared=prepared,
            backend=backend,
            output_root=root,
        )
    saved = json.loads(
        (root / "worker_diagnostic_failure_receipt.json").read_text()
    )
    failure = json.loads((root / "FAILURE.json").read_text())
    assert saved == diagnostic
    assert failure["failure_stage"] == "pair_result_receive"
    assert failure["worker_diagnostic_failure_receipt_exists"] is True
    assert failure["worker_diagnostic_failure_receipt_sha256"] == (
        diagnostic["receipt_sha256"]
    )
    assert backend.calls[-2:] == [
        "pair_result_receive",
        "failure_cleanup",
    ]


def test_pair_controller_cleanup_failure_triggers_outer_verified_recovery(
    runner: Any,
    tmp_path: Path,
) -> None:
    prepared = _prepared(runner)
    backend = _FakeBackend(
        runner,
        prepared,
        fail_stage="exact_pair_attempt0",
        pair_owned_failure=True,
    )
    root = tmp_path / "pair-owned-failure"
    with pytest.raises(
        runner.pair_controller.ExactPairCloudControllerError
    ):
        runner.execute_prepared(
            prepared=prepared,
            backend=backend,
            output_root=root,
        )
    assert backend.calls[-2:] == [
        "exact_pair_attempt0",
        "failure_cleanup",
    ]
    assert backend.calls.count("exact_pair_attempt0") == 1
    assert backend.calls.count("failure_cleanup") == 1
    failure = json.loads((root / "FAILURE.json").read_text())
    assert failure["pair_controller_owned_cleanup"] is True
    assert failure["outer_cleanup_always_attempted"] is True
    assert failure["outer_cleanup_call_returned"] is True
    assert failure["outer_cleanup_verified"] is True
    assert isinstance(failure["outer_cleanup_receipt_sha256"], str)
    assert failure["outer_failure_cleanup_receipt_path"] == (
        "outer_failure_cleanup_receipt.json"
    )
    assert failure["outer_failure_cleanup_receipt_exists"] is True
    assert failure["pair_controller_failure_receipt_path"] == (
        "pair_controller_failure_receipt.json"
    )
    assert failure["pair_controller_failure_receipt_exists"] is True
    outer = json.loads(
        (root / "outer_failure_cleanup_receipt.json").read_text()
    )
    pair = json.loads(
        (root / "pair_controller_failure_receipt.json").read_text()
    )
    assert outer["receipt_sha256"] == (
        failure["outer_cleanup_receipt_sha256"]
    )
    assert pair["receipt_sha256"] == (
        failure["pair_controller_failure_receipt_sha256"]
    )
    assert pair["cleanup_records"][0]["completed"] is False


def test_incomplete_outer_cleanup_payload_is_saved_for_recovery(
    runner: Any,
    tmp_path: Path,
) -> None:
    prepared = _prepared(runner)
    backend = _FakeBackend(
        runner,
        prepared,
        fail_stage="pair_result_receive",
        outer_cleanup_complete=False,
    )
    root = tmp_path / "incomplete-outer-cleanup"
    with pytest.raises(RuntimeError, match=backend.raw_secret):
        runner.execute_prepared(
            prepared=prepared,
            backend=backend,
            output_root=root,
        )
    cleanup = json.loads(
        (root / "outer_failure_cleanup_receipt.json").read_text()
    )
    failure = json.loads((root / "FAILURE.json").read_text())
    assert cleanup["mandatory_verification_complete"] is False
    assert failure["outer_cleanup_verified"] is False
    assert failure["outer_cleanup_unverified_operations"] == ["phase2"]
    assert failure["outer_failure_cleanup_receipt_exists"] is True
    assert failure["outer_cleanup_receipt_sha256"] == (
        cleanup["receipt_sha256"]
    )


def test_failure_receipt_validation_rejects_digest_drift_and_secrets(
    runner: Any,
) -> None:
    valid = _sealed(runner, status="failure", private_key_stored=False)
    assert runner._validated_sealed_receipt(
        valid, label="fixture"
    ) == valid
    drifted = dict(valid)
    drifted["status"] = "tampered"
    with pytest.raises(ValueError, match="digest"):
        runner._validated_sealed_receipt(
            drifted, label="fixture"
        )
    secret = _sealed(runner, status="failure", access_token="secret")
    with pytest.raises(ValueError, match="secret"):
        runner._validated_sealed_receipt(
            secret, label="fixture"
        )


def test_fresh_output_and_immutable_roots_fail_closed(
    runner: Any,
    tmp_path: Path,
) -> None:
    prepared = _prepared(runner)
    existing = tmp_path / "existing"
    existing.mkdir()
    backend = _FakeBackend(runner, prepared)
    with pytest.raises(FileExistsError, match="not fresh"):
        runner.execute_prepared(
            prepared=prepared,
            backend=backend,
            output_root=existing,
        )
    assert backend.calls == []
    with pytest.raises(PermissionError, match="immutable"):
        runner._safe_output_root(runner.STEP11_ROOT)
    with pytest.raises(PermissionError, match="immutable"):
        runner._safe_output_root(runner.PACKAGE_DIR / "new-output")


def test_real_offline_prepare_uses_dynamic_runtime_source_closure(
    runner: Any,
) -> None:
    prepared = runner.prepare_run(
        now_unix_seconds=1_900_000_000,
        run_nonce="cd" * 32,
        signer=runner.step11_controller.generate_ephemeral_controller_key(
            key_size=2_048
        ),
    )
    assert tuple(sorted(prepared.runtime_source_files)) == tuple(
        sorted(runner.bootstrap_source.REQUIRED_RUNTIME_SOURCE_PATHS)
    )
    assert len(prepared.runtime_source_files) >= 12
    dry = runner.dry_run_receipt(prepared)
    assert dry["runtime_source_file_count"] == len(
        runner.bootstrap_source.REQUIRED_RUNTIME_SOURCE_PATHS
    )
    assert dry["vm_count"] == 2
    assert dry["attempt_index"] == 0
    assert dry["automatic_retry_authorized"] is False


class _CleanupIam:
    def __init__(self, runner: Any, log: list[str]) -> None:
        self.runner = runner
        self.log = log
        self.remove_changed = True
        self.raise_unknown_remove = False
        self.controller_target_deleted = False
        self.stale_token_reads = 0
        self.stale_token_forever = False
        self.removed_token_binding: dict[str, Any] | None = None
        self.token_get_count = 0
        self.omit_empty_bindings = False
        self.null_empty_bindings = False

    def remove_binding(
        self,
        target: Any,
        *,
        role: str,
        member: str,
        condition: Mapping[str, Any],
    ) -> Any:
        del target
        self.log.append("token_remove")
        if self.raise_unknown_remove:
            raise OSError("unknown token remove outcome")
        self.removed_token_binding = {
            "role": role,
            "members": [member],
            "condition": dict(condition),
        }
        return SimpleNamespace(changed=self.remove_changed, attempts=1)

    def get_policy(self, target: Any) -> dict[str, Any]:
        self.log.append("token_get")
        if (
            self.controller_target_deleted
            and target
            is self.runner.rest_iam.PolicyTarget.CONTROLLER_SERVICE_ACCOUNT
        ):
            raise self.runner.rest_iam.RestIamAdminError(
                "iam_policy_get_failed",
                operation="get_controller_service_account_policy",
                status_code=404,
            )
        if self.removed_token_binding is not None:
            self.token_get_count += 1
            if (
                self.stale_token_forever
                or self.token_get_count <= self.stale_token_reads
            ):
                return {
                    "version": 3,
                    "etag": "fixture",
                    "bindings": [dict(self.removed_token_binding)],
                }
        if self.omit_empty_bindings:
            return {"version": 3, "etag": "fixture"}
        if self.null_empty_bindings:
            return {"version": 3, "etag": "fixture", "bindings": None}
        return {"version": 3, "etag": "fixture", "bindings": []}


def _cleanup_backend(
    runner: Any,
    prepared: Any,
    *,
    log: list[str],
    phase2_entered: bool,
    token_entered: bool,
) -> Any:
    backend = object.__new__(runner.LiveExecutionBackend)
    backend.prepared = prepared
    backend.iam_admin = _CleanupIam(runner, log)
    backend.phase2_install_entered = phase2_entered
    backend.installed = None
    backend.worker_zero = None
    backend.teardown = SimpleNamespace(
        controller_zero=None,
        failure_cleanup_complete=False,
    )
    backend.token_barrier_entered = token_entered
    backend.token_creator_binding = (
        runner.token_barrier.TokenCreatorBinding(
            target=runner.rest_iam.PolicyTarget.CONTROLLER_SERVICE_ACCOUNT,
            role=runner.token_barrier.TOKEN_CREATOR_ROLE,
            member=runner.token_barrier.DEFAULT_INITIATING_PRINCIPAL,
            condition={
                "title": "fixture",
                "expression": (
                    'request.time < timestamp("2030-01-01T00:00:00Z")'
                ),
            },
            controller_service_account=(
                "ofc-m31-s2b-fixture@"
                "ofc-solver-485418.iam.gserviceaccount.com"
            ),
        )
        if token_entered
        else None
    )

    class _Sa:
        raise_unknown_cleanup = False

        def cleanup_delete_if_present(self) -> dict[str, Any]:
            log.append("sa")
            if self.raise_unknown_cleanup:
                raise OSError("unknown SA delete outcome")
            return _sealed(
                runner,
                status="sa-absent",
                final_get_status=404,
                readback_verified=True,
            )

        def get(self) -> None:
            log.append("sa_get")
            return None

    backend.controller_sa_admin = _Sa()

    def _compute() -> dict[str, Any]:
        log.append("compute")
        return _sealed(
            runner,
            status="compute-absent",
            exact_cleanup_complete=True,
            instance_final_statuses=[404, 404],
            disk_final_statuses=[404, 404],
        )

    backend.ensure_compute_absent = _compute
    clock = SimpleNamespace(elapsed=0.0)

    def _monotonic() -> float:
        return float(clock.elapsed)

    def _sleep(seconds: float) -> None:
        clock.elapsed += seconds

    backend._monotonic = _monotonic
    backend._sleep = _sleep
    backend._test_clock = clock
    return backend


def test_phase2_unknown_install_outcome_recleans_before_token_sa_compute(
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(runner)
    log: list[str] = []
    backend = _cleanup_backend(
        runner,
        prepared,
        log=log,
        phase2_entered=True,
        token_entered=True,
    )

    def _phase2_cleanup(**_: Any) -> dict[str, Any]:
        log.append("phase2_cleanup")
        raise OSError("unknown Phase2 cleanup outcome")

    def _phase2_zero() -> dict[str, Any]:
        log.append("phase2_zero_get")
        return _sealed(runner, status="phase2-zero")

    monkeypatch.setattr(
        runner.phase2_lifecycle,
        "cleanup_step12b_phase2_iam_on_failure",
        _phase2_cleanup,
    )
    backend._verify_phase2_principals_absent = _phase2_zero
    receipt = backend.cleanup_failure(
        failure_evidence_sha256="41" * 32
    )
    assert log == [
        "phase2_cleanup",
        "phase2_zero_get",
        "token_remove",
        "token_get",
        "sa",
        "sa_get",
        "compute",
    ]
    assert [row["operation"] for row in receipt["records"]] == [
        "phase2_controller_then_worker_zero",
        "token_creator_residual_zero",
        "controller_service_account_absent",
        "exact_instances_and_disks_absent",
    ]
    assert all(row["completed"] for row in receipt["records"])


def test_token_barrier_unknown_outcome_removes_and_reads_zero_before_sa(
    runner: Any,
) -> None:
    prepared = _prepared(runner)
    log: list[str] = []
    backend = _cleanup_backend(
        runner,
        prepared,
        log=log,
        phase2_entered=False,
        token_entered=True,
    )
    receipt = backend.cleanup_failure(
        failure_evidence_sha256="42" * 32
    )
    assert log == [
        "token_remove",
        "token_get",
        "sa",
        "sa_get",
        "compute",
    ]
    assert [row["operation"] for row in receipt["records"]] == [
        "phase2_not_installed",
        "token_creator_residual_zero",
        "controller_service_account_absent",
        "exact_instances_and_disks_absent",
    ]
    assert receipt["records"][1]["completed"] is True


def test_unknown_token_remove_transport_still_resolves_with_zero_get(
    runner: Any,
) -> None:
    prepared = _prepared(runner)
    log: list[str] = []
    backend = _cleanup_backend(
        runner,
        prepared,
        log=log,
        phase2_entered=False,
        token_entered=True,
    )
    backend.iam_admin.raise_unknown_remove = True
    receipt = backend._cleanup_token_creator_residual()
    assert log == ["token_remove", "token_get"]
    assert receipt["remove_call_returned"] is False
    assert receipt["remove_changed"] is None
    assert receipt["remove_attempts"] is None
    assert receipt["token_creator_member_count"] == 0
    assert receipt["readback_complete"] is True


def test_outer_token_cleanup_polls_stale_present_then_absent_without_readd(
    runner: Any,
) -> None:
    prepared = _prepared(runner)
    log: list[str] = []
    backend = _cleanup_backend(
        runner,
        prepared,
        log=log,
        phase2_entered=False,
        token_entered=True,
    )
    backend.iam_admin.stale_token_reads = 2
    receipt = backend.cleanup_failure(
        failure_evidence_sha256="44" * 32
    )
    assert log[:4] == [
        "token_remove",
        "token_get",
        "token_get",
        "token_get",
    ]
    assert log.count("token_remove") == 1
    assert "token_add" not in log
    assert "exact_pair_attempt0" not in log
    assert backend._test_clock.elapsed == 3
    assert receipt["token_creator_zero_verified"] is True
    assert receipt["mandatory_verification_complete"] is True


def test_outer_token_zero_accepts_missing_bindings_but_rejects_null(
    runner: Any,
) -> None:
    prepared = _prepared(runner)
    missing_log: list[str] = []
    missing = _cleanup_backend(
        runner,
        prepared,
        log=missing_log,
        phase2_entered=False,
        token_entered=True,
    )
    missing.iam_admin.omit_empty_bindings = True
    missing_receipt = missing.cleanup_failure(
        failure_evidence_sha256="46" * 32
    )
    assert missing_receipt["token_creator_zero_verified"] is True
    assert missing_receipt["mandatory_verification_complete"] is True

    null_log: list[str] = []
    malformed = _cleanup_backend(
        runner,
        prepared,
        log=null_log,
        phase2_entered=False,
        token_entered=True,
    )
    malformed.iam_admin.null_empty_bindings = True
    malformed_receipt = malformed.cleanup_failure(
        failure_evidence_sha256="47" * 32
    )
    assert malformed_receipt["token_creator_zero_verified"] is False
    assert malformed_receipt["mandatory_verification_complete"] is False
    assert null_log[-3:] == ["sa", "sa_get", "compute"]


def test_outer_token_cleanup_timeout_never_readds_or_blocks_other_cleanup(
    runner: Any,
) -> None:
    prepared = _prepared(runner)
    log: list[str] = []
    backend = _cleanup_backend(
        runner,
        prepared,
        log=log,
        phase2_entered=False,
        token_entered=True,
    )
    backend.iam_admin.stale_token_forever = True
    receipt = backend.cleanup_failure(
        failure_evidence_sha256="45" * 32
    )
    assert log.count("token_remove") == 1
    assert "token_add" not in log
    assert "exact_pair_attempt0" not in log
    assert log[-3:] == ["sa", "sa_get", "compute"]
    assert backend._test_clock.elapsed == (
        runner.token_barrier.MAX_PROPAGATION_SECONDS
    )
    assert receipt["token_creator_zero_verified"] is False
    assert receipt["mandatory_verification_complete"] is False


def test_deleted_controller_sa_policy_get404_proves_phase2_and_token_zero(
    runner: Any,
) -> None:
    prepared = _prepared(runner)
    prepared.phase2_iam_plan.update(
        {
            "phase2_bindings": {
                "controller": [
                    {
                        "target": "controller_service_account",
                    }
                ],
                "worker": [{"target": "project"}],
            },
            "principals": {
                "controller_principal": "serviceAccount:controller",
                "worker_principal": "serviceAccount:worker",
            },
        }
    )
    log: list[str] = []
    backend = _cleanup_backend(
        runner,
        prepared,
        log=log,
        phase2_entered=True,
        token_entered=True,
    )
    backend.iam_admin.controller_target_deleted = True
    phase2 = backend._verify_phase2_principals_absent()
    token = backend._verify_token_creator_absent()
    assert phase2["targeted_principal_binding_count"] == 0
    assert phase2["deleted_target_get404s"] == [
        "controller_service_account"
    ]
    assert token["target_policy_get_status"] == 404
    assert token["token_creator_member_count"] == 0


def test_unknown_controller_sa_cleanup_still_requires_and_accepts_get404(
    runner: Any,
) -> None:
    prepared = _prepared(runner)
    log: list[str] = []
    backend = _cleanup_backend(
        runner,
        prepared,
        log=log,
        phase2_entered=False,
        token_entered=False,
    )
    backend.controller_sa_admin.raise_unknown_cleanup = True
    receipt = backend.cleanup_failure(
        failure_evidence_sha256="43" * 32
    )
    assert log == ["sa", "sa_get", "compute"]
    assert receipt["controller_service_account_get404_verified"] is True
    assert receipt["mandatory_verification_complete"] is True


def test_phase2_zero_readback_attempts_every_target_before_failing(
    runner: Any,
) -> None:
    prepared = _prepared(runner)
    prepared.phase2_iam_plan.update(
        {
            "phase2_bindings": {
                "controller": [
                    {"target": "bucket"},
                    {"target": "controller_service_account"},
                ],
                "worker": [{"target": "project"}],
            },
            "principals": {
                "controller_principal": "serviceAccount:controller",
                "worker_principal": "serviceAccount:worker",
            },
        }
    )
    log: list[str] = []
    backend = _cleanup_backend(
        runner,
        prepared,
        log=log,
        phase2_entered=True,
        token_entered=False,
    )
    targets_seen: list[Any] = []

    def get_policy(target: Any) -> dict[str, Any]:
        targets_seen.append(target)
        if target is runner.rest_iam.PolicyTarget.BUCKET:
            raise runner.rest_iam.RestIamAdminError(
                "iam_policy_get_failed",
                operation="get_bucket_policy",
                status_code=500,
            )
        return {"version": 3, "etag": "fixture", "bindings": []}

    backend.iam_admin.get_policy = get_policy
    with pytest.raises(RuntimeError, match="readback incomplete"):
        backend._verify_phase2_principals_absent()
    assert targets_seen == [
        runner.rest_iam.PolicyTarget.BUCKET,
        runner.rest_iam.PolicyTarget.CONTROLLER_SERVICE_ACCOUNT,
        runner.rest_iam.PolicyTarget.PROJECT,
    ]
