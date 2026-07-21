from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter as adapter,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as payload_plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_release_v2
    as pair_release,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_phase2_iam_lifecycle_v2
    as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_phase2_iam_plan_v2
    as phase2_iam,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_token_barrier_v1
    as token_barrier,
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
RUN_NONCE = "e5" * 32
ISSUED = 1_900_100_000
EXPIRES = ISSUED + 3_600
RAW_SECRET = "raw-controller-token-must-never-serialize"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def controller_public_key_record() -> dict[str, Any]:
    return _read_json(STEP11_ROOT / "controller_public_key.json")


@pytest.fixture(scope="module")
def payload_contracts(
    controller_public_key_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage1 = _read_json(STEP11_ROOT / "transport_contract.json")
    done = _read_json(STEP11_ROOT / "late_done_envelope.json")
    stage1_receive = adapter.build_receive(
        stage1["adapter_preview"],
        done_records=[done],
    )
    wheel = next(
        row
        for row in stage1["outer_package_manifest"]["objects"]
        if row["kind"] == "offline_numpy_cp311_manylinux_x86_64_wheel"
    )
    contracts = [
        payload_transport.build_job_contract(
            package_dir=PACKAGE_DIR,
            stage_id=payload_plan.STAGE2_ID,
            job_id=job_id,
            attempt_index=0,
            offline_wheel_record=wheel,
            controller_public_key_record=controller_public_key_record,
            prerequisite_stage1_preview=stage1["adapter_preview"],
            prerequisite_stage1_receive=stage1_receive,
        )
        for job_id in payload_plan.STAGE2_JOB_IDS
    ]
    return contracts[0], contracts[1]


@pytest.fixture(scope="module")
def deployment(
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> dict[str, Any]:
    candidate, reference = payload_contracts
    return deployment_v2.build_step12b_deployment_contract(
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
    )


@pytest.fixture(scope="module")
def plan(
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> dict[str, Any]:
    candidate, reference = payload_contracts
    return phase2_iam.build_step12b_phase2_iam_plan(
        deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
        issued_at_unix_seconds=ISSUED,
        expires_at_unix_seconds=EXPIRES,
    )


def _role_readbacks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": row["name"],
            "stage": row["stage"],
            "included_permissions": list(row["included_permissions"]),
            "deleted": False,
            "get_status": 200,
            "readback_complete": True,
        }
        for row in plan["custom_role_readback_contract"]["requirements"]
    ]


class _NoReadTokenSource:
    expire_time = "2030-03-17T19:46:40Z"

    def __init__(self) -> None:
        self.access_count = 0
        self.raw = RAW_SECRET

    def access_token(self) -> str:
        self.access_count += 1
        raise AssertionError("Phase2 lifecycle must not access raw token")


def _barrier_outcome(
    deployment: dict[str, Any],
) -> tuple[token_barrier.TokenBarrierOutcome, _NoReadTokenSource]:
    source = _NoReadTokenSource()
    zero_readback = {
        "zero_observed": True,
        "poll_count": 1,
        "stale_present_count": 0,
        "observation_sha256s": ["3" * 64],
        "observation_sha256s_sha256": token_barrier.canonical_sha256(
            ["3" * 64]
        ),
        "sleep_delays_seconds": [],
        "elapsed_seconds": 0.5,
        "maximum_seconds": token_barrier.MAX_PROPAGATION_SECONDS,
        "failure_reason": None,
        "second_add_performed": False,
        "controller_token_reminted": False,
    }
    body = {
        "schema": token_barrier.SCHEMA,
        "status": token_barrier.SUCCESS_STATUS,
        "controller_service_account": deployment[
            "controller_service_account"
        ]["email"],
        "run_scoped_controller_required": True,
        "binding_purpose": "initiator_controller_token_creator",
        "binding_role": phase2_iam.TOKEN_CREATOR_ROLE,
        "binding_member": "user:giovinco.080807@gmail.com",
        "binding_condition_sha256": "1" * 64,
        "binding_add_attempts": 1,
        "binding_add_outcome": "confirmed_changed",
        "binding_add_readback_sha256": "2" * 64,
        "token_attempt_count": 1,
        "failed_token_attempt_count": 0,
        "token_attempts": [],
        "started_at_unix_seconds": ISSUED - 20,
        "finished_at_unix_seconds": ISSUED - 10,
        "elapsed_seconds": 10.0,
        "maximum_propagation_seconds": 480,
        "token_lifetime_seconds": 3_600,
        "token_expire_time": source.expire_time,
        "fixed_nonrefreshing_controller_credential": True,
        "controller_token_remint_forbidden": True,
        "token_creator_revoke_attempts": 1,
        "token_creator_revoke_changed": True,
        "token_creator_revoke_readback_sha256": "3" * 64,
        "token_creator_revoke_zero_readback": zero_readback,
        "token_creator_revoke_zero_readback_evidence_sha256": (
            token_barrier.canonical_sha256(zero_readback)
        ),
        "token_creator_live_after_barrier": False,
        "phase2_started": False,
        "phase2_must_exclude_token_creator": True,
        "run_scoped_controller_delete_after_pair_claim_required": True,
        "vm_insert_attempt_count": 0,
        "access_token_stored": False,
        "authorization_header_stored": False,
        "response_body_stored": False,
        "cloud_mutation_performed": True,
        "current_profile_changed": False,
    }
    receipt = {
        **body,
        "receipt_sha256": token_barrier.canonical_sha256(body),
    }
    return token_barrier.TokenBarrierOutcome(source, receipt), source


def _condition_key(value: Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    return phase2_iam.canonical_sha256(dict(value))


class _FakeIamAdmin:
    def __init__(
        self,
        deployment: dict[str, Any],
        *,
        fail_add_at: int | None = None,
        omit_add_at: int | None = None,
        extra_after_install: bool = False,
        fail_remove_at: int | None = None,
    ) -> None:
        self.project = phase2_iam.PROJECT
        self.bucket = phase2_iam.BUCKET
        self.worker_service_account = phase2_iam.WORKER_SERVICE_ACCOUNT
        self.controller_service_account = deployment[
            "controller_service_account"
        ]["email"]
        self.policies = {
            target.value: {
                "version": 3,
                "etag": f"etag-{target.value}",
                "bindings": [],
            }
            for target in (
                rest_iam.PolicyTarget.PROJECT,
                rest_iam.PolicyTarget.BUCKET,
                rest_iam.PolicyTarget.WORKER_SERVICE_ACCOUNT,
            )
        }
        self.log: list[dict[str, Any]] = []
        self.add_count = 0
        self.remove_count = 0
        self.fail_add_at = fail_add_at
        self.omit_add_at = omit_add_at
        self.extra_after_install = extra_after_install
        self.extra_installed = False
        self.fail_remove_at = fail_remove_at
        self.remove_failure_used = False

    @staticmethod
    def _target_value(
        target: rest_iam.PolicyTarget | str,
    ) -> str:
        return target.value if isinstance(target, rest_iam.PolicyTarget) else target

    def get_policy(
        self, target: rest_iam.PolicyTarget | str
    ) -> Mapping[str, Any]:
        key = self._target_value(target)
        self.log.append({"operation": "get", "target": key})
        if (
            self.extra_after_install
            and self.add_count >= 8
            and not self.extra_installed
        ):
            self.policies["project"]["bindings"].append(
                {
                    "role": "roles/viewer",
                    "members": [
                        f"serviceAccount:{self.controller_service_account}"
                    ],
                    "condition": {
                        "title": "unexpected",
                        "expression": "request.time < timestamp(\"2030-01-01T00:00:00Z\")",
                    },
                }
            )
            self.extra_installed = True
        return copy.deepcopy(self.policies[key])

    def _mutate(
        self,
        target: rest_iam.PolicyTarget | str,
        *,
        role: str,
        member: str,
        condition: Mapping[str, Any] | None,
        add: bool,
    ) -> rest_iam.PolicyMutationResult:
        target_enum = (
            target
            if isinstance(target, rest_iam.PolicyTarget)
            else rest_iam.PolicyTarget(target)
        )
        key = target_enum.value
        operation = "add" if add else "remove"
        self.log.append(
            {
                "operation": operation,
                "target": key,
                "role": role,
                "member": member,
                "condition": copy.deepcopy(condition),
            }
        )
        if add:
            self.add_count += 1
            if self.fail_add_at == self.add_count:
                raise RuntimeError("injected add failure with secret: " + RAW_SECRET)
            if self.omit_add_at == self.add_count:
                return rest_iam.PolicyMutationResult(
                    target=target_enum,
                    changed=True,
                    attempts=1,
                    policy=copy.deepcopy(self.policies[key]),
                )
        else:
            self.remove_count += 1
            if (
                self.fail_remove_at == self.remove_count
                and not self.remove_failure_used
            ):
                self.remove_failure_used = True
                raise RuntimeError(
                    "injected remove failure with secret: " + RAW_SECRET
                )

        policy = self.policies[key]
        matching = None
        for row in policy["bindings"]:
            if (
                row["role"] == role
                and _condition_key(row.get("condition"))
                == _condition_key(condition)
            ):
                matching = row
                break
        if add:
            if matching is None:
                matching = {
                    "role": role,
                    "members": [],
                    "condition": copy.deepcopy(condition),
                }
                policy["bindings"].append(matching)
            changed = member not in matching["members"]
            if changed:
                matching["members"].append(member)
                matching["members"].sort()
        else:
            changed = matching is not None and member in matching["members"]
            if changed:
                matching["members"].remove(member)
                if not matching["members"]:
                    policy["bindings"].remove(matching)
        return rest_iam.PolicyMutationResult(
            target=target_enum,
            changed=changed,
            attempts=1,
            policy=copy.deepcopy(policy),
        )

    def add_binding(
        self,
        target: rest_iam.PolicyTarget | str,
        *,
        role: str,
        member: str,
        condition: Mapping[str, Any] | None,
    ) -> rest_iam.PolicyMutationResult:
        return self._mutate(
            target,
            role=role,
            member=member,
            condition=condition,
            add=True,
        )

    def remove_binding(
        self,
        target: rest_iam.PolicyTarget | str,
        *,
        role: str,
        member: str,
        condition: Mapping[str, Any] | None,
    ) -> rest_iam.PolicyMutationResult:
        return self._mutate(
            target,
            role=role,
            member=member,
            condition=condition,
            add=False,
        )

    def preinstall(self, binding: Mapping[str, Any]) -> None:
        self._mutate(
            rest_iam.PolicyTarget(binding["target"]),
            role=binding["role"],
            member=binding["member"],
            condition=binding["condition"],
            add=True,
        )
        self.log.clear()
        self.add_count = 0


def _install(
    *,
    admin: _FakeIamAdmin,
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> tuple[subject.Phase2IamInstalledCapability, _NoReadTokenSource]:
    candidate, reference = payload_contracts
    barrier, source = _barrier_outcome(deployment)
    installed = subject.install_step12b_phase2_iam(
        iam_admin=admin,
        phase2_iam_plan=plan,
        deployment_contract=deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
        token_barrier_outcome=barrier,
        custom_role_readbacks=_role_readbacks(plan),
        observed_at_unix_seconds=ISSUED + 30,
    )
    return installed, source


def _claim_receipts(
    deployment: dict[str, Any],
) -> list[dict[str, Any]]:
    receipts = []
    for position, (job_id, instance) in enumerate(
        zip(
            deployment["selected_job_ids"],
            deployment["instances"],
            strict=True,
        )
    ):
        provider_id = str(123_456 + position)
        pre = f"preFingerprint{position}"
        run_nonce_digest = pair_release.canonical_sha256(
            {"nonce": deployment["run_nonce"]}
        )
        authorization_nonce_digest = pair_release.canonical_sha256(
            {"nonce": f"{position + 6}" * 64}
        )
        claim_nonce_digest = pair_release.canonical_sha256(
            {"nonce": f"{position + 8}" * 64}
        )
        authorization_sha = f"{position + 1}" * 64
        claim_sha = f"{position + 3}" * 64
        body = {
            "schema": pair_release.CLAIM_CAS_READBACK_SCHEMA,
            "deployment_contract_sha256": deployment[
                "deployment_contract_sha256"
            ],
            "external_job_id": job_id,
            "inner_job_id": instance["inner_job_id"],
            "source_role": instance["source_role"],
            "instance_name": instance["instance_name"],
            "provider_instance_id": provider_id,
            "authorization_sha256": authorization_sha,
            "claim_sha256": claim_sha,
            "claim_metadata_value_sha256": claim_sha,
            "claimed_metadata_sha256": f"{position + 5}" * 64,
            "controller_pre_cas_metadata_fingerprint": pre,
            "controller_post_cas_metadata_fingerprint": (
                f"postFingerprint{position}"
            ),
            "run_nonce_digest": run_nonce_digest,
            "authorization_nonce_digest": authorization_nonce_digest,
            "claim_nonce_digest": claim_nonce_digest,
            "run_authorization_claim_nonces_unique": True,
            "claim_present_exactly_once": True,
            "claim_cas_readback_complete": True,
            "worker_observed_metadata_fingerprint": False,
        }
        receipts.append(
            {
                **body,
                "receipt_sha256": pair_release.canonical_sha256(body),
            }
        )
    return receipts


def _controller_zero(
    *,
    admin: _FakeIamAdmin,
    installed: subject.Phase2IamInstalledCapability,
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> subject.ControllerBindingsZeroCapability:
    candidate, reference = payload_contracts
    claims = subject.validate_pair_claim_cas_readbacks_capability(
        deployment_contract=deployment,
        claim_cas_readback_receipts=_claim_receipts(deployment),
    )
    return subject.remove_step12b_phase2_controller_bindings(
        iam_admin=admin,
        installed=installed,
        pair_claims=claims,
        phase2_iam_plan=plan,
        deployment_contract=deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
    )


def test_install_sequence_is_exact_and_token_is_never_read(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    admin = _FakeIamAdmin(deployment)
    installed, source = _install(
        admin=admin,
        plan=plan,
        deployment=deployment,
        payload_contracts=payload_contracts,
        controller_public_key_record=controller_public_key_record,
    )
    assert source.access_count == 0
    assert installed.phase2_iam_plan_sha256 == plan["plan_sha256"]
    receipt = installed.install_receipt
    assert receipt["schema"] == subject.INSTALL_RECEIPT_SCHEMA
    assert receipt["installed_controller_binding_count"] == 5
    assert receipt["installed_worker_binding_count"] == 3
    assert receipt["all_phase2_bindings_present_exactly_once"] is True
    assert receipt["token_creator_binding_included"] is False
    adds = [row for row in admin.log if row["operation"] == "add"]
    assert len(adds) == 8
    expected = {
        row["binding_sha256"]: row["purpose"]
        for row in [
            *plan["phase2_bindings"]["controller"],
            *plan["phase2_bindings"]["worker"],
        ]
    }
    observed_purposes = [
        expected[
            next(
                binding["binding_sha256"]
                for binding in [
                    *plan["phase2_bindings"]["controller"],
                    *plan["phase2_bindings"]["worker"],
                ]
                if binding["target"] == row["target"]
                and binding["role"] == row["role"]
                and binding["member"] == row["member"]
                and binding["condition"] == row["condition"]
            )
        ]
        for row in adds
    ]
    assert observed_purposes == [
        "worker_package_and_result_reader",
        "worker_result_creator",
        "worker_self_delete",
        "controller_service_usage",
        "controller_zone_operation_reader",
        "controller_instance_lifecycle",
        "controller_worker_act_as",
        "controller_vm_launch",
    ]
    serialized = json.dumps(receipt, sort_keys=True)
    assert RAW_SECRET not in serialized
    assert phase2_iam.TOKEN_CREATOR_ROLE not in serialized


def test_installed_capability_returns_fresh_validated_full_readback(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    admin = _FakeIamAdmin(deployment)
    installed, source = _install(
        admin=admin,
        plan=plan,
        deployment=deployment,
        payload_contracts=payload_contracts,
        controller_public_key_record=controller_public_key_record,
    )
    candidate, reference = payload_contracts
    readback = subject.get_step12b_phase2_exact_readback_receipt(
        installed,
        phase2_iam_plan=plan,
        deployment_contract=deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
    )
    assert source.access_count == 0
    assert (
        readback["receipt_sha256"]
        == installed.exact_readback_receipt_sha256
    )
    assert readback["controller_binding_count"] == 5
    assert readback["worker_binding_count"] == 3
    assert readback["all_phase2_bindings_present_exactly_once"] is True
    readback["binding_readbacks"].clear()
    fresh = subject.get_step12b_phase2_exact_readback_receipt(
        installed,
        phase2_iam_plan=plan,
        deployment_contract=deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
    )
    assert len(fresh["binding_readbacks"]) == 8
    serialized = json.dumps(fresh, sort_keys=True)
    assert RAW_SECRET not in serialized
    assert '"raw_policy"' not in serialized
    assert '"access_token"' not in serialized


def test_stale_initial_binding_fails_without_clear_and_continue(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    admin = _FakeIamAdmin(deployment)
    admin.preinstall(plan["phase2_bindings"]["worker"][0])
    with pytest.raises(subject.Phase2IamLifecycleError) as captured:
        _install(
            admin=admin,
            plan=plan,
            deployment=deployment,
            payload_contracts=payload_contracts,
            controller_public_key_record=controller_public_key_record,
        )
    receipt = captured.value.receipt
    assert receipt["stage"] == "initial_exact_zero"
    assert receipt["stale_binding_cleared"] is False
    assert receipt["clear_and_continue_forbidden"] is True
    assert receipt["iam_mutation_attempted"] is False
    assert not any(
        row["operation"] in {"add", "remove"} for row in admin.log
    )


def test_wrong_plan_digest_fails_before_any_admin_call(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    admin = _FakeIamAdmin(deployment)
    changed = copy.deepcopy(plan)
    changed["instance_boundary"]["third_vm_authorized"] = True
    with pytest.raises(ValueError, match="digest changed"):
        _install(
            admin=admin,
            plan=changed,
            deployment=deployment,
            payload_contracts=payload_contracts,
            controller_public_key_record=controller_public_key_record,
        )
    assert admin.log == []


@pytest.mark.parametrize(
    "admin_kwargs",
    [
        {"omit_add_at": 4},
        {"extra_after_install": True},
    ],
)
def test_omitted_or_extra_binding_fails_and_cleans_controller_first(
    admin_kwargs: dict[str, Any],
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    admin = _FakeIamAdmin(deployment, **admin_kwargs)
    with pytest.raises(subject.Phase2IamLifecycleError) as captured:
        _install(
            admin=admin,
            plan=plan,
            deployment=deployment,
            payload_contracts=payload_contracts,
            controller_public_key_record=controller_public_key_record,
        )
    removes = [row for row in admin.log if row["operation"] == "remove"]
    assert len(removes) == 8
    controller_principal = plan["principals"]["controller_principal"]
    assert all(row["member"] == controller_principal for row in removes[:5])
    assert all(
        row["member"] == plan["principals"]["worker_principal"]
        for row in removes[5:]
    )
    assert captured.value.receipt["cleanup_controller_before_worker"] is True


def test_add_failure_is_secret_free_and_cleanup_is_ordered(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    admin = _FakeIamAdmin(deployment, fail_add_at=4)
    with pytest.raises(subject.Phase2IamLifecycleError) as captured:
        _install(
            admin=admin,
            plan=plan,
            deployment=deployment,
            payload_contracts=payload_contracts,
            controller_public_key_record=controller_public_key_record,
        )
    receipt = captured.value.receipt
    assert receipt["stage"] == "install_and_exact_readback"
    assert receipt["cleanup_controller_before_worker"] is True
    serialized = json.dumps(receipt, sort_keys=True)
    assert RAW_SECRET not in serialized
    assert "Authorization" not in serialized
    removes = [row for row in admin.log if row["operation"] == "remove"]
    assert len(removes) == 8
    assert all(
        row["member"] == plan["principals"]["controller_principal"]
        for row in removes[:5]
    )


def test_controller_removal_requires_both_claims_and_yields_pair_receipt(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    admin = _FakeIamAdmin(deployment)
    installed, _ = _install(
        admin=admin,
        plan=plan,
        deployment=deployment,
        payload_contracts=payload_contracts,
        controller_public_key_record=controller_public_key_record,
    )
    with pytest.raises(ValueError, match="exactly two"):
        subject.validate_pair_claim_cas_readbacks_capability(
            deployment_contract=deployment,
            claim_cas_readback_receipts=_claim_receipts(deployment)[:1],
        )
    before = len(
        [row for row in admin.log if row["operation"] == "remove"]
    )
    zero = _controller_zero(
        admin=admin,
        installed=installed,
        plan=plan,
        deployment=deployment,
        payload_contracts=payload_contracts,
        controller_public_key_record=controller_public_key_record,
    )
    removes = [row for row in admin.log if row["operation"] == "remove"]
    assert len(removes) - before == 5
    assert all(
        row["member"] == plan["principals"]["controller_principal"]
        for row in removes[before:]
    )
    receipt = zero.phase2_zero_receipt
    candidate, reference = payload_contracts
    assert pair_release.validate_phase2_zero_receipt(
        receipt,
        deployment_contract=deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
        phase2_iam_plan=plan,
    ) == receipt
    memberships = admin.get_policy(
        rest_iam.PolicyTarget.BUCKET
    )["bindings"]
    assert any(
        plan["principals"]["worker_principal"] in row["members"]
        for row in memberships
    )


def test_partial_controller_remove_never_issues_zero_and_cleans_all(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    admin = _FakeIamAdmin(deployment, fail_remove_at=2)
    installed, _ = _install(
        admin=admin,
        plan=plan,
        deployment=deployment,
        payload_contracts=payload_contracts,
        controller_public_key_record=controller_public_key_record,
    )
    claims = subject.validate_pair_claim_cas_readbacks_capability(
        deployment_contract=deployment,
        claim_cas_readback_receipts=_claim_receipts(deployment),
    )
    candidate, reference = payload_contracts
    with pytest.raises(subject.Phase2IamLifecycleError) as captured:
        subject.remove_step12b_phase2_controller_bindings(
            iam_admin=admin,
            installed=installed,
            pair_claims=claims,
            phase2_iam_plan=plan,
            deployment_contract=deployment,
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=controller_public_key_record,
            run_nonce=RUN_NONCE,
        )
    assert captured.value.receipt["pair_release_authorized"] is False
    cleanup = captured.value.receipt["cleanup_records"]
    assert [row["cleanup_group"] for row in cleanup[:5]] == [
        "controller"
    ] * 5
    assert [row["cleanup_group"] for row in cleanup[5:]] == [
        "worker"
    ] * 3
    assert RAW_SECRET not in json.dumps(captured.value.receipt)


def test_final_worker_remove_is_exactly_three_and_leaves_zero(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    admin = _FakeIamAdmin(deployment)
    installed, _ = _install(
        admin=admin,
        plan=plan,
        deployment=deployment,
        payload_contracts=payload_contracts,
        controller_public_key_record=controller_public_key_record,
    )
    controller_zero = _controller_zero(
        admin=admin,
        installed=installed,
        plan=plan,
        deployment=deployment,
        payload_contracts=payload_contracts,
        controller_public_key_record=controller_public_key_record,
    )
    candidate, reference = payload_contracts
    before = len(
        [row for row in admin.log if row["operation"] == "remove"]
    )
    worker_zero = subject.remove_step12b_phase2_worker_bindings_final(
        iam_admin=admin,
        installed=installed,
        controller_zero=controller_zero,
        completion_kind="pair_done_readback",
        completion_evidence_sha256="9" * 64,
        phase2_iam_plan=plan,
        deployment_contract=deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
    )
    removes = [row for row in admin.log if row["operation"] == "remove"]
    assert len(removes) - before == 3
    assert all(
        row["member"] == plan["principals"]["worker_principal"]
        for row in removes[before:]
    )
    assert worker_zero.receipt["removed_worker_binding_count"] == 3
    assert worker_zero.receipt["final_targeted_binding_count"] == 0
    assert all(
        not admin.get_policy(target)["bindings"]
        for target in (
            rest_iam.PolicyTarget.PROJECT,
            rest_iam.PolicyTarget.BUCKET,
            rest_iam.PolicyTarget.WORKER_SERVICE_ACCOUNT,
        )
    )


def test_explicit_failure_cleanup_never_mutates_token_creator(
    plan: dict[str, Any],
    deployment: dict[str, Any],
    payload_contracts: tuple[dict[str, Any], dict[str, Any]],
    controller_public_key_record: dict[str, Any],
) -> None:
    admin = _FakeIamAdmin(deployment)
    installed, _ = _install(
        admin=admin,
        plan=plan,
        deployment=deployment,
        payload_contracts=payload_contracts,
        controller_public_key_record=controller_public_key_record,
    )
    assert installed.phase2_iam_plan_sha256 == plan["plan_sha256"]
    candidate, reference = payload_contracts
    receipt = subject.cleanup_step12b_phase2_iam_on_failure(
        iam_admin=admin,
        phase2_iam_plan=plan,
        deployment_contract=deployment,
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=controller_public_key_record,
        run_nonce=RUN_NONCE,
        failure_evidence_sha256="a" * 64,
    )
    assert receipt["cleanup_controller_before_worker"] is True
    assert len(receipt["cleanup_records"]) == 8
    assert all(
        row.get("role") != phase2_iam.TOKEN_CREATOR_ROLE
        for row in admin.log
    )
    assert RAW_SECRET not in json.dumps(receipt)


_EMPTY_POLICY_PLAN = {
    "principals": {
        "controller_principal": (
            "serviceAccount:ctrl@ofc-solver-485418.iam.gserviceaccount.com"
        ),
        "worker_principal": (
            "serviceAccount:ofc-m31-t3-diagnostic@"
            "ofc-solver-485418.iam.gserviceaccount.com"
        ),
    }
}


def test_targeted_memberships_treats_empty_policy_as_zero():
    # Regression: a fresh worker service account returns a policy with no
    # "bindings" key (or an explicit null). That legitimately means zero
    # memberships - the initial-zero state - and must not abort as a shape
    # violation. This is the live-only bug that stopped the Step12h canary.
    policies = {
        "project": {
            "bindings": [
                {"role": "roles/viewer", "members": ["user:someone@x.com"]}
            ]
        },
        "worker_service_account": {"version": 1, "etag": "e"},  # no bindings key
        "bucket": {"bindings": None},  # explicit null
    }
    rows = subject._targeted_memberships(policies, _EMPTY_POLICY_PLAN)
    assert rows == []


def test_targeted_memberships_still_finds_present_principal():
    policies = {
        "worker_service_account": {
            "bindings": [
                {
                    "role": "roles/iam.serviceAccountUser",
                    "members": [
                        "serviceAccount:ofc-m31-t3-diagnostic@"
                        "ofc-solver-485418.iam.gserviceaccount.com"
                    ],
                }
            ]
        }
    }
    rows = subject._targeted_memberships(policies, _EMPTY_POLICY_PLAN)
    assert len(rows) == 1
    assert rows[0]["target"] == "worker_service_account"
    assert rows[0]["role"] == "roles/iam.serviceAccountUser"


def test_targeted_memberships_still_rejects_malformed_bindings():
    policies = {"project": {"bindings": "not-a-list"}}
    with pytest.raises(subject._LifecycleAbort, match="policy_binding_shape_changed"):
        subject._targeted_memberships(policies, _EMPTY_POLICY_PLAN)



def test_failure_receipt_carries_underlying_step11_error_detail():
    # Diagnostic instrumentation: a step11 set/readback RestIamAdminError must
    # be preserved in the sealed failure receipt as non-sensitive detail (code,
    # HTTP status, operation) - no body, no token - so the cause is visible.
    err = rest_iam.RestIamAdminError(
        "iam_policy_set_failed",
        operation="add_bucket_policy_binding",
        status_code=400,
    )
    exc = subject._failure(
        stage="install_and_exact_readback",
        reason=err.code,
        plan={"source_deployment": {"deployment_contract_sha256": "x"}, "plan_sha256": "y"},
        mutation_records=[],
        cleanup_records=[],
        underlying_error={
            "error_code": err.code,
            "status_code": err.status_code,
            "operation": err.operation,
        },
    )
    receipt = exc.receipt
    assert receipt["failure_reason"] == "iam_policy_set_failed"
    assert receipt["underlying_error"] == {
        "error_code": "iam_policy_set_failed",
        "status_code": 400,
        "operation": "add_bucket_policy_binding",
    }
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    assert subject.canonical_sha256(body) == receipt["receipt_sha256"]


def test_failure_receipt_underlying_error_defaults_to_none():
    exc = subject._failure(
        stage="initial_exact_zero",
        reason="stale_targeted_binding_present",
        plan={"source_deployment": {"deployment_contract_sha256": "x"}, "plan_sha256": "y"},
        mutation_records=[],
        cleanup_records=[],
    )
    assert exc.receipt["underlying_error"] is None
