from __future__ import annotations

import copy

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave
from ofc_regular import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as worker_iam
from ofc_regular import hu_m31_t3_step6d_full100_wave_worker_identity_v2 as subject


OBSERVED_AT = "2027-01-15T08:00:00Z"
PROVIDER = "fake-provider-test-v1"


def _evidence(
    run_name: str = "regular-hu-m31-c02-f100wv2-workeridentity-001",
) -> tuple[dict, dict, dict, dict]:
    plan = wave.build_wave_plan(
        run_name=run_name,
        identity_salt="1234567890abcdef1234567890abcdef",
        package_sha256="2" * 64,
        image_digest="sha256:" + "3" * 64,
    )
    transition = wave.build_observed_transition(
        plan,
        project_id="ofc-solver-485418",
        zone="asia-northeast1-b",
        observed_at_utc="2027-01-15T07:00:00Z",
        previous_transition_digest=None,
        attempt_history=wave.empty_attempt_history(plan),
    )
    ledger = wave.build_attempt_ledger(plan, transitions=[transition])
    resume = wave.build_resume_plan(plan, attempt_ledger=ledger)
    identity = subject.build_worker_identity_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        wave_index=0,
    )
    return plan, ledger, resume, identity


def _inventory_observations() -> list[dict]:
    return [
        {
            **row,
            "project_id": subject.PROJECT,
            "unique_id": str(100_000_000_000_000_000_000 + index),
            "disabled": False,
            "exists": True,
        }
        for index, row in enumerate(subject.fixed_pool_accounts())
    ]


def _inventory(evidence: tuple[dict, dict, dict, dict]) -> dict:
    plan, ledger, resume, identity = evidence
    return subject.build_inventory_receipt(
        identity_plan=identity,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        observed_accounts=_inventory_observations(),
        observed_at_utc=OBSERVED_AT,
        provider_source=PROVIDER,
    )


def _act_as_observations(identity: dict) -> list[dict]:
    return [
        {
            "account_id": row["account_id"],
            "email": row["service_account_email"],
            "name": row["service_account_name"],
            "http_method": "POST",
            "requested_permissions": [subject.ACT_AS_PERMISSION],
            "granted_permissions": [subject.ACT_AS_PERMISSION],
        }
        for row in identity["selected_workers"]
    ]


def _empty_project_policy() -> dict:
    return {
        "version": 3,
        "etag": "BwYFixedPoolAudit==",
        "bindings": [
            {
                "role": "roles/viewer",
                "members": ["user:unrelated@example.com"],
            },
            {
                "role": "roles/logging.viewer",
                "members": [
                    "serviceAccount:unrelated@ofc-solver-485418.iam.gserviceaccount.com"
                ],
                "condition": {
                    "title": "unrelated",
                    "expression": "request.time < timestamp(\"2028-01-01T00:00:00Z\")",
                },
            },
        ],
        "auditConfigs": [],
    }


def _reseal(value: dict, field: str) -> None:
    value[field] = subject.canonical_sha256(
        {key: item for key, item in value.items() if key != field}
    )


def test_fixed_pool_and_wave_mapping_are_exact_and_run_independent() -> None:
    plan, ledger, resume, identity = _evidence()
    assert subject.PROJECT == "ofc-solver-485418"
    assert subject.ACCOUNT_IDS == tuple(
        f"ofc-f100-worker-{index:02d}" for index in range(8)
    )
    assert identity["pool_account_emails"] == list(subject.ACCOUNT_EMAILS)
    assert identity["selected_count"] == 8
    assert [row["job_id"] for row in identity["selected_workers"]] == [
        row["job_id"] for row in resume["selected_attempts"]
    ]
    assert [row["worker_slot"] for row in identity["selected_workers"]] == list(range(8))
    assert [row["service_account_email"] for row in identity["selected_workers"]] == list(
        subject.ACCOUNT_EMAILS
    )
    assert subject.validate_worker_identity_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        value=identity,
    ) == identity

    plan2, _, resume2, _ = _evidence(
        "regular-hu-m31-c02-f100wv2-workeridentity-002"
    )
    assert worker_iam.default_service_accounts(plan, resume) == {
        row["job_id"]: subject.ACCOUNT_EMAILS[index]
        for index, row in enumerate(resume["selected_attempts"])
    }
    assert list(worker_iam.default_service_accounts(plan2, resume2).values()) == list(
        subject.ACCOUNT_EMAILS
    )
    assert identity["run_specific_account_creation"] is False
    assert identity["bucket_iam_managed_here"] is False
    assert identity["cloud_mutated"] is False
    assert identity["current_profile_changed"] is False


def test_inventory_proves_all_pool_and_selected_accounts_enabled_and_unique() -> None:
    evidence = _evidence()
    plan, ledger, resume, identity = evidence
    receipt = _inventory(evidence)
    assert subject.validate_inventory_receipt(
        identity_plan=identity,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        value=receipt,
    ) == receipt
    assert receipt["pool_account_count"] == 8
    assert receipt["selected_account_count"] == 8
    assert receipt["all_pool_accounts_exist"] is True
    assert receipt["all_selected_accounts_exist"] is True
    assert receipt["all_enabled"] is True
    assert receipt["unique_ids_unique"] is True


@pytest.mark.parametrize("mode", ["missing", "disabled", "wrong_project", "duplicate_uid"])
def test_inventory_fails_closed_on_incomplete_or_wrong_identity(mode: str) -> None:
    plan, ledger, resume, identity = _evidence()
    rows = _inventory_observations()
    if mode == "missing":
        rows.pop()
    elif mode == "disabled":
        rows[2]["disabled"] = True
    elif mode == "wrong_project":
        rows[3]["project_id"] = "some-other-project"
    else:
        rows[4]["unique_id"] = rows[0]["unique_id"]
    with pytest.raises(ValueError):
        subject.build_inventory_receipt(
            identity_plan=identity,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            observed_accounts=rows,
            observed_at_utc=OBSERVED_AT,
            provider_source=PROVIDER,
        )


def test_act_as_receipt_requires_exact_post_test_for_every_selected_account() -> None:
    evidence = _evidence()
    plan, ledger, resume, identity = evidence
    inventory = _inventory(evidence)
    receipt = subject.build_act_as_receipt(
        identity_plan=identity,
        inventory_receipt=inventory,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        test_iam_permissions_observations=_act_as_observations(identity),
        tested_at_utc="2027-01-15T08:01:00Z",
        provider_source=PROVIDER,
    )
    assert subject.validate_act_as_receipt(
        identity_plan=identity,
        inventory_receipt=inventory,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        value=receipt,
    ) == receipt
    assert receipt["selected_count"] == 8
    assert receipt["required_permission"] == "iam.serviceAccounts.actAs"
    assert receipt["every_selected_account_grants_exact_act_as"] is True


@pytest.mark.parametrize("mode", ["missing", "extra_permission", "get", "not_granted"])
def test_act_as_fails_closed_on_missing_extra_or_malformed(mode: str) -> None:
    plan, ledger, resume, identity = _evidence()
    inventory = _inventory((plan, ledger, resume, identity))
    rows = _act_as_observations(identity)
    if mode == "missing":
        rows.pop()
    elif mode == "extra_permission":
        rows[0]["granted_permissions"].append("iam.serviceAccounts.get")
    elif mode == "get":
        rows[0]["http_method"] = "GET"
    else:
        rows[0]["granted_permissions"] = []
    with pytest.raises((PermissionError, ValueError)):
        subject.build_act_as_receipt(
            identity_plan=identity,
            inventory_receipt=inventory,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            test_iam_permissions_observations=rows,
            tested_at_utc="2027-01-15T08:01:00Z",
            provider_source=PROVIDER,
        )


def test_project_iam_scan_is_self_contained_and_proves_zero_pool_roles() -> None:
    evidence = _evidence()
    plan, ledger, resume, identity = evidence
    inventory = _inventory(evidence)
    receipt = subject.build_project_iam_scan_receipt(
        identity_plan=identity,
        inventory_receipt=inventory,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        project_iam_policy=_empty_project_policy(),
        observed_at_utc="2027-01-15T08:02:00Z",
        provider_source=PROVIDER,
    )
    assert subject.validate_project_iam_scan_receipt(
        identity_plan=identity,
        inventory_receipt=inventory,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        value=receipt,
    ) == receipt
    assert receipt["pool_role_membership_count"] == 0
    assert receipt["all_pool_accounts_have_zero_project_roles"] is True
    assert receipt["bucket_iam_outside_scope"] is True
    assert all(row["role_memberships"] == [] for row in receipt["rows"])


def test_project_iam_scan_normalizes_live_policy_shape_without_audit_configs() -> None:
    evidence = _evidence()
    plan, ledger, resume, identity = evidence
    inventory = _inventory(evidence)
    policy = _empty_project_policy()
    del policy["auditConfigs"]

    receipt = subject.build_project_iam_scan_receipt(
        identity_plan=identity,
        inventory_receipt=inventory,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        project_iam_policy=policy,
        observed_at_utc="2027-01-15T08:02:00Z",
        provider_source=PROVIDER,
    )

    assert receipt["scanned_policy"] == {**policy, "auditConfigs": []}
    assert subject.validate_project_iam_scan_receipt(
        identity_plan=identity,
        inventory_receipt=inventory,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        value=receipt,
    ) == receipt


@pytest.mark.parametrize(
    "mutate",
    [
        lambda policy: policy.update(unexpected=True),
        lambda policy: policy.pop("version"),
        lambda policy: policy.update(bindings=[{"role": "roles/viewer"}]),
        lambda policy: policy.update(bindings=[{"role": "roles/viewer", "members": [], "extra": True}]),
    ],
)
def test_project_iam_scan_live_shape_still_rejects_unknown_or_malformed_entries(
    mutate,
) -> None:
    evidence = _evidence()
    plan, ledger, resume, identity = evidence
    inventory = _inventory(evidence)
    policy = _empty_project_policy()
    del policy["auditConfigs"]
    mutate(policy)

    with pytest.raises(ValueError, match="fields changed|malformed"):
        subject.build_project_iam_scan_receipt(
            identity_plan=identity,
            inventory_receipt=inventory,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            project_iam_policy=policy,
            observed_at_utc="2027-01-15T08:02:00Z",
            provider_source=PROVIDER,
        )


@pytest.mark.parametrize("deleted", [False, True])
def test_project_iam_scan_rejects_any_pool_account_membership(deleted: bool) -> None:
    evidence = _evidence()
    plan, ledger, resume, identity = evidence
    inventory = _inventory(evidence)
    policy = _empty_project_policy()
    principal = f"serviceAccount:{subject.ACCOUNT_EMAILS[0]}"
    if deleted:
        principal = f"deleted:{principal}?uid=100000000000000000000"
    policy["bindings"].append(
        {"role": "roles/viewer", "members": [principal]}
    )
    with pytest.raises(PermissionError, match="project-level role"):
        subject.build_project_iam_scan_receipt(
            identity_plan=identity,
            inventory_receipt=inventory,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            project_iam_policy=policy,
            observed_at_utc="2027-01-15T08:02:00Z",
            provider_source=PROVIDER,
        )


def _setup_observations(plan: dict) -> tuple[list[dict], list[dict]]:
    absence = [
        {
            "account_id": row["account_id"],
            "email": row["email"],
            "name": row["name"],
            "http_method": "GET",
            "http_status": 404,
            "exists": False,
        }
        for row in plan["accounts"]
    ]
    creation = [
        {
            "account_id": row["account_id"],
            "email": row["email"],
            "name": row["name"],
            "project_id": subject.PROJECT,
            "unique_id": str(200_000_000_000_000_000_000 + index),
            "disabled": False,
            "http_method": "POST",
            "http_status": 200,
            "created": True,
            "already_existed": False,
        }
        for index, row in enumerate(plan["accounts"])
    ]
    return absence, creation


def test_optional_setup_surface_is_one_time_create_only_and_pure() -> None:
    plan = subject.build_create_only_setup_plan(
        setup_nonce="fixed-worker-pool-setup-0001",
        issued_at_utc="2027-01-15T06:00:00Z",
    )
    assert subject.validate_create_only_setup_plan(plan) == plan
    absence, creation = _setup_observations(plan)
    receipt = subject.build_create_only_setup_receipt(
        setup_plan=plan,
        absence_observations=absence,
        creation_observations=creation,
        completed_at_utc="2027-01-15T06:05:00Z",
        provider_source=PROVIDER,
    )
    assert subject.validate_create_only_setup_receipt(
        setup_plan=plan, value=receipt
    ) == receipt
    assert plan["create_only"] is True
    assert plan["run_specific"] is False
    assert plan["cloud_mutated"] is False
    assert receipt["created_account_count"] == 8
    assert receipt["existing_account_accepted"] is False
    assert receipt["create_only_contract_complete"] is True
    assert receipt["provider_mutation_observed"] is True
    assert receipt["module_executed_cloud_mutation"] is False


def test_setup_rejects_existing_account_or_non_post_create() -> None:
    plan = subject.build_create_only_setup_plan(
        setup_nonce="fixed-worker-pool-setup-0002",
        issued_at_utc="2027-01-15T06:00:00Z",
    )
    absence, creation = _setup_observations(plan)
    bad_absence = copy.deepcopy(absence)
    bad_absence[0].update(http_status=200, exists=True)
    with pytest.raises(FileExistsError):
        subject.build_create_only_setup_receipt(
            setup_plan=plan,
            absence_observations=bad_absence,
            creation_observations=creation,
            completed_at_utc="2027-01-15T06:05:00Z",
            provider_source=PROVIDER,
        )
    bad_creation = copy.deepcopy(creation)
    bad_creation[0]["http_method"] = "PUT"
    with pytest.raises(FileExistsError):
        subject.build_create_only_setup_receipt(
            setup_plan=plan,
            absence_observations=absence,
            creation_observations=bad_creation,
            completed_at_utc="2027-01-15T06:05:00Z",
            provider_source=PROVIDER,
        )


def test_resealed_mapping_or_receipt_claim_cannot_bypass_revalidation() -> None:
    plan, ledger, resume, identity = _evidence()
    forged = copy.deepcopy(identity)
    forged["selected_workers"][0]["service_account_email"] = subject.ACCOUNT_EMAILS[1]
    _reseal(forged, "plan_sha256")
    with pytest.raises(ValueError, match="deterministic fixed-pool mapping"):
        subject.validate_worker_identity_plan(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            value=forged,
        )

    inventory = _inventory((plan, ledger, resume, identity))
    forged_inventory = copy.deepcopy(inventory)
    forged_inventory["all_enabled"] = False
    _reseal(forged_inventory, "receipt_sha256")
    with pytest.raises(ValueError, match="contract changed"):
        subject.validate_inventory_receipt(
            identity_plan=identity,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            value=forged_inventory,
        )
