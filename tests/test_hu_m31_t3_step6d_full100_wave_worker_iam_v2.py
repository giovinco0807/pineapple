from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave
from ofc_regular import hu_m31_t3_step6d_full100_wave_package_v2 as package_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as subject


ISSUED = 1_800_000_000
CONTENT_PAYLOAD_SHA = "4" * 64
OUTER_MANIFEST_SHA = "5" * 64
IMMUTABLE_CONTENT_PREFIX = (
    f"{package_v2.CONTENT_PREFIX_ROOT}/{CONTENT_PAYLOAD_SHA}"
)


def _content_kwargs() -> dict[str, str]:
    return {
        "immutable_content_prefix": IMMUTABLE_CONTENT_PREFIX,
        "content_payload_sha256": CONTENT_PAYLOAD_SHA,
        "outer_manifest_sha256": OUTER_MANIFEST_SHA,
    }


def _wave_evidence() -> tuple[dict, dict, dict]:
    plan = wave.build_wave_plan(
        run_name="regular-hu-m31-c02-f100wv2-workeriam-001",
        identity_salt="1234567890abcdef1234567890abcdef",
        package_sha256="2" * 64,
        image_digest="sha256:" + "3" * 64,
    )
    transition = wave.build_observed_transition(
        plan,
        project_id="ofc-project-123",
        zone="asia-northeast1-b",
        observed_at_utc="2026-07-22T00:00:00Z",
        previous_transition_digest=None,
        attempt_history=wave.empty_attempt_history(plan),
    )
    ledger = wave.build_attempt_ledger(plan, transitions=[transition])
    resume = wave.build_resume_plan(plan, attempt_ledger=ledger)
    return plan, ledger, resume


@pytest.fixture(scope="module")
def evidence() -> tuple[dict, dict, dict, dict]:
    plan, ledger, resume = _wave_evidence()
    iam = subject.build_worker_iam_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        wave_index=0,
        **_content_kwargs(),
        issued_at_unix_seconds=ISSUED,
    )
    return plan, ledger, resume, iam


def _base_policy() -> dict[str, Any]:
    return {
        "kind": "storage#policy",
        "resourceId": f"projects/_/buckets/{subject.BUCKET}",
        "version": 3,
        "etag": "BwWInitialEtag==",
        "bindings": [
            {
                "role": "roles/storage.legacyBucketReader",
                "members": ["projectViewer:unrelated-project"],
            },
            {
                "role": "roles/storage.objectViewer",
                "members": ["user:unrelated@example.com"],
                "condition": {
                    "title": "unrelated-condition",
                    "expression": (
                        'resource.name.startsWith('
                        '"projects/_/buckets/other/objects/prefix/")'
                    ),
                },
            },
        ],
        "auditConfigs": [],
    }


class FakeBackend:
    def __init__(self, *, mode: str = "normal") -> None:
        self.policy = _base_policy()
        self.mode = mode
        self.set_count = 0
        self.set_inputs: list[dict[str, Any]] = []
        self.target_principal: str | None = None

    def get_bucket_policy(self) -> Mapping[str, Any]:
        return copy.deepcopy(self.policy)

    def set_bucket_policy(self, *, policy: Mapping[str, Any]) -> Mapping[str, Any]:
        supplied = copy.deepcopy(dict(policy))
        self.set_count += 1
        self.set_inputs.append(supplied)
        if self.mode == "etag_race":
            self.policy["etag"] = "BwWRacingWriter=="
            raise subject.WorkerIamCasError("fake ETag precondition failed")
        if supplied["etag"] != self.policy["etag"]:
            raise subject.WorkerIamCasError("fake stale ETag")
        installing = len(supplied["bindings"]) > len(self.policy["bindings"])
        new_policy = supplied
        if self.mode == "partial_add" and installing:
            new_policy["bindings"] = new_policy["bindings"][:-1]
        elif self.mode == "extra_add" and installing:
            assert self.target_principal is not None
            new_policy["bindings"].append(
                {
                    "role": "roles/storage.objectAdmin",
                    "members": [self.target_principal],
                }
            )
        elif self.mode == "partial_remove" and not installing:
            # Keep one binding which the requested cleanup removed.
            removed = [
                row
                for row in self.policy["bindings"]
                if row not in supplied["bindings"]
            ]
            assert removed
            new_policy["bindings"].append(copy.deepcopy(removed[0]))
        new_policy["etag"] = f"BwWAfterSet{self.set_count}=="
        self.policy = new_policy
        return copy.deepcopy(self.policy)


def _prepare(
    evidence: tuple[dict, dict, dict, dict], backend: FakeBackend
) -> dict:
    plan, ledger, resume, iam = evidence
    return subject.prepare_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        backend=backend,
    )


def _install(
    evidence: tuple[dict, dict, dict, dict], backend: FakeBackend
) -> tuple[dict, dict]:
    plan, ledger, resume, iam = evidence
    prepare = _prepare(evidence, backend)
    install = subject.install_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        backend=backend,
    )
    return prepare, install


def _readback(
    evidence: tuple[dict, dict, dict, dict], backend: FakeBackend
) -> tuple[dict, dict, dict]:
    plan, ledger, resume, iam = evidence
    prepare, install = _install(evidence, backend)
    readback = subject.readback_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        install_receipt=install,
        backend=backend,
    )
    return prepare, install, readback


def _reseal(value: dict, digest_field: str) -> None:
    value[digest_field] = subject.canonical_sha256(
        {key: item for key, item in value.items() if key != digest_field}
    )


def test_plan_binds_exact_wave_jobs_roles_vms_accounts_and_prefixes(
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    plan, ledger, resume, iam = evidence
    assert subject.validate_worker_iam_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        value=iam,
    ) == iam
    assert iam["wave_plan_schedule_sha256"] == plan["schedule_sha256"]
    assert iam["wave_index"] == 0
    assert iam["attempt_ledger_sha256"] == ledger["ledger_sha256"]
    assert iam["resume_plan_sha256"] == resume["resume_sha256"]
    assert iam["immutable_content_prefix"] == IMMUTABLE_CONTENT_PREFIX
    assert subject.IMMUTABLE_CONTENT_PREFIX_ROOT == package_v2.CONTENT_PREFIX_ROOT
    assert iam["content_payload_sha256"] == CONTENT_PAYLOAD_SHA
    assert iam["outer_manifest_sha256"] == OUTER_MANIFEST_SHA
    assert iam["worker_count"] == 8
    assert iam["exact_binding_count"] == 16
    assert iam["exact_binding_count"] <= subject.MAX_BINDINGS_PER_WAVE
    assert len({row["service_account"] for row in iam["workers"]}) == 8
    assert [row["job_id"] for row in iam["workers"]] == [
        row["job_id"] for row in resume["selected_attempts"]
    ]
    for worker, selected in zip(
        iam["workers"], resume["selected_attempts"], strict=True
    ):
        assert worker["source_role"] == selected["source_role"]
        assert worker["attempt_id"] == selected["attempt_id"]
        assert worker["vm_instance_id"] == selected["instance_id"]
        assert worker["creator_prefix"] == selected["artifact_prefix"] + "/"
        assert worker["reader_control_prefix"].endswith("/control/")
        assert worker["reader_immutable_content_prefix"] == (
            IMMUTABLE_CONTENT_PREFIX + "/"
        )
    for index, worker in enumerate(iam["workers"]):
        reader = iam["expected_bindings"][2 * index]
        creator = iam["expected_bindings"][2 * index + 1]
        reader_expression = reader["condition"]["expression"]
        creator_expression = creator["condition"]["expression"]
        assert reader["role"] == subject.READER_ROLE
        assert reader_expression.count("resource.name.startsWith") == 3
        assert worker["reader_control_prefix"] in reader_expression
        assert worker["reader_immutable_content_prefix"] in reader_expression
        assert worker["creator_prefix"] in reader_expression
        assert creator["role"] == subject.CREATOR_ROLE
        assert creator_expression.count("resource.name.startsWith") == 1
        assert worker["creator_prefix"] in creator_expression
        assert worker["reader_control_prefix"] not in creator_expression
        assert worker["reader_immutable_content_prefix"] not in creator_expression


def test_worker_policy_allows_only_own_attempt_read_and_create(
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    _, _, _, iam = evidence
    first = iam["workers"][0]
    other = iam["workers"][1]
    reader, creator = iam["expected_bindings"][:2]
    principal = first["principal"]
    own_object = (
        f"projects/_/buckets/{subject.BUCKET}/objects/"
        f"{first['creator_prefix']}result.json"
    )
    other_object = (
        f"projects/_/buckets/{subject.BUCKET}/objects/"
        f"{other['creator_prefix']}result.json"
    )

    def prefix_allowed(binding: Mapping[str, Any], object_name: str) -> bool:
        expression = binding["condition"]["expression"]
        prefixes = [
            part.split('"', 1)[0]
            for part in expression.split('resource.name.startsWith("')[1:]
        ]
        return principal in binding["members"] and any(
            object_name.startswith(prefix) for prefix in prefixes
        )

    assert reader["role"] == subject.READER_ROLE
    assert creator["role"] == subject.CREATOR_ROLE
    assert prefix_allowed(reader, own_object)
    assert not prefix_allowed(reader, other_object)
    assert prefix_allowed(creator, own_object)
    assert not prefix_allowed(creator, other_object)

    # The selected predefined roles intentionally expose get for the reader
    # and create-only for the creator; neither binding grants mutation/admin or
    # list surfaces used for overwrite, delete, or bucket-wide enumeration.
    allowed = {
        "storage.objects.get": prefix_allowed(reader, own_object),
        "storage.objects.create": prefix_allowed(creator, own_object),
        "storage.objects.update": False,
        "storage.objects.delete": False,
        "storage.objects.list": False,
    }
    assert allowed == {
        "storage.objects.get": True,
        "storage.objects.create": True,
        "storage.objects.update": False,
        "storage.objects.delete": False,
        "storage.objects.list": False,
    }
    assert all(
        binding["members"] == [iam["workers"][index // 2]["principal"]]
        for index, binding in enumerate(iam["expected_bindings"])
    )
    assert iam["cloud_mutated"] is False
    assert iam["current_profile_changed"] is False


def test_wrong_wave_or_resealed_schedule_ledger_vm_tamper_fails_closed(
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    plan, ledger, resume, iam = evidence
    with pytest.raises(ValueError, match="wave index"):
        subject.build_worker_iam_plan(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            wave_index=1,
            **_content_kwargs(),
            issued_at_unix_seconds=ISSUED,
        )
    for key, replacement in (
        ("wave_plan_schedule_sha256", "a" * 64),
        ("attempt_ledger_sha256", "b" * 64),
    ):
        changed = copy.deepcopy(iam)
        changed[key] = replacement
        _reseal(changed, "plan_sha256")
        with pytest.raises(ValueError):
            subject.validate_worker_iam_plan(
                wave_plan=plan,
                attempt_ledger=ledger,
                resume_plan=resume,
                **_content_kwargs(),
                value=changed,
            )
    changed = copy.deepcopy(iam)
    changed["workers"][0]["vm_instance_id"] = changed["workers"][1][
        "vm_instance_id"
    ]
    _reseal(changed, "plan_sha256")
    with pytest.raises(ValueError, match="derives"):
        subject.validate_worker_iam_plan(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            **_content_kwargs(),
            value=changed,
        )


def test_prepare_requires_exact_absence_and_records_only_etag_hash(
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    backend = FakeBackend()
    receipt = _prepare(evidence, backend)
    assert receipt["targeted_binding_count"] == 0
    assert receipt["set_attempt_count"] == 0
    assert "BwWInitialEtag" not in subject.canonical_bytes(receipt).decode("ascii")
    assert backend.set_count == 0

    backend = FakeBackend()
    backend.policy["bindings"].append(
        {
            "role": "roles/storage.objectViewer",
            "members": [evidence[3]["workers"][0]["principal"]],
        }
    )
    with pytest.raises(ValueError, match="stale, duplicate, or drifted"):
        _prepare(evidence, backend)
    assert backend.set_count == 0


def test_content_manifest_prefix_and_reader_scope_fail_closed(
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    plan, ledger, resume, iam = evidence
    wrong_content = "6" * 64
    with pytest.raises(ValueError, match="safety boundary"):
        subject.validate_worker_iam_plan(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            immutable_content_prefix=(
                f"{subject.IMMUTABLE_CONTENT_PREFIX_ROOT}/{wrong_content}"
            ),
            content_payload_sha256=wrong_content,
            outer_manifest_sha256=OUTER_MANIFEST_SHA,
            value=iam,
        )
    with pytest.raises(ValueError, match="safety boundary"):
        subject.validate_worker_iam_plan(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            immutable_content_prefix=IMMUTABLE_CONTENT_PREFIX,
            content_payload_sha256=CONTENT_PAYLOAD_SHA,
            outer_manifest_sha256="7" * 64,
            value=iam,
        )
    with pytest.raises(ValueError, match="exact content-addressed"):
        subject.build_worker_iam_plan(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            wave_index=0,
            immutable_content_prefix=subject.IMMUTABLE_CONTENT_PREFIX_ROOT,
            content_payload_sha256=CONTENT_PAYLOAD_SHA,
            outer_manifest_sha256=OUTER_MANIFEST_SHA,
            issued_at_unix_seconds=ISSUED,
        )

    missing_content = copy.deepcopy(iam)
    reader = missing_content["expected_bindings"][0]["condition"]
    reader["expression"] = (
        'resource.name.startsWith("projects/_/buckets/'
        f'{subject.BUCKET}/objects/{iam["reader_control_prefix"]}") '
        f'&& request.time < timestamp("{iam["expires_at_utc"]}")'
    )
    _reseal(missing_content, "plan_sha256")
    with pytest.raises(ValueError, match="derives"):
        subject.validate_worker_iam_plan(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            **_content_kwargs(),
            value=missing_content,
        )

    expanded_prefix = copy.deepcopy(iam)
    expanded_reader = expanded_prefix["expected_bindings"][0]["condition"]
    expanded_reader["expression"] = expanded_reader["expression"].replace(
        IMMUTABLE_CONTENT_PREFIX + "/",
        subject.IMMUTABLE_CONTENT_PREFIX_ROOT + "/",
    )
    _reseal(expanded_prefix, "plan_sha256")
    with pytest.raises(ValueError, match="derives"):
        subject.validate_worker_iam_plan(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            **_content_kwargs(),
            value=expanded_prefix,
        )


def test_same_service_account_for_two_jobs_is_rejected(
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    plan, ledger, resume, _iam = evidence
    accounts = subject.default_service_accounts(plan, resume)
    jobs = list(accounts)
    accounts[jobs[1]] = accounts[jobs[0]]
    with pytest.raises(ValueError, match="duplicated"):
        subject.build_worker_iam_plan(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            wave_index=0,
            **_content_kwargs(),
            issued_at_unix_seconds=ISSUED,
            service_accounts_by_job=accounts,
        )

def test_successful_single_cas_install_readback_cleanup_preserves_unrelated(
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    plan, ledger, resume, iam = evidence
    backend = FakeBackend()
    unrelated = copy.deepcopy(backend.policy["bindings"])
    prepare, install, readback = _readback(evidence, backend)
    assert backend.set_count == 1
    assert install["installed_binding_count"] == 16
    assert readback["observed_binding_count"] == 16
    assert backend.policy["bindings"][: len(unrelated)] == unrelated
    cleanup = subject.cleanup_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        install_receipt=install,
        readback_receipt=readback,
        backend=backend,
    )
    assert backend.set_count == 2
    assert backend.policy["bindings"] == unrelated
    assert cleanup["removed_binding_count"] == 16
    assert cleanup["remaining_targeted_binding_count"] == 0
    assert cleanup["post_cleanup_absence_readback"] is True
    for receipt in (prepare, install, readback, cleanup):
        assert receipt["immutable_content_prefix"] == IMMUTABLE_CONTENT_PREFIX
        assert receipt["content_payload_sha256"] == CONTENT_PAYLOAD_SHA
        assert receipt["outer_manifest_sha256"] == OUTER_MANIFEST_SHA
    assert subject.validate_cleanup_receipt(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        install_receipt=install,
        readback_receipt=readback,
        value=cleanup,
    ) == cleanup


def test_public_receipt_validators_bind_outer_content_without_backend(
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    plan, ledger, resume, iam = evidence
    prepare, install, readback = _readback(evidence, FakeBackend())
    common = {
        "iam_plan": iam,
        "wave_plan": plan,
        "attempt_ledger": ledger,
        "resume_plan": resume,
        **_content_kwargs(),
    }
    assert subject.validate_prepare_receipt(
        **common, value=prepare
    ) == prepare
    assert subject.validate_install_receipt(
        **common, prepare_receipt=prepare, value=install
    ) == install
    assert subject.validate_readback_receipt(
        **common,
        prepare_receipt=prepare,
        install_receipt=install,
        value=readback,
    ) == readback

    wrong_manifest = {**common, "outer_manifest_sha256": "7" * 64}
    with pytest.raises(ValueError, match="safety boundary"):
        subject.validate_prepare_receipt(**wrong_manifest, value=prepare)
    wrong_content = "6" * 64
    wrong_payload = {
        **common,
        "content_payload_sha256": wrong_content,
        "immutable_content_prefix": (
            f"{subject.IMMUTABLE_CONTENT_PREFIX_ROOT}/{wrong_content}"
        ),
    }
    with pytest.raises(ValueError, match="safety boundary"):
        subject.validate_install_receipt(
            **wrong_payload, prepare_receipt=prepare, value=install
        )
    wrong_prefix = {
        **common,
        "immutable_content_prefix": subject.IMMUTABLE_CONTENT_PREFIX_ROOT,
    }
    with pytest.raises(ValueError, match="exact content-addressed"):
        subject.validate_readback_receipt(
            **wrong_prefix,
            prepare_receipt=prepare,
            install_receipt=install,
            value=readback,
        )


def test_cleanup_recovery_receipt_attributes_only_its_get_only_probe(
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    plan, ledger, resume, iam = evidence
    backend = FakeBackend()
    prepare, install, readback = _readback(evidence, backend)
    subject.cleanup_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        install_receipt=install,
        readback_receipt=readback,
        backend=backend,
    )
    before = backend.set_count
    recovered = subject.reconcile_cleanup_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        install_receipt=install,
        readback_receipt=readback,
        backend=backend,
    )
    assert backend.set_count == before
    assert recovered["schema"].endswith("_cleanup_receipt_v3")
    assert recovered["recovered_after_outcome_ambiguity"] is True
    assert recovered["removed_binding_count"] == 0
    assert recovered["set_attempt_count"] == 0
    assert recovered["cloud_mutation_performed"] is False
    assert recovered["source_mutation_outcome"] == "unknown"


def test_pre_set_etag_drift_and_in_set_etag_race_never_retry(
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    plan, ledger, resume, iam = evidence
    backend = FakeBackend()
    prepare = _prepare(evidence, backend)
    backend.policy["etag"] = "BwWDriftBeforeSet=="
    with pytest.raises(ValueError, match="ETag"):
        subject.install_worker_iam(
            iam_plan=iam,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            **_content_kwargs(),
            prepare_receipt=prepare,
            backend=backend,
        )
    assert backend.set_count == 0

    backend = FakeBackend(mode="etag_race")
    prepare = _prepare(evidence, backend)
    with pytest.raises(subject.WorkerIamCasError, match="ETag"):
        subject.install_worker_iam(
            iam_plan=iam,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            **_content_kwargs(),
            prepare_receipt=prepare,
            backend=backend,
        )
    assert backend.set_count == 1


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("partial_add", "partial add"),
        ("extra_add", "extra or drift"),
    ],
)
def test_partial_add_and_extra_binding_fail_closed_after_one_set(
    evidence: tuple[dict, dict, dict, dict], mode: str, message: str
) -> None:
    plan, ledger, resume, iam = evidence
    backend = FakeBackend(mode=mode)
    backend.target_principal = iam["workers"][0]["principal"]
    prepare = _prepare(evidence, backend)
    with pytest.raises(ValueError, match=message):
        subject.install_worker_iam(
            iam_plan=iam,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            **_content_kwargs(),
            prepare_receipt=prepare,
            backend=backend,
        )
    assert backend.set_count == 1


def test_cleanup_incomplete_and_wrong_wave_receipt_fail_closed(
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    plan, ledger, resume, iam = evidence
    backend = FakeBackend()
    prepare, install, readback = _readback(evidence, backend)
    backend.mode = "partial_remove"
    with pytest.raises(ValueError, match="absence is incomplete"):
        subject.cleanup_worker_iam(
            iam_plan=iam,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            **_content_kwargs(),
            prepare_receipt=prepare,
            install_receipt=install,
            readback_receipt=readback,
            backend=backend,
        )
    assert backend.set_count == 2

    backend = FakeBackend()
    prepare, install, readback = _readback(evidence, backend)
    wrong = copy.deepcopy(readback)
    wrong["wave_index"] = 1
    _reseal(wrong, "receipt_sha256")
    with pytest.raises(ValueError, match="wrong wave"):
        subject.cleanup_worker_iam(
            iam_plan=iam,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            **_content_kwargs(),
            prepare_receipt=prepare,
            install_receipt=install,
            readback_receipt=wrong,
            backend=backend,
        )
    assert backend.set_count == 1


def test_plan_and_receipts_are_create_only(tmp_path: Path, evidence: tuple) -> None:
    target = tmp_path / "worker_iam_plan.json"
    subject.write_json_once(target, evidence[3])
    assert target.read_bytes() == subject.canonical_bytes(evidence[3])
    with pytest.raises(FileExistsError, match="immutable"):
        subject.write_json_once(target, evidence[3])
