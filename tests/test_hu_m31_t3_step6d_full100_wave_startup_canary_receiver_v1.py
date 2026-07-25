from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_controller_v2 as controller_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_launch_bundle_v2 as launch_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_package_v2 as package_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_production_cleanup_orchestrator_v2 as cleanup_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_production_receiver_v2 as production_receiver_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_result_receiver_v2 as result_receiver_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_startup_canary_receiver_v1 as subject
from ofc_regular import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as worker_iam_v2


RUN = "regular-hu-m31-c02-f100wv2-canaryrx"
PROJECT = "ofc-solver-485418"
ZONE = "asia-northeast1-b"


def _sha(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _initial() -> tuple[dict, dict, dict]:
    plan = wave_v2.build_wave_plan(
        run_name=RUN,
        identity_salt="1234567890abcdef1234567890abcdef",
        package_sha256="2" * 64,
        image_digest="sha256:" + "3" * 64,
        execution_scope=wave_v2.STARTUP_CANARY_SCOPE,
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


def _binding(name: str) -> dict:
    return {
        "object_name": f"content/{name}",
        "sha256": _sha(name),
        "bytes": 100 + len(name),
    }


def _bootstrap(plan: dict, ledger: dict, resume: dict) -> dict:
    selected = resume["selected_attempts"][0]
    principal = worker_iam_v2.default_service_accounts(plan, resume)[
        selected["job_id"]
    ]
    value = {
        "schema": package_v2.JOB_BOOTSTRAP_SCHEMA,
        "status": "single_job_bootstrap_bound_to_prelaunch_authorization",
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_sha256": resume["resume_sha256"],
        "observed_transition_digest": resume["observed_transition_digest"],
        "wave_index": resume["resume_wave_index"],
        "job_id": selected["job_id"],
        "source_role": selected["source_role"],
        "attempt_id": selected["attempt_id"],
        "instance_name": selected["instance_id"],
        "artifact_prefix": selected["artifact_prefix"],
        "bucket": "ofc-test-bucket",
        "content_prefix": "content/full100-wave-v2",
        "outer_manifest_sha256": _sha("outer"),
        "content_payload_sha256": _sha("content"),
        "scientific_source": _binding("source.zip"),
        "scientific_manifest": _binding("science.json"),
        "wheelhouse": _binding("wheelhouse.zip"),
        "wheelhouse_manifest": _binding("wheelhouse.json"),
        "startup": {
            "object_name": "content/startup.sh",
            "sha256": launch_v2.EXPECTED_STARTUP_SHA256,
            "bytes": 12345,
        },
        "wave_plan": _binding("wave.json"),
        "job_manifest": _binding("candidate-shard-00.json"),
        "prelaunch_authorization_sha256": _sha("authorization"),
        "worker_principal": principal,
        "one_vm_one_job_one_role": True,
        "additional_create_authorized": False,
        "hidden_truth_exposed": False,
    }
    value["bootstrap_sha256"] = package_v2.canonical_sha256(value)
    return value


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[dict, bytes]] = {}
        self.next_generation = 1
        self.create_count = 0

    def put(self, path: str, raw: bytes) -> None:
        self.objects[path] = (
            {
                "path": path,
                "generation": self.next_generation,
                "bytes": len(raw),
                "sha256": _sha(raw),
            },
            raw,
        )
        self.next_generation += 1

    def list_prefix(self, *, prefix: str):
        return [
            deepcopy(record)
            for path, (record, _) in sorted(self.objects.items())
            if path.startswith(prefix)
        ]

    def read_bytes(self, *, path: str, generation: int) -> bytes:
        record, raw = self.objects[path]
        if generation != record["generation"]:
            raise ValueError("wrong generation")
        return raw

    def read_current(self, *, path: str, allow_missing: bool = False):
        found = self.objects.get(path)
        if found is None:
            if allow_missing:
                return None
            raise ValueError("missing")
        return deepcopy(found[0]), found[1]

    def create_only(self, *, path: str, data: bytes):
        del path, data
        self.create_count += 1
        raise AssertionError("startup canary receiver must not create objects")


def _publish(
    plan: dict,
    ledger: dict,
    resume: dict,
    bootstrap: dict,
    store: FakeStore,
    *,
    hidden: bool = False,
) -> None:
    selected = resume["selected_attempts"][0]
    work = next(
        row["work_hand_indices"]
        for row in plan["full100_plan"]["jobs"]
        if row["job_id"] == subject.CANARY_JOB_ID
    )
    artifacts = []
    roots = []
    for hand in work:
        root = {
            "schema": "test-root-v1",
            "hand_index": hand,
        }
        source = {
            "schema": "test-source-v1",
            "job_id": subject.CANARY_JOB_ID,
            "source_role": subject.CANARY_SOURCE_ROLE,
            "hand_index": hand,
        }
        if hidden and hand == work[0]:
            source["opponent_private_discards"] = ["As"]
        for relative, value in (
            (f"roots/hand_{hand:03d}.json", root),
            (f"hands/candidate/hand_{hand:03d}.json", source),
        ):
            raw = json.dumps(
                value, sort_keys=True, separators=(",", ":")
            ).encode("ascii")
            store.put(f"{selected['artifact_prefix']}/{relative}", raw)
            artifacts.append(
                {"path": relative, "sha256": _sha(raw), "bytes": len(raw)}
            )
            if relative.startswith("roots/"):
                roots.append({"hand_index": hand, "sha256": _sha(raw)})
    root_digest = wave_v2.canonical_sha256(roots)
    done = {
        "schema": result_receiver_v2.DONE_SCHEMA,
        "status": "complete_validated_single_job_attempt",
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_sha256": resume["resume_sha256"],
        "observed_transition_digest": resume["observed_transition_digest"],
        "wave_index": resume["resume_wave_index"],
        "job_id": subject.CANARY_JOB_ID,
        "source_role": subject.CANARY_SOURCE_ROLE,
        "attempt_id": subject.CANARY_ATTEMPT_ID,
        "package_sha256": plan["runtime_binding"]["package_sha256"],
        "image_digest": plan["runtime_binding"]["image_digest"],
        "binary_sha256": plan["runtime_binding"]["binary_sha256_by_role"][
            "candidate"
        ],
        "allocation_digest": plan["runtime_binding"]["allocation_digest"],
        "run_contract_digest": plan["run_contract_digest"],
        "root_digest": root_digest,
        "done_identity_sha256": wave_v2.expected_done_identity_sha256(
            plan,
            job_id=subject.CANARY_JOB_ID,
            attempt_id=subject.CANARY_ATTEMPT_ID,
            root_digest=root_digest,
        ),
        "content_payload_sha256": bootstrap["content_payload_sha256"],
        "outer_manifest_sha256": bootstrap["outer_manifest_sha256"],
        "prelaunch_authorization_sha256": bootstrap[
            "prelaunch_authorization_sha256"
        ],
        "worker_principal": bootstrap["worker_principal"],
        "work_hand_indices": list(work),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "runner_done_sha256": _sha("runner-done"),
        "metadata_hidden_truth_exposed": False,
        "opponent_private_discards_used": False,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    raw = json.dumps(
        done,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    store.put(f"{selected['artifact_prefix']}/DONE.json", raw)


def _proof(resume: dict) -> dict:
    selected = resume["selected_attempts"][0]
    return {
        "proof_sha256": "9" * 64,
        "all_owned_instances_absent": True,
        "all_owned_boot_disks_absent": True,
        "worker_iam_bindings_absent": True,
        "additional_create_authorized": False,
        "current_profile_changed": False,
        "selected_instance_mapping": [
            {
                **selected,
                "exact_instance_created": True,
                "final_instance_absent": True,
                "final_boot_disk_absent": True,
            }
        ],
    }


def _receive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    hidden: bool = False,
):
    plan, ledger, resume = _initial()
    bootstrap = _bootstrap(plan, ledger, resume)
    store = FakeStore()
    _publish(plan, ledger, resume, bootstrap, store, hidden=hidden)
    profile = tmp_path / "ai_profiles.py"
    profile.write_bytes(b"frozen-current-profile\n")
    profile_sha = _sha(profile.read_bytes())
    monkeypatch.setattr(subject, "CURRENT_PROFILE_SHA256", profile_sha)
    monkeypatch.setattr(
        subject,
        "_validate_cleanup_manifest_and_replay",
        lambda **kwargs: ({"manifest_sha256": "8" * 64}, _proof(resume)),
    )
    receipt = subject.receive_startup_canary(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        job_bootstrap=bootstrap,
        cleanup_manifest={},
        journal_dir=tmp_path,
        profile_path=profile,
        store=store,
    )
    return receipt, store, plan, ledger, resume, bootstrap, profile


def test_exact_done_plus_twenty_is_diagnostic_only_and_never_mutates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt, store, *_ = _receive(monkeypatch, tmp_path)
    assert receipt["status"] == subject.RECEIPT_STATUS
    assert receipt["data_object_count"] == 20
    assert receipt["job_id"] == "candidate-shard-00"
    assert receipt["attempt_id"] == "a00"
    assert receipt["execution_scope"] == wave_v2.STARTUP_CANARY_SCOPE
    assert receipt["diagnostic_only"] is True
    assert receipt["cleanup_complete"] is True
    assert store.create_count == 0
    for key in (
        "acceptance_create_authorized",
        "acceptance_create_performed",
        "attempt_ledger_transition_authorized",
        "attempt_ledger_transition_performed",
        "merge_authorized",
        "merge_evidence",
        "scientific_merge_eligible",
        "training_eligible",
        "performance_evidence",
        "performance_evaluation_eligible",
        "quality_evidence",
        "promotion_evidence",
        "provider_mutation_performed",
        "current_profile_changed",
    ):
        assert receipt[key] is False
    assert subject.validate_receipt(receipt) == receipt


def test_missing_extra_hidden_and_accepted_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, store, plan, ledger, resume, bootstrap, profile = _receive(
        monkeypatch, tmp_path
    )
    selected = resume["selected_attempts"][0]
    first_data = next(
        path for path in store.objects if not path.endswith("DONE.json")
    )
    store.objects.pop(first_data)
    with pytest.raises(ValueError, match="DONE plus twenty"):
        subject.receive_startup_canary(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            job_bootstrap=bootstrap,
            cleanup_manifest={},
            journal_dir=tmp_path,
            profile_path=profile,
            store=store,
        )

    store = FakeStore()
    _publish(plan, ledger, resume, bootstrap, store)
    store.put(f"{selected['artifact_prefix']}/unexpected.json", b"{}")
    with pytest.raises(ValueError, match="DONE plus twenty"):
        subject.receive_startup_canary(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            job_bootstrap=bootstrap,
            cleanup_manifest={},
            journal_dir=tmp_path,
            profile_path=profile,
            store=store,
        )

    store = FakeStore()
    _publish(plan, ledger, resume, bootstrap, store, hidden=True)
    with pytest.raises(ValueError, match="hidden truth"):
        subject.receive_startup_canary(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            job_bootstrap=bootstrap,
            cleanup_manifest={},
            journal_dir=tmp_path,
            profile_path=profile,
            store=store,
        )

    store = FakeStore()
    _publish(plan, ledger, resume, bootstrap, store)
    acceptance = plan["artifact_contract"][
        "job_acceptance_path_template"
    ].format(job_id=subject.CANARY_JOB_ID)
    store.put(acceptance, b"{}")
    with pytest.raises(ValueError, match="must not observe ACCEPTED"):
        subject.receive_startup_canary(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            job_bootstrap=bootstrap,
            cleanup_manifest={},
            journal_dir=tmp_path,
            profile_path=profile,
            store=store,
        )


def test_profile_change_and_receipt_evidence_escalation_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt, _, *_ = _receive(monkeypatch, tmp_path)
    changed = deepcopy(receipt)
    changed["performance_evidence"] = True
    changed["receipt_sha256"] = wave_v2.canonical_sha256(
        {key: value for key, value in changed.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="safety contract"):
        subject.validate_receipt(changed)

    plan, ledger, resume = _initial()
    bootstrap = _bootstrap(plan, ledger, resume)
    store = FakeStore()
    _publish(plan, ledger, resume, bootstrap, store)
    profile = tmp_path / "changed.py"
    profile.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed before"):
        subject.receive_startup_canary(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            job_bootstrap=bootstrap,
            cleanup_manifest={},
            journal_dir=tmp_path,
            profile_path=profile,
            store=store,
        )


def _cleanup_manifest(
    tmp_path: Path, plan: dict, ledger: dict, resume: dict
) -> dict:
    journal = (tmp_path / "journal").resolve()
    journal.mkdir()
    payload = {
        "closeout_event_sha256": "1" * 64,
        "launch_bundle": {},
        "launch_validation": {},
        "startup_script_path": str(tmp_path / "startup.sh"),
        "gce_create_receipt": {},
        "gce_delete_receipt": {"receipt_sha256": "2" * 64},
        "worker_iam_cleanup_receipt": {"receipt_sha256": "2" * 64},
        "content_binding": {},
        "observed_at_utc": "2026-07-23T00:00:01Z",
    }
    request = production_receiver_v2.build_production_receive_request(
        execution_namespace=production_receiver_v2.execution_namespace(
            ledger, resume
        ),
        controller_journal_dir=journal,
        payload=payload,
    )
    context = {
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "wave_index": resume["resume_wave_index"],
    }
    lifecycle_core = {
        "schema": controller_v2.LIFECYCLE_SCHEMA,
        "status": "exact_owned_gce_lifecycle_absence_attested",
        "controller_context_sha256": controller_v2.canonical_sha256(context),
        **context,
        "launch_event_sha256": "2" * 64,
        "gce_create_receipt_sha256": "2" * 64,
        "actual_launch_receipt_sha256": None,
        "delete_event_sha256": "2" * 64,
        "gce_delete_receipt_sha256": "2" * 64,
        "absence_event_sha256": "2" * 64,
        "gce_absence_receipt_sha256": "2" * 64,
        "worker_iam_cleanup_event_sha256": "2" * 64,
        "worker_iam_cleanup_receipt_sha256": "2" * 64,
        "worker_iam_bindings_absent": True,
        "content_cleanup_event_sha256": None,
        "content_cleanup_receipt_sha256": None,
        "all_owned_instances_absent": True,
        "all_owned_boot_disks_absent": True,
        "additional_create_authorized": False,
        "attested_at_utc": "2026-07-23T00:00:01Z",
        "current_profile_changed": False,
    }
    lifecycle = {
        **lifecycle_core,
        "receipt_sha256": controller_v2.canonical_sha256(lifecycle_core),
    }
    core = {
        key: "2" * 64 for key in subject._CLEANUP_MANIFEST_FIELDS
    }
    core.update(
        {
            "schema": cleanup_v2.CLEANUP_MANIFEST_SCHEMA,
            "status": cleanup_v2.CLEANUP_MANIFEST_STATUS,
            "execution_namespace": production_receiver_v2.execution_namespace(
                ledger, resume
            ),
            "run_name": plan["run_name"],
            "execution_identity_sha256": plan["execution_identity_sha256"],
            "wave_index": resume["resume_wave_index"],
            "lifecycle_receipt": lifecycle,
            "receiver_request": request,
            "receiver_request_sha256": request["request_sha256"],
            "closeout_event_sha256": request["closeout_event_sha256"],
            "all_owned_instances_absent": True,
            "all_owned_boot_disks_absent": True,
            "worker_iam_bindings_absent": True,
            "shared_content_preserved_for_later_executions": True,
            "content_cleanup_event_sha256": None,
            "credentials_from_environment_only": True,
            "additional_create_authorized": False,
            "current_profile_changed": False,
        }
    )
    core.pop("manifest_sha256")
    return {**core, "manifest_sha256": cleanup_v2.canonical_sha256(core)}


def _production_proof(
    plan: dict, ledger: dict, resume: dict, manifest: dict
) -> dict:
    selected = resume["selected_attempts"][0]
    context = {
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "wave_index": resume["resume_wave_index"],
    }
    mapping = {
        **selected,
        "launch_receipt_sha256": "3" * 64,
        "exact_instance_created": True,
        "provider_instance_id": "instance-123",
        "provider_boot_disk_id": "disk-123",
        "gce_spec_sha256": "4" * 64,
        "gce_operation_id": "operation-123",
        "actual_launch_operation_id": None,
        "actual_launch_instance_status": None,
        "ownership_label": "owned",
        "final_instance_absent": True,
        "final_boot_disk_absent": True,
    }
    core = {
        "schema": controller_v2.LIFECYCLE_PROOF_SCHEMA,
        "status": "controller_journal_and_all_producer_receipts_revalidated",
        "controller_context_sha256": controller_v2.canonical_sha256(context),
        **context,
        "lifecycle_event_sha256": manifest["closeout_event_sha256"],
        "lifecycle_receipt_sha256": manifest["lifecycle_receipt"][
            "receipt_sha256"
        ],
        "lifecycle_attested_at_utc": manifest["receiver_request"][
            "observed_at_utc"
        ],
        "launch_event_sha256": manifest["launch_event_sha256"],
        "delete_event_sha256": manifest["delete_event_sha256"],
        "absence_event_sha256": manifest["absence_event_sha256"],
        "worker_iam_cleanup_event_sha256": manifest[
            "worker_iam_cleanup_event_sha256"
        ],
        "launch_bundle_sha256": "5" * 64,
        "gce_create_receipt": {
            "receipt_sha256": manifest["gce_create_receipt_sha256"]
        },
        "actual_launch_receipt": None,
        "gce_delete_receipt": {
            "receipt_sha256": manifest["gce_delete_receipt_sha256"]
        },
        "gce_absence_receipt": {
            "receipt_sha256": manifest["gce_absence_receipt_sha256"]
        },
        "worker_iam_cleanup_receipt": {
            "receipt_sha256": manifest["worker_iam_cleanup_receipt_sha256"]
        },
        "selected_instance_mapping": [mapping],
        "gce_create_rows": [],
        "actual_launch_rows": [],
        "selected_instance_count": 1,
        "exact_created_instance_count": 1,
        "exact_uncreated_instance_count": 0,
        "create_classification": "all_selected_created",
        "actual_launch_receipt_present": False,
        "journal_event_count": 12,
        "journal_hash_chain_valid": True,
        "all_producer_receipts_valid": True,
        "all_owned_instances_absent": True,
        "all_owned_boot_disks_absent": True,
        "worker_iam_bindings_absent": True,
        "additional_create_authorized": False,
        "current_profile_changed": False,
    }
    return {**core, "proof_sha256": controller_v2.canonical_sha256(core)}


def _install_replay_stubs(
    monkeypatch: pytest.MonkeyPatch, manifest: dict, proof: dict
) -> None:
    monkeypatch.setattr(
        production_receiver_v2,
        "_validated_material",
        lambda **kwargs: (manifest["receiver_request"], {}, b"startup", lambda x: x),
    )
    monkeypatch.setattr(
        controller_v2,
        "Full100WaveControllerV2",
        lambda **kwargs: object(),
    )

    class Adapter:
        def validate(self, **kwargs):
            return deepcopy(proof)

    monkeypatch.setattr(
        production_receiver_v2,
        "_lifecycle_adapter",
        lambda **kwargs: Adapter(),
    )


def test_cleanup_manifest_is_replayed_and_tamper_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, ledger, resume = _initial()
    manifest = _cleanup_manifest(tmp_path, plan, ledger, resume)
    proof = _production_proof(plan, ledger, resume, manifest)
    _install_replay_stubs(monkeypatch, manifest, proof)
    checked, replayed = subject._validate_cleanup_manifest_and_replay(
        plan=plan,
        ledger=ledger,
        resume=resume,
        cleanup_manifest=manifest,
        journal_dir=tmp_path / "journal",
    )
    assert checked["manifest_sha256"] == manifest["manifest_sha256"]
    assert replayed == proof

    tampered = deepcopy(manifest)
    tampered["worker_iam_bindings_absent"] = False
    tampered["manifest_sha256"] = cleanup_v2.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="cleanup safety"):
        subject._validate_cleanup_manifest_and_replay(
            plan=plan,
            ledger=ledger,
            resume=resume,
            cleanup_manifest=tampered,
            journal_dir=tmp_path / "journal",
        )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("lifecycle_event_sha256", "6" * 64),
        ("lifecycle_receipt_sha256", "6" * 64),
        ("run_name", "regular-hu-m31-c02-f100wv2-wrong"),
        ("gce_create_receipt", {"receipt_sha256": "6" * 64}),
    ],
)
def test_replayed_proof_digest_context_and_receipt_bindings_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    plan, ledger, resume = _initial()
    manifest = _cleanup_manifest(tmp_path, plan, ledger, resume)
    proof = _production_proof(plan, ledger, resume, manifest)
    proof[field] = replacement
    proof["proof_sha256"] = controller_v2.canonical_sha256(
        {key: value for key, value in proof.items() if key != "proof_sha256"}
    )
    _install_replay_stubs(monkeypatch, manifest, proof)
    with pytest.raises(ValueError, match="replayed cleanup is incomplete"):
        subject._validate_cleanup_manifest_and_replay(
            plan=plan,
            ledger=ledger,
            resume=resume,
            cleanup_manifest=manifest,
            journal_dir=tmp_path / "journal",
        )


def test_replayed_proof_requires_exactly_one_selected_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, ledger, resume = _initial()
    manifest = _cleanup_manifest(tmp_path, plan, ledger, resume)
    proof = _production_proof(plan, ledger, resume, manifest)
    extra = deepcopy(proof["selected_instance_mapping"][0])
    extra["job_id"] = "reference-shard-00"
    proof["selected_instance_mapping"].append(extra)
    proof["proof_sha256"] = controller_v2.canonical_sha256(
        {key: value for key, value in proof.items() if key != "proof_sha256"}
    )
    _install_replay_stubs(monkeypatch, manifest, proof)
    with pytest.raises(ValueError, match="lifecycle mapping changed"):
        subject._validate_cleanup_manifest_and_replay(
            plan=plan,
            ledger=ledger,
            resume=resume,
            cleanup_manifest=manifest,
            journal_dir=tmp_path / "journal",
        )

    proof["selected_instance_mapping"] = None
    proof["proof_sha256"] = controller_v2.canonical_sha256(
        {key: value for key, value in proof.items() if key != "proof_sha256"}
    )
    with pytest.raises(ValueError, match="lifecycle mapping changed"):
        subject._validate_cleanup_manifest_and_replay(
            plan=plan,
            ledger=ledger,
            resume=resume,
            cleanup_manifest=manifest,
            journal_dir=tmp_path / "journal",
        )


def test_manifest_lifecycle_must_be_the_replayed_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, ledger, resume = _initial()
    manifest = _cleanup_manifest(tmp_path, plan, ledger, resume)
    proof = _production_proof(plan, ledger, resume, manifest)
    lifecycle = manifest["lifecycle_receipt"]
    lifecycle["actual_launch_receipt_sha256"] = "6" * 64
    lifecycle["receipt_sha256"] = controller_v2.canonical_sha256(
        {key: value for key, value in lifecycle.items() if key != "receipt_sha256"}
    )
    manifest["manifest_sha256"] = cleanup_v2.canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    _install_replay_stubs(monkeypatch, manifest, proof)
    with pytest.raises(ValueError, match="replayed cleanup is incomplete"):
        subject._validate_cleanup_manifest_and_replay(
            plan=plan,
            ledger=ledger,
            resume=resume,
            cleanup_manifest=manifest,
            journal_dir=tmp_path / "journal",
        )


def test_cli_requires_explicit_cloud_read_and_exact_run(
    tmp_path: Path,
) -> None:
    plan, ledger, resume = _initial()
    paths = {}
    for name, value in (
        ("plan", plan),
        ("ledger", ledger),
        ("resume", resume),
        ("bootstrap", _bootstrap(plan, ledger, resume)),
        ("cleanup", {}),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    common = [
        "--wave-plan",
        str(paths["plan"]),
        "--attempt-ledger",
        str(paths["ledger"]),
        "--resume-plan",
        str(paths["resume"]),
        "--job-bootstrap",
        str(paths["bootstrap"]),
        "--cleanup-manifest",
        str(paths["cleanup"]),
        "--journal-dir",
        str(tmp_path),
        "--profile-path",
        str(tmp_path / "profile.py"),
        "--output",
        str(tmp_path / "out.json"),
        "--confirm-run-name",
        RUN,
    ]
    with pytest.raises(PermissionError, match="explicit cloud read"):
        subject.main(common)
    with pytest.raises(PermissionError, match="confirmation changed"):
        subject.main([*common, "--allow-cloud-read", "--confirm-run-name", "wrong"])


def test_standard_production_acceptance_core_rejects_canary_half_pair() -> None:
    """The canary can only enter through its diagnostic-only receiver."""

    plan, _, resume = _initial()
    with pytest.raises(ValueError, match="candidate/reference half-pair"):
        result_receiver_v2._selected_candidate_reference_pairs(plan, resume)
