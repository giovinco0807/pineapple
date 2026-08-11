from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_step6d_performance_lock_rearm2_open as subject
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subject.canonical_bytes(value))


@pytest.fixture
def inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> subject.PerformanceLockRearm2Inputs:
    repository = tmp_path / "repository"
    ai_profiles = repository / "src/ofc_regular/ai_profiles.py"
    ai_profiles.parent.mkdir(parents=True)
    ai_profiles.write_bytes(b"CURRENT = 'unchanged'\n")
    ai_sha = subject.sha256_file(ai_profiles)
    monkeypatch.setattr(subject, "AI_PROFILES_CURRENT_SHA256", ai_sha)
    step6d = repository / "configs/hu_joint_policy_m31_t3_step6d_contract.json"
    _write(step6d, {"anchors": {"policy_registry": {"byte_sha256": ai_sha}}})
    monkeypatch.setattr(
        subject, "STEP6D_CONTRACT_BYTE_SHA256", subject.sha256_file(step6d)
    )
    runner_source = (
        repository / "src/ofc_regular/run_hu_m31_t3_step6d_performance_v2.py"
    )
    runner_source.write_bytes(b"# recovery-v3\n")
    monkeypatch.setattr(subject, "MODEL_INPUT_PATHS", ())
    monkeypatch.setattr(subject, "ROOT_GENERATOR_INPUT_PATHS", ())

    candidate = tmp_path / "candidate.so"
    reference = tmp_path / "reference.so"
    feature = tmp_path / "feature.so"
    startup = tmp_path / "startup.sh"
    for path, raw in (
        (candidate, b"candidate"),
        (reference, b"reference"),
        (feature, b"feature"),
        (startup, b"#!/bin/sh\n"),
    ):
        path.write_bytes(raw)
    monkeypatch.setattr(
        subject, "CANDIDATE_LIBRARY_SHA256", subject.sha256_file(candidate)
    )
    monkeypatch.setattr(
        subject, "REFERENCE_LIBRARY_SHA256", subject.sha256_file(reference)
    )
    monkeypatch.setattr(
        subject, "FEATURE_ENCODER_SHA256", subject.sha256_file(feature)
    )
    monkeypatch.setattr(
        runner,
        "CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256",
        subject.CANDIDATE_LIBRARY_SHA256,
    )
    monkeypatch.setattr(
        runner,
        "CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256",
        subject.REFERENCE_LIBRARY_SHA256,
    )
    contract = runner.build_run_contract(
        candidate_library_sha256=subject.CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=subject.REFERENCE_LIBRARY_SHA256,
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT,
    )
    digest = runner.canonical_sha256(contract)
    monkeypatch.setattr(subject, "LOCK_RUN_CONTRACT_DIGEST", digest)

    receipt = {"schema": "test-closeout", "terminal": True}
    incident = tmp_path / "closeout.json"
    _write(incident, receipt)
    monkeypatch.setattr(subject, "_validate_incident", lambda _path: receipt)
    authority = {
        "fresh_lock_ordinal": "rearm2",
        "fresh_root_set_required": True,
        "fresh_seed_set_required": True,
        "fresh_global_claim_required_before_root_content": True,
        "fresh_root_content_authorized_before_global_claim": False,
        "rearm1_attempt1_authorized": False,
        "rearm1_package_authorized": False,
        "rearm1_roots_authorized": False,
        "rearm1_seeds_authorized": False,
        "authorization_before_actual_package_smoke_allowed": False,
    }
    disposition = {
        "rearm1_attempt1_authorized": False,
        "rearm1_package_reuse_authorized": False,
        "rearm1_root_reuse_authorized": False,
        "rearm1_seed_reuse_authorized": False,
        "rearm1_claim_reuse_authorized": False,
    }
    smoke = subject._expected_smoke_requirement()
    plan = {
        "schema": "test-rearm2-plan",
        "startup_failure_closeout": {
            "sha256": subject.REARM1_STARTUP_FAILURE_RECEIPT_SHA256,
            "receipt": receipt,
        },
        "rearm1_disposition": disposition,
        "fresh_lock_authority": authority,
        "actual_package_smoke_requirement": smoke,
        "run_contract": contract,
        "run_contract_digest": digest,
        "cloud_started": False,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }
    plan_path = tmp_path / "plan.json"
    _write(plan_path, plan)
    plan_module = SimpleNamespace(
        PRECONTENT_PLAN_SHA256=subject.sha256_file(plan_path),
        RECOVERY_V3_RUN_CONTRACT_DIGEST=digest,
        _rearm1_disposition=lambda: disposition,
        _fresh_lock_authority=lambda: authority,
        _smoke_requirement=lambda: smoke,
    )
    monkeypatch.setattr(
        subject, "_load_and_validate_plan", lambda _path: (plan, plan_module)
    )
    monkeypatch.setattr(
        subject.rearm1_open.legacy_open,
        "_validate_official_development_go",
        lambda *_args: {"all_gates_passed": True},
    )
    monkeypatch.setattr(
        subject,
        "_validate_prior_lock_controls",
        lambda _inputs: {
            "old_v1": {
                "global_claim_sha256": subject.OLD_V1_GLOBAL_CLAIM_SHA256,
                "seal_sha256": subject.OLD_V1_SEAL_SHA256,
            },
            "rearm1": {
                "global_claim_sha256": subject.REARM1_GLOBAL_CLAIM_SHA256,
                "seal_sha256": subject.REARM1_SEAL_SHA256,
            },
            "all_prior_reuse_authorized": False,
        },
    )
    global_claim = tmp_path / "global" / "GLOBAL_REARM2.json"
    monkeypatch.setattr(subject, "DEFAULT_GLOBAL_CLAIM_PATH", global_claim)
    return subject.PerformanceLockRearm2Inputs(
        repository_root=repository,
        plan_path=plan_path,
        lock_output_directory=tmp_path / "rearm2-roots",
        candidate_library=candidate,
        reference_library=reference,
        feature_encoder=feature,
        startup_source=startup,
        incident_receipt_path=incident,
        old_v1_root_directory=tmp_path / "v1-roots",
        rearm1_root_directory=tmp_path / "rearm1-roots",
        global_claim_path=global_claim,
    )


def test_claim_binds_v3_fresh_seeds_and_exhaustive_smoke_without_output_touch(
    inputs: subject.PerformanceLockRearm2Inputs,
) -> None:
    claim = subject._claim_payload(inputs, opened_unix_ns=123)

    assert not inputs.lock_output_directory.exists()
    assert claim["schema"] == subject.CLAIM_SCHEMA
    assert claim["status"] == subject.CLAIM_STATUS
    assert runner.contract_variant(claim["lock_run_contract"]) == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT
    )
    assert claim["seed_contract"]["seed_min"] == 710_108_071_901
    assert claim["seed_contract"]["performance_lock_rearm1_overlap_count"] == 0
    assert claim["actual_package_smoke_requirement"] == (
        subject._expected_smoke_requirement()
    )
    assert claim["restrictions"][
        "authorization_requires_exhaustive_actual_package_smoke"
    ] is True
    for key in (
        "rearm1_attempt1_reused",
        "rearm1_package_reused",
        "rearm1_root_reused",
        "rearm1_seed_reused",
        "rearm1_claim_reused",
    ):
        assert claim["rearm_guards"][key] is False


def test_preflight_is_deterministic_and_writes_no_claim_or_root(
    inputs: subject.PerformanceLockRearm2Inputs,
) -> None:
    first = subject.preflight_performance_lock_rearm2(inputs)
    second = subject.preflight_performance_lock_rearm2(inputs)
    claim_template = subject._claim_payload(inputs, opened_unix_ns=1)

    assert first == second
    assert first["schema"] == subject.PREFLIGHT_SCHEMA
    assert first["status"] == subject.PREFLIGHT_STATUS
    assert first["claim_template_opened_unix_ns"] == 1
    assert first["claim_template_sha256"] == subject.canonical_sha256(
        claim_template
    )
    assert first["lock_run_contract_digest"] == (
        claim_template["lock_run_contract_digest"]
    )
    assert first["accepted_binary_sha256"] == {
        role: claim_template["accepted_binaries"][role]["sha256"]
        for role in ("candidate", "reference", "feature_encoder")
    }
    assert first["seed_set_sha256"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SEED_SET_SHA256
    )
    assert first["proposed_global_claim_absent"] is True
    assert first["proposed_lock_output_absent"] is True
    assert first["persistent_write_executed"] is False
    assert first["new_root_content_opened"] is False
    assert first["cloud_mutation_executed"] is False
    assert first["current_profile_changed"] is False
    assert not Path(inputs.global_claim_path).exists()
    assert not Path(inputs.lock_output_directory).exists()


@pytest.mark.parametrize("occupied", ("claim", "output"))
def test_preflight_refuses_every_occupied_target_before_validation(
    inputs: subject.PerformanceLockRearm2Inputs,
    monkeypatch: pytest.MonkeyPatch,
    occupied: str,
) -> None:
    target = (
        Path(inputs.global_claim_path)
        if occupied == "claim"
        else Path(inputs.lock_output_directory)
    )
    if occupied == "claim":
        target.parent.mkdir(parents=True)
        target.write_bytes(b"already claimed\n")
    else:
        target.mkdir(parents=True)

    def unexpected(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("preflight validated after observing an occupied target")

    monkeypatch.setattr(subject, "_claim_payload", unexpected)
    with pytest.raises(FileExistsError, match="requires unused targets"):
        subject.preflight_performance_lock_rearm2(inputs)


def test_preflight_replays_real_binary_prerequisites(
    inputs: subject.PerformanceLockRearm2Inputs,
) -> None:
    inputs.candidate_library.write_bytes(b"tampered candidate")

    with pytest.raises(ValueError, match="accepted rearm2 binary identity"):
        subject.preflight_performance_lock_rearm2(inputs)
    assert not Path(inputs.global_claim_path).exists()
    assert not Path(inputs.lock_output_directory).exists()


def test_preflight_cli_is_public_and_uses_no_write_path(
    inputs: subject.PerformanceLockRearm2Inputs,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence = {
        "schema": subject.PREFLIGHT_SCHEMA,
        "status": subject.PREFLIGHT_STATUS,
    }
    calls: list[subject.PerformanceLockRearm2Inputs] = []
    monkeypatch.setattr(subject, "_inputs_from_args", lambda _args: inputs)
    monkeypatch.setattr(
        subject,
        "preflight_performance_lock_rearm2",
        lambda value: calls.append(value) or evidence,
    )
    result = subject.main(
        [
            "preflight",
            "--lock-output",
            str(inputs.lock_output_directory),
            "--candidate-library",
            str(inputs.candidate_library),
            "--reference-library",
            str(inputs.reference_library),
            "--feature-encoder",
            str(inputs.feature_encoder),
            "--startup-source",
            str(inputs.startup_source),
            "--incident-receipt",
            str(inputs.incident_receipt_path),
            "--old-v1-roots",
            str(inputs.old_v1_root_directory),
            "--rearm1-roots",
            str(inputs.rearm1_root_directory),
        ]
    )

    assert result == 0
    assert calls == [inputs]
    assert json.loads(capsys.readouterr().out) == evidence
    assert "preflight_performance_lock_rearm2" in subject.__all__


def test_crash_after_claim_consumes_rearm2_before_output_touch(
    inputs: subject.PerformanceLockRearm2Inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Crash(RuntimeError):
        pass

    def crash(_path: Path) -> None:
        assert not inputs.lock_output_directory.exists()
        raise Crash

    monkeypatch.setattr(subject, "_after_claim_persisted", crash)
    with pytest.raises(Crash):
        subject.open_performance_lock_rearm2(inputs)
    before = Path(inputs.global_claim_path).read_bytes()
    with pytest.raises(FileExistsError):
        subject.open_performance_lock_rearm2(inputs)
    assert Path(inputs.global_claim_path).read_bytes() == before
    assert not inputs.lock_output_directory.exists()


def test_rearm2_output_must_be_disjoint_from_both_prior_root_trees(
    inputs: subject.PerformanceLockRearm2Inputs,
) -> None:
    for prior in (
        inputs.old_v1_root_directory,
        inputs.rearm1_root_directory,
        inputs.rearm1_root_directory / "nested",
    ):
        changed = subject.PerformanceLockRearm2Inputs(
            **{
                **inputs.__dict__,
                "lock_output_directory": prior,
            }
        )
        with pytest.raises(ValueError, match="disjoint|reuse"):
            subject._claim_payload(changed, opened_unix_ns=123)


def test_materialization_payload_forbids_all_prior_identity_reuse() -> None:
    claim = {
        "precontent_plan": {"sha256": "a" * 64},
        "lock_run_contract_digest": "b" * 64,
    }
    roots = [{"hand_index": index} for index in range(100)]
    value = subject._materialization_payload(claim, roots)
    assert value["schema"] == subject.MATERIALIZATION_SCHEMA
    assert value["same_identity_resume_only"] is True
    assert value["fresh_recovery_v3_seed_schedule"] is True
    assert value["old_v1_root_reused"] is False
    assert value["rearm1_attempt1_reused"] is False
    assert value["rearm1_package_reused"] is False
    assert value["rearm1_root_reused"] is False
    assert value["rearm1_seed_reused"] is False
    assert value["rearm1_claim_reused"] is False
    assert value["reseeded"] is False


def test_stored_claim_validation_replays_exact_identity(
    inputs: subject.PerformanceLockRearm2Inputs,
) -> None:
    claim = subject.open_performance_lock_rearm2(inputs)
    assert subject.validate_global_claim(inputs) == claim
    stored = json.loads(Path(inputs.global_claim_path).read_text(encoding="utf-8"))
    stored["restrictions"]["reseed_allowed"] = True
    Path(inputs.global_claim_path).write_bytes(subject.canonical_bytes(stored))
    with pytest.raises(ValueError, match="identity"):
        subject.validate_global_claim(inputs)


def _claim_for_roots() -> dict[str, Any]:
    contract = runner.build_run_contract(
        candidate_library_sha256=subject.CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=subject.REFERENCE_LIBRARY_SHA256,
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT,
    )
    return {
        "precontent_plan": {"sha256": "a" * 64},
        "lock_run_contract": contract,
        "lock_run_contract_digest": runner.canonical_sha256(contract),
        "actual_package_smoke_requirement": subject._expected_smoke_requirement(),
    }


def test_materialize_routes_only_to_v3_after_claim_validation(
    inputs: subject.PerformanceLockRearm2Inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim_for_roots()
    roots = [{"hand_index": index} for index in range(100)]
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(subject, "validate_global_claim", lambda _inputs: claim)

    def materialize(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        return roots

    monkeypatch.setattr(runner, "_materialize_roots", materialize)
    monkeypatch.setattr(
        subject.rearm1_open,
        "_load_roots",
        lambda *_args, **_kwargs: (roots, []),
    )
    value = subject.materialize_performance_lock_rearm2(inputs)

    assert runner.contract_variant(calls[0]["contract"]) == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT
    )
    assert calls[0]["indices"] == runner.CONTRACT_HAND_INDICES
    assert value == subject._materialization_payload(claim, roots)


@dataclass(frozen=True)
class _Observation:
    value: str

    def fingerprint(self) -> str:
        return self.value


def _roots(prefix: str, seed_base: int) -> tuple[list[dict], list[tuple]]:
    profile_names = tuple(subject.M31_T3_BEHAVIOR_PROFILES)
    roots: list[dict] = []
    observations: list[tuple] = []
    for index in runner.CONTRACT_HAND_INDICES:
        first = f"{prefix}-first-{index}"
        second = f"{prefix}-second-{index}"
        roots.append(
            {
                "hand_index": index,
                "profile": profile_names[index % len(profile_names)],
                "seeds": {
                    "hand": seed_base + index * 3,
                    "behavior": seed_base + index * 3 + 1,
                    "future": seed_base + index * 3 + 2,
                },
                "observations": [
                    {"seat": "first", "observation_fingerprint": first},
                    {"seat": "second", "observation_fingerprint": second},
                ],
            }
        )
        observations.append((_Observation(first), _Observation(second)))
    return roots, observations


def test_seal_proves_zero_overlap_against_development_v1_and_rearm1(
    inputs: subject.PerformanceLockRearm2Inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim_for_roots()
    new, observations = _roots("new", 900_000)
    development, _ = _roots("development", 100_000)
    old_v1, _ = _roots("v1", 200_000)
    rearm1, _ = _roots("rearm1", 300_000)
    output = inputs.lock_output_directory
    output.mkdir(parents=True)
    _write(output / "materialization.json", subject._materialization_payload(claim, new))
    monkeypatch.setattr(subject, "validate_global_claim", lambda _inputs: claim)
    monkeypatch.setattr(
        subject.rearm1_open,
        "_load_roots",
        lambda path, *_args, **_kwargs: (
            (new, observations)
            if Path(path) == output
            else (_ for _ in ()).throw(AssertionError(path))
        ),
    )
    monkeypatch.setattr(
        subject.development_roots,
        "load_frozen_roots",
        lambda _path: development,
    )
    monkeypatch.setattr(
        subject.rearm1_open,
        "_load_old_v1_roots",
        lambda _inputs: (old_v1, []),
    )
    monkeypatch.setattr(
        subject,
        "_load_rearm1_roots",
        lambda _inputs: (rearm1, []),
    )
    monkeypatch.setattr(
        subject.rearm1_open.legacy_open,
        "_topology_rows",
        lambda roots, parsed: [
            {"hand_index": root["hand_index"]} for root in roots
        ],
    )
    value = subject._build_root_seal(inputs)

    assert value["schema"] == subject.SEAL_SCHEMA
    assert value["development_comparison"][
        "lock_root_hash_overlap_count"
    ] == 0
    assert value["development_comparison"]["lock_fingerprint_overlap_count"] == 0
    assert value["development_comparison"]["lock_seed_overlap_count"] == 0
    prior = value["prior_lock_comparison"]
    for prefix in ("old_v1", "rearm1"):
        assert prior[f"{prefix}_root_hash_overlap_count"] == 0
        assert prior[f"{prefix}_fingerprint_overlap_count"] == 0
        assert prior[f"{prefix}_seed_overlap_count"] == 0
    assert prior["rearm1_attempt1_reused"] is False
    assert prior["rearm1_package_reused"] is False
    assert prior["rearm1_root_reused"] is False
    assert prior["rearm1_seed_reused"] is False
    assert prior["rearm1_claim_reused"] is False
