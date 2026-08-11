from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_step6d_performance_lock_rearm1_open as subject
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner
from ofc_regular.hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subject.canonical_bytes(value))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture
def fake_claim_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> subject.PerformanceLockRearm1Inputs:
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
    runner_source.write_bytes(b"# frozen recovery runner\n")
    models = ("models/a", "models/b")
    generators = ("src/root_a.py", "src/root_b.py")
    for index, relative in enumerate((*models, *generators)):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"input-{index}".encode())
    monkeypatch.setattr(subject, "MODEL_INPUT_PATHS", models)
    monkeypatch.setattr(subject, "ROOT_GENERATOR_INPUT_PATHS", generators)

    candidate = tmp_path / "candidate.so"
    reference = tmp_path / "reference.so"
    feature = tmp_path / "feature.py"
    startup = tmp_path / "startup.sh"
    incident = tmp_path / "incident.json"
    for path, raw in (
        (candidate, b"candidate"),
        (reference, b"reference"),
        (feature, b"feature"),
        (startup, b"#!/bin/sh\n"),
    ):
        path.write_bytes(raw)
    receipt = {
        "schema": "test-closeout",
        "status": "terminal-startup-failure",
        "terminal": True,
    }
    _write(incident, receipt)
    monkeypatch.setattr(subject, "_validate_incident", lambda _path: receipt)
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
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT,
    )
    digest = runner.canonical_sha256(contract)
    monkeypatch.setattr(subject, "LOCK_RUN_CONTRACT_DIGEST", digest)
    old_disposition = {
        "old_attempt1_authorized": False,
        "old_root_reuse_authorized": False,
        "old_seed_reuse_authorized": False,
        "old_package_reuse_authorized": False,
    }
    fresh_authority = {
        "fresh_lock_ordinal": "rearm1",
        "authorized_fresh_lock_count": 1,
        "fresh_lock_plan_authorized": True,
        "fresh_root_set_required": True,
        "fresh_seed_set_required": True,
        "fresh_global_claim_required_before_root_content": True,
        "fresh_root_content_authorized_before_global_claim": False,
        "old_attempt1_authorized": False,
        "old_roots_authorized": False,
        "old_seeds_authorized": False,
    }
    plan = {
        "schema": "test-rearm1-plan",
        "startup_failure_closeout": {
            "sha256": subject.STARTUP_FAILURE_RECEIPT_SHA256,
            "receipt": receipt,
        },
        "old_run_disposition": old_disposition,
        "fresh_lock_authority": fresh_authority,
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
        RECOVERY_RUN_CONTRACT_DIGEST=digest,
        _old_run_disposition=lambda: old_disposition,
        _fresh_lock_authority=lambda: fresh_authority,
    )
    monkeypatch.setattr(
        subject, "_load_and_validate_plan", lambda _path: (plan, plan_module)
    )
    monkeypatch.setattr(
        subject,
        "_validate_official_development_go"
        if hasattr(subject, "_validate_official_development_go")
        else "unused",
        lambda *_args: {},
        raising=False,
    )
    monkeypatch.setattr(
        subject.legacy_open,
        "_validate_official_development_go",
        lambda *_args: {
            "summary": {"sha256": "1" * 64},
            "validation": {"sha256": "2" * 64},
            "all_gates_passed": True,
            "performance_lock_authorized": True,
        },
    )
    monkeypatch.setattr(
        subject,
        "_validate_old_v1_controls",
        lambda _inputs: {
            "global_claim": {"sha256": subject.OLD_V1_GLOBAL_CLAIM_SHA256},
            "materialization": {
                "sha256": subject.OLD_V1_MATERIALIZATION_SHA256
            },
            "seal": {"sha256": subject.OLD_V1_SEAL_SHA256},
            "run_contract_digest": subject.legacy_open.LOCK_RUN_CONTRACT_DIGEST,
            "seed_set_sha256": (
                runner.CANDIDATE02_PERFORMANCE_LOCK_SEED_SET_SHA256
            ),
            "attempt1_used": False,
            "root_reuse_authorized": False,
        },
    )
    global_claim = tmp_path / "global" / "GLOBAL_PERFORMANCE_LOCK_REARM1_CLAIM.json"
    monkeypatch.setattr(subject, "DEFAULT_GLOBAL_CLAIM_PATH", global_claim)
    return subject.PerformanceLockRearm1Inputs(
        repository_root=repository,
        plan_path=plan_path,
        lock_output_directory=tmp_path / "new-rearm1-roots",
        candidate_library=candidate,
        reference_library=reference,
        feature_encoder=feature,
        startup_source=startup,
        incident_receipt_path=incident,
        old_v1_root_directory=tmp_path / "old-v1-roots",
        global_claim_path=global_claim,
    )


def test_frozen_plan_and_incident_receipt_are_exact() -> None:
    plan, module = subject._load_and_validate_plan(
        subject.DEFAULT_PRECONTENT_PLAN_PATH
    )
    assert subject.sha256_file(subject.DEFAULT_PRECONTENT_PLAN_PATH) == (
        subject.PRECONTENT_PLAN_SHA256
    )
    assert plan["run_contract_digest"] == subject.LOCK_RUN_CONTRACT_DIGEST
    assert module.RECOVERY_RUN_CONTRACT_DIGEST == subject.LOCK_RUN_CONTRACT_DIGEST
    receipt = subject._validate_incident(
        subject.DEFAULT_PRECONTENT_PLAN_PATH.parent
        / "performance_lock_v1_startup_failure_closeout.json"
    )
    assert subject.canonical_sha256(receipt) == subject.STARTUP_FAILURE_RECEIPT_SHA256


def test_claim_is_distinct_and_binds_fresh_recovery_without_touching_output(
    fake_claim_inputs: subject.PerformanceLockRearm1Inputs,
) -> None:
    claim = subject._claim_payload(fake_claim_inputs, opened_unix_ns=123)
    assert not fake_claim_inputs.lock_output_directory.exists()
    assert claim["schema"] == subject.CLAIM_SCHEMA
    assert claim["status"] == subject.CLAIM_STATUS
    assert claim["lock_run_contract_digest"] == subject.LOCK_RUN_CONTRACT_DIGEST
    assert runner.contract_variant(claim["lock_run_contract"]) == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT
    )
    assert claim["seed_contract"]["seed_min"] == 700_108_071_901
    assert claim["seed_contract"]["performance_lock_v1_overlap_count"] == 0
    assert claim["image"] == subject.IMAGE
    assert claim["allocation"] == subject.ALLOCATION
    assert claim["rearm_guards"]["old_v1_attempt1_reused"] is False
    assert claim["rearm_guards"]["old_v1_root_reused"] is False
    assert claim["restrictions"]["cloud_authorized"] is False
    assert claim["restrictions"]["training_authorized"] is False
    assert claim["restrictions"]["quality_authorized"] is False
    assert claim["restrictions"]["current_profile_resolution_allowed"] is False
    assert claim["restrictions"]["opponent_private_discards_allowed"] is False


def test_crash_after_claim_consumes_rearm1_without_touching_output(
    fake_claim_inputs: subject.PerformanceLockRearm1Inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Crash(RuntimeError):
        pass

    def crash(path: Path) -> None:
        assert _read(path)["schema"] == subject.CLAIM_SCHEMA
        assert not fake_claim_inputs.lock_output_directory.exists()
        raise Crash

    monkeypatch.setattr(subject, "_after_claim_persisted", crash)
    with pytest.raises(Crash):
        subject.open_performance_lock_rearm1(fake_claim_inputs)
    before = Path(fake_claim_inputs.global_claim_path).read_bytes()
    with pytest.raises(FileExistsError):
        subject.open_performance_lock_rearm1(fake_claim_inputs)
    assert Path(fake_claim_inputs.global_claim_path).read_bytes() == before
    assert not fake_claim_inputs.lock_output_directory.exists()


def _recovery_claim() -> dict[str, Any]:
    contract = runner.build_run_contract(
        candidate_library_sha256=subject.CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=subject.REFERENCE_LIBRARY_SHA256,
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT,
    )
    return {
        "precontent_plan": {"sha256": "a" * 64},
        "lock_run_contract": contract,
        "lock_run_contract_digest": runner.canonical_sha256(contract),
    }


def _minimal_inputs(tmp_path: Path, output: Path) -> subject.PerformanceLockRearm1Inputs:
    return subject.PerformanceLockRearm1Inputs(
        repository_root=tmp_path,
        plan_path=tmp_path / "plan",
        lock_output_directory=output,
        candidate_library=tmp_path / "candidate",
        reference_library=tmp_path / "reference",
        feature_encoder=tmp_path / "feature",
        startup_source=tmp_path / "startup",
        incident_receipt_path=tmp_path / "incident",
        old_v1_root_directory=tmp_path / "old",
    )


def test_materialize_uses_recovery_variant_and_exact_identity_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "rearm1"
    inputs = _minimal_inputs(tmp_path, output)
    claim = _recovery_claim()
    calls = 0

    def validate(_inputs: subject.PerformanceLockRearm1Inputs) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert not output.exists()
        return claim

    monkeypatch.setattr(subject, "validate_global_claim", validate)
    monkeypatch.setattr(
        runner,
        "_validate_root_artifact",
        lambda _contract, value, *, index: (
            f"first-{value['hand_index']}",
            f"second-{index}",
        ),
    )
    stop_after = {"count": 25}

    def materialize(
        *,
        contract: dict[str, Any],
        repository_root: Path,
        output_dir: Path,
        indices: tuple[int, ...],
    ) -> list[dict[str, Any]]:
        assert runner.contract_variant(contract) == (
            runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT
        )
        assert repository_root == tmp_path.resolve()
        roots = []
        created = 0
        for index in indices:
            path = output_dir / "roots" / f"hand_{index:03d}.json"
            if path.exists():
                value = _read(path)
            else:
                value = {"hand_index": index, "seed": 700_000 + index}
                _write(path, value)
                created += 1
                if stop_after["count"] and created == stop_after["count"]:
                    raise RuntimeError("crash")
            roots.append(value)
        return roots

    monkeypatch.setattr(runner, "_materialize_roots", materialize)
    with pytest.raises(RuntimeError, match="crash"):
        subject.materialize_performance_lock_rearm1(inputs)
    partial = {
        path.name: path.read_bytes()
        for path in (output / "roots").glob("hand_*.json")
    }
    assert len(partial) == 25

    stop_after["count"] = 0
    result = subject.materialize_performance_lock_rearm1(inputs)
    assert result["schema"] == subject.MATERIALIZATION_SCHEMA
    assert result["status"] == subject.MATERIALIZATION_STATUS
    assert result["root_count"] == 100
    assert result["same_identity_resume_only"] is True
    assert result["fresh_recovery_seed_schedule"] is True
    assert result["old_v1_attempt1_reused"] is False
    assert result["old_v1_root_reused"] is False
    assert result["reseeded"] is False
    assert {
        name: (output / "roots" / name).read_bytes() for name in partial
    } == partial
    assert subject.materialize_performance_lock_rearm1(inputs) == result


def test_claim_rejection_precedes_new_output_touch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "must-not-exist"
    inputs = _minimal_inputs(tmp_path, output)
    monkeypatch.setattr(
        subject,
        "validate_global_claim",
        lambda _inputs: (_ for _ in ()).throw(ValueError("claim rejected")),
    )
    with pytest.raises(ValueError, match="claim rejected"):
        subject.materialize_performance_lock_rearm1(inputs)
    assert not output.exists()


@dataclass(frozen=True)
class _Observation:
    index: int
    seat: str
    prefix: str = "rearm1"

    def fingerprint(self) -> str:
        return f"{self.prefix}-{self.seat}-{self.index}"


def _roots(prefix: str, seed_base: int, *, profiles: bool) -> list[dict[str, Any]]:
    profile_names = tuple(M31_T3_BEHAVIOR_PROFILES)
    rows = []
    for index in runner.CONTRACT_HAND_INDICES:
        value = {
            "hand_index": index,
            "seeds": {
                "hand": seed_base + index * 3,
                "behavior": seed_base + index * 3 + 1,
                "future": seed_base + index * 3 + 2,
            },
            "observations": [
                {
                    "seat": "first",
                    "observation_fingerprint": f"{prefix}-first-{index}",
                },
                {
                    "seat": "second",
                    "observation_fingerprint": f"{prefix}-second-{index}",
                },
            ],
        }
        if profiles:
            value["profile"] = profile_names[index % len(profile_names)]
        rows.append(value)
    return rows


def _seal_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    subject.PerformanceLockRearm1Inputs,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    output = tmp_path / "rearm1"
    new = _roots("rearm1", 700_000, profiles=True)
    development = _roots("development", 800_000, profiles=False)
    old = _roots("old-v1", 900_000, profiles=True)
    for index, value in enumerate(new):
        _write(output / "roots" / f"hand_{index:03d}.json", value)
    claim = _recovery_claim()
    _write(
        output / "materialization.json",
        subject._materialization_payload(claim, new),
    )
    inputs = _minimal_inputs(tmp_path, output)
    monkeypatch.setattr(subject, "validate_global_claim", lambda _inputs: claim)

    def validate_root(
        _contract: dict[str, Any], value: dict[str, Any], *, index: int
    ) -> tuple[_Observation, _Observation]:
        assert value["hand_index"] == index
        return _Observation(index, "first"), _Observation(index, "second")

    monkeypatch.setattr(runner, "_validate_root_artifact", validate_root)
    monkeypatch.setattr(
        subject.development_roots,
        "load_frozen_roots",
        lambda _path: development,
    )
    monkeypatch.setattr(
        subject,
        "_load_old_v1_roots",
        lambda _inputs: (
            old,
            [
                (
                    _Observation(index, "first", "old-v1"),
                    _Observation(index, "second", "old-v1"),
                )
                for index in runner.CONTRACT_HAND_INDICES
            ],
        ),
    )
    monkeypatch.setattr(
        subject.legacy_open,
        "_topology_rows",
        lambda roots, observations: [
            {
                "hand_index": root["hand_index"],
                "first": pair[0].fingerprint(),
                "second": pair[1].fingerprint(),
            }
            for root, pair in zip(roots, observations, strict=True)
        ],
    )
    return inputs, new, development, old


def test_seal_is_balanced_hidden_safe_and_disjoint_from_both_prior_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, _new, _development, _old = _seal_fixture(tmp_path, monkeypatch)
    seal = subject.seal_performance_lock_rearm1(inputs)
    assert seal["schema"] == subject.SEAL_SCHEMA
    assert seal["status"] == subject.SEAL_STATUS
    assert seal["root_count"] == 100
    assert seal["observation_count"] == 200
    assert seal["profile_counts"] == {
        profile: 20 for profile in M31_T3_BEHAVIOR_PROFILES
    }
    assert seal["seat_counts"] == {"first": 100, "second": 100}
    assert set(seal["development_comparison"].values()) >= {0}
    assert seal["development_comparison"]["lock_seed_overlap_count"] == 0
    assert seal["development_comparison"]["lock_root_hash_overlap_count"] == 0
    assert seal["development_comparison"]["lock_fingerprint_overlap_count"] == 0
    prior = seal["old_performance_lock_comparison"]
    assert prior["rearm1_seed_overlap_count"] == 0
    assert prior["rearm1_root_hash_overlap_count"] == 0
    assert prior["rearm1_fingerprint_overlap_count"] == 0
    assert prior["old_v1_attempt1_reused"] is False
    assert prior["old_v1_root_reused"] is False
    assert seal["visibility"]["opponent_private_discards_used"] is False
    assert seal["visibility"]["current_profile_resolved"] is False
    assert seal["training_eligible"] is False
    assert seal["quality_evidence"] is False
    assert seal["current_profile_changed"] is False
    assert subject.validate_root_seal(inputs) == seal


@pytest.mark.parametrize("prior_name", ("development", "old_v1"))
@pytest.mark.parametrize("kind", ("fingerprint", "seed", "root"))
def test_seal_rejects_every_overlap_with_development_or_old_v1(
    prior_name: str,
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, new, development, old = _seal_fixture(tmp_path, monkeypatch)
    prior = development if prior_name == "development" else old
    if kind == "fingerprint":
        prior[0]["observations"][0]["observation_fingerprint"] = (
            new[0]["observations"][0]["observation_fingerprint"]
        )
    elif kind == "seed":
        prior[0]["seeds"]["hand"] = new[0]["seeds"]["hand"]
    else:
        prior[0] = new[0]
    with pytest.raises(ValueError, match="overlap development or old v1"):
        subject._build_root_seal(inputs)


def test_rearm_schemas_are_not_legacy_v1() -> None:
    assert subject.CLAIM_SCHEMA != subject.legacy_open.CLAIM_SCHEMA
    assert subject.MATERIALIZATION_SCHEMA != subject.legacy_open.MATERIALIZATION_SCHEMA
    assert subject.SEAL_SCHEMA != subject.legacy_open.SEAL_SCHEMA
    assert subject.PerformanceLockInputs is subject.PerformanceLockRearm1Inputs
