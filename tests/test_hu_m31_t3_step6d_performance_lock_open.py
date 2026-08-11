from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_step6d_performance_lock_open as subject
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner
from ofc_regular.hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


def _write_canonical(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subject.canonical_bytes(value))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture
def fake_open_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> subject.PerformanceLockInputs:
    repository = tmp_path / "repository"
    ai_profiles = repository / "src/ofc_regular/ai_profiles.py"
    ai_profiles.parent.mkdir(parents=True)
    ai_profiles.write_bytes(b"CURRENT = 'unchanged'\n")
    ai_sha = subject.sha256_file(ai_profiles)
    monkeypatch.setattr(subject, "AI_PROFILES_CURRENT_SHA256", ai_sha)

    step6d = repository / "configs/hu_joint_policy_m31_t3_step6d_contract.json"
    _write_canonical(
        step6d,
        {"anchors": {"policy_registry": {"byte_sha256": ai_sha}}},
    )
    monkeypatch.setattr(
        subject, "STEP6D_CONTRACT_BYTE_SHA256", subject.sha256_file(step6d)
    )

    runner_source = (
        repository / "src/ofc_regular/run_hu_m31_t3_step6d_performance_v2.py"
    )
    runner_source.write_bytes(b"# frozen runner\n")
    model_paths = ("models/a.model", "models/b.model")
    root_generator_paths = ("src/root_a.py", "src/root_b.py")
    for index, relative in enumerate((*model_paths, *root_generator_paths)):
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"input-{index}".encode("ascii"))
    monkeypatch.setattr(subject, "MODEL_INPUT_PATHS", model_paths)
    monkeypatch.setattr(subject, "ROOT_GENERATOR_INPUT_PATHS", root_generator_paths)

    candidate = tmp_path / "candidate.so"
    reference = tmp_path / "reference.so"
    feature = tmp_path / "feature_encoder.py"
    startup = tmp_path / "startup.sh"
    candidate.write_bytes(b"candidate-library")
    reference.write_bytes(b"reference-library")
    feature.write_bytes(b"feature-encoder")
    startup.write_bytes(b"#!/bin/sh\n")
    candidate_sha = subject.sha256_file(candidate)
    reference_sha = subject.sha256_file(reference)
    feature_sha = subject.sha256_file(feature)
    monkeypatch.setattr(subject, "CANDIDATE_LIBRARY_SHA256", candidate_sha)
    monkeypatch.setattr(subject, "REFERENCE_LIBRARY_SHA256", reference_sha)
    monkeypatch.setattr(subject, "FEATURE_ENCODER_SHA256", feature_sha)

    contract = runner.build_run_contract(
        candidate_library_sha256=candidate_sha,
        reference_library_sha256=reference_sha,
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
    )
    contract_digest = runner.canonical_sha256(contract)
    monkeypatch.setattr(subject, "LOCK_RUN_CONTRACT_DIGEST", contract_digest)
    plan = {
        "schema": "test_precontent_plan",
        "candidate_variant": runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
        "run_contract": contract,
        "run_contract_digest": contract_digest,
        "cloud_started": False,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }
    plan_path = tmp_path / "precontent_plan.json"
    _write_canonical(plan_path, plan)
    monkeypatch.setattr(subject, "_load_and_validate_plan", lambda _path: plan)
    monkeypatch.setattr(
        subject,
        "_validate_official_development_go",
        lambda _summary, _validation: {
            "summary": {"sha256": "1" * 64},
            "validation": {"sha256": "2" * 64},
            "validation_report_sha256": "3" * 64,
            "development_run_name": "frozen-development",
            "all_gates_passed": True,
            "performance_lock_authorized": True,
        },
    )

    global_claim = tmp_path / "global" / "GLOBAL_PERFORMANCE_LOCK_CLAIM.json"
    monkeypatch.setattr(subject, "DEFAULT_GLOBAL_CLAIM_PATH", global_claim)
    return subject.PerformanceLockInputs(
        repository_root=repository,
        plan_path=plan_path,
        lock_output_directory=tmp_path / "performance-lock-roots",
        candidate_library=candidate,
        reference_library=reference,
        feature_encoder=feature,
        startup_source=startup,
        development_summary_path=tmp_path / "development-summary.json",
        development_validation_path=tmp_path / "development-validation.json",
        development_root_directory=tmp_path / "development-roots",
        global_claim_path=global_claim,
    )


def test_frozen_precontent_plan_adapter_requires_exact_producer_hash(
    tmp_path: Path,
) -> None:
    plan = subject._load_and_validate_plan(subject.DEFAULT_PRECONTENT_PLAN_PATH)
    assert plan["run_contract_digest"] == subject.LOCK_RUN_CONTRACT_DIGEST
    assert (
        subject.sha256_file(subject.DEFAULT_PRECONTENT_PLAN_PATH)
        == subject.PRECONTENT_PLAN_SHA256
    )

    tampered = dict(plan)
    tampered["status"] = "tampered"
    path = tmp_path / "tampered-plan.json"
    _write_canonical(path, tampered)
    with pytest.raises(ValueError):
        subject._load_and_validate_plan(path)


def test_claim_binds_every_frozen_input_without_touching_lock_output(
    fake_open_inputs: subject.PerformanceLockInputs,
) -> None:
    claim = subject._claim_payload(fake_open_inputs, opened_unix_ns=123)

    assert not fake_open_inputs.lock_output_directory.exists()
    assert claim["precontent_plan"]["sha256"] == subject.sha256_file(
        fake_open_inputs.plan_path
    )
    assert claim["step6d_contract"]["sha256"] == subject.sha256_file(
        Path(fake_open_inputs.repository_root)
        / "configs/hu_joint_policy_m31_t3_step6d_contract.json"
    )
    assert claim["runner_source"]["sha256"] == subject.sha256_file(
        Path(fake_open_inputs.repository_root)
        / "src/ofc_regular/run_hu_m31_t3_step6d_performance_v2.py"
    )
    assert claim["ai_profiles_current"]["sha256"] == subject.AI_PROFILES_CURRENT_SHA256
    assert {
        key: record["sha256"] for key, record in claim["accepted_binaries"].items()
    } == {
        "candidate": subject.CANDIDATE_LIBRARY_SHA256,
        "reference": subject.REFERENCE_LIBRARY_SHA256,
        "feature_encoder": subject.FEATURE_ENCODER_SHA256,
    }
    assert set(claim["model_inputs"]) == set(subject.MODEL_INPUT_PATHS)
    assert set(claim["root_generator_inputs"]) == set(
        subject.ROOT_GENERATOR_INPUT_PATHS
    )
    assert claim["image"] == subject.IMAGE
    assert claim["allocation"] == subject.ALLOCATION
    assert (
        claim["seed_contract"]["seed_set_sha256"]
        == runner.CANDIDATE02_PERFORMANCE_LOCK_SEED_SET_SHA256
    )
    assert claim["seed_contract"]["seed_count"] == 600
    assert claim["startup_source"]["sha256"] == subject.sha256_file(
        fake_open_inputs.startup_source
    )
    assert claim["restrictions"]["training_authorized"] is False
    assert claim["restrictions"]["promotion_authorized"] is False
    assert claim["restrictions"]["current_profile_resolution_allowed"] is False


def test_global_claim_is_durable_before_crash_and_crash_consumes_it(
    fake_open_inputs: subject.PerformanceLockInputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedCrash(RuntimeError):
        pass

    def crash_after_persist(claim_path: Path) -> None:
        stored = _read_json(claim_path)
        assert stored["schema"] == subject.CLAIM_SCHEMA
        assert claim_path.read_bytes() == subject.canonical_bytes(stored)
        assert not fake_open_inputs.lock_output_directory.exists()
        raise SimulatedCrash

    monkeypatch.setattr(subject, "_after_claim_persisted", crash_after_persist)
    with pytest.raises(SimulatedCrash):
        subject.open_performance_lock(fake_open_inputs)

    claim_bytes = Path(fake_open_inputs.global_claim_path).read_bytes()
    with pytest.raises(FileExistsError):
        subject.open_performance_lock(fake_open_inputs)
    assert Path(fake_open_inputs.global_claim_path).read_bytes() == claim_bytes
    assert not fake_open_inputs.lock_output_directory.exists()


def test_claim_replay_rejects_input_drift_and_non_global_path(
    fake_open_inputs: subject.PerformanceLockInputs,
) -> None:
    subject.open_performance_lock(fake_open_inputs)
    assert (
        subject.validate_open_claim(fake_open_inputs)["schema"] == subject.CLAIM_SCHEMA
    )

    Path(fake_open_inputs.startup_source).write_bytes(b"changed startup\n")
    with pytest.raises(ValueError, match="identity changed"):
        subject.validate_global_claim(fake_open_inputs)
    assert not fake_open_inputs.lock_output_directory.exists()

    wrong_path = replace(
        fake_open_inputs,
        global_claim_path=Path(fake_open_inputs.global_claim_path).with_name(
            "run-scoped-claim.json"
        ),
    )
    with pytest.raises(ValueError, match="fixed global path"):
        subject._claim_payload(wrong_path, opened_unix_ns=1)


def _materialization_claim() -> dict[str, Any]:
    contract = runner.build_run_contract(
        candidate_library_sha256=subject.CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=subject.REFERENCE_LIBRARY_SHA256,
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
    )
    return {
        "precontent_plan": {"sha256": "a" * 64},
        "lock_run_contract": contract,
        "lock_run_contract_digest": runner.canonical_sha256(contract),
    }


def test_materialize_validates_claim_first_and_resumes_same_100_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "lock-output"
    claim = _materialization_claim()
    inputs = subject.PerformanceLockInputs(
        repository_root=tmp_path,
        plan_path=tmp_path / "plan.json",
        lock_output_directory=output,
        candidate_library=tmp_path / "candidate",
        reference_library=tmp_path / "reference",
        feature_encoder=tmp_path / "feature",
        startup_source=tmp_path / "startup",
    )
    validation_calls = 0

    def validate_claim(_inputs: subject.PerformanceLockInputs) -> dict[str, Any]:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            assert not output.exists()
        return claim

    monkeypatch.setattr(subject, "validate_global_claim", validate_claim)
    monkeypatch.setattr(
        runner,
        "_validate_root_artifact",
        lambda _contract, _value, *, index: (f"first-{index}", f"second-{index}"),
    )
    crash_after = {"count": 25}

    def materialize(
        *,
        contract: dict[str, Any],
        repository_root: Path,
        output_dir: Path,
        indices: tuple[int, ...],
    ) -> list[dict[str, Any]]:
        assert contract == claim["lock_run_contract"]
        assert repository_root == tmp_path.resolve()
        assert tuple(indices) == runner.CONTRACT_HAND_INDICES
        root_dir = output_dir / "roots"
        root_dir.mkdir(parents=True, exist_ok=True)
        roots: list[dict[str, Any]] = []
        created = 0
        for index in indices:
            path = root_dir / f"hand_{index:03d}.json"
            if path.exists():
                value = _read_json(path)
            else:
                value = {"hand_index": index, "deterministic_seed": 700_000 + index}
                _write_canonical(path, value)
                created += 1
                if crash_after["count"] and created == crash_after["count"]:
                    raise RuntimeError("simulated materialization crash")
            roots.append(value)
        return roots

    monkeypatch.setattr(runner, "_materialize_roots", materialize)
    with pytest.raises(RuntimeError, match="simulated"):
        subject.materialize_performance_lock(inputs)
    partial = sorted((output / "roots").glob("hand_*.json"))
    assert len(partial) == 25
    partial_bytes = {path.name: path.read_bytes() for path in partial}
    assert not (output / "materialization.json").exists()

    crash_after["count"] = 0
    materialization = subject.materialize_performance_lock(inputs)
    assert materialization["root_count"] == 100
    assert materialization["reseeded"] is False
    assert materialization["same_identity_resume_only"] is True
    assert len(materialization["root_artifact_sha256"]) == 100
    assert {path.name: path.read_bytes() for path in partial} == partial_bytes

    artifact = output / "materialization.json"
    before = artifact.read_bytes()
    before_mtime = artifact.stat().st_mtime_ns
    assert subject.materialize_performance_lock(inputs) == materialization
    assert artifact.read_bytes() == before
    assert artifact.stat().st_mtime_ns == before_mtime
    assert validation_calls == 3


def test_materialize_rejects_claim_before_touching_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "must-not-exist"
    inputs = subject.PerformanceLockInputs(
        repository_root=tmp_path,
        plan_path=tmp_path / "plan",
        lock_output_directory=output,
        candidate_library=tmp_path / "candidate",
        reference_library=tmp_path / "reference",
        feature_encoder=tmp_path / "feature",
        startup_source=tmp_path / "startup",
    )

    def reject(_inputs: subject.PerformanceLockInputs) -> dict[str, Any]:
        raise ValueError("claim rejected")

    monkeypatch.setattr(subject, "validate_global_claim", reject)
    with pytest.raises(ValueError, match="claim rejected"):
        subject.materialize_performance_lock(inputs)
    assert not output.exists()


@dataclass(frozen=True)
class _FakeObservation:
    index: int
    seat: str

    @property
    def hero_board(self) -> str:
        return f"hero-{self.index}-{self.seat}"

    @property
    def opponent_public_board(self) -> str:
        return f"opponent-{self.index}-{self.seat}"

    @property
    def dealt_cards(self) -> str:
        return f"dealt-{self.index}-{self.seat}"

    def fingerprint(self) -> str:
        return f"lock-{self.seat}-{self.index}"


def _lock_roots() -> list[dict[str, Any]]:
    profiles = tuple(M31_T3_BEHAVIOR_PROFILES)
    return [
        {
            "hand_index": index,
            "profile": profiles[index % len(profiles)],
            "seeds": {
                "hand": 800_000 + index * 3,
                "behavior": 800_001 + index * 3,
                "future": 800_002 + index * 3,
            },
            "observations": [
                {"observation_fingerprint": f"raw-lock-first-{index}"},
                {"observation_fingerprint": f"raw-lock-second-{index}"},
            ],
        }
        for index in runner.CONTRACT_HAND_INDICES
    ]


def _development_roots() -> list[dict[str, Any]]:
    return [
        {
            "development_index": index,
            "seeds": {
                "hand": 900_000 + index * 3,
                "behavior": 900_001 + index * 3,
                "future": 900_002 + index * 3,
            },
            "observations": [
                {"observation_fingerprint": f"development-first-{index}"},
                {"observation_fingerprint": f"development-second-{index}"},
            ],
        }
        for index in runner.CONTRACT_HAND_INDICES
    ]


def _prepare_seal_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[subject.PerformanceLockInputs, list[dict[str, Any]], list[dict[str, Any]]]:
    output = tmp_path / "lock-output"
    roots = _lock_roots()
    for index, root in enumerate(roots):
        _write_canonical(output / "roots" / f"hand_{index:03d}.json", root)
    claim = _materialization_claim()
    _write_canonical(
        output / "materialization.json",
        subject._materialization_payload(claim, roots),
    )
    inputs = subject.PerformanceLockInputs(
        repository_root=tmp_path,
        plan_path=tmp_path / "plan",
        lock_output_directory=output,
        candidate_library=tmp_path / "candidate",
        reference_library=tmp_path / "reference",
        feature_encoder=tmp_path / "feature",
        startup_source=tmp_path / "startup",
        development_root_directory=tmp_path / "development-roots",
    )
    monkeypatch.setattr(subject, "validate_global_claim", lambda _inputs: claim)
    validator_calls: list[int] = []

    def validate_root(
        _contract: dict[str, Any], value: dict[str, Any], *, index: int
    ) -> tuple[_FakeObservation, _FakeObservation]:
        assert value["hand_index"] == index
        validator_calls.append(index)
        return _FakeObservation(index, "first"), _FakeObservation(index, "second")

    monkeypatch.setattr(runner, "_validate_root_artifact", validate_root)
    monkeypatch.setattr(subject, "generate_turn_actions", lambda *_args: range(21))
    development = _development_roots()
    monkeypatch.setattr(
        subject.development_roots,
        "load_frozen_roots",
        lambda _path: development,
    )
    return inputs, roots, development


def test_seal_replays_all_roots_and_is_nonwriting_when_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, roots, _development = _prepare_seal_fixture(tmp_path, monkeypatch)
    validator_calls: list[int] = []

    def validate_root(
        _contract: dict[str, Any], value: dict[str, Any], *, index: int
    ) -> tuple[_FakeObservation, _FakeObservation]:
        assert value == roots[index]
        validator_calls.append(index)
        return _FakeObservation(index, "first"), _FakeObservation(index, "second")

    monkeypatch.setattr(runner, "_validate_root_artifact", validate_root)
    seal = subject.seal_performance_lock(inputs)
    assert validator_calls == list(runner.CONTRACT_HAND_INDICES)
    assert seal["root_count"] == 100
    assert seal["observation_count"] == 200
    assert seal["profile_counts"] == {
        profile: 20 for profile in M31_T3_BEHAVIOR_PROFILES
    }
    assert seal["seat_counts"] == {"first": 100, "second": 100}
    assert seal["development_comparison"]["lock_fingerprint_overlap_count"] == 0
    assert seal["development_comparison"]["lock_root_hash_overlap_count"] == 0
    assert seal["development_comparison"]["lock_seed_overlap_count"] == 0
    assert seal["visibility"]["runner_validator_replayed_all_roots"] is True
    assert seal["visibility"]["opponent_private_discards_used"] is False
    assert seal["selection_inputs"] == {
        "timing_used": False,
        "q_used": False,
        "ev_used": False,
        "all_100_preregistered_hands_used": True,
    }
    assert seal["training_eligible"] is False
    assert seal["current_profile_changed"] is False

    artifact = Path(inputs.lock_output_directory) / "seal.json"
    before = artifact.read_bytes()
    before_mtime = artifact.stat().st_mtime_ns
    assert subject.validate_root_seal(inputs) == seal
    assert artifact.read_bytes() == before
    assert artifact.stat().st_mtime_ns == before_mtime
    assert validator_calls == list(runner.CONTRACT_HAND_INDICES) * 2


def test_seal_rejects_fingerprint_seed_and_root_hash_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, roots, development = _prepare_seal_fixture(tmp_path, monkeypatch)

    development[0]["observations"][0]["observation_fingerprint"] = "lock-first-0"
    with pytest.raises(ValueError, match="overlap development"):
        subject._build_root_seal(inputs)

    development[0] = _development_roots()[0]
    development[0]["seeds"]["hand"] = roots[0]["seeds"]["hand"]
    with pytest.raises(ValueError, match="overlap development"):
        subject._build_root_seal(inputs)

    development[0] = roots[0]
    with pytest.raises(ValueError, match="overlap development"):
        subject._build_root_seal(inputs)


def test_seal_replay_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, _roots, _development = _prepare_seal_fixture(tmp_path, monkeypatch)
    subject.seal_performance_lock(inputs)
    path = Path(inputs.lock_output_directory) / "seal.json"
    tampered = _read_json(path)
    tampered["training_eligible"] = True
    _write_canonical(path, tampered)
    with pytest.raises(ValueError, match="replay changed"):
        subject.validate_root_seal(inputs)
