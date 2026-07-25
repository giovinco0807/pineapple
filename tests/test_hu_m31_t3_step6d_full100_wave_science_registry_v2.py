from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave
from ofc_regular import hu_m31_t3_step6d_full100_wave_science_registry_v2 as subject


RUN_NAME = "regular-hu-m31-c02-f100wv2-20260722-003"
SALT = "0123456789abcdef0123456789abcdef"
PACKAGE_SHA = "2" * 64
IMAGE_DIGEST = "sha256:" + "3" * 64


def _development_plan() -> dict:
    return wave.full100.validate_full100_plan(
        json.loads(wave.DEFAULT_FULL100_PLAN_PATH.read_text("utf-8"))
    )


def _build_development_wave() -> dict:
    return wave.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt=SALT,
        package_sha256=PACKAGE_SHA,
        image_digest=IMAGE_DIGEST,
    )


def test_development_descriptor_and_serialized_wave_remain_bit_exact() -> None:
    descriptor = subject.descriptor_for_kind(subject.DEVELOPMENT_SCIENCE_KIND)
    frozen = _development_plan()
    assert subject.descriptor_for_plan(frozen) is descriptor
    assert descriptor.validate_plan(frozen) == frozen
    assert descriptor.plan_sha256 == wave.full100.FULL100_PLAN_SHA256
    assert descriptor.run_contract_digest == wave.full100.FULL_RUN_CONTRACT_DIGEST
    assert descriptor.execution_scope == wave.FULL100_EXECUTION_SCOPE
    assert descriptor.package_schema == subject.DEVELOPMENT_PACKAGE_SCHEMA
    assert descriptor.package_source_name == subject.DEVELOPMENT_PACKAGE_SOURCE_NAME
    startup = descriptor.resolved_startup_path()
    assert startup.as_posix().endswith(
        subject.DEVELOPMENT_STARTUP_RELATIVE_PATH
    )
    assert startup.stat().st_size == 37217
    assert hashlib.sha256(startup.read_bytes()).hexdigest() == (
        subject.DEVELOPMENT_STARTUP_SHA256
    )
    assert descriptor.package_canonical_bytes({"schema": "probe"}) == (
        b'{"schema":"probe"}\n'
    )

    plan = _build_development_wave()
    raw = wave.canonical_bytes(plan)
    assert len(raw) == 22635
    assert hashlib.sha256(raw).hexdigest() == (
        "80dfa56e7341b6e2935f64dc596b869af78d48776bdca8a25cb31976b3ba3ecb"
    )
    assert plan["execution_identity_sha256"] == (
        "18885a46e2a5125be972d9fb5b250209e344fa02f6597068ab6ddb5d8284d6f8"
    )
    assert plan["schedule_sha256"] == (
        "8b6dea07202290ae41293ecd2c3e76307a5298848ad4eee3133d9565ce97c8c5"
    )


def test_unknown_schema_fails_before_import_or_validator_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped = _build_development_wave()

    def forbidden_import(name: str) -> object:
        raise AssertionError(f"unexpected import for unknown schema: {name}")

    monkeypatch.setattr(subject.importlib, "import_module", forbidden_import)
    unknown = deepcopy(_development_plan())
    unknown["schema"] = "hu_m31_t3_step6d_unknown_science_plan_v999"
    with pytest.raises(ValueError, match="unsupported scientific plan schema"):
        subject.validate_scientific_plan(unknown)

    wrapped["full100_plan"] = unknown
    with pytest.raises(ValueError, match="unsupported scientific plan schema"):
        wave.validate_wave_plan(wrapped)


def test_performance_lock_v4_descriptor_binds_the_frozen_module_interface() -> None:
    descriptor = subject.descriptor_for_kind(
        subject.PERFORMANCE_LOCK_V4_SCIENCE_KIND
    )
    assert descriptor.plan_schema == subject.PERFORMANCE_LOCK_V4_PLAN_SCHEMA
    assert descriptor.plan_scope == subject.PERFORMANCE_LOCK_V4_PLAN_SCOPE
    assert descriptor.execution_scope == wave.PERFORMANCE_LOCK_V4_EXECUTION_SCOPE
    assert descriptor.plan_sha256 == (
        "2ad08116835a58f5b5927e4de986f2717915d0dd128e7a5f3fa288e0cac6e5be"
    )
    assert descriptor.run_contract_digest == (
        "669c1efa1afeebe41fcc531c6458c9d72fffdd5df2cca751c99988a872f3e2b6"
    )
    assert descriptor.resolved_default_plan_path().name == (
        "performance_lock_v4_plan.json"
    )
    assert descriptor.startup_canary_allowed is False
    assert descriptor.legacy_development_identity is False
    assert descriptor.package_schema == (
        subject.PERFORMANCE_LOCK_V4_PACKAGE_SCHEMA
    )
    assert descriptor.package_source_name == (
        subject.PERFORMANCE_LOCK_V4_PACKAGE_SOURCE_NAME
    )
    startup = descriptor.resolved_startup_path()
    assert startup.as_posix().endswith(
        subject.PERFORMANCE_LOCK_V4_STARTUP_RELATIVE_PATH
    )
    assert startup.stat().st_size == 35205
    assert hashlib.sha256(startup.read_bytes()).hexdigest() == (
        subject.PERFORMANCE_LOCK_V4_STARTUP_SHA256
    )
    assert descriptor.package_canonical_bytes({"schema": "probe"}) == (
        b'{"schema":"probe"}\n'
    )


def test_performance_lock_v4_uses_separate_identity_and_strict_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = _development_plan()
    lock_plan = deepcopy(development)
    lock_plan.update(
        {
            "schema": subject.PERFORMANCE_LOCK_V4_PLAN_SCHEMA,
            "scope": subject.PERFORMANCE_LOCK_V4_PLAN_SCOPE,
            "status": subject.PERFORMANCE_LOCK_V4_WAVE_STATUS,
            "decision": subject.PERFORMANCE_LOCK_V4_WAVE_DECISION,
        }
    )
    plan_sha256 = hashlib.sha256(wave.canonical_bytes(lock_plan)).hexdigest()
    run_contract_digest = wave.canonical_sha256(lock_plan["run_contract"])
    fake_module = SimpleNamespace(
        PLAN_SCHEMA=subject.PERFORMANCE_LOCK_V4_PLAN_SCHEMA,
        PLAN_SCOPE=subject.PERFORMANCE_LOCK_V4_PLAN_SCOPE,
        PLAN_SHA256=plan_sha256,
        RUN_CONTRACT_DIGEST=run_contract_digest,
        DEFAULT_PLAN_PATH="unused-in-this-test.json",
        validate_performance_lock_v4_plan=lambda value: deepcopy(dict(value)),
    )
    real_import = subject.importlib.import_module

    def import_science(name: str) -> object:
        if name.endswith("hu_m31_t3_step6d_candidate02_performance_lock_v4_plan"):
            return fake_module
        return real_import(name)

    monkeypatch.setattr(subject.importlib, "import_module", import_science)
    lock_wave = wave.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt=SALT,
        package_sha256=PACKAGE_SHA,
        image_digest=IMAGE_DIGEST,
        full100_plan=lock_plan,
        execution_scope=wave.PERFORMANCE_LOCK_V4_EXECUTION_SCOPE,
    )
    development_wave = _build_development_wave()

    assert wave.validate_wave_plan(lock_wave) == lock_wave
    assert lock_wave["scope"] == subject.PERFORMANCE_LOCK_V4_EXECUTION_SCOPE
    assert lock_wave["status"] == subject.PERFORMANCE_LOCK_V4_WAVE_STATUS
    assert lock_wave["decision"] == subject.PERFORMANCE_LOCK_V4_WAVE_DECISION
    assert lock_wave["full100_plan_sha256"] == plan_sha256
    assert lock_wave["run_contract_digest"] == run_contract_digest
    assert (
        lock_wave["execution_identity_sha256"]
        != development_wave["execution_identity_sha256"]
    )
    assert lock_wave["schedule_sha256"] != development_wave["schedule_sha256"]
    observed = wave.build_observed_transition(
        lock_wave,
        project_id="ofc-project-123",
        zone="asia-northeast1-b",
        observed_at_utc="2026-07-23T03:00:00Z",
        previous_transition_digest=None,
        attempt_history=wave.empty_attempt_history(lock_wave),
    )
    ledger = wave.build_attempt_ledger(
        lock_wave,
        transitions=[observed],
        consumed_transition_digests=[],
    )
    resume = wave.build_resume_plan(lock_wave, attempt_ledger=ledger)
    assert resume["resume_wave_index"] == 0
    assert len(resume["selected_attempts"]) == 8
    assert {row["attempt_id"] for row in resume["selected_attempts"]} == {"a00"}

    with pytest.raises(ValueError, match="scientific plan and execution scope"):
        wave.build_wave_plan(
            run_name=RUN_NAME,
            identity_salt=SALT,
            package_sha256=PACKAGE_SHA,
            image_digest=IMAGE_DIGEST,
            full100_plan=lock_plan,
            execution_scope=wave.FULL100_EXECUTION_SCOPE,
        )
    with pytest.raises(ValueError, match="scientific plan and execution scope"):
        wave.build_wave_plan(
            run_name=RUN_NAME,
            identity_salt=SALT,
            package_sha256=PACKAGE_SHA,
            image_digest=IMAGE_DIGEST,
            full100_plan=development,
            execution_scope=wave.PERFORMANCE_LOCK_V4_EXECUTION_SCOPE,
        )
    with pytest.raises(ValueError, match="scientific plan and execution scope"):
        wave.build_wave_plan(
            run_name=RUN_NAME,
            identity_salt=SALT,
            package_sha256=PACKAGE_SHA,
            image_digest=IMAGE_DIGEST,
            full100_plan=lock_plan,
            execution_scope=wave.STARTUP_CANARY_SCOPE,
        )


def test_startup_hash_dispatch_is_exact_for_dev_and_v4() -> None:
    development = _build_development_wave()
    v4_plan = json.loads(
        subject.descriptor_for_kind(
            subject.PERFORMANCE_LOCK_V4_SCIENCE_KIND
        ).resolved_default_plan_path().read_text("utf-8")
    )
    lock = wave.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt=SALT,
        package_sha256=PACKAGE_SHA,
        image_digest=IMAGE_DIGEST,
        full100_plan=v4_plan,
    )
    assert (
        subject.resolve_startup_sha256(development)
        == subject.DEVELOPMENT_STARTUP_SHA256
    )
    assert (
        subject.resolve_startup_sha256(lock)
        == subject.PERFORMANCE_LOCK_V4_STARTUP_SHA256
    )
    assert subject.descriptor_for_wave_plan(lock).science_kind == (
        subject.PERFORMANCE_LOCK_V4_SCIENCE_KIND
    )
    with pytest.raises(ValueError, match="does not match scientific plan"):
        subject.resolve_startup_sha256(
            lock, subject.DEVELOPMENT_STARTUP_SHA256
        )
    with pytest.raises(ValueError, match="does not match scientific plan"):
        subject.resolve_startup_sha256(
            development, subject.PERFORMANCE_LOCK_V4_STARTUP_SHA256
        )
