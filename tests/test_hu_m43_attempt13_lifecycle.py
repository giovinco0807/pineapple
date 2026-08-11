from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import pytest

import ofc_regular.freeze_hu_m43_attempt09_development as freeze_engine
import ofc_regular.freeze_hu_m43_attempt13_development as freeze13
import ofc_regular.hu_m43_attempt13_spot as spot
import ofc_regular.select_hu_m43_attempt09_development as selector_engine
import ofc_regular.select_hu_m43_attempt13_audit50 as audit13
import ofc_regular.select_hu_m43_attempt13_development as selector13
from ofc_regular import run_hu_m43_attempt09 as runner_engine
from ofc_regular.hu_m43_attempt13_contract import (
    M43_ATTEMPT13_PLAN_SHA256,
    M43_ATTEMPT13_PROFILES,
    enumerate_attempt13_seed_schedules,
    load_and_validate_attempt13_plan,
    validate_attempt13_artifact_bindings,
)
from ofc_regular.run_hu_m43_attempt13 import (
    ATTEMPT13_AUTHORIZATION_SCHEMA,
    ATTEMPT13_ROW_SCHEMA,
    _attempt13_bindings,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt13.json"
_HASH = "a" * 64


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _plan() -> dict[str, Any]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def _validated_row(
    _row: Mapping[str, Any],
    root: int,
    seeds: Mapping[str, int],
    **_kwargs: Any,
) -> dict[str, Any]:
    fired = root < (40 if root < 200 else 210)
    profile = M43_ATTEMPT13_PROFILES[root % 5]
    return {
        "root_index": root,
        "profile": profile,
        "hand_seed": seeds["hand"],
        "observation_fingerprint": f"observation-{root}",
        "baseline_action_key": f"baseline-{root}",
        "selected_action_key": f"selected-{root}" if fired else f"baseline-{root}",
        "fired": fired,
        "mean": 1.0 if fired else 0.0,
        "loss": (
            {"p95": 25.0, "p99": 40.0, "max": 50.0}
            if fired
            else {"p95": 0.0, "p99": 0.0, "max": 0.0}
        ),
        "rng_digests": [f"rng-{root}"],
        "belief_digests": [f"belief-{root}"],
        "config_sha256": f"{root:064x}",
    }


def _phase_order_row(order: tuple[str, ...]) -> dict[str, Any]:
    return {
        "teacher": {
            "rng_key_digests": {phase: [f"rng-{phase}"] for phase in order},
            "belief_digests": {phase: f"belief-{phase}" for phase in order},
            "phase_child_information_set_counts": {
                phase: 1 for phase in order
            },
        }
    }


def test_attempt13_fresh_seed_schedules_and_three_package_modes() -> None:
    plan = load_and_validate_attempt13_plan(PLAN)
    schedules: dict[str, tuple[int, ...]] = {}
    for population, count in (
        ("preflight", 5),
        ("development", 200),
        ("future_audit", 50),
    ):
        values = enumerate_attempt13_seed_schedules(plan, population=population)
        assert all(len(schedule) == count for schedule in values.values())
        schedules.update(
            {f"{population}.{domain}": sequence for domain, sequence in values.items()}
        )
    for left, right in combinations(schedules, 2):
        assert set(schedules[left]).isdisjoint(schedules[right])
    assert schedules["development.hand"][0] == 230_108_071_901
    assert schedules["preflight.hand"][0] == 240_108_071_901

    preflight = spot.build_schedule("preflight", "attempt13-preflight-test")
    assert [row["root_index"] for row in preflight] == [0, 0, 0, 1, 2, 3, 4]
    assert [row["batch_child_selectors"] for row in preflight] == [
        True,
        True,
        False,
        True,
        True,
        True,
        True,
    ]
    assert {row["root_profile"] for row in preflight} == set(M43_ATTEMPT13_PROFILES)
    assert len(spot.build_schedule("development", "attempt13-development-test")) == 200
    assert len(spot.build_schedule("future_audit", "attempt13-audit-test")) == 50


def test_attempt13_runner_selector_audit_and_freeze_bindings_are_scoped() -> None:
    old_runner_plan = runner_engine.M43_ATTEMPT09_PLAN_SHA256
    old_runner_row = runner_engine.ATTEMPT09_ROW_SCHEMA
    with _attempt13_bindings():
        assert runner_engine.M43_ATTEMPT09_PLAN_SHA256 == M43_ATTEMPT13_PLAN_SHA256
        assert runner_engine.ATTEMPT09_ROW_SCHEMA == ATTEMPT13_ROW_SCHEMA
    assert runner_engine.M43_ATTEMPT09_PLAN_SHA256 == old_runner_plan
    assert runner_engine.ATTEMPT09_ROW_SCHEMA == old_runner_row

    old_selector_plan = selector_engine.M43_ATTEMPT09_PLAN_SHA256
    old_selector_validator = selector_engine._validate_row
    old_attempt12_aggregate = selector13._base._aggregate_attempt12_rows
    with selector13._attempt13_selector_bindings():
        assert selector_engine.M43_ATTEMPT09_PLAN_SHA256 == M43_ATTEMPT13_PLAN_SHA256
        assert selector_engine.EVALUATION_SAMPLE_COUNT == 512
        assert "extreme_tail_statistic" in selector13._INTEGRITY_NAMES
        assert selector_engine._validate_row is old_selector_validator
        assert (
            selector13._base._aggregate_attempt12_rows
            is selector13._aggregate_attempt13_rows
        )
    assert selector_engine.M43_ATTEMPT09_PLAN_SHA256 == old_selector_plan
    assert selector_engine._validate_row is old_selector_validator
    assert selector13._base._aggregate_attempt12_rows is old_attempt12_aggregate

    old_population = selector_engine.POPULATION
    with audit13._attempt13_audit_bindings():
        assert selector_engine.POPULATION == "future_audit"
        assert selector_engine.ROOT_INDEX_FIRST == 200
        assert selector_engine._ROOTS == 50
    assert selector_engine.POPULATION == old_population

    old_freeze_plan = freeze_engine.M43_ATTEMPT09_PLAN_SHA256
    with freeze13._attempt13_freeze_bindings():
        assert freeze_engine.M43_ATTEMPT09_PLAN_SHA256 == M43_ATTEMPT13_PLAN_SHA256
        assert (
            freeze_engine.DEVELOPMENT_GO_FREEZE_STATUS
            == "go_freeze_attempt13_development"
        )
    assert freeze_engine.M43_ATTEMPT09_PLAN_SHA256 == old_freeze_plan


@pytest.mark.parametrize(
    "phase_order",
    [
        (
            "rerank_r128",
            "veto_v256",
            "stress_x1024",
            "confirmation_c1024",
            "evaluation_e512",
        ),
        (
            "confirmation_c1024",
            "evaluation_e512",
            "rerank_r128",
            "stress_x1024",
            "veto_v256",
        ),
    ],
    ids=("logical", "canonical-json"),
)
def test_attempt13_selector_accepts_only_valid_serialized_phase_orders(
    monkeypatch: pytest.MonkeyPatch, phase_order: tuple[str, ...]
) -> None:
    logical_order = list(selector13._ATTEMPT13_OPEN_PHASE_ORDER)
    validated_orders: list[list[str]] = []

    def _capture_normalized_row(
        row: Mapping[str, Any],
        root: int,
        seeds: Mapping[str, int],
        **kwargs: Any,
    ) -> dict[str, Any]:
        teacher = row["teacher"]
        for field in selector13._ATTEMPT13_PHASE_MAPPING_FIELDS:
            assert list(teacher[field]) == logical_order
        validated_orders.append(list(teacher["rng_key_digests"]))
        return _validated_row(row, root, seeds, **kwargs)

    monkeypatch.setattr(selector_engine, "_validate_row", _capture_normalized_row)
    result = selector13.aggregate_attempt13_development_rows(
        [_phase_order_row(phase_order) for _ in range(200)],
        plan=_plan(),
        source_input_sha256=_HASH,
        source_plan_sha256=M43_ATTEMPT13_PLAN_SHA256,
        authorization_sha256="b" * 64,
        source_package_sha256="c" * 64,
        run_name="attempt13-phase-order-regression",
    )

    assert result["decision"] == "go"
    assert validated_orders == [logical_order] * 200


def test_attempt13_selector_rejects_arbitrary_phase_permutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arbitrary_order = (
        "veto_v256",
        "rerank_r128",
        "stress_x1024",
        "confirmation_c1024",
        "evaluation_e512",
    )

    def _must_not_validate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("arbitrary phase order reached inherited validator")

    monkeypatch.setattr(selector_engine, "_validate_row", _must_not_validate)
    with pytest.raises(
        ValueError, match="Attempt13 opened RNG phase order changed at root 0"
    ):
        selector13.aggregate_attempt13_development_rows(
            [_phase_order_row(arbitrary_order) for _ in range(200)],
            plan=_plan(),
            source_input_sha256=_HASH,
            source_plan_sha256=M43_ATTEMPT13_PLAN_SHA256,
            authorization_sha256="b" * 64,
            source_package_sha256="c" * 64,
            run_name="attempt13-arbitrary-phase-order",
        )


def test_attempt13_development_and_audit_use_frozen_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(selector_engine, "_validate_row", _validated_row)
    development = selector13.aggregate_attempt13_development_rows(
        [{} for _ in range(200)],
        plan=_plan(),
        source_input_sha256=_HASH,
        source_plan_sha256=M43_ATTEMPT13_PLAN_SHA256,
        authorization_sha256="b" * 64,
        source_package_sha256="c" * 64,
        run_name="attempt13-development-synthetic",
    )
    gates = {gate["name"]: gate for gate in development["gates"]}
    assert development["decision"] == "go"
    assert development["metrics"]["overall"]["fires"] == 40
    assert gates["fires_total"]["requirement"] == ">= 40"
    assert gates["fires_each_profile"]["requirement"] == "each >= 3"
    assert gates["maximum_per_fired_root_max_loss"]["observed"] == 50
    assert gates["extreme_tail_statistic_violation_count"]["observed"] == 0
    assert development["selected_threshold"] is None

    audit = audit13.aggregate_attempt13_audit50_rows(
        [{} for _ in range(50)],
        plan=_plan(),
        source_input_sha256=_HASH,
        source_plan_sha256=M43_ATTEMPT13_PLAN_SHA256,
        authorization_sha256="b" * 64,
        source_package_sha256="c" * 64,
        run_name="attempt13-audit-synthetic",
    )
    audit_gates = {gate["name"]: gate for gate in audit["gates"]}
    assert audit["decision"] == "go"
    assert audit["metrics"]["overall"]["fires"] == 10
    assert audit_gates["fires_total"]["requirement"] == ">= 10"
    assert audit_gates["fires_each_profile"]["requirement"] == "each >= 1"


def test_attempt13_audit_pure_selector_recomputes_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def _select(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        assert selector_engine.POPULATION == "future_audit"
        assert selector_engine.ROOT_INDEX_FIRST == 200
        return {"decision": "go", "read_only": True}

    old_population = selector_engine.POPULATION
    monkeypatch.setattr(selector_engine, "select_attempt09_development", _select)
    result = audit13.select_attempt13_audit50(
        input_path=tmp_path / "teacher.jsonl",
        plan_path=tmp_path / "plan.json",
        authorization_path=tmp_path / "authorization.json",
        source_package_sha256="a" * 64,
        run_name="attempt13-audit-pure",
    )

    assert result == {"decision": "go", "read_only": True}
    assert captured["run_name"] == "attempt13-audit-pure"
    assert selector_engine.POPULATION == old_population
    assert list(tmp_path.iterdir()) == []


def test_attempt13_development_selector_is_write_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(selector_engine, "_validate_row", _validated_row)
    input_path = tmp_path / "teacher.jsonl"
    input_path.write_bytes(b"".join(_canonical({}) for _ in range(200)))
    package = tmp_path / "run" / "package_src"
    gate = package / "artifacts" / "attempt13" / "preceding_gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_bytes(
        _canonical(
            {
                "schema": "hu_m43_attempt13_preflight_result_v1",
                "status": "pass_correctness_preflight",
                "decision": "authorize_development200_package_only",
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
            }
        )
    )
    authorization = tmp_path / "run" / "execution_authorization.json"
    authorization.write_bytes(
        _canonical(
            {
                "schema": ATTEMPT13_AUTHORIZATION_SCHEMA,
                "status": "authorized",
                "mode": "development",
                "plan_sha256": M43_ATTEMPT13_PLAN_SHA256,
                "source_package_sha256": "c" * 64,
                "preceding_gate_artifact": "artifacts/attempt13/preceding_gate.json",
                "preceding_gate_sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
                "root_index_first": 0,
                "root_index_last": 199,
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
            }
        )
    )
    output = tmp_path / "selector"
    receipt = selector13.execute_attempt13_development_selector(
        input_path=input_path,
        plan_path=PLAN,
        authorization_path=authorization,
        source_package_sha256="c" * 64,
        run_name="attempt13-development-write-once",
        output_dir=output,
    )
    assert receipt["schema"] == selector13.ATTEMPT13_DEVELOPMENT_RECEIPT_SCHEMA
    assert receipt["gate_evaluation_count"] == 1
    with pytest.raises(FileExistsError):
        selector13.execute_attempt13_development_selector(
            input_path=input_path,
            plan_path=PLAN,
            authorization_path=authorization,
            source_package_sha256="c" * 64,
            run_name="attempt13-development-write-once",
            output_dir=output,
        )


def test_attempt13_spot_package_rejects_attempt12_closeout(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    template = tmp_path / "template"
    template.mkdir()
    (template / "source_closure_manifest.json").write_bytes(
        spot.canonical_json_bytes({"schema": "minimal_template_v1", "status": "frozen"})
    )
    for relative in spot.OVERLAY_RELATIVES:
        source = ROOT / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    plan = _plan()
    for relative in (
        plan["search_protocol"]["candidate_generator_artifact_path"],
        plan["baseline_hash_audit"]["policy_registry_path"],
    ):
        source = ROOT / relative
        for base in (repository, template):
            destination = base / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    startup = repository / "scripts" / spot.STARTUP_NAME
    startup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts" / spot.STARTUP_NAME, startup)
    gate = repository / "gates" / "preflight.json"
    gate.parent.mkdir(parents=True)
    gate.write_bytes(
        spot.canonical_json_bytes(
            {
                "schema": "hu_m43_attempt13_local_correctness_v1",
                "status": "pass_local_correctness",
                "decision": "authorize_preflight_only",
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
            }
        )
    )
    run_name = "attempt13-preflight-package-test"
    run_dir = repository / "outputs" / "gcp_runs" / run_name
    manifest = spot.package_attempt13(
        mode="preflight",
        run_name=run_name,
        run_dir=run_dir,
        repository_root=repository,
        template_package=template,
        plan=repository / spot.PLAN_RELATIVE,
        startup=startup,
        preceding_gate=gate,
    )
    assert manifest == spot.validate_package(run_dir)
    assert manifest["total_shards"] == 7
    assert manifest["plan_sha256"] == M43_ATTEMPT13_PLAN_SHA256
    assert not (run_dir / "execution_authorization.json").exists()
    unpacked = tmp_path / "unpacked-source"
    with zipfile.ZipFile(run_dir / spot.SOURCE_NAME) as archive:
        archive.extractall(unpacked)
    validate_attempt13_artifact_bindings(_plan(), repository_root=unpacked)

    closeout = repository / "configs" / "hu_joint_policy_m43_attempt12_closeout.json"
    canonical_closeout = repository / "gates" / "attempt12-closeout.json"
    canonical_closeout.write_bytes(
        spot.canonical_json_bytes(json.loads(closeout.read_text(encoding="utf-8")))
    )
    rejected_name = "attempt13-closeout-must-not-authorize"
    with pytest.raises(ValueError, match="preceding gate did not pass"):
        spot.package_attempt13(
            mode="preflight",
            run_name=rejected_name,
            run_dir=repository / "outputs" / "gcp_runs" / rejected_name,
            repository_root=repository,
            template_package=template,
            plan=repository / spot.PLAN_RELATIVE,
            startup=startup,
            preceding_gate=canonical_closeout,
        )


def test_attempt13_startup_script_identity_and_bash_syntax() -> None:
    relative = Path("scripts") / spot.STARTUP_NAME
    completed = subprocess.run(
        ["bash", "-n", relative.as_posix()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    source = (ROOT / relative).read_text(encoding="utf-8")
    assert "ofc_regular.run_hu_m43_attempt13" in source
    assert "configs/hu_joint_policy_m43_attempt13.json" in source
    assert "hu_m43_attempt13_done_v1" in source
    assert "scalar child selectors are allowed only in Attempt13 preflight" in source
    assert "--if-generation-match=0" in source
