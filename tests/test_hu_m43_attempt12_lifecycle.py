from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import pytest

import ofc_regular.freeze_hu_m43_attempt09_development as freeze_base
import ofc_regular.freeze_hu_m43_attempt12_development as freeze12
import ofc_regular.hu_m43_attempt12_spot as spot
import ofc_regular.select_hu_m43_attempt09_development as selector_base
import ofc_regular.select_hu_m43_attempt12_audit50 as audit12
import ofc_regular.select_hu_m43_attempt12_development as selector12
from ofc_regular import run_hu_m43_attempt09 as runner_base
from ofc_regular.hu_m43_attempt12_contract import (
    ATTEMPT12_CANDIDATE_MAX,
    ATTEMPT12_CONFIRMATION_SAMPLES,
    ATTEMPT12_EVALUATION_SAMPLES,
    ATTEMPT12_POOLED_SAMPLES,
    M43_ATTEMPT12_PLAN_SHA256,
    M43_ATTEMPT12_PROFILES,
    enumerate_attempt12_seed_schedules,
    enumerate_known_seed_schedules,
    load_and_validate_attempt12_plan,
    validate_attempt12_artifact_bindings,
)
from ofc_regular.hu_m43_attempt12_spot import (
    _validate_root4_variable_candidate_teacher,
    build_schedule,
)
from ofc_regular.run_hu_m43_attempt12 import (
    ATTEMPT12_ROW_SCHEMA,
    _attempt12_bindings,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt12.json"


def test_attempt12_plan_architecture_hash_and_fresh_seed_schedules() -> None:
    plan = load_and_validate_attempt12_plan(PLAN)
    validate_attempt12_artifact_bindings(plan, repository_root=ROOT)
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == M43_ATTEMPT12_PLAN_SHA256
    assert ATTEMPT12_CANDIDATE_MAX == 26
    assert ATTEMPT12_CONFIRMATION_SAMPLES == 1024
    assert ATTEMPT12_POOLED_SAMPLES == 2048
    assert ATTEMPT12_EVALUATION_SAMPLES == 512
    assert plan["attempt11_boundary"]["status"] == "development_no_go_closed"
    assert plan["attempt11_boundary"]["all_attempt11_files_and_artifacts_preserved"]
    assert all(value is False for value in plan["activation_guards"].values())

    search = plan["search_protocol"]
    assert search["rerank"]["action_scope"].startswith("all_n_complete_unique_legal")
    assert set(search["veto"]["eligibility"]) == {
        "paired_delta_mean_strictly_greater_than",
        "paired_delta_p05_min",
        "paired_delta_p01_min",
    }
    assert search["stress"]["retention"].startswith("no_filter")
    assert search["confirmation"]["samples"] == 1024
    assert search["confirmation"]["pooled_decision"]["samples_per_action"] == 2048
    assert "paired_delta_min_min" not in search["confirmation"]["pooled_decision"]["eligibility"]
    assert search["evaluation"]["symbol"] == "E512"

    schedules: dict[str, tuple[int, ...]] = {}
    for population, count in (("development", 200), ("future_audit", 50)):
        values = enumerate_attempt12_seed_schedules(plan, population=population)
        assert all(len(schedule) == count for schedule in values.values())
        schedules.update({f"{population}.{key}": value for key, value in values.items()})
    preflight = enumerate_attempt12_seed_schedules(plan, population="preflight")
    assert all(len(schedule) == 5 for schedule in preflight.values())
    schedules.update({f"preflight.{key}": value for key, value in preflight.items()})
    for left, right in combinations(schedules, 2):
        assert set(schedules[left]).isdisjoint(schedules[right])
    assert schedules["development.hand"][0] == 200_108_071_901
    assert schedules["preflight.hand"] == tuple(
        210_108_071_901 + 1_000_003 * index for index in range(5)
    )
    known = enumerate_known_seed_schedules()
    for values in schedules.values():
        for prior in known.values():
            assert set(values).isdisjoint(prior)


def test_attempt12_seven_slot_preflight_and_variable_all_legal_guard() -> None:
    schedule = build_schedule("preflight", "attempt12-preflight-test")
    assert [row["root_index"] for row in schedule] == [0, 0, 0, 1, 2, 3, 4]
    assert [row["batch_child_selectors"] for row in schedule] == [
        True,
        True,
        False,
        True,
        True,
        True,
        True,
    ]
    assert {row["root_profile"] for row in schedule} == set(M43_ATTEMPT12_PROFILES)

    candidates = [f"a{index}" for index in range(26)]
    payload = {
        "candidate_nonbaseline_count": 26,
        "baseline_action_key": "baseline",
        "all_legal_nonbaseline_action_keys": candidates,
        "proposal_mapping": {
            "action_count": 27,
            "action_keys": [*candidates, "baseline"],
        },
        "rerank": {
            "action_count": 27,
            "action_keys": [*candidates, "baseline"],
        },
        "shortlist": {
            "nonbaseline_count": 8,
            "action_count": 8,
            "action_keys": candidates[:8],
        },
        "veto": {
            "action_count": 9,
            "action_keys": [*candidates[:8], "baseline"],
        },
    }
    assert _validate_root4_variable_candidate_teacher(payload) == 26
    payload["veto"]["action_keys"].insert(0, candidates[0])
    with pytest.raises(ValueError, match="mapping changed"):
        _validate_root4_variable_candidate_teacher(payload)


def test_attempt12_runner_selector_audit_and_freeze_bindings_are_scoped() -> None:
    old_runner_plan = runner_base.M43_ATTEMPT09_PLAN_SHA256
    old_runner_row = runner_base.ATTEMPT09_ROW_SCHEMA
    with _attempt12_bindings():
        assert runner_base.M43_ATTEMPT09_PLAN_SHA256 == M43_ATTEMPT12_PLAN_SHA256
        assert runner_base.ATTEMPT09_ROW_SCHEMA == ATTEMPT12_ROW_SCHEMA
    assert runner_base.M43_ATTEMPT09_PLAN_SHA256 == old_runner_plan
    assert runner_base.ATTEMPT09_ROW_SCHEMA == old_runner_row

    old_selector_plan = selector_base.M43_ATTEMPT09_PLAN_SHA256
    with selector12._attempt12_selector_bindings():
        assert selector_base.M43_ATTEMPT09_PLAN_SHA256 == M43_ATTEMPT12_PLAN_SHA256
        assert selector_base.EVALUATION_SAMPLE_COUNT == 512
    assert selector_base.M43_ATTEMPT09_PLAN_SHA256 == old_selector_plan

    old_population = selector_base.POPULATION
    with audit12._attempt12_audit_bindings():
        assert selector_base.POPULATION == "future_audit"
        assert selector_base.ROOT_INDEX_FIRST == 200
        assert selector_base._ROOTS == 50
    assert selector_base.POPULATION == old_population

    old_freeze_plan = freeze_base.M43_ATTEMPT09_PLAN_SHA256
    with freeze12._attempt12_freeze_bindings():
        assert freeze_base.M43_ATTEMPT09_PLAN_SHA256 == M43_ATTEMPT12_PLAN_SHA256
        assert freeze_base.DEVELOPMENT_GO_FREEZE_STATUS == "go_freeze_attempt12_development"
    assert freeze_base.M43_ATTEMPT09_PLAN_SHA256 == old_freeze_plan


def test_attempt12_spot_package_is_immutable_and_preflight_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    startup = repository / "scripts" / spot.STARTUP_NAME
    startup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts" / spot.STARTUP_NAME, startup)
    monkeypatch.setitem(
        spot._ATTEMPT12_BINDINGS,
        "validate_attempt09_artifact_bindings",
        lambda *args, **kwargs: None,
    )
    gate = repository / "gates" / "preflight.json"
    gate.parent.mkdir(parents=True)
    gate.write_bytes(
        spot.canonical_json_bytes(
            {
                "schema": "attempt12_gate_v1",
                "status": "pass_local_correctness",
                "decision": "authorize_preflight_only",
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
            }
        )
    )
    run_name = "attempt12-preflight-package-test"
    run_dir = repository / "outputs" / "gcp_runs" / run_name
    manifest = spot.package_attempt12(
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
    assert manifest["plan_sha256"] == M43_ATTEMPT12_PLAN_SHA256
    assert {row["path"] for row in manifest["overlays"]} == set(spot.OVERLAY_RELATIVES)
    assert "configs/hu_joint_policy_m43_attempt11.json" in spot.OVERLAY_RELATIVES
    assert not (run_dir / "execution_authorization.json").exists()


def test_attempt12_startup_script_identity_and_bash_syntax() -> None:
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
    assert "ofc_regular.run_hu_m43_attempt12" in source
    assert "configs/hu_joint_policy_m43_attempt12.json" in source
    assert "hu_m43_attempt12_done_v1" in source
    assert "scalar child selectors are allowed only in Attempt12 preflight" in source
    assert "--if-generation-match=0" in source
