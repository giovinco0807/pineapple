from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import pytest

import ofc_regular.hu_m43_attempt11_teacher as teacher
import ofc_regular.hu_m43_attempt11_spot as spot
import ofc_regular.freeze_hu_m43_attempt09_development as freeze_base
import ofc_regular.freeze_hu_m43_attempt11_development as freeze11
import ofc_regular.select_hu_m43_attempt09_development as selector_base
import ofc_regular.select_hu_m43_attempt11_audit50 as audit11
import ofc_regular.select_hu_m43_attempt11_development as selector11
from ofc_regular import hu_m43_attempt09_spot as spot_base
from ofc_regular import run_hu_m43_attempt09 as runner_base
from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.ai_profiles import ModelPaths, build_policy, load_model_bundle
from ofc_regular.generate_hu_m4_t1_data import (
    _profile_policy_seed,
    generate_t1_second_root,
)
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m4_t1_teacher import _ActionScores
from ofc_regular.hu_m43_attempt11_contract import (
    M43_ATTEMPT11_PLAN_SHA256,
    M43_ATTEMPT11_PROFILES,
    enumerate_attempt11_seed_schedules,
    load_and_validate_attempt11_plan,
)
from ofc_regular.hu_m43_attempt11_spot import (
    _validate_root4_variable_candidate_teacher,
    build_schedule,
)
from ofc_regular.run_hu_m43_attempt11 import (
    ATTEMPT11_ROW_SCHEMA,
    _attempt11_bindings,
)
from ofc_regular.select_hu_m43_attempt11_development import (
    _expected_search_config,
)
from ofc_regular.play_ai import _choose_from_observation, _hand_decision_seed
from ofc_regular.state import Board


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt11.json"


def _root() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("9h",), middle=("Th", "Jh"), bottom=("Qh", "Kh")
        ),
        opponent_public_board=Board.from_rows(
            top=("2h",),
            middle=("3h", "4h", "5h"),
            bottom=("6h", "7h", "8h"),
        ),
        dealt_cards=("Ah", "2d", "3d"),
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )


class _Stage9fPolicy:
    def __init__(self, seat: str) -> None:
        self.seat = seat
        self.topk_context = {
            "runtime_profile": "stage9f_p2",
            "runtime_status": "p2_fixed",
        }


def _policies() -> dict[str, object]:
    return {
        "first": _Stage9fPolicy("first"),
        "second": _Stage9fPolicy("second"),
    }


class _Ranker:
    artifact_sha256 = teacher.ATTEMPT11_FROZEN_MODEL_SHA256
    model_id = teacher.ATTEMPT11_FROZEN_MODEL_ID

    def score_actions(self, observation, actions, *, baseline_index):
        size = len(actions)
        return teacher.Attempt11RankScores(
            rank_mean=tuple(float(size - index) for index in range(size)),
            rank_disagreement=tuple(0.25 for _ in actions),
            raw_downside_p95=tuple(10.0 + index for index in range(size)),
            raw_downside_p99=tuple(20.0 + index for index in range(size)),
            raw_downside_max=tuple(30.0 + index for index in range(size)),
        )


def _config(*, batch: bool = False) -> teacher.Attempt11TeacherConfig:
    return teacher.Attempt11TeacherConfig(
        frozen_model_sha256=teacher.ATTEMPT11_FROZEN_MODEL_SHA256,
        hand_seed=1001,
        rerank_seed=1002,
        veto_seed=1003,
        stress_seed=1004,
        confirmation_seed=1005,
        evaluation_seed=1006,
        child_policy_seed=1007,
        run_id="attempt11-test",
        batch_child_selectors=batch,
    )


def _score(_observation, actions, batch, _selector):
    phase = batch.run_id.rsplit(":", 1)[-1]
    expected = {
        "rerank_r128": 128,
        "veto_v256": 256,
        "stress_x1024": 1024,
        "confirmation_c512": 512,
        "evaluation_e256": 256,
    }
    assert len(batch.particles) == expected[phase]
    means = list(range(len(actions) - 1, -1, -1))
    return tuple(
        _ActionScores(tuple(float(mean) for _ in batch.particles))
        for mean in means
    )


def _limited_actions(n: int):
    actions = tuple(generate_turn_actions(_root().hero_board, _root().dealt_cards))
    baseline = actions[-1]
    nonbaseline = [action for action in actions[:-1] if action != baseline]
    assert len(nonbaseline) >= 12
    return tuple([*nonbaseline[:n], baseline])


def _evaluate(monkeypatch: pytest.MonkeyPatch, n: int, *, batch: bool = False):
    actions = _limited_actions(n)
    monkeypatch.setattr(teacher, "generate_turn_actions", lambda *_: actions)
    monkeypatch.setattr(teacher, "_score_actions", _score)
    monkeypatch.setattr(teacher, "_score_actions_batched", _score)
    payload = teacher.evaluate_attempt11_t1_second(
        _root(),
        baseline_action_key=action_key(actions[-1]).to_token(),
        ranker=_Ranker(),
        t2_policies=_policies(),
        config=_config(batch=batch),
    )
    return actions, payload


def test_attempt11_plan_hash_seeds_balance_and_seven_slot_preflight() -> None:
    plan = load_and_validate_attempt11_plan(PLAN)
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == M43_ATTEMPT11_PLAN_SHA256
    assert plan["attempt10_boundary"]["failed_root_indices"] == [4, 24, 79]
    assert plan["attempt10_boundary"]["all_attempt10_files_and_artifacts_preserved"]
    assert all(value is False for value in plan["activation_guards"].values())
    schedules = {}
    for population, count in (("development", 200), ("future_audit", 50)):
        values = enumerate_attempt11_seed_schedules(plan, population=population)
        assert all(len(schedule) == count for schedule in values.values())
        schedules.update({f"{population}.{key}": value for key, value in values.items()})
    preflight = enumerate_attempt11_seed_schedules(plan, population="preflight")
    assert all(len(schedule) == 5 for schedule in preflight.values())
    schedules.update({f"preflight.{key}": value for key, value in preflight.items()})
    for left, right in combinations(schedules, 2):
        assert set(schedules[left]).isdisjoint(schedules[right])
    assert schedules["development.hand"][0] == 140_108_071_901
    assert schedules["preflight.hand"] == tuple(
        150_108_071_901 + 1_000_003 * index for index in range(5)
    )

    schedule = build_schedule("preflight", "attempt11-preflight-test")
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
    assert {row["root_profile"] for row in schedule} == set(M43_ATTEMPT11_PROFILES)
    assert schedule[-1]["root_profile"] == "random_exact_final"


@pytest.mark.parametrize("n", [0, 1, 3, 4, 5, 7, 8, 9, 12])
def test_attempt11_variable_candidate_and_shortlist_contract(
    monkeypatch: pytest.MonkeyPatch, n: int
) -> None:
    _, payload = _evaluate(monkeypatch, n)
    baseline = payload["baseline_action_key"]
    k = min(8, n)
    h = min(4, n)
    assert payload["candidate_nonbaseline_count"] == n
    assert payload["proposal_mapping"]["action_count"] == n + 1
    assert payload["proposal_mapping"]["action_keys"][-1] == baseline
    assert len(set(payload["proposal_mapping"]["action_keys"])) == n + 1
    assert payload["rerank"]["action_count"] == n + 1
    assert payload["shortlist"]["nonbaseline_count"] == k
    assert payload["shortlist"]["action_count"] == k
    assert len(payload["shortlist"]["head_rerank_positions"]) == h
    assert len(payload["shortlist"]["risk_reserve_rerank_positions"]) == k - h
    assert len(set(payload["shortlist"]["action_keys"])) == k
    assert payload["veto"]["action_count"] == k + 1
    assert payload["veto"]["action_keys"][-1] == baseline
    assert payload["veto"]["action_keys"][:-1] == payload["shortlist"]["action_keys"]


def test_attempt11_zero_alternatives_is_baseline_only_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, payload = _evaluate(monkeypatch, 0)
    assert payload["rerank"]["action_count"] == 1
    assert payload["veto"]["action_count"] == 1
    assert list(payload["rng_key_digests"]) == ["rerank_r128", "veto_v256"]
    assert payload["stress"]["opened"] is False
    assert payload["confirmation"]["opened"] is False
    assert payload["evaluation"]["opened"] is False
    assert payload["decision"] == {
        "final_selected_action_key": payload["baseline_action_key"],
        "override_fired": False,
        "exact_baseline_fallback": True,
        "fallback_reason": "no_v256_candidate_passed",
        "candidate_fallback_across_V_X_C_allowed": True,
        "frozen_before_evaluation_namespace_open": True,
    }
    tampered = copy.deepcopy(payload)
    tampered["stress"]["action_count"] = 1
    with pytest.raises(ValueError, match="closed phase mapping/count"):
        teacher.validate_attempt11_teacher_output(
            _root(),
            baseline_action_key=payload["baseline_action_key"],
            payload=tampered,
            config=_config(),
        )


def test_attempt11_scalar_batch_parity_hidden_info_and_tamper_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions, scalar = _evaluate(monkeypatch, 5, batch=False)
    _, batch = _evaluate(monkeypatch, 5, batch=True)
    normalized = copy.deepcopy(batch)
    normalized["search_config"]["batch_child_selectors"] = False
    assert normalized == scalar
    encoded = json.dumps(scalar, sort_keys=True)
    assert "opponent_private_discard" not in encoded
    assert '"particles"' not in encoded
    assert teacher.validate_attempt11_teacher_output(
        _root(),
        baseline_action_key=action_key(actions[-1]).to_token(),
        payload=scalar,
        config=_config(),
    )["selected_action_key"] == scalar["decision"]["final_selected_action_key"]

    hidden = copy.deepcopy(scalar)
    hidden["opponent_private_discard"] = ["As"]
    with pytest.raises(ValueError, match="hidden opponent"):
        teacher.validate_attempt11_teacher_output(
            _root(),
            baseline_action_key=action_key(actions[-1]).to_token(),
            payload=hidden,
            config=_config(),
        )
    tampered = copy.deepcopy(scalar)
    tampered["shortlist"]["action_keys"].append(
        tampered["shortlist"]["action_keys"][0]
    )
    with pytest.raises(ValueError, match="shortlist"):
        teacher.validate_attempt11_teacher_output(
            _root(),
            baseline_action_key=action_key(actions[-1]).to_token(),
            payload=tampered,
            config=_config(),
        )


def test_attempt11_duplicate_legal_action_mapping_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = _limited_actions(1)
    duplicate = (actions[0], actions[0], actions[-1])
    monkeypatch.setattr(teacher, "generate_turn_actions", lambda *_: duplicate)
    with pytest.raises(ValueError, match="not unique"):
        teacher.evaluate_attempt11_t1_second(
            _root(),
            baseline_action_key=action_key(actions[-1]).to_token(),
            ranker=_Ranker(),
            t2_policies=_policies(),
            config=_config(),
        )


def test_attempt11_old_failed_root4_real_cardinality_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = 120_108_071_901 + 1_000_003 * 4
    root_profile = "random_exact_final"
    baseline_profile = "stage18_p1"
    bundle = load_model_bundle(ModelPaths(), profiles={root_profile, baseline_profile})
    root_policies = {
        seat: build_policy(
            root_profile,
            bundle,
            seed=_profile_policy_seed(seed, root_profile, seat),
            seat=seat,
            opening_lookahead_samples=0,
        )
        for seat in ("first", "second")
    }
    observation = generate_t1_second_root(seed, root_policies=root_policies)
    baseline_policy = build_policy(
        baseline_profile,
        bundle,
        seed=_profile_policy_seed(seed, baseline_profile, "second"),
        seat="second",
        opening_lookahead_samples=0,
    )
    baseline = _choose_from_observation(
        baseline_policy,
        observation,
        hand_id=seed,
        game_id=seed,
        decision_seed=_hand_decision_seed(base_seed=seed, observation=observation),
    )
    actions = tuple(generate_turn_actions(observation.hero_board, observation.dealt_cards))
    tokens = [action_key(action).to_token() for action in actions]
    baseline_token = action_key(baseline).to_token()
    assert len(actions) == len(set(tokens)) == 12
    assert tokens.count(baseline_token) == 1
    assert len(actions) - 1 == 11

    monkeypatch.setattr(teacher, "generate_turn_actions", lambda *_: actions)
    monkeypatch.setattr(teacher, "_score_actions", _score)
    payload = teacher.evaluate_attempt11_t1_second(
        observation,
        baseline_action_key=baseline_token,
        ranker=_Ranker(),
        t2_policies=_policies(),
        config=_config(),
    )
    assert payload["candidate_nonbaseline_count"] == 11
    assert payload["proposal_mapping"]["action_count"] == 12
    assert payload["shortlist"]["action_count"] == 8
    assert payload["veto"]["action_count"] == 9


def test_attempt11_root4_preflight_variable_mapping_guard() -> None:
    teacher_payload = {
        "candidate_nonbaseline_count": 3,
        "baseline_action_key": "baseline",
        "learned_nonbaseline_action_keys": ["a", "b", "c"],
        "proposal_mapping": {
            "action_count": 4,
            "action_keys": ["a", "b", "c", "baseline"],
        },
        "rerank": {
            "action_count": 4,
            "action_keys": ["a", "b", "c", "baseline"],
        },
        "shortlist": {
            "nonbaseline_count": 3,
            "action_count": 3,
            "action_keys": ["a", "b", "c"],
        },
        "veto": {
            "action_count": 4,
            "action_keys": ["a", "b", "c", "baseline"],
        },
    }
    assert _validate_root4_variable_candidate_teacher(teacher_payload) == 3
    teacher_payload["veto"]["action_keys"].insert(0, "a")
    with pytest.raises(ValueError, match="mapping changed"):
        _validate_root4_variable_candidate_teacher(teacher_payload)


def test_attempt11_runner_and_selector_bindings_are_opt_in_and_scoped() -> None:
    old_plan = runner_base.M43_ATTEMPT09_PLAN_SHA256
    old_row = runner_base.ATTEMPT09_ROW_SCHEMA
    with _attempt11_bindings():
        assert runner_base.M43_ATTEMPT09_PLAN_SHA256 == M43_ATTEMPT11_PLAN_SHA256
        assert runner_base.ATTEMPT09_ROW_SCHEMA == ATTEMPT11_ROW_SCHEMA
    assert runner_base.M43_ATTEMPT09_PLAN_SHA256 == old_plan
    assert runner_base.ATTEMPT09_ROW_SCHEMA == old_row
    expected = _expected_search_config(
        _config(batch=True),
        {
            "hand": 1,
            "rerank": 2,
            "veto": 3,
            "stress": 4,
            "confirmation": 5,
            "evaluation": 6,
            "child": 7,
        },
    )
    assert expected["learned_nonbaseline_max"] == 12
    assert expected["shortlist_max"] == 8
    assert "learned_nonbaseline_count" not in expected
    assert spot_base.M43_ATTEMPT09_PROFILES != ()

    old_selector_plan = selector_base.M43_ATTEMPT09_PLAN_SHA256
    with selector11._attempt11_selector_bindings():
        assert selector_base.M43_ATTEMPT09_PLAN_SHA256 == M43_ATTEMPT11_PLAN_SHA256
        assert (
            selector_base.ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA
            == selector11.ATTEMPT11_DEVELOPMENT_DECISION_SCHEMA
        )
    assert selector_base.M43_ATTEMPT09_PLAN_SHA256 == old_selector_plan

    old_population = selector_base.POPULATION
    with audit11._attempt11_audit_bindings():
        assert selector_base.POPULATION == "future_audit"
        assert selector_base.ROOT_INDEX_FIRST == 200
    assert selector_base.POPULATION == old_population

    old_freeze_plan = freeze_base.M43_ATTEMPT09_PLAN_SHA256
    with freeze11._attempt11_freeze_bindings():
        assert freeze_base.M43_ATTEMPT09_PLAN_SHA256 == M43_ATTEMPT11_PLAN_SHA256
        assert (
            freeze_base.DEVELOPMENT_GO_FREEZE_STATUS
            == "go_freeze_attempt11_development"
        )
    assert freeze_base.M43_ATTEMPT09_PLAN_SHA256 == old_freeze_plan


def test_attempt11_spot_package_has_seven_shards_and_clean_overlay(
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
        spot._ATTEMPT11_BINDINGS,
        "validate_attempt09_artifact_bindings",
        lambda *args, **kwargs: None,
    )
    gate = repository / "gates" / "preflight.json"
    gate.parent.mkdir(parents=True)
    gate.write_bytes(
        spot.canonical_json_bytes(
            {
                "schema": "attempt11_gate_v1",
                "status": "pass_local_correctness",
                "decision": "authorize_preflight_only",
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
            }
        )
    )
    run_name = "attempt11-preflight-package-test"
    run_dir = repository / "outputs" / "gcp_runs" / run_name
    manifest = spot.package_attempt11(
        mode="preflight",
        run_name=run_name,
        run_dir=run_dir,
        repository_root=repository,
        template_package=template,
        plan=repository / spot.PLAN_RELATIVE,
        startup=startup,
        preceding_gate=gate,
    )
    validated = spot.validate_package(run_dir)
    assert manifest == validated
    assert manifest["total_shards"] == 7
    assert manifest["plan_sha256"] == M43_ATTEMPT11_PLAN_SHA256
    assert {row["path"] for row in manifest["overlays"]} == set(
        spot.OVERLAY_RELATIVES
    )
    assert "configs/hu_joint_policy_m43_attempt10.json" in spot.OVERLAY_RELATIVES
    assert not any("attempt09_development" in value for value in spot.OVERLAY_RELATIVES)
    assert not (run_dir / "execution_authorization.json").exists()


def test_attempt11_startup_script_identity_and_bash_syntax() -> None:
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
    assert "ofc_regular.run_hu_m43_attempt11" in source
    assert "configs/hu_joint_policy_m43_attempt11.json" in source
    assert "hu_m43_attempt11_done_v1" in source
    assert "scalar child selectors are allowed only in Attempt11 preflight" in source
    assert "--if-generation-match=0" in source
