from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import pytest

import ofc_regular.hu_m43_attempt06_teacher as attempt06
import ofc_regular.hu_m43_attempt07_teacher as teacher
import ofc_regular.run_hu_m43_attempt07_development as runner
from ofc_regular.action_key import ActionKey, action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt07_contract import (
    M43_ATTEMPT07_ARMS,
    M43_ATTEMPT07_PLAN_SHA256,
    M43_ATTEMPT07_PROFILES,
    enumerate_attempt07_seed_schedules,
    load_and_validate_attempt07_plan,
)
from ofc_regular.hu_m4_t1_teacher import (
    M4_PAIRED_DELTA_SUMMARY_SCHEMA,
    _ActionScores,
)
from ofc_regular.select_hu_m43_attempt07_development_arm import (
    ATTEMPT07_DEVELOPMENT_SELECTION_SCHEMA,
    aggregate_attempt07_development_rows,
    main,
    select_attempt07_development_arm,
    write_attempt07_development_selection,
)
from ofc_regular.state import Board


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt07.json"
MODEL_HASH = attempt06.ATTEMPT06_FROZEN_MODEL_SHA256


class _Stage9fPolicy:
    def __init__(self, seat: str) -> None:
        self.seat = seat
        self.topk_context = {
            "runtime_profile": "stage9f_p2",
            "runtime_status": "p2_fixed",
        }


class _Ranker:
    artifact_sha256 = MODEL_HASH
    model_id = attempt06.ATTEMPT06_FROZEN_MODEL_ID

    def score_actions(self, _observation, actions, *, baseline_index):
        assert 0 <= baseline_index < len(actions)
        return attempt06.Attempt06RankScores(
            mean=tuple(float(len(actions) - index) for index in range(len(actions))),
            standard_deviation=tuple(0.25 for _ in actions),
        )


def _root(root_index: int) -> ActorObservation:
    cards = random.Random(70_000 + root_index).sample(list(ALL_CARDS), 15)
    return ActorObservation(
        hero_board=Board.from_rows(
            top=(cards[0],),
            middle=(cards[1], cards[2]),
            bottom=(cards[3], cards[4]),
        ),
        opponent_public_board=Board.from_rows(
            top=(cards[5],),
            middle=(cards[6], cards[7], cards[8]),
            bottom=(cards[9], cards[10], cards[11]),
        ),
        dealt_cards=(cards[12], cards[13], cards[14]),
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )


def _scorer(_observation, actions, batch, _selector):
    count = len(batch.particles)
    if count == 8:
        means = [float(9 - index) for index in range(len(actions))]
        means[-1] = 0.0
        return tuple(
            _ActionScores(tuple(mean for _ in range(count))) for mean in means
        )
    if count == 64:
        assert len(actions) == 4
        return (
            _ActionScores(tuple([4.0] * 32 + [-4.0] * 32)),
            _ActionScores(tuple([2.0] * 64)),
            _ActionScores(tuple([1.0] * 64)),
            _ActionScores(tuple([0.0] * 64)),
        )
    if count == 128 and len(actions) != 9:
        return tuple(
            _ActionScores(tuple((0.0 if index == len(actions) - 1 else 7.0) for _ in range(count)))
            for index in range(len(actions))
        )
    if count == 128 and len(actions) == 9:
        means = [1.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]
        return tuple(
            _ActionScores(tuple(mean for _ in range(count))) for mean in means
        )
    raise AssertionError((len(actions), count))


def _provenance(
    *,
    root_index: int,
    profile: str,
    seeds: dict[str, int],
    observation: ActorObservation,
    baseline_token: str,
) -> dict:
    run_id = "attempt07-development-selector-test"
    fixed = runner._fixed_contract(
        root_index=root_index,
        root_profile=profile,
        seeds=seeds,
        run_id=run_id,
        plan_sha256=M43_ATTEMPT07_PLAN_SHA256,
        ai_profiles_sha256=runner.AI_PROFILES_SHA256,
        model_sha256=MODEL_HASH,
        batch_child_selectors=True,
        native_batch_threads=4,
    )
    config_sha256 = runner._canonical_sha256(fixed)
    root_input_sha256 = runner._root_input_sha256(
        root_index=root_index,
        hand_seed=seeds["hand"],
        root_profile=profile,
        observation=observation,
        baseline_action_key=baseline_token,
    )
    root_policy_seeds = {
        seat: runner._profile_policy_seed(seeds["hand"], profile, seat)
        for seat in ("first", "second")
    }
    payload = {
        "schema": runner.ATTEMPT07_PROVENANCE_SCHEMA,
        "run_id": run_id,
        "root_index": root_index,
        "root_profile": profile,
        "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
        "plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
        "ai_profiles_sha256": runner.AI_PROFILES_SHA256,
        "model_sha256": MODEL_HASH,
        "config_sha256": config_sha256,
        "root_input_sha256": root_input_sha256,
        "seeds": dict(seeds),
        "rng_domains": dict(runner._RNG_DOMAIN_NAMES),
        "root_generation_policy": runner.ATTEMPT07_ROOT_GENERATION_POLICY,
        "root_policy_profiles": {"first": profile, "second": profile},
        "root_policy_seeds": root_policy_seeds,
        "baseline_profile": "stage18_p1",
        "baseline_policy_seed": runner._profile_policy_seed(
            seeds["hand"], "stage18_p1", "second"
        ),
        "continuation_profile": "stage9f_p2",
        "continuation_policy_seeds": {
            "first": seeds["child"],
            "second": seeds["child"] + 1,
        },
        "explicit_loaded_profiles": sorted({profile, "stage18_p1", "stage9f_p2"}),
        "batch_child_selectors": True,
        "native_batch_threads": 4,
        "current_profile_resolved": False,
        "opponent_private_discard_input_allowed": False,
        "teacher_value_status": "diagnostic_not_match_EV",
        "teacher_values_are_realized_match_ev": False,
        "teacher_ev_or_lcb_runtime_gate_allowed": False,
        "development_only": True,
        "fit_allowed": False,
        "threshold_selection_allowed": False,
        "runtime_activation_allowed": False,
        "full_replacement_enabled": False,
    }
    runner._validate_provenance(
        payload,
        fixed_contract=fixed,
        config_sha256=config_sha256,
        root_input_sha256=root_input_sha256,
    )
    return payload


@pytest.fixture(scope="module")
def development_rows() -> list[dict]:
    plan = load_and_validate_attempt07_plan(PLAN)
    schedules = enumerate_attempt07_seed_schedules(plan, population="development")
    policies = {
        "first": _Stage9fPolicy("first"),
        "second": _Stage9fPolicy("second"),
    }
    original_scalar = teacher._score_actions
    original_batch = teacher._score_actions_batched
    teacher._score_actions = _scorer
    teacher._score_actions_batched = _scorer
    rows: list[dict] = []
    try:
        for root_index in range(100):
            observation = _root(root_index)
            profile = M43_ATTEMPT07_PROFILES[root_index % 5]
            seeds = {
                domain: values[root_index] for domain, values in schedules.items()
            }
            actions = generate_turn_actions(
                observation.hero_board, observation.dealt_cards
            )
            baseline_token = action_key(actions[-1]).to_token()
            base_run_id = "attempt07-development-selector-test"
            teacher_run_id = (
                f"{base_run_id}:root={root_index}:seed={seeds['hand']}:"
                f"obs={observation.fingerprint()}"
            )
            teacher_row = teacher.evaluate_attempt07_t1_second(
                observation,
                baseline_action_key=baseline_token,
                ranker=_Ranker(),
                t2_policies=policies,
                config=teacher.Attempt07TeacherConfig(
                    frozen_model_sha256=MODEL_HASH,
                    screen_seed=seeds["screen"],
                    rerank_seed=seeds["rerank"],
                    veto_seed=seeds["veto"],
                    assessment_seed=seeds["assessment"],
                    child_policy_seed=seeds["child"],
                    run_id=teacher_run_id,
                    batch_child_selectors=True,
                ),
            )
            rows.append(
                {
                    "schema": runner.ATTEMPT07_SHARD_ROW_SCHEMA,
                    "root_index": root_index,
                    "hand_seed": seeds["hand"],
                    "root_profile": profile,
                    "policy_observation": observation.to_dict(),
                    "baseline_action_key": baseline_token,
                    "provenance": _provenance(
                        root_index=root_index,
                        profile=profile,
                        seeds=seeds,
                        observation=observation,
                        baseline_token=baseline_token,
                    ),
                    "teacher": teacher_row,
                }
            )
    finally:
        teacher._score_actions = original_scalar
        teacher._score_actions_batched = original_batch
    return rows


def _aggregate(rows: list[dict]) -> dict:
    return aggregate_attempt07_development_rows(
        rows,
        plan=load_and_validate_attempt07_plan(PLAN),
        source_input_sha256="a" * 64,
        source_plan_sha256=M43_ATTEMPT07_PLAN_SHA256,
    )


def test_development100_selects_highest_a128_mean_then_lower_cost(
    development_rows,
) -> None:
    report = _aggregate(development_rows)
    assert report["schema"] == ATTEMPT07_DEVELOPMENT_SELECTION_SCHEMA
    assert report["decision"] == "go"
    assert report["selected_arm"] == "r64_v64"
    assert report["winner"]["arm_name"] == "r64_v64"
    assert report["development_population"]["profile_counts"] == {
        profile: 20 for profile in M43_ATTEMPT07_PROFILES
    }
    assert report["development_population"]["unique_observation_fingerprints"] == 100
    assert report["arms"]["r32_v64"]["metrics"]["overall"][
        "mean_delta_per_state"
    ] == pytest.approx(1.0)
    assert report["arms"]["r64_v64"]["metrics"]["overall"][
        "mean_delta_per_state"
    ] == pytest.approx(2.0)
    assert all(report["arms"][arm]["eligible"] for arm in M43_ATTEMPT07_ARMS)


def test_quality_metrics_use_a128_raw_never_veto_summary(development_rows) -> None:
    report = _aggregate(development_rows)
    assert report["arms"]["r64_v64"]["metrics"]["overall"][
        "mean_delta_per_fire"
    ] == pytest.approx(2.0)
    # The validated V64 mean is 7, which must not leak into development metrics.
    assert development_rows[0]["teacher"]["arms"]["R64_V64"][
        "veto_paired_delta_vs_baseline"
    ]["mean"] == pytest.approx(7.0)
    assert report["science_boundary"][
        "veto_values_used_as_development_quality_metrics"
    ] is False
    assert report["science_boundary"]["assessment_source"].startswith(
        "disjoint_A128_raw"
    )


def _rewrite_assessment_row(action_row: dict, delta: float) -> None:
    raw = [float(delta)] * 128
    std = 0.0
    summary = {
        "mean": float(delta),
        "standard_error": 0.0,
        "std": std,
        "min": float(delta),
        "p01": float(delta),
        "p05": float(delta),
        "p25": float(delta),
        "p50": float(delta),
        "p75": float(delta),
        "p95": float(delta),
        "p99": float(delta),
        "max": float(delta),
        "lt0_rate": 1.0 if delta < 0.0 else 0.0,
        "le_neg6_rate": 1.0 if delta <= -6.0 else 0.0,
        "le_neg12_rate": 1.0 if delta <= -12.0 else 0.0,
        "le_neg20_rate": 1.0 if delta <= -20.0 else 0.0,
    }
    action_row["mean"] = float(delta)
    action_row["standard_error"] = 0.0
    action_row["raw_paired_deltas_vs_baseline"] = raw
    action_row["raw_paired_deltas_sha256"] = teacher._raw_digest(raw)
    action_row["paired_delta_vs_baseline"] = {
        "schema": M4_PAIRED_DELTA_SUMMARY_SCHEMA,
        "count": 128,
        **summary,
    }


def _recompute_assessment_best(row: dict) -> None:
    actions = row["teacher"]["assessment"]["actions"]
    best = max(float(action["mean"]) for action in actions)
    tied = [action for action in actions if float(action["mean"]) == best]
    selected = min(
        tied, key=lambda action: ActionKey.from_token(action["action_key"]).sort_key()
    )
    row["teacher"]["assessment"]["sample_best_action_key"] = selected["action_key"]


def test_none_eligible_returns_no_go_without_threshold_or_fit(development_rows) -> None:
    rows = copy.deepcopy(development_rows)
    for row in rows:
        baseline = row["baseline_action_key"]
        for action_row in row["teacher"]["assessment"]["actions"]:
            if action_row["action_key"] != baseline:
                _rewrite_assessment_row(action_row, -1.0)
        _recompute_assessment_best(row)
    report = _aggregate(rows)
    assert report["decision"] == "no_go"
    assert report["selected_arm"] is None
    assert report["winner"] is None
    assert all(not report["arms"][arm]["eligible"] for arm in M43_ATTEMPT07_ARMS)
    assert report["science_boundary"]["fit_performed"] is False
    assert report["science_boundary"]["threshold_selected"] is False
    assert report["science_boundary"]["gate_reselected"] is False


def test_exact_metric_tie_uses_cost_then_fixed_arm_name(development_rows) -> None:
    rows = copy.deepcopy(development_rows)
    for row in rows:
        first_proposal = row["teacher"]["learned_top8_action_keys"][0]
        action_row = next(
            action
            for action in row["teacher"]["assessment"]["actions"]
            if action["action_key"] == first_proposal
        )
        _rewrite_assessment_row(action_row, 2.0)
        _recompute_assessment_best(row)
    report = _aggregate(rows)
    assert report["selected_arm"] == "r32_v64"
    assert report["arms"]["r32_v64"]["total_action_futures_per_root"] == 1480


def test_integrity_is_fail_closed_and_runtime_trajectory_is_deferred(
    development_rows,
) -> None:
    report = _aggregate(development_rows)
    assert report["integrity"] == {
        "action_mapping_violation_count": 0,
        "rng_domain_violation_count": 0,
        "hidden_information_violation_count": 0,
        "nonfire_exact_baseline_action_fallback_verified": True,
        "nonfire_complete_trajectory_cancellation_verified": False,
        "nonfire_complete_trajectory_acceptance_deferred": True,
    }
    assert report["science_boundary"]["runtime_trajectory_cancellation_claimed"] is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
        lambda rows: rows[0]["provenance"].__setitem__(
            "plan_sha256", "0" * 64
        ),
        lambda rows: rows[0]["provenance"].__setitem__(
            "batch_child_selectors", False
        ),
        lambda rows: rows[0]["provenance"].__setitem__(
            "current_profile_resolved", 0
        ),
        lambda rows: rows[0]["teacher"]["assessment"]["actions"][0].__setitem__(
            "raw_paired_deltas_vs_baseline", [999.0] * 128
        ),
        lambda rows: rows[0]["teacher"]["assessment"]["actions"][0].__setitem__(
            "is_explicit_baseline", 0
        ),
        lambda rows: rows[0]["teacher"]["legal_actions"][0].__setitem__(
            "is_explicit_baseline", 0
        ),
        lambda rows: rows[0]["teacher"]["legal_actions"][0].__setitem__(
            "in_learned_top8", 1
        ),
        lambda rows: rows[0]["teacher"]["rerank"]["prefixes"]["R32"].__setitem__(
            "winner_is_explicit_baseline", 0
        ),
        lambda rows: rows[0]["teacher"]["assessment"]["action_keys"].reverse(),
        lambda rows: rows[0]["teacher"]["rng_key_digests"].__setitem__(
            "screen_s8",
            rows[0]["teacher"]["rng_key_digests"]["rerank_r64"][:8],
        ),
        lambda rows: rows[0]["teacher"].__setitem__(
            "opponent_private_discards", ["As"]
        ),
        lambda rows: rows[0]["teacher"]["arms"]["R32_V64"].__setitem__(
            "selected_action_key", rows[0]["baseline_action_key"]
        ),
    ],
)
def test_adversarial_row_mutations_fail_closed(development_rows, mutator) -> None:
    rows = copy.deepcopy(development_rows)
    mutator(rows)
    with pytest.raises(ValueError):
        _aggregate(rows)


def test_exactly_100_unique_roots_are_required(development_rows) -> None:
    with pytest.raises(ValueError, match="exactly 100"):
        _aggregate(development_rows[:-1])
    duplicated = copy.deepcopy(development_rows)
    duplicated[99]["root_index"] = 98
    with pytest.raises(ValueError, match="duplicate root_index"):
        _aggregate(duplicated)


def test_jsonl_cli_hashes_sources_and_refuses_output_clobber(
    tmp_path: Path, development_rows
) -> None:
    input_path = tmp_path / "development.jsonl"
    input_bytes = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in development_rows
    )
    input_path.write_bytes(input_bytes)
    report = select_attempt07_development_arm(input_path=input_path, plan_path=PLAN)
    assert report["source"]["input_jsonl_sha256"] == hashlib.sha256(
        input_bytes
    ).hexdigest()
    assert report["source"]["plan_sha256"] == M43_ATTEMPT07_PLAN_SHA256
    assert len(report["source"]["selector_source_sha256"]) == 64

    output_path = tmp_path / "selection.json"
    assert main(
        [
            "--input",
            str(input_path),
            "--plan",
            str(PLAN),
            "--output",
            str(output_path),
        ]
    ) == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["selected_arm"] == "r64_v64"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_attempt07_development_selection(output_path, report)
