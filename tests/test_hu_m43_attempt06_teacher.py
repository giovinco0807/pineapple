from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import ofc_regular.hu_m43_attempt06_teacher as teacher
from ofc_regular.action_key import (
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m4_t1_teacher import _ActionScores
from ofc_regular.state import Board


MODEL_HASH = teacher.ATTEMPT06_FROZEN_MODEL_SHA256


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


def _policies():
    return {
        "first": _Stage9fPolicy("first"),
        "second": _Stage9fPolicy("second"),
    }


class _Ranker:
    artifact_sha256 = MODEL_HASH
    model_id = "fake-frozen-lambda"

    def __init__(self, scores=None, events=None) -> None:
        self.scores = scores
        self.events = events

    def score_actions(self, observation, actions, *, baseline_index):
        assert observation == _root()
        assert 0 <= baseline_index < len(actions)
        if self.events is not None:
            self.events.append("rank")
        scores = self.scores
        if scores is None:
            scores = tuple(float(len(actions) - index) for index in range(len(actions)))
        return teacher.Attempt06RankScores(
            mean=tuple(scores),
            standard_deviation=tuple(0.25 for _ in actions),
        )


def _config(**overrides) -> teacher.Attempt06TeacherConfig:
    values = {
        "frozen_model_sha256": MODEL_HASH,
        "candidate_seed": 101,
        "evaluation_seed": 102,
        "child_policy_seed": 103,
        "run_id": "attempt06-test",
    }
    values.update(overrides)
    return teacher.Attempt06TeacherConfig(**values)


def _baseline_token() -> str:
    actions = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    return action_key(actions[-1]).to_token()


def _controlled_scorer(
    call_shapes: list[tuple[int, int]], events: list[str] | None = None
):
    calls = 0

    def score(_observation, actions, batch, _selector):
        nonlocal calls
        calls += 1
        call_shapes.append((len(actions), len(batch.particles)))
        count = len(batch.particles)
        if events is not None:
            events.append("score-c8" if count == 8 else "score-e128")
        if calls == 1:
            means = [10.0, 9.0, *([0.0] * (len(actions) - 2))]
            return tuple(_ActionScores(tuple([mean] * count)) for mean in means)
        rows = []
        for index in range(len(actions)):
            if index == 0:
                values = tuple(-float(value) for value in range(count))
            elif index == 1:
                values = tuple(100.0 for _ in range(count))
            else:
                values = tuple(0.0 for _ in range(count))
            rows.append(_ActionScores(values))
        return tuple(rows)

    return score


def test_config_hard_locks_top8_c8_e128_stage9f_and_mc1() -> None:
    config = _config()
    assert config.candidate_top_k == 8
    assert config.candidate_samples == 8
    assert config.evaluation_samples == 128
    assert config.t2_policy_id == "stage9f_p2"
    assert config.t3_candidate_samples == 1
    assert config.t4_evaluation_samples == 1
    with pytest.raises(ValueError, match="fixed at 8"):
        _config(candidate_samples=4)
    with pytest.raises(ValueError, match="fixed at 128"):
        _config(evaluation_samples=64)
    with pytest.raises(ValueError, match="stage9f_p2"):
        _config(t2_policy_id="stage3_baseline")
    with pytest.raises(ValueError, match="distinct"):
        _config(evaluation_seed=101)
    with pytest.raises(ValueError, match="SHA-256"):
        _config(frozen_model_sha256="bad")
    with pytest.raises(ValueError, match="frozen Lambda"):
        _config(frozen_model_sha256="a" * 64)


def test_top8_is_actionkey_tied_and_fixed_before_particles(monkeypatch) -> None:
    events: list[str] = []
    original_sample = teacher.sample_hidden_card_particles

    def sample(*args, **kwargs):
        events.append("sample")
        return original_sample(*args, **kwargs)

    call_shapes: list[tuple[int, int]] = []
    monkeypatch.setattr(teacher, "sample_hidden_card_particles", sample)
    monkeypatch.setattr(
        teacher, "_score_actions", _controlled_scorer(call_shapes, events)
    )
    legal = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    baseline = legal[-1]
    ranker = _Ranker(scores=tuple(1.0 for _ in legal), events=events)
    result = teacher.evaluate_attempt06_t1_second(
        _root(),
        baseline_action_key=action_key(baseline).to_token(),
        ranker=ranker,
        t2_policies=_policies(),
        config=_config(),
    )

    expected = sorted(
        (action for action in legal if action_key(action) != action_key(baseline)),
        key=lambda action: action_key(action).sort_key(),
    )[:8]
    assert events == ["rank", "sample", "score-c8", "sample", "score-e128"]
    assert result["learned_top8_action_keys"] == [
        action_key(action).to_token() for action in expected
    ]
    assert result["root_selection_lock"] == (
        "top8_fixed_before_sampling_then_c8_locked_before_e128"
    )
    assert call_shapes == [(9, 8), (2, 128)]


def test_c8_lock_survives_e128_rerank_and_emits_paired_tails(monkeypatch) -> None:
    call_shapes: list[tuple[int, int]] = []
    monkeypatch.setattr(teacher, "_score_actions", _controlled_scorer(call_shapes))
    result = teacher.evaluate_attempt06_t1_second(
        _root(),
        baseline_action_key=_baseline_token(),
        ranker=_Ranker(),
        t2_policies=_policies(),
        config=_config(),
    )

    selected = next(row for row in result["actions"] if row["selected_by_c8"])
    evaluation_best = next(
        row for row in result["actions"] if row["evaluation_sample_best"]
    )
    assert result["selected_action_key"] == selected["action_key"]
    assert result["override_fired"] is True
    assert evaluation_best["action_key"] != selected["action_key"]
    assert result["evaluation_sample_best_action_key"] == evaluation_best["action_key"]
    assert result["evaluation_action_count"] == 2
    assert result["evaluation_action_keys"] == [
        selected["action_key"],
        result["baseline_action_key"],
    ]
    assert result["evaluation_scope"] == (
        "c8_locked_action_plus_explicit_baseline_only"
    )
    delta = selected["paired_evaluation_delta_vs_baseline"]
    raw = result["selected_action_paired_evaluation_deltas_vs_baseline"]
    loss = selected["paired_evaluation_override_loss"]
    baseline = next(row for row in result["actions"] if row["is_explicit_baseline"])
    assert len(raw) == 128
    assert delta["mean"] == pytest.approx(float(np.mean(raw)))
    assert delta["p05"] == pytest.approx(
        float(np.quantile(np.asarray(raw), 0.05, method="linear"))
    )
    assert delta["p01"] == pytest.approx(
        float(np.quantile(np.asarray(raw), 0.01, method="linear"))
    )
    assert result["selected_action_evaluation_mean"] - baseline[
        "evaluation_mean"
    ] == pytest.approx(delta["mean"])
    assert delta["count"] == 128
    assert loss["p95"] == pytest.approx(max(0.0, -delta["p05"]))
    assert loss["p99"] == pytest.approx(max(0.0, -delta["p01"]))
    assert loss["max"] == pytest.approx(max(0.0, -delta["min"]))
    assert selected["paired_evaluation_state_mean_loss_diagnostic"] == pytest.approx(
        max(0.0, -delta["mean"])
    )
    assert result["audit_aggregation_contract"]["aggregate_over_fires"] == {
        "p95": "maximum_of_per_root_e128_p95",
        "p99": "maximum_of_per_root_e128_p99",
        "max": "maximum_of_per_root_e128_max",
    }
    assert result["audit_aggregation_contract"][
        "state_mean_loss_is_non_gate_diagnostic"
    ] is True
    assert baseline["paired_evaluation_delta_vs_baseline"]["mean"] == 0.0
    assert baseline["paired_evaluation_override_loss"] == {
        "p95": 0.0,
        "p99": 0.0,
        "max": 0.0,
    }
    unscored = [
        row
        for row in result["actions"]
        if not row["selected_by_c8"] and not row["is_explicit_baseline"]
    ]
    assert len(unscored) == 7
    for row in unscored:
        assert row["evaluation_scope"] == "not_evaluated_after_c8_lock"
        assert row["evaluation_mean"] is None
        assert row["evaluation_standard_error"] is None
        assert row["paired_evaluation_delta_vs_baseline"] is None
        assert row["paired_evaluation_override_loss"] is None
        assert row["paired_evaluation_state_mean_loss_diagnostic"] is None
        assert row["evaluation_sample_best"] is None


def test_fixed_contract_binds_v3_full_search_and_native4() -> None:
    kwargs = {
        "root_index": 0,
        "input_sha256": "a" * 64,
        "model_sha256": teacher.ATTEMPT06_FROZEN_MODEL_SHA256,
        "source_model_manifest_sha256": (
            teacher.ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256
        ),
        "source_native_manifest_sha256": (
            teacher.ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256
        ),
        "run_id": "attempt06-test:shard=0",
        "batch_child_selectors": True,
        "native_batch_threads": 4,
    }
    contract = teacher.attempt06_fixed_contract(**kwargs)
    assert contract["teacher_schema"].endswith("_v3")
    assert contract["evaluation_action_scope"] == (
        "locked_action_plus_explicit_baseline_only"
    )
    assert contract["t3_candidate_samples"] == 1
    assert contract["hypothetical_t4_evaluation_samples"] == 1
    assert contract["real_live_t4_exact_unchanged"] is True
    assert teacher.attempt06_fixed_contract_sha256(**kwargs) == hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    with pytest.raises(ValueError, match="native batch threads"):
        teacher.attempt06_fixed_contract(**{**kwargs, "native_batch_threads": 3})


def test_locked_baseline_is_evaluated_once_and_cancels_exactly(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    def baseline_scorer(_observation, actions, batch, _selector):
        calls.append((len(actions), len(batch.particles)))
        count = len(batch.particles)
        if count == 8:
            values = [0.0 for _ in actions]
            values[-1] = 10.0
            return tuple(_ActionScores((value,) * count) for value in values)
        assert len(actions) == 1
        return (_ActionScores(tuple(float(index) for index in range(count))),)

    monkeypatch.setattr(teacher, "_score_actions", baseline_scorer)
    result = teacher.evaluate_attempt06_t1_second(
        _root(),
        baseline_action_key=_baseline_token(),
        ranker=_Ranker(),
        t2_policies=_policies(),
        config=_config(),
    )

    assert calls == [(9, 8), (1, 128)]
    assert result["override_fired"] is False
    assert result["selected_action_key"] == result["baseline_action_key"]
    assert result["evaluation_action_count"] == 1
    assert result["evaluation_action_keys"] == [result["baseline_action_key"]]
    assert result["selected_action_paired_evaluation_delta_vs_baseline"]["mean"] == 0.0
    assert result["selected_action_paired_evaluation_override_loss"] == {
        "p95": 0.0,
        "p99": 0.0,
        "max": 0.0,
    }
    assert sum(row["evaluation_mean"] is not None for row in result["actions"]) == 1


def test_complete_legal_and_candidate_action_mapping_digests(monkeypatch) -> None:
    monkeypatch.setattr(teacher, "_score_actions", _controlled_scorer([]))
    result = teacher.evaluate_attempt06_t1_second(
        _root(),
        baseline_action_key=_baseline_token(),
        ranker=_Ranker(),
        t2_policies=_policies(),
        config=_config(),
    )
    legal = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    candidate_by_key = {action_key(action).to_token(): action for action in legal}
    candidate = tuple(
        candidate_by_key[row["action_key"]] for row in result["actions"]
    )
    assert result["legal_action_count"] == len(legal)
    assert len(result["legal_actions"]) == len(legal)
    assert result["legal_action_set_digest"] == legal_action_set_digest(legal)
    assert result["legal_action_order_digest"] == ordered_action_mapping_digest(legal)
    assert result["candidate_action_count"] == 9
    assert result["candidate_action_set_digest"] == legal_action_set_digest(candidate)
    assert result["candidate_action_order_digest"] == ordered_action_mapping_digest(
        candidate
    )
    assert result["actions"][-1]["is_explicit_baseline"] is True
    assert result["frozen_candidate_generator"]["runtime_authorized"] is False
    assert result["runtime_gate_allowed"] is False
    assert result["teacher_value_status"] == "diagnostic_not_match_EV"
    encoded = json.dumps(result, sort_keys=True)
    assert "opponent_private_discards" not in encoded

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield str(key)
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    assert not any("lcb" in key.casefold() for key in keys(result))


def test_hash_baseline_and_actual_stage9f_policy_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(teacher, "_score_actions", _controlled_scorer([]))
    wrong_hash = _Ranker()
    wrong_hash.artifact_sha256 = "b" * 64
    with pytest.raises(ValueError, match="hash"):
        teacher.evaluate_attempt06_t1_second(
            _root(),
            baseline_action_key=_baseline_token(),
            ranker=wrong_hash,
            t2_policies=_policies(),
            config=_config(),
        )
    with pytest.raises(ValueError, match="not legal"):
        teacher.evaluate_attempt06_t1_second(
            _root(),
            baseline_action_key=teacher.ActionKey().to_token(),
            ranker=_Ranker(),
            t2_policies=_policies(),
            config=_config(),
        )
    bad_policies = _policies()
    bad_policies["first"].topk_context["runtime_profile"] = "stage3_baseline"
    with pytest.raises(ValueError, match="runtime_profile"):
        teacher.evaluate_attempt06_t1_second(
            _root(),
            baseline_action_key=_baseline_token(),
            ranker=_Ranker(),
            t2_policies=bad_policies,
            config=_config(),
        )


@pytest.mark.filterwarnings("ignore:X does not have valid feature names")
def test_frozen_lambda_two_loads_are_rank_and_action_mapping_identical() -> None:
    artifact = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "hu_joint_policy"
        / "m43_attempt05_old_dev900_architecture_once"
        / "lambda_rank_candidate.pkl"
    )
    if not artifact.exists():
        pytest.skip("frozen Attempt06 Lambda artifact is not present in this checkout")
    first = teacher.FrozenAttempt06LambdaRanker.load(
        artifact, expected_sha256=MODEL_HASH
    )
    second = teacher.FrozenAttempt06LambdaRanker.load(
        artifact, expected_sha256=MODEL_HASH
    )
    assert first.model.runtime_enabled is False
    assert first.model.winner_frozen is False
    actions = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    mapping_before = tuple(action_key(action).to_token() for action in actions)
    baseline_index = len(actions) - 1
    first_scores = first.score_actions(
        _root(), actions, baseline_index=baseline_index
    )
    second_scores = second.score_actions(
        _root(), actions, baseline_index=baseline_index
    )
    mapping_after = tuple(action_key(action).to_token() for action in actions)
    assert first_scores == second_scores
    assert mapping_before == mapping_after
    assert len(first_scores.mean) == len(mapping_before)


def test_same_seed_controlled_scalar_batch_action_value_delta_digest_parity(
    monkeypatch,
) -> None:
    def parity_scorer(_observation, actions, batch, _selector):
        count = len(batch.particles)
        rows = []
        for action_index, _action in enumerate(actions):
            if count == 8:
                values = tuple(float(20 - action_index) for _ in range(count))
            else:
                values = tuple(
                    float(action_index * 3 - (future_index % 11))
                    for future_index in range(count)
                )
            rows.append(_ActionScores(values))
        return tuple(rows)

    monkeypatch.setattr(teacher, "_score_actions", parity_scorer)
    monkeypatch.setattr(teacher, "_score_actions_batched", parity_scorer)
    scalar = teacher.evaluate_attempt06_t1_second(
        _root(),
        baseline_action_key=_baseline_token(),
        ranker=_Ranker(),
        t2_policies=_policies(),
        config=_config(batch_child_selectors=False),
    )
    batched = teacher.evaluate_attempt06_t1_second(
        _root(),
        baseline_action_key=_baseline_token(),
        ranker=_Ranker(),
        t2_policies=_policies(),
        config=_config(batch_child_selectors=True),
    )

    def parity_payload(result):
        return {
            "legal_action_set_digest": result["legal_action_set_digest"],
            "legal_action_order_digest": result["legal_action_order_digest"],
            "candidate_action_set_digest": result["candidate_action_set_digest"],
            "candidate_action_order_digest": result["candidate_action_order_digest"],
            "selected_action_key": result["selected_action_key"],
            "candidate_belief_digest": result["candidate_belief_digest"],
            "evaluation_belief_digest": result["evaluation_belief_digest"],
            "actions": [
                {
                    "action_key": row["action_key"],
                    "candidate_mean": row["candidate_mean"],
                    "candidate_standard_error": row["candidate_standard_error"],
                    "evaluation_mean": row["evaluation_mean"],
                    "evaluation_standard_error": row["evaluation_standard_error"],
                    "paired": row["paired_evaluation_delta_vs_baseline"],
                }
                for row in result["actions"]
            ],
        }

    scalar_encoded = json.dumps(
        parity_payload(scalar), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    batched_encoded = json.dumps(
        parity_payload(batched), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(scalar_encoded).hexdigest() == hashlib.sha256(
        batched_encoded
    ).hexdigest()
    assert scalar_encoded == batched_encoded


def _root_payload(**overrides):
    actions = generate_turn_actions(_root().hero_board, _root().dealt_cards)
    payload = {
        "schema": teacher.ATTEMPT06_ROOT_SCHEMA,
        "root_index": 0,
        "hand_seed": teacher.ATTEMPT06_HAND_SEED_START,
        "root_profile": teacher.ATTEMPT06_ROOT_PROFILES[0],
        "policy_observation": _root().to_dict(),
        "baseline_action_key": action_key(actions[-1]).to_token(),
        "provenance": {
            "schema": teacher.ATTEMPT06_ROOT_PROVENANCE_SCHEMA,
            "run_name": "attempt06-test",
            "run_id": "attempt06-test:shard=0",
            "root_index": 0,
            "root_profile": teacher.ATTEMPT06_ROOT_PROFILES[0],
            "root_policy_seed_base": teacher.ATTEMPT06_HAND_SEED_START,
            "root_generation_policy": teacher.ATTEMPT06_ROOT_GENERATION_POLICY,
            "baseline_profile": "stage18_p1",
            "plan_sha256": teacher.ATTEMPT06_PLAN_SHA256,
            "schedule_sha256": "b" * 64,
            "candidate_model_sha256": MODEL_HASH,
            "source_model_manifest_sha256": (
                teacher.ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256
            ),
            "source_native_manifest_sha256": (
                teacher.ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256
            ),
            "source_sha256": "f" * 64,
            "startup_sha256": "0" * 64,
            "status_sha256": "1" * 64,
            "source_closure_sha256": "2" * 64,
            "native_batch_threads": teacher.ATTEMPT06_NATIVE_BATCH_THREADS,
            "package_manifest_sha256": "c" * 64,
            "global_consumption_marker_sha256": "d" * 64,
            "root_consumption_claim_sha256": "e" * 64,
            "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
            "current_profile_resolved": False,
            "opponent_private_discard_input_allowed": False,
            "teacher_value_status": "diagnostic_not_match_EV",
            "fresh_audit_retry_or_alternate_sample_allowed": False,
            "deterministic_claim_recovery_allowed": True,
            "deterministic_claim_recovery_mode": (
                teacher.ATTEMPT06_ROOT_REMATERIALIZATION_MODE
            ),
        },
    }
    payload.update(overrides)
    return payload


def test_one_root_shard_input_is_canonical_scheduled_and_hidden_safe(tmp_path) -> None:
    path = tmp_path / "root.jsonl"
    path.write_text(json.dumps(_root_payload()) + "\n", encoding="utf-8")
    roots = teacher.load_attempt06_roots(path)
    assert len(roots) == 1
    assert roots[0].root_index == 0
    assert roots[0].observation == _root()

    second = tmp_path / "two.jsonl"
    second.write_text(
        json.dumps(_root_payload()) + "\n" + json.dumps(_root_payload()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        teacher.load_attempt06_roots(second)

    poison = tmp_path / "poison.jsonl"
    poison.write_text(
        json.dumps(
            _root_payload(
                provenance={
                    **_root_payload()["provenance"],
                    "root_profile": teacher.ATTEMPT06_ROOT_PROFILES[0],
                    "sampled_discards": ["As"],
                }
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="card token"):
        teacher.load_attempt06_roots(poison)

    unknown = tmp_path / "unknown-provenance.jsonl"
    unknown.write_text(
        json.dumps(
            _root_payload(
                provenance={
                    **_root_payload()["provenance"],
                    "sneaky_hidden_hash": "3" * 64,
                }
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provenance fields"):
        teacher.load_attempt06_roots(unknown)

    wrong_profile = tmp_path / "wrong-profile.jsonl"
    wrong_profile.write_text(
        json.dumps(_root_payload(root_profile="stage3_baseline")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mod 5"):
        teacher.load_attempt06_roots(wrong_profile)


def test_resume_truncates_only_uncheckpointed_tail(tmp_path) -> None:
    input_hash = "e" * 64
    model_hash = teacher.ATTEMPT06_FROZEN_MODEL_SHA256
    config_hash = "c" * 64
    row = {
        "schema": teacher.ATTEMPT06_SHARD_ROW_SCHEMA,
        "root_index": 0,
        "provenance": {
            "config_sha256": config_hash,
            "input_sha256": input_hash,
            "model_sha256": model_hash,
            "source_model_manifest_sha256": (
                teacher.ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256
            ),
            "source_native_manifest_sha256": (
                teacher.ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256
            ),
        },
    }
    committed = (json.dumps(row) + "\n").encode("utf-8")
    partial = tmp_path / "out.jsonl.partial"
    partial.write_bytes(committed + b'{"torn":')
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema": teacher.ATTEMPT06_CHECKPOINT_SCHEMA,
                "config_sha256": config_hash,
                "input_sha256": input_hash,
                "model_sha256": model_hash,
                "source_model_manifest_sha256": (
                    teacher.ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256
                ),
                "source_native_manifest_sha256": (
                    teacher.ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256
                ),
                "target_roots": 1,
                "root_index": 0,
                "completed_roots": 1,
                "partial_sha256": __import__("hashlib").sha256(committed).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    assert teacher._resume_one_root(
        partial,
        checkpoint,
        config_sha256=config_hash,
        input_sha256=input_hash,
        model_sha256=model_hash,
        source_model_manifest_sha256=(
            teacher.ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256
        ),
        source_native_manifest_sha256=(
            teacher.ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256
        ),
        expected_root_index=0,
    ) == 1
    assert partial.read_bytes() == committed
    with pytest.raises(ValueError, match="configuration"):
        teacher._resume_one_root(
            partial,
            checkpoint,
            config_sha256="d" * 64,
            input_sha256=input_hash,
            model_sha256=model_hash,
            source_model_manifest_sha256=(
                teacher.ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256
            ),
            source_native_manifest_sha256=(
                teacher.ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256
            ),
            expected_root_index=0,
        )


def test_restart_accepts_preemption_before_partial_and_discards_uncommitted_partial(
    tmp_path,
) -> None:
    input_hash = "e" * 64
    config_hash = "c" * 64
    model_hash = teacher.ATTEMPT06_FROZEN_MODEL_SHA256
    common = {
        "schema": teacher.ATTEMPT06_CHECKPOINT_SCHEMA,
        "config_sha256": config_hash,
        "input_sha256": input_hash,
        "model_sha256": model_hash,
        "source_model_manifest_sha256": (
            teacher.ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256
        ),
        "source_native_manifest_sha256": (
            teacher.ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256
        ),
        "completed_roots": 0,
        "target_roots": 1,
        "root_index": 0,
        "partial_sha256": hashlib.sha256(b"").hexdigest(),
    }
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps(common), encoding="utf-8")
    partial = tmp_path / "out.jsonl.partial"

    assert teacher._resume_one_root(
        partial,
        checkpoint,
        config_sha256=config_hash,
        input_sha256=input_hash,
        model_sha256=model_hash,
        source_model_manifest_sha256=(
            teacher.ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256
        ),
        source_native_manifest_sha256=(
            teacher.ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256
        ),
        expected_root_index=0,
    ) == 0

    checkpoint.unlink()
    partial.write_bytes(b'{"uncheckpointed":true}\n')
    assert teacher._resume_one_root(
        partial,
        checkpoint,
        config_sha256=config_hash,
        input_sha256=input_hash,
        model_sha256=model_hash,
        source_model_manifest_sha256=(
            teacher.ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256
        ),
        source_native_manifest_sha256=(
            teacher.ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256
        ),
        expected_root_index=0,
    ) == 0
    assert partial.read_bytes() == b""


def test_one_root_shard_fsync_checkpoint_finalize_and_no_clobber(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "root.jsonl"
    input_path.write_text(json.dumps(_root_payload()) + "\n", encoding="utf-8")
    output = tmp_path / "result.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    heartbeat = tmp_path / "heartbeat.json"

    monkeypatch.setattr(
        teacher.FrozenAttempt06LambdaRanker,
        "load",
        classmethod(lambda cls, path, *, expected_sha256: _Ranker()),
    )
    monkeypatch.setattr(teacher, "load_model_bundle", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        teacher,
        "build_policy",
        lambda _profile, _bundle, *, seed, seat, opening_lookahead_samples: (
            _Stage9fPolicy(seat)
        ),
    )
    def evaluate_with_live_state(*args, **kwargs):
        running = json.loads(heartbeat.read_text(encoding="utf-8"))
        initial_checkpoint = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert running["status"] == "running"
        assert running["completed_roots"] == 0
        assert initial_checkpoint["completed_roots"] == 0
        return {
            "status": "ok",
            "schema": teacher.ATTEMPT06_TEACHER_SCHEMA,
        }

    monkeypatch.setattr(
        teacher, "evaluate_attempt06_t1_second", evaluate_with_live_state
    )
    monkeypatch.setattr(
        teacher, "_require_concrete_stage9f_p2_policies", lambda policies: None
    )
    with pytest.raises(ValueError, match="run_id disagrees"):
        teacher.run_attempt06_shard(
            input_roots=input_path,
            output=tmp_path / "wrong-run.jsonl",
            checkpoint=tmp_path / "wrong-run-checkpoint.json",
            heartbeat=tmp_path / "wrong-run-heartbeat.json",
            model=tmp_path / "wrong-run-frozen.pkl",
            model_sha256=MODEL_HASH,
            source_model_manifest_sha256=(
                teacher.ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256
            ),
            source_native_manifest_sha256=(
                teacher.ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256
            ),
            run_id="different-run:shard=0",
        )
    assert not (tmp_path / "wrong-run.jsonl").exists()
    summary = teacher.run_attempt06_shard(
        input_roots=input_path,
        output=output,
        checkpoint=checkpoint,
        heartbeat=heartbeat,
        model=tmp_path / "frozen.pkl",
        model_sha256=MODEL_HASH,
        source_model_manifest_sha256=(
            teacher.ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256
        ),
        source_native_manifest_sha256=(
            teacher.ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256
        ),
        run_id="attempt06-test:shard=0",
    )
    assert summary["status"] == "complete"
    assert output.exists()
    assert not (tmp_path / "result.jsonl.partial").exists()
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["schema"] == teacher.ATTEMPT06_SHARD_ROW_SCHEMA
    assert row["root_index"] == 0
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_payload["input_sha256"] == row["provenance"]["input_sha256"]
    assert checkpoint_payload["model_sha256"] == MODEL_HASH
    heartbeat_payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert heartbeat_payload["schema"] == teacher.ATTEMPT06_HEARTBEAT_SCHEMA
    assert heartbeat_payload["summary_schema"] == teacher.ATTEMPT06_SHARD_SUMMARY_SCHEMA
    with pytest.raises(FileExistsError, match="already complete"):
        teacher.run_attempt06_shard(
            input_roots=input_path,
            output=output,
            checkpoint=checkpoint,
            heartbeat=heartbeat,
            model=tmp_path / "frozen.pkl",
            model_sha256=MODEL_HASH,
            source_model_manifest_sha256=(
                teacher.ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256
            ),
            source_native_manifest_sha256=(
                teacher.ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256
            ),
            run_id="attempt06-test:shard=0",
        )


def test_shard_rejects_canonical_path_aliases_before_writing(tmp_path) -> None:
    input_path = tmp_path / "root.jsonl"
    input_path.write_text(json.dumps(_root_payload()) + "\n", encoding="utf-8")
    output = tmp_path / "aliased.jsonl"
    with pytest.raises(ValueError, match="paths must be distinct"):
        teacher.run_attempt06_shard(
            input_roots=input_path,
            output=output,
            checkpoint=tmp_path / "checkpoint.json",
            heartbeat=output,
            model=tmp_path / "frozen.pkl",
            model_sha256=MODEL_HASH,
            source_model_manifest_sha256=(
                teacher.ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256
            ),
            source_native_manifest_sha256=(
                teacher.ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256
            ),
            run_id="attempt06-path-alias-test",
        )
    assert not output.exists()
    assert not (tmp_path / "aliased.jsonl.lock").exists()

    lock_alias_output = tmp_path / "lock-alias.jsonl"
    lock_alias = tmp_path / "lock-alias.jsonl.lock"
    with pytest.raises(ValueError, match="paths must be distinct"):
        teacher.run_attempt06_shard(
            input_roots=input_path,
            output=lock_alias_output,
            checkpoint=lock_alias,
            heartbeat=tmp_path / "lock-alias-heartbeat.json",
            model=tmp_path / "frozen.pkl",
            model_sha256=MODEL_HASH,
            source_model_manifest_sha256=(
                teacher.ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256
            ),
            source_native_manifest_sha256=(
                teacher.ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256
            ),
            run_id="attempt06-lock-path-alias-test",
        )
    assert not lock_alias.exists()


def test_shard_rejects_existing_hardlink_alias_before_truncation(tmp_path) -> None:
    input_path = tmp_path / "root.jsonl"
    original = (json.dumps(_root_payload()) + "\n").encode("utf-8")
    input_path.write_bytes(original)
    output = tmp_path / "hardlink.jsonl"
    partial = tmp_path / "hardlink.jsonl.partial"
    try:
        partial.hardlink_to(input_path)
    except OSError:
        pytest.skip("workspace filesystem does not support hard links")
    with pytest.raises(ValueError, match="hard-links or aliases"):
        teacher.run_attempt06_shard(
            input_roots=input_path,
            output=output,
            checkpoint=tmp_path / "hardlink-checkpoint.json",
            heartbeat=tmp_path / "hardlink-heartbeat.json",
            model=tmp_path / "frozen.pkl",
            model_sha256=MODEL_HASH,
            source_model_manifest_sha256=(
                teacher.ATTEMPT06_SOURCE_MODEL_MANIFEST_SHA256
            ),
            source_native_manifest_sha256=(
                teacher.ATTEMPT06_SOURCE_NATIVE_MANIFEST_SHA256
            ),
            run_id="attempt06-hardlink-alias-test",
        )
    assert input_path.read_bytes() == original
    assert not (tmp_path / "hardlink.jsonl.lock").exists()


def test_shard_process_lock_rejects_a_concurrent_owner(tmp_path) -> None:
    lock_path = tmp_path / "result.jsonl.lock"
    with teacher._ShardFileLock(lock_path):
        with pytest.raises(RuntimeError, match="already owned"):
            with teacher._ShardFileLock(lock_path):
                raise AssertionError("concurrent lock unexpectedly acquired")
