import hashlib
import json
from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from ofc_regular.action_key import action_key, action_key_from_payload, resolve_action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_turn2_teacher_data import evaluate_hu_turn2_actions
from ofc_regular.hu_belief import sample_hidden_card_particles
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_turn2_teacher_data import (
    STATE_PROFILE_COLUMNS,
    aggregate_hu_turn2_profiles,
    write_hu_turn2_state_profile_csv,
)
from ofc_regular.hu_turn3_batch_continuation import (
    HuTurn3ActionCache,
    HuTurn3BatchModels,
    HuTurn3DecisionCache,
    HuTurn3Stage3ReferenceCache,
    HuTurn3Stage7BatchConfig,
    Stage3ReferenceDecision,
    HuTurn3State,
    canonical_t3_state_key,
    decide_hu_turn3_stage7_batch,
    decide_hu_turn3_stage3_reference_batch,
)
from ofc_regular.benchmark_hu_turn3_stage3_features import benchmark_replay
from ofc_regular.hu_turn3_model import hu_policy_sample, sample_to_matrix as hu_sample_to_matrix
from ofc_regular.hu_turn3_stage3_feature_fast import (
    FEATURE_SCHEMA_VERSION,
    Stage3StateFeatureCache,
    build_hu_turn3_stage3_feature_matrix_batch,
)
from ofc_regular.hu_turn3_stage3_feature_manifest import (
    build_hu_turn3_stage3_feature_manifest,
)
from ofc_regular.hu_turn3_stage3_feature_rust import rust_direct_available
from ofc_regular.hu_turn3_stage3_feature_replay import (
    append_stage3_feature_replay,
    expected_scores_path_for_replay,
    load_stage3_feature_replay,
    resolve_stage3_feature_replay_path,
    schema_path_for_replay,
)
from ofc_regular.hu_turn3_stage3_rust_encoder_fixture import write_rust_encoder_fixture
from ofc_regular.policy import RegularAiPolicy, action_to_json
from ofc_regular.state import Board


class FixedIndexModel:
    def __init__(self, index=0, margin=20.0, bad_length=False, nan=False):
        self.index = index
        self.margin = margin
        self.bad_length = bad_length
        self.nan = nan
        self.predict_calls = 0

    def choose_action_index(self, sample):
        return min(self.index, len(sample["actions"]) - 1)

    def predict_sample(self, sample):
        self.predict_calls += 1
        length = len(sample["actions"])
        if self.bad_length:
            return np.zeros(length + 1, dtype=np.float64)
        values = np.zeros(length, dtype=np.float64)
        if self.nan:
            values[:] = np.nan
            return values
        if length:
            values[min(self.index, length - 1)] = self.margin
        return values


class SemanticAhTopModel:
    def predict_sample(self, sample):
        return np.asarray(
            [
                100.0
                if any(
                    card == "Ah" and row == "top"
                    for card, row in action["placements"]
                )
                else 0.0
                for action in sample["actions"]
            ],
            dtype=np.float64,
        )


class FixedGateModel:
    def __init__(self, probability):
        self.probability = probability
        self.calls = 0
        self.estimator = self
        self.classes_ = np.asarray([0, 1])

    def predict_proba(self, matrix):
        self.calls += 1
        return np.asarray(
            [[1.0 - self.probability, self.probability] for _row in matrix],
            dtype=np.float64,
        )


def _t3_board():
    return Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "5c"],
        bottom=["9c", "9d", "9s", "2d"],
    )


def _opponent_t3_board():
    return Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )


def _state():
    return HuTurn3State(
        board=_t3_board(),
        dealt_cards=("Qs", "Ah", "7d"),
        opponent_board=_opponent_t3_board(),
        dead_cards=("2c",),
        seat="first",
        to_act_order="first",
    )


def _second_seat_state():
    return HuTurn3State(
        board=_t3_board(),
        dealt_cards=("Qs", "Ah", "7d"),
        opponent_board=Board.from_rows(
            top=["2h", "3c"],
            middle=["3h", "4h", "5h", "6h"],
            bottom=["7h", "8h", "Th", "Jh", "4c"],
        ),
        dead_cards=("2c", "5d"),
        seat="second",
        to_act_order="second",
    )


def _batch_decision(
    *,
    state=None,
    stage7=None,
    reference=None,
    enabled=True,
    cache=None,
    batch_size=8192,
    models=None,
):
    state = state or _state()
    if models is None:
        fallback = FixedIndexModel(index=0, margin=1.0)
        models = HuTurn3BatchModels(
            fallback_turn3_model=fallback,
            stage7_model=stage7 or FixedIndexModel(index=2, margin=40.0),
            stage3_reference_model=reference or FixedIndexModel(index=1, margin=20.0),
        )
    result = decide_hu_turn3_stage7_batch(
        [state],
        HuTurn3Stage7BatchConfig(stage7_enabled=enabled, use_cache=cache is not None),
        models,
        batch_size,
        cache=cache,
    )
    return result.decisions[0], result.profile, models


@pytest.mark.parametrize(
    "state",
    [_state(), _second_seat_state()],
    ids=["first-seat", "second-seat"],
)
def test_scalar_vs_batch_parity_for_stage7_override(state):
    fallback = FixedIndexModel(index=0, margin=1.0)
    reference = FixedIndexModel(index=1, margin=20.0)
    stage7 = FixedIndexModel(index=2, margin=40.0)
    scalar = RegularAiPolicy(
        turn3_model=fallback,
        hu_turn3_model=stage7,
        hu_turn3_reference_model=reference,
        hu_turn3_min_margin=5.0,
        hu_turn3_reference_min_margin=10.0,
        hu_turn3_decision_log=[],
        seat=state.seat,
    )

    scalar_action = scalar.choose_action(
        state.board,
        state.dealt_cards,
        dead_cards=state.dead_cards,
        opponent_board=state.opponent_board,
    )
    batch_decision, _profile, _models = _batch_decision(
        state=state,
        stage7=stage7,
        reference=reference,
    )
    scalar_record = scalar.hu_turn3_decision_log[-1]

    assert batch_decision.final_action == scalar_action
    assert batch_decision.override_fired == scalar_record["override_fired"]
    assert batch_decision.no_override_reason == scalar_record["no_override_reason"]
    assert batch_decision.stage7_predicted_margin == scalar_record["stage7_predicted_margin"]
    assert batch_decision.reference_margin == scalar_record["reference_margin"]


def test_batch_size_one_matches_large_batch_and_cache_off_matches_cache_on():
    small_decision, _profile, models = _batch_decision(batch_size=1)
    cache = HuTurn3DecisionCache(max_size=8)
    cached_decision, _profile, _models = _batch_decision(
        cache=cache,
        batch_size=8192,
        models=models,
    )
    cached_again, profile_again, _models = _batch_decision(
        cache=cache,
        batch_size=8192,
        models=models,
    )

    assert small_decision.final_action == cached_decision.final_action
    assert small_decision.override_fired == cached_decision.override_fired
    assert cached_again.final_action == cached_decision.final_action
    assert profile_again["cache_hits"] == 1


def test_t3_decision_cache_namespace_changes_with_config_and_every_model():
    state = _state()
    cache = HuTurn3DecisionCache(max_size=32)
    config = HuTurn3Stage7BatchConfig(use_cache=True)
    models = HuTurn3BatchModels(
        fallback_turn3_model=FixedIndexModel(index=0, margin=1.0),
        stage7_model=FixedIndexModel(index=2, margin=40.0),
        stage3_reference_model=FixedIndexModel(index=1, margin=20.0),
        support_model=FixedIndexModel(index=2, margin=5.0),
        gate_model=FixedGateModel(0.9),
    )

    first = decide_hu_turn3_stage7_batch(
        [state],
        config,
        models,
        cache=cache,
    )
    assert first.profile["cache_hits"] == 0

    replacements = {
        "fallback_turn3_model": FixedIndexModel(index=0, margin=1.0),
        "stage7_model": FixedIndexModel(index=2, margin=40.0),
        "stage3_reference_model": FixedIndexModel(index=1, margin=20.0),
        "support_model": FixedIndexModel(index=2, margin=5.0),
        "gate_model": FixedGateModel(0.9),
    }
    for field_name, replacement_model in replacements.items():
        changed = replace(models, **{field_name: replacement_model})
        result = decide_hu_turn3_stage7_batch(
            [state],
            config,
            changed,
            cache=cache,
        )
        assert result.profile["cache_hits"] == 0, field_name

    disabled = decide_hu_turn3_stage7_batch(
        [state],
        replace(config, stage7_enabled=False),
        models,
        cache=cache,
    )
    assert disabled.profile["cache_hits"] == 0
    assert disabled.decisions[0].override_fired is False
    assert disabled.decisions[0].no_override_reason == "stage7_disabled"

    original_again = decide_hu_turn3_stage7_batch(
        [state],
        config,
        models,
        cache=cache,
    )
    assert original_again.profile["cache_hits"] == 1


def test_reference_cache_remaps_semantic_action_across_dealt_permutations():
    first_state = _state()
    second_state = replace(first_state, dealt_cards=("7d", "Qs", "Ah"))
    model = SemanticAhTopModel()
    cache = HuTurn3Stage3ReferenceCache(max_size=8)

    first = decide_hu_turn3_stage3_reference_batch(
        [first_state],
        model,
        cache=cache,
    )
    second = decide_hu_turn3_stage3_reference_batch(
        [second_state],
        model,
        cache=cache,
    )
    fresh = decide_hu_turn3_stage3_reference_batch([second_state], model)

    first_decision = first.decisions[0]
    second_decision = second.decisions[0]
    fresh_decision = fresh.decisions[0]
    assert second.profile["stage3_reference_cache_hit"] == 1
    assert first_decision.stage3_action is not None
    assert second_decision.stage3_action is not None
    assert fresh_decision.stage3_action is not None
    expected_key = action_key(first_decision.stage3_action)
    assert action_key(second_decision.stage3_action) == expected_key
    assert action_key(fresh_decision.stage3_action) == expected_key
    second_actions = generate_turn_actions(
        second_state.board,
        second_state.dealt_cards,
    )
    assert second_decision.action_index == resolve_action_key(
        second_actions,
        expected_key,
    )


def test_stage3_feature_cache_is_namespaced_by_legal_action_order():
    first_state = _state()
    second_state = replace(first_state, dealt_cards=("7d", "Qs", "Ah"))
    first_actions = generate_turn_actions(
        first_state.board,
        first_state.dealt_cards,
    )
    second_actions = generate_turn_actions(
        second_state.board,
        second_state.dealt_cards,
    )
    cache = Stage3StateFeatureCache(max_size=8)

    first = build_hu_turn3_stage3_feature_matrix_batch(
        [first_state],
        [first_actions],
        state_keys=[canonical_t3_state_key(first_state)],
        state_feature_cache=cache,
    )
    second = build_hu_turn3_stage3_feature_matrix_batch(
        [second_state],
        [second_actions],
        state_keys=[canonical_t3_state_key(second_state)],
        state_feature_cache=cache,
    )
    second_again = build_hu_turn3_stage3_feature_matrix_batch(
        [second_state],
        [second_actions],
        state_keys=[canonical_t3_state_key(second_state)],
        state_feature_cache=cache,
    )

    assert first.profile["stage3_state_feature_cache_hit"] == 0
    assert second.profile["stage3_state_feature_cache_hit"] == 0
    assert second_again.profile["stage3_state_feature_cache_hit"] == 1
    assert [action_key_from_payload(row) for row in second.action_encodings] == [
        action_key(action) for action in second_actions
    ]


def test_permutation_cache_on_off_action_key_parity_and_record_rebinding():
    flat = FixedIndexModel(index=0, margin=0.0)
    models = HuTurn3BatchModels(
        fallback_turn3_model=flat,
        stage7_model=None,
        stage3_reference_model=flat,
    )
    config = HuTurn3Stage7BatchConfig(stage7_enabled=False, use_cache=True)
    decision_cache = HuTurn3DecisionCache(max_size=8)
    reference_cache = HuTurn3Stage3ReferenceCache(max_size=8)
    action_cache = HuTurn3ActionCache(max_size=8)
    cached_keys = []
    fresh_keys = []

    for index, dealt in enumerate(permutations(_state().dealt_cards)):
        state = replace(
            _state(),
            dealt_cards=dealt,
            hand_id=f"hand-{index}",
            game_id=f"game-{index}",
            decision_seed=1000 + index,
        )
        cached = decide_hu_turn3_stage7_batch(
            [state],
            config,
            models,
            cache=decision_cache,
            reference_cache=reference_cache,
            action_cache=action_cache,
        ).decisions[0]
        fresh = decide_hu_turn3_stage7_batch(
            [state],
            config,
            models,
        ).decisions[0]
        assert cached.final_action is not None
        assert fresh.final_action is not None
        cached_keys.append(action_key(cached.final_action))
        fresh_keys.append(action_key(fresh.final_action))
        assert cached.record["hand_id"] == state.hand_id
        assert cached.record["game_id"] == state.game_id
        assert cached.record["seed"] == state.decision_seed
        assert cached.record["cards_to_place"] == list(dealt)

    assert cached_keys == fresh_keys
    assert len(set(cached_keys)) == 1


def test_reference_margin_pre_gate_skips_stage7_model():
    stage7 = FixedIndexModel(index=2, margin=40.0)
    decision, profile, _models = _batch_decision(
        stage7=stage7,
        reference=FixedIndexModel(index=1, margin=5.0),
    )

    assert decision.no_override_reason == "below_reference_margin"
    assert decision.override_fired is False
    assert stage7.predict_calls == 0
    assert profile["skipped_by_reference_margin"] == 1
    assert profile["stage7_model_called_count"] == 0


def test_batched_support_margin_can_reject_candidate():
    models = HuTurn3BatchModels(
        fallback_turn3_model=FixedIndexModel(index=0, margin=1.0),
        stage7_model=FixedIndexModel(index=2, margin=40.0),
        stage3_reference_model=FixedIndexModel(index=1, margin=20.0),
        support_model=FixedIndexModel(index=2, margin=1.0),
    )

    result = decide_hu_turn3_stage7_batch(
        [_state()],
        HuTurn3Stage7BatchConfig(
            hu_turn3_min_margin=5.0,
            hu_turn3_reference_min_margin=10.0,
            hu_turn3_min_support_margin=5.0,
        ),
        models,
        8192,
    )

    decision = result.decisions[0]
    assert decision.override_fired is False
    assert decision.no_override_reason == "below_support_margin"
    assert result.profile["support_model_called_count"] == 1


def test_batched_gate_probability_can_reject_candidate():
    gate = FixedGateModel(0.25)
    models = HuTurn3BatchModels(
        fallback_turn3_model=FixedIndexModel(index=0, margin=1.0),
        stage7_model=FixedIndexModel(index=2, margin=40.0),
        stage3_reference_model=FixedIndexModel(index=1, margin=20.0),
        gate_model=gate,
    )

    result = decide_hu_turn3_stage7_batch(
        [_state()],
        HuTurn3Stage7BatchConfig(
            hu_turn3_min_margin=5.0,
            hu_turn3_reference_min_margin=10.0,
            hu_turn3_min_gate_probability=0.7,
        ),
        models,
        8192,
    )

    decision = result.decisions[0]
    assert decision.override_fired is False
    assert decision.no_override_reason == "fallback_to_stage3"
    assert decision.record["gate_probability"] == 0.25
    assert gate.calls == 1
    assert result.profile["gate_model_called_count"] == 1
    assert result.profile["stage3_fallback_recomputed_count"] == 0
    assert result.profile["duplicate_stage3_compute_avoided_count"] == 1


def test_stage3_reference_scalar_vs_batch_parity():
    state = _state()
    reference = FixedIndexModel(index=1, margin=20.0)
    actions = generate_turn_actions(state.board, state.dealt_cards)
    scalar_index = reference.choose_action_index({"actions": [action_to_json(state.board, action) for action in actions]})
    result = decide_hu_turn3_stage3_reference_batch([state], reference, batch_size=1)
    decision = result.decisions[0]

    assert decision.action_index == scalar_index
    assert decision.stage3_action == actions[scalar_index]
    assert decision.reference_margin == 20.0
    assert result.profile["stage3_reference_computed_count"] == 1


def test_stage3_fast_feature_row_equals_scalar_feature_row():
    state = _state()
    actions = generate_turn_actions(state.board, state.dealt_cards)
    sample = hu_policy_sample(
        state.board,
        state.dealt_cards,
        actions,
        opponent_board=state.opponent_board,
        dead_cards=state.dead_cards,
        seat=state.seat,
        to_act_order=state.to_act_order,
    )
    scalar, _targets = hu_sample_to_matrix(sample)
    fast = build_hu_turn3_stage3_feature_matrix_batch(
        [state],
        [actions],
        FEATURE_SCHEMA_VERSION,
    )

    assert fast.feature_column_names == [f"f{index}" for index in range(scalar.shape[1])]
    assert fast.X.dtype == np.float32
    assert fast.X.shape == scalar.shape
    assert np.allclose(fast.X, scalar, rtol=0.0, atol=1e-6, equal_nan=True)
    assert fast.profile["feature_column_count"] == scalar.shape[1]


def test_stage3_numpy_direct_feature_row_equals_scalar_fast_feature_row():
    state = _state()
    actions = generate_turn_actions(state.board, state.dealt_cards)
    scalar_fast = build_hu_turn3_stage3_feature_matrix_batch(
        [state],
        [actions],
        FEATURE_SCHEMA_VERSION,
        encoder_mode="scalar_fast",
    )
    direct = build_hu_turn3_stage3_feature_matrix_batch(
        [state],
        [actions],
        FEATURE_SCHEMA_VERSION,
        encoder_mode="numpy_direct_full",
    )

    assert direct.feature_column_names == scalar_fast.feature_column_names
    assert direct.X.dtype == np.float32
    assert direct.X.shape == scalar_fast.X.shape
    assert np.allclose(direct.X, scalar_fast.X, rtol=0.0, atol=1e-6, equal_nan=True)
    assert direct.profile["stage3_feature_mode"] == "numpy_direct_full"
    assert direct.profile["direct_column_count"] == 1076
    assert direct.profile["scalar_fallback_column_count"] == 0
    assert direct.profile["direct_column_coverage_ratio"] == 1.0
    for key in (
        "stage3_encoder_matrix_build_seconds",
        "stage3_after_board_construction_seconds",
        "stage3_row_summary_seconds",
        "stage3_global_summary_seconds",
        "stage3_action_delta_seconds",
        "stage3_non_encoder_overhead_seconds",
    ):
        assert key in direct.profile


def test_stage3_rust_direct_feature_row_equals_scalar_fast_feature_row():
    if not rust_direct_available():
        pytest.skip("Rust Stage3 feature encoder library is not built")
    state = _state()
    actions = generate_turn_actions(state.board, state.dealt_cards)
    scalar_fast = build_hu_turn3_stage3_feature_matrix_batch(
        [state],
        [actions],
        FEATURE_SCHEMA_VERSION,
        encoder_mode="scalar_fast",
    )
    direct = build_hu_turn3_stage3_feature_matrix_batch(
        [state],
        [actions],
        FEATURE_SCHEMA_VERSION,
        encoder_mode="rust_direct",
    )

    assert direct.feature_column_names == scalar_fast.feature_column_names
    assert direct.X.dtype == np.float32
    assert direct.X.shape == scalar_fast.X.shape
    assert np.allclose(direct.X, scalar_fast.X, rtol=0.0, atol=1e-6, equal_nan=True)
    assert direct.profile["stage3_feature_mode"] == "rust_direct"
    assert direct.profile["direct_column_count"] == 1076
    assert direct.profile["scalar_fallback_column_count"] == 0
    assert direct.profile["direct_column_coverage_ratio"] == 1.0
    assert direct.profile["rust_encoder_core_seconds"] >= 0.0


def test_stage3_feature_manifest_matches_hgb_input_order():
    manifest = build_hu_turn3_stage3_feature_manifest()

    assert manifest["feature_count"] == 1076
    assert manifest["feature_dtype"] == "float32"
    assert manifest["feature_column_names"] == [f"f{index}" for index in range(1076)]
    assert [feature["index"] for feature in manifest["features"]] == list(range(1076))
    assert manifest["checks"]["missing_columns"] == 0
    assert manifest["checks"]["extra_columns"] == 0


def test_disable_fast_path_falls_back_to_scalar_stage3_reference():
    state = _state()
    reference = FixedIndexModel(index=1, margin=20.0)
    fast_disabled = decide_hu_turn3_stage3_reference_batch(
        [state],
        reference,
        batch_size=8192,
        use_fast_feature_path=False,
    )

    assert fast_disabled.decisions[0].action_index == 1
    assert fast_disabled.profile["stage3_feature_mode"] == "scalar"


def test_stage3_reference_cache_on_off_and_batch_size_parity():
    state = _state()
    reference = FixedIndexModel(index=1, margin=20.0)
    uncached = decide_hu_turn3_stage3_reference_batch([state], reference, batch_size=1)
    cache = HuTurn3Stage3ReferenceCache(max_size=8)
    cached = decide_hu_turn3_stage3_reference_batch([state], reference, batch_size=8192, cache=cache)
    cached_again = decide_hu_turn3_stage3_reference_batch([state], reference, batch_size=8192, cache=cache)

    assert uncached.decisions[0].stage3_action == cached.decisions[0].stage3_action
    assert cached_again.decisions[0].stage3_action == cached.decisions[0].stage3_action
    assert cached_again.profile["stage3_reference_cache_hit"] == 1


def test_t3_action_generation_cache_parity():
    state = _state()
    models = HuTurn3BatchModels(
        fallback_turn3_model=FixedIndexModel(index=0, margin=1.0),
        stage7_model=FixedIndexModel(index=2, margin=40.0),
        stage3_reference_model=FixedIndexModel(index=1, margin=20.0),
    )
    uncached = decide_hu_turn3_stage7_batch(
        [state],
        HuTurn3Stage7BatchConfig(use_cache=False),
        models,
        8192,
    )
    action_cache = HuTurn3ActionCache(max_size=8)
    cached = decide_hu_turn3_stage7_batch(
        [state],
        HuTurn3Stage7BatchConfig(use_cache=False),
        models,
        8192,
        action_cache=action_cache,
    )
    cached_again = decide_hu_turn3_stage7_batch(
        [state],
        HuTurn3Stage7BatchConfig(use_cache=False),
        models,
        8192,
        action_cache=action_cache,
    )

    assert uncached.decisions[0].final_action == cached.decisions[0].final_action
    assert cached_again.decisions[0].final_action == cached.decisions[0].final_action
    assert cached_again.profile["t3_action_generation_cache_hit"] == 1


def test_no_duplicate_fallback_hgb_after_reference_pre_gate():
    fallback = FixedIndexModel(index=0, margin=1.0)
    result = decide_hu_turn3_stage7_batch(
        [_state()],
        HuTurn3Stage7BatchConfig(),
        HuTurn3BatchModels(
            fallback_turn3_model=fallback,
            stage7_model=FixedIndexModel(index=2, margin=40.0),
            stage3_reference_model=FixedIndexModel(index=1, margin=5.0),
        ),
        8192,
    )

    assert result.decisions[0].no_override_reason == "below_reference_margin"
    assert fallback.predict_calls == 0
    assert result.profile["stage3_fallback_recomputed_count"] == 0
    assert result.profile["stage3_fallback_reuse_count"] == 1


def test_nan_and_illegal_candidate_fallback_to_stage3():
    nan_decision, _profile, _models = _batch_decision(stage7=FixedIndexModel(index=2, nan=True))
    illegal_decision, _profile, _models = _batch_decision(
        stage7=FixedIndexModel(index=2, bad_length=True)
    )

    assert nan_decision.no_override_reason == "nan_prediction"
    assert nan_decision.final_action == nan_decision.stage3_action
    assert illegal_decision.no_override_reason == "illegal_candidate"
    assert illegal_decision.final_action == illegal_decision.stage3_action


def test_stage7_off_matches_stage3_reference_action():
    decision, _profile, _models = _batch_decision(enabled=False)
    actions = generate_turn_actions(_t3_board(), ("Qs", "Ah", "7d"))

    assert decision.override_fired is False
    assert decision.final_action == actions[1]


def test_deterministic_canonical_output_hash():
    def digest_for_run():
        decision, _profile, _models = _batch_decision()
        payload = {
            "final_action": action_to_json(_t3_board(), decision.final_action),
            "override_fired": decision.override_fired,
            "no_override_reason": decision.no_override_reason,
            "stage7_predicted_margin": decision.stage7_predicted_margin,
            "reference_margin": decision.reference_margin,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    assert digest_for_run() == digest_for_run()


def test_batched_turn2_teacher_preserves_common_future_digest():
    policy = RegularAiPolicy(
        turn3_model=FixedIndexModel(index=0, margin=1.0),
        hu_turn3_model=FixedIndexModel(index=2, margin=40.0),
        hu_turn3_reference_model=FixedIndexModel(index=1, margin=20.0),
        hu_turn3_min_margin=5.0,
        hu_turn3_reference_min_margin=10.0,
        seed=1,
    )
    board = Board.from_rows(
            top=["Qh"],
            middle=["Kh", "Kd", "6c"],
            bottom=["9c", "9d", "9s"],
        )
    opponent = Board.from_rows(
            top=["2h"],
            middle=["3h", "4h", "5h"],
            bottom=["7h", "8h", "Th"],
        )
    dealt = ("Qs", "Ah", "7d")
    observation = ActorObservation(
        hero_board=board,
        opponent_public_board=opponent,
        dealt_cards=dealt,
        hero_private_discards=("2c",),
        seat="first",
        street="T2",
        to_act_order="first",
    )
    belief = sample_hidden_card_particles(
        observation,
        base_seed=99,
        run_id="t2_batch_common_future",
        sample_count=2,
    )
    sample = evaluate_hu_turn2_actions(
        board=board,
        dealt_cards=dealt,
        opponent_board=opponent,
        hero_seat="first",
        continuation_policy=policy,
        opponent_policy=policy,
        baseline_turn2_model=FixedIndexModel(index=0, margin=1.0),
        future_samples=2,
        future_rollout_seed=99,
        use_batched_continuation=True,
        batched_continuation_config=HuTurn3Stage7BatchConfig(),
        batched_continuation_cache=HuTurn3DecisionCache(max_size=32),
        batched_continuation_batch_size=1,
        observation=observation,
        belief_batch=belief,
    )

    assert sample is not None
    assert all(
        action["common_random_future_digest"] == sample["common_random_future_digest"]
        for action in sample["actions"]
    )
    assert sample["profiling"]["raw_t3_states"] > 0
    assert "opponent_policy_decision_time" in sample["profiling"]
    assert "non_t3_turn_T2_seconds" in sample["profiling"]
    assert "stage3_reference_cache_hit_rate" in sample["profiling"]


@pytest.mark.parametrize("hero_seat", ["first", "second"])
def test_t2_full_rollout_scalar_batch_parity_with_asymmetric_actor_policies(hero_seat):
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c"],
        bottom=["9c", "9d", "9s"],
    )
    if hero_seat == "first":
        opponent = Board.from_rows(
            top=["2h"],
            middle=["3h", "4h", "5h"],
            bottom=["7h", "8h", "Th"],
        )
    else:
        opponent = Board.from_rows(
            top=["2h"],
            middle=["3h", "4h", "5h", "Jh"],
            bottom=["7h", "8h", "Th", "6h"],
        )
    dealt = ("Qs", "Ah", "7d")
    observation = ActorObservation(
        hero_board=board,
        opponent_public_board=opponent,
        dealt_cards=dealt,
        hero_private_discards=("2c",),
        seat=hero_seat,
        street="T2",
        to_act_order=hero_seat,
    )
    belief = sample_hidden_card_particles(
        observation,
        base_seed=991,
        run_id=f"scalar_batch_asymmetric|seat={hero_seat}",
        sample_count=2,
    )

    def actor_policies():
        opponent_seat = "second" if hero_seat == "first" else "first"
        hero = RegularAiPolicy(
            turn1_model=FixedIndexModel(index=0, margin=3.0),
            turn2_model=FixedIndexModel(index=1, margin=4.0),
            turn3_model=FixedIndexModel(index=0, margin=5.0),
            hu_turn3_model=FixedIndexModel(index=2, margin=40.0),
            hu_turn3_reference_model=FixedIndexModel(index=1, margin=20.0),
            hu_turn3_min_margin=5.0,
            hu_turn3_reference_min_margin=10.0,
            seed=1,
            seat=hero_seat,
        )
        opponent_policy = RegularAiPolicy(
            turn1_model=FixedIndexModel(index=2, margin=6.0),
            turn2_model=FixedIndexModel(index=3, margin=7.0),
            turn3_model=FixedIndexModel(index=4, margin=8.0),
            hu_turn3_model=FixedIndexModel(index=5, margin=50.0),
            hu_turn3_reference_model=FixedIndexModel(index=3, margin=25.0),
            hu_turn3_min_margin=5.0,
            hu_turn3_reference_min_margin=10.0,
            seed=2,
            seat=opponent_seat,
        )
        return hero, opponent_policy

    def evaluate(*, batched: bool):
        hero, opponent_policy = actor_policies()
        return evaluate_hu_turn2_actions(
            board=board,
            dealt_cards=dealt,
            opponent_board=opponent,
            hero_seat=hero_seat,
            continuation_policy=hero,
            opponent_policy=opponent_policy,
            baseline_turn2_model=FixedIndexModel(index=0, margin=1.0),
            future_samples=2,
            future_rollout_seed=991,
            action_indices=(0, 1),
            use_batched_continuation=batched,
            batched_continuation_config=HuTurn3Stage7BatchConfig(
                hu_turn3_min_margin=5.0,
                hu_turn3_reference_min_margin=10.0,
                use_cache=False,
            ),
            batched_continuation_batch_size=1,
            use_final_turn_cache=False,
            observation=observation,
            belief_batch=belief,
        )

    scalar = evaluate(batched=False)
    batch = evaluate(batched=True)
    assert scalar is not None and batch is not None

    def values(sample):
        return {
            row["canonical_action_key"]: (
                row["score"],
                row["rollout_count"],
            )
            for row in sample["actions"]
        }

    assert values(batch) == values(scalar)
    assert batch["common_random_future_digest"] == scalar["common_random_future_digest"]


def test_summary_aggregation_reduce_rules_and_state_profile_csv(tmp_path):
    profiles = [
        {
            "seconds_total": 10.0,
            "t3_continuation_total_seconds": 2.0,
            "raw_t3_states": 100.0,
            "unique_t3_states": 80.0,
            "memory_peak_mb": 1000.0,
            "feature_column_count": 1076.0,
            "feature_dtype": "float32",
            "stage3_feature_mode": "fast",
            "stage3_feature_generation_time": 4.0,
            "stage3_hgb_predict_time": 1.0,
            "stage3_feature_rows": 400.0,
            "stage3_reference_cache_hit": 10.0,
            "stage3_reference_cache_miss": 90.0,
            "legal_actions": 24.0,
        },
        {
            "seconds_total": 20.0,
            "t3_continuation_total_seconds": 4.0,
            "raw_t3_states": 200.0,
            "unique_t3_states": 120.0,
            "memory_peak_mb": 1500.0,
            "feature_column_count": 1076.0,
            "feature_dtype": "float32",
            "stage3_feature_mode": "fast",
            "stage3_feature_generation_time": 8.0,
            "stage3_hgb_predict_time": 3.0,
            "stage3_feature_rows": 800.0,
            "stage3_reference_cache_hit": 30.0,
            "stage3_reference_cache_miss": 70.0,
            "legal_actions": 27.0,
        },
    ]

    summary = aggregate_hu_turn2_profiles(profiles)

    assert summary["feature_column_count"] == 1076.0
    assert summary["memory_peak_mb"] == 1500.0
    assert summary["ms_per_raw_t3_continuation_decision"] == 20.0
    assert summary["unique_raw_ratio"] == 200.0 / 300.0
    assert summary["stage3_reference_cache_hit_rate"] == 40.0 / 200.0
    assert summary["max_legal_actions"] == 27.0

    output = tmp_path / "state_profile.csv"
    write_hu_turn2_state_profile_csv(profiles, output)
    header = output.read_text().splitlines()[0].split(",")
    assert header == STATE_PROFILE_COLUMNS


def test_stage3_feature_replay_dump_load_and_benchmark(tmp_path):
    state = _state()
    actions = generate_turn_actions(state.board, state.dealt_cards)
    replay_path = tmp_path / "stage3_feature_replay.jsonl"
    predictions = np.arange(len(actions), dtype=np.float64)
    append_stage3_feature_replay(
        replay_path,
        states=[state],
        actions_by_state=[actions],
        state_keys=[("test", "state")],
        decisions=[
            Stage3ReferenceDecision(
                stage3_action=actions[-1],
                action_index=len(actions) - 1,
                reference_margin=1.0,
                score=float(predictions[-1]),
                rank_score=float(predictions[-1]),
                action_count=len(actions),
                legality_status="legal",
                fallback_reason="",
            )
        ],
        predictions=[predictions],
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_column_names=[f"f{index}" for index in range(1076)],
        feature_dtype="float32",
        profile={"stage3_feature_rows": len(actions), "feature_column_count": 1076},
        metadata={"stage3_reference_model_path": ""},
        sample_limit=1,
    )

    assert replay_path.exists()
    assert schema_path_for_replay(replay_path).exists()
    assert expected_scores_path_for_replay(replay_path).exists()
    replay = load_stage3_feature_replay(replay_path)
    assert len(replay.states) == 1
    assert len(replay.actions_by_state[0]) == len(actions)

    summary = benchmark_replay(
        input_path=replay_path,
        schema_path=schema_path_for_replay(replay_path),
        output_path=tmp_path / "bench.json",
        batch_size=8192,
        limit_states=0,
        repeat=1,
        check_model_scores=False,
        encoder_mode="scalar_fast",
        compare_encoder_modes=False,
        manifest_output=None,
    )
    assert summary["states"] == 1
    assert summary["actions"] == len(actions)
    assert summary["feature_columns"] == 1076

    compare_summary = benchmark_replay(
        input_path=replay_path,
        schema_path=schema_path_for_replay(replay_path),
        output_path=tmp_path / "bench_compare.json",
        batch_size=8192,
        limit_states=0,
        repeat=1,
        check_model_scores=False,
        encoder_mode="scalar_fast",
        compare_encoder_modes=True,
        manifest_output=tmp_path / "manifest.json",
    )
    assert set(compare_summary["mode_results"]) == {
        "scalar_fast",
        "numpy_direct_partial",
        "numpy_direct_full",
    }
    assert compare_summary["mode_results"]["numpy_direct_full"]["feature_columns"] == 1076
    assert compare_summary["feature_parity_vs_scalar_fast"]["numpy_direct_full"]["allclose_1e_6"]
    assert compare_summary["reference_parity_vs_scalar_fast"]["numpy_direct_full"]["checked"] is False
    assert compare_summary["attribution_by_mode"]["numpy_direct_full"]["direct_column_count"] == 1076
    assert (tmp_path / "manifest.json").exists()


def test_representative_replay_directory_path_and_rust_fixture(tmp_path):
    state = _state()
    actions = generate_turn_actions(state.board, state.dealt_cards)
    replay_dir = tmp_path / "replay_dir"
    predictions = np.arange(len(actions), dtype=np.float64)
    metadata = {
        "replay_source": "mc512_1state",
        "source_teacher_run_hash": "abc123",
        "stage3_reference_model_path": "",
        "stage7_model_path": "stage7.pt",
        "raw_t3_states": 1,
    }
    append_stage3_feature_replay(
        replay_dir,
        states=[state],
        actions_by_state=[actions],
        state_keys=[("test", "state")],
        decisions=[
            Stage3ReferenceDecision(
                stage3_action=actions[-1],
                action_index=len(actions) - 1,
                reference_margin=1.0,
                score=float(predictions[-1]),
                rank_score=float(predictions[-1]),
                action_count=len(actions),
                legality_status="legal",
                fallback_reason="",
            )
        ],
        predictions=[predictions],
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_column_names=[f"f{index}" for index in range(1076)],
        feature_dtype="float32",
        profile={
            "stage3_feature_rows": len(actions),
            "feature_column_count": 1076,
            "stage3_feature_mode": "numpy_direct_full",
            "direct_column_count": 1076,
            "scalar_fallback_column_count": 0,
            "direct_column_coverage_ratio": 1.0,
        },
        metadata=metadata,
        sample_limit=1,
    )

    replay_path = resolve_stage3_feature_replay_path(replay_dir, metadata)
    assert replay_path.name == "stage3_feature_replay_mc512_1state.jsonl"
    assert replay_path.exists()
    record = json.loads(replay_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["profile"]["direct_column_count"] == 1076
    assert record["profile"]["scalar_fallback_column_count"] == 0
    assert record["profile"]["feature_group_distribution"]
    assert record["profile"]["source_teacher_run_hash"] == "abc123"

    fixture_path = tmp_path / "fixture.npz"
    spec_path = tmp_path / "spec.md"
    summary = write_rust_encoder_fixture(
        input_path=replay_path,
        spec_output=spec_path,
        fixture_output=fixture_path,
    )
    assert summary["feature_columns"] == 1076
    assert spec_path.exists()
    assert fixture_path.exists()
    fixture = np.load(fixture_path)
    assert fixture["hero_board_masks"].shape == (1, 3)
    assert fixture["expected_features"].shape[1] == 1076
