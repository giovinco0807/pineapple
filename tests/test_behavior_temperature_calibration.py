import copy
from fractions import Fraction

import pytest

from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.tutor.behavior_calibration_contract import (
    build_behavior_decision_log,
    canonical_sha256,
    commit_hidden_root_trace,
    root_split,
)
from ai.tutor.behavior_temperature_calibration import (
    ALLOW_DEV_SELECTED_CANDIDATE,
    CALIBRATION_SCHEMA,
    GATE_CONFIG_SCHEMA,
    GATE_RESULT_SCHEMA,
    MODEL_EVALUATION_SCHEMA,
    REQUIRE_NONIDENTITY_FIT,
    build_behavior_temperature_calibration,
    build_model_evaluation_row,
    build_temperature_gate_config,
    verify_behavior_temperature_calibration,
    verify_model_evaluation_row,
    verify_temperature_gate_config,
)
from ai.tutor.exact_late import action_key
from ai.tutor.t3_hu_full_card_range import BehaviorInfoSet
from ai.tutor.t3_hu_public_cfr import PrivateRecall


ROWS = ("top", "middle", "bottom")
BB_T0 = (("2c", "3d", "4h"), ("7c",), ("Qc",))
BTN_T0 = (("5c", "6d", "8c"), ("9c",), ("As",))
T0_HISTORY = (
    (
        0,
        "bb",
        (
            ("Qc", "bottom"),
            ("7c", "middle"),
            ("2c", "top"),
            ("3d", "top"),
            ("4h", "top"),
        ),
    ),
    (
        0,
        "btn",
        (
            ("As", "bottom"),
            ("9c", "middle"),
            ("5c", "top"),
            ("6d", "top"),
            ("8c", "top"),
        ),
    ),
)
POLICY_SHA256 = "a" * 64
SAMPLING_CONTRACT = {
    "name": "synthetic_observed_behavior_v1",
    "temperature_numerator": 1,
    "temperature_denominator": 1,
}


def _board(rows):
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def _legal_ids(draw, rows):
    return tuple(
        sorted(action_key(action) for action in get_turn_actions(list(draw), _board(rows)))
    )


def _append_placements(rows, placements):
    out = {row: list(cards) for row, cards in zip(ROWS, rows)}
    for card, row in placements:
        out[row].append(card)
    return tuple(tuple(sorted(out[row])) for row in ROWS)


def _canonical_placements(action):
    return tuple(sorted(action.placements, key=lambda item: (item[1], item[0])))


def _first_action(information):
    rows = information.board_bb if information.actor == "bb" else information.board_btn
    return min(
        get_turn_actions(list(information.current_draw), _board(rows)), key=action_key
    )


def _t1_bb_information(draw=("Ad", "Kd", "Qd")):
    draw = tuple(sorted(draw))
    return BehaviorInfoSet(
        actor="bb",
        turn=1,
        board_bb=BB_T0,
        board_btn=BTN_T0,
        public_action_history=T0_HISTORY,
        own_recall_before=PrivateRecall(),
        current_draw=draw,
        legal_action_ids=_legal_ids(draw, BB_T0),
    )


def _t1_btn_information(draw=("2h", "3h", "5h")):
    bb_info = _t1_bb_information()
    bb_action = _first_action(bb_info)
    bb_rows = _append_placements(BB_T0, bb_action.placements)
    history = T0_HISTORY + ((1, "bb", _canonical_placements(bb_action)),)
    draw = tuple(sorted(draw))
    return BehaviorInfoSet(
        actor="btn",
        turn=1,
        board_bb=bb_rows,
        board_btn=BTN_T0,
        public_action_history=history,
        own_recall_before=PrivateRecall(),
        current_draw=draw,
        legal_action_ids=_legal_ids(draw, BTN_T0),
    )


def _t2_bb_information(draw=("Ah", "Jd", "Td")):
    bb_t1 = _t1_bb_information()
    bb_action = _first_action(bb_t1)
    bb_rows = _append_placements(BB_T0, bb_action.placements)
    btn_t1 = _t1_btn_information()
    btn_action = _first_action(btn_t1)
    btn_rows = _append_placements(BTN_T0, btn_action.placements)
    history = T0_HISTORY + (
        (1, "bb", _canonical_placements(bb_action)),
        (1, "btn", _canonical_placements(btn_action)),
    )
    draw = tuple(sorted(draw))
    recall = PrivateRecall(
        dealt_by_turn=((1, bb_t1.current_draw),),
        discards_by_turn=((1, bb_action.discard),),
    )
    return BehaviorInfoSet(
        actor="bb",
        turn=2,
        board_bb=bb_rows,
        board_btn=btn_rows,
        public_action_history=history,
        own_recall_before=recall,
        current_draw=draw,
        legal_action_ids=_legal_ids(draw, bb_rows),
    )


def _t2_btn_information(draw=("4d", "8d", "9d")):
    bb_t1 = _t1_bb_information()
    bb_t1_action = _first_action(bb_t1)
    bb_rows_t1 = _append_placements(BB_T0, bb_t1_action.placements)
    btn_t1 = _t1_btn_information()
    btn_t1_action = _first_action(btn_t1)
    btn_rows_t1 = _append_placements(BTN_T0, btn_t1_action.placements)
    bb_t2 = _t2_bb_information()
    bb_t2_action = _first_action(bb_t2)
    bb_rows_t2 = _append_placements(bb_rows_t1, bb_t2_action.placements)
    history = T0_HISTORY + (
        (1, "bb", _canonical_placements(bb_t1_action)),
        (1, "btn", _canonical_placements(btn_t1_action)),
        (2, "bb", _canonical_placements(bb_t2_action)),
    )
    draw = tuple(sorted(draw))
    recall = PrivateRecall(
        dealt_by_turn=((1, btn_t1.current_draw),),
        discards_by_turn=((1, btn_t1_action.discard),),
    )
    return BehaviorInfoSet(
        actor="btn",
        turn=2,
        board_bb=bb_rows_t2,
        board_btn=btn_rows_t1,
        public_action_history=history,
        own_recall_before=recall,
        current_draw=draw,
        legal_action_ids=_legal_ids(draw, btn_rows_t1),
    )


INFO_BUILDERS = {
    "t1_bb": _t1_bb_information,
    "t1_btn": _t1_btn_information,
    "t2_bb": _t2_bb_information,
    "t2_btn": _t2_btn_information,
}
CHALLENGE_DRAWS = {
    "t1_bb": {
        0: ("Ad", "Kd", "Qd"),
        1: ("Ad", "Kd", "X1"),
        2: ("Ad", "X1", "X2"),
    },
    "t1_btn": {
        0: ("2h", "3h", "5h"),
        1: ("2h", "3h", "X1"),
        2: ("2h", "X1", "X2"),
    },
    "t2_bb": {
        0: ("Ah", "Jd", "Td"),
        1: ("Ah", "Jd", "X1"),
        2: ("Ah", "X1", "X2"),
    },
    "t2_btn": {
        0: ("4d", "8d", "9d"),
        1: ("4d", "8d", "X2"),
        2: ("4d", "X1", "X2"),
    },
}


def _record(information, *, root_id, observed="first"):
    rows = information.board_bb if information.actor == "bb" else information.board_btn
    actions = sorted(
        get_turn_actions(list(information.current_draw), _board(rows)), key=action_key
    )
    if isinstance(observed, int) and not isinstance(observed, bool):
        chosen = actions[observed]
    elif observed == "first":
        chosen = actions[0]
    elif observed == "last":
        chosen = actions[-1]
    else:
        raise ValueError("observed must be 'first', 'last', or an action index")
    return build_behavior_decision_log(
        root_id=root_id,
        root_commitment=commit_hidden_root_trace({"root": root_id, "deck": "fixed"}),
        seed_namespace="synthetic-calibration-seed-v1",
        behavior_target_id="synthetic-behavior-population-v1",
        policy_sha256=POLICY_SHA256,
        sampling_contract=SAMPLING_CONTRACT,
        information=information,
        observed_action_key=action_key(chosen),
        source_action_probability=Fraction(1, information.legal_action_count),
    )


def _root_for(split, suffix):
    for index in range(100_000):
        root = f"main-{split}-{suffix}-{index}"
        if root_split(root) == split:
            return root
    raise AssertionError("could not find split root")


class _SyntheticPreTemperatureEvaluator:
    model_sha256 = "b" * 64
    row_extractor_sha256 = "c" * 64
    adapter_source_sha256 = "d" * 64

    def __init__(self, role):
        self.checkpoint_sha256 = canonical_sha256({"role": role, "checkpoint": "v1"})

    def pre_temperature_legal_logits(self, information, legal_action_ids):
        del information
        return [2.0 if index == 0 else 0.0 for index, _ in enumerate(legal_action_ids)]


def _evaluator_for(record):
    return _SyntheticPreTemperatureEvaluator(f"t{record['turn']}_{record['actor']}")


class _IdentityCalibratedEvaluator(_SyntheticPreTemperatureEvaluator):
    model_sha256 = "e" * 64
    row_extractor_sha256 = "f" * 64

    def pre_temperature_legal_logits(self, information, legal_action_ids):
        del information
        return [0.0 for _ in legal_action_ids]


def _identity_evaluator_for(record):
    return _IdentityCalibratedEvaluator(f"t{record['turn']}_{record['actor']}")


def _dataset(*, dev_observed="first", include_joker2=True):
    records = []
    for split in ("fit", "dev", "test"):
        root = _root_for(split, "complete")
        for role, builder in INFO_BUILDERS.items():
            observed = dev_observed if split == "dev" else "first"
            records.append(_record(builder(), root_id=root, observed=observed))
    rows = [build_model_evaluation_row(record, _evaluator_for(record)) for record in records]

    challenge_records = []
    for joker_count in range(3):
        if joker_count == 2 and not include_joker2:
            continue
        root = f"m3-joker-challenge-v1/joker-{joker_count}"
        for role, builder in INFO_BUILDERS.items():
            challenge_records.append(
                _record(
                    builder(CHALLENGE_DRAWS[role][joker_count]),
                    root_id=root,
                    observed="first",
                )
            )
    challenge_rows = [
        build_model_evaluation_row(record, _evaluator_for(record))
        for record in challenge_records
    ]
    return records, rows, challenge_records, challenge_rows


def _identity_calibrated_dataset():
    """Uniform outcomes for uniform logits make T=1 exactly calibrated."""

    records = []
    for split in ("fit", "dev", "test"):
        for role, builder in INFO_BUILDERS.items():
            information = builder()
            action_count = len(information.legal_action_ids)
            for observed_index in range(action_count):
                records.append(
                    _record(
                        information,
                        root_id=_root_for(
                            split, f"identity-{role}-{observed_index}"
                        ),
                        observed=observed_index,
                    )
                )
    rows = [
        build_model_evaluation_row(record, _identity_evaluator_for(record))
        for record in records
    ]

    challenge_records = []
    for joker_count in range(3):
        root = f"m3-joker-challenge-v1/identity-joker-{joker_count}"
        for role, builder in INFO_BUILDERS.items():
            challenge_records.append(
                _record(
                    builder(CHALLENGE_DRAWS[role][joker_count]),
                    root_id=root,
                    observed=0,
                )
            )
    challenge_rows = [
        build_model_evaluation_row(record, _identity_evaluator_for(record))
        for record in challenge_records
    ]
    return records, rows, challenge_records, challenge_rows


def _small_gate(
    *, require_joker_challenge=True, require_nonidentity_temperature=False
):
    return build_temperature_gate_config(
        gate_id="synthetic_temperature_gate_v2",
        min_fit_decisions_per_role=1,
        min_dev_decisions_per_role=1,
        min_test_decisions_per_role=1,
        min_challenge_decisions_per_role_joker=1,
        min_roots_per_split_role=1,
        min_challenge_roots_per_role_joker=1,
        require_joker_challenge=require_joker_challenge,
        require_nonidentity_temperature=require_nonidentity_temperature,
        bootstrap_replicates=20,
        max_test_nll_delta_vs_t1=100,
        max_test_nll_delta_vs_t1_ucb=100,
        min_test_uniform_improvement_role_lcb=-100,
        min_test_uniform_improvement_joker_lcb=-100,
        max_test_role_marginal_ece=1,
        max_test_joker_marginal_ece=1,
        max_test_brier_delta_vs_t1=1,
        min_test_residual_temperature=Fraction(1, 20),
        max_test_residual_temperature=20,
    )


def test_positive_artifact_is_deterministic_content_bound_and_gate_passes():
    records, rows, challenge_records, challenge_rows = _dataset()
    config = _small_gate()
    artifact = build_behavior_temperature_calibration(
        records,
        rows,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        gate_config=config,
    )
    replay = build_behavior_temperature_calibration(
        list(reversed(records)),
        list(reversed(rows)),
        challenge_records=list(reversed(challenge_records)),
        challenge_evaluation_rows=list(reversed(challenge_rows)),
        gate_config=config,
    )

    assert artifact == replay
    assert artifact["schema"] == CALIBRATION_SCHEMA
    assert artifact["gate_config"]["schema"] == GATE_CONFIG_SCHEMA
    assert artifact["gate_result"]["schema"] == GATE_RESULT_SCHEMA
    assert artifact["promotion_eligible"] is True
    assert artifact["gate_result"]["all_required_gates_passed"] is True
    assert artifact["selection_contract"]["locked_test_was_selection_input"] is False
    assert artifact["joker_challenge"]["used_for_temperature_fit"] is False
    assert all(
        role["dev_selected_candidate"] == "fit_temperature"
        for role in artifact["temperatures"].values()
    )
    assert all(
        role["final_temperature"]["denominator"] == 1_000_000
        for role in artifact["temperatures"].values()
    )
    assert verify_behavior_temperature_calibration(
        records,
        rows,
        artifact,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
    ) == artifact


def test_perfectly_calibrated_identity_selection_passes_default_gate():
    records, rows, challenge_records, challenge_rows = _identity_calibrated_dataset()
    artifact = build_behavior_temperature_calibration(
        records,
        rows,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        gate_config=_small_gate(),
    )

    assert artifact["promotion_eligible"] is True
    assert artifact["gate_config"]["selection_policy"] == (
        ALLOW_DEV_SELECTED_CANDIDATE
    )
    assert artifact["selection_contract"]["selection_frozen_before_locked_test"] is True
    assert artifact["selection_contract"]["published_metrics_temperature_bound"] is True
    for role, temperature in artifact["temperatures"].items():
        assert temperature["dev_selected_candidate"] == "identity_temperature"
        assert temperature["fit_temperature"] == {
            "numerator": 1_000_000,
            "denominator": 1_000_000,
        }
        assert temperature["final_temperature"] == temperature["fit_temperature"]
        assert temperature["dev_selection_frozen_before_locked_test"] is True
        assert artifact["metrics"]["test"]["by_role"][role][
            "applied_temperature"
        ] == temperature["final_temperature"]
    assert not any(
        name.startswith("selection.")
        for name in artifact["gate_result"]["failures"]
    )
    assert verify_behavior_temperature_calibration(
        records,
        rows,
        artifact,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
    ) == artifact


def test_explicit_strict_nonidentity_policy_rejects_identity_selection():
    records, rows, challenge_records, challenge_rows = _identity_calibrated_dataset()
    artifact = build_behavior_temperature_calibration(
        records,
        rows,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        gate_config=_small_gate(require_nonidentity_temperature=True),
    )

    assert artifact["gate_config"]["selection_policy"] == REQUIRE_NONIDENTITY_FIT
    assert artifact["promotion_eligible"] is False
    assert all(
        f"selection.{role}.promotion_policy"
        in artifact["gate_result"]["failures"]
        for role in INFO_BUILDERS
    )


def test_v2_selection_policy_config_is_content_bound_and_fail_closed():
    config = _small_gate()
    assert verify_temperature_gate_config(config) == config
    assert config["schema"] == "ofc_behavior_temperature_gate_config/v2"

    legacy = copy.deepcopy(config)
    legacy["schema"] = "ofc_behavior_temperature_gate_config/v1"
    legacy["gate_config_sha256"] = canonical_sha256(
        {key: value for key, value in legacy.items() if key != "gate_config_sha256"}
    )
    with pytest.raises(ValueError, match="unsupported temperature gate config schema"):
        verify_temperature_gate_config(legacy)

    unknown_policy = copy.deepcopy(config)
    unknown_policy["selection_policy"] = "accept_any_temperature"
    unknown_policy["gate_config_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in unknown_policy.items()
            if key != "gate_config_sha256"
        }
    )
    with pytest.raises(ValueError, match="unsupported temperature dev-selection"):
        verify_temperature_gate_config(unknown_policy)

    with pytest.raises(TypeError, match="require_nonidentity_temperature must be bool"):
        build_temperature_gate_config(require_nonidentity_temperature=1)


def test_evaluation_row_and_artifact_tampering_fail_closed():
    records, rows, challenge_records, challenge_rows = _dataset()
    assert rows[0]["schema"] == MODEL_EVALUATION_SCHEMA
    verify_model_evaluation_row(rows[0], records[0])

    tampered_row = copy.deepcopy(rows[0])
    tampered_row["logits_f64_hex"][0] = (0.0).hex()
    with pytest.raises(ValueError, match="model evaluation row SHA-256 mismatch"):
        verify_model_evaluation_row(tampered_row, records[0])

    artifact = build_behavior_temperature_calibration(
        records,
        rows,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        gate_config=_small_gate(),
    )
    resigned_rows = copy.deepcopy(rows)
    resigned_rows[0]["logits_f64_hex"][0] = (0.0).hex()
    resigned_rows[0]["evaluation_row_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in resigned_rows[0].items()
            if key != "evaluation_row_sha256"
        }
    )
    # A self-consistent row is still bound by the published artifact's row-set
    # commitment and raw-derived metric replay.
    verify_model_evaluation_row(resigned_rows[0], records[0])
    with pytest.raises(ValueError, match="does not match raw inputs"):
        verify_behavior_temperature_calibration(
            records,
            resigned_rows,
            artifact,
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
        )

    tampered_selection = copy.deepcopy(artifact)
    tampered_selection["temperatures"]["t1_bb"]["dev_selected_candidate"] = (
        "identity_temperature"
    )
    tampered_selection["artifact_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in tampered_selection.items()
            if key != "artifact_sha256"
        }
    )
    with pytest.raises(ValueError, match="does not match raw inputs"):
        verify_behavior_temperature_calibration(
            records,
            rows,
            tampered_selection,
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
        )

    tampered_metric_binding = copy.deepcopy(artifact)
    tampered_metric_binding["metrics"]["test"]["by_role"]["t1_bb"][
        "applied_temperature"
    ]["numerator"] += 1
    tampered_metric_binding["artifact_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in tampered_metric_binding.items()
            if key != "artifact_sha256"
        }
    )
    with pytest.raises(ValueError, match="does not match raw inputs"):
        verify_behavior_temperature_calibration(
            records,
            rows,
            tampered_metric_binding,
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
        )

    tampered_artifact = copy.deepcopy(artifact)
    tampered_artifact["temperatures"]["t1_bb"]["final_temperature"]["numerator"] += 1
    tampered_artifact["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in tampered_artifact.items() if key != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="does not match raw inputs"):
        verify_behavior_temperature_calibration(
            records,
            rows,
            tampered_artifact,
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
        )


def test_dev_rejects_fit_overfit_and_identity_fallback_is_default_valid():
    records, rows, challenge_records, challenge_rows = _dataset(dev_observed="last")
    artifact = build_behavior_temperature_calibration(
        records,
        rows,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        gate_config=_small_gate(),
    )

    assert artifact["promotion_eligible"] is True
    assert all(
        role["dev_selected_candidate"] == "identity_temperature"
        for role in artifact["temperatures"].values()
    )
    assert all(
        float.fromhex(role["dev_candidate_nll"]["identity_temperature_f64_hex"])
        <= float.fromhex(role["dev_candidate_nll"]["fit_temperature_f64_hex"])
        for role in artifact["temperatures"].values()
    )
    assert not any(
        name.startswith("selection.")
        for name in artifact["gate_result"]["failures"]
    )

    strict = build_behavior_temperature_calibration(
        records,
        rows,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        gate_config=_small_gate(require_nonidentity_temperature=True),
    )
    assert strict["promotion_eligible"] is False
    assert all(
        f"selection.{role}.promotion_policy" in strict["gate_result"]["failures"]
        for role in INFO_BUILDERS
    )


def test_missing_targeted_joker_cell_is_a_deterministic_gate_failure():
    records, rows, challenge_records, challenge_rows = _dataset(include_joker2=False)
    artifact = build_behavior_temperature_calibration(
        records,
        rows,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        gate_config=_small_gate(),
    )

    assert artifact["promotion_eligible"] is False
    failures = artifact["gate_result"]["failures"]
    assert any("challenge.t1_bb.joker_2.decisions" in name for name in failures)
    assert any("challenge.t2_btn.joker_2.roots" in name for name in failures)


def test_production_thresholds_fail_on_small_synthetic_collection():
    records, rows, challenge_records, challenge_rows = _dataset()
    artifact = build_behavior_temperature_calibration(
        records,
        rows,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
    )

    assert artifact["promotion_eligible"] is False
    assert "count.fit.t1_bb.decisions" in artifact["gate_result"]["failures"]
    assert (
        "count.challenge.t2_btn.joker_2.decisions"
        in artifact["gate_result"]["failures"]
    )


def test_optional_challenge_has_safe_empty_schema_and_is_not_selection_input():
    records, rows, _challenge_records, _challenge_rows = _dataset()
    artifact = build_behavior_temperature_calibration(
        records,
        rows,
        gate_config=_small_gate(require_joker_challenge=False),
    )

    assert artifact["promotion_eligible"] is True
    assert artifact["joker_challenge"]["present"] is False
    assert artifact["joker_challenge"]["raw_decision_manifest"] is None
    assert artifact["joker_challenge"]["used_for_dev_selection"] is False
    assert artifact["metrics"]["challenge"]["by_role_joker"]["t1_bb"][
        "joker_2"
    ]["decision_count"] == 0
