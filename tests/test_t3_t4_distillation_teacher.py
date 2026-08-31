import copy
import hashlib
import json
import shutil
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType

import numpy as np
import pytest

import ai.tutor.t3_t4_distillation_teacher as teacher_module
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.t3_hu_public_cfr import InfoSetKey, PrivateRecall
from ai.tutor.t3_t4_distillation_teacher import (
    MANIFEST_NAME,
    PAYOFF_SEED_NAMESPACE,
    Q32_DENOMINATOR,
    SOLVER_SEED_NAMESPACE,
    SOLVER_METHOD_CONTRACT_IDS,
    TeacherContractError,
    build_split_assignment,
    build_t4_btn_exact_teacher_row,
    build_teacher_row,
    canonical_json,
    canonical_sha256,
    derive_payoff_seed,
    derive_solver_seed,
    self_hash,
    verify_split_assignment,
    verify_teacher_bundle,
    verify_teacher_row,
    write_teacher_bundle,
)
from ai.tutor.t3_t4_infoset_encoder import (
    INFOSET_ENCODER_MANIFEST_SHA256,
    INFOSET_VECTOR_DIM,
    decode_infoset_key,
    semantic_action_ids,
)
from ai.tutor.t4_btn_exact_resolver import (
    T4_BTN_EXACT_METHOD,
    resolve_t4_second_btn_exact,
)


BB_T0 = (
    ("4h", "top"),
    ("2c", "top"),
    ("3d", "top"),
    ("7c", "middle"),
    ("Qc", "bottom"),
)
BTN_T0 = (
    ("8c", "top"),
    ("5c", "top"),
    ("6d", "top"),
    ("9c", "middle"),
    ("As", "bottom"),
)
BB_T1 = (("7d", "middle"), ("8h", "middle"))
BTN_T1 = (("9d", "middle"), ("Jh", "middle"))
BB_T2 = (("9s", "middle"), ("Tc", "middle"))
BTN_T2 = (("Qs", "middle"), ("Kc", "middle"))
BB_T3 = (("Qd", "bottom"), ("X1", "bottom"))
BTN_T3 = (("2d", "bottom"), ("3h", "bottom"))
BB_T4 = (("Ad", "bottom"), ("Kd", "bottom"))

HISTORY_T3_FIRST = (
    (0, "bb", BB_T0),
    (0, "btn", BTN_T0),
    (1, "bb", BB_T1),
    (1, "btn", BTN_T1),
    (2, "bb", BB_T2),
    (2, "btn", BTN_T2),
)
HISTORY_T3_SECOND = HISTORY_T3_FIRST + ((3, "bb", BB_T3),)
HISTORY_T4_FIRST = HISTORY_T3_SECOND + ((3, "btn", BTN_T3),)
HISTORY_T4_SECOND = HISTORY_T4_FIRST + ((4, "bb", BB_T4),)

BB_BOARD_9 = (
    ("2c", "3d", "4h"),
    ("7c", "7d", "8h", "9s", "Tc"),
    ("Qc",),
)
BTN_BOARD_9 = (
    ("5c", "6d", "8c"),
    ("9c", "9d", "Jh", "Kc", "Qs"),
    ("As",),
)
BB_BOARD_11 = (BB_BOARD_9[0], BB_BOARD_9[1], ("Qc", "Qd", "X1"))
BTN_BOARD_11 = (BTN_BOARD_9[0], BTN_BOARD_9[1], ("2d", "3h", "As"))
BB_BOARD_13 = (
    BB_BOARD_9[0],
    BB_BOARD_9[1],
    ("Ad", "Kd", "Qc", "Qd", "X1"),
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _recall(actor: str, through_turn: int) -> PrivateRecall:
    public_cards = {
        "bb": {
            1: ("7d", "8h"),
            2: ("9s", "Tc"),
            3: ("Qd", "X1"),
        },
        "btn": {
            1: ("9d", "Jh"),
            2: ("Qs", "Kc"),
            3: ("2d", "3h"),
        },
    }[actor]
    discards = {
        "bb": {1: "6c", 2: "6s", 3: "X2"},
        "btn": {1: "4s", 2: "7h", 3: "4d"},
    }[actor]
    return PrivateRecall(
        dealt_by_turn=tuple(
            (turn, (*public_cards[turn], discards[turn]))
            for turn in range(1, through_turn + 1)
        ),
        discards_by_turn=tuple(
            (turn, discards[turn]) for turn in range(1, through_turn + 1)
        ),
    )


def _key(phase: str, *, fantasy_state: str | None = None) -> InfoSetKey:
    spec = {
        "t3_first": (
            "bb",
            3,
            BB_BOARD_9,
            BTN_BOARD_9,
            HISTORY_T3_FIRST,
            _recall("bb", 2),
            ("Qd", "X1", "X2"),
        ),
        "t3_second": (
            "btn",
            3,
            BB_BOARD_11,
            BTN_BOARD_9,
            HISTORY_T3_SECOND,
            _recall("btn", 2),
            ("2d", "3h", "4d"),
        ),
        "t4_first": (
            "bb",
            4,
            BB_BOARD_11,
            BTN_BOARD_11,
            HISTORY_T4_FIRST,
            _recall("bb", 3),
            ("Ad", "Jd", "Kd"),
        ),
        "t4_second": (
            "btn",
            4,
            BB_BOARD_13,
            BTN_BOARD_11,
            HISTORY_T4_SECOND,
            _recall("btn", 3),
            ("Ah", "Qh", "Th"),
        ),
    }[phase]
    actor, turn, board_bb, board_btn, history, recall, draw = spec
    return InfoSetKey(
        contract_version=POSITION_CONTRACT_VERSION,
        actor=actor,
        turn=turn,
        phase=phase,
        board_bb=board_bb,
        board_btn=board_btn,
        public_action_history=history,
        own_recall=recall,
        current_draw=draw,
        fantasy_state=fantasy_state,
    )


def _assignment(tag: str = "base") -> dict:
    return build_split_assignment(
        full_deal_commitment_sha256=_hash(f"full-deal:{tag}"),
        public_root_family_commitment_sha256=_hash(f"root-family:{tag}"),
    )


def _assignment_for_split(split: str) -> dict:
    for index in range(100_000):
        assignment = _assignment(f"{split}:{index}")
        if assignment["split"] == split:
            return assignment
    raise AssertionError(f"could not find deterministic {split} fixture")


def _bindings(tag: str) -> dict[str, str]:
    return {
        field: _hash(f"{tag}:{field}")
        for field in (
            "public_root_commitment_sha256",
            "public_root_mixture_sha256",
            "range_content_sha256",
            "range_build_sha256",
            "behavior_model_sha256",
            "source_manifest_sha256",
            "solver_source_sha256",
            "solver_config_sha256",
            "solver_checkpoint_sha256",
        )
    }


def _exact_bindings(tag: str) -> dict[str, str]:
    return {
        field: value
        for field, value in _bindings(tag).items()
        if field
        not in {
            "solver_source_sha256",
            "solver_config_sha256",
            "solver_checkpoint_sha256",
        }
    }


def _lineage(tag: str, *, seat_swap: int = 0, suit: int = 0) -> dict:
    return {
        "descendant_public_path_sha256": _hash(f"{tag}:descendant"),
        "restricted_variant_root_commitment_sha256": _hash(f"{tag}:variant"),
        "restricted_private_type_commitment_sha256": _hash(f"{tag}:private-type"),
        "seat_swap_index": seat_swap,
        "suit_augmentation_index": suit,
    }


def _labels(key: InfoSetKey) -> tuple[dict[str, str], dict[str, dict[str, float | int]]]:
    legal_ids = [action_id for action_id in semantic_action_ids(key) if action_id]
    strategy = {
        action_id: f"{index + 1}/{len(legal_ids)}"
        for index, action_id in enumerate(legal_ids)
    }
    moments = {}
    for index, action_id in enumerate(legal_ids):
        mean = float(index - 1)
        values = (mean - 1.0, mean, mean, mean + 1.0)
        moments[action_id] = {
            "count": len(values),
            "sum": sum(values),
            "sum_squares": sum(value * value for value in values),
        }
    return strategy, moments


def _row(
    phase: str = "t3_first",
    *,
    assignment: dict | None = None,
    tag: str = "row-0",
    solver_seed_index: int = 0,
    payoff_seed_index: int = 0,
    seat_swap: int = 0,
    suit: int = 0,
    lineage: dict | None = None,
    solver_method: str = SOLVER_METHOD_CONTRACT_IDS[0],
) -> dict:
    key = _key(phase)
    strategy, moments = _labels(key)
    return build_teacher_row(
        key,
        split_assignment=assignment or _assignment(),
        lineage=(
            lineage
            if lineage is not None
            else _lineage(tag, seat_swap=seat_swap, suit=suit)
        ),
        bindings=_bindings(tag),
        solver_method=solver_method,
        solver_iterations_completed=10_000,
        solver_seed_index=solver_seed_index,
        payoff_seed_index=payoff_seed_index,
        average_strategy_by_action_id=strategy,
        action_payoff_moments_by_action_id=moments,
        infoset_visit_count=1234,
        infoset_reach_probability=Fraction(1, 100),
    )


def _exact_row(
    *,
    assignment: dict | None = None,
    tag: str = "exact-row",
    seat_swap: int = 0,
    suit: int = 0,
) -> dict:
    key = _key("t4_second")
    return build_t4_btn_exact_teacher_row(
        key,
        resolution=resolve_t4_second_btn_exact(key),
        split_assignment=assignment or _assignment(),
        lineage=_lineage(tag, seat_swap=seat_swap, suit=suit),
        bindings=_exact_bindings(tag),
    )


def _rehash_row(row: dict) -> None:
    row["row_sha256"] = self_hash(row, "row_sha256")


def _rewrite_canonical(path, value) -> None:
    path.write_bytes((canonical_json(value) + "\n").encode("utf-8"))


def test_split_is_prelabel_full_deal_locked_across_all_variant_axes():
    first = _assignment("same-deal")
    different_public_family = build_split_assignment(
        full_deal_commitment_sha256=first["full_deal_commitment_sha256"],
        public_root_family_commitment_sha256=_hash("another-public-family"),
    )

    assert verify_split_assignment(first) == first
    assert first["assigned_before_labels"] is True
    assert first["split_material"] == (
        "original_pre_augmentation_full_deal_commitment_sha256"
    )
    assert first["split"] == different_public_family["split"]
    assert first["split_bucket"] == different_public_family["split_bucket"]
    assert first["split_group_sha256"] == different_public_family["split_group_sha256"]
    assert set(first["invariant_axes"]) == {
        "descendants",
        "private_types",
        "solver_seeds",
        "payoff_seeds",
        "seat_swaps",
        "suit_augmentations",
    }


def test_solver_and_payoff_seed_namespaces_are_deterministic_and_disjoint():
    assignment = _assignment()
    group = assignment["split_group_sha256"]
    solver = derive_solver_seed(group, 7)
    payoff = derive_payoff_seed(group, 7)

    assert solver == derive_solver_seed(group, 7)
    assert payoff == derive_payoff_seed(group, 7)
    assert solver % 2 == 0
    assert payoff % 2 == 1
    assert solver != payoff

    row = _row(solver_seed_index=7, payoff_seed_index=7)
    plan = row["solver"]["seed_plan"]
    assert plan["solver"] == {
        "namespace": SOLVER_SEED_NAMESPACE,
        "index": 7,
        "seed": solver,
    }
    assert plan["payoff"] == {
        "namespace": PAYOFF_SEED_NAMESPACE,
        "index": 7,
        "seed": payoff,
    }


def test_raw_moment_feasibility_is_overflow_safe_without_rejecting_valid_scale():
    key = _key("t3_first")
    strategy, valid_moments = _labels(key)
    action_id = next(iter(valid_moments))
    large_value = 5e153
    valid_moments[action_id] = {
        "count": 4,
        "sum": 4 * large_value,
        "sum_squares": 4 * large_value * large_value,
    }
    valid = build_teacher_row(
        key,
        split_assignment=_assignment("valid-large-moments"),
        lineage=_lineage("valid-large-moments"),
        bindings=_bindings("valid-large-moments"),
        solver_method=SOLVER_METHOD_CONTRACT_IDS[0],
        solver_iterations_completed=100,
        solver_seed_index=0,
        payoff_seed_index=0,
        average_strategy_by_action_id=strategy,
        action_payoff_moments_by_action_id=valid_moments,
        infoset_visit_count=1,
        infoset_reach_probability=Fraction(1, 2),
    )
    valid_stat = next(
        row
        for row in valid["targets"]["action_payoff_statistics"]
        if row is not None and row["action_id"] == action_id
    )
    assert valid_stat["mean"] == large_value
    assert np.isfinite(valid_stat["standard_error"])

    impossible_moments = copy.deepcopy(valid_moments)
    impossible_moments[action_id] = {
        "count": 4,
        "sum": 1e308,
        "sum_squares": 0.0,
    }
    with pytest.raises(TeacherContractError, match="impossible raw moments"):
        build_teacher_row(
            key,
            split_assignment=_assignment("impossible-large-moments"),
            lineage=_lineage("impossible-large-moments"),
            bindings=_bindings("impossible-large-moments"),
            solver_method=SOLVER_METHOD_CONTRACT_IDS[0],
            solver_iterations_completed=100,
            solver_seed_index=0,
            payoff_seed_index=0,
            average_strategy_by_action_id=strategy,
            action_payoff_moments_by_action_id=impossible_moments,
            infoset_visit_count=1,
            infoset_reach_probability=Fraction(1, 2),
        )

    tampered = copy.deepcopy(valid)
    tampered_stat = next(
        row
        for row in tampered["targets"]["action_payoff_statistics"]
        if row is not None and row["action_id"] == action_id
    )
    tampered_stat["sum"] = 1e308
    tampered_stat["sum_squares"] = 0.0
    _rehash_row(tampered)
    with pytest.raises(TeacherContractError, match="impossible raw moments"):
        verify_teacher_row(tampered)


def test_solver_method_requires_a_known_contract_id_and_cannot_carry_raw_cards():
    exact_terminal = _exact_row(tag="known-t4-terminal-method")
    assert exact_terminal["solver"]["method"] == T4_BTN_EXACT_METHOD
    assert exact_terminal["solver"]["policy_source"] == "exact_terminal_argmax"
    assert exact_terminal["solver"]["payoff_source"] == "exact_terminal_utility"
    assert exact_terminal["solver"]["seed_usage"] == "none"
    assert "iterations_completed" not in exact_terminal["solver"]
    assert "seed_plan" not in exact_terminal["solver"]
    assert "solver_checkpoint_sha256" not in exact_terminal["bindings"]
    assert "action_payoff_statistics" not in exact_terminal["targets"]
    assert exact_terminal["targets"]["policy_semantics"] == "exact_terminal_argmax"
    assert exact_terminal["targets"]["action_payoff_semantics"] == (
        "exact_terminal_utility"
    )
    assert verify_teacher_row(exact_terminal) == exact_terminal

    with pytest.raises(TeacherContractError, match="build_t4_btn_exact_teacher_row"):
        _row(
            "t4_second",
            tag="wrong-exact-interface",
            solver_method=T4_BTN_EXACT_METHOD,
        )

    private_text = "opponent_private_cards=X2,Ad,Kd"
    with pytest.raises(TeacherContractError, match="known contract ID"):
        _row(tag="private-solver-method", solver_method=private_text)

    tampered = _row(tag="tampered-solver-method")
    tampered["solver"]["method"] = private_text
    _rehash_row(tampered)
    with pytest.raises(TeacherContractError, match="known contract ID"):
        verify_teacher_row(tampered)


def test_exact_terminal_adapter_binds_fresh_result_and_rejects_rehashed_tamper():
    row = _exact_row(tag="exact-binding")
    key = _key("t4_second")
    resolution = resolve_t4_second_btn_exact(key)
    assert row["solver"]["resolver_manifest_sha256"] == resolution.manifest_sha256
    assert len(row["solver"]["resolver_result_sha256"]) == 64
    assert row["quality"] == {
        "legal_action_count": len(resolution.utility_by_action_id),
        "exact_terminal_result_reverified": True,
        "all_legal_actions_evaluated_exactly": True,
        "structural_quality_passed": True,
        "quality_status": "exact_terminal_verified",
        "promotion_quality_gate_evaluated": False,
    }

    forged_manifest = copy.deepcopy(row)
    forged_manifest["solver"]["resolver_manifest_sha256"] = "0" * 64
    forged_manifest["row_identity_sha256"] = canonical_sha256(
        {
            "schema": "forged-row-identity",
            "solver": forged_manifest["solver"],
        }
    )
    _rehash_row(forged_manifest)
    with pytest.raises(TeacherContractError, match="exact resolver provenance mismatch"):
        verify_teacher_row(forged_manifest)

    forged_target = copy.deepcopy(row)
    legal_slot = next(
        item
        for item in forged_target["targets"]["terminal_utility_by_action"]
        if item is not None
    )
    legal_slot["utility"] = bool(legal_slot["utility"])
    _rehash_row(forged_target)
    with pytest.raises(TeacherContractError, match="exact terminal resolver binding mismatch"):
        verify_teacher_row(forged_target)


def test_exact_teacher_rejects_paired_forged_resolver_aliases(monkeypatch):
    key = _key("t4_second")
    honest = resolve_t4_second_btn_exact(key)
    forged_utility = MappingProxyType(
        {
            action_id: utility + 999.0
            for action_id, utility in honest.utility_by_action_id.items()
        }
    )
    forged = replace(honest, utility_by_action_id=forged_utility)
    assert {
        forged.utility_by_action_id[action_id]
        - honest.utility_by_action_id[action_id]
        for action_id in honest.utility_by_action_id
    } == {999.0}

    forged_row = copy.deepcopy(_exact_row(tag="forged-alias-row"))
    forged_row["solver"] = teacher_module._exact_solver_payload(forged)
    forged_row["targets"] = teacher_module._exact_target_payload(key, forged)
    forged_row["quality"] = teacher_module._exact_quality_payload(forged)
    forged_row["row_identity_sha256"] = teacher_module._row_identity(
        assignment=forged_row["split_assignment"],
        lineage=forged_row["lineage"],
        model_input=forged_row["model_input"],
        bindings=forged_row["bindings"],
        solver=forged_row["solver"],
    )
    _rehash_row(forged_row)

    with monkeypatch.context() as patch:
        patch.setattr(
            teacher_module,
            "resolve_t4_second_btn_exact",
            lambda _key: forged,
        )
        patch.setattr(
            teacher_module,
            "verify_t4_second_btn_exact",
            lambda *_args, **_kwargs: {"verified": True},
        )
        with pytest.raises(
            TeacherContractError,
            match="canonical exact resolver alias drifted",
        ):
            teacher_module.verify_teacher_row(forged_row)
        with pytest.raises(
            TeacherContractError,
            match="canonical exact resolver alias drifted",
        ):
            teacher_module.build_t4_btn_exact_teacher_row(
                key,
                resolution=forged,
                split_assignment=_assignment("forged-alias-build"),
                lineage=_lineage("forged-alias-build"),
                bindings=_exact_bindings("forged-alias-build"),
            )


@pytest.mark.parametrize("phase", ("t3_first", "t3_second", "t4_first", "t4_second"))
def test_teacher_row_binds_lossless_input_actions_targets_and_quality(phase: str):
    key = _key(phase)
    row = _row(phase, tag=f"row:{phase}")
    verified = verify_teacher_row(row)
    model_input = verified["model_input"]
    decoded = decode_infoset_key(
        np.asarray(model_input["encoded_vector"], dtype=np.float32)
    )
    targets = verified["targets"]
    quality = verified["quality"]

    assert decoded == key
    assert model_input["information_canonical_json"] == key.canonical_json()
    assert model_input["information_digest"] == key.digest()
    assert model_input["encoder_manifest_sha256"] == INFOSET_ENCODER_MANIFEST_SHA256
    assert model_input["encoded_vector_dimension"] == INFOSET_VECTOR_DIM == 3313
    assert model_input["opponent_hidden_cards_included"] is False
    assert sum(targets["average_strategy_q32"]) == Q32_DENOMINATOR
    assert len(targets["average_strategy_q32"]) == 27
    assert all(
        weight == 0 or verified["action_contract"]["legal_action_mask"][index]
        for index, weight in enumerate(targets["average_strategy_q32"])
    )
    assert len(targets["action_payoff_statistics"]) == 27
    assert targets["payoff_perspective"] == "acting_player"
    assert targets["one_step_deviation_regret_estimate"] >= 0
    assert targets["exact_exploitability_computed"] is False
    assert quality["infoset_visit_count"] == 1234
    assert quality["infoset_reach_probability_exact"] == "1/100"
    assert quality["all_legal_actions_sampled"] is True
    assert quality["structural_quality_passed"] is True
    assert quality["promotion_quality_gate_evaluated"] is False
    assert verified["algorithm_teacher_only"] is True
    assert verified["promotion_eligible"] is False
    assert verified["training_performed"] is False
    assert verified["serving_changed"] is False


def test_model_input_has_no_hidden_world_and_restricted_metadata_is_hash_only():
    row = _row()
    serialized_input = canonical_json(row["model_input"]).lower()
    for forbidden in (
        "opponent_private",
        "opponent_discards",
        "undealt_cards",
        "remaining_deck",
        "particle_id",
        "rng_seed",
    ):
        assert f'"{forbidden}"' not in serialized_input
    assert row["model_input"]["restricted_commitments_in_input"] is False
    assert all(
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
        for value in row["bindings"].values()
    )
    assert all(
        isinstance(value, str) and len(value) == 64
        for field, value in row["lineage"].items()
        if field.endswith("sha256")
    )


def test_teacher_rejects_any_fantasy_or_private_looking_fantasy_text():
    private_text = 'opponent_private={"X1":"discard"}'
    key = _key("t3_first", fantasy_state=private_text)
    strategy, moments = _labels(key)
    with pytest.raises(TeacherContractError, match="fantasy_state=None"):
        build_teacher_row(
            key,
            split_assignment=_assignment(),
            lineage=_lineage("private-text"),
            bindings=_bindings("private-text"),
            solver_method=SOLVER_METHOD_CONTRACT_IDS[0],
            solver_iterations_completed=1,
            solver_seed_index=0,
            payoff_seed_index=0,
            average_strategy_by_action_id=strategy,
            action_payoff_moments_by_action_id=moments,
            infoset_visit_count=1,
            infoset_reach_probability="1/1",
        )


def test_rehashed_vector_payoff_split_seed_and_hidden_field_tamper_fail_closed():
    base = _row()

    vector_tamper = copy.deepcopy(base)
    vector = vector_tamper["model_input"]["encoded_vector"]
    vector[0] = 1 - vector[0]
    vector_tamper["model_input"]["encoded_vector_sha256"] = canonical_sha256(vector)
    vector_tamper["model_input"]["model_input_sha256"] = self_hash(
        vector_tamper["model_input"],
        "model_input_sha256",
    )
    _rehash_row(vector_tamper)
    with pytest.raises(TeacherContractError, match="manifest SHA-256|canonical information"):
        verify_teacher_row(vector_tamper)

    payoff_tamper = copy.deepcopy(base)
    statistic = next(
        item
        for item in payoff_tamper["targets"]["action_payoff_statistics"]
        if item is not None
    )
    statistic["sum"] += 1.0
    _rehash_row(payoff_tamper)
    with pytest.raises(TeacherContractError, match="mean/raw moment mismatch"):
        verify_teacher_row(payoff_tamper)

    split_tamper = copy.deepcopy(base)
    split_tamper["split_assignment"]["split"] = "test"
    split_tamper["split_assignment"]["assignment_sha256"] = self_hash(
        split_tamper["split_assignment"],
        "assignment_sha256",
    )
    _rehash_row(split_tamper)
    with pytest.raises(TeacherContractError, match="stale or label-dependent"):
        verify_teacher_row(split_tamper)

    seed_tamper = copy.deepcopy(base)
    seed_tamper["solver"]["seed_plan"]["solver"]["namespace"] = PAYOFF_SEED_NAMESPACE
    _rehash_row(seed_tamper)
    with pytest.raises(TeacherContractError, match="namespace, derivation, or parity"):
        verify_teacher_row(seed_tamper)

    hidden_field = copy.deepcopy(base)
    hidden_field["model_input"]["opponent_private"] = ["X2"]
    hidden_field["model_input"]["model_input_sha256"] = self_hash(
        hidden_field["model_input"],
        "model_input_sha256",
    )
    _rehash_row(hidden_field)
    with pytest.raises(TeacherContractError, match="exact fields"):
        verify_teacher_row(hidden_field)


def test_content_addressed_manifest_last_bundle_roundtrip(tmp_path):
    assignment = _assignment("bundle")
    rows = [
        _row("t3_first", assignment=assignment, tag="bundle-0"),
        _row("t3_second", assignment=assignment, tag="bundle-1"),
        _row("t4_first", assignment=assignment, tag="bundle-2", seat_swap=1),
        _exact_row(assignment=assignment, tag="bundle-3", suit=1),
        _row(
            "t3_first",
            assignment=assignment,
            tag="bundle-4",
            solver_seed_index=1,
            payoff_seed_index=1,
            seat_swap=1,
            suit=2,
        ),
    ]
    output = tmp_path / "teacher"
    built = write_teacher_bundle(output, list(reversed(rows)), shard_size=2)
    verified = verify_teacher_bundle(output)

    assert built == verified
    assert verified.manifest["manifest_written_last"] is True
    assert verified.manifest["row_count"] == 5
    assert verified.manifest["shard_count"] == 3
    assert verified.manifest["algorithm_teacher_only"] is True
    assert verified.manifest["promotion_eligible"] is False
    assert verified.manifest["training_performed"] is False
    assert verified.manifest["serving_changed"] is False
    assert verified.manifest["overlap_audit"]["cross_split_overlap_count"] == 0
    assert (
        verified.manifest["overlap_audit"][
            "each_lineage_commitment_has_one_full_deal"
        ]
        is True
    )
    assert sum(verified.manifest["split_group_counts"].values()) == 1
    assert verified.manifest["seed_namespace_overlap_count"] == 0
    assert verified.manifest["seed_contract"]["seedless_methods"] == [
        T4_BTN_EXACT_METHOD
    ]
    assert verified.manifest["solver_method_contract_ids"] == list(
        SOLVER_METHOD_CONTRACT_IDS
    )
    assert [row["row_identity_sha256"] for row in verified.rows] == sorted(
        row["row_identity_sha256"] for row in verified.rows
    )
    assert (output / MANIFEST_NAME).is_file()
    for entry in verified.manifest["shards"]:
        assert entry["shard_sha256"] in entry["relative_path"]
        path = output / entry["relative_path"]
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n") and text.count("\n") == 1
        assert canonical_json(json.loads(text)) + "\n" == text


def test_duplicate_and_cross_split_observation_overlap_are_rejected(tmp_path):
    row = _row()
    with pytest.raises(TeacherContractError, match="duplicate teacher row identity"):
        write_teacher_bundle(tmp_path / "duplicate", [row, row], shard_size=1)

    fit = _assignment_for_split("fit")
    test = _assignment_for_split("test")
    fit_row = _row(assignment=fit, tag="fit-row")
    test_row = _row(assignment=test, tag="test-row")
    assert fit_row["model_input"]["information_digest"] == test_row["model_input"][
        "information_digest"
    ]
    with pytest.raises(TeacherContractError, match="overlap locked splits"):
        write_teacher_bundle(
            tmp_path / "overlap",
            [fit_row, test_row],
            shard_size=2,
        )


def test_public_root_family_cannot_be_rebound_to_another_full_deal(tmp_path):
    first = _assignment("family-owner-a")
    second = build_split_assignment(
        full_deal_commitment_sha256=_hash("family-owner-b"),
        public_root_family_commitment_sha256=first[
            "public_root_family_commitment_sha256"
        ],
    )
    rows = [
        _row(assignment=first, tag="family-owner-a"),
        _row("t3_second", assignment=second, tag="family-owner-b"),
    ]
    with pytest.raises(TeacherContractError, match="multiple full-deal"):
        write_teacher_bundle(tmp_path / "family-rebind", rows, shard_size=2)


def test_lineage_commitments_cannot_be_reused_by_another_full_deal_in_same_split(
    tmp_path,
):
    first = _assignment_for_split("fit")
    second = None
    for index in range(100_000):
        candidate = _assignment(f"same-split-lineage-owner:{index}")
        if (
            candidate["split"] == first["split"]
            and candidate["full_deal_commitment_sha256"]
            != first["full_deal_commitment_sha256"]
        ):
            second = candidate
            break
    assert second is not None
    shared_lineage = _lineage("shared-across-two-full-deals")
    rows = [
        _row(
            "t3_first",
            assignment=first,
            tag="lineage-owner-a",
            lineage=shared_lineage,
        ),
        _row(
            "t3_second",
            assignment=second,
            tag="lineage-owner-b",
            lineage=shared_lineage,
        ),
    ]
    with pytest.raises(
        TeacherContractError,
        match="lineage commitment is bound to multiple full-deal",
    ):
        write_teacher_bundle(tmp_path / "lineage-rebind", rows, shard_size=2)


def test_tamper_gap_orphan_and_missing_manifest_fail_closed(tmp_path):
    rows = [_row(tag=f"artifact-{index}", solver_seed_index=index) for index in range(3)]
    source = tmp_path / "source"
    bundle = write_teacher_bundle(source, rows, shard_size=1)

    tamper = tmp_path / "tamper"
    shutil.copytree(source, tamper)
    first_path = tamper / bundle.manifest["shards"][0]["relative_path"]
    first_path.write_bytes(first_path.read_bytes() + b" ")
    with pytest.raises(TeacherContractError, match="file SHA-256|canonical JSON"):
        verify_teacher_bundle(tamper)

    gap = tmp_path / "gap"
    shutil.copytree(source, gap)
    (gap / bundle.manifest["shards"][1]["relative_path"]).unlink()
    with pytest.raises(TeacherContractError, match="missing or escaping file"):
        verify_teacher_bundle(gap)

    orphan = tmp_path / "orphan"
    shutil.copytree(source, orphan)
    (orphan / "shards" / "orphan.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(TeacherContractError, match="orphan files"):
        verify_teacher_bundle(orphan)

    no_manifest = tmp_path / "no-manifest"
    shutil.copytree(source, no_manifest)
    (no_manifest / MANIFEST_NAME).unlink()
    with pytest.raises(TeacherContractError, match="missing or escaping file"):
        verify_teacher_bundle(no_manifest)

    duplicate_key = tmp_path / "duplicate-json-key"
    shutil.copytree(source, duplicate_key)
    manifest_path = duplicate_key / MANIFEST_NAME
    original = manifest_path.read_bytes()
    manifest_path.write_bytes(b'{"schema":"duplicate",' + original[1:])
    with pytest.raises(TeacherContractError, match="duplicate JSON key"):
        verify_teacher_bundle(duplicate_key)


def test_rehashed_manifest_range_gap_and_duplicate_entry_fail_closed(tmp_path):
    source = tmp_path / "source"
    bundle = write_teacher_bundle(
        source,
        [_row(tag=f"range-{index}", solver_seed_index=index) for index in range(3)],
        shard_size=1,
    )

    gap = tmp_path / "range-gap"
    shutil.copytree(source, gap)
    gap_manifest_path = gap / MANIFEST_NAME
    gap_manifest = json.loads(gap_manifest_path.read_text(encoding="utf-8"))
    gap_manifest["shards"][1]["row_start"] += 1
    gap_manifest["shards"][1]["entry_sha256"] = self_hash(
        gap_manifest["shards"][1],
        "entry_sha256",
    )
    gap_manifest["manifest_sha256"] = self_hash(gap_manifest, "manifest_sha256")
    _rewrite_canonical(gap_manifest_path, gap_manifest)
    with pytest.raises(TeacherContractError, match="gap or overlap"):
        verify_teacher_bundle(gap)

    duplicate = tmp_path / "duplicate-entry"
    shutil.copytree(source, duplicate)
    duplicate_manifest_path = duplicate / MANIFEST_NAME
    duplicate_manifest = json.loads(
        duplicate_manifest_path.read_text(encoding="utf-8")
    )
    duplicate_manifest["shards"][1]["index"] = 0
    duplicate_manifest["shards"][1]["entry_sha256"] = self_hash(
        duplicate_manifest["shards"][1],
        "entry_sha256",
    )
    duplicate_manifest["manifest_sha256"] = self_hash(
        duplicate_manifest,
        "manifest_sha256",
    )
    _rewrite_canonical(duplicate_manifest_path, duplicate_manifest)
    with pytest.raises(TeacherContractError, match="duplicate/out-of-order"):
        verify_teacher_bundle(duplicate)


def test_output_must_be_empty_and_bundle_is_bounded(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "unrelated.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(TeacherContractError, match="must be empty"):
        write_teacher_bundle(occupied, [_row()], shard_size=1)
    assert (occupied / "unrelated.txt").read_text(encoding="utf-8") == "keep"

    with pytest.raises(TeacherContractError, match="integer <= 256"):
        write_teacher_bundle(tmp_path / "too-large", [_row()], shard_size=257)
