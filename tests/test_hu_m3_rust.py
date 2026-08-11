from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import replace

import pytest

from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation, WorldState
from ofc_regular.hu_late_street_teacher import (
    T4SearchConfig,
    evaluate_t4_sequential_actions,
)
from ofc_regular.hu_m3_rust import (
    HuM3RustError,
    build_native_engine,
    engine_version,
    evaluate_batch,
    evaluate_request,
    evaluate_t3,
    evaluate_t3_abr_components,
    evaluate_t4,
    load_native_engine,
    t3_request,
    t4_request,
)
from ofc_regular.hu_turn3_joint_exact_teacher import (
    JointExactConfig,
    evaluate_t3_joint_exact_actions,
)
from ofc_regular.state import Board


def _constrained_board(cards: tuple[str, ...]) -> Board:
    """A complete top/middle with bottom space keeps parity tests tiny."""

    return Board.from_rows(
        top=cards[:3],
        middle=cards[3:8],
        bottom=cards[8:],
    )


def _t3_observation(to_act_order: str) -> ActorObservation:
    opponent_count = 9 if to_act_order == "first" else 11
    cursor = 0
    hero_cards = ALL_CARDS[cursor : cursor + 9]
    cursor += 9
    opponent_cards = ALL_CARDS[cursor : cursor + opponent_count]
    cursor += opponent_count
    dealt = ALL_CARDS[cursor : cursor + 3]
    cursor += 3
    hero_discards = ALL_CARDS[cursor : cursor + 2]
    return ActorObservation(
        hero_board=_constrained_board(hero_cards),
        opponent_public_board=_constrained_board(opponent_cards),
        dealt_cards=dealt,
        hero_private_discards=hero_discards,
        seat=to_act_order,  # type: ignore[arg-type]
        street="T3",
        to_act_order=to_act_order,  # type: ignore[arg-type]
    )


def _t4_first_observation() -> ActorObservation:
    """Three legal actions per player, including the full C(24, 3) tree."""

    return ActorObservation(
        hero_board=Board.from_rows(
            top=("Kh",),
            middle=("7d", "Jh", "Ac", "6s", "3c"),
            bottom=("Qs", "Qc", "Ks", "As", "Js"),
        ),
        opponent_public_board=Board.from_rows(
            top=("Ad",),
            middle=("3s", "8c", "4c", "7c", "6d"),
            bottom=("Td", "3d", "9d", "8d", "6h"),
        ),
        dealt_cards=("4h", "2c", "Ah"),
        hero_private_discards=("Tc", "7h", "Th"),
        seat="first",
        street="T4",
        to_act_order="first",
    )


def _t4_second_observation() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("Qh", "Kc"),
            middle=("Ah", "Ac", "4s", "5s"),
            bottom=("Td", "7h", "7s", "7c", "Th"),
        ),
        opponent_public_board=Board.from_rows(
            top=("2h", "3h", "4h"),
            middle=("5c", "2s", "6h", "4c", "8c"),
            bottom=("7d", "Jd", "9d", "Qd", "Kd"),
        ),
        dealt_cards=("2c", "3c", "4d"),
        hero_private_discards=("6c", "8h", "9c"),
        seat="second",
        street="T4",
        to_act_order="second",
    )


@pytest.fixture(scope="session")
def m3_native():
    build = build_native_engine(release=False)
    return load_native_engine(path=build.library_path)


def _action_projection(result: dict) -> dict[str, tuple[int, float, float]]:
    """Compare action semantics by stable key, never by sorted row position."""

    rows = result["actions"]
    assert rows
    assert all(
        {"original_index", "action_key", "selection_score", "score"} <= set(row)
        for row in rows
    )
    return {
        row["action_key"]: (
            int(row["original_index"]),
            float(row["selection_score"]),
            float(row["score"]),
        )
        for row in rows
    }


def _assert_reference_parity(actual: dict, expected: dict) -> None:
    assert actual["observation_fingerprint"] == expected["observation_fingerprint"]
    assert actual["selected_action_key"] == expected["selected_action_key"]
    assert actual["selected_action_original_index"] == expected[
        "selected_action_original_index"
    ]
    assert actual["selected_action_evaluation_score"] == pytest.approx(
        expected["selected_action_evaluation_score"], rel=0.0, abs=1e-12
    )
    assert actual["selection_score_gap"] == pytest.approx(
        expected["selection_score_gap"], rel=0.0, abs=1e-12
    )

    actual_rows = _action_projection(actual)
    expected_rows = _action_projection(expected)
    assert actual_rows.keys() == expected_rows.keys()
    for key in actual_rows:
        assert actual_rows[key][0] == expected_rows[key][0]
        assert actual_rows[key][1:] == pytest.approx(
            expected_rows[key][1:], rel=0.0, abs=1e-12
        )


def _semantic_digest(result: dict, *, include_order_mapping: bool = True) -> str:
    actions = _action_projection(result)
    payload = {
        "observation_fingerprint": result["observation_fingerprint"],
        "selected_action_key": result["selected_action_key"],
        "selected_action_original_index": (
            result["selected_action_original_index"] if include_order_mapping else None
        ),
        "selected_action_evaluation_score": result[
            "selected_action_evaluation_score"
        ],
        "selection_score_gap": result["selection_score_gap"],
        "actions": [
            [
                key,
                original_index if include_order_mapping else None,
                selection_score,
                score,
            ]
            for key, (original_index, selection_score, score) in sorted(actions.items())
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_native_engine_builds_and_exposes_a_version(m3_native):
    version = engine_version(library=m3_native)

    assert version
    assert "m3" in version.casefold()


def test_t4_second_matches_python_terminal_exhaustive(m3_native):
    observation = _t4_second_observation()
    expected = evaluate_t4_sequential_actions(observation)

    actual = evaluate_t4(observation, library=m3_native)

    _assert_reference_parity(actual, expected)


def test_t4_first_counter_mc_matches_candidate_and_locked_evaluation(m3_native):
    observation = _t4_first_observation()
    config = T4SearchConfig(
        candidate_samples=3,
        evaluation_samples=4,
        seed=2026071301,
        candidate_seed=2026071302,
        evaluation_seed=2026071303,
        run_id="m3-python-parity-t4-first",
    )
    expected = evaluate_t4_sequential_actions(observation, config=config)

    actual = evaluate_t4(observation, config=config, library=m3_native)

    _assert_reference_parity(actual, expected)


def test_t4_first_full_uniform_2024_tree_matches_python(m3_native):
    observation = _t4_first_observation()
    config = T4SearchConfig(
        candidate_samples=0,
        evaluation_samples=0,
        seed=2026071304,
        run_id="m3-python-parity-t4-exact",
    )
    expected = evaluate_t4_sequential_actions(observation, config=config)

    actual = evaluate_t4(observation, config=config, library=m3_native)

    _assert_reference_parity(actual, expected)


@pytest.mark.parametrize("to_act_order", ["first", "second"])
def test_t3_one_particle_tree_matches_python_reference(to_act_order, m3_native):
    observation = _t3_observation(to_act_order)
    config = JointExactConfig(
        candidate_samples=1,
        evaluation_samples=1,
        downstream_t3_samples=1,
        downstream_t4_samples=1,
        seed=2026071310,
        candidate_seed=2026071311,
        evaluation_seed=2026071312,
        run_id=f"m3-python-parity-t3-{to_act_order}",
    )
    expected = evaluate_t3_joint_exact_actions(
        observation=observation,
        config=config,
    )

    actual = evaluate_t3(observation, config=config, library=m3_native)

    _assert_reference_parity(actual, expected)


def test_t3_abr_diagnostic_preserves_every_legacy_q_and_hides_worlds(
    m3_native,
):
    observation = _t3_observation("second")
    config = JointExactConfig(
        candidate_samples=1,
        evaluation_samples=1,
        downstream_t3_samples=1,
        downstream_t4_samples=0,
        seed=2026071313,
        candidate_seed=2026071314,
        evaluation_seed=2026071315,
        run_id="m3-abr-diagnostic-python-binding",
    )
    legacy = evaluate_t3(observation, config=config, library=m3_native)
    diagnostic = evaluate_t3_abr_components(
        observation,
        config=config,
        library=m3_native,
    )
    assert diagnostic["kind"] == "t3_abr_components"
    assert diagnostic["selected_action_key"] == legacy["selected_action_key"]
    legacy_by_key = {
        row["action_key"]: row for row in legacy["actions"]
    }
    diagnostic_by_key = {
        row["action_key"]: row for row in diagnostic["actions"]
    }
    assert diagnostic_by_key.keys() == legacy_by_key.keys()
    for key, row in diagnostic_by_key.items():
        expected = legacy_by_key[key]
        assert row["selection_score"] == expected["selection_score"]
        assert row["score"] == expected["score"]
        assert row["terminal_components"]["hu_score_mean"] == row["score"]
    def keys(value):
        if isinstance(value, dict):
            return set(value).union(
                *(keys(item) for item in value.values())
            )
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert not {"opponent_private_discards", "future_cards"} & keys(
        diagnostic
    )


def test_scalar_and_batch_have_identical_semantic_digests(m3_native):
    t4_config = T4SearchConfig(
        candidate_samples=2,
        evaluation_samples=3,
        seed=2026071320,
        run_id="m3-scalar-batch-t4",
    )
    t3_config = JointExactConfig(
        candidate_samples=1,
        evaluation_samples=1,
        downstream_t3_samples=1,
        downstream_t4_samples=1,
        seed=2026071321,
        run_id="m3-scalar-batch-t3",
    )
    requests = [
        t4_request(_t4_first_observation(), config=t4_config),
        t4_request(_t4_second_observation(), config=t4_config),
        t3_request(_t3_observation("second"), config=t3_config),
    ]
    scalar = [evaluate_request(request, library=m3_native) for request in requests]

    batched = evaluate_batch(requests, library=m3_native)

    assert [_semantic_digest(row) for row in batched] == [
        _semantic_digest(row) for row in scalar
    ]


def test_dealt_card_permutations_do_not_change_t4_semantics(m3_native):
    baseline = _t4_second_observation()
    observations = [
        replace(baseline, dealt_cards=tuple(cards))
        for cards in itertools.permutations(baseline.dealt_cards)
    ]
    assert {observation.fingerprint() for observation in observations} == {
        baseline.fingerprint()
    }

    results = [evaluate_t4(observation, library=m3_native) for observation in observations]

    # Legacy positional indices are allowed to follow the dealt-card order;
    # semantic ActionKeys and values must not. Per-state index/key mapping is
    # checked against Python in the parity tests above.
    expected_digest = _semantic_digest(results[0], include_order_mapping=False)
    assert {
        _semantic_digest(result, include_order_mapping=False) for result in results
    } == {expected_digest}


def test_binding_and_native_entrypoint_reject_raw_world_state(m3_native):
    observation = _t4_second_observation()
    world = WorldState(
        boards=(observation.opponent_public_board, observation.hero_board),
        private_discards=((), observation.hero_private_discards),
        street="T4",
        next_player=1,
    )

    with pytest.raises(TypeError, match="requires ActorObservation"):
        evaluate_t4(world, library=m3_native)  # type: ignore[arg-type]

    unsafe = {
        "schema": "hu_m3_engine_request_v1",
        "kind": "t4",
        "observation": {
            "schema": "hu_world_state_v1",
            "boards": [],
            "private_discards": [["As"], ["Ks"]],
            "draw_pile": ["Qs"],
        },
        "observation_fingerprint": "forged",
        "config": {
            "candidate_samples": 1,
            "evaluation_samples": 1,
            "seed": 1,
            "candidate_seed": 1,
            "evaluation_seed": 2,
            "run_id": "unsafe-raw-world",
        },
    }
    with pytest.raises(HuM3RustError):
        evaluate_request(unsafe, library=m3_native)
