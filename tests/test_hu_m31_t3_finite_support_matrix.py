from __future__ import annotations

import os
from pathlib import Path

import pytest

from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m3_rust import (
    build_native_engine,
    evaluate_batch,
    evaluate_t3_explicit_support,
    load_native_engine,
    native_library_path,
    repository_root,
    t3_explicit_support_request,
)
from ofc_regular.hu_turn3_joint_exact_teacher import (
    T3ExplicitWorld,
    evaluate_t3_exact_explicit_support_actions,
)
from ofc_regular.state import Board


CONTINUATION_POLICY_ID = "m31_test_canonical_min_action_key_v1"


def _canonical_min_action(observation: ActorObservation):
    actions = generate_turn_actions(
        observation.hero_board,
        observation.dealt_cards,
    )
    return min(actions, key=lambda action: action_key(action).sort_key())


def _finite_support_case(
    to_act_order: str,
) -> tuple[ActorObservation, tuple[T3ExplicitWorld, ...], list[float]]:
    if to_act_order == "first":
        observation = ActorObservation(
            hero_board=Board.from_rows(
                middle=("2c", "Td", "7c", "2s"),
                bottom=("9h", "Js", "Ad", "Jc", "Kc"),
            ),
            opponent_public_board=Board.from_rows(
                middle=("Qc", "Tc", "2h", "9s"),
                bottom=("2d", "Th", "3h", "4s", "5c"),
            ),
            dealt_cards=("Ts", "8h", "4h"),
            hero_private_discards=("9c", "3s"),
            seat="first",
            street="T3",
            to_act_order="first",
        )
        worlds = (
            T3ExplicitWorld(
                opponent_private_discards=("3d", "4d"),
                future_cards=(
                    "Kh",
                    "8d",
                    "Ac",
                    "5h",
                    "7h",
                    "Qh",
                    "Ah",
                    "4c",
                    "5s",
                ),
                weight=0.25,
                world_id="first-world-0",
            ),
            T3ExplicitWorld(
                opponent_private_discards=("5h", "6s"),
                future_cards=(
                    "Jh",
                    "8d",
                    "Jd",
                    "5s",
                    "8s",
                    "Ks",
                    "7h",
                    "5d",
                    "3c",
                ),
                weight=0.75,
                world_id="first-world-1",
            ),
        )
        expected_scores = [6.0, 6.0, 6.0, 4.5, 4.5, 4.5, 1.5, 0.0, 0.0]
        return observation, worlds, expected_scores
    if to_act_order == "second":
        observation = ActorObservation(
            hero_board=Board.from_rows(
                middle=("Qc", "7d", "Th", "2d"),
                bottom=("2c", "2s", "7h", "8c", "7s"),
            ),
            opponent_public_board=Board.from_rows(
                top=("4c",),
                middle=("4d", "3c", "3h", "5h", "9c"),
                bottom=("3d", "Ah", "6s", "5c", "6h"),
            ),
            dealt_cards=("8d", "8s", "As"),
            hero_private_discards=("Js", "6d"),
            seat="second",
            street="T3",
            to_act_order="second",
        )
        worlds = (
            T3ExplicitWorld(
                opponent_private_discards=("9d", "Ad", "Ac"),
                future_cards=("Jh", "9s", "Qs", "5d", "Kd", "Jc"),
                weight=0.25,
                world_id="second-world-0",
            ),
            T3ExplicitWorld(
                opponent_private_discards=("4h", "7c", "9s"),
                future_cards=("8h", "Kd", "5s", "2h", "Jh", "Jc"),
                weight=0.75,
                world_id="second-world-1",
            ),
        )
        expected_scores = [
            -0.5,
            -0.5,
            -2.25,
            -2.25,
            -6.0,
            -6.0,
            -6.0,
            -6.0,
            -6.0,
        ]
        return observation, worlds, expected_scores
    raise ValueError("to_act_order must be first or second")


def _world_payload(worlds: tuple[T3ExplicitWorld, ...]) -> list[dict[str, object]]:
    return [
        {
            "opponent_private_discards": list(world.opponent_private_discards),
            "future_cards": list(world.future_cards),
            "weight": world.weight,
            "world_id": world.world_id,
        }
        for world in worlds
    ]


@pytest.fixture(scope="session")
def m31_finite_support_native():
    accepted = native_library_path(
        release=True,
        target_dir=repository_root() / "target" / "m30_build",
    )
    configured = os.environ.get("HU_M31_TEST_NATIVE_LIBRARY")
    path = Path(configured) if configured else accepted
    if not path.is_file():
        path = build_native_engine(
            release=False,
            target_dir=repository_root() / "target" / "hu_m31_support_test",
        ).library_path
    return load_native_engine(path=path)


def _assert_exact_result_parity(actual: dict, expected: dict) -> None:
    for field in (
        "schema",
        "mode",
        "full_52_card_tree_claimed",
        "observation_fingerprint",
        "seat",
        "to_act_order",
        "continuation_policy_id",
        "continuation_policy_fingerprint",
        "support_count",
        "support_digest",
        "legal_action_count",
        "selected_action_original_index",
        "selected_action_key",
    ):
        assert actual[field] == expected[field]
    assert actual["support_weight_sum"] == pytest.approx(
        expected["support_weight_sum"], rel=0.0, abs=1e-12
    )
    assert actual["best_score"] == pytest.approx(
        expected["best_score"], rel=0.0, abs=1e-12
    )

    actual_rows = {row["action_key"]: row for row in actual["actions"]}
    expected_rows = {row["action_key"]: row for row in expected["actions"]}
    assert actual_rows.keys() == expected_rows.keys()
    for key, expected_row in expected_rows.items():
        actual_row = actual_rows[key]
        for field in ("original_index", "sorted_index", "future_count"):
            assert actual_row[field] == expected_row[field]
        for field in ("score", "joint_ev", "regret_vs_best"):
            assert actual_row[field] == pytest.approx(
                expected_row[field], rel=0.0, abs=1e-12
            )


def test_both_seat_python_scalar_and_native_batch_finite_support_exact_parity(
    m31_finite_support_native,
):
    cases = []
    requests = []
    scalar_results = []
    for to_act_order in ("first", "second"):
        observation, worlds, expected_scores = _finite_support_case(to_act_order)
        expected = evaluate_t3_exact_explicit_support_actions(
            observation=observation,
            worlds=worlds,
            t4_selector=_canonical_min_action,
            t3_second_selector=(
                _canonical_min_action if to_act_order == "first" else None
            ),
            continuation_policy_id=CONTINUATION_POLICY_ID,
        )
        assert [row["score"] for row in expected["actions"]] == expected_scores
        payload = _world_payload(worlds)
        cases.append(expected)
        requests.append(
            t3_explicit_support_request(
                observation,
                worlds=payload,
                continuation_policy_id=CONTINUATION_POLICY_ID,
            )
        )
        scalar_results.append(
            evaluate_t3_explicit_support(
                observation,
                worlds=payload,
                continuation_policy_id=CONTINUATION_POLICY_ID,
                library=m31_finite_support_native,
            )
        )

    batch_results = evaluate_batch(requests, library=m31_finite_support_native)
    assert len(cases) == len(scalar_results) == len(batch_results) == 2
    for expected, scalar, batch in zip(
        cases, scalar_results, batch_results, strict=True
    ):
        _assert_exact_result_parity(scalar, expected)
        _assert_exact_result_parity(batch, expected)
        assert batch == scalar


@pytest.mark.parametrize(
    ("weights", "error_type", "match"),
    (
        ((0.0, 1.0), ValueError, "finite and strictly positive"),
        ((-0.25, 1.25), ValueError, "finite and strictly positive"),
        ((float("nan"), 1.0), ValueError, "finite and strictly positive"),
        ((0.2, 0.2), ValueError, "weights must sum to one"),
        ((True, 0.5), TypeError, "weight must be a real number"),
    ),
)
def test_public_binding_rejects_invalid_explicit_support_weights(
    weights,
    error_type,
    match,
):
    observation, worlds, _expected_scores = _finite_support_case("second")
    payload = _world_payload(worlds)
    payload[0]["weight"], payload[1]["weight"] = weights

    with pytest.raises(error_type, match=match):
        t3_explicit_support_request(
            observation,
            worlds=payload,
            continuation_policy_id=CONTINUATION_POLICY_ID,
        )


@pytest.mark.parametrize(
    ("mutate", "error_type", "match"),
    (
        (
            lambda world: world.pop("world_id"),
            ValueError,
            "invalid explicit T3 world 0 schema",
        ),
        (
            lambda world: world.update({"outer_truth": []}),
            ValueError,
            "invalid explicit T3 world 0 schema",
        ),
        (
            lambda world: world.update({"future_cards": "not-a-card-sequence"}),
            TypeError,
            "future_cards must be a card sequence",
        ),
        (
            lambda world: world.update({"world_id": 7}),
            TypeError,
            "world_id must be a string",
        ),
    ),
)
def test_public_binding_rejects_malformed_explicit_support_worlds(
    mutate,
    error_type,
    match,
):
    observation, worlds, _expected_scores = _finite_support_case("second")
    payload = _world_payload(worlds)
    mutate(payload[0])

    with pytest.raises(error_type, match=match):
        t3_explicit_support_request(
            observation,
            worlds=payload,
            continuation_policy_id=CONTINUATION_POLICY_ID,
        )
