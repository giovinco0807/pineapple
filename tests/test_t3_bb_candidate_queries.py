from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction

import pytest

from ai.tutor.frozen_behavior_torch import _quantize_largest_remainder
from ai.tutor.t3_bb_candidate_queries import (
    Q32_DENOMINATOR,
    behavior_t3_bb_to_t3_first_key,
    quantize_mccfr_distribution_q32,
)
from ai.tutor.t3_hu_full_card_range import (
    BehaviorDistribution,
    build_history_weighted_full_card_range,
)
from ai.tutor.t3_hu_public_cfr import FORBIDDEN_INFOSET_FIELDS
from test_t3_hu_full_card_range import _later_phase_observation


class _CaptureExactUniformBehavior:
    model_id = "capture_t3_bb_candidate_query"

    def __init__(self) -> None:
        self.informations = []

    @property
    def model_manifest(self):
        return {
            "schema": "ofc_frozen_behavior_model/v1",
            "model_id": self.model_id,
            "model_type": type(self).__name__,
            "position_contract_version": "bb_first_v1",
        }

    @property
    def model_sha256(self):
        import hashlib

        encoded = json.dumps(
            self.model_manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def action_distribution(self, information):
        self.informations.append(information)
        probability = Fraction(1, information.legal_action_count)
        return BehaviorDistribution(
            information_digest=information.digest(),
            probabilities={
                action_id: probability for action_id in information.legal_action_ids
            },
            source="model",
            used_fallback=False,
        )


@pytest.fixture
def real_btn_root_t3_bb_query():
    """Capture the fifth behavior route from a physical BTN T3 root fixture."""

    observation = _later_phase_observation("t3_second")
    capture = _CaptureExactUniformBehavior()
    build_history_weighted_full_card_range(
        observation,
        capture,
        epsilon=0,
        max_particles=1,
        seed=17,
    )
    matches = [
        information
        for information in capture.informations
        if information.turn == 3 and information.actor == "bb"
    ]
    assert len(matches) == 1
    return matches[0]


def test_real_btn_fixture_t3_bb_query_maps_exactly_without_hidden_state(
    real_btn_root_t3_bb_query,
):
    information = real_btn_root_t3_bb_query
    key = behavior_t3_bb_to_t3_first_key(information)

    assert key.actor == information.actor == "bb"
    assert key.turn == information.turn == 3
    assert key.phase == "t3_first"
    assert key.board_bb == information.board_bb
    assert key.board_btn == information.board_btn
    assert key.public_action_history == information.public_action_history
    assert key.own_recall == information.own_recall_before
    assert key.current_draw == information.current_draw
    assert key.fantasy_state == information.fantasy_state
    assert [(turn, actor) for turn, actor, _ in key.public_action_history][-1] == (
        2,
        "btn",
    )
    serialized = key.canonical_json().lower()
    assert not [
        field for field in FORBIDDEN_INFOSET_FIELDS if f'"{field}"' in serialized
    ]


def test_t3_bb_query_action_support_tamper_fails_closed(real_btn_root_t3_bb_query):
    tampered = replace(
        real_btn_root_t3_bb_query,
        legal_action_ids=real_btn_root_t3_bb_query.legal_action_ids[:-1],
    )
    with pytest.raises(ValueError, match="legal action support"):
        behavior_t3_bb_to_t3_first_key(tampered)


def test_t3_bb_query_noncanonical_draw_tamper_is_not_silently_repaired(
    real_btn_root_t3_bb_query,
):
    tampered = replace(
        real_btn_root_t3_bb_query,
        current_draw=tuple(reversed(real_btn_root_t3_bb_query.current_draw)),
    )
    with pytest.raises(ValueError, match="not already canonical.*current_draw"):
        behavior_t3_bb_to_t3_first_key(tampered)


@pytest.mark.parametrize(
    "tampered",
    [
        lambda information: replace(information, actor="btn"),
        lambda information: replace(information, turn=2),
        lambda information: replace(
            information,
            public_action_history=information.public_action_history[:-1],
        ),
    ],
)
def test_t3_bb_query_route_or_history_tamper_fails_closed(
    real_btn_root_t3_bb_query, tampered
):
    with pytest.raises((TypeError, ValueError)):
        behavior_t3_bb_to_t3_first_key(tampered(real_btn_root_t3_bb_query))


def test_q32_quantization_matches_frozen_largest_remainder_reference():
    probabilities = {"z": 0.1, "a": 0.2, "m": 0.7}
    actual = quantize_mccfr_distribution_q32(
        probabilities,
        legal_action_ids=("m", "z", "a"),
    )
    expected = _quantize_largest_remainder(
        probabilities,
        denominator=Q32_DENOMINATOR,
    )

    assert dict(actual) == dict(expected)
    assert tuple(actual) == ("a", "m", "z")
    assert sum(actual.values(), Fraction(0, 1)) == 1
    assert all(
        (probability * Q32_DENOMINATOR).denominator == 1
        for probability in actual.values()
    )


def test_q32_equal_remainder_tie_break_is_lexical_and_insertion_stable():
    forward = {"z": 1.0 / 3.0, "a": 1.0 / 3.0, "m": 1.0 / 3.0}
    reverse = dict(reversed(tuple(forward.items())))
    first = quantize_mccfr_distribution_q32(
        forward, legal_action_ids=("z", "a", "m")
    )
    second = quantize_mccfr_distribution_q32(
        reverse, legal_action_ids=("m", "a", "z")
    )

    assert dict(first) == dict(second)
    units = {
        action_id: int(probability * Q32_DENOMINATOR)
        for action_id, probability in first.items()
    }
    assert units == {
        "a": Q32_DENOMINATOR // 3 + 1,
        "m": Q32_DENOMINATOR // 3,
        "z": Q32_DENOMINATOR // 3,
    }


@pytest.mark.parametrize(
    ("probabilities", "legal", "error"),
    [
        ({"a": 0.5}, ("a", "b"), "support"),
        ({"a": 0.5, "b": 0.5, "x": 0.0}, ("a", "b"), "support"),
        ({"a": float("nan"), "b": 0.0}, ("a", "b"), "finite"),
        ({"a": float("inf"), "b": 0.0}, ("a", "b"), "finite"),
        ({"a": -0.1, "b": 1.1}, ("a", "b"), "non-negative"),
        ({"a": 0.4, "b": 0.5}, ("a", "b"), "sum to one"),
        ({"a": 1, "b": 0.0}, ("a", "b"), "built-in float"),
        ({"a": True, "b": 0.0}, ("a", "b"), "built-in float"),
    ],
)
def test_q32_quantization_rejects_tampered_distributions(
    probabilities, legal, error
):
    with pytest.raises((TypeError, ValueError), match=error):
        quantize_mccfr_distribution_q32(
            probabilities,
            legal_action_ids=legal,
        )


def test_q32_quantization_rejects_duplicate_or_invalid_declared_support():
    with pytest.raises(ValueError, match="unique"):
        quantize_mccfr_distribution_q32(
            {"a": 1.0}, legal_action_ids=("a", "a")
        )
    with pytest.raises(TypeError, match="non-empty strings"):
        quantize_mccfr_distribution_q32(
            {"a": 1.0}, legal_action_ids=("a", 1)  # type: ignore[arg-type]
        )


def test_q32_result_is_read_only():
    result = quantize_mccfr_distribution_q32(
        {"a": 1.0}, legal_action_ids=("a",)
    )
    with pytest.raises(TypeError):
        result["a"] = Fraction(0, 1)  # type: ignore[index]
