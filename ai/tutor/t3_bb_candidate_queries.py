"""Information-safe T3-BB candidate-policy query helpers.

The T3-BB likelihood used by a BTN ``t3_second`` posterior is produced by
the same public information set that the full-card MCCFR solver calls
``t3_first``.  This module is the strict boundary between those two
representations.  It also converts a solver float strategy into a stable,
exact Q32 table suitable for a frozen behavior artifact.

Neither helper accepts a hidden particle, opponent recall, a remaining deck,
or an observed action.  A conversion succeeds only when every public/private
recall field and the complete legal action support rederive exactly.
"""
from __future__ import annotations

import math
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.exact_late import action_key
from ai.tutor.t3_hu_full_card_range import BehaviorInfoSet
from ai.tutor.t3_hu_public_cfr import InfoSetKey, PrivateRecall


Q32_DENOMINATOR = 1 << 32
MCCFR_SUM_ABS_TOLERANCE = 1e-12


def _board(rows: Sequence[Sequence[str]]) -> Board:
    if len(rows) != 3:
        raise ValueError("board must contain top/middle/bottom rows")
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def _legal_action_ids(key: InfoSetKey) -> tuple[str, ...]:
    actor_rows = key.board_bb if key.actor == "bb" else key.board_btn
    actions = get_turn_actions(list(key.current_draw), _board(actor_rows))
    action_ids = tuple(sorted(action_key(action) for action in actions))
    if not action_ids:
        raise ValueError("T3-BB information set has no legal actions")
    if len(action_ids) != len(set(action_ids)):
        raise RuntimeError("action engine generated duplicate canonical action IDs")
    return action_ids


def behavior_t3_bb_to_t3_first_key(information: BehaviorInfoSet) -> InfoSetKey:
    """Convert an exact T3-BB behavior query to its public MCCFR key.

    The returned key is constructed only from fields already present in the
    acting BB's pre-action :class:`BehaviorInfoSet`.  ``InfoSetKey`` then
    independently validates the T3-first public-history cutoff, boards,
    private perfect recall, current draw, and physical-card disjointness.
    Finally, every copied field and the action-engine-derived legal support is
    compared back to the input.  Canonicalization is therefore validation,
    never a silent repair of a malformed behavior query.
    """

    if not isinstance(information, BehaviorInfoSet):
        raise TypeError("information must be a BehaviorInfoSet")
    if information.actor != "bb":
        raise ValueError("candidate-policy conversion requires actor='bb'")
    if isinstance(information.turn, bool) or not isinstance(information.turn, int):
        raise TypeError("candidate-policy conversion requires integer turn=3")
    if information.turn != 3:
        raise ValueError("candidate-policy conversion requires turn=3")
    if not isinstance(information.own_recall_before, PrivateRecall):
        raise TypeError("own_recall_before must be PrivateRecall")
    if information.fantasy_state is not None and not isinstance(
        information.fantasy_state, str
    ):
        raise TypeError("fantasy_state must be a string or None")

    key = InfoSetKey(
        contract_version=POSITION_CONTRACT_VERSION,
        actor="bb",
        turn=3,
        phase="t3_first",
        board_bb=information.board_bb,
        board_btn=information.board_btn,
        public_action_history=information.public_action_history,
        own_recall=information.own_recall_before,
        current_draw=information.current_draw,
        fantasy_state=information.fantasy_state,
    )

    # InfoSetKey canonicalizes its inputs.  Require the input to have already
    # been canonical so conversion cannot silently sort/fix a producer bug.
    comparisons: tuple[tuple[str, Any, Any], ...] = (
        ("actor", information.actor, key.actor),
        ("turn", information.turn, key.turn),
        ("board_bb", information.board_bb, key.board_bb),
        ("board_btn", information.board_btn, key.board_btn),
        (
            "public_action_history",
            information.public_action_history,
            key.public_action_history,
        ),
        ("own_recall", information.own_recall_before, key.own_recall),
        ("current_draw", information.current_draw, key.current_draw),
        ("fantasy_state", information.fantasy_state, key.fantasy_state),
    )
    mismatches = [label for label, source, converted in comparisons if source != converted]
    if mismatches:
        raise ValueError(
            "T3-BB behavior query is not already canonical; mismatched fields="
            f"{mismatches}"
        )

    derived_action_ids = _legal_action_ids(key)
    supplied_action_ids = information.legal_action_ids
    if supplied_action_ids != derived_action_ids:
        supplied = set(supplied_action_ids)
        derived = set(derived_action_ids)
        raise ValueError(
            "T3-BB legal action support does not match the action engine: "
            f"missing={sorted(derived - supplied)}, "
            f"extra={sorted(supplied - derived)}"
        )

    # Runs InfoSetKey's explicit forbidden-field serialization defense.
    key.canonical_json()
    return key


def _validated_legal_support(legal_action_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(legal_action_ids, (str, bytes)) or not isinstance(
        legal_action_ids, Sequence
    ):
        raise TypeError("legal_action_ids must be a sequence of strings")
    supplied = tuple(legal_action_ids)
    if not supplied:
        raise ValueError("legal_action_ids must not be empty")
    if any(not isinstance(action_id, str) or not action_id for action_id in supplied):
        raise TypeError("legal_action_ids must contain non-empty strings")
    if len(supplied) != len(set(supplied)):
        raise ValueError("legal_action_ids must be unique")
    return tuple(sorted(supplied))


def quantize_mccfr_distribution_q32(
    action_probabilities: Mapping[str, float],
    *,
    legal_action_ids: Sequence[str],
) -> Mapping[str, Fraction]:
    """Return deterministic largest-remainder Q32 probabilities.

    The input must be a complete MCCFR float distribution over exactly the
    supplied legal action IDs.  Values must be built-in finite non-negative
    floats and sum to one within ``1e-12`` absolute error.  Quotas are then
    calculated from each float's exact binary rational value.  Equal
    remainders are awarded in lexical action-ID order, making the result
    independent of mapping insertion order.
    """

    if not isinstance(action_probabilities, Mapping):
        raise TypeError("action_probabilities must be a mapping")
    ordered = _validated_legal_support(legal_action_ids)
    if any(not isinstance(action_id, str) for action_id in action_probabilities):
        raise TypeError("action_probabilities keys must be strings")
    actual_support = set(action_probabilities)
    expected_support = set(ordered)
    if actual_support != expected_support:
        raise ValueError(
            "MCCFR distribution support does not match legal actions: "
            f"missing={sorted(expected_support - actual_support)}, "
            f"extra={sorted(actual_support - expected_support)}"
        )

    raw: dict[str, Fraction] = {}
    float_values: list[float] = []
    for action_id in ordered:
        probability = action_probabilities[action_id]
        if isinstance(probability, bool) or not isinstance(probability, float):
            raise TypeError(
                f"MCCFR probability for {action_id!r} must be a built-in float"
            )
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError("MCCFR probabilities must be finite and non-negative")
        float_values.append(probability)
        raw[action_id] = Fraction.from_float(probability)

    float_total = math.fsum(float_values)
    if not math.isclose(
        float_total,
        1.0,
        rel_tol=0.0,
        abs_tol=MCCFR_SUM_ABS_TOLERANCE,
    ):
        raise ValueError(
            "MCCFR probabilities must sum to one within absolute tolerance "
            f"{MCCFR_SUM_ABS_TOLERANCE}; got {float_total!r}"
        )
    exact_total = sum(raw.values(), Fraction(0, 1))
    if exact_total <= 0:
        raise ValueError("MCCFR probabilities must carry positive mass")

    floor_units: dict[str, int] = {}
    remainders: list[tuple[Fraction, str]] = []
    for action_id in ordered:
        quota = raw[action_id] * Q32_DENOMINATOR / exact_total
        units = quota.numerator // quota.denominator
        floor_units[action_id] = units
        remainders.append((quota - units, action_id))

    units_left = Q32_DENOMINATOR - sum(floor_units.values())
    if not 0 <= units_left <= len(ordered):
        raise RuntimeError("largest-remainder Q32 residual is outside its valid range")
    for _remainder, action_id in sorted(
        remainders,
        key=lambda item: (-item[0], item[1]),
    )[:units_left]:
        floor_units[action_id] += 1

    quantized = {
        action_id: Fraction(floor_units[action_id], Q32_DENOMINATOR)
        for action_id in ordered
    }
    if set(quantized) != expected_support:
        raise RuntimeError("Q32 quantization changed legal action support")
    if any(probability < 0 for probability in quantized.values()):
        raise RuntimeError("Q32 quantization produced negative probability")
    if sum(quantized.values(), Fraction(0, 1)) != 1:
        raise RuntimeError("Q32 probabilities do not sum exactly to one")
    if any(
        (probability * Q32_DENOMINATOR).denominator != 1
        for probability in quantized.values()
    ):
        raise RuntimeError("Q32 probability is not an integral Q32 unit count")
    return MappingProxyType(quantized)


__all__ = [
    "MCCFR_SUM_ABS_TOLERANCE",
    "Q32_DENOMINATOR",
    "behavior_t3_bb_to_t3_first_key",
    "quantize_mccfr_distribution_q32",
]
