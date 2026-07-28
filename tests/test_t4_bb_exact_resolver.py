import hashlib
import json
from dataclasses import replace
from itertools import combinations
from math import comb
from types import MappingProxyType

import pytest

import ai.tutor.exact_late as exact_late_module
import ai.tutor.t4_bb_exact_resolver as resolver_module
from ai.engine.action_space import (
    get_action_from_semantic_index_if_valid,
    get_turn_actions,
)
from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.t3_hu_public_cfr import InfoSetKey, PrivateRecall
from ai.tutor.t3_t4_infoset_encoder import semantic_action_ids
from ai.tutor.t4_bb_exact_resolver import (
    T4_BB_BELIEF_MODEL,
    T4_BB_EXACT_METHOD,
    T4BbExactResolveError,
    resolve_t4_first_bb_exact,
    verify_t4_first_bb_exact,
)


BB_BOARD_9 = (
    ("4h", "2c", "3d"),
    ("9s", "7c", "8h", "7d", "Tc"),
    ("Qc",),
)
BTN_BOARD_9 = (
    ("8c", "5c", "6d"),
    ("Kc", "9c", "Jh", "9d", "Qs"),
    ("As",),
)
PUBLIC_HISTORY_T3_FIRST = (
    (
        0,
        "bb",
        (
            ("4h", "top"),
            ("2c", "top"),
            ("3d", "top"),
            ("7c", "middle"),
            ("Qc", "bottom"),
        ),
    ),
    (
        0,
        "btn",
        (
            ("8c", "top"),
            ("5c", "top"),
            ("6d", "top"),
            ("9c", "middle"),
            ("As", "bottom"),
        ),
    ),
    (1, "bb", (("7d", "middle"), ("8h", "middle"))),
    (1, "btn", (("9d", "middle"), ("Jh", "middle"))),
    (2, "bb", (("9s", "middle"), ("Tc", "middle"))),
    (2, "btn", (("Qs", "middle"), ("Kc", "middle"))),
)
BB_T3 = (("Qd", "bottom"), ("Kh", "bottom"))
BTN_T3 = (("2h", "bottom"), ("3h", "bottom"))
BB_BOARD_11 = (BB_BOARD_9[0], BB_BOARD_9[1], ("Qc", "Qd", "Kh"))
BTN_BOARD_11 = (BTN_BOARD_9[0], BTN_BOARD_9[1], ("As", "2h", "3h"))
HISTORY_AFTER_T3 = PUBLIC_HISTORY_T3_FIRST + (
    (3, "bb", BB_T3),
    (3, "btn", BTN_T3),
)
BB_DISCARDS = ("6c", "6s", "5d")
CURRENT_DRAW = ("5s", "8d", "9h")


def _bb_recall() -> PrivateRecall:
    return PrivateRecall(
        dealt_by_turn=(
            (1, ("7d", "8h", "6c")),
            (2, ("9s", "Tc", "6s")),
            (3, ("Qd", "Kh", "5d")),
        ),
        discards_by_turn=((1, "6c"), (2, "6s"), (3, "5d")),
    )


def _t4_first(draw=CURRENT_DRAW) -> InfoSetKey:
    return InfoSetKey(
        contract_version=POSITION_CONTRACT_VERSION,
        actor="bb",
        turn=4,
        phase="t4_first",
        board_bb=BB_BOARD_11,
        board_btn=BTN_BOARD_11,
        public_action_history=HISTORY_AFTER_T3,
        own_recall=_bb_recall(),
        current_draw=draw,
        fantasy_state=None,
    )


def _t4_second_wrong_phase() -> InfoSetKey:
    btn_recall = PrivateRecall(
        dealt_by_turn=(
            (1, ("9d", "Jh", "4s")),
            (2, ("Qs", "Kc", "7h")),
            (3, ("2h", "3h", "4d")),
        ),
        discards_by_turn=((1, "4s"), (2, "7h"), (3, "4d")),
    )
    bb_board_13 = (
        BB_BOARD_11[0],
        BB_BOARD_11[1],
        ("Qc", "Qd", "Kh", "Ad", "Kd"),
    )
    history = HISTORY_AFTER_T3 + ((4, "bb", (("Ad", "bottom"), ("Kd", "bottom"))),)
    return InfoSetKey(
        contract_version=POSITION_CONTRACT_VERSION,
        actor="btn",
        turn=4,
        phase="t4_second",
        board_bb=bb_board_13,
        board_btn=BTN_BOARD_11,
        public_action_history=history,
        own_recall=btn_recall,
        current_draw=("5s", "8d", "X1"),
        fantasy_state=None,
    )


@pytest.fixture(scope="module")
def resolved():
    information = _t4_first()
    result = resolve_t4_first_bb_exact(information)
    return information, result


def test_t4_bb_exact_resolver_matches_independent_uniform_deal_brute_force(resolved):
    information, result = resolved

    bb_board = Board(
        top=list(information.board_bb[0]),
        middle=list(information.board_bb[1]),
        bottom=list(information.board_bb[2]),
    )
    btn_board = Board(
        top=list(information.board_btn[0]),
        middle=list(information.board_btn[1]),
        bottom=list(information.board_btn[2]),
    )

    expected: dict[str, float] = {}
    for index, action_id in enumerate(semantic_action_ids(information)):
        if action_id is None:
            continue
        action = get_action_from_semantic_index_if_valid(
            index,
            list(information.current_draw),
            bb_board,
        )
        assert action is not None
        final_bb = exact_late_module.apply_action(bb_board, action)
        known = (
            set(final_bb.all_cards())
            | set(btn_board.all_cards())
            | set(BB_DISCARDS)
            | {action.discard}
        )
        deck = [card for card in ALL_CARDS if card not in known]
        assert len(deck) == 26
        total = 0.0
        draws = 0
        for opponent_draw in combinations(deck, 3):
            best_opponent_score = None
            for response in get_turn_actions(list(opponent_draw), btn_board):
                final_btn = exact_late_module.apply_action(btn_board, response)
                score = exact_late_module.terminal_metrics(final_btn, final_bb)[
                    "score"
                ]
                if best_opponent_score is None or score > best_opponent_score:
                    best_opponent_score = score
            assert best_opponent_score is not None
            total += -best_opponent_score
            draws += 1
        assert draws == comb(26, 3)
        expected[action_id] = total / draws

    assert set(result.utility_by_action_id) == set(expected)
    for action_id, value in expected.items():
        assert result.utility_by_action_id[action_id] == pytest.approx(
            value, rel=1e-9, abs=1e-9
        )

    best_value = max(result.utility_by_action_id.values())
    assert result.selected_action_id == min(
        action_id
        for action_id, value in result.utility_by_action_id.items()
        if value == best_value
    )
    assert result.optimal_action_ids == tuple(
        action_id
        for action_id, value in sorted(result.utility_by_action_id.items())
        if value == best_value
    )
    assert sum(result.action_probabilities.values()) == 1.0
    assert result.action_probabilities[result.selected_action_id] == 1.0

    for row in result.distribution_by_action_id.values():
        assert row["remaining_deck_size"] == 26
        assert row["enumerated_draws"] == comb(26, 3)


def test_t4_bb_exact_manifest_declares_belief_and_no_hidden_information(resolved):
    _information, result = resolved
    manifest = json.loads(result.manifest_json)

    assert manifest["schema"] == "ofc_t4_bb_exact_resolve/v1"
    assert manifest["method"] == T4_BB_EXACT_METHOD
    assert manifest["belief_model"] == T4_BB_BELIEF_MODEL
    assert manifest["phase"] == "t4_first"
    assert manifest["actor"] == "bb"
    assert manifest["utility_perspective"] == "bb"
    assert manifest["all_legal_actions_enumerated"] is True
    assert manifest["all_opponent_draws_enumerated"] is True
    assert manifest["chance_sampling_used"] is False
    assert manifest["policy_sampling_used"] is False
    assert manifest["opponent_hidden_cards_used"] is False
    assert manifest["exact_under_declared_belief"] is True
    assert manifest["bayes_posterior_used"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["runtime_integrated"] is False
    assert manifest["global_unseen_state_policy_claim"] is False
    assert manifest["serving_changed"] is False
    assert "opponent_discards" not in result.manifest_json
    assert "opponent_private_discards" not in result.manifest_json
    assert "undealt_cards" not in result.manifest_json

    source = manifest["source_binding"]
    assert source["schema"] == "ofc_t4_bb_exact_source_binding/v1"
    assert set(source["files"]) == {
        "t4_bb_exact_resolver",
        "exact_terminal_scoring",
        "action_space",
        "canonical_encoding",
        "canonical_turn_order",
        "canonical_game_evaluator",
        "canonical_joker_scoring",
        "fantasyland_utility",
        "fantasyland_utility_config",
        "infoset_contract",
        "infoset_action_contract",
        "runtime_semantic_graph",
    }
    assert resolver_module._verify_source_binding(source) == source


def test_t4_bb_exact_resolver_rejects_wrong_phase_and_fantasy_inputs():
    with pytest.raises(T4BbExactResolveError, match="BB t4_first"):
        resolve_t4_first_bb_exact(_t4_second_wrong_phase())
    with pytest.raises(T4BbExactResolveError, match="fantasy_state=None"):
        resolve_t4_first_bb_exact(replace(_t4_first(), fantasy_state="fantasy"))


def test_t4_bb_exact_verifier_recomputes_tables_and_rejects_tampering(resolved):
    information, result = resolved

    utility = dict(result.utility_by_action_id)
    utility[result.selected_action_id] += 1.0
    with pytest.raises(T4BbExactResolveError, match="utility table mismatch"):
        verify_t4_first_bb_exact(
            information,
            replace(result, utility_by_action_id=MappingProxyType(utility)),
        )

    manifest = json.loads(result.manifest_json)
    manifest["promotion_eligible"] = True
    tampered_json = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with pytest.raises(T4BbExactResolveError, match="manifest content mismatch"):
        verify_t4_first_bb_exact(
            information,
            replace(
                result,
                manifest_json=tampered_json,
                manifest_sha256=hashlib.sha256(tampered_json.encode()).hexdigest(),
            ),
        )


def test_t4_bb_exact_verifier_requires_external_hash_when_supplied(resolved):
    information, result = resolved
    with pytest.raises(T4BbExactResolveError, match="externally trusted"):
        verify_t4_first_bb_exact(
            information,
            result,
            expected_manifest_sha256="0" * 64,
        )


def test_t4_bb_exact_resolver_fails_closed_on_runtime_callable_or_helper_drift(
    monkeypatch,
):
    information = _t4_first()
    original_terminal = exact_late_module.terminal_metrics
    with monkeypatch.context() as patch:
        patch.setattr(
            exact_late_module,
            "terminal_metrics",
            lambda board, opponent=None: original_terminal(board, opponent),
        )
        with pytest.raises(T4BbExactResolveError, match="runtime callable drifted"):
            resolve_t4_first_bb_exact(information)

    original_manifest = resolver_module._build_manifest
    with monkeypatch.context() as patch:
        patch.setattr(
            resolver_module,
            "_build_manifest",
            lambda *args, **kwargs: original_manifest(*args, **kwargs),
        )
        with pytest.raises(T4BbExactResolveError, match="runtime helper drifted"):
            resolve_t4_first_bb_exact(information)


@pytest.mark.parametrize(
    "alias",
    (
        "_action_space",
        "_encoding",
        "_turn_order",
        "_exact_late",
        "_multi_root",
        "_public_cfr",
        "_infoset_encoder",
    ),
)
def test_t4_bb_exact_rejects_each_canonical_module_alias_substitution(
    monkeypatch,
    alias,
):
    with monkeypatch.context() as patch:
        patch.setattr(resolver_module, alias, object())
        with pytest.raises(T4BbExactResolveError, match="canonical module alias drifted"):
            resolve_t4_first_bb_exact(_t4_first())


def test_t4_bb_exact_rejects_live_fl_ev_drift(monkeypatch):
    information = _t4_first()
    with monkeypatch.context() as patch:
        changed_fl_ev = dict(exact_late_module.RolloutEvaluator.FL_EV)
        changed_fl_ev[15] = float(changed_fl_ev.get(15, 0.0)) + 1.0
        patch.setattr(exact_late_module.RolloutEvaluator, "FL_EV", changed_fl_ev)
        with pytest.raises(T4BbExactResolveError, match="semantic binding drifted"):
            resolve_t4_first_bb_exact(information)


def test_t4_bb_exact_rejects_runtime_guard_alias_bypass(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(resolver_module, "_CANONICAL_RUNTIME_GUARD", lambda: None)
        with pytest.raises(T4BbExactResolveError, match="runtime guard alias drifted"):
            resolve_t4_first_bb_exact(_t4_first())
