import hashlib
import json
from dataclasses import replace
from types import MappingProxyType

import pytest

import ai.tutor.exact_late as exact_late_module
import ai.tutor.t4_btn_exact_resolver as resolver_module
from ai.engine.action_space import get_action_from_semantic_index_if_valid
from ai.engine.encoding import Board
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.t3_hu_public_cfr import InfoSetKey, PrivateRecall
from ai.tutor.t3_t4_infoset_encoder import semantic_action_ids
from ai.tutor.t4_btn_exact_resolver import (
    T4_BTN_EXACT_METHOD,
    T4BtnExactResolveError,
    resolve_t4_second_btn_exact,
    verify_t4_second_btn_exact,
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
BB_T4 = (("Ad", "bottom"), ("Kd", "bottom"))
BB_BOARD_11 = (BB_BOARD_9[0], BB_BOARD_9[1], ("Qc", "Qd", "Kh"))
BTN_BOARD_11 = (BTN_BOARD_9[0], BTN_BOARD_9[1], ("As", "2h", "3h"))
BB_BOARD_13 = (
    BB_BOARD_11[0],
    BB_BOARD_11[1],
    ("Qc", "Qd", "Kh", "Ad", "Kd"),
)
HISTORY_AFTER_T3 = PUBLIC_HISTORY_T3_FIRST + (
    (3, "bb", BB_T3),
    (3, "btn", BTN_T3),
)
HISTORY_T4_SECOND = HISTORY_AFTER_T3 + ((4, "bb", BB_T4),)


def _btn_recall() -> PrivateRecall:
    return PrivateRecall(
        dealt_by_turn=(
            (1, ("9d", "Jh", "4s")),
            (2, ("Qs", "Kc", "7h")),
            (3, ("2h", "3h", "4d")),
        ),
        discards_by_turn=((1, "4s"), (2, "7h"), (3, "4d")),
    )


def _bb_recall() -> PrivateRecall:
    return PrivateRecall(
        dealt_by_turn=(
            (1, ("7d", "8h", "6c")),
            (2, ("9s", "Tc", "6s")),
            (3, ("Qd", "Kh", "5d")),
        ),
        discards_by_turn=((1, "6c"), (2, "6s"), (3, "5d")),
    )


def _t4_second(draw=("5s", "8d", "X1")) -> InfoSetKey:
    return InfoSetKey(
        contract_version=POSITION_CONTRACT_VERSION,
        actor="btn",
        turn=4,
        phase="t4_second",
        board_bb=BB_BOARD_13,
        board_btn=BTN_BOARD_11,
        public_action_history=HISTORY_T4_SECOND,
        own_recall=_btn_recall(),
        current_draw=draw,
        fantasy_state=None,
    )


def _t4_first() -> InfoSetKey:
    return InfoSetKey(
        contract_version=POSITION_CONTRACT_VERSION,
        actor="bb",
        turn=4,
        phase="t4_first",
        board_bb=BB_BOARD_11,
        board_btn=BTN_BOARD_11,
        public_action_history=HISTORY_AFTER_T3,
        own_recall=_bb_recall(),
        current_draw=("5s", "8d", "X1"),
        fantasy_state=None,
    )


@pytest.mark.parametrize(
    "draw",
    [("5s", "8d", "9h"), ("5s", "X1", "X2")],
    ids=["natural", "two-jokers"],
)
def test_t4_btn_exact_resolver_enumerates_all_actions_and_selects_true_max(draw):
    information = _t4_second(draw)
    result = resolve_t4_second_btn_exact(information)
    verification = verify_t4_second_btn_exact(
        information,
        result,
        expected_manifest_sha256=result.manifest_sha256,
    )

    btn_board = Board(
        top=list(information.board_btn[0]),
        middle=list(information.board_btn[1]),
        bottom=list(information.board_btn[2]),
    )
    bb_board = Board(
        top=list(information.board_bb[0]),
        middle=list(information.board_bb[1]),
        bottom=list(information.board_bb[2]),
    )
    expected = {}
    for index, action_id in enumerate(semantic_action_ids(information)):
        if action_id is None:
            continue
        action = get_action_from_semantic_index_if_valid(
            index,
            list(information.current_draw),
            btn_board,
        )
        assert action is not None
        final_btn = exact_late_module.apply_action(btn_board, action)
        btn_score = exact_late_module.terminal_metrics(final_btn, bb_board)["score"]
        bb_score = exact_late_module.terminal_metrics(bb_board, final_btn)["score"]
        assert btn_score == pytest.approx(-bb_score)
        expected[action_id] = btn_score

    assert dict(result.utility_by_action_id) == expected
    assert result.selected_action_id == min(
        action_id
        for action_id, value in expected.items()
        if value == max(expected.values())
    )
    assert result.optimal_action_ids == tuple(
        action_id
        for action_id, value in sorted(expected.items())
        if value == max(expected.values())
    )
    assert sum(result.action_probabilities.values()) == 1.0
    assert result.action_probabilities[result.selected_action_id] == 1.0
    assert verification["legal_action_count"] == len(expected) == 3
    assert verification["exact_terminal_utility"] is True
    assert verification["promotion_eligible"] is False
    assert verification["runtime_integrated"] is False

    manifest = json.loads(result.manifest_json)
    assert manifest["method"] == T4_BTN_EXACT_METHOD
    assert manifest["opponent_hidden_cards_used"] is False
    assert manifest["chance_sampling_used"] is False
    assert manifest["all_legal_actions_enumerated"] is True
    assert "opponent_discards" not in result.manifest_json
    assert "undealt_cards" not in result.manifest_json


def test_t4_btn_exact_resolver_rejects_nonterminal_and_fantasy_inputs():
    with pytest.raises(T4BtnExactResolveError, match="BTN t4_second"):
        resolve_t4_second_btn_exact(_t4_first())
    with pytest.raises(T4BtnExactResolveError, match="fantasy_state=None"):
        resolve_t4_second_btn_exact(replace(_t4_second(), fantasy_state="fantasy"))


def test_t4_btn_exact_verifier_recomputes_tables_and_rejects_tampering():
    information = _t4_second()
    result = resolve_t4_second_btn_exact(information)

    utility = dict(result.utility_by_action_id)
    utility[result.selected_action_id] += 1.0
    with pytest.raises(T4BtnExactResolveError, match="utility table mismatch"):
        verify_t4_second_btn_exact(
            information,
            replace(result, utility_by_action_id=MappingProxyType(utility)),
        )

    policy = dict(result.action_probabilities)
    policy[result.selected_action_id] = 0.5
    with pytest.raises(T4BtnExactResolveError, match="policy mismatch"):
        verify_t4_second_btn_exact(
            information,
            replace(result, action_probabilities=MappingProxyType(policy)),
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
    with pytest.raises(T4BtnExactResolveError, match="manifest content mismatch"):
        verify_t4_second_btn_exact(
            information,
            replace(
                result,
                manifest_json=tampered_json,
                manifest_sha256=hashlib.sha256(tampered_json.encode()).hexdigest(),
            ),
        )


def test_t4_btn_exact_resolver_fails_closed_on_runtime_callable_or_helper_drift(
    monkeypatch,
):
    information = _t4_second()
    original_terminal = exact_late_module.terminal_metrics
    with monkeypatch.context() as patch:
        patch.setattr(
            exact_late_module,
            "terminal_metrics",
            lambda board, opponent=None: original_terminal(board, opponent),
        )
        with pytest.raises(T4BtnExactResolveError, match="runtime callable drifted"):
            resolve_t4_second_btn_exact(information)

    original_manifest = resolver_module._build_manifest
    with monkeypatch.context() as patch:
        patch.setattr(
            resolver_module,
            "_build_manifest",
            lambda *args, **kwargs: original_manifest(*args, **kwargs),
        )
        with pytest.raises(T4BtnExactResolveError, match="runtime helper drifted"):
            resolve_t4_second_btn_exact(information)


def test_t4_btn_exact_verifier_requires_external_hash_when_supplied():
    information = _t4_second()
    result = resolve_t4_second_btn_exact(information)
    with pytest.raises(T4BtnExactResolveError, match="externally trusted"):
        verify_t4_second_btn_exact(
            information,
            result,
            expected_manifest_sha256="0" * 64,
        )


def test_t4_btn_exact_source_binding_has_exact_live_schema_and_dependencies():
    information = _t4_second()
    result = resolve_t4_second_btn_exact(information)
    source = json.loads(result.manifest_json)["source_binding"]

    assert source["schema"] == "ofc_t4_btn_exact_source_binding/v2"
    assert set(source) == {
        "schema",
        "files",
        "infoset_encoder_manifest_sha256",
        "action_semantics_sha256",
        "runtime_semantic_binding_schema",
        "runtime_semantic_binding_sha256",
        "binding_sha256",
    }
    assert set(source["files"]) == {
        "t4_btn_exact_resolver",
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


@pytest.mark.parametrize("tamper", ("empty-files", "false-hash"))
def test_t4_btn_exact_verifier_rejects_resigned_source_binding_tampering(tamper):
    information = _t4_second()
    result = resolve_t4_second_btn_exact(information)
    manifest = json.loads(result.manifest_json)
    source = manifest["source_binding"]
    if tamper == "empty-files":
        source["files"] = {}
        expected_message = "file set drifted"
    else:
        first = min(source["files"])
        source["files"][first] = "0" * 64
        expected_message = "file hash drifted"
    unsigned_source = dict(source)
    unsigned_source.pop("binding_sha256")
    source["binding_sha256"] = resolver_module._canonical_sha256(unsigned_source)
    tampered_json = resolver_module._canonical_json(manifest)
    tampered = replace(
        result,
        manifest_json=tampered_json,
        manifest_sha256=hashlib.sha256(tampered_json.encode("utf-8")).hexdigest(),
    )

    with pytest.raises(T4BtnExactResolveError, match=expected_message):
        verify_t4_second_btn_exact(information, tampered)


def test_t4_btn_exact_rejects_transitive_scoring_or_live_fl_ev_drift(monkeypatch):
    information = _t4_second()
    result = resolve_t4_second_btn_exact(information)
    original_score = exact_late_module._score_against_complete_opponent

    with monkeypatch.context() as patch:
        def forged_score(my_board, opponent_board, *, include_fl_ev):
            value = original_score(
                my_board,
                opponent_board,
                include_fl_ev=include_fl_ev,
            )
            return value + (1000.0 if "8d" in my_board.bottom else 0.0)

        patch.setattr(
            exact_late_module,
            "_score_against_complete_opponent",
            forged_score,
        )
        with pytest.raises(T4BtnExactResolveError, match="semantic binding drifted"):
            resolve_t4_second_btn_exact(information)
        with pytest.raises(T4BtnExactResolveError, match="semantic binding drifted"):
            verify_t4_second_btn_exact(information, result)

    with monkeypatch.context() as patch:
        changed_fl_ev = dict(exact_late_module.RolloutEvaluator.FL_EV)
        changed_fl_ev[15] = float(changed_fl_ev.get(15, 0.0)) + 1.0
        patch.setattr(exact_late_module.RolloutEvaluator, "FL_EV", changed_fl_ev)
        with pytest.raises(T4BtnExactResolveError, match="semantic binding drifted"):
            resolve_t4_second_btn_exact(information)


@pytest.mark.parametrize(
    ("helper", "replacement"),
    (
        ("_source_paths", lambda: {}),
        ("_file_sha256", lambda _path: "0" * 64),
    ),
)
def test_t4_btn_exact_rejects_source_attestor_helper_bypass(
    monkeypatch,
    helper,
    replacement,
):
    with monkeypatch.context() as patch:
        patch.setattr(resolver_module, helper, replacement)
        with pytest.raises(T4BtnExactResolveError, match="runtime helper drifted"):
            resolve_t4_second_btn_exact(_t4_second())


def test_t4_btn_exact_rejects_runtime_guard_alias_bypass(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(resolver_module, "_CANONICAL_RUNTIME_GUARD", lambda: None)
        with pytest.raises(T4BtnExactResolveError, match="runtime guard alias drifted"):
            resolve_t4_second_btn_exact(_t4_second())


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
def test_t4_btn_exact_rejects_each_canonical_module_alias_substitution(
    monkeypatch,
    alias,
):
    expected_aliases = {
        name for name, _module in resolver_module._CANONICAL_MODULE_ALIASES
    }
    assert expected_aliases == {
        "_action_space",
        "_encoding",
        "_turn_order",
        "_exact_late",
        "_multi_root",
        "_public_cfr",
        "_infoset_encoder",
    }
    with monkeypatch.context() as patch:
        patch.setattr(resolver_module, alias, object())
        with pytest.raises(T4BtnExactResolveError, match="canonical module alias drifted"):
            resolve_t4_second_btn_exact(_t4_second())


def test_t4_btn_exact_rejects_paired_scoring_and_graph_module_proxies(monkeypatch):
    class ModuleProxy:
        def __init__(self, target, overrides):
            self._target = target
            self._overrides = overrides

        def __getattr__(self, name):
            if name in self._overrides:
                return self._overrides[name]
            return getattr(self._target, name)

    original_terminal = exact_late_module.terminal_metrics

    def forged_terminal(board, opponent=None):
        metrics = dict(original_terminal(board, opponent))
        metrics["score"] += 999.0
        metrics["raw_score"] += 999.0
        return metrics

    canonical_graph = resolver_module._live_runtime_semantic_binding()
    forged_exact_late = ModuleProxy(
        resolver_module._exact_late,
        {"terminal_metrics": forged_terminal},
    )
    forged_multi_root = ModuleProxy(
        resolver_module._multi_root,
        {"_runtime_semantic_graph": lambda *_args, **_kwargs: canonical_graph},
    )
    with monkeypatch.context() as patch:
        patch.setattr(resolver_module, "_exact_late", forged_exact_late)
        patch.setattr(resolver_module, "_multi_root", forged_multi_root)
        with pytest.raises(T4BtnExactResolveError, match="canonical module alias drifted"):
            resolve_t4_second_btn_exact(_t4_second())


def test_t4_btn_exact_verifier_requires_strict_float_result_maps():
    information = _t4_second()
    result = resolve_t4_second_btn_exact(information)

    bool_policy = MappingProxyType(
        {action_id: bool(value) for action_id, value in result.action_probabilities.items()}
    )
    with pytest.raises(T4BtnExactResolveError, match="finite strict float"):
        verify_t4_second_btn_exact(
            information,
            replace(result, action_probabilities=bool_policy),
        )

    integer_utility = MappingProxyType(
        {action_id: int(value) for action_id, value in result.utility_by_action_id.items()}
    )
    with pytest.raises(T4BtnExactResolveError, match="finite strict float"):
        verify_t4_second_btn_exact(
            information,
            replace(result, utility_by_action_id=integer_utility),
        )

    metrics = {
        action_id: dict(row)
        for action_id, row in result.terminal_metrics_by_action_id.items()
    }
    metrics[result.selected_action_id]["score"] = int(
        metrics[result.selected_action_id]["score"]
    )
    with pytest.raises(T4BtnExactResolveError, match="finite strict float"):
        verify_t4_second_btn_exact(
            information,
            replace(result, terminal_metrics_by_action_id=MappingProxyType(metrics)),
        )
