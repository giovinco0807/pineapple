import copy
import hashlib
from fractions import Fraction

import pytest

from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.behavior_calibration_contract import (
    DECISION_LOG_SCHEMA,
    MANIFEST_SCHEMA,
    SPLIT_NAMESPACE,
    build_behavior_decision_log,
    build_behavior_decision_manifest,
    canonical_json,
    canonical_sha256,
    commit_hidden_root_trace,
    root_split,
    root_split_bucket,
    verify_behavior_decision_dataset,
    verify_behavior_decision_log,
)
from ai.tutor.exact_late import action_key
from ai.tutor.t3_hu_full_card_range import BehaviorInfoSet
from ai.tutor.t3_hu_public_cfr import PrivateRecall


ROWS = ("top", "middle", "bottom")
BB_T0 = (
    ("2c", "3d", "4h"),
    ("7c",),
    ("Qc",),
)
BTN_T0 = (
    ("5c", "6d", "8c"),
    ("9c",),
    ("As",),
)
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
    "name": "frozen_policy_categorical_v1",
    "temperature_numerator": 1,
    "temperature_denominator": 1,
}


def _board(rows):
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def _legal_ids(draw, rows):
    return tuple(sorted(action_key(action) for action in get_turn_actions(list(draw), _board(rows))))


def _append_placements(rows, placements):
    out = {row: list(cards) for row, cards in zip(ROWS, rows)}
    for card, row in placements:
        out[row].append(card)
    return tuple(tuple(sorted(out[row])) for row in ROWS)


def _canonical_placements(action):
    return tuple(sorted(action.placements, key=lambda item: (item[1], item[0])))


def _first_action(information):
    rows = information.board_bb if information.actor == "bb" else information.board_btn
    actions = get_turn_actions(list(information.current_draw), _board(rows))
    return min(actions, key=action_key)


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


def _t2_bb_information(draw=("Ah", "Jd", "X1")):
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


def _t2_btn_information(draw=("4d", "8d", "X2")):
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


def _record(
    information,
    *,
    root_id="root-001",
    root_commitment=None,
    seed_namespace="worker-00/seed-20260713",
    target="selfplay-population-v1",
    policy_sha256=POLICY_SHA256,
):
    observed = _first_action(information)
    commitment = root_commitment or commit_hidden_root_trace(
        {
            "root_id": root_id,
            "canonical_initial_deck": ["X1", "X2", "Ac", "Kd"],
        }
    )
    return build_behavior_decision_log(
        root_id=root_id,
        root_commitment=commitment,
        seed_namespace=seed_namespace,
        behavior_target_id=target,
        policy_sha256=policy_sha256,
        sampling_contract=SAMPLING_CONTRACT,
        information=information,
        observed_action_key=action_key(observed),
        source_action_probability=Fraction(1, information.legal_action_count),
    )


def _resign(record):
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = canonical_sha256(unsigned)


def _root_for(split):
    for index in range(100_000):
        candidate = f"{split}-root-{index}"
        if root_split(candidate) == split:
            return candidate
    raise AssertionError(f"failed to find root for {split}")


def test_positive_record_is_canonical_content_bound_and_information_safe():
    record = _record(_t2_bb_information())
    verified = verify_behavior_decision_log(record)

    assert record["schema"] == DECISION_LOG_SCHEMA
    assert record["promotion_eligible"] is False
    assert canonical_sha256({k: v for k, v in record.items() if k != "record_sha256"}) == record[
        "record_sha256"
    ]
    assert canonical_json(record) == canonical_json(copy.deepcopy(record))
    assert verified.information.digest() == record["information_digest"]
    assert action_key(verified.observed_action) == record["observed_action_key"]
    assert verified.observed_action.discard == record["observed_discard"]
    assert record["mask27"][record["observed_semantic_index"]] is True
    assert verified.source_action_probability == Fraction(
        record["source_action_probability"]
    )
    serialized_information = canonical_json(record["information"])
    assert "root_id" not in serialized_information
    assert "root_commitment" not in serialized_information
    assert "seed_namespace" not in serialized_information
    assert "observed_action" not in serialized_information
    assert "hidden_full_trace" not in serialized_information


def test_t1_btn_bb_first_cutoff_and_two_physical_jokers_are_supported():
    record = _record(_t1_btn_information(("2h", "X1", "X2")))
    verified = verify_behavior_decision_log(record)

    assert verified.actor == "btn"
    assert verified.turn == 1
    assert record["visible_joker_count"] == 2
    assert set(record["current_draw"]) == {"2h", "X1", "X2"}
    assert record["rules"]["physical_joker_ids"] == ["X1", "X2"]
    assert record["rules"]["action_order"] == "bb_first_every_turn"


def test_all_t1_t2_bb_btn_decision_cells_verify_under_one_root_split():
    root_id = "complete-four-cell-root"
    commitment = commit_hidden_root_trace({"root": root_id, "deck": "committed"})
    records = [
        _record(_t1_bb_information(), root_id=root_id, root_commitment=commitment),
        _record(_t1_btn_information(), root_id=root_id, root_commitment=commitment),
        _record(_t2_bb_information(), root_id=root_id, root_commitment=commitment),
        _record(_t2_btn_information(), root_id=root_id, root_commitment=commitment),
    ]
    verified = [verify_behavior_decision_log(record) for record in records]
    manifest = build_behavior_decision_manifest(records)

    assert {(item.turn, item.actor) for item in verified} == {
        (1, "bb"),
        (1, "btn"),
        (2, "bb"),
        (2, "btn"),
    }
    assert len({item.split for item in verified}) == 1
    assert manifest["root_count"] == 1
    assert manifest["turn_actor_counts"] == {
        "t1": {"bb": 1, "btn": 1},
        "t2": {"bb": 1, "btn": 1},
    }


def test_raw_hash_detects_unresigned_tampering():
    record = _record(_t1_bb_information())
    record["observed_semantic_index"] = (record["observed_semantic_index"] + 1) % 27
    with pytest.raises(ValueError, match="record SHA-256 mismatch"):
        verify_behavior_decision_log(record)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda r: r.__setitem__("information_digest", "0" * 64), "information digest"),
        (lambda r: r.__setitem__("current_draw", ["2h", "3h", "5h"]), "current_draw"),
        (lambda r: r.__setitem__("observed_discard", "X1"), "observed discard"),
        (
            lambda r: r.__setitem__(
                "observed_semantic_index", (r["observed_semantic_index"] + 1) % 27
            ),
            "semantic index",
        ),
        (lambda r: r["legal_action_ids"].pop(), "legal action IDs"),
        (lambda r: r["mask27"].__setitem__(0, not r["mask27"][0]), "mask27"),
        (
            lambda r: r.__setitem__("visible_joker_count", 2),
            "visible Joker count",
        ),
    ],
)
def test_independent_derivations_reject_tampering_even_after_rehash(mutator, message):
    record = _record(_t1_bb_information())
    mutator(record)
    _resign(record)
    with pytest.raises(ValueError, match=message):
        verify_behavior_decision_log(record)


def test_hidden_trace_cannot_be_injected_into_policy_information_even_after_rehash():
    record = _record(_t1_bb_information())
    record["information"]["hidden_full_trace"] = {
        "opponent_discards": ["X1"],
        "undealt_cards": ["X2"],
    }
    _resign(record)
    with pytest.raises(ValueError, match="information keys mismatch"):
        verify_behavior_decision_log(record)


def test_bb_first_public_history_is_rederived_not_trusted():
    record = _record(_t1_btn_information())
    history = record["information"]["public_action_history"]
    history[0], history[1] = history[1], history[0]
    record["information_digest"] = canonical_sha256(record["information"])
    _resign(record)
    with pytest.raises(ValueError, match="BB-first history"):
        verify_behavior_decision_log(record)


def test_source_probability_is_exact_nonzero_and_canonical():
    info = _t1_bb_information()
    with pytest.raises(ValueError, match=r"\(0,1\]"):
        build_behavior_decision_log(
            root_id="zero-probability-root",
            root_commitment="1" * 64,
            seed_namespace="worker-0",
            behavior_target_id="target",
            policy_sha256=POLICY_SHA256,
            sampling_contract="categorical-v1",
            information=info,
            observed_action_key=action_key(_first_action(info)),
            source_action_probability=0,
        )

    record = _record(info)
    record["source_action_probability"] = "2/4"
    _resign(record)
    with pytest.raises(ValueError, match="not canonical"):
        verify_behavior_decision_log(record)


def test_root_split_matches_preregistered_sha256_uint64_contract():
    roots = {name: _root_for(name) for name in ("fit", "dev", "test")}
    for expected_split, root_id in roots.items():
        digest = hashlib.sha256((SPLIT_NAMESPACE + root_id).encode("utf-8")).digest()
        expected_bucket = int.from_bytes(digest[:8], "big") % 10_000
        assert root_split_bucket(root_id) == expected_bucket
        assert root_split(root_id) == expected_split
        assert (
            expected_bucket < 7_000
            if expected_split == "fit"
            else 7_000 <= expected_bucket < 8_500
            if expected_split == "dev"
            else expected_bucket >= 8_500
        )


def test_manifest_is_order_independent_counts_role_joker_cells_and_has_no_overlap():
    fit_root = _root_for("fit")
    dev_root = _root_for("dev")
    test_root = _root_for("test")
    fit_commitment = commit_hidden_root_trace({"root": fit_root})
    records = [
        _record(
            _t1_bb_information(("Ad", "Kd", "Qd")),
            root_id=fit_root,
            root_commitment=fit_commitment,
        ),
        _record(
            _t1_btn_information(("2h", "3h", "X1")),
            root_id=fit_root,
            root_commitment=fit_commitment,
        ),
        _record(_t1_bb_information(("Ad", "Kd", "X1")), root_id=dev_root),
        _record(_t1_btn_information(("2h", "X1", "X2")), root_id=test_root),
    ]
    manifest = build_behavior_decision_manifest(records)
    reversed_manifest = build_behavior_decision_manifest(list(reversed(records)))

    assert manifest == reversed_manifest
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["promotion_eligible"] is False
    assert manifest["calibration_gate_status"] == "not_evaluated"
    assert manifest["record_count"] == 4
    assert manifest["root_count"] == 3
    assert manifest["decision_counts_by_split"] == {"fit": 2, "dev": 1, "test": 1}
    assert manifest["root_counts_by_split"] == {"fit": 1, "dev": 1, "test": 1}
    assert manifest["root_overlap_audit"]["overlap_count"] == 0
    assert manifest["duplicate_decision_count"] == 0
    assert manifest["role_joker_counts"] == {
        "bb": {"joker_0": 1, "joker_1": 1, "joker_2": 0},
        "btn": {"joker_0": 0, "joker_1": 1, "joker_2": 1},
    }
    assert manifest["information_contract"]["hidden_full_trace_in_policy_input"] is False
    assert verify_behavior_decision_dataset(records, manifest) == manifest


def test_duplicate_decision_is_rejected_fail_closed():
    record = _record(_t1_bb_information())
    with pytest.raises(ValueError, match="duplicate behavior decision"):
        build_behavior_decision_manifest([record, copy.deepcopy(record)])


def test_same_root_must_keep_one_commitment_seed_and_split():
    root_id = "shared-root-contract"
    first = _record(
        _t1_bb_information(), root_id=root_id, root_commitment="1" * 64
    )
    second = _record(
        _t1_btn_information(), root_id=root_id, root_commitment="2" * 64
    )
    assert first["split"] == second["split"] == root_split(root_id)
    with pytest.raises(ValueError, match="inconsistent commitment/seed/split"):
        build_behavior_decision_manifest([first, second])


def test_manifest_cannot_mix_behavior_targets_or_sampling_contracts():
    first = _record(_t1_bb_information(), root_id="target-a")
    second = _record(
        _t1_btn_information(), root_id="target-b", target="different-population"
    )
    with pytest.raises(ValueError, match="cannot mix behavior targets"):
        build_behavior_decision_manifest([first, second])

    second = _record(_t1_btn_information(), root_id="sampling-b")
    second["sampling_contract"] = "different-sampler"
    _resign(second)
    with pytest.raises(ValueError, match="cannot mix sampling contracts"):
        build_behavior_decision_manifest([first, second])


def test_empty_dataset_never_builds_a_passing_manifest():
    with pytest.raises(ValueError, match="dataset is empty"):
        build_behavior_decision_manifest([])
    with pytest.raises(ValueError, match="dataset is empty"):
        verify_behavior_decision_dataset([])


def test_manifest_tampering_fails_with_or_without_recomputed_hash():
    records = [_record(_t1_bb_information())]
    manifest = build_behavior_decision_manifest(records)
    tampered = copy.deepcopy(manifest)
    tampered["role_joker_counts"]["bb"]["joker_0"] += 1
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        verify_behavior_decision_dataset(records, tampered)

    tampered["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="does not match raw records"):
        verify_behavior_decision_dataset(records, tampered)


def test_sampling_contract_rejects_binary_float_metadata():
    info = _t1_bb_information()
    with pytest.raises(TypeError, match="binary floats"):
        build_behavior_decision_log(
            root_id="float-sampler-root",
            root_commitment="1" * 64,
            seed_namespace="worker-0",
            behavior_target_id="target",
            policy_sha256=POLICY_SHA256,
            sampling_contract={"temperature": 0.7},
            information=info,
            observed_action_key=action_key(_first_action(info)),
            source_action_probability=Fraction(1, info.legal_action_count),
        )
