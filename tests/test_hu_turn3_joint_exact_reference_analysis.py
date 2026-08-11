import json

import pytest

from ofc_regular.analyze_hu_turn3_joint_exact_references import (
    analyze_file,
    analyze_sample,
    write_markdown,
)
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_turn3_joint_exact_teacher import (
    JointExactConfig,
    evaluate_state_record,
)
from ofc_regular.state import Board


def _constrained_board(cards):
    cards = tuple(cards)
    return Board.from_rows(
        top=cards[:3],
        middle=cards[3:8],
        bottom=cards[8:],
    )


def _state():
    observation = ActorObservation(
        hero_board=_constrained_board(ALL_CARDS[:9]),
        opponent_public_board=_constrained_board(ALL_CARDS[9:18]),
        dealt_cards=ALL_CARDS[18:21],
        hero_private_discards=ALL_CARDS[21:23],
        seat="first",
        street="T3",
        to_act_order="first",
    )
    return {
        "state_id": "s-ref-v2",
        "source": "unit_test_fresh_state",
        "hand_seed": 101,
        "hand_index": 7,
        "visibility_model": "actor_observation_v1",
        "discard_visibility": "own_private_only",
        "policy_observation": observation.to_dict(),
        "selection": {
            "baseline_index": 0,
            "hu_index": 1,
            "compare_hu_index": 2,
            "disagreement": True,
            "compare_hu_disagreement": True,
            "predicted_margin_vs_baseline": 8.0,
            "compare_predicted_margin_vs_baseline": 3.0,
        },
    }


@pytest.fixture(scope="module")
def teacher_sample():
    return evaluate_state_record(
        _state(),
        sample_id=0,
        config=JointExactConfig(
            candidate_samples=1,
            evaluation_samples=1,
            downstream_t3_samples=1,
            downstream_t4_samples=1,
            seed=20,
            run_id="reference-analysis-v2",
        ),
    )


def test_analyze_sample_reports_v2_locked_evaluation_selection_delta(teacher_sample):
    row = analyze_sample(teacher_sample, delta_threshold=0.25)

    assert row is not None
    by_index = {
        action["original_index"]: action for action in teacher_sample["actions"]
    }
    expected_delta = by_index[1]["score"] - by_index[0]["score"]
    assert row["baseline_index"] == 0
    assert row["hu_index"] == 1
    assert row["compare_hu_index"] == 2
    assert row["source"] == "unit_test_fresh_state"
    assert row["state_id"] == "s-ref-v2"
    assert row["hand_index"] == 7
    assert row["delta_hu_vs_baseline"] == pytest.approx(expected_delta)
    assert row["best_index"] == teacher_sample["actions"][0]["original_index"]
    assert row["best_is_hu"] == int(row["best_index"] == 1)
    assert row["hu_materially_positive"] == int(expected_delta >= 0.25)
    assert row["hu_materially_negative"] == int(expected_delta <= -0.25)


def test_analyze_file_reads_v2_teacher_jsonl(tmp_path, teacher_sample):
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(
        json.dumps(teacher_sample, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    analysis = analyze_file(teacher, delta_threshold=0.25)

    assert analysis["summary"]["samples"] == 1
    assert analysis["summary"]["reason"] == "joint_exact_selection_delta_summary"
    assert analysis["rows"][0]["state_id"] == "s-ref-v2"
    assert analysis["hu_loss_audit_rows"][0]["source"] == (
        "unit_test_fresh_state"
    )
    assert analysis["hu_loss_audit_rows"][0]["baseline_action"][
        "original_index"
    ] == 0


def test_write_markdown_contains_decision(tmp_path):
    out = tmp_path / "summary.md"

    write_markdown(
        out,
        {
            "samples": 2,
            "delta_hu_vs_baseline_mean": 1.5,
            "hu_materially_positive": 1,
            "hu_materially_negative": 0,
            "hu_near_tie": 1,
            "baseline_regret_mean": 2.0,
            "hu_regret_mean": 0.5,
            "decision": "Needs larger validation",
        },
    )

    text = out.read_text(encoding="utf-8")
    assert "Needs larger validation" in text
    assert "production evidence" in text
