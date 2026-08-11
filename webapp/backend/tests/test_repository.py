from __future__ import annotations

from dataclasses import replace
import json

from ofc_webapp.domain import (
    ActionSubmission,
    FinalScore,
    HandStatus,
    apply_action,
    create_match,
    legal_actions,
    settle_hand,
    start_hand,
)
from ofc_webapp.ports import AIMetadata
from ofc_webapp.repository import DecisionRecord, SQLiteRepository


SHA_A = "a" * 64
SHA_B = "b" * 64


def _meta() -> AIMetadata:
    return AIMetadata(
        evaluator="fake-evaluator",
        weights_sha=(SHA_A,),
        assembly_sha=SHA_B,
        scores_topk=(
            {"score": 3.0, "action": "a"},
            {"score": 2.0, "action": "b"},
            {"score": 1.0, "action": "c"},
        ),
    )


def test_repository_round_trip_and_replay_complete_jsonl(tmp_path):
    repository = SQLiteRepository(tmp_path / "ofc.sqlite3")
    repository.initialize()
    match = replace(
        create_match(
            seed=909,
            match_id="db-match",
            assembly_sha=SHA_B,
            app_version="test",
            created_at="2026-07-29T00:00:00+00:00",
        ),
        first_hand_first="human",
    )
    active, hand = start_hand(
        match,
        hand_id="db-hand",
        started_at="2026-07-29T00:00:01+00:00",
    )
    repository.save_transition(match=active, hand=hand)

    while hand.status == HandStatus.PLAYING:
        turn = hand.current_turn
        action = legal_actions(hand)[0]
        transition = apply_action(
            hand,
            ActionSubmission(action.placements, action.discards),
            actor=turn.actor,
        )
        record = DecisionRecord.from_applied(
            transition.decision,
            hand_id=hand.id,
            decision_id=f"decision-{transition.decision.sequence_no}",
            think_ms=12,
            created_at=(
                f"2026-07-29T00:00:{transition.decision.sequence_no + 2:02d}"
                "+00:00"
            ),
            ai_meta=_meta() if turn.actor == "ai" else None,
        )
        hand = transition.hand
        repository.save_transition(
            match=active,
            hand=hand,
            decision=record,
        )

    settled, hand = settle_hand(
        active,
        hand,
        FinalScore(
            hu_score=-5,
            breakdown={
                "row_wins": [-1, 0, -1],
                "scoop": False,
                "royalties": [0, 3],
                "fouls": [False, False],
            },
        ),
        ended_at="2026-07-29T00:01:00+00:00",
    )
    repository.save_transition(match=settled, hand=hand)

    assert repository.get_match(match.id) == settled
    assert repository.get_hand(hand.id) == hand
    assert repository.count_decisions(hand.id) == 10
    assert len(repository.list_decisions(hand.id)) == 10
    assert len(repository.list_hands(match.id)) == 1

    lines = [
        json.loads(line)
        for line in repository.export_jsonl(match.id).splitlines()
    ]
    decision_lines = [
        line for line in lines if line["record_type"] == "decision"
    ]
    summaries = [
        line for line in lines if line["record_type"] == "hand_summary"
    ]
    assert len(decision_lines) == 10
    assert len(summaries) == 1
    assert summaries[0]["match_seed"] == 909
    assert summaries[0]["deck_order"] == list(hand.deck_order)
    assert summaries[0]["result"]["raw_score"] == -5
    assert sum(line["actor"] == "ai" for line in decision_lines) == 5
    assert all(
        line["ai_meta"] is not None
        for line in decision_lines
        if line["actor"] == "ai"
    )


def test_ai_record_requires_complete_top_three_metadata():
    try:
        AIMetadata(
            evaluator="broken",
            weights_sha=(SHA_A,),
            assembly_sha=SHA_B,
            scores_topk=({"score": 1},),
        )
    except ValueError as exc:
        assert "top three" in str(exc)
    else:
        raise AssertionError("incomplete top-k metadata was accepted")

