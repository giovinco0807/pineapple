from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import random
import subprocess

import pytest

from ofc_regular.hu_m3_rust import HU_M3_REQUEST_SCHEMA, evaluate_request

from ofc_webapp.domain import (
    ActionSubmission,
    HandStatus,
    MatchStatus,
    apply_action,
    build_observation,
    continue_match,
    create_match,
    legal_actions,
    settle_hand,
    start_hand,
)
from ofc_webapp.repository import DecisionRecord, SQLiteRepository
from ofc_webapp.runtime import (
    Assembly,
    NativeFinalScoreAdapter,
    RegularAIRuntime,
    _parse_fl_solution,
    evaluation_scoring_context,
    settlement_scoring_context,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("OFC_RUN_NATIVE_ACCEPTANCE") != "1",
    reason="set OFC_RUN_NATIVE_ACCEPTANCE=1 after building isolated native artifacts",
)


def _board_payload(board):
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }


def _human_fl_action(cards: tuple[str, ...]) -> ActionSubmission:
    # Every full 3/5/5 placement is legal in FL, including a foul.
    return ActionSubmission(
        tuple(
            [(card, "top") for card in cards[:3]]
            + [(card, "middle") for card in cards[3:8]]
            + [(card, "bottom") for card in cards[8:13]]
        ),
        (cards[13],),
    )


def _play_match(
    tmp_path: Path,
    *,
    seed: int,
    minimum_hands: int,
) -> tuple[int, bool, int, int]:
    assembly = Assembly.load()
    ai = RegularAIRuntime(assembly)
    scorer = NativeFinalScoreAdapter(assembly)
    repository = SQLiteRepository(tmp_path / f"golden-{seed}.sqlite3")
    repository.initialize()
    match = create_match(
        seed=seed,
        assembly_sha=assembly.sha256,
        app_version="native-golden",
    )
    repository.save_match(match)
    rng = random.Random(seed ^ 0xA11CE)
    saw_fantasyland = False
    total_decisions = 0

    while match.hand_count < minimum_hands and match.status != MatchStatus.COMPLETED:
        if match.status == MatchStatus.AWAITING_CONTINUE:
            match = continue_match(match, should_continue=True)
            repository.save_match(match)
        assert match.status == MatchStatus.READY
        match, hand = start_hand(match)
        saw_fantasyland |= any(hand.fantasyland)
        repository.save_transition(match=match, hand=hand)

        while hand.status == HandStatus.PLAYING:
            assert hand.current_turn is not None
            if hand.current_turn.actor == "human":
                action = (
                    _human_fl_action(hand.current_turn.dealt_cards)
                    if hand.current_turn.street == "FL"
                    else ActionSubmission(
                        tuple(
                            (
                                chosen := rng.choice(legal_actions(hand))
                            ).placements
                        ),
                        tuple(chosen.discards),
                    )
                )
                transition = apply_action(hand, action, actor="human")
                record = DecisionRecord.from_applied(
                    transition.decision,
                    hand_id=hand.id,
                    think_ms=0,
                )
            else:
                decision = ai.decide(
                    build_observation(
                        hand,
                        actor="ai",
                        scoring=evaluation_scoring_context(),
                    )
                )
                transition = apply_action(hand, decision.action, actor="ai")
                assert len(decision.meta.scores_topk) == 3
                assert decision.meta.assembly_sha == assembly.sha256
                record = DecisionRecord.from_applied(
                    transition.decision,
                    hand_id=hand.id,
                    think_ms=decision.think_ms,
                    ai_meta=decision.meta,
                )
            hand = transition.hand
            total_decisions += 1
            repository.save_transition(
                match=match, hand=hand, decision=record
            )

        first_player = "human" if hand.positions[0] == "first" else "ai"
        second_player = "ai" if first_player == "human" else "human"
        first_board = hand.board_for(first_player)
        second_board = hand.board_for(second_player)
        scoring = settlement_scoring_context()
        normalized = scorer.score_final(
            first_board=first_board,
            second_board=second_board,
            scoring=scoring,
            first_in_fantasyland=hand.in_fantasyland(first_player),
            second_in_fantasyland=hand.in_fantasyland(second_player),
        )
        direct = evaluate_request(
            {
                "schema": HU_M3_REQUEST_SCHEMA,
                "kind": "score_final",
                "hero_board": _board_payload(first_board),
                "opponent_board": _board_payload(second_board),
                "scoring": scoring.to_dict(),
            },
            library=scorer.library,
        )
        assert normalized.hu_score == round(float(direct["hu_score"]))
        assert normalized.breakdown["source"] == "canonical_rust_hu_m3_engine"
        assert (
            normalized.breakdown["point_components"]["total"]
            == normalized.hu_score
        )
        before = match.stacks
        match, hand = settle_hand(match, hand, normalized)
        assert sum(match.stacks) == 400
        assert abs(match.stacks[0] - before[0]) <= before[1]
        assert abs(match.stacks[1] - before[1]) <= before[0]
        repository.save_transition(match=match, hand=hand)

    exported = [
        json.loads(line)
        for line in repository.export_jsonl(match.id).splitlines()
    ]
    decision_lines = [
        row for row in exported if row["record_type"] == "decision"
    ]
    summary_lines = [
        row for row in exported if row["record_type"] == "hand_summary"
    ]
    assert len(decision_lines) == total_decisions
    assert len(summary_lines) == match.hand_count
    return match.hand_count, saw_fantasyland, total_decisions, len(exported)


def test_fixed_seed_match_uses_native_score_and_complete_records(tmp_path):
    hand_count, saw_fl, decisions, export_lines = _play_match(
        tmp_path,
        seed=1717,
        minimum_hands=5,
    )
    assert hand_count >= 5
    assert saw_fl
    assert decisions > hand_count
    assert export_lines == decisions + hand_count


def test_ai_fantasyland_action_is_the_solver_output():
    assembly = Assembly.load()
    ai = RegularAIRuntime(assembly)
    match = replace(
        create_match(
            seed=9042,
            assembly_sha=assembly.sha256,
            app_version="native-fl",
        ),
        pending_fantasyland=(False, True),
    )
    _active_match, hand = start_hand(match)
    while hand.current_turn is not None and hand.current_turn.actor == "human":
        chosen = legal_actions(hand)[0]
        hand = apply_action(
            hand,
            ActionSubmission(
                tuple(chosen.placements), tuple(chosen.discards)
            ),
            actor="human",
        ).hand
    assert hand.current_turn is not None
    assert hand.current_turn.actor == "ai"
    assert hand.current_turn.street == "FL"
    observation = build_observation(
        hand, actor="ai", scoring=evaluation_scoring_context()
    )
    result = ai.decide(observation)

    completed = subprocess.run(
        [
            str(ai.fl_solver_path),
            "--solve",
            ",".join(hand.current_turn.dealt_cards),
            "--stay-bonus",
            str(dict(observation.scoring.fl_ev)[14]),
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
        timeout=30,
    )
    expected = _parse_fl_solution(completed.stdout)
    expected_placements = tuple(
        (card, row)
        for row in ("top", "middle", "bottom")
        for card in expected[row]
    )
    assert result.action.placements == expected_placements
    assert result.action.discards == (expected["discard"][0],)
    assert result.meta.evaluator == "regular_fl_solver+canonical_rust_top3"
    assert len(result.meta.weights_sha) == 2
    assert len(result.meta.scores_topk) == 3
    assert all(
        isinstance(candidate["score"], (int, float))
        for candidate in result.meta.scores_topk
    )
    rank1 = ActionSubmission.from_parts(
        result.meta.scores_topk[0]["placements"],
        result.meta.scores_topk[0]["discards"],
    )
    assert rank1 == result.action
    exact_expected_score = expected["total_royalty"] + (
        dict(observation.scoring.fl_ev)[14] if expected["can_stay"] else 0.0
    )
    assert result.meta.scores_topk[0]["score"] == pytest.approx(
        exact_expected_score, abs=1e-12
    )
    assert result.meta.scores_topk[0]["can_stay"] is expected["can_stay"]
