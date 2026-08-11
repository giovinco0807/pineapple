from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ofc_regular.action_space import generate_actions, generate_turn_actions
from ofc_regular.state import Board

from ofc_webapp.api import create_app
from ofc_webapp.domain import ActionSubmission, FinalScore, create_match
from ofc_webapp.ports import AIDecision, AIMetadata
from ofc_webapp.repository import SQLiteRepository


AUTH = {"Authorization": "Bearer test-token"}
ASSEMBLY_SHA = "a" * 64
WEIGHT_SHA = "b" * 64
CARD_RE = re.compile(r"^[2-9TJQKA][hdcs]$")


class FakeAssembly:
    sha256 = ASSEMBLY_SHA

    def public_meta(self) -> dict[str, Any]:
        return {
            "assembly_sha256": self.sha256,
            "dispatch": {
                "T0": {"evaluator": "fake"},
                "FL": {"evaluator": "fake"},
            },
            "weights": {
                "fake": {"sha256": WEIGHT_SHA},
            },
            "rules": {
                "joker": False,
                "fantasyland_cards": 14,
            },
            "artifacts": {},
        }


@dataclass
class FakeAI:
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def decide(self, observation) -> AIDecision:
        if observation.street == "FL":
            cards = observation.dealt_cards
            action = ActionSubmission(
                placements=tuple(
                    [(card, "top") for card in cards[:3]]
                    + [(card, "middle") for card in cards[3:8]]
                    + [(card, "bottom") for card in cards[8:13]]
                ),
                discards=(cards[13],),
            )
            candidates = (
                {
                    "rank": 1,
                    "score": 10.0,
                    "action_key": "private-fl-action",
                    "placements": [
                        [card, row] for card, row in action.placements
                    ],
                    "discards": list(action.discards),
                },
                {
                    "rank": 2,
                    "score": None,
                    "available": False,
                    "reason": "fake_best_only",
                },
                {
                    "rank": 3,
                    "score": None,
                    "available": False,
                    "reason": "fake_best_only",
                },
            )
        else:
            actions = (
                generate_actions(
                    observation.hero_board, observation.dealt_cards
                )
                if observation.street == "T0"
                else generate_turn_actions(
                    observation.hero_board, observation.dealt_cards
                )
            )
            chosen = actions[0]
            action = ActionSubmission(chosen.placements, chosen.discards)
            candidates = tuple(
                {
                    "rank": rank,
                    "score": float(4 - rank),
                    "action_key": f"private:{candidate.placements!r}",
                    "placements": [
                        [card, row]
                        for card, row in candidate.placements
                    ],
                    "discards": list(candidate.discards),
                }
                for rank, candidate in enumerate(actions[:3], 1)
            )
        result = AIDecision(
            action=action,
            think_ms=7,
            meta=AIMetadata(
                evaluator=f"fake:{observation.street}",
                weights_sha=(WEIGHT_SHA,),
                assembly_sha=ASSEMBLY_SHA,
                scores_topk=candidates,
            ),
        )
        self.decisions.append(
            {
                "street": observation.street,
                "dealt_cards": tuple(observation.dealt_cards),
                "action": action,
            }
        )
        return result


@dataclass
class FakeScorer:
    outcomes: list[tuple[int, bool, bool]] = field(default_factory=list)
    breakdowns: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def score_final(
        self,
        *,
        first_board,
        second_board,
        scoring,
        first_in_fantasyland,
        second_in_fantasyland,
    ) -> FinalScore:
        assert first_board.is_complete()
        assert second_board.is_complete()
        self.calls.append(
            {
                "scoring": scoring,
                "first_in_fantasyland": first_in_fantasyland,
                "second_in_fantasyland": second_in_fantasyland,
            }
        )
        score, first_fl, second_fl = (
            self.outcomes.pop(0)
            if self.outcomes
            else (6, False, False)
        )
        breakdown = (
            self.breakdowns.pop(0)
            if self.breakdowns
            else {
                "source": "fake-score-final",
                "fouls": {"first": False, "second": False},
                "royalties": {"first": 0, "second": 0},
                "row_results": {
                    "top": "tie",
                    "middle": "tie",
                    "bottom": "tie",
                },
                "scoop": {"first": False, "second": False},
                "point_components": {
                    "perspective": "first",
                    "foul_base": 0,
                    "line_total": score,
                    "scoop_bonus": 0,
                    "royalty_delta": 0,
                    "total": score,
                },
            }
        )
        return FinalScore(
            hu_score=score,
            breakdown=breakdown,
            first_next_fantasyland=first_fl,
            second_next_fantasyland=second_fl,
        )


@dataclass
class FakeBackup:
    enabled: bool = False
    restored: list[Path] = field(default_factory=list)
    backed_up: list[Path] = field(default_factory=list)
    error: Exception | None = None

    def restore(self, destination: Path) -> bool:
        self.restored.append(destination)
        return False

    def backup(self, source: Path) -> None:
        assert source.is_file()
        self.backed_up.append(source)
        if self.error is not None:
            raise self.error


@pytest.fixture
def api_env(tmp_path):
    repository = SQLiteRepository(tmp_path / "api.sqlite3")
    ai = FakeAI()
    scorer = FakeScorer()
    backup = FakeBackup()
    assembly = FakeAssembly()
    app = create_app(
        assembly=assembly,  # type: ignore[arg-type]
        repository=repository,
        ai_factory=lambda _assembly: ai,
        scorer_factory=lambda _assembly: scorer,
        backup=backup,
        shared_token="test-token",
        static_dir=tmp_path / "missing-static",
    )
    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client,
            repository=repository,
            ai=ai,
            scorer=scorer,
            backup=backup,
        )


def _create_and_start(api_env, *, seed: int = 1234):
    response = api_env.client.post(
        "/api/match", headers=AUTH, json={"seed": seed}
    )
    assert response.status_code == 200, response.text
    match = response.json()
    response = api_env.client.post(
        f"/api/match/{match['match_id']}/hand",
        headers=AUTH,
    )
    assert response.status_code == 200, response.text
    return match, response.json()


def _normal_action_payload(hand_payload):
    board_payload = hand_payload["boards"]["human"]
    board = Board.from_rows(
        top=board_payload["top"],
        middle=board_payload["middle"],
        bottom=board_payload["bottom"],
    )
    dealt = tuple(hand_payload["dealt_cards"])
    actions = (
        generate_actions(board, dealt)
        if hand_payload["street"] == "T0"
        else generate_turn_actions(board, dealt)
    )
    action = actions[0]
    return {
        "placements": [list(item) for item in action.placements],
        "discards": list(action.discards),
    }


def _fl_action_payload(hand_payload):
    cards = hand_payload["dealt_cards"]
    return {
        "placements": (
            [[card, "top"] for card in cards[:3]]
            + [[card, "middle"] for card in cards[3:8]]
            + [[card, "bottom"] for card in cards[8:13]]
        ),
        "discards": [cards[13]],
    }


def _play_hand(api_env, hand_payload):
    payload = hand_payload
    for _ in range(12):
        if payload["status"] == "complete":
            return payload
        assert payload["status"] == "playing"
        assert payload["to_act"] == "human"
        assert payload["action_required"] in {"normal", "fl"}
        is_fl = payload["action_required"] == "fl"
        action = (
            _fl_action_payload(payload)
            if is_fl
            else _normal_action_payload(payload)
        )
        suffix = "fl-placement" if is_fl else "action"
        response = api_env.client.post(
            f"/api/hand/{payload['hand_id']}/{suffix}",
            headers=AUTH,
            json=action,
        )
        assert response.status_code == 200, response.text
        payload = response.json()
    raise AssertionError("hand did not complete within the expected turns")


def _card_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if CARD_RE.fullmatch(value) else set()
    if isinstance(value, dict):
        return {
            card
            for nested in value.values()
            for card in _card_values(nested)
        }
    if isinstance(value, list):
        return {
            card for nested in value for card in _card_values(nested)
        }
    return set()


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(
            _contains_key(nested, target) for nested in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(nested, target) for nested in value)
    return False


def test_bearer_authentication_and_public_health(api_env):
    for path in ("/healthz", "/api/healthz"):
        health = api_env.client.get(path)
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

    missing = api_env.client.get("/api/meta")
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"

    wrong = api_env.client.get(
        "/api/meta", headers={"Authorization": "Bearer wrong"}
    )
    assert wrong.status_code == 401

    authorized = api_env.client.get("/api/meta", headers=AUTH)
    assert authorized.status_code == 200
    payload = authorized.json()
    assert payload["assembly_sha"] == ASSEMBLY_SHA
    assert payload["rules"]["fantasyland_cards"] == 14
    assert payload["rules"]["settlement_fl_ev"] == {"14": 0.0}
    assert payload["rules"]["scoring_authority"] == "rust_score_final"


def test_missing_shared_token_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("SHARED_TOKEN", raising=False)
    app = create_app(
        assembly=FakeAssembly(),  # type: ignore[arg-type]
        repository=SQLiteRepository(tmp_path / "missing-token.sqlite3"),
        ai_factory=lambda _assembly: FakeAI(),
        scorer_factory=lambda _assembly: FakeScorer(),
        backup=FakeBackup(),
        shared_token=None,
        static_dir=tmp_path / "missing-static",
    )
    with pytest.raises(RuntimeError, match="SHARED_TOKEN"):
        with TestClient(app):
            pass


def test_illegal_overflow_duplicate_and_discard_mismatch_return_400(api_env):
    _match, hand = _create_and_start(api_env, seed=2001)
    assert hand["street"] == "T0"
    assert hand["to_act"] == "human"
    cards = hand["dealt_cards"]
    initial = api_env.client.get(
        f"/api/hand/{hand['hand_id']}", headers=AUTH
    ).json()
    invalid_actions = [
        {
            "placements": [[card, "top"] for card in cards],
            "discards": [],
        },
        {
            "placements": [
                [cards[0], "top"],
                [cards[0], "middle"],
                [cards[2], "middle"],
                [cards[3], "bottom"],
                [cards[4], "bottom"],
            ],
            "discards": [],
        },
        {
            **_normal_action_payload(hand),
            "discards": [cards[0]],
        },
    ]

    for invalid in invalid_actions:
        response = api_env.client.post(
            f"/api/hand/{hand['hand_id']}/action",
            headers=AUTH,
            json=invalid,
        )
        assert response.status_code == 400
        assert "detail" in response.json()
        unchanged = api_env.client.get(
            f"/api/hand/{hand['hand_id']}", headers=AUTH
        ).json()
        assert unchanged == initial


def test_complete_normal_hand_is_safe_and_export_counts_match_decisions(api_env):
    match, hand = _create_and_start(api_env, seed=3001)
    complete = _play_hand(api_env, hand)

    assert complete["status"] == "complete"
    assert complete["result"]["breakdown"]["source"] == "fake-score-final"
    assert len(complete["replay"]) == 10
    assert not _contains_key(complete, "deck_order")
    assert not _contains_key(complete, "turns")

    hidden_ai_discards = {
        card
        for decision in api_env.ai.decisions
        for card in decision["action"].discards
    }
    assert len(hidden_ai_discards) == 4
    assert hidden_ai_discards.isdisjoint(_card_values(complete))
    for step in complete["replay"]:
        if step["actor"] == "ai":
            assert step["dealt_cards"] == []
            assert step["discards"] == []
            assert step["ai_meta"] is not None
            assert len(step["ai_meta"]["scores_topk"]) == 3
            assert all(
                "placements" not in candidate
                and "discards" not in candidate
                and "action_key" not in candidate
                for candidate in step["ai_meta"]["scores_topk"]
            )

    assert len(api_env.scorer.calls) == 1
    scoring = api_env.scorer.calls[0]["scoring"]
    assert dict(scoring.fl_ev) == {14: 0.0}
    assert scoring.fantasyland_cards == 14
    assert len(api_env.backup.backed_up) == 1

    match_response = api_env.client.get(
        f"/api/match/{match['match_id']}", headers=AUTH
    )
    assert match_response.status_code == 200
    match_payload = match_response.json()
    assert match_payload["hand_count"] == 1
    assert match_payload["can_continue"] is True
    assert sum(match_payload["stacks"].values()) == 400

    finish = api_env.client.post(
        f"/api/match/{match['match_id']}/continue",
        headers=AUTH,
        json={"continue": False},
    )
    assert finish.status_code == 200
    assert finish.json()["status"] == "completed"
    assert len(api_env.backup.backed_up) == 2

    exported = api_env.client.get(
        f"/api/match/{match['match_id']}/export", headers=AUTH
    )
    assert exported.status_code == 200
    assert exported.headers["content-type"].startswith(
        "application/x-ndjson"
    )
    assert "attachment;" in exported.headers["content-disposition"]
    lines = [json.loads(line) for line in exported.text.splitlines()]
    decisions = [
        line for line in lines if line["record_type"] == "decision"
    ]
    summaries = [
        line for line in lines if line["record_type"] == "hand_summary"
    ]
    assert len(decisions) == 10
    assert len(summaries) == 1
    assert api_env.repository.count_decisions(complete["hand_id"]) == 10
    assert len(summaries[0]["deck_order"]) == 52
    assert any(
        row["actor"] == "ai"
        and row["street"] != "T0"
        and len(row["discards"]) == 1
        for row in decisions
    )


def test_continue_choice_backs_up_database(api_env):
    match, hand = _create_and_start(api_env, seed=3002)
    _play_hand(api_env, hand)
    assert len(api_env.backup.backed_up) == 1

    response = api_env.client.post(
        f"/api/match/{match['match_id']}/continue",
        headers=AUTH,
        json={"continue": True},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert len(api_env.backup.backed_up) == 2


def test_continue_backup_failure_is_recorded_without_losing_response(api_env):
    match, hand = _create_and_start(api_env, seed=3003)
    _play_hand(api_env, hand)
    api_env.backup.error = RuntimeError("simulated GCS outage")

    response = api_env.client.post(
        f"/api/match/{match['match_id']}/continue",
        headers=AUTH,
        json={"continue": False},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"

    meta = api_env.client.get("/api/meta", headers=AUTH)
    assert meta.status_code == 200
    assert meta.json()["storage"]["last_backup_error"] == (
        "simulated GCS outage"
    )


def test_foul_point_components_are_signed_for_human_perspective(api_env):
    seed = next(
        candidate
        for candidate in range(10_000, 10_100)
        if create_match(seed=candidate).first_hand_first == "ai"
    )
    api_env.scorer.outcomes[:] = [(-10, False, False)]
    api_env.scorer.breakdowns[:] = [
        {
            "source": "canonical_rust_hu_m3_engine",
            "fouls": {"first": True, "second": False},
            "row_results": {
                "top": "not_scored_foul",
                "middle": "not_scored_foul",
                "bottom": "not_scored_foul",
            },
            "scoop": {"first": False, "second": False},
            "royalties": {
                "first": {"top": 0, "middle": 0, "bottom": 0, "total": 0},
                "second": {"top": 4, "middle": 0, "bottom": 0, "total": 4},
            },
            "point_components": {
                "perspective": "first",
                "foul_base": -6,
                "line_total": 0,
                "scoop_bonus": 0,
                "royalty_delta": -4,
                "total": -10,
            },
        }
    ]
    _match, hand = _create_and_start(api_env, seed=seed)
    assert hand["positions"]["human"] == "second"
    complete = _play_hand(api_env, hand)
    result = complete["result"]

    assert result["human_raw_score"] == 10
    assert result["point_components"] == {
        "perspective": "human",
        "foul_base": 6,
        "line_total": 0,
        "scoop_bonus": 0,
        "royalty_delta": 4,
        "total": 10,
    }
    assert set(result["row_wins"].values()) == {"not_scored_foul"}
    assert result["scoop"] is None


def test_fixed_14_card_fl_uses_dedicated_endpoint_and_auto_continues(api_env):
    api_env.scorer.outcomes[:] = [
        (2, True, True),
        (0, False, False),
    ]
    match, hand = _create_and_start(api_env, seed=4001)
    first_complete = _play_hand(api_env, hand)
    assert first_complete["status"] == "complete"

    match_state = api_env.client.get(
        f"/api/match/{match['match_id']}", headers=AUTH
    ).json()
    assert match_state["status"] == "ready"
    assert match_state["can_continue"] is False
    forbidden_continue = api_env.client.post(
        f"/api/match/{match['match_id']}/continue",
        headers=AUTH,
        json={"continue": True},
    )
    assert forbidden_continue.status_code == 400

    response = api_env.client.post(
        f"/api/match/{match['match_id']}/hand", headers=AUTH
    )
    assert response.status_code == 200
    fl_hand = response.json()
    assert fl_hand["action_required"] == "fl"
    assert fl_hand["to_act"] == "human"
    assert len(fl_hand["dealt_cards"]) == 14

    wrong_endpoint = api_env.client.post(
        f"/api/hand/{fl_hand['hand_id']}/action",
        headers=AUTH,
        json=_fl_action_payload(fl_hand),
    )
    assert wrong_endpoint.status_code == 400
    invalid_fl = api_env.client.post(
        f"/api/hand/{fl_hand['hand_id']}/fl-placement",
        headers=AUTH,
        json={"placements": [], "discards": []},
    )
    assert invalid_fl.status_code == 400

    complete = _play_hand(api_env, fl_hand)
    assert complete["status"] == "complete"
    assert len(
        api_env.repository.list_decisions(complete["hand_id"])
    ) == 2
    assert len(api_env.scorer.calls) == 2
    assert api_env.scorer.calls[1]["first_in_fantasyland"] is True
    assert api_env.scorer.calls[1]["second_in_fantasyland"] is True
    for step in complete["replay"]:
        if step["actor"] == "ai":
            assert step["dealt_cards"] == []
            assert step["discards"] == []

    exported = api_env.client.get(
        f"/api/match/{match['match_id']}/export", headers=AUTH
    )
    lines = [json.loads(line) for line in exported.text.splitlines()]
    assert sum(line["record_type"] == "decision" for line in lines) == 12
    assert sum(line["record_type"] == "hand_summary" for line in lines) == 2
