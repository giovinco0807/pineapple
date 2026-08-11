from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ofc_regular import evaluate_hu_turn2_stage8b_topk_mc_rerank as topk_eval
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.decision_trace import attach_replay_truth, capture_decision_log_positions
from ofc_regular.hu_infoset import ReplayTruth
from ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank import (
    HuTurn2Stage8bTopKMcRerankPolicy,
    TopKMcRerankConfig,
    _add_prefixed_t3_summary,
    _add_t3_summary_delta,
    _paired_delta_summary_for_action,
    _trace_t3_decision_summary,
    load_topk_parts,
    parse_topk_configs,
    risk_veto_candidate_metrics,
    risk_veto_count,
)
from ofc_regular.policy import action_to_json, board_to_json
from ofc_regular.state import Board


class BaselineTurn2Model:
    def predict_sample(self, sample):
        values = np.zeros(len(sample["actions"]), dtype=np.float64)
        values[min(1, len(values) - 1)] = 1.0
        return values


class Stage8bTopKModel:
    def predict_sample(self, sample):
        predictions = np.zeros((len(sample["actions"]), 5), dtype=np.float64)
        predictions[:, 4] = 5.0
        predictions[0, 0] = 10.0
        predictions[0, 1] = 2.0
        predictions[1:, 0] = 1.0
        predictions[1:, 1] = 0.0
        return predictions


class FakeLocalEvRiskScorer:
    path = "fake_local_ev_risk.pt"

    def __init__(self, probability):
        self.probability = probability
        self.rows = []

    def predict_probability(self, row):
        self.rows.append(dict(row))
        return self.probability


class FakeStage8cFireSelectorScorer(FakeLocalEvRiskScorer):
    path = "fake_stage8c_fire_selector.pt"


class FakeBatchStage8cFireSelectorScorer(FakeStage8cFireSelectorScorer):
    def __init__(self, probabilities):
        super().__init__(probability=0.0)
        self.probabilities = list(probabilities)
        self.batch_calls = 0
        self.scalar_calls = 0

    def predict_probability(self, row):
        self.scalar_calls += 1
        return super().predict_probability(row)

    def predict_probabilities(self, rows):
        self.batch_calls += 1
        self.rows.extend(dict(row) for row in rows)
        return self.probabilities[: len(rows)]


def board_and_dealt():
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c"],
        bottom=["9c", "9d", "9s"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h"],
        bottom=["7h", "8h", "Th"],
    )
    dealt = ["Qs", "Ah", "7d"]
    return board, opponent, dealt


def _load_parts_args(*, allow_missing_stage8b_model_fallback):
    return SimpleNamespace(
        opening_model=Path("opening.pt"),
        turn1_model=Path("turn1.pt"),
        turn2_baseline_model=Path("turn2.pkl"),
        turn3_model=Path("turn3.pkl"),
        hu_turn3_stage7_model=Path("stage7.pt"),
        hu_turn3_reference_model=Path("reference.pt"),
        hu_turn2_stage8b_model=Path("missing-stage9f.pt"),
        device="cpu",
        allow_missing_stage8b_model_fallback=allow_missing_stage8b_model_fallback,
    )


def test_load_topk_parts_requires_stage8b_model_by_default(monkeypatch):
    def fake_load_parts(_args):
        raise OSError("missing stage9f")

    monkeypatch.setattr(topk_eval, "load_parts", fake_load_parts)

    args = _load_parts_args(allow_missing_stage8b_model_fallback=False)

    try:
        load_topk_parts(args)
    except OSError as exc:
        assert "missing stage9f" in str(exc)
    else:
        raise AssertionError("expected Stage8b model load failure to be fatal by default")


def test_load_topk_parts_allows_stage8b_model_fallback_only_when_flagged(monkeypatch):
    def fake_action_loader(path):
        return f"action:{path}"

    def fake_hu_loader(path):
        return f"hu:{path}"

    def fake_stage8_loader(path, *, device):
        raise OSError(f"missing {path} on {device}")

    monkeypatch.setattr(topk_eval, "load_action_value_model", fake_action_loader)
    monkeypatch.setattr(topk_eval, "load_hu_action_value_model", fake_hu_loader)
    monkeypatch.setattr(topk_eval, "load_hu_turn2_stage8_model", fake_stage8_loader)

    args = _load_parts_args(allow_missing_stage8b_model_fallback=True)
    parts = load_topk_parts(args)

    assert parts.opening == "action:opening.pt"
    assert parts.turn2_baseline == "action:turn2.pkl"
    assert parts.hu_turn3_stage7 == "hu:stage7.pt"
    assert parts.hu_turn3_reference == "hu:reference.pt"
    assert parts.hu_turn2_stage8 is None
    assert args.stage8b_model_load_failed is True
    assert "missing missing-stage9f.pt on cpu" in args.stage8b_model_load_error


def test_parse_topk_config_accepts_confirm_se_max_guard():
    config = parse_topk_configs("k3/mc16/d0/se0/confirm32/cse1.5/csemax3/pd0/seat=first/bygate_delta")[0]

    assert config.confirm_se_multiplier == 1.5
    assert config.max_confirm_se == 3.0
    assert "csemax3" in config.config_id


def test_topk_confirm_se_max_guard_falls_back_to_baseline():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    policy = HuTurn2Stage8bTopKMcRerankPolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8b_model=Stage8bTopKModel(),
        topk_rerank_config=TopKMcRerankConfig(
            top_k=1,
            mc_samples=1,
            min_delta=0.0,
            confirm_mc_samples=1,
            confirm_se_multiplier=0.0,
            min_confirm_delta=1.0,
            max_confirm_se=2.5,
        ),
        topk_decision_log=log,
        seed=1,
        seat="first",
    )
    call_count = 0

    def fake_rerank_sample(**kwargs):
        nonlocal call_count
        call_count += 1
        candidate_index = 0
        baseline_index = 1
        return {
            "common_random_future_digest": "stage-a" if call_count == 1 else "confirm",
            "paired_delta_candidate_index": candidate_index,
            "paired_delta_baseline_index": baseline_index,
            "paired_delta_mean": 5.0,
            "paired_delta_standard_error": 3.0,
            "paired_delta_count": 1,
            "actions": [
                {"original_index": candidate_index, "score": 5.0, "standard_error": 0.1},
                {"original_index": baseline_index, "score": 0.0, "standard_error": 0.1},
            ],
        }

    policy._rerank_sample = fake_rerank_sample

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=[*opponent.all_cards()],
        opponent_board=opponent,
    )

    assert action == actions[1]
    record = log[-1]
    assert record["override_fired"] is False
    assert record["no_override_reason"] == "above_confirm_se"
    assert record["confirm_delta"] == 5.0
    assert record["confirm_delta_se"] == 3.0
    assert record["max_confirm_se"] == 2.5


def test_topk_runtime_log_separates_true_and_visible_dead_cards_on_fallback():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    policy = HuTurn2Stage8bTopKMcRerankPolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8b_model=None,
        topk_rerank_config=TopKMcRerankConfig(top_k=3, mc_samples=1, min_delta=0.0),
        topk_decision_log=log,
        topk_context={},
        seed=1,
        seat="first",
    )

    positions = capture_decision_log_positions(policy)
    action = policy.choose_action(
        board,
        dealt,
        dead_cards=[*opponent.all_cards(), "2c"],
        opponent_board=opponent,
    )
    assert log[-1]["replay_ready"] is False
    attach_replay_truth(
        positions,
        ReplayTruth(
            true_dead_cards=("2c", "3c"),
            visible_dead_cards=(*opponent.all_cards(), "2c"),
            hero_private_discards=("2c",),
            opponent_private_discards=("3c",),
        ),
    )

    assert action == actions[1]
    record = log[-1]
    assert record["no_override_reason"] == "model_load_failed"
    assert record["dead_cards"] == [*opponent.all_cards(), "2c"]
    assert record["visible_dead_cards"] == [*opponent.all_cards(), "2c"]
    assert "2c" in record["visible_dead_cards"]
    assert "3c" not in record["visible_dead_cards"]
    assert record["hero_private_discards"] == ["2c"]
    assert "opponent_private_discards" not in record
    assert record["true_dead_cards"] == ["2c", "3c"]
    assert record["true_hero_private_discards"] == ["2c"]
    assert record["true_opponent_private_discards"] == ["3c"]


def test_t3_decision_summary_records_turn_and_downstream_override():
    hand = {
        "turns": [
            {
                "turn": "T3",
                "player": 0,
                "profile": "stage8b_topk_mc",
                "dealt": ["Ah", "Ks", "3d"],
                "placements": [["Ah", "top"], ["Ks", "middle"]],
                "discards": ["3d"],
                "board": {"top": ["Ah"], "middle": ["Ks"], "bottom": []},
            },
            {
                "turn": "T3",
                "player": 1,
                "profile": "baseline",
                "dealt": ["2c", "4c", "6c"],
                "placements": [["2c", "top"], ["4c", "middle"]],
                "discards": ["6c"],
                "board": {"top": ["2c"], "middle": ["4c"], "bottom": []},
            },
        ]
    }
    stage7_log = [
        {
            "turn": "T3",
            "seat": "first",
            "override_fired": True,
            "no_override_reason": "",
            "reference_margin": 11.0,
            "stage7_predicted_margin": 5.5,
            "stage3_action": {"placements": [], "discards": []},
            "stage7_action": {"placements": [], "discards": []},
            "final_action": {"placements": [], "discards": []},
        },
        {"turn": "T3", "seat": "second", "override_fired": False, "no_override_reason": "stage7_disabled"},
    ]

    candidate = _trace_t3_decision_summary(hand, 0, stage7_log)
    baseline = _trace_t3_decision_summary(hand, 1, stage7_log)
    row = {}
    _add_prefixed_t3_summary(row, "candidate", candidate)
    _add_prefixed_t3_summary(row, "baseline", baseline)
    _add_t3_summary_delta(row)

    assert candidate["player_t3_turn"]["dealt"] == ["Ah", "Ks", "3d"]
    assert candidate["opponent_t3_turn"]["dealt"] == ["2c", "4c", "6c"]
    assert candidate["stage7_record_count"] == 1
    assert candidate["stage7_override_fired"] is True
    assert baseline["stage7_override_fired"] is False
    assert row["candidate_downstream_override_fired"] is True
    assert row["baseline_downstream_override_fired"] is False
    assert row["downstream_override_fired"] is True
    assert row["t3_decision_summary"]["candidate"]["seat"] == "first"


def test_paired_delta_summary_for_action_prefers_by_action_distribution():
    sample = {
        "paired_delta_candidate_index": 1,
        "paired_delta_baseline_index": 0,
        "paired_delta_mean": 9.0,
        "paired_delta_standard_error": 0.1,
        "paired_delta_count": 2,
        "paired_delta_by_action": [
            {
                "candidate_index": 3,
                "baseline_index": 0,
                "count": 4,
                "mean": -1.25,
                "standard_error": 0.25,
                "p05": -2.0,
                "p50": -1.0,
                "p95": -0.5,
                "component_delta_summaries": {
                    "fl_delta": {
                        "mean": -3.0,
                        "p05": -10.0,
                        "le_neg6_rate": 0.25,
                    }
                },
            }
        ],
    }

    summary = _paired_delta_summary_for_action(sample, 3, 0)

    assert summary is not None
    assert summary["candidate_index"] == 3
    assert summary["mean"] == -1.25
    assert summary["standard_error"] == 0.25
    assert summary["component_delta_summaries"]["fl_delta"]["le_neg6_rate"] == 0.25


def test_risk_veto_count_counts_only_vetoed_rows():
    decisions = [
        {"local_ev_risk_vetoed": True},
        {"local_ev_risk_vetoed": False},
        {"override_fired": True},
        {"local_ev_risk_vetoed": 1},
        {"local_ev_risk_vetoed": ""},
        {"local_ev_risk_would_veto": True, "local_ev_risk_vetoed": False},
    ]

    assert risk_veto_count(decisions) == 3


def test_risk_veto_candidate_metrics_estimates_veto_utility_from_would_veto_rows():
    rows = [
        {
            "config_id": "cfg",
            "local_ev_risk_would_veto": True,
            "local_ev_risk_audit_only": True,
            "local_ev_risk_vetoed": False,
            "realized_delta_valid": True,
            "realized_candidate_seat_delta": -4.0,
        },
        {
            "config_id": "cfg",
            "local_ev_risk_would_veto": True,
            "local_ev_risk_audit_only": True,
            "local_ev_risk_vetoed": False,
            "realized_delta_valid": True,
            "realized_candidate_seat_delta": 2.0,
        },
        {
            "config_id": "cfg",
            "override_fired": True,
            "realized_delta_valid": True,
            "realized_candidate_seat_delta": 10.0,
        },
    ]

    [metrics] = risk_veto_candidate_metrics(rows)

    assert metrics["would_veto_count"] == 2
    assert metrics["audit_only_would_veto_count"] == 2
    assert metrics["actual_veto_count"] == 0
    assert metrics["realized_would_veto_count"] == 2
    assert metrics["realized_candidate_delta_mean"] == -1.0
    assert metrics["veto_utility_per_veto"] == 1.0
    assert metrics["estimated_veto_utility_per_hand"] == 2 / 3


def test_stage8c_fire_selector_direct_fire_skips_mc_rerank():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    selector = FakeBatchStage8cFireSelectorScorer([0.96])
    policy = HuTurn2Stage8bTopKMcRerankPolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8b_model=Stage8bTopKModel(),
        topk_rerank_config=TopKMcRerankConfig(top_k=1, mc_samples=999, min_delta=0.0),
        topk_decision_log=log,
        seed=1,
        seat="first",
        stage8c_fire_selector_scorer=selector,
        stage8c_fire_selector_threshold=0.95,
        stage8c_fire_selector_direct_fire=True,
    )

    def fail_rerank_sample(**kwargs):
        raise AssertionError("direct-fire must not call MC rerank")

    policy._rerank_sample = fail_rerank_sample

    action = policy.choose_action(board, dealt, dead_cards=opponent.all_cards(), opponent_board=opponent)

    assert action == actions[0]
    record = log[-1]
    assert record["override_fired"] is True
    assert record["stage8c_fire_selector_direct_fire_enabled"] is True
    assert record["stage8c_fire_selector_direct_fire_used"] is True
    assert record["stage8c_fire_selector_direct_fire_candidate_count"] == 1
    assert record["stage8c_fire_selector_direct_fire_probability"] == 0.96
    assert record["mc_rerank_latency_ms"] == 0.0
    assert record["confirm_mc_latency_ms"] == 0.0
    assert record["common_random_future_digest"] == ""
    assert record["rerank_best_index"] == 0


def test_confirm_delta_guard_blocks_after_stage_a_without_raising_min_delta():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    policy = HuTurn2Stage8bTopKMcRerankPolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8b_model=Stage8bTopKModel(),
        topk_rerank_config=TopKMcRerankConfig(
            top_k=1,
            mc_samples=1,
            min_delta=0.0,
            confirm_mc_samples=1,
            confirm_se_multiplier=0.0,
            min_confirm_delta=1.0,
        ),
        topk_decision_log=log,
        seed=1,
        seat="first",
    )
    call_count = 0

    def fake_rerank_sample(**kwargs):
        nonlocal call_count
        call_count += 1
        candidate_index = 0
        baseline_index = 1
        if call_count == 1:
            return {
                "common_random_future_digest": "stage-a",
                "actions": [
                    {"original_index": candidate_index, "score": 2.0, "standard_error": 0.1},
                    {"original_index": baseline_index, "score": 0.0, "standard_error": 0.1},
                ],
            }
        return {
            "common_random_future_digest": "confirm",
            "paired_delta_candidate_index": candidate_index,
            "paired_delta_baseline_index": baseline_index,
            "paired_delta_mean": 0.5,
            "paired_delta_standard_error": 0.1,
            "paired_delta_count": 1,
            "actions": [
                {"original_index": candidate_index, "score": 0.5, "standard_error": 0.1},
                {"original_index": baseline_index, "score": 0.0, "standard_error": 0.1},
            ],
        }

    policy._rerank_sample = fake_rerank_sample

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=[*opponent.all_cards()],
        opponent_board=opponent,
    )

    assert action == actions[1]
    assert call_count == 2
    record = log[-1]
    assert record["override_fired"] is False
    assert record["no_override_reason"] == "below_confirm_delta"
    assert record["confirm_delta"] == 0.5
    assert record["min_rerank_delta"] == 0.0
    assert record["min_confirm_delta"] == 1.0


def test_second_confirm_delta_guard_does_not_block_first_seat():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    policy = HuTurn2Stage8bTopKMcRerankPolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8b_model=Stage8bTopKModel(),
        topk_rerank_config=TopKMcRerankConfig(
            top_k=1,
            mc_samples=1,
            min_delta=0.0,
            confirm_mc_samples=1,
            confirm_se_multiplier=0.0,
            second_min_confirm_delta=1.0,
        ),
        topk_decision_log=log,
        seed=1,
        seat="first",
    )
    call_count = 0

    def fake_rerank_sample(**kwargs):
        nonlocal call_count
        call_count += 1
        candidate_index = 0
        baseline_index = 1
        if call_count == 1:
            return {
                "common_random_future_digest": "stage-a",
                "actions": [
                    {"original_index": candidate_index, "score": 2.0, "standard_error": 0.1},
                    {"original_index": baseline_index, "score": 0.0, "standard_error": 0.1},
                ],
            }
        return {
            "common_random_future_digest": "confirm",
            "paired_delta_candidate_index": candidate_index,
            "paired_delta_baseline_index": baseline_index,
            "paired_delta_mean": 0.5,
            "paired_delta_standard_error": 0.1,
            "paired_delta_count": 1,
            "actions": [
                {"original_index": candidate_index, "score": 0.5, "standard_error": 0.1},
                {"original_index": baseline_index, "score": 0.0, "standard_error": 0.1},
            ],
        }

    policy._rerank_sample = fake_rerank_sample

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=[*opponent.all_cards()],
        opponent_board=opponent,
    )

    assert action == actions[0]
    record = log[-1]
    assert record["override_fired"] is True
    assert record["confirm_delta"] == 0.5
    assert record["min_confirm_delta"] is None
    assert record["second_min_confirm_delta"] == 1.0
    assert record["seat_min_confirm_delta"] is None


def test_second_confirm_delta_guard_blocks_second_seat():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    policy = HuTurn2Stage8bTopKMcRerankPolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8b_model=Stage8bTopKModel(),
        topk_rerank_config=TopKMcRerankConfig(
            top_k=1,
            mc_samples=1,
            min_delta=0.0,
            confirm_mc_samples=1,
            confirm_se_multiplier=0.0,
            second_min_confirm_delta=1.0,
        ),
        topk_decision_log=log,
        seed=1,
        seat="second",
    )
    call_count = 0

    def fake_rerank_sample(**kwargs):
        nonlocal call_count
        call_count += 1
        candidate_index = 0
        baseline_index = 1
        if call_count == 1:
            return {
                "common_random_future_digest": "stage-a",
                "actions": [
                    {"original_index": candidate_index, "score": 2.0, "standard_error": 0.1},
                    {"original_index": baseline_index, "score": 0.0, "standard_error": 0.1},
                ],
            }
        return {
            "common_random_future_digest": "confirm",
            "paired_delta_candidate_index": candidate_index,
            "paired_delta_baseline_index": baseline_index,
            "paired_delta_mean": 0.5,
            "paired_delta_standard_error": 0.1,
            "paired_delta_count": 1,
            "actions": [
                {"original_index": candidate_index, "score": 0.5, "standard_error": 0.1},
                {"original_index": baseline_index, "score": 0.0, "standard_error": 0.1},
            ],
        }

    policy._rerank_sample = fake_rerank_sample

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=[*opponent.all_cards()],
        opponent_board=opponent,
    )

    assert action == actions[1]
    record = log[-1]
    assert record["override_fired"] is False
    assert record["no_override_reason"] == "below_confirm_delta"
    assert record["confirm_delta"] == 0.5
    assert record["second_min_confirm_delta"] == 1.0
    assert record["seat_min_confirm_delta"] == 1.0


def test_local_ev_risk_veto_blocks_after_confirm_passes():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    risk = FakeLocalEvRiskScorer(0.91)
    policy = HuTurn2Stage8bTopKMcRerankPolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8b_model=Stage8bTopKModel(),
        topk_rerank_config=TopKMcRerankConfig(
            top_k=1,
            mc_samples=1,
            min_delta=0.0,
            confirm_mc_samples=1,
            confirm_se_multiplier=0.0,
            min_confirm_delta=1.0,
        ),
        topk_decision_log=log,
        local_ev_risk_scorer=risk,
        local_ev_risk_threshold=0.8,
        seed=1,
        seat="second",
    )
    call_count = 0

    def fake_rerank_sample(**kwargs):
        nonlocal call_count
        call_count += 1
        candidate_index = 0
        baseline_index = 1
        if call_count == 1:
            return {
                "common_random_future_digest": "stage-a",
                "actions": [
                    {"original_index": candidate_index, "score": 3.0, "standard_error": 0.1},
                    {"original_index": baseline_index, "score": 0.0, "standard_error": 0.1},
                ],
            }
        return {
            "common_random_future_digest": "confirm",
            "paired_delta_candidate_index": candidate_index,
            "paired_delta_baseline_index": baseline_index,
            "paired_delta_mean": 2.5,
            "paired_delta_standard_error": 0.5,
            "paired_delta_count": 1,
            "actions": [
                {"original_index": candidate_index, "score": 2.5, "standard_error": 0.1},
                {"original_index": baseline_index, "score": 0.0, "standard_error": 0.1},
            ],
        }

    policy._rerank_sample = fake_rerank_sample

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=[*opponent.all_cards()],
        opponent_board=opponent,
    )

    assert action == actions[1]
    assert len(risk.rows) == 1
    assert risk.rows[0]["confirm_delta"] == 2.5
    assert risk.rows[0]["confirm_delta_se"] == 0.5
    assert risk.rows[0]["seat"] == "second"
    assert risk.rows[0]["hero_board"] == {
        "top": ["Qh"],
        "middle": ["Kh", "Kd", "6c"],
        "bottom": ["9c", "9d", "9s"],
    }
    assert risk.rows[0]["candidate_action"] == action_to_json(board, actions[0])
    assert risk.rows[0]["baseline_action"] == action_to_json(board, actions[1])
    assert risk.rows[0]["cards_to_place"] == dealt
    assert risk.rows[0]["candidate_ev_rank"] == 1
    record = log[-1]
    assert record["override_fired"] is False
    assert record["no_override_reason"] == "local_ev_risk_veto"
    assert record["local_ev_risk_probability"] == 0.91
    assert record["local_ev_risk_would_veto"] is True
    assert record["local_ev_risk_vetoed"] is True
    assert record["local_ev_risk_threshold"] == 0.8


def test_local_ev_risk_audit_only_logs_would_veto_without_blocking():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    risk = FakeLocalEvRiskScorer(0.91)
    policy = HuTurn2Stage8bTopKMcRerankPolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8b_model=Stage8bTopKModel(),
        topk_rerank_config=TopKMcRerankConfig(
            top_k=1,
            mc_samples=1,
            min_delta=0.0,
            confirm_mc_samples=1,
            confirm_se_multiplier=0.0,
            min_confirm_delta=1.0,
        ),
        topk_decision_log=log,
        local_ev_risk_scorer=risk,
        local_ev_risk_threshold=0.8,
        local_ev_risk_audit_only=True,
        seed=1,
        seat="second",
    )
    call_count = 0

    def fake_rerank_sample(**kwargs):
        nonlocal call_count
        call_count += 1
        candidate_index = 0
        baseline_index = 1
        return {
            "common_random_future_digest": "stage-a" if call_count == 1 else "confirm",
            "paired_delta_candidate_index": candidate_index,
            "paired_delta_baseline_index": baseline_index,
            "paired_delta_mean": 2.5,
            "paired_delta_standard_error": 0.5,
            "paired_delta_count": 1,
            "actions": [
                {"original_index": candidate_index, "score": 2.5, "standard_error": 0.1},
                {"original_index": baseline_index, "score": 0.0, "standard_error": 0.1},
            ],
        }

    policy._rerank_sample = fake_rerank_sample

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=[*opponent.all_cards()],
        opponent_board=opponent,
    )

    assert action == actions[0]
    assert len(risk.rows) == 1
    record = log[-1]
    assert record["override_fired"] is True
    assert record["no_override_reason"] == ""
    assert record["local_ev_risk_audit_only"] is True
    assert record["local_ev_risk_probability"] == 0.91
    assert record["local_ev_risk_would_veto"] is True
    assert record["local_ev_risk_vetoed"] is False


def test_local_ev_risk_allows_low_risk_confirmed_override():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    policy = HuTurn2Stage8bTopKMcRerankPolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8b_model=Stage8bTopKModel(),
        topk_rerank_config=TopKMcRerankConfig(
            top_k=1,
            mc_samples=1,
            min_delta=0.0,
            confirm_mc_samples=1,
            confirm_se_multiplier=0.0,
            min_confirm_delta=1.0,
        ),
        topk_decision_log=log,
        local_ev_risk_scorer=FakeLocalEvRiskScorer(0.25),
        local_ev_risk_threshold=0.8,
        seed=1,
        seat="first",
    )
    call_count = 0

    def fake_rerank_sample(**kwargs):
        nonlocal call_count
        call_count += 1
        candidate_index = 0
        baseline_index = 1
        if call_count == 1:
            return {
                "common_random_future_digest": "stage-a",
                "actions": [
                    {"original_index": candidate_index, "score": 3.0, "standard_error": 0.1},
                    {"original_index": baseline_index, "score": 0.0, "standard_error": 0.1},
                ],
            }
        return {
            "common_random_future_digest": "confirm",
            "paired_delta_candidate_index": candidate_index,
            "paired_delta_baseline_index": baseline_index,
            "paired_delta_mean": 2.5,
            "paired_delta_standard_error": 0.5,
            "paired_delta_count": 1,
            "actions": [
                {"original_index": candidate_index, "score": 2.5, "standard_error": 0.1},
                {"original_index": baseline_index, "score": 0.0, "standard_error": 0.1},
            ],
        }

    policy._rerank_sample = fake_rerank_sample

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=[*opponent.all_cards()],
        opponent_board=opponent,
    )

    assert action == actions[0]
    record = log[-1]
    assert record["override_fired"] is True
    assert record["no_override_reason"] == ""
    assert record["local_ev_risk_probability"] == 0.25
    assert record["local_ev_risk_would_veto"] is False
    assert record["local_ev_risk_vetoed"] is False


def test_stage8c_fire_selector_filters_before_mc():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    fire_selector = FakeStage8cFireSelectorScorer(0.49)
    policy = HuTurn2Stage8bTopKMcRerankPolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8b_model=Stage8bTopKModel(),
        topk_rerank_config=TopKMcRerankConfig(top_k=1, mc_samples=1, min_delta=0.0),
        topk_decision_log=log,
        stage8c_fire_selector_scorer=fire_selector,
        stage8c_fire_selector_threshold=0.5,
        seed=1,
        seat="first",
    )
    call_count = 0

    def fake_rerank_sample(**kwargs):
        nonlocal call_count
        call_count += 1
        return {}

    policy._rerank_sample = fake_rerank_sample

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=[*opponent.all_cards()],
        opponent_board=opponent,
    )

    assert action == actions[1]
    assert call_count == 0
    assert len(fire_selector.rows) == 1
    assert fire_selector.rows[0]["candidate_action"] == action_to_json(board, actions[0])
    assert fire_selector.rows[0]["baseline_action"] == action_to_json(board, actions[1])
    assert fire_selector.rows[0]["topk_score"] == "delta"
    record = log[-1]
    assert record["override_fired"] is False
    assert record["no_override_reason"] == "below_fire_selector_threshold"
    assert record["stage8c_fire_selector_enabled"] is True
    assert record["stage8c_fire_selector_probability"] == 0.49
    assert record["stage8c_fire_selector_max_probability"] == 0.49
    assert record["stage8c_fire_selector_evaluated_count"] == 1
    assert record["stage8c_fire_selector_passed_count"] == 0
    assert record["stage8c_fire_selector_candidates"][0]["passed"] is False
    assert record["stage8c_fire_selector_candidates"][0]["action"] == action_to_json(board, actions[0])
    assert record["stage8c_fire_selector_candidates"][0]["post_t2_board"] == board_to_json(
        board.place(actions[0].placements)
    )


def test_stage8c_fire_selector_audit_only_does_not_filter():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    policy = HuTurn2Stage8bTopKMcRerankPolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8b_model=Stage8bTopKModel(),
        topk_rerank_config=TopKMcRerankConfig(top_k=1, mc_samples=1, min_delta=0.0),
        topk_decision_log=log,
        stage8c_fire_selector_scorer=FakeStage8cFireSelectorScorer(0.10),
        stage8c_fire_selector_threshold=0.9,
        stage8c_fire_selector_audit_only=True,
        seed=1,
        seat="first",
    )
    call_count = 0

    def fake_rerank_sample(**kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "common_random_future_digest": "stage-a",
            "actions": [
                {"original_index": 0, "score": 2.0, "standard_error": 0.1},
                {"original_index": 1, "score": 0.0, "standard_error": 0.1},
            ],
        }

    policy._rerank_sample = fake_rerank_sample

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=[*opponent.all_cards()],
        opponent_board=opponent,
    )

    assert action == actions[0]
    assert call_count == 1
    record = log[-1]
    assert record["override_fired"] is True
    assert record["no_override_reason"] == ""
    assert record["stage8c_fire_selector_audit_only"] is True
    assert record["stage8c_fire_selector_probability"] == 0.10
    assert record["stage8c_fire_selector_passed_count"] == 0


def test_stage8c_fire_selector_high_probability_runs_mc_without_veto_semantics():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    policy = HuTurn2Stage8bTopKMcRerankPolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8b_model=Stage8bTopKModel(),
        topk_rerank_config=TopKMcRerankConfig(top_k=1, mc_samples=1, min_delta=0.0),
        topk_decision_log=log,
        stage8c_fire_selector_scorer=FakeStage8cFireSelectorScorer(0.91),
        stage8c_fire_selector_threshold=0.7,
        seed=1,
        seat="first",
    )
    call_count = 0

    def fake_rerank_sample(**kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "common_random_future_digest": "stage-a",
            "actions": [
                {"original_index": 0, "score": 2.0, "standard_error": 0.1},
                {"original_index": 1, "score": 0.0, "standard_error": 0.1},
            ],
        }

    policy._rerank_sample = fake_rerank_sample

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=[*opponent.all_cards()],
        opponent_board=opponent,
    )

    assert action == actions[0]
    assert call_count == 1
    record = log[-1]
    assert record["override_fired"] is True
    assert record["stage8c_fire_selector_probability"] == 0.91
    assert record["stage8c_fire_selector_passed_count"] == 1
    assert record["local_ev_risk_vetoed"] is False


def test_stage8c_fire_selector_uses_batch_scoring_for_multiple_candidates():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    assert len(actions) > 2
    log = []
    fire_selector = FakeBatchStage8cFireSelectorScorer([0.40, 0.80])
    policy = HuTurn2Stage8bTopKMcRerankPolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8b_model=Stage8bTopKModel(),
        topk_rerank_config=TopKMcRerankConfig(top_k=2, mc_samples=1, min_delta=0.0),
        topk_decision_log=log,
        stage8c_fire_selector_scorer=fire_selector,
        stage8c_fire_selector_threshold=0.5,
        seed=1,
        seat="first",
    )
    call_count = 0
    selected_index = None

    def fake_rerank_sample(**kwargs):
        nonlocal call_count, selected_index
        call_count += 1
        assert kwargs["action_indices"][0] == 1
        assert len(kwargs["action_indices"]) == 2
        selected_index = kwargs["action_indices"][1]
        return {
            "common_random_future_digest": "stage-a",
            "actions": [
                {"original_index": selected_index, "score": 2.0, "standard_error": 0.1},
                {"original_index": 1, "score": 0.0, "standard_error": 0.1},
            ],
        }

    policy._rerank_sample = fake_rerank_sample

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=[*opponent.all_cards()],
        opponent_board=opponent,
    )

    assert selected_index is not None
    assert action == actions[selected_index]
    assert call_count == 1
    assert fire_selector.batch_calls == 1
    assert fire_selector.scalar_calls == 0
    assert len(fire_selector.rows) == 2
    assert fire_selector.rows[0]["candidate_action"] == action_to_json(board, actions[0])
    assert fire_selector.rows[1]["candidate_action"] == action_to_json(board, actions[selected_index])
    record = log[-1]
    assert record["override_fired"] is True
    assert record["stage8c_fire_selector_evaluated_count"] == 2
    assert record["stage8c_fire_selector_passed_count"] == 1
    assert record["stage8c_fire_selector_probability"] == 0.80
    assert record["stage8c_fire_selector_max_probability"] == 0.80
    assert [item["passed"] for item in record["stage8c_fire_selector_candidates"]] == [False, True]


def test_stage8c_risk_rank_guard_skips_veto_outside_rank_bucket():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    risk = FakeLocalEvRiskScorer(0.99)
    policy = HuTurn2Stage8bTopKMcRerankPolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8b_model=Stage8bTopKModel(),
        topk_rerank_config=TopKMcRerankConfig(
            top_k=1,
            mc_samples=1,
            min_delta=0.0,
            confirm_mc_samples=1,
            confirm_se_multiplier=0.0,
            min_confirm_delta=1.0,
        ),
        topk_decision_log=log,
        local_ev_risk_scorer=risk,
        local_ev_risk_threshold=0.35,
        local_ev_risk_rank_min=4,
        local_ev_risk_rank_max=5,
        seed=1,
        seat="first",
    )
    call_count = 0

    def fake_rerank_sample(**kwargs):
        nonlocal call_count
        call_count += 1
        candidate_index = 0
        baseline_index = 1
        return {
            "common_random_future_digest": "stage-a" if call_count == 1 else "confirm",
            "paired_delta_candidate_index": candidate_index,
            "paired_delta_baseline_index": baseline_index,
            "paired_delta_mean": 2.5,
            "paired_delta_standard_error": 0.5,
            "paired_delta_count": 1,
            "actions": [
                {"original_index": candidate_index, "score": 2.5, "standard_error": 0.1},
                {"original_index": baseline_index, "score": 0.0, "standard_error": 0.1},
            ],
        }

    policy._rerank_sample = fake_rerank_sample

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=[*opponent.all_cards()],
        opponent_board=opponent,
    )

    assert action == actions[0]
    assert risk.rows == []
    record = log[-1]
    assert record["override_fired"] is True
    assert record["local_ev_risk_rank_min"] == 4
    assert record["local_ev_risk_rank_max"] == 5
    assert record["local_ev_risk_rank_guard_passed"] is False
    assert record["local_ev_risk_probability"] is None
    assert record["local_ev_risk_vetoed"] is False
