import torch

from ai.engine.action_space import get_initial_actions, get_turn_actions
from ai.engine.encoding import Board, Observation
from ai.mcts.rollout_evaluator import RolloutEvaluator


class DummyPolicy(torch.nn.Module):
    def forward(self, state, mask=None):
        width = mask.shape[-1] if mask is not None else 1
        return torch.ones((state.shape[0], width), dtype=torch.float32)


class AscendingActionValue(torch.nn.Module):
    def predict_components(self, batch):
        scores = torch.arange(batch.shape[0], dtype=torch.float32, device=batch.device)
        zeros = torch.zeros_like(scores)
        return {
            "score": scores,
            "bust_prob": zeros,
            "fl_prob": zeros,
            "fl_type_probs": torch.zeros((batch.shape[0], 4), dtype=torch.float32, device=batch.device),
        }


class CountingActionValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def predict_components(self, batch):
        self.batch_sizes.append(int(batch.shape[0]))
        scores = torch.zeros(batch.shape[0], dtype=torch.float32, device=batch.device)
        return {
            "score": scores,
            "bust_prob": scores,
            "fl_prob": scores,
            "fl_type_probs": torch.zeros((batch.shape[0], 4), dtype=torch.float32, device=batch.device),
        }


class TurnAwareActionValue(torch.nn.Module):
    def __init__(self, expected_turn):
        super().__init__()
        self.expected_turn = int(expected_turn)
        self.batch_sizes = []

    def predict_components(self, batch, turn=None):
        assert turn is not None
        assert set(turn.detach().cpu().tolist()) == {self.expected_turn}
        self.batch_sizes.append(int(batch.shape[0]))
        scores = torch.arange(batch.shape[0], dtype=torch.float32, device=batch.device)
        return {
            "score": scores,
            "bust_prob": torch.zeros_like(scores),
            "fl_prob": torch.zeros_like(scores),
            "fl_type_probs": torch.zeros((batch.shape[0], 4), dtype=torch.float32, device=batch.device),
        }


class CountingRolloutEvaluator(RolloutEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_evals = 0

    def _evaluate_action(self, obs, action):
        self.rollout_evals += 1
        return 0.0


def _obs(turn: int) -> Observation:
    return Observation(
        board_self=Board(
            top=["As"],
            middle=["2h", "3h"],
            bottom=["4h", "5h"],
        ),
        board_opponent=Board(),
        dealt_cards=["Ah", "Kd", "Qc"],
        known_discards_self=[],
        turn=turn,
        is_btn=True,
    )


def _t0_obs() -> Observation:
    return Observation(
        board_self=Board(),
        board_opponent=Board(),
        dealt_cards=["Ah", "Kd", "Qc", "7s", "2h"],
        known_discards_self=[],
        turn=0,
        is_btn=True,
    )


def test_action_value_prefilter_uses_t0_safety_budget():
    obs = _t0_obs()
    valid_actions = get_initial_actions(obs.dealt_cards, obs.board_self)
    assert len(valid_actions) > 64

    evaluator = RolloutEvaluator(
        policy_net=DummyPolicy(),
        action_value_net=AscendingActionValue(),
        action_value_top_k=24,
        action_value_top_k_by_turn={0: 64, 2: 24},
    )

    kept = evaluator._action_value_prefilter(obs, list(enumerate(valid_actions)))

    assert len(kept) == 64
    assert {idx for idx, _ in kept} == set(range(len(valid_actions) - 64, len(valid_actions)))


def test_action_value_prefilter_keeps_t2_safety_budget():
    obs = _obs(turn=2)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    assert len(valid_actions) == 27

    evaluator = RolloutEvaluator(
        policy_net=DummyPolicy(),
        action_value_net=AscendingActionValue(),
        action_value_top_k=20,
        action_value_top_k_by_turn={2: 24},
    )

    kept = evaluator._action_value_prefilter(obs, list(enumerate(valid_actions)))

    assert len(kept) == 24
    assert {idx for idx, _ in kept} == set(range(3, 27))


def test_action_value_prefilter_uses_base_budget_for_non_t2():
    obs = _obs(turn=1)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)

    evaluator = RolloutEvaluator(
        policy_net=DummyPolicy(),
        action_value_net=AscendingActionValue(),
        action_value_top_k=20,
        action_value_top_k_by_turn={2: 24},
    )

    kept = evaluator._action_value_prefilter(obs, list(enumerate(valid_actions)))

    assert len(kept) == 20
    assert {idx for idx, _ in kept} == set(range(7, 27))


def test_action_value_suit_ensemble_expands_selected_turn():
    obs = _obs(turn=2)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    reranker = CountingActionValue()

    evaluator = RolloutEvaluator(
        policy_net=DummyPolicy(),
        action_value_net=reranker,
        action_value_suit_ensemble_turns={2},
        action_value_suit_ensemble_size=4,
    )

    scores = evaluator._score_candidates_action_value(obs, list(enumerate(valid_actions)))

    assert len(scores) == len(valid_actions)
    assert reranker.batch_sizes == [len(valid_actions) * 4]


def test_action_value_suit_ensemble_default_size_is_eight():
    obs = _obs(turn=2)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    reranker = CountingActionValue()

    evaluator = RolloutEvaluator(
        policy_net=DummyPolicy(),
        action_value_net=reranker,
        action_value_suit_ensemble_turns={2},
    )

    scores = evaluator._score_candidates_action_value(obs, list(enumerate(valid_actions)))

    assert len(scores) == len(valid_actions)
    assert reranker.batch_sizes == [len(valid_actions) * 8]


def test_action_value_suit_ensemble_skips_other_turns():
    obs = _obs(turn=1)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    reranker = CountingActionValue()

    evaluator = RolloutEvaluator(
        policy_net=DummyPolicy(),
        action_value_net=reranker,
        action_value_suit_ensemble_turns={2},
    )

    scores = evaluator._score_candidates_action_value(obs, list(enumerate(valid_actions)))

    assert len(scores) == len(valid_actions)
    assert reranker.batch_sizes == [len(valid_actions)]


def test_action_value_turn_override_uses_turn_specific_model():
    obs = _obs(turn=3)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    base_reranker = CountingActionValue()
    turn_reranker = TurnAwareActionValue(expected_turn=3)

    evaluator = RolloutEvaluator(
        policy_net=DummyPolicy(),
        action_value_net=base_reranker,
        action_value_nets_by_turn={3: turn_reranker},
    )

    scores = evaluator._score_candidates_action_value(obs, list(enumerate(valid_actions)))

    assert len(scores) == len(valid_actions)
    assert base_reranker.batch_sizes == []
    assert turn_reranker.batch_sizes == [len(valid_actions)]


def test_action_value_turn_override_works_without_base_model():
    obs = _obs(turn=3)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    turn_reranker = TurnAwareActionValue(expected_turn=3)

    evaluator = RolloutEvaluator(
        policy_net=DummyPolicy(),
        action_value_nets_by_turn={3: turn_reranker},
    )

    scores = evaluator._score_candidates_action_value(obs, list(enumerate(valid_actions)))

    assert len(scores) == len(valid_actions)
    assert turn_reranker.batch_sizes == [len(valid_actions)]


def test_full_width_bypasses_action_value_prefilter_for_rollout_scores():
    obs = _obs(turn=1)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    reranker = CountingActionValue()

    evaluator = CountingRolloutEvaluator(
        policy_net=DummyPolicy(),
        action_value_net=reranker,
        action_value_top_k=5,
        n_rollouts=1,
        full_width=True,
    )

    _best_idx, _action, scores = evaluator.select_action_with_scores(obs)

    assert evaluator.rollout_evals == len(valid_actions)
    assert sum(torch.isfinite(torch.tensor(scores)).tolist()) == len(valid_actions)
    assert reranker.batch_sizes == []
