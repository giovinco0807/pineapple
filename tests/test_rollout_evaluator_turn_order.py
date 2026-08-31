import pytest
import torch

from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board, Observation
from ai.mcts.rollout_evaluator import RolloutEvaluator


class _DummyPolicy(torch.nn.Module):
    def forward(self, state, mask=None):
        width = mask.shape[-1] if mask is not None else 1
        return torch.ones((state.shape[0], width), dtype=torch.float32)


class _RecordingEvaluator(RolloutEvaluator):
    def __init__(self):
        super().__init__(
            policy_net=_DummyPolicy(),
            n_rollouts=1,
            use_policy_playout=True,
        )
        self.steps = []
        self.final_counts = None

    def _bc_playout_turn(self, board, opp_board, cards, turn, discards, is_btn):
        before = board.card_count()
        other_before = opp_board.card_count()
        placements = 5 if before == 0 else 2
        card_index = 0
        for row_name, capacity in (("top", 3), ("middle", 5), ("bottom", 5)):
            row = getattr(board, row_name)
            while len(row) < capacity and card_index < placements:
                row.append(cards[card_index])
                card_index += 1
        if len(cards) > placements:
            discards.append(cards[placements])
        self.steps.append(
            (int(turn), bool(is_btn), before, other_before, board.card_count())
        )

    def _compute_score(self, my_board, opp_board):
        self.final_counts = (my_board.card_count(), opp_board.card_count())
        return 0.0


def _board(cards, count):
    row_sizes = {
        5: (1, 2, 2),
        7: (1, 3, 3),
        9: (2, 3, 4),
    }
    top_n, middle_n, bottom_n = row_sizes[count]
    return Board(
        top=list(cards[:top_n]),
        middle=list(cards[top_n : top_n + middle_n]),
        bottom=list(cards[top_n + middle_n : top_n + middle_n + bottom_n]),
    )


def _root_case(turn, is_btn):
    before = 5 + 2 * (turn - 1)
    cards = iter(ALL_CARDS)
    hero_cards = [next(cards) for _ in range(before)]
    opponent_count = before + (2 if is_btn else 0)
    opponent_cards = [next(cards) for _ in range(opponent_count)]
    dealt = [next(cards) for _ in range(3)]
    board = _board(hero_cards, before)
    opponent = _board(opponent_cards, opponent_count)
    obs = Observation(
        board_self=board,
        board_opponent=opponent,
        dealt_cards=dealt,
        known_discards_self=[],
        turn=turn,
        is_btn=is_btn,
    )
    action = get_turn_actions(dealt, board)[0]
    return obs, action, before


@pytest.mark.parametrize("turn", [1, 2])
@pytest.mark.parametrize("is_btn", [False, True])
@pytest.mark.parametrize("detailed", [False, True])
def test_rollout_uses_bb_first_street_order_and_role_flags(turn, is_btn, detailed):
    evaluator = _RecordingEvaluator()
    obs, action, before = _root_case(turn, is_btn)

    if detailed:
        evaluator._single_rollout_detailed(obs, action)
    else:
        evaluator._single_rollout(obs, action)

    next_street_pair = [
        (turn + 1, False, before + 2, before + 2, before + 4),
        (turn + 1, True, before + 2, before + 4, before + 4),
    ]
    if is_btn:
        # BTN completed the root street; opponent BB opens the next street.
        assert evaluator.steps[:2] == next_street_pair
    else:
        # BB completed its root action; BTN must answer on the same street.
        assert evaluator.steps[:3] == [
            (turn, True, before, before + 2, before + 2),
            *next_street_pair,
        ]

    assert evaluator.final_counts == (13, 13)
    for street in range(turn + 1, 5):
        flags = [is_btn_flag for step_turn, is_btn_flag, *_ in evaluator.steps if step_turn == street]
        assert flags == [False, True]
