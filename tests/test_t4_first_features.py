import pytest

import ai.tutor.exact_late as exact_late
import ai.tutor.t4_first_features as features
import ai.tutor.t4_bb_exact_vs_myopic_probe as probe
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board


def _root_actions(seed: int):
    root = probe.sample_random_root(seed)
    board = Board(
        top=list(root["bb_board"][0]),
        middle=list(root["bb_board"][1]),
        bottom=list(root["bb_board"][2]),
    )
    return root, board, get_turn_actions(list(root["draw"]), board)


@pytest.mark.parametrize("seed", (20260729, 20260730, 20260731))
def test_encode_shape_and_finiteness_across_roots(seed):
    root, board, actions = _root_actions(seed)
    assert actions
    for action in actions:
        final = exact_late.apply_action(board, action)
        vector = features.encode(
            (final.top, final.middle, final.bottom),
            root["btn_board"],
            list(root["bb_discards"]) + [action.discard],
        )
        assert len(vector) == features.FEATURE_SIZE == 109
        assert all(isinstance(value, float) for value in vector)
        assert all(value == value for value in vector)  # no NaN
        assert all(-1.0 <= value <= 2.0 for value in vector)


def test_block_sizes_sum_to_feature_size():
    assert (
        features.HERO_SIZE
        + features.OPPONENT_SIZE
        + features.JOINT_SIZE
        + features.CONTEXT_SIZE
        == features.FEATURE_SIZE
    )


@pytest.mark.parametrize("seed", (20260729, 20260732))
def test_node_cache_is_action_independent_and_exact(seed):
    """The unknown pool is identical for every action, so the shared block is."""
    root, board, actions = _root_actions(seed)
    cache = features.NodeCache.for_root(
        root["bb_board"], root["btn_board"], root["draw"], root["bb_discards"]
    )
    assert len(cache.pool) == 26
    for action in actions:
        final = exact_late.apply_action(board, action)
        hero_rows = (final.top, final.middle, final.bottom)
        cached = features.encode_action(hero_rows, cache)
        plain = features.encode(
            hero_rows,
            root["btn_board"],
            list(root["bb_discards"]) + [action.discard],
        )
        assert cached == plain


def test_partial_category_counts_jokers_as_wild():
    # Two naturals of a rank plus a joker is trips even though the row is short.
    assert features.partial_category(["7c", "7d", "X1"], 5) == 3
    assert features.partial_category(["7c", "8d", "X1"], 5) == 1
    assert features.partial_category(["7c", "8d", "9h"], 5) == 0
    # A complete row is evaluated for real, so flushes become visible.
    assert features.partial_category(["2c", "5c", "9c", "Jc", "Kc"], 5) == 5


def test_hero_block_marks_bust_and_zeroes_royalty():
    busted = (["As", "Ah", "Ad"], ["2c", "3d", "4h", "5s", "7c"], ["2h", "3s", "4c", "5d", "8h"])
    block = features.hero_block(busted)
    assert block[0] == 1.0  # busted flag
    assert block[1] == 0.0  # total royalty zeroed
    assert block[5] == 0.0  # no Fantasyland from a busted board


def test_joint_block_flags_locked_opponent_foul():
    # Opponent bottom is full and weaker than a full middle: foul is fixed.
    opponent = (
        ["2h"],
        ["Kc", "Kd", "Ks", "3h", "4s"],  # trips middle, complete
        ["5c", "6d", "7h", "8s", "Tc"],  # high card bottom (no straight), complete
    )
    hero = (["Qs", "Qh", "2d"], ["3c", "3d", "4h", "5s", "6c"], ["7c", "7d", "7h", "8s", "9d"])
    pool = features.unknown_pool(hero, opponent, [])
    _opponent_features, categories = features.opponent_block(opponent, pool)
    block = features.joint_block(hero, opponent, categories)
    assert block[0] == 1.0  # locked middle-over-bottom foul
    assert block[2] == 1.0  # any locked foul


def test_encode_rejects_incomplete_boards():
    root, board, actions = _root_actions(20260729)
    with pytest.raises(ValueError, match="complete"):
        features.encode(root["bb_board"], root["btn_board"], root["bb_discards"])


def test_joint_block_detects_certain_opponent_foul():
    """seed 5000103 from the diagnosis: opponent fouls on every draw."""
    root = probe.sample_random_root(5000103)
    cache = features.NodeCache.for_root(
        root["bb_board"], root["btn_board"], root["draw"], root["bb_discards"]
    )
    block = features.opponent_joint_block(root["btn_board"], cache.pool)
    assert block[0] == 1.0          # foul_rate
    assert block[2] == 0.0          # no surviving royalty
    assert block[3] == 0.0          # no surviving Fantasyland
    assert block[5] == 0.0 and block[6] == 0.0   # both tail probabilities


def test_joint_block_is_bounded_and_finite_across_strata():
    for seed in (5000000, 5000034, 5000568, 5000706, 5000763):
        root = probe.sample_random_root(seed)
        cache = features.NodeCache.for_root(
            root["bb_board"], root["btn_board"], root["draw"], root["bb_discards"]
        )
        block = features.opponent_joint_block(root["btn_board"], cache.pool)
        assert len(block) == features.JOINT_SIZE_ADDED == 8
        assert all(value == value for value in block)
        assert 0.0 <= block[0] <= 1.0
        assert 0.0 <= block[5] <= 1.0 and 0.0 <= block[6] <= 1.0
        assert block[6] <= block[5] + 1e-9   # P(>=15) cannot exceed P(>=6)
