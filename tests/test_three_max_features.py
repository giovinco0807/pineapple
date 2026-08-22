"""3-max feature encoding, and the T3 teacher corpus it consumes."""

from __future__ import annotations

import hashlib
import json

import pytest

from ofc_regular.cards import ALL_CARDS
from ofc_regular.state import Board
from ofc_regular.three_max.features import (
    CONTEST_SIZE,
    CONTEXT_END,
    CONTEXT_SIZE,
    DECK_BLOCK_END,
    DECK_BLOCK_SIZE,
    FEATURE_SIZE,
    HEAD_TO_HEAD_END,
    HEAD_TO_HEAD_SIZE,
    OUTLOOK_BLOCK_END,
    OUTLOOK_SIZE,
    SIDE_BLOCK_END,
    SIDE_SIZE,
    deck_block,
    encode,
    encode_record_action,
    outlook_block,
    side_block,
)
from ofc_regular.three_max.teacher import (
    TEACHER_SCHEMA,
    TeacherConfig,
    label_root,
    sample_root,
    write_corpus,
)

HERO = Board.from_rows(
    top=["Qh", "Qd"],
    middle=["Ks", "Kd", "3c", "4c"],
    bottom=["As", "Ad", "6c", "7d", "8h"],
)
OPP1 = Board.from_rows(
    top=["2s", "3d"],
    middle=["5s", "6d", "7h", "8c"],
    bottom=["9s", "9c", "Js", "Qs", "Kc"],
)
OPP2 = Board.from_rows(
    top=["3h", "5h"],
    middle=["7s", "8s", "Th", "Jh"],
    bottom=["Ac", "9h", "Td", "4d", "2d"],
)


def _unseen(*boards: Board) -> list[str]:
    seen: set[str] = set()
    for board in boards:
        seen.update(board.all_cards())
    return [card for card in ALL_CARDS if card not in seen]


def _encode_fixture() -> tuple[float, ...]:
    return encode(
        hero_board=HERO,
        opponent_boards=[OPP1, OPP2],
        dealt_cards=["Qh", "Qd", "2h"],
        hero_private_discards=["2c", "4h"],
        unseen=_unseen(HERO, OPP1, OPP2),
    ).features


# --- Layout -----------------------------------------------------------------


def test_block_boundaries_add_up():
    assert SIDE_BLOCK_END == SIDE_SIZE * 3 == 93
    assert OUTLOOK_BLOCK_END == SIDE_BLOCK_END + OUTLOOK_SIZE * 3 == 201
    assert HEAD_TO_HEAD_END == OUTLOOK_BLOCK_END + HEAD_TO_HEAD_SIZE * 2 == 221
    assert CONTEXT_END == HEAD_TO_HEAD_END + CONTEXT_SIZE == 225
    # The feature-v2 probe appended the deck and contest blocks; it did not
    # reorder anything, so a 225-dim checkpoint still reads a valid prefix.
    # That is load-bearing -- ``interior.model_interior`` slices on it -- so it
    # is asserted rather than remembered.
    assert DECK_BLOCK_END == CONTEXT_END + DECK_BLOCK_SIZE == 245
    assert FEATURE_SIZE == DECK_BLOCK_END + CONTEST_SIZE * 2 == 261


def test_the_wide_layout_only_appends_to_the_narrow_one():
    """The v2 blocks sit AFTER context, so ``features[:225]`` is still v1.

    A 225-dim checkpoint is read by slicing this prefix, so if the deck or
    contest block ever moved in front of context every such model would keep
    loading, keep scoring, and be wrong.
    """
    features = _encode_fixture()
    unseen = _unseen(HERO, OPP1, OPP2)
    assert list(features[CONTEXT_END:DECK_BLOCK_END]) == deck_block(unseen)
    assert len(features[DECK_BLOCK_END:]) == CONTEST_SIZE * 2
    # And the prefix is exactly the four v1 blocks, in the v1 order.
    assert list(features[:SIDE_SIZE]) == side_block(HERO)


def test_encoding_is_the_declared_width_and_finite():
    features = _encode_fixture()
    assert len(features) == FEATURE_SIZE
    assert all(value == value for value in features)  # no NaN
    assert all(-1.0001 <= value <= 1.0001 for value in features)


def test_encoding_is_pinned():
    """A golden pin, so any layout change has to be a deliberate one.

    Heads-up keeps the same kind of pin on its encoders precisely because a
    silent reordering would invalidate every trained weight without failing a
    single other test.

    Re-pinned on 2026-08-13 for the 261-dim layout the feature-v2 probe left in
    place.  The old 225-dim pin was
    ``5a1d613e721b246ca996c3e66c9d40d276ea4ee9efa08099cfec8ea556edfed8``; the
    two agree on the first 225 dims, which
    ``test_the_wide_layout_only_appends_to_the_narrow_one`` checks structurally.
    """
    features = _encode_fixture()
    digest = hashlib.sha256(
        json.dumps([round(value, 9) for value in features]).encode()
    ).hexdigest()
    assert digest == (
        "8bbe4627ddbb74a2cad26da1eeceb958035cf65f3129a55466cb451d00735512"
    )


def test_encoding_has_no_randomness():
    assert _encode_fixture() == _encode_fixture()


def test_side_block_is_31_dims_and_ignores_within_row_order():
    """Placement order is not information; a feature that reads it is a bug.

    The first version sliced ``cards[:3]`` to rank a partial row, so reversing
    a four-card middle row changed the encoding.  Every board here has partial
    rows precisely so that regression cannot come back unnoticed.
    """
    assert len(side_block(HERO)) == SIDE_SIZE
    for board in (HERO, OPP1, OPP2):
        shuffled = Board.from_rows(
            top=list(reversed(board.top)),
            middle=list(reversed(board.middle)),
            bottom=list(reversed(board.bottom)),
        )
        assert side_block(shuffled) == side_block(board)


def test_outlook_rejects_a_board_with_more_than_two_open_slots():
    from ofc_regular.three_max.features import _completions

    early = Board.from_rows(top=["2h"], middle=["3d"], bottom=["4c"])
    with pytest.raises(ValueError, match="at most two open slots"):
        _completions(early, _unseen(early))


def test_encode_rejects_the_wrong_opponent_count():
    with pytest.raises(ValueError, match="needs 2 opponent boards"):
        encode(
            hero_board=HERO,
            opponent_boards=[OPP1],
            dealt_cards=["2h"],
            hero_private_discards=[],
            unseen=_unseen(HERO, OPP1),
        )


def test_outlook_block_of_a_completed_board_is_a_point_mass():
    from ofc_regular.three_max.features import board_terminal

    complete = Board.from_rows(
        top=["Qh", "Qd", "2h"],
        middle=["Ks", "Kd", "3c", "4c", "5c"],
        bottom=["As", "Ad", "6c", "7d", "8h"],
    )
    block = outlook_block([board_terminal(complete)], 0)
    assert len(block) == OUTLOOK_SIZE
    # Each row's category histogram is one-hot when there is a single outcome.
    for row in range(3):
        row_histogram = block[row * 9 : (row + 1) * 9]
        assert sum(row_histogram) == pytest.approx(1.0)
        assert max(row_histogram) == pytest.approx(1.0)


# --- Against a real teacher corpus ------------------------------------------


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> list[dict]:
    output = tmp_path_factory.mktemp("t3") / "corpus.jsonl"
    config = TeacherConfig(samples=8, holdout_samples=8, rollout_policy_sims=2)
    write_corpus(output=output, count=2, base_seed=8_400_000, config=config)
    return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]


def test_teacher_records_describe_a_btn_t3_root(corpus):
    for record in corpus:
        assert record["schema"] == TEACHER_SCHEMA
        assert record["seat"] == "btn"
        assert record["street"] == "T3"
        assert record["unseen_count"] == 16
        assert len(record["dealt"]) == 3
        assert len(record["hero_private_discards"]) == 2
        assert record["opponent_seats"] == ["sb", "bb"]
        for board in record["opponent_boards"]:
            assert sum(len(board[row]) for row in ("top", "middle", "bottom")) == 11


def test_teacher_records_rank_every_legal_action(corpus):
    from ofc_regular.action_space import generate_actions

    for record in corpus:
        hero = Board.from_rows(
            record["hero_board"]["top"],
            record["hero_board"]["middle"],
            record["hero_board"]["bottom"],
        )
        legal = generate_actions(hero, record["dealt"])
        assert len(record["actions"]) == len(legal)
        evs = [action["ev"] for action in record["actions"]]
        assert evs == sorted(evs, reverse=True)
        assert record["best_action"] == 0
        assert record["score_gap"] == pytest.approx(evs[0] - evs[1])


def test_only_the_winner_carries_a_held_out_value(corpus):
    for record in corpus:
        assert record["ev_holdout"] is not None
        assert isinstance(record["ev_holdout"], float)


def test_corpus_manifest_hashes_the_bytes_it_describes(tmp_path):
    output = tmp_path / "corpus.jsonl"
    config = TeacherConfig(samples=4, holdout_samples=4, rollout_policy_sims=2)
    manifest = write_corpus(
        output=output, count=2, base_seed=8_500_000, config=config
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest["content_sha256"] == digest
    assert manifest["rows"] == 2
    assert manifest["seed_block"] == [8_500_000, 8_500_001]
    assert manifest["config_fingerprint"] == config.fingerprint()

    sidecar = json.loads(
        output.with_suffix(output.suffix + ".manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar == manifest


def test_labelling_the_same_root_twice_gives_the_same_record():
    config = TeacherConfig(samples=8, holdout_samples=8, rollout_policy_sims=2)
    world = sample_root(8_600_000, config=config)
    first = label_root(world.observe(), seed=8_600_000, config=config)
    second = label_root(world.observe(), seed=8_600_000, config=config)
    assert first == second


def test_every_action_of_a_root_encodes_to_a_distinct_vector(corpus):
    record = corpus[0]
    vectors = {
        encode_record_action(record, index).features
        for index in range(len(record["actions"]))
    }
    assert len(vectors) == len(record["actions"])
    for vector in vectors:
        assert len(vector) == FEATURE_SIZE
