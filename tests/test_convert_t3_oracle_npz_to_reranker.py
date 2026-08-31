from argparse import Namespace

import numpy as np

from ai.engine.action_space import create_regular_turn_mask
from ai.engine.encoding import Board, Observation, encode_state
from ai.training.action_feature_encoding import ACTION_FEATURE_DIM
from ai.training.convert_t3_oracle_npz_to_reranker import convert, decode_observation


def test_decode_observation_round_trips_card_locations() -> None:
    obs = Observation(
        board_self=Board(top=["Ah"], middle=["2h", "3d"], bottom=["4c", "5s"]),
        board_opponent=Board(top=["Kd"], middle=["6h"], bottom=["7c", "8s"]),
        dealt_cards=["Qh", "Jd", "Ts"],
        known_discards_self=["9c"],
        turn=3,
        is_btn=False,
    )

    decoded = decode_observation(encode_state(obs))

    assert decoded.turn == 3
    assert decoded.is_btn is False
    assert decoded.board_self.to_dict() == obs.board_self.to_dict()
    assert decoded.board_opponent.to_dict() == obs.board_opponent.to_dict()
    assert decoded.dealt_cards == obs.dealt_cards
    assert decoded.known_discards_self == obs.known_discards_self


def test_convert_t3_oracle_npz_writes_candidate_reranker_arrays(tmp_path) -> None:
    obs = Observation(
        board_self=Board(
            top=["Ah"],
            middle=["2h", "3d", "4c"],
            bottom=["5h", "6d", "7c", "8s"],
        ),
        board_opponent=Board(top=["Kd"], middle=["9h", "9d"], bottom=["Tc", "Jc"]),
        dealt_cards=["Qh", "Jd", "Ts"],
        known_discards_self=["9c"],
        turn=3,
        is_btn=True,
    )
    mask = create_regular_turn_mask(obs.dealt_cards, obs.board_self)
    action_evs = np.full((1, 27), -10000.0, dtype=np.float16)
    valid_ids = np.flatnonzero(mask)
    action_evs[0, valid_ids] = np.linspace(1.0, 3.0, num=len(valid_ids), dtype=np.float16)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    np.savez(
        input_dir / "chunk_0.npz",
        states=np.asarray([encode_state(obs)], dtype=np.float16),
        action_evs=action_evs,
        valid_masks=np.asarray([mask], dtype=bool),
    )

    convert(
        Namespace(
            input=str(input_dir),
            output=str(output_dir),
            state_dim=520 + ACTION_FEATURE_DIM,
            limit_records=0,
            decode_threshold=0.5,
            teacher_best_weight=0.5,
            gap_weight=0.5,
            max_sample_weight=4.0,
            progress_every=1000,
        )
    )

    states = np.load(output_dir / "states.npy")
    scores = np.load(output_dir / "scores.npy")
    action_indices = np.load(output_dir / "action_indices.npy")
    candidate_ranks = np.load(output_dir / "candidate_ranks.npy")
    metadata = __import__("json").load((output_dir / "metadata.json").open("r", encoding="utf-8"))

    assert states.shape == (len(valid_ids), 520 + ACTION_FEATURE_DIM)
    assert len(scores) == len(valid_ids)
    assert set(action_indices.tolist()) == set(valid_ids.tolist())
    assert candidate_ranks.min() == 0
    assert states[0, 520:547].sum() == 1.0
    assert metadata["n_records"] == 1
    assert metadata["n_samples"] == len(valid_ids)
