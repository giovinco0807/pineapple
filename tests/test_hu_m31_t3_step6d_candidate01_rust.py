from __future__ import annotations

import struct

from ofc_regular.action_key import ActionKey
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_late_street_teacher import T4SearchConfig
from ofc_regular.hu_m3_rust import (
    build_native_engine,
    evaluate_t4,
    load_native_engine,
)
from ofc_regular.state import Board


def _rounding_sensitive_t4_first_observation() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("6s", "Qh", "Ts"),
            middle=("9c", "3c", "8d"),
            bottom=("Ac", "5h", "5d", "8c", "Td"),
        ),
        opponent_public_board=Board.from_rows(
            top=("4d", "8s"),
            middle=("3s", "3d", "Js", "7d"),
            bottom=("Tc", "9s", "Qc", "8h", "Jh"),
        ),
        dealt_cards=("Kd", "9h", "2h"),
        hero_private_discards=("4h", "Ks", "Jc"),
        seat="first",
        street="T4",
        to_act_order="first",
    )


def _f64_bits(value: object) -> str:
    return struct.pack(">d", float(value)).hex()


def test_candidate01_exact_t4_preserves_every_q_bit_and_actionkey_tie_order() -> None:
    build = build_native_engine(release=False)
    library = load_native_engine(path=build.library_path)
    result = evaluate_t4(
        _rounding_sensitive_t4_first_observation(),
        config=T4SearchConfig(
            candidate_samples=0,
            evaluation_samples=0,
            seed=617,
            run_id="candidate01-rounding-order",
        ),
        library=library,
    )

    expected = {
        "rak1:0000000000000:0000000000081:0000000000000:0000001000000": (
            2,
            "c020800000000000",
        ),
        "rak1:0000000000000:0000001000001:0000000000000:0000000000080": (
            1,
            "c018db92b828796c",
        ),
        "rak1:0000000000000:0000001000080:0000000000000:0000000000001": (
            0,
            "c020800000000000",
        ),
    }
    observed = {
        row["action_key"]: (
            row["original_index"],
            _f64_bits(row["selection_score"]),
            _f64_bits(row["score"]),
        )
        for row in result["actions"]
    }
    assert observed == {
        key: (original_index, bits, bits)
        for key, (original_index, bits) in expected.items()
    }
    assert result["selected_action_key"] == (
        "rak1:0000000000000:0000001000001:0000000000000:0000000000080"
    )

    tied = [
        row
        for row in result["actions"]
        if _f64_bits(row["selection_score"]) == "c020800000000000"
    ]
    assert [row["action_key"] for row in tied] == [
        key
        for key in sorted(
            (row["action_key"] for row in tied),
            key=lambda token: ActionKey.from_token(token).sort_key(),
        )
    ]
