import json
from itertools import combinations, islice

from ofc_regular.prepare_hu_turn0_candidate_dataset import (
    MC1_SOURCE,
    MC4_SOURCE,
    canonical_t0_state_key,
    prepare_hu_turn0_candidate_dataset,
)


def _sample(index: int, seat: str, source: str) -> dict:
    deck = [f"{rank}{suit}" for rank in "23456789TJQKA" for suit in "cdhs"]
    dealt = list(next(islice(combinations(deck, 5), index, index + 1)))
    return {
        "schema": "hu_turn0_stage1_teacher_v1",
        "phase": "hu_turn0_0card",
        "sample_id": index,
        "hand_seed": 1000 + index,
        "player": 0 if seat == "first" else 1,
        "seat": seat,
        "source": source,
        "board": {"top": [], "middle": [], "bottom": []},
        "opponent_board": {
            "top": [f"{2 + index % 7}h"] if seat == "second" else [],
            "middle": [],
            "bottom": [],
        },
        "dead_cards": [],
        "dealt": dealt,
        "actions": [{"score": 1.0, "ev": 1.0, "se": 0.1}],
    }


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_canonical_t0_state_key_ignores_card_order():
    left = _sample(1, "first", "old")
    right = json.loads(json.dumps(left))
    right["dealt"] = list(reversed(right["dealt"]))

    assert canonical_t0_state_key(left) == canonical_t0_state_key(right)


def test_prepare_candidate_dataset_has_leak_free_seat_balanced_splits(tmp_path):
    mc1 = [_sample(index, "first" if index % 2 == 0 else "second", "legacy") for index in range(20)]
    mc4 = [
        _sample(100 + index, "first" if index % 2 == 0 else "second", "legacy")
        for index in range(10)
    ]
    mc1_path = tmp_path / "mc1.jsonl"
    mc4_path = tmp_path / "mc4.jsonl"
    _write(mc1_path, mc1)
    _write(mc4_path, mc4)

    summary = prepare_hu_turn0_candidate_dataset(
        mc1_input=mc1_path,
        mc4_input=mc4_path,
        output_dir=tmp_path / "prepared",
        mc1_holdout_fraction=0.2,
        mc4_holdout_fraction=0.2,
        seed=42,
    )

    assert summary["train"]["rows"] == 24
    assert summary["holdout_mc1"]["rows"] == 4
    assert summary["holdout_mc4"]["rows"] == 2
    assert summary["holdout_mc1"]["seats"] == {"first": 2, "second": 2}
    assert summary["holdout_mc4"]["seats"] == {"first": 1, "second": 1}
    assert summary["split_overlap"] == {
        "train_vs_mc1": 0,
        "train_vs_mc4": 0,
        "mc1_vs_mc4": 0,
    }

    train_rows = [
        json.loads(line)
        for line in (tmp_path / "prepared" / "train_mixed_mc1_mc4.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {row["source"] for row in train_rows} == {MC1_SOURCE, MC4_SOURCE}
    assert {row["teacher_mc_samples"] for row in train_rows} == {1, 4}
    assert all(row["dataset_split"] == "train" for row in train_rows)


def test_prepare_candidate_dataset_prefers_duplicate_mc4_state(tmp_path):
    duplicate_mc1 = _sample(1, "first", "old_mc1")
    duplicate_mc4 = json.loads(json.dumps(duplicate_mc1))
    duplicate_mc4["source"] = "old_mc4"
    mc1_rows = [duplicate_mc1, _sample(2, "second", "old_mc1")]
    mc4_rows = [duplicate_mc4, _sample(3, "second", "old_mc4")]
    mc1_path = tmp_path / "mc1.jsonl"
    mc4_path = tmp_path / "mc4.jsonl"
    _write(mc1_path, mc1_rows)
    _write(mc4_path, mc4_rows)

    summary = prepare_hu_turn0_candidate_dataset(
        mc1_input=mc1_path,
        mc4_input=mc4_path,
        output_dir=tmp_path / "prepared",
        mc1_holdout_fraction=0.0,
        mc4_holdout_fraction=0.0,
        seed=7,
    )

    assert summary["duplicate_mc1_states_dropped"] == 1
    assert summary["train"]["rows"] == 3
