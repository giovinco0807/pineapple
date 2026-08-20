"""The runner's decisions and file handling, checked without the native engine.

`next_step` is the whole control flow: seed the field, keep measuring, or stop.
`write_batch` is the crash safety. Both are exercised here; the engine call
between them is not, because it needs the DLL and has no logic in it.
"""

from __future__ import annotations

import json
import pathlib

from ofc_regular.hu_t0_elimination_runner_v1 import (
    batch_path,
    existing,
    next_step,
    observation_dict,
    placement_label,
    write_batch,
)
from ofc_regular.hu_t0_sequential_elimination_v1 import Batch


def batch(samples: int = 1024, **scores: float) -> Batch:
    return Batch(samples=samples, scores=dict(scores))


def test_no_batches_means_seed_the_field() -> None:
    state, alive = next_step([], budget=40.0)
    assert state == "seed"
    assert alive == []


def test_a_decided_hand_stops() -> None:
    state, alive = next_step([batch(a=5.0, b=0.0)], budget=40.0)
    assert state == "converged"
    assert alive == ["a"]


def test_an_open_hand_returns_who_to_measure_next() -> None:
    state, alive = next_step([batch(a=1.0, b=0.6, c=-9.0)], budget=40.0)
    assert state == "open"
    assert set(alive) == {"a", "b"}      # c is 10 points back, far past 1.90


def test_budget_stops_a_hand_the_particles_cannot_separate() -> None:
    close = [batch(a=0.05, b=0.0)] * 4
    assert next_step(close, budget=40.0)[0] == "open"
    assert next_step(close, budget=4.0)[0] == "budget"


def test_write_batch_leaves_no_partial_behind(tmp_path: pathlib.Path) -> None:
    target = batch_path(tmp_path, 7, 3)
    write_batch(target, {"samples": 1024, "scores": {"a": 1.0}})
    assert target.name == "h007_b03.json"
    assert target.is_file()
    assert not list(tmp_path.glob("*.partial"))


def test_write_batch_replaces_an_earlier_file(tmp_path: pathlib.Path) -> None:
    target = batch_path(tmp_path, 0, 0)
    write_batch(target, {"samples": 1024, "scores": {"a": 1.0}})
    write_batch(target, {"samples": 4096, "scores": {"a": 2.0}})
    assert json.loads(target.read_text(encoding="utf-8"))["samples"] == 4096


def test_existing_reads_back_what_was_written(tmp_path: pathlib.Path) -> None:
    write_batch(batch_path(tmp_path, 4, 0),
                {"samples": 1024, "runs": [{"scores": {"a": 1.0}}]})
    write_batch(batch_path(tmp_path, 4, 1),
                {"samples": 4096, "runs": [{"scores": {"a": 2.0}}]})
    write_batch(batch_path(tmp_path, 5, 0),
                {"samples": 1024, "runs": [{"scores": {"z": 9.0}}]})

    batches = existing(tmp_path, 4)
    assert [b["samples"] for b in batches] == [1024, 4096]
    assert all("z" not in b["scores"] for b in batches)


def test_existing_orders_batches_by_number(tmp_path: pathlib.Path) -> None:
    """Ten must not sort before two; the filenames are zero-padded for this."""

    for number in (0, 2, 10):
        write_batch(batch_path(tmp_path, 1, number),
                    {"samples": 1024 * (number + 1), "scores": {"a": 0.0}})
    assert [b["samples"] for b in existing(tmp_path, 1)] == [1024, 3072, 11264]


def test_observation_is_the_five_cards_and_nothing_else() -> None:
    observation = observation_dict(["Ah", "Kc", "Qs", "8h", "7c"])
    assert observation["street"] == "T0" and observation["seat"] == "first"
    assert observation["dealt_cards"] == ["Ah", "Kc", "Qs", "8h", "7c"]
    for board in ("hero_board", "opponent_public_board"):
        assert all(not rows for rows in observation[board].values())
    assert observation["hero_private_discards"] == []
    assert observation["scoring"]["fl_ev"] == {"14": 9.6}


def test_placement_label_sorts_each_row_high_to_low() -> None:
    label = placement_label([["7c", "middle"], ["Kc", "middle"],
                             ["8h", "bottom"], ["Ah", "bottom"], ["Qs", "top"]])
    assert label == "Qs / Kc 7c / Ah 8h"


def test_placement_label_marks_an_empty_row() -> None:
    assert placement_label([["3c", "middle"]]) == "- / 3c / -"
