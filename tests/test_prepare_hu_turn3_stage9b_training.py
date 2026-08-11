from ofc_regular.prepare_hu_turn3_stage9b_training import build_stage9b_samples


def _sample(
    state_id: str,
    baseline_index: int,
    hu_index: int,
    *,
    source: str = "base",
    source_input_path: str = "input-a",
):
    return {
        "source": source,
        "source_state": {"state_id": state_id, "source_input_path": source_input_path},
        "selection": {"baseline_index": baseline_index, "hu_index": hu_index},
        "actions": [
            {"original_index": baseline_index, "score": 1.0},
            {"original_index": hu_index, "score": 0.0},
        ],
    }


def test_build_stage9b_samples_tags_runtime_hard_negatives():
    base = [_sample("base-1", 0, 1)]
    runtime = [
        _sample("bad-1", 3, 4, source="runtime"),
        _sample("ok-1", 5, 6, source="runtime"),
    ]
    hard_rows = [{"state_id": "bad-1", "baseline_index": 3, "hu_index": 4}]

    samples, summary = build_stage9b_samples(
        base_samples=base,
        runtime_samples=runtime,
        hard_negative_rows=hard_rows,
    )

    assert len(samples) == 3
    assert summary["runtime_hard_negative_samples"] == 1
    assert summary["runtime_fire_non_negative_samples"] == 1
    assert summary["source_counts"]["stage9b_runtime_hard_negative"] == 1
    assert summary["source_counts"]["stage9b_runtime_fire_non_negative"] == 1
    assert samples[1]["source"] == "stage9b_runtime_hard_negative"
    assert samples[2]["source"] == "stage9b_runtime_fire_non_negative"


def test_build_stage9b_samples_can_dedupe_base_samples():
    base = [_sample("base-1", 0, 1), _sample("base-1", 0, 1)]

    samples, summary = build_stage9b_samples(
        base_samples=base,
        runtime_samples=[],
        hard_negative_rows=[],
    )

    assert len(samples) == 1
    assert summary["base_samples"] == 2
    assert summary["output_samples"] == 1


def test_build_stage9b_samples_keeps_same_state_id_from_different_inputs():
    base = [
        _sample("base-1", 0, 1, source_input_path="input-a"),
        _sample("base-1", 0, 1, source_input_path="input-b"),
    ]

    samples, summary = build_stage9b_samples(
        base_samples=base,
        runtime_samples=[],
        hard_negative_rows=[],
    )

    assert len(samples) == 2
    assert summary["output_samples"] == 2
