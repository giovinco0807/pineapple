import json

import pytest

from ofc_regular.aggregate_hu_turn0_counterfactual import aggregate_shards


def _write_shard(tmp_path, name, *, config_id, seed, first_delta, second_delta):
    shard = tmp_path / name
    shard.mkdir()
    events = [
        {
            "event_id": f"{seed}:first",
            "config_id": config_id,
            "seed": seed,
            "seat": "first",
            "realized_delta": first_delta,
            "override_fired": first_delta != 0.0,
            "no_override_reason": "" if first_delta != 0.0 else "same_as_baseline",
        },
        {
            "event_id": f"{seed}:second",
            "config_id": config_id,
            "seed": seed,
            "seat": "second",
            "realized_delta": second_delta,
            "override_fired": second_delta != 0.0,
            "no_override_reason": "" if second_delta != 0.0 else "same_as_baseline",
        },
    ]
    (shard / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )
    (shard / "summary.json").write_text(
        json.dumps(
            {
                "config_id": config_id,
                "candidate_model": "model.pkl",
                "candidate_topk": 60,
                "safe_selector_model": "selector.pkl",
                "safe_selector_threshold_by_seat": {"first": 0.9, "second": 1.0},
                "min_margin_by_seat": {"first": 1.0, "second": 1.0},
                "allowed_seats": ["first", "second"],
                "profile": "stage18_p1",
                "seed_stride": 1000003,
                "paired_seeds": 1,
                "elapsed_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return shard / "events.jsonl", shard / "summary.json"


def test_aggregate_uses_seed_paired_ci_and_preserves_seats(tmp_path):
    shards = [
        _write_shard(tmp_path, "a", config_id="m1", seed=10, first_delta=2.0, second_delta=0.0),
        _write_shard(tmp_path, "b", config_id="m1", seed=20, first_delta=0.0, second_delta=-2.0),
    ]

    summary, events, seed_rows = aggregate_shards(
        shards, expected_paired_seeds_per_config=2
    )

    result = summary["config_results"][0]
    assert result["paired_seeds"] == 2
    assert result["events"] == 4
    assert result["avg_delta_per_hand"] == 0.0
    assert result["fires"] == 2
    assert result["safe_selector_model"] == "selector.pkl"
    assert result["safe_selector_threshold_by_seat"] == {"first": 0.9, "second": 1.0}
    assert result["by_seat"]["first"]["fires"] == 1
    assert result["by_seat"]["second"]["fires"] == 1
    assert len(events) == 4
    assert [row["paired_delta"] for row in seed_rows] == [1.0, -1.0]


def test_aggregate_rejects_duplicate_events(tmp_path):
    shard = _write_shard(
        tmp_path, "a", config_id="m1", seed=10, first_delta=1.0, second_delta=0.0
    )

    with pytest.raises(ValueError, match="duplicate counterfactual events"):
        aggregate_shards([shard, shard])
