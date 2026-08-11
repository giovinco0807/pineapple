import json

import pytest

from ofc_regular.aggregate_hu_turn1_pilot import aggregate_hu_turn1_pilot


def _write_shard(root, shard, *, done=True, seconds=1.0, records=1):
    name = f"shard_{shard:03d}"
    shard_dir = root / name
    shard_dir.mkdir(parents=True)
    if done:
        (shard_dir / "DONE").write_text("", encoding="utf-8")
    with (shard_dir / "teacher.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(records):
            handle.write(json.dumps({"shard": shard, "row": index}) + "\n")
    summary = {
        "elapsed_seconds": seconds,
        "seconds_per_sample": seconds / records,
        "mean_action_count": 24 + shard,
        "topk_decisions": 10 + shard,
        "topk_overrides": shard,
        "duplicate_profile_stats": {
            "t2_state_only": {
                "raw": 10 + shard,
                "unique": 8 + shard,
                "repeated": 2,
                "duplicate_rate": 2 / (10 + shard),
                "max_occurrence": 2,
            },
            "t2_state_plus_decision_seed": {
                "raw": 10 + shard,
                "unique": 10 + shard,
                "repeated": 0,
                "duplicate_rate": 0.0,
                "max_occurrence": 1,
            },
        },
        "profile_stats": {
            "choose_action_T2_seconds": seconds * 0.8,
            "rollout_seconds": seconds * 0.9,
        },
    }
    (shard_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return name


def test_aggregate_hu_turn1_pilot_merges_completed_shards(tmp_path):
    results = tmp_path / "results"
    output = tmp_path / "out"
    first = _write_shard(results, 0, seconds=2.0, records=2)
    second = _write_shard(results, 1, seconds=4.0, records=1)
    manifest = tmp_path / "shards_manifest.jsonl"
    manifest.write_text(
        json.dumps({"shard": 0, "output_prefix": first})
        + "\n"
        + json.dumps({"shard": 1, "output_prefix": second})
        + "\n",
        encoding="utf-8",
    )

    summary = aggregate_hu_turn1_pilot(
        results_root=results,
        shard_manifest=manifest,
        output_dir=output,
    )
    records = [
        json.loads(line)
        for line in (output / "hu_turn1_stage1_pilot.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["records"] == 3
    assert summary["completed_shards"] == 2
    assert summary["missing_shards"] == []
    assert summary["sample_id_globalized"] is True
    assert summary["topk_decisions"] == 21
    assert summary["topk_overrides"] == 1
    assert summary["duplicate_profile_stats"]["t2_state_only"]["raw"] == 21
    assert summary["duplicate_profile_stats"]["t2_state_only"]["repeated"] == 4
    assert summary["duplicate_profile_stats"]["t2_state_plus_decision_seed"]["repeated"] == 0
    assert (output / "hu_turn1_stage1_pilot.jsonl").read_text(encoding="utf-8").count("\n") == 3
    assert (output / "summary.json").exists()
    assert [record["sample_id"] for record in records] == [0, 1, 1_000_000]
    assert [record["local_sample_id"] for record in records] == [0, 1, 0]
    assert [record["aggregate_shard"] for record in records] == [0, 0, 1]


def test_aggregate_hu_turn1_pilot_rejects_missing_by_default(tmp_path):
    results = tmp_path / "results"
    output = tmp_path / "out"
    completed = _write_shard(results, 0)
    missing = _write_shard(results, 1, done=False)
    manifest = tmp_path / "shards_manifest.jsonl"
    manifest.write_text(
        json.dumps({"shard": 0, "output_prefix": completed})
        + "\n"
        + json.dumps({"shard": 1, "output_prefix": missing})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Missing HU T1 pilot result shards"):
        aggregate_hu_turn1_pilot(
            results_root=results,
            shard_manifest=manifest,
            output_dir=output,
        )
