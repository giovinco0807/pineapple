import json
from pathlib import Path

import pytest

from ofc_regular.aggregate_hu_turn0_pilot import aggregate_hu_turn0_pilot


def _write_shard(root: Path, prefix: str, *, seat: str, regret: float) -> None:
    shard = root / prefix
    shard.mkdir(parents=True)
    (shard / "DONE").write_text("complete\n", encoding="utf-8")
    row = {
        "sample_id": 0,
        "seat": seat,
        "score_gap": 0.5,
        "delta_best_vs_baseline": regret,
        "total_legal_actions": 232,
        "evaluated_action_count": 2,
        "common_random_futures_verified": True,
        "replay_ready": True,
        "actions": [
            {"ev": 2.0, "score": 2.0, "se": 0.25, "rollout_count": 4},
            {"ev": 1.0, "score": 1.0, "se": 0.25, "rollout_count": 4},
        ],
    }
    (shard / "teacher.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary = {
        "future_samples": 4,
        "seconds_per_sample": 3.0,
        "profile_stats": {
            "action_eval_seconds": 2.5,
            "continuation_T2_seconds": 1.5,
            "raw_rollout_count": 8,
        },
    }
    (shard / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_aggregate_hu_turn0_pilot_validates_and_globalizes_rows(tmp_path):
    results = tmp_path / "results"
    _write_shard(results, "s0", seat="first", regret=1.0)
    _write_shard(results, "s1", seat="second", regret=3.0)
    manifest = tmp_path / "shards.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"shard": 0, "samples": 1, "output_prefix": "s0"}),
                json.dumps({"shard": 1, "samples": 1, "output_prefix": "s1"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = aggregate_hu_turn0_pilot(
        results_root=results,
        shard_manifest=manifest,
        output_dir=tmp_path / "aggregate",
    )

    assert summary["records"] == 2
    assert summary["expected_records"] == 2
    assert summary["completed_shards"] == 2
    assert summary["missing_shards"] == []
    assert summary["seat_counts"] == {"first": 1, "second": 1}
    assert summary["mean_legal_actions"] == 232
    assert summary["baseline_regret_mean"] == 2.0
    assert summary["non_finite_actions"] == 0
    assert summary["wrong_rollout_counts"] == 0
    assert summary["common_random_future_failures"] == 0
    assert summary["replay_not_ready"] == 0
    assert summary["profile_totals"]["raw_rollout_count"] == 16

    rows = [
        json.loads(line)
        for line in Path(summary["merged_output"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sample_id"] for row in rows] == [0, 1_000_000]
    assert [row["aggregate_shard"] for row in rows] == [0, 1]


def test_aggregate_hu_turn0_pilot_rejects_missing_shard(tmp_path):
    manifest = tmp_path / "shards.jsonl"
    manifest.write_text(
        json.dumps({"shard": 0, "samples": 1, "output_prefix": "missing"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Missing HU T0"):
        aggregate_hu_turn0_pilot(
            results_root=tmp_path / "results",
            shard_manifest=manifest,
            output_dir=tmp_path / "aggregate",
        )
