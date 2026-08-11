import json

from ofc_regular.collect_hu_turn2_stage8b_loss_targets import (
    DOWNSTREAM_TRAJECTORY_FIELDS,
    collect_rows,
    downstream_complete,
    downstream_coverage_rows,
    main,
    split_rows,
    summary_rows,
)


def _target(**overrides):
    row = {
        "schema": "hu_turn2_stage8b_counterfactual_loss_target_v1",
        "source_log": "outputs\\run\\runtime_decisions.jsonl",
        "config_id": "cfg",
        "hand_seed": 123,
        "seat": "first",
        "candidate_index": 7,
        "baseline_index": 2,
        "realized_delta": -6.0,
        "recommended_training_use": "whole_game_risk_only",
        "local_replay_bucket": "local_positive_lcb",
        "local_replay_future_samples": 128,
        "use_for_local_ev_hard_negative": 0,
        "use_for_whole_game_risk_head": 1,
        "downstream_trajectory_complete": 1,
        "downstream_trajectory_present_fields": 16,
        "downstream_trajectory_total_fields": 16,
    }
    row.update(overrides)
    return row


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_collection_dedupes_and_prefers_higher_mc_replay(tmp_path):
    low = tmp_path / "low.jsonl"
    high = tmp_path / "high.jsonl"
    _write_jsonl(low, [_target(local_replay_future_samples=128, recommended_training_use="whole_game_risk_only")])
    _write_jsonl(
        high,
        [
            _target(
                local_replay_future_samples=512,
                recommended_training_use="local_ev_hard_negative",
                local_replay_bucket="local_negative",
                use_for_local_ev_hard_negative=1,
                use_for_whole_game_risk_head=0,
            )
        ],
    )

    rows, sources = collect_rows([low, high])

    assert len(rows) == 1
    assert len(sources) == 2
    assert rows[0]["local_replay_future_samples"] == 512
    assert rows[0]["recommended_training_use"] == "local_ev_hard_negative"


def test_collection_splits_local_hard_negatives_from_whole_game_risk(tmp_path):
    path = tmp_path / "targets.jsonl"
    _write_jsonl(
        path,
        [
            _target(hand_seed=1, recommended_training_use="whole_game_risk_only"),
            _target(
                hand_seed=2,
                recommended_training_use="local_ev_hard_negative",
                local_replay_bucket="local_negative",
                use_for_local_ev_hard_negative=1,
                use_for_whole_game_risk_head=0,
            ),
            _target(hand_seed=3, recommended_training_use="requires_local_replay", local_replay_bucket="missing"),
            _target(
                hand_seed=4,
                realized_delta=2.0,
                recommended_training_use="whole_game_non_loss_control",
                use_for_whole_game_risk_head=1,
            ),
        ],
    )

    rows, _sources = collect_rows([path])
    groups = split_rows(rows)

    assert len(groups["whole_game_risk_only"]) == 1
    assert len(groups["whole_game_non_loss_control"]) == 1
    assert len(groups["local_ev_hard_negative"]) == 1
    assert len(groups["requires_local_replay"]) == 1
    assert groups["whole_game_risk_only"][0]["use_for_local_ev_hard_negative"] == 0


def test_collection_summary_counts_uses_and_buckets(tmp_path):
    path = tmp_path / "targets.jsonl"
    _write_jsonl(
        path,
        [
            _target(hand_seed=1, realized_delta=-6.0, recommended_training_use="whole_game_risk_only"),
            _target(
                hand_seed=2,
                realized_delta=-2.0,
                recommended_training_use="local_ev_hard_negative",
                local_replay_bucket="local_negative",
            ),
            _target(
                hand_seed=3,
                realized_delta=3.0,
                recommended_training_use="whole_game_non_loss_control",
                use_for_whole_game_risk_head=1,
            ),
        ],
    )

    rows, sources = collect_rows([path])
    summary = {row["metric"]: row["value"] for row in summary_rows(rows, sources)}

    assert summary["input_rows"] == 3
    assert summary["deduped_rows"] == 3
    assert summary["recommended_use.local_ev_hard_negative"] == 1
    assert summary["recommended_use.whole_game_risk_only"] == 1
    assert summary["recommended_use.whole_game_non_loss_control"] == 1
    assert summary["local_replay_bucket.local_negative"] == 1
    assert summary["local_replay_bucket.local_positive_lcb"] == 2
    assert summary["realized_delta_sum"] == -5.0
    assert summary["downstream_trajectory_complete_rows"] == 3
    assert summary["downstream_trajectory_incomplete_rows"] == 0


def test_collection_reports_downstream_trajectory_coverage(tmp_path):
    complete_path = tmp_path / "complete.jsonl"
    local_replay_only_path = tmp_path / "local_replay_only.jsonl"
    _write_jsonl(
        complete_path,
        [_target(hand_seed=1)],
    )
    _write_jsonl(
        local_replay_only_path,
        [
            _target(
                hand_seed=2,
                downstream_trajectory_complete=0,
                downstream_trajectory_present_fields=2,
            )
        ],
    )

    rows, sources = collect_rows([complete_path, local_replay_only_path])
    summary = {row["metric"]: row["value"] for row in summary_rows(rows, sources)}
    coverage = {
        (row["group_field"], row["group_value"]): row for row in downstream_coverage_rows(rows)
    }

    assert summary["downstream_trajectory_complete_rows"] == 1
    assert summary["downstream_trajectory_incomplete_rows"] == 1
    assert coverage[("overall", "all")]["downstream_trajectory_complete_rows"] == 1
    assert coverage[("collection_source_path", str(complete_path))]["downstream_trajectory_complete_rate"] == 1.0
    assert (
        coverage[("collection_source_path", str(local_replay_only_path))][
            "downstream_trajectory_complete_rate"
        ]
        == 0.0
    )


def test_downstream_complete_falls_back_to_concrete_trajectory_fields():
    row = _target()
    row.pop("downstream_trajectory_complete")
    row.pop("downstream_trajectory_present_fields")
    row.pop("downstream_trajectory_total_fields")
    for field in DOWNSTREAM_TRAJECTORY_FIELDS:
        row[field] = {"present": True}

    assert downstream_complete(row) == 1

    row.pop("final_board_hero")
    assert downstream_complete(row) == 0


def test_collection_manifest_reports_downstream_trajectory_coverage(tmp_path, monkeypatch):
    path = tmp_path / "targets.jsonl"
    _write_jsonl(
        path,
        [
            _target(hand_seed=1),
            _target(
                hand_seed=2,
                downstream_trajectory_complete=0,
                downstream_trajectory_present_fields=2,
            ),
        ],
    )
    output_dir = tmp_path / "collection"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--input-jsonl",
            str(path),
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    manifest = json.loads((output_dir / "topk_loss_target_collection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["deduped_rows"] == 2
    assert manifest["downstream_trajectory_complete_rows"] == 1
    assert manifest["downstream_trajectory_incomplete_rows"] == 1
    assert manifest["downstream_trajectory_complete_rate"] == 0.5
