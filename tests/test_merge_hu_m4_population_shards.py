from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular.evaluate_hu_m4_population import (
    M4_HAND_RECORD_SCHEMA,
    M4_OPPONENT_PROFILES,
    summarize_hu_m4_population_records,
)
from ofc_regular.merge_hu_m4_population_shards import (
    M43_POPULATION_PLAN_SCHEMA,
    load_population_plan,
    main,
    merge_population_shards,
)


def _plan() -> dict[str, object]:
    return {
        "schema": M43_POPULATION_PLAN_SCHEMA,
        "status": "frozen_before_population_evaluation",
        "opponents": list(M4_OPPONENT_PROFILES),
        "paired_seeds_per_opponent": 4,
        "seed": 1000,
        "seed_stride": 101,
        "shards": 2,
        "paired_seeds_per_shard": 2,
        "minimum_valid_overrides": 300,
        "activation_guards": {
            "current_profile_changed": False,
            "runtime_policy_activated": False,
            "full_replacement_enabled": False,
            "threshold_changed_after_lock": False,
        },
        "freshness": {
            "excluded_population_schedules": [
                {"seed": 9000, "seed_stride": 101, "paired_seeds": 2}
            ]
        },
    }


def _runtime() -> dict[str, object]:
    return {
        "baseline_profile": "stage18_p1",
        "candidate_model": "model.pkl",
        "candidate_model_sha256": "a" * 64,
        "safety_model": "model.pkl",
        "safety_model_sha256": "a" * 64,
        "safety_threshold": 0.8,
        "opponents": list(M4_OPPONENT_PROFILES),
        "current_profile_used": False,
        "promotion_artifact_contract": True,
        "diagnostic_legacy": False,
    }


def _records(seed_start: int, count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for opponent_index, opponent in enumerate(M4_OPPONENT_PROFILES):
        for index in range(count):
            hand_seed = seed_start + index * 101
            for seat in ("first", "second"):
                fired = seat == "second"
                delta = float(1 + opponent_index) if fired else 0.0
                rows.append(
                    {
                        "schema": M4_HAND_RECORD_SCHEMA,
                        "opponent": opponent,
                        "seed": hand_seed,
                        "seat": seat,
                        "hero_policy_seed": hand_seed * 4 + (seat == "second"),
                        "opponent_policy_seed": hand_seed * 4 + (seat == "first"),
                        "candidate_score": 2.0 + delta,
                        "baseline_score": 2.0,
                        "delta": delta,
                        "override_log_valid": True,
                        "override_fired": fired,
                        "candidate_gameplay_digest": (
                            ("b" if fired else "a") * 64
                        ),
                        "baseline_gameplay_digest": "a" * 64,
                        "nonfire_cancellation_valid": None if fired else True,
                        "counterfactual_basis": (
                            "same_seed_physical_seat_opponent_policy_seeds_v1"
                        ),
                    }
                )
    return rows


def _shards(tmp_path: Path):
    evaluations = []
    records = []
    for shard, seed_start in enumerate((1000, 1202)):
        shard_records = _records(seed_start, 2)
        summary = summarize_hu_m4_population_records(
            shard_records,
            opponents=M4_OPPONENT_PROFILES,
            paired_seeds=2,
            seed=seed_start,
            seed_stride=101,
            elapsed_seconds=5.0 + shard,
        )
        summary["runtime_config"] = _runtime()
        evaluation_path = tmp_path / f"evaluation_{shard}.json"
        records_path = tmp_path / f"records_{shard}.jsonl"
        evaluation_path.write_text(json.dumps(summary), encoding="utf-8")
        records_path.write_text(
            "".join(json.dumps(row) + "\n" for row in shard_records),
            encoding="utf-8",
        )
        evaluations.append((evaluation_path, summary))
        records.append((records_path, shard_records))
    return evaluations, records


def test_merge_recomputes_one_complete_seed_clustered_population(tmp_path: Path) -> None:
    evaluations, records = _shards(tmp_path)

    final, merge_manifest, merged_records = merge_population_shards(
        plan=_plan(),
        shard_evaluations=evaluations,
        shard_records=records,
        plan_sha256="b" * 64,
        records_output=tmp_path / "merged.jsonl",
    )

    assert len(merged_records) == 32
    assert final["paired_seeds_per_opponent"] == 4
    assert final["population"]["paired_seat_swap"]["delta_ev_per_hand"]["n"] == 4
    assert final["population"]["by_seat"]["first"]["delta_ev_per_hand"][
        "mean"
    ] == 0.0
    assert final["runtime_config"]["current_profile_used"] is False
    assert final["runtime_config"]["final_metrics_recomputed_from_merged_records"] is True
    assert merge_manifest["metrics_recomputed_from_merged_seed_clusters"] is True


@pytest.mark.parametrize(
    "mutation",
    ["summary", "records", "runtime", "current", "missing"],
)
def test_merge_rejects_malformed_incomplete_or_mixed_shards(
    tmp_path: Path, mutation: str
) -> None:
    evaluations, records = _shards(tmp_path)
    if mutation == "summary":
        evaluations[0][1]["population"]["all_seats"]["overrides"] += 1
    elif mutation == "records":
        records[0][1].append(deepcopy(records[0][1][0]))
    elif mutation == "runtime":
        evaluations[1][1]["runtime_config"]["safety_threshold"] = 0.9
    elif mutation == "current":
        evaluations[0][1]["runtime_config"]["current_profile_used"] = True
    elif mutation == "missing":
        evaluations.pop()
        records.pop()

    with pytest.raises(ValueError):
        merge_population_shards(
            plan=_plan(),
            shard_evaluations=evaluations,
            shard_records=records,
            plan_sha256="b" * 64,
            records_output=tmp_path / "merged.jsonl",
        )


def test_plan_loader_rejects_current_or_lowered_override_gate(tmp_path: Path) -> None:
    plan = _plan()
    path = tmp_path / "plan.json"
    plan["minimum_valid_overrides"] = 299
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="300"):
        load_population_plan(path)

    plan = _plan()
    plan["opponents"] = ["current", *M4_OPPONENT_PROFILES[1:]]
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="opponent"):
        load_population_plan(path)


@pytest.mark.parametrize(
    "guards",
    [
        {},
        {"current_profile_changed": False},
        {
            "current_profile_changed": False,
            "runtime_policy_activated": False,
            "full_replacement_enabled": False,
            "threshold_changed_after_lock": False,
            "unexpected_fail_open_guard": False,
        },
    ],
)
def test_plan_loader_rejects_missing_activation_guards(
    tmp_path: Path, guards: dict[str, bool]
) -> None:
    plan = _plan()
    plan["activation_guards"] = guards
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="activation guards"):
        load_population_plan(path)


def test_plan_loader_rejects_missing_or_overlapping_prior_schedule(
    tmp_path: Path,
) -> None:
    plan = _plan()
    plan.pop("freshness")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="prior population exclusions"):
        load_population_plan(path)

    plan = _plan()
    plan["freshness"] = {
        "excluded_population_schedules": [
            {"seed": 1000, "seed_stride": 101, "paired_seeds": 1}
        ]
    }
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps"):
        load_population_plan(path)


def test_merge_rejects_forged_nonfire_digest_or_delta(tmp_path: Path) -> None:
    evaluations, records = _shards(tmp_path)
    forged = records[0][1][0]
    forged["candidate_gameplay_digest"] = "c" * 64
    assert forged["nonfire_cancellation_valid"] is True
    with pytest.raises(ValueError, match="cancellation claim"):
        merge_population_shards(
            plan=_plan(),
            shard_evaluations=evaluations,
            shard_records=records,
            plan_sha256="b" * 64,
            records_output=tmp_path / "merged.jsonl",
        )

    evaluations, records = _shards(tmp_path)
    records[0][1][1]["delta"] = 999.0
    with pytest.raises(ValueError, match="terminal scores"):
        merge_population_shards(
            plan=_plan(),
            shard_evaluations=evaluations,
            shard_records=records,
            plan_sha256="b" * 64,
            records_output=tmp_path / "merged.jsonl",
        )


def test_cli_writes_hash_chained_merge_outputs(tmp_path: Path) -> None:
    evaluations, records = _shards(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
    output = tmp_path / "evaluation.json"
    records_output = tmp_path / "merged.jsonl"
    manifest_output = tmp_path / "merge_manifest.json"
    args = ["--plan", str(plan_path)]
    for path, _payload in evaluations:
        args.extend(("--shard-evaluation", str(path)))
    for path, _payload in records:
        args.extend(("--shard-records", str(path)))
    args.extend(
        (
            "--records-output",
            str(records_output),
            "--output",
            str(output),
            "--merge-manifest-output",
            str(manifest_output),
        )
    )

    assert main(args) == 0
    merged = json.loads(output.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
    assert merged["paired_seeds_per_opponent"] == 4
    assert manifest["status"] == "complete_content_verified"
    assert len(manifest["evaluation_sha256"]) == 64
    assert len(manifest["merged_records_sha256"]) == 64
