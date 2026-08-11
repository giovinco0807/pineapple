from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import ofc_regular.validate_hu_m3_engine as validation
from ofc_regular.validate_hu_m3_engine import (
    M3_ENGINE_RESULT_SCHEMA,
    M3_PILOT_INPUT_SCHEMA,
    M3_RUNNER_RESULT_SCHEMA,
    M3_VALIDATION_SCHEMA,
    TEACHER_VALUE_STATUS,
    M3PilotConfig,
    generate_pilot_roots,
    run_m3_validation,
    validate_runner_results,
    write_runner_input,
)


ACTION_KEY = (
    "rak1:0000000000001:0000000000002:0000000000004:0000000000008"
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _fake_engine_result(row: dict) -> dict:
    request = row["request"]
    fingerprint = request["observation_fingerprint"]
    seat = request["observation"]["seat"]
    return {
        "status": "ok",
        "schema": M3_ENGINE_RESULT_SCHEMA,
        "engine_version": "test-engine",
        "kind": "t3",
        "street": "T3",
        "seat": seat,
        "to_act_order": seat,
        "observation_fingerprint": fingerprint,
        "legal_action_count": 1,
        "legal_action_set_digest": _digest("set:" + fingerprint),
        "legal_action_order_digest": _digest("order:" + fingerprint),
        "selected_action_original_index": 0,
        "selected_action_key": ACTION_KEY,
        "selected_action_evaluation_score": 1.0,
        "teacher_value_status": TEACHER_VALUE_STATUS,
        "sample_independence": "disjoint_particle_rng_keys",
        "candidate_rng_key_digests": [_digest("candidate-key:" + fingerprint)],
        "evaluation_rng_key_digests": [_digest("evaluation-key:" + fingerprint)],
        "candidate_belief": {
            "particle_digests": [_digest("candidate:" + fingerprint)]
        },
        "evaluation_belief": {
            "particle_digests": [_digest("evaluation:" + fingerprint)]
        },
        "actions": [
            {
                "original_index": 0,
                "action_key": ACTION_KEY,
                "score": 1.0,
                "selection_score": 1.25,
            }
        ],
    }


def _install_fake_runner(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, bool, int]] = []

    def fake_invoke(executable, paths, *, run_id, resume):
        input_rows = [
            json.loads(line)
            for line in paths.input.read_text(encoding="utf-8").splitlines()
        ]
        envelopes = []
        for index, row in enumerate(input_rows):
            envelopes.append(
                {
                    "schema_version": M3_RUNNER_RESULT_SCHEMA,
                    "run_id": run_id,
                    "input_index": index,
                    "root_id": row["root_id"],
                    "request_schema_version": row["schema_version"],
                    "result": _fake_engine_result(row),
                }
            )
        paths.output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in envelopes),
            encoding="utf-8",
        )
        paths.checkpoint.write_text(
            json.dumps(
                {
                    "schema_version": "hu_m3_shard_runner_v1",
                    "committed_rows": len(input_rows),
                    "complete": True,
                }
            ),
            encoding="utf-8",
        )
        paths.heartbeat.write_text(
            json.dumps(
                {
                    "schema_version": "hu_m3_shard_runner_v1",
                    "committed_rows": len(input_rows),
                    "status": "complete",
                }
            ),
            encoding="utf-8",
        )
        calls.append((run_id, resume, len(input_rows)))
        return {
            "schema_version": "hu_m3_shard_runner_v1",
            "input_rows": len(input_rows),
            "output_rows": len(input_rows),
            "resumed": resume,
        }

    monkeypatch.setattr(validation, "_invoke_runner", fake_invoke)
    return calls


def _passing_benchmark(*args, minimum_speedup=5.0, **kwargs):
    return {
        "schema": "test-parity",
        "passed": True,
        "aggregate_speedup": minimum_speedup + 1.0,
        "gates": {
            "action_key_value_parity": True,
            "aggregate_speedup_at_least_5x": True,
        },
    }


def test_root_generation_is_balanced_and_uses_seed_stride():
    config = M3PilotConfig(
        roots_100=2,
        roots_1000=4,
        seed_start=101,
        seed_stride=37,
    )
    roots = generate_pilot_roots(4, config, run_id="unit")

    assert [root.seat for root in roots] == ["first", "second"] * 2
    assert [root.hand_seed for root in roots] == [101, 101, 138, 138]
    assert len({root.fingerprint for root in roots}) == 4
    assert all(root.request["schema"] == "hu_m3_engine_request_v1" for root in roots)
    assert all(
        root.request["config"][name] == 1
        for root in roots
        for name in (
            "candidate_samples",
            "evaluation_samples",
            "downstream_t3_samples",
            "downstream_t4_samples",
        )
    )


def test_versioned_runner_input_is_atomic_and_idempotent(tmp_path):
    config = M3PilotConfig(roots_100=2, roots_1000=4)
    roots = generate_pilot_roots(2, config, run_id="input-test")
    path = tmp_path / "roots.jsonl"

    first_digest = write_runner_input(path, roots)
    second_digest = write_runner_input(path, roots)
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert first_digest == second_digest
    assert [row["schema_version"] for row in rows] == [M3_PILOT_INPUT_SCHEMA] * 2
    assert [row["root_id"] for row in rows] == [root.root_id for root in roots]
    assert all(row["request"] == root.request for row, root in zip(rows, roots))
    assert not list(tmp_path.glob("*.tmp"))


def test_small_2_then_4_root_pilot_goes_through_both_phases(
    tmp_path, monkeypatch
):
    calls = _install_fake_runner(monkeypatch)
    config = M3PilotConfig(
        roots_100=2,
        roots_1000=4,
        deterministic_subset_roots=2,
        parity_roots=2,
        seed_start=2026071391,
        seed_stride=1_000_003,
    )

    report = run_m3_validation(
        config,
        tmp_path,
        executable=tmp_path / "debug-fixture-runner",
        parity_benchmark=_passing_benchmark,
    )
    persisted = json.loads(
        (tmp_path / "validation_summary.json").read_text(encoding="utf-8")
    )

    assert persisted == report
    assert report["schema"] == M3_VALIDATION_SCHEMA
    assert report["status"] == "complete_pass"
    assert report["go_1000"] is True
    assert report["phase_100"]["root_count"] == 2
    assert report["phase_1000"]["root_count"] == 4
    assert report["phase_100"]["seat_counts"] == {"first": 1, "second": 1}
    assert report["phase_1000"]["seat_counts"] == {"first": 2, "second": 2}
    assert all(report["gates"].values())
    assert all(report["phase_100"]["gates"].values())
    assert all(report["phase_1000"]["gates"].values())
    assert [count for _run_id, _resume, count in calls] == [2, 2, 4, 2]
    assert report["root_generation"]["current_profile_read"] is False
    assert report["compute_scope"]["cloud_actions_performed"] is False
    assert report["teacher_values"] == TEACHER_VALUE_STATUS


def test_failed_parity_gate_does_not_run_1000_phase(tmp_path, monkeypatch):
    calls = _install_fake_runner(monkeypatch)

    def failed_benchmark(*args, **kwargs):
        return {
            "passed": False,
            "gates": {
                "action_key_value_parity": True,
                "aggregate_speedup_at_least_5x": False,
            },
        }

    report = run_m3_validation(
        M3PilotConfig(roots_100=2, roots_1000=4),
        tmp_path,
        executable=tmp_path / "debug-fixture-runner",
        parity_benchmark=failed_benchmark,
    )

    assert report["status"] == "no_go"
    assert report["go_1000"] is False
    assert report["phase_1000"]["status"] == "not_run"
    assert [count for _run_id, _resume, count in calls] == [2, 2]


def test_result_validation_rejects_rng_overlap_and_bad_selected_mapping():
    config = M3PilotConfig(roots_100=2, roots_1000=4)
    root = generate_pilot_roots(2, config, run_id="bad-output")[0]
    row = {
        "schema_version": M3_RUNNER_RESULT_SCHEMA,
        "request_schema_version": M3_PILOT_INPUT_SCHEMA,
        "input_index": root.root_index,
        "root_id": root.root_id,
        "result": _fake_engine_result(root.input_row()),
    }
    shared = _digest("shared")
    row["result"]["candidate_rng_key_digests"] = [shared]
    row["result"]["evaluation_rng_key_digests"] = [shared]
    row["result"]["candidate_belief"]["particle_digests"] = [shared]
    row["result"]["evaluation_belief"]["particle_digests"] = [shared]
    row["result"]["selected_action_original_index"] = 9

    report = validate_runner_results([root], [row])

    assert report["passed"] is False
    assert report["gates"][
        "candidate_evaluation_rng_digest_disjoint_per_root"
    ] is False
    assert report["gates"]["selected_results_well_formed"] is False


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"roots_100": 3}, "positive even"),
        ({"roots_100": 4, "roots_1000": 2}, "at least"),
        ({"seed_stride": 0}, "positive"),
        ({"candidate_samples": 2}, "must be 1"),
        ({"parity_roots": 5}, "between 1 and 4"),
    ],
)
def test_config_enforces_bounded_balanced_pilot(kwargs, message):
    with pytest.raises(ValueError, match=message):
        M3PilotConfig(**kwargs)
