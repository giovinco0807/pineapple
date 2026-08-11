from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ofc_regular.hu_rl_native import native_available
from ofc_regular.hu_rl_native_benchmark import generate_benchmark_decks
from ofc_regular.hu_rl_replay_resume_smoke import (
    DECISIONS_PER_HAND,
    HU_RL_REPLAY_RESUME_PROVENANCE_SCHEMA,
    HU_RL_REPLAY_RESUME_REQUEST_SCHEMA,
    HU_RL_REPLAY_RESUME_SMOKE_SCHEMA,
    SEED_NAMESPACE,
    HuRlReplayResumeSmokeError,
    _REQUIRED_SOURCE_PATHS,
    _canonical_digest,
    _deck_commitment,
    _prefix_action_digest,
    _receipt_digest,
    _seed_commitment,
    canonical_replay_resume_json,
    collect_replay_resume_provenance,
    evaluate_replay_resume_worker_request,
    validate_replay_resume_receipt,
    validate_replay_resume_request,
)


_ZERO_ACTION = "rak1:0000000000000:0000000000000:0000000000000:0000000000000"


def _request(*, prefix_decisions: int = 5) -> dict[str, Any]:
    prefix = [
        {
            "decision_ordinal": decision,
            "action_tokens": [_ZERO_ACTION] * 100,
        }
        for decision in range(prefix_decisions)
    ]
    seed = 7
    return {
        "schema": HU_RL_REPLAY_RESUME_REQUEST_SCHEMA,
        "seed_namespace": SEED_NAMESPACE,
        "run_seed": seed,
        "seed_commitment_sha256": _seed_commitment(seed),
        "deck_commitment_sha256": "d" * 64,
        "lane_count": 100,
        "chunk_width": 32,
        "thread_count": 8,
        "prefix_decisions": prefix_decisions,
        "prefix_actions": prefix,
        "prefix_action_history_sha256": _prefix_action_digest(prefix),
        "provenance_sha256": "e" * 64,
    }


def _receipt() -> dict[str, Any]:
    checkpoint = {
        "packed_observation_sha256": "1" * 64,
        "packed_action_keys_sha256": "2" * 64,
        "packed_mask_sha256": "3" * 64,
        "packed_action_counts_sha256": "4" * 64,
        "packed_action_set_digests_sha256": "5" * 64,
        "packed_action_order_digests_sha256": "6" * 64,
    }
    source_hashes = {path: "7" * 64 for path in _REQUIRED_SOURCE_PATHS}
    receipt: dict[str, Any] = {
        "schema": HU_RL_REPLAY_RESUME_SMOKE_SCHEMA,
        "status": "deterministic_replay_resume_smoke_passed",
        "artifact_role": "deterministic_replay_resume_smoke",
        "scope": "bounded_100_to_1000_lane_local_pilot",
        "resume_mechanism": "fresh_env_seed_and_prefix_action_replay",
        "subprocess_boundary_exercised": True,
        "production_checkpoint_eligible": False,
        "serializable_snapshot_claimed": False,
        "cross_process_snapshot_claimed": False,
        "performance_gate_evaluated": False,
        "performance_gate_pass": None,
        "lane_count": 100,
        "chunk_width": 32,
        "thread_count": 8,
        "decision_count_per_lane": DECISIONS_PER_HAND,
        "prefix_decisions": 5,
        "seed_namespace": SEED_NAMESPACE,
        "seed_commitment_sha256": "a" * 64,
        "deck_commitment_sha256": "b" * 64,
        "prefix_action_history_sha256": "c" * 64,
        "prefix_replay_checkpoint": checkpoint,
        "remaining_public_step_sha256": ["d" * 64] * 5,
        "terminal_rewards_sha256": "e" * 64,
        "public_aggregate_sha256": "f" * 64,
        "decision_counts_sha256": "0" * 64,
        "byte_identical": True,
        "provenance": {
            "schema": HU_RL_REPLAY_RESUME_PROVENANCE_SCHEMA,
            "package_name": "ofc-hu-rl-engine-native",
            "package_version": "0.1.0",
            "wheel_filename": "engine.whl",
            "wheel_sha256": "1" * 64,
            "native_extension_filename": "engine.pyd",
            "native_extension_sha256": "2" * 64,
            "source_sha256": source_hashes,
        },
        "runtime": {
            "python_version": "3.13.0",
            "python_implementation": "CPython",
            "platform": "test-platform",
            "machine": "test-machine",
            "processor": "test-processor",
            "cpu_count": 8,
        },
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = _receipt_digest(receipt)
    return receipt


def test_request_contract_accepts_complete_prefix() -> None:
    validate_replay_resume_request(_request())


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda value: value["prefix_actions"][1].__setitem__(
                "decision_ordinal", 0
            ),
            "duplicated",
        ),
        (
            lambda value: value["prefix_actions"][1].__setitem__(
                "decision_ordinal", 2
            ),
            "missing or reordered",
        ),
        (lambda value: value["prefix_actions"].pop(), "prefix length"),
        (
            lambda value: value.__setitem__(
                "prefix_action_history_sha256", "0" * 64
            ),
            "ActionKey digest mismatch",
        ),
        (
            lambda value: value["prefix_actions"][0]["action_tokens"].pop(),
            "lane geometry",
        ),
        (
            lambda value: value["prefix_actions"][0]["action_tokens"].__setitem__(
                0, "invalid"
            ),
            "ActionKey encoding",
        ),
        (
            lambda value: value.__setitem__("seed_commitment_sha256", "0" * 64),
            "seed commitment",
        ),
        (lambda value: value.__setitem__("unexpected", True), "fields"),
    ],
)
def test_request_tamper_duplicate_missing_and_mismatch_fail_closed(
    mutator, match: str
) -> None:
    request = _request()
    mutator(request)
    with pytest.raises(HuRlReplayResumeSmokeError, match=match):
        validate_replay_resume_request(request)


@pytest.mark.parametrize("lane_count", (0, 99, 1001, 100_000, 1_000_000))
def test_request_rejects_out_of_pilot_lane_counts(lane_count: int) -> None:
    request = _request()
    request["lane_count"] = lane_count
    with pytest.raises(HuRlReplayResumeSmokeError, match="bounded pilot range"):
        validate_replay_resume_request(request)


@pytest.mark.skipif(not native_available(), reason="optional PyO3 extension not built")
def test_semantically_illegal_prefix_action_fails_closed_during_replay() -> None:
    request = _request(prefix_decisions=1)
    provenance = collect_replay_resume_provenance()
    decks = generate_benchmark_decks(
        lane_count=request["lane_count"],
        seed=request["run_seed"],
    )
    request["deck_commitment_sha256"] = _deck_commitment(decks)
    request["provenance_sha256"] = _canonical_digest(provenance)

    with pytest.raises(
        HuRlReplayResumeSmokeError,
        match="prefix replay or remaining step failed",
    ):
        evaluate_replay_resume_worker_request(request)


def test_receipt_is_explicitly_smoke_only_and_contains_no_private_payload() -> None:
    receipt = _receipt()
    validate_replay_resume_receipt(receipt)
    encoded = canonical_replay_resume_json(receipt)
    assert receipt["production_checkpoint_eligible"] is False
    assert receipt["serializable_snapshot_claimed"] is False
    assert receipt["cross_process_snapshot_claimed"] is False
    assert receipt["performance_gate_evaluated"] is False
    assert receipt["performance_gate_pass"] is None
    assert receipt["byte_identical"] is True
    lowered = encoded.lower()
    for forbidden in (
        '"deck_tail"',
        '"opponent_private_discard"',
        '"opponent_private_discards"',
        '"world_state"',
        '"action_tokens"',
        '"run_seed"',
    ):
        assert forbidden not in lowered


def test_receipt_tampering_and_overclaim_fail_closed() -> None:
    for field, value, match in (
        ("production_checkpoint_eligible", True, "production_checkpoint_eligible"),
        ("cross_process_snapshot_claimed", True, "cross_process_snapshot_claimed"),
        ("byte_identical", False, "byte_identical"),
        ("receipt_sha256", "0" * 64, "receipt digest"),
    ):
        receipt = _receipt()
        receipt[field] = value
        with pytest.raises(HuRlReplayResumeSmokeError, match=match):
            validate_replay_resume_receipt(receipt)


@pytest.mark.skipif(not native_available(), reason="optional PyO3 extension not built")
def test_actual_cli_crosses_subprocess_boundary_and_output_is_write_once(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    script = repository / "scripts" / "run_hu_rl_replay_resume_smoke.py"
    output = tmp_path / "receipt.json"
    command = [
        sys.executable,
        str(script),
        "--seed",
        "2026072201",
        "--lanes",
        "100",
        "--prefix-decisions",
        "5",
        "--chunk-width",
        "32",
        "--thread-count",
        "8",
        "--output",
        str(output),
    ]
    first = subprocess.run(
        command,
        cwd=repository,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert first.returncode == 0, first.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    validate_replay_resume_receipt(receipt)
    assert receipt["subprocess_boundary_exercised"] is True
    assert receipt["lane_count"] == 100
    before = output.read_bytes()

    second = subprocess.run(
        command,
        cwd=repository,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert second.returncode == 1
    assert "already exists" in second.stderr
    assert output.read_bytes() == before
