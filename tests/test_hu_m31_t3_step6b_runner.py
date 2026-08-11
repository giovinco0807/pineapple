from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ofc_regular.action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.ai_profiles import ModelBundle
from ofc_regular.hu_m31_t3_behavior_roots import (
    M31_T3_BEHAVIOR_PROFILES,
    behavior_profile_for_index,
    generate_behavior_t3_roots,
)
from ofc_regular.run_hu_m31_t3_step6a_shard import (
    RAYON_THREADS,
    STEP5_CONTRACT_CANONICAL_SHA256,
    STEP6A_RUN_ID,
    STEP6A_SEARCH_TASK_SCHEMA,
    seed_values,
)
from ofc_regular.run_hu_m31_t3_step6b_shard import (
    AUTHORIZED_SHARDS,
    EXPECTED_FEATURE_ENCODER_SHA256,
    EXPECTED_MODELS,
    EXPECTED_NATIVE_LIBRARY_SHA256,
    STEP6B_PACKAGE_SCHEMA,
    STEP6B_PARITY_SCHEMA,
    STEP6B_SHARD_SCHEMA,
    STEP6B_SUMMARY_SCHEMA,
    ShardSpec,
    _INTEGRITY_KEYS,
    _PARITY_GATE_KEYS,
    _load_manifest_and_spec,
    _validate_parity_report,
    _validate_recovered_search_task,
)
from ofc_regular.validate_hu_m31_t3_convergence import REFERENCE_BUDGET


REPO_ROOT = Path(__file__).resolve().parents[1]


def _schedule(run_name: str) -> list[dict[str, Any]]:
    return [
        {
            "schema": STEP6B_SHARD_SCHEMA,
            "run_name": run_name,
            "shard": shard,
            "global_hand_start": shard * 25,
            "hand_count": 25,
            "root_count": 50,
            "output_prefix": f"shard-{shard:03d}",
        }
        for shard in range(10)
    ]


def _write_manifest_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    run_name = "regular-hu-m31-step6b-test"
    schedule_path = tmp_path / "shards_manifest.jsonl"
    schedule_bytes = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        for row in _schedule(run_name)
    )
    schedule_path.write_bytes(schedule_bytes)
    source_sha = "a" * 64
    manifest = {
        "schema": STEP6B_PACKAGE_SCHEMA,
        "status": "packaged_local_no_gcloud",
        "run_name": run_name,
        "source_sha256": source_sha,
        "schedule_sha256": hashlib.sha256(schedule_bytes).hexdigest(),
        "total_shards": 10,
        "authorized_shards": list(AUTHORIZED_SHARDS),
        "paired_hands_per_shard": 25,
        "roots_per_shard": 50,
        "step5_contract_canonical_sha256": STEP5_CONTRACT_CANONICAL_SHA256,
        "parity_golden_sha256": "b" * 64,
        "native_library": {
            "path": "native/release/libofc_hu_m3_engine.so",
            "sha256": EXPECTED_NATIVE_LIBRARY_SHA256,
            "engine_version": "ofc_hu_m3_engine/0.1.0",
        },
        "feature_encoder_library": {
            "path": "target/release/libofc_stage3_feature_encoder.so",
            "sha256": EXPECTED_FEATURE_ENCODER_SHA256,
        },
        "models": EXPECTED_MODELS,
        "canary_rows_training_eligible": False,
        "remaining_canary_shards_authorized": True,
        "production_fanout_authorized": False,
        "current_profile_changed": False,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, schedule_path, source_sha


def _valid_parity() -> dict[str, Any]:
    rows = []
    for seat, character in (("first", "d"), ("second", "e")):
        digest = character * 64
        rows.append(
            {
                "seat": seat,
                "observation_fingerprint": "f" * 64,
                "expected_portable_sha256": digest,
                "observed_portable_sha256": digest,
                "match": True,
            }
        )
    return {
        "schema": STEP6B_PARITY_SCHEMA,
        "status": "pass",
        "source_package_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "parity_golden_sha256": "c" * 64,
        "native_library_sha256": EXPECTED_NATIVE_LIBRARY_SHA256,
        "rows": rows,
        "gates": {key: True for key in _PARITY_GATE_KEYS},
        "all_gates_passed": True,
        "current_profile_changed": False,
        "teacher_generation_started": False,
        "production_fanout_authorized": False,
    }


def _root_and_search_task() -> tuple[dict[str, Any], dict[str, Any]]:
    first, second = generate_behavior_t3_roots(
        hand_seed=12345,
        behavior_seed=54321,
        profile="random_exact_final",
        bundle=ModelBundle(),
    )
    global_index = 25
    seeds = seed_values(global_index)
    root_task = {
        "contract_digest": "contract",
        "global_hand_index": global_index,
        "profile": behavior_profile_for_index(global_index),
        "seeds": seeds,
        "observations": [
            {
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "observation": observation.to_dict(),
            }
            for observation in (first, second)
        ],
    }
    task_rows = []
    for offset, observation in enumerate((first, second)):
        actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
        action_values = []
        for index, action in enumerate(actions):
            action_values.append(
                {
                    "action_key": action_key(action).to_token(),
                    "placements": [list(item) for item in action.placements],
                    "discards": list(action.discards),
                    "original_index": index,
                    "rank": index,
                    "selection_ev": float(index),
                    "evaluation_ev": float(index),
                    "evaluation_regret": 0.0,
                }
            )
        selected = action_values[0]
        decision = {
            "seat": observation.seat,
            "observation_fingerprint": observation.fingerprint(),
            "run_id": STEP6A_RUN_ID,
            "continuation_seed": seeds["child"],
            "candidate_seed": seeds["candidate"],
            "evaluation_seed": seeds["evaluation"],
            "candidate_samples": REFERENCE_BUDGET.candidate_samples,
            "evaluation_samples": REFERENCE_BUDGET.evaluation_samples,
            "downstream_t3_samples": REFERENCE_BUDGET.downstream_t3_samples,
            "downstream_t4_samples": 0,
            "native_library_sha256": EXPECTED_NATIVE_LIBRARY_SHA256,
            "execution_mode": "scalar",
            "batch_size": 1,
            "teacher_value_status": "diagnostic_not_match_EV",
            "action_key_schema": ACTION_KEY_SCHEMA,
            "legal_action_set_digest": legal_action_set_digest(actions),
            "legal_action_order_digest": ordered_action_mapping_digest(actions),
            "action_values": action_values,
            "selected_action_key": selected["action_key"],
            "selected_action": {
                "placements": selected["placements"],
                "discards": selected["discards"],
            },
            "selected_selection_ev": selected["selection_ev"],
            "selected_evaluation_ev": selected["evaluation_ev"],
        }
        task_rows.append(
            {
                "root_index": global_index * 2 + offset,
                "global_hand_index": global_index,
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "legal_action_count": len(actions),
                "wall_seconds": 0.1,
                "decision": decision,
                "integrity": {key: True for key in _INTEGRITY_KEYS},
            }
        )
    task = {
        "schema": STEP6A_SEARCH_TASK_SCHEMA,
        "contract_digest": root_task["contract_digest"],
        "global_hand_index": global_index,
        "profile": root_task["profile"],
        "seeds": seeds,
        "rayon_threads": str(RAYON_THREADS),
        "memory": {"peak_rss_bytes": 1024},
        "engine": {"library_sha256": EXPECTED_NATIVE_LIBRARY_SHA256},
        "rows": task_rows,
        "all_gates_passed": True,
    }
    return root_task, task


def test_step6b_shard_spec_accepts_only_remaining_canary_shards():
    assert ShardSpec("regular-hu-m31-step6b-test", 1, 25).root_count == 50
    assert ShardSpec("regular-hu-m31-step6b-test", 9, 225).root_count == 50
    for shard, start in ((0, 0), (10, 250), (True, 25)):
        with pytest.raises(ValueError, match="shards 1 through 9 only"):
            ShardSpec("regular-hu-m31-step6b-test", shard, start)
    with pytest.raises(ValueError, match="schedule changed"):
        ShardSpec("regular-hu-m31-step6b-test", 1, 26)


def test_step6b_manifest_and_schedule_bind_exact_authorized_range(tmp_path):
    manifest, schedule, source_sha = _write_manifest_fixture(tmp_path)
    payload, spec, digest = _load_manifest_and_spec(
        manifest_path=manifest,
        schedule_path=schedule,
        source_package_sha256=source_sha,
        shard=9,
    )
    assert payload["authorized_shards"] == list(range(1, 10))
    assert (spec.shard, spec.global_hand_start) == (9, 225)
    assert len(digest) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authorized_shards", list(range(10))),
        ("remaining_canary_shards_authorized", False),
        ("production_fanout_authorized", True),
        ("current_profile_changed", True),
        ("canary_rows_training_eligible", True),
    ],
)
def test_step6b_manifest_boundary_mutations_fail_closed(
    tmp_path, field: str, value: Any
):
    manifest, schedule, source_sha = _write_manifest_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload[field] = value
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest boundary changed"):
        _load_manifest_and_spec(
            manifest_path=manifest,
            schedule_path=schedule,
            source_package_sha256=source_sha,
            shard=1,
        )


def test_step6b_frozen_model_identity_mutation_fails_closed(tmp_path):
    manifest, schedule, source_sha = _write_manifest_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    model = next(iter(payload["models"]))
    payload["models"][model] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="native/model identity changed"):
        _load_manifest_and_spec(
            manifest_path=manifest,
            schedule_path=schedule,
            source_package_sha256=source_sha,
            shard=1,
        )


def test_step6b_remaining_seed_space_is_exact_and_disjoint_from_shard0():
    shard0 = {value for index in range(25) for value in seed_values(index).values()}
    remaining = [
        value for index in range(25, 250) for value in seed_values(index).values()
    ]
    assert len(shard0) == len(set(shard0)) == 150
    assert len(remaining) == len(set(remaining)) == 1350
    assert not (shard0 & set(remaining))
    assert all(
        sum(
            behavior_profile_for_index(index) == profile
            for index in range(shard * 25, shard * 25 + 25)
        )
        == 5
        for shard in AUTHORIZED_SHARDS
        for profile in M31_T3_BEHAVIOR_PROFILES
    )


def test_step6b_parity_report_is_bound_to_package_manifest_golden_and_native():
    report = _valid_parity()
    kwargs = {
        "source_package_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "parity_golden_sha256": "c" * 64,
        "native_library_sha256": EXPECTED_NATIVE_LIBRARY_SHA256,
    }
    _validate_parity_report(report, **kwargs)
    for field in (
        "source_package_sha256",
        "manifest_sha256",
        "parity_golden_sha256",
        "native_library_sha256",
    ):
        mutated = deepcopy(report)
        mutated[field] = "0" * 64
        with pytest.raises(ValueError, match="parity provenance changed"):
            _validate_parity_report(mutated, **kwargs)


def test_step6b_recovered_search_task_validates_full_row_binding():
    root, task = _root_and_search_task()
    _validate_recovered_search_task(
        task, root_task=root, library_sha256=EXPECTED_NATIVE_LIBRARY_SHA256
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda task: task["rows"][0].__setitem__("seat", "second"),
        lambda task: task["rows"][0].__setitem__("root_index", 999),
        lambda task: task["rows"][0].__setitem__("observation_fingerprint", "0" * 64),
        lambda task: task["rows"][0]["decision"].__setitem__("candidate_seed", 1),
        lambda task: task["rows"][0]["decision"]["action_values"][1].__setitem__(
            "original_index", 0
        ),
        lambda task: task["rows"][0]["integrity"].pop("run_id"),
    ],
)
def test_step6b_recovered_search_task_mutations_fail_closed(mutation):
    root, task = _root_and_search_task()
    mutation(task)
    with pytest.raises(ValueError, match="Step 6b recovered"):
        _validate_recovered_search_task(
            task, root_task=root, library_sha256=EXPECTED_NATIVE_LIBRARY_SHA256
        )


def test_step6b_keeps_frozen_run_id_summary_schema_and_spawn_start_method():
    source = (REPO_ROOT / "src/ofc_regular/run_hu_m31_t3_step6b_shard.py").read_text(
        encoding="utf-8"
    )
    assert STEP6A_RUN_ID == "hu-m31-step6a-infrastructure-canary-v1"
    assert STEP6B_SUMMARY_SCHEMA == "hu_m31_t3_step6b_summary_v1"
    assert 'multiprocessing.get_context("spawn")' in source
