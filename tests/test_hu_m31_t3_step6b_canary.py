from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

import pytest

import ofc_regular.validate_hu_m31_t3_step6b_canary as subject
from ofc_regular.action_key import (
    ACTION_KEY_SCHEMA,
    ActionKey,
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_belief import sample_hidden_card_particles
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.state import Board
from ofc_regular.validate_hu_m31_t3_convergence import REFERENCE_BUDGET


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def _take(cards: list[str], cursor: int, count: int) -> tuple[tuple[str, ...], int]:
    return tuple(cards[cursor : cursor + count]), cursor + count


def _observation(global_hand_index: int, seat: str) -> ActorObservation:
    cards = list(ALL_CARDS)
    random.Random(77_000 + global_hand_index * 2 + (seat == "second")).shuffle(cards)
    cursor = 0
    hero_top, cursor = _take(cards, cursor, 2)
    hero_middle, cursor = _take(cards, cursor, 3)
    hero_bottom, cursor = _take(cards, cursor, 4)
    opponent_top, cursor = _take(cards, cursor, 2)
    opponent_middle, cursor = _take(cards, cursor, 3 if seat == "first" else 4)
    opponent_bottom, cursor = _take(cards, cursor, 4 if seat == "first" else 5)
    dealt, cursor = _take(cards, cursor, 3)
    hero_discards, cursor = _take(cards, cursor, 2)
    assert cursor in {23, 25}
    return ActorObservation(
        hero_board=Board.from_rows(hero_top, hero_middle, hero_bottom),
        opponent_public_board=Board.from_rows(
            opponent_top, opponent_middle, opponent_bottom
        ),
        dealt_cards=dealt,
        hero_private_discards=hero_discards,
        seat=seat,
        street="T3",
        to_act_order=seat,
    )


def _decision(
    observation: ActorObservation,
    *,
    seeds: dict[str, int],
    global_hand_index: int,
) -> dict[str, object]:
    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    values = []
    selection_values = [float(len(legal) - index) for index in range(len(legal))]
    evaluation_values = [value / 2.0 for value in selection_values]
    best_evaluation = max(evaluation_values)
    ranking = sorted(range(len(legal)), key=lambda index: -selection_values[index])
    rank_by_index = {original: rank for rank, original in enumerate(ranking)}
    for original_index, action in enumerate(legal):
        values.append(
            {
                "action_key": action_key(action).to_token(),
                "original_index": original_index,
                "rank": rank_by_index[original_index],
                "placements": [list(item) for item in action.placements],
                "discards": list(action.discards),
                "selection_ev": selection_values[original_index],
                "evaluation_ev": evaluation_values[original_index],
                "evaluation_regret": best_evaluation
                - evaluation_values[original_index],
            }
        )
    values.sort(key=lambda row: ActionKey.from_token(row["action_key"]).sort_key())
    selected_index = ranking[0]
    selected = legal[selected_index]
    selected_token = action_key(selected).to_token()
    candidate = sample_hidden_card_particles(
        observation,
        base_seed=seeds["candidate"],
        run_id=f"{subject.STEP6A_RUN_ID}:candidate_selection",
        sample_count=REFERENCE_BUDGET.candidate_samples,
    )
    evaluation = sample_hidden_card_particles(
        observation,
        base_seed=seeds["evaluation"],
        run_id=f"{subject.STEP6A_RUN_ID}:locked_evaluation",
        sample_count=REFERENCE_BUDGET.evaluation_samples,
    )
    candidate_keys = tuple(row.rng_key_digest for row in candidate.particles)
    evaluation_keys = tuple(row.rng_key_digest for row in evaluation.particles)
    return {
        "schema": "hu_m31_t3_runtime_decision_v1",
        "runtime_id": "synthetic-step6b-test",
        "seat": observation.seat,
        "value_scope": "diagnostic_teacher_value",
        "observation_fingerprint": observation.fingerprint(),
        "selected_action_key": selected_token,
        "selected_selection_ev": selection_values[selected_index],
        "selected_evaluation_ev": evaluation_values[selected_index],
        "selection_gap": selection_values[selected_index]
        - selection_values[ranking[1]],
        "evaluation_sample_regret": 0.0,
        "selected_action": {
            "placements": [list(item) for item in selected.placements],
            "discards": list(selected.discards),
        },
        "action_key_schema": ACTION_KEY_SCHEMA,
        "legal_action_set_digest": legal_action_set_digest(legal),
        "legal_action_order_digest": ordered_action_mapping_digest(legal),
        "action_values": values,
        "belief_prior": "uniform_hidden_assignment_v1",
        "candidate_belief_digest": candidate.digest(),
        "evaluation_belief_digest": evaluation.digest(),
        "candidate_rng_digest": _token_digest(candidate_keys),
        "evaluation_rng_digest": _token_digest(evaluation_keys),
        "candidate_samples": REFERENCE_BUDGET.candidate_samples,
        "evaluation_samples": REFERENCE_BUDGET.evaluation_samples,
        "downstream_t3_samples": REFERENCE_BUDGET.downstream_t3_samples,
        "downstream_t4_samples": 0,
        "run_id": subject.STEP6A_RUN_ID,
        "continuation_seed": seeds["child"],
        "candidate_seed": seeds["candidate"],
        "evaluation_seed": seeds["evaluation"],
        "use_t4_action_cache": True,
        "continuation_policy_id": "synthetic",
        "strategy_fusion_guard": "synthetic",
        "search_contract_digest": _token_digest(
            [global_hand_index, observation.seat, seeds]
        ),
        "downstream_t4_native_semantics_id": "synthetic-exact-t4",
        "downstream_t4_native_anchor": "same_pinned_m30_native_engine",
        "downstream_t4_mode": "exact",
        "child_information_set_count": len(legal),
        "solver_id": "synthetic",
        "engine_version": "ofc_hu_m3_engine/0.1.0",
        "native_library_sha256": subject.EXPECTED_NATIVE_LIBRARY_SHA256,
        "teacher_value_status": "diagnostic_not_match_EV",
        "native_latency_ms": 1.0,
        "validation_latency_ms": 1.0,
        "total_latency_ms": 2.0,
        "execution_mode": "scalar",
        "batch_size": 1,
        "semantic_result_digest_schema": "synthetic",
        "semantic_result_digest_scope": "synthetic",
        "semantic_result_digest": _token_digest(
            ["semantic", global_hand_index, observation.seat]
        ),
        "result_digest_scope": "synthetic",
        "result_digest": _token_digest(["result", global_hand_index, observation.seat]),
    }


def _root_and_task(
    global_hand_index: int,
    *,
    duplicate_first: ActorObservation | None,
    slow_first: bool,
) -> tuple[dict[str, object], dict[str, object], set[str], set[str]]:
    seeds = subject.seed_values(global_hand_index)
    observations = [
        duplicate_first or _observation(global_hand_index, "first"),
        _observation(global_hand_index, "second"),
    ]
    contract_digest = _token_digest(["synthetic-contract", global_hand_index, seeds])
    root = {
        "schema": subject.STEP6A_ROOT_TASK_SCHEMA,
        "contract_digest": contract_digest,
        "global_hand_index": global_hand_index,
        "profile": subject.behavior_profile_for_index(global_hand_index),
        "seeds": seeds,
        "observations": [
            {
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "observation": observation.to_dict(),
            }
            for observation in observations
        ],
        "current_profile_resolved": False,
        "opponent_private_discards_used": False,
    }
    rows = []
    candidate_keys: set[str] = set()
    evaluation_keys: set[str] = set()
    for offset, observation in enumerate(observations):
        decision = _decision(
            observation, seeds=seeds, global_hand_index=global_hand_index
        )
        checks, candidate, evaluation = subject._validate_decision(
            decision,
            observation=observation,
            seeds=seeds,
            native_library_sha256=subject.EXPECTED_NATIVE_LIBRARY_SHA256,
        )
        candidate_keys.update(candidate)
        evaluation_keys.update(evaluation)
        latency = 200.0 if slow_first and offset == 0 else (1.0 if offset == 0 else 0.1)
        rows.append(
            {
                "root_index": global_hand_index * 2 + offset,
                "global_hand_index": global_hand_index,
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "legal_action_count": len(decision["action_values"]),
                "wall_seconds": latency,
                "decision": decision,
                "integrity": checks,
            }
        )
    task = {
        "schema": subject.STEP6A_SEARCH_TASK_SCHEMA,
        "contract_digest": contract_digest,
        "global_hand_index": global_hand_index,
        "profile": subject.behavior_profile_for_index(global_hand_index),
        "seeds": seeds,
        "rayon_threads": str(subject.RAYON_THREADS),
        "engine": {
            "version": "ofc_hu_m3_engine/0.1.0",
            "library_sha256": subject.EXPECTED_NATIVE_LIBRARY_SHA256,
            "build_or_fallback": False,
        },
        "process_id": 1000 + global_hand_index,
        "rows": rows,
        "memory": {
            "supported": True,
            "source": "synthetic",
            "rss_bytes": None,
            "private_bytes": None,
            "peak_rss_bytes": 128 * 1024 * 1024,
        },
        "wall_seconds": sum(row["wall_seconds"] for row in rows),
        "all_gates_passed": True,
    }
    return root, task, candidate_keys, evaluation_keys


_GATES = {
    name: True
    for name in (
        "exactly_50_roots",
        "exactly_25_each_seat",
        "five_profiles_equal_quota",
        "unique_observation_fingerprints",
        "all_task_and_decision_integrity",
        "candidate_evaluation_rng_disjoint",
        "candidate_rng_unique",
        "evaluation_rng_unique",
        "linux_portable_parity",
        "resume_drill_recovered_task",
        "first_p95_within_180_seconds",
        "second_p95_within_6_seconds",
        "peak_rss_within_1_gib",
        "canary_rows_not_training_eligible",
        "no_current_profile_or_production_fanout",
    )
}


def _build_shard(
    directory: Path,
    *,
    shard: int,
    duplicate_first: ActorObservation | None = None,
    slow_first_count: int = 0,
) -> None:
    start = shard * subject.PAIRED_HANDS_PER_SHARD
    task_hashes = []
    fingerprints = []
    profiles = []
    first_latencies = []
    second_latencies = []
    candidate_keys: set[str] = set()
    evaluation_keys: set[str] = set()
    for local_index, global_index in enumerate(
        range(start, start + subject.PAIRED_HANDS_PER_SHARD)
    ):
        root, task, candidate, evaluation = _root_and_task(
            global_index,
            duplicate_first=duplicate_first if local_index == 0 else None,
            slow_first=local_index < slow_first_count,
        )
        root_path = directory / "roots" / f"hand_{global_index:03d}.json"
        task_path = directory / "tasks" / f"hand_{global_index:03d}.json"
        _write_json(root_path, root)
        _write_json(task_path, task)
        task_hashes.append(
            {"global_hand_index": global_index, "sha256": _sha(task_path)}
        )
        fingerprints.extend(
            row["observation_fingerprint"] for row in root["observations"]
        )
        profiles.append(root["profile"])
        first_latencies.append(task["rows"][0]["wall_seconds"])
        second_latencies.append(task["rows"][1]["wall_seconds"])
        candidate_keys.update(candidate)
        evaluation_keys.update(evaluation)

    source_sha = _token_digest(["source", "step6a" if shard == 0 else "step6b"])
    manifest_sha = _token_digest(["manifest", "step6a" if shard == 0 else "step6b"])
    gates = dict(_GATES)
    if shard != 0:
        gates.update(
            {
                "exact_shard_hand_indices": True,
                "profiles_follow_frozen_cycle": True,
                "linux_portable_parity_bound": True,
            }
        )
    summary = {
        "schema": (
            subject.STEP6A_SUMMARY_SCHEMA
            if shard == 0
            else subject.STEP6B_SUMMARY_SCHEMA
        ),
        "status": "pass",
        "run_name": "synthetic-step6a" if shard == 0 else "synthetic-step6b",
        "shard": shard,
        "source_package_sha256": source_sha,
        "manifest_sha256": manifest_sha,
        "native_library_sha256": subject.EXPECTED_NATIVE_LIBRARY_SHA256,
        "contract": {
            "paired_hands": subject.PAIRED_HANDS_PER_SHARD,
            "roots": subject.ROOTS_PER_SHARD,
            "workers": subject.WORKERS,
            "rayon_threads_per_worker": subject.RAYON_THREADS,
            "budget": REFERENCE_BUDGET.to_dict(),
            "run_id": subject.STEP6A_RUN_ID,
        },
        "integrity": {
            "fingerprints": 50,
            "unique_fingerprints": len(set(fingerprints)),
            "candidate_rng_keys": len(candidate_keys),
            "evaluation_rng_keys": len(evaluation_keys),
            "candidate_evaluation_overlap": len(candidate_keys & evaluation_keys),
            "profile_counts": {
                profile: profiles.count(profile)
                for profile in subject.M31_T3_BEHAVIOR_PROFILES
            },
        },
        "performance": {
            "latency_by_seat": {
                "first": subject._latency(first_latencies),
                "second": subject._latency(second_latencies),
            },
            "peak_process_rss_bytes": 128 * 1024 * 1024,
        },
        "resumed_task_count": 1,
        "gates": gates,
        "all_gates_passed": True,
        "task_manifest": task_hashes,
        "teacher_value_status": "diagnostic_not_match_EV",
        "training_eligible": False,
        "current_profile_changed": False,
        "spot_vm_started": True,
        "production_fanout_authorized": False,
        "m31_complete": False,
    }
    if shard != 0:
        summary["global_hand_start"] = start
        summary["remaining_canary_shards_authorized"] = True
    _write_json(
        directory / "heartbeat.json",
        {
            "schema": subject.STEP6A_HEARTBEAT_SCHEMA,
            "status": "pass",
            "total_tasks": 25,
            "completed_tasks": 25,
            "pending_tasks": 0,
            "resumed_task_count": 1,
            "process_id": 123,
            "updated_at": "2026-07-17T00:00:00+00:00",
        },
    )
    parity = {
        "schema": (
            subject.STEP6A_PARITY_SCHEMA if shard == 0 else subject.STEP6B_PARITY_SCHEMA
        ),
        "status": "pass",
        "all_gates_passed": True,
        "native_library_sha256": subject.EXPECTED_NATIVE_LIBRARY_SHA256,
        "current_profile_changed": False,
        "gates": {"portable_action_values_exact": True},
    }
    if shard != 0:
        parity.update(
            {
                "source_package_sha256": source_sha,
                "manifest_sha256": manifest_sha,
                "production_fanout_authorized": False,
            }
        )
    _write_json(directory / "parity.json", parity)
    if shard != 0:
        summary["parity_report_sha256"] = _sha(directory / "parity.json")
    _write_json(directory / "summary.json", summary)
    for name, data in {
        "drill_stdout.json": b"{}\n",
        "run.log": b"synthetic\n",
        "runner_stdout.json": b"{}\n",
        "time.txt": b"synthetic\n",
    }.items():
        (directory / name).write_bytes(data)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "DONE.json":
            relative = path.relative_to(directory).as_posix()
            files[relative] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    _write_json(
        directory / "DONE.json",
        {
            "schema": (
                "hu_m31_t3_step6a_done_v1" if shard == 0 else subject.STEP6B_DONE_SCHEMA
            ),
            "status": "complete",
            "run_name": summary["run_name"],
            "shard": shard,
            "source_sha256": source_sha,
            "manifest_sha256": manifest_sha,
            "schedule_sha256": _token_digest(
                ["schedule", "step6a" if shard == 0 else "step6b"]
            ),
            "authorization_sha256": _token_digest(
                ["authorization", "step6a" if shard == 0 else "step6b"]
            ),
            "native_library_sha256": subject.EXPECTED_NATIVE_LIBRARY_SHA256,
            "files": files,
            "resume_drill_passed": True,
            "training_eligible": False,
            "remaining_canary_shards_authorized": shard != 0,
            "authorized_shards": list(range(1, 10)) if shard != 0 else [0],
            "production_fanout_authorized": False,
            "current_profile_changed": False,
            "completed_unix_seconds": 1.0 + shard,
        },
    )


def _refresh_done(directory: Path) -> None:
    done = json.loads((directory / "DONE.json").read_text())
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "DONE.json":
            relative = path.relative_to(directory).as_posix()
            files[relative] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    done["files"] = files
    _write_json(directory / "DONE.json", done)


def _accepted_identity(step6a: Path) -> dict[str, object]:
    done = json.loads((step6a / "DONE.json").read_text())
    return {
        "run_name": done["run_name"],
        "done_sha256": _sha(step6a / "DONE.json"),
        "summary_sha256": _sha(step6a / "summary.json"),
        "receive_receipt_sha256": _sha(step6a / "receive_receipt.json"),
        "source_sha256": done["source_sha256"],
        "manifest_sha256": done["manifest_sha256"],
        "schedule_sha256": done["schedule_sha256"],
        "authorization_sha256": done["authorization_sha256"],
        "native_library_sha256": subject.EXPECTED_NATIVE_LIBRARY_SHA256,
        "feature_encoder_sha256": subject.EXPECTED_FEATURE_ENCODER_SHA256,
    }


def _refresh_receive_receipt(
    step6a: Path, received: Path, *, preserve_step6a_receipt: bool = False
) -> dict[str, object]:
    if not preserve_step6a_receipt:
        _write_json(
            step6a / "receive_receipt.json",
            {
                "schema": "hu_m31_t3_step6a_receive_v1",
                "status": "pass",
                "done_sha256": _sha(step6a / "DONE.json"),
                "summary_sha256": _sha(step6a / "summary.json"),
                "all_gates_passed": True,
                "training_eligible": False,
                "production_fanout_authorized": False,
                "current_profile_changed": False,
            },
        )
    per_shard = {}
    for shard in range(1, 10):
        directory = received / "shards" / f"shard-{shard:03d}"
        per_shard[f"{shard:03d}"] = {
            "done_sha256": _sha(directory / "DONE.json"),
            "summary_sha256": _sha(directory / "summary.json"),
            "heartbeat_sha256": _sha(directory / "heartbeat.json"),
            "task_count": 25,
            "root_task_count": 25,
            "resumed_task_count": 1,
        }
    first_done = json.loads(
        (received / "shards" / "shard-001" / "DONE.json").read_text()
    )
    step6a_identity = _accepted_identity(step6a)
    receipt = {
        "schema": subject.STEP6B_RECEIVE_SCHEMA,
        "status": "pass",
        "run_name": first_done["run_name"],
        "source_sha256": first_done["source_sha256"],
        "manifest_sha256": first_done["manifest_sha256"],
        "schedule_sha256": first_done["schedule_sha256"],
        "authorization_sha256": first_done["authorization_sha256"],
        "native_library_sha256": subject.EXPECTED_NATIVE_LIBRARY_SHA256,
        "feature_encoder_sha256": subject.EXPECTED_FEATURE_ENCODER_SHA256,
        "step6a_done_sha256": step6a_identity["done_sha256"],
        "step6a_summary_sha256": step6a_identity["summary_sha256"],
        "per_shard": per_shard,
        "all_shards_received": True,
        "training_eligible": False,
        "remaining_canary_shards_authorized": True,
        "production_fanout_authorized": False,
        "current_profile_changed": False,
        "m31_complete": False,
    }
    _write_json(received / "receive_receipt.json", receipt)
    return step6a_identity


def _build_canary(
    root: Path,
    *,
    duplicate_cross_shard_fingerprint: bool = False,
    slow_first_shard: int | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    step6a = root / "step6a"
    received = root / "step6b"
    _build_shard(step6a, shard=0)
    duplicate = _observation(0, "first") if duplicate_cross_shard_fingerprint else None
    for shard in range(1, 10):
        _build_shard(
            received / "shards" / f"shard-{shard:03d}",
            shard=shard,
            duplicate_first=duplicate if shard == 1 else None,
            slow_first_count=2 if slow_first_shard == shard else 0,
        )
    identity = _refresh_receive_receipt(step6a, received)
    return step6a, received, identity


@pytest.fixture(scope="module")
def base_canary(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("step6b-base")
    step6a, received, _identity = _build_canary(root)
    return step6a, received


@pytest.fixture
def canary(
    tmp_path: Path, base_canary: tuple[Path, Path]
) -> tuple[Path, Path, dict[str, object]]:
    base_step6a, base_received = base_canary
    step6a = tmp_path / "step6a"
    received = tmp_path / "step6b"
    shutil.copytree(base_step6a, step6a)
    shutil.copytree(base_received, received)
    return step6a, received, _accepted_identity(step6a)


def test_synthetic_500_root_canary_passes_and_output_is_write_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canary: tuple[Path, Path, dict[str, object]],
):
    step6a, received, identity = canary
    monkeypatch.setattr(subject, "ACCEPTED_STEP6A_IDENTITY", identity)
    output = tmp_path / "validation.json"

    result = subject.validate_step6b_canary(
        step6a_shard0_dir=step6a,
        step6b_received_dir=received,
        output_path=output,
    )

    assert result["status"] == "pass"
    assert result["integrity"]["unique_fingerprints"] == 500
    assert result["integrity"]["unique_namespace_seed_values"] == 1500
    assert result["integrity"]["candidate_rng_keys"] == 2000
    assert result["integrity"]["evaluation_rng_keys"] == 4000
    assert result["training_eligible"] is False
    assert result["production_fanout_authorized"] is False
    assert json.loads(output.read_text()) == result
    with pytest.raises(FileExistsError):
        subject.validate_step6b_canary(
            step6a_shard0_dir=step6a,
            step6b_received_dir=received,
            output_path=output,
        )


def test_cross_shard_duplicate_fingerprint_fails_closed(
    canary: tuple[Path, Path, dict[str, object]],
):
    step6a, received, identity = canary
    first_root = json.loads((step6a / "roots" / "hand_000.json").read_text())
    duplicate = ActorObservation.from_dict(first_root["observations"][0]["observation"])
    root, task, _candidate, _evaluation = _root_and_task(
        25, duplicate_first=duplicate, slow_first=False
    )
    shard = received / "shards" / "shard-001"
    root_path = shard / "roots" / "hand_025.json"
    task_path = shard / "tasks" / "hand_025.json"
    _write_json(root_path, root)
    _write_json(task_path, task)
    summary_path = shard / "summary.json"
    summary = json.loads(summary_path.read_text())
    for record in summary["task_manifest"]:
        if record["global_hand_index"] == 25:
            record["sha256"] = _sha(task_path)
    _write_json(summary_path, summary)
    _refresh_done(shard)
    _refresh_receive_receipt(step6a, received, preserve_step6a_receipt=True)

    with pytest.raises(ValueError, match="merged Step 6b canary gates failed"):
        subject._validate_step6b_canary_core(
            step6a_shard0_dir=step6a,
            step6b_received_dir=received,
            accepted_step6a_identity=identity,
        )


def test_hidden_opponent_discard_field_fails_closed(
    canary: tuple[Path, Path, dict[str, object]],
):
    step6a, received, identity = canary
    shard = received / "shards" / "shard-001"
    root_path = shard / "roots" / "hand_025.json"
    root = json.loads(root_path.read_text())
    root["observations"][0]["observation"]["opponent_private_discards"] = ["As"]
    _write_json(root_path, root)
    _refresh_done(shard)
    _refresh_receive_receipt(step6a, received, preserve_step6a_receipt=True)

    with pytest.raises(ValueError, match="forbidden hidden-information field"):
        subject._validate_step6b_canary_core(
            step6a_shard0_dir=step6a,
            step6b_received_dir=received,
            accepted_step6a_identity=identity,
        )


def test_seed_schedule_mutation_fails_closed(
    canary: tuple[Path, Path, dict[str, object]],
):
    step6a, received, identity = canary
    shard = received / "shards" / "shard-001"
    root_path = shard / "roots" / "hand_025.json"
    root = json.loads(root_path.read_text())
    root["seeds"]["candidate"] += 1
    _write_json(root_path, root)
    _refresh_done(shard)
    _refresh_receive_receipt(step6a, received, preserve_step6a_receipt=True)

    with pytest.raises(ValueError, match="root seed schedule changed"):
        subject._validate_step6b_canary_core(
            step6a_shard0_dir=step6a,
            step6b_received_dir=received,
            accepted_step6a_identity=identity,
        )


def test_missing_shard_and_extra_file_fail_closed(
    canary: tuple[Path, Path, dict[str, object]],
):
    step6a, received, identity = canary
    (received / "unexpected.txt").write_text("no", encoding="utf-8")

    with pytest.raises(ValueError, match="receive root file set changed"):
        subject._validate_step6b_canary_core(
            step6a_shard0_dir=step6a,
            step6b_received_dir=received,
            accepted_step6a_identity=identity,
        )


def test_shard_latency_tail_is_recomputed_from_sorted_rows(
    canary: tuple[Path, Path, dict[str, object]],
):
    step6a, received, identity = canary
    shard = received / "shards" / "shard-001"
    for hand in (25, 26):
        task_path = shard / "tasks" / f"hand_{hand:03d}.json"
        task = json.loads(task_path.read_text())
        task["rows"][0]["wall_seconds"] = 200.0
        task["wall_seconds"] = sum(row["wall_seconds"] for row in task["rows"])
        _write_json(task_path, task)
    summary_path = shard / "summary.json"
    summary = json.loads(summary_path.read_text())
    first_latencies = []
    for task_path in sorted((shard / "tasks").glob("hand_*.json")):
        task = json.loads(task_path.read_text())
        first_latencies.append(task["rows"][0]["wall_seconds"])
        hand = task["global_hand_index"]
        for record in summary["task_manifest"]:
            if record["global_hand_index"] == hand:
                record["sha256"] = _sha(task_path)
    summary["performance"]["latency_by_seat"]["first"] = subject._latency(
        first_latencies
    )
    _write_json(summary_path, summary)
    _refresh_done(shard)
    _refresh_receive_receipt(step6a, received, preserve_step6a_receipt=True)

    with pytest.raises(ValueError, match="first-seat p95 exceeds 180 seconds"):
        subject._validate_step6b_canary_core(
            step6a_shard0_dir=step6a,
            step6b_received_dir=received,
            accepted_step6a_identity=identity,
        )


def test_training_flag_and_receive_hash_manifest_fail_closed(
    canary: tuple[Path, Path, dict[str, object]],
):
    step6a, received, identity = canary
    shard = received / "shards" / "shard-001"
    summary_path = shard / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["training_eligible"] = True
    _write_json(summary_path, summary)
    _refresh_done(shard)
    _refresh_receive_receipt(step6a, received, preserve_step6a_receipt=True)

    with pytest.raises(ValueError, match="canary became training eligible"):
        subject._validate_step6b_canary_core(
            step6a_shard0_dir=step6a,
            step6b_received_dir=received,
            accepted_step6a_identity=identity,
        )


def test_accepted_step6a_trust_anchor_is_exact(
    canary: tuple[Path, Path, dict[str, object]],
):
    step6a, received, identity = canary
    wrong = dict(identity)
    wrong["done_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="accepted immutable v4 run"):
        subject._validate_step6b_canary_core(
            step6a_shard0_dir=step6a,
            step6b_received_dir=received,
            accepted_step6a_identity=wrong,
        )
