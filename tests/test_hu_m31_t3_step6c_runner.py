from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import ofc_regular.run_hu_m31_t3_step6c_shard as subject
from ofc_regular.action_key import (
    ACTION_KEY_SCHEMA,
    ActionKey,
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m31_t3_step6c_contract import (
    EXPECTED_STEP6C_CONTRACT_BYTE_SHA256,
    EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256,
    PILOT_HAND_INDICES,
    STEP6C_BEHAVIOR_SCHEDULE_SCHEMA,
    STEP6C_RUN_ID,
    canonical_sha256,
    schedule_rows,
    train_seed_values,
)
from ofc_regular.state import Board


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


def _observation(index: int, seat: str) -> ActorObservation:
    cards = list(ALL_CARDS)
    random.Random(910_000 + index * 2 + (seat == "second")).shuffle(cards)
    cursor = 0

    def take(count: int) -> tuple[str, ...]:
        nonlocal cursor
        result = tuple(cards[cursor : cursor + count])
        cursor += count
        return result

    hero = Board.from_rows(take(2), take(3), take(4))
    opponent = (
        Board.from_rows(take(2), take(3), take(4))
        if seat == "first"
        else Board.from_rows(take(2), take(4), take(5))
    )
    return ActorObservation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=take(3),
        hero_private_discards=take(2),
        seat=seat,
        street="T3",
        to_act_order=seat,
    )


def _decision(
    observation: ActorObservation,
    *,
    seeds: dict[str, int],
    confirmation: bool,
    selected_regret: float = 0.0,
) -> dict[str, Any]:
    budget = subject._CONFIRMATION_BUDGET if confirmation else subject._PRIMARY_BUDGET
    evaluation_seed_key = "confirmation" if confirmation else "evaluation"
    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    selection = [float(len(legal) - index) for index in range(len(legal))]
    selected_index = 0
    evaluation = [0.0 for _ in legal]
    evaluation[selected_index] = 10.0 - selected_regret
    if len(legal) > 1:
        evaluation[1] = 10.0
    best = max(evaluation)
    rank_order = sorted(
        range(len(legal)),
        key=lambda index: (-selection[index], action_key(legal[index]).sort_key()),
    )
    rank_by_index = {index: rank for rank, index in enumerate(rank_order)}
    values = []
    for index, action in enumerate(legal):
        values.append(
            {
                "action_key": action_key(action).to_token(),
                "original_index": index,
                "rank": rank_by_index[index],
                "placements": [list(item) for item in action.placements],
                "discards": list(action.discards),
                "selection_ev": selection[index],
                "evaluation_ev": evaluation[index],
                "evaluation_regret": best - evaluation[index],
            }
        )
    values.sort(key=lambda row: ActionKey.from_token(row["action_key"]).sort_key())
    selected = legal[selected_index]
    selected_key = action_key(selected).to_token()
    candidate_keys, candidate_belief, candidate_rng = subject._particle_evidence(
        observation,
        base_seed=seeds["candidate"],
        role="candidate_selection",
        sample_count=8,
    )
    evaluation_keys, evaluation_belief, evaluation_rng = subject._particle_evidence(
        observation,
        base_seed=seeds[evaluation_seed_key],
        role="locked_evaluation",
        sample_count=budget["evaluation_samples"],
    )
    assert not (candidate_keys & evaluation_keys)
    selection_gap = selection[selected_index] - (
        selection[1] if len(selection) > 1 else selection[selected_index]
    )
    child_information_set_count = len(legal) * budget["downstream_t3_samples"]
    search_contract = {
        "run_id": STEP6C_RUN_ID,
        "continuation_seed": seeds["child"],
        "candidate_seed": seeds["candidate"],
        "evaluation_seed": seeds[evaluation_seed_key],
        "candidate_samples": budget["candidate_samples"],
        "evaluation_samples": budget["evaluation_samples"],
        "downstream_t3_samples": budget["downstream_t3_samples"],
        "downstream_t4_samples": 0,
        "use_t4_action_cache": True,
        "continuation_policy_id": "local_infoset_response_t3_second_t4_v1",
        "strategy_fusion_guard": "child_actions_keyed_only_by_actor_observation",
        "downstream_t4_native_semantics_id": (
            "m30_exact_t4_native_kernel_semantics_v1"
        ),
    }
    search_contract_digest = subject._runtime_digest(search_contract)
    decision = {
        "schema": "hu_m31_t3_runtime_decision_v2",
        "runtime_id": "hu_m31_t3_crn_exact_t4_v1",
        "seat": observation.seat,
        "value_scope": "q_pi_uniform_exchangeable_t3_crn_with_exact_t4_children",
        "observation_fingerprint": observation.fingerprint(),
        "run_id": STEP6C_RUN_ID,
        "continuation_seed": seeds["child"],
        "candidate_seed": seeds["candidate"],
        "evaluation_seed": seeds[evaluation_seed_key],
        "candidate_samples": budget["candidate_samples"],
        "evaluation_samples": budget["evaluation_samples"],
        "downstream_t3_samples": budget["downstream_t3_samples"],
        "downstream_t4_samples": 0,
        "downstream_t4_mode": "exact",
        "use_t4_action_cache": True,
        "continuation_policy_id": "local_infoset_response_t3_second_t4_v1",
        "strategy_fusion_guard": "child_actions_keyed_only_by_actor_observation",
        "search_contract_digest": search_contract_digest,
        "downstream_t4_native_semantics_id": (
            "m30_exact_t4_native_kernel_semantics_v1"
        ),
        "downstream_t4_native_anchor": "same_pinned_m30_native_engine",
        "child_information_set_count": child_information_set_count,
        "solver_id": "rust_crn_sequential_t3_v1",
        "engine_version": "ofc_hu_m3_engine/0.1.0",
        "native_library_sha256": subject.EXPECTED_NATIVE_LIBRARY_SHA256,
        "native_latency_ms": 1.0,
        "validation_latency_ms": 0.25,
        "total_latency_ms": 1.25,
        "execution_mode": "scalar",
        "batch_size": 1,
        "teacher_value_status": "diagnostic_not_match_EV",
        "action_key_schema": ACTION_KEY_SCHEMA,
        "legal_action_set_digest": legal_action_set_digest(legal),
        "legal_action_order_digest": ordered_action_mapping_digest(legal),
        "action_values": values,
        "selected_action_key": selected_key,
        "selected_action": {
            "placements": [list(item) for item in selected.placements],
            "discards": list(selected.discards),
        },
        "selected_selection_ev": selection[selected_index],
        "selected_evaluation_ev": evaluation[selected_index],
        "selection_gap": selection_gap,
        "evaluation_sample_regret": best - evaluation[selected_index],
        "belief_prior": subject.HIDDEN_CARD_PRIOR,
        "candidate_belief_digest": candidate_belief,
        "evaluation_belief_digest": evaluation_belief,
        "candidate_rng_digest": candidate_rng,
        "evaluation_rng_digest": evaluation_rng,
        "semantic_result_digest_schema": ("hu_m31_t3_semantic_result_digest_v1"),
        "semantic_result_digest_scope": ("dealt_order_independent_action_value_result"),
        "semantic_result_digest": "",
        "result_digest_scope": "ordered_action_mapping_bound",
        "result_digest": "",
    }
    mapping_payload = {
        "runtime_id": decision["runtime_id"],
        "request_schema": "hu_m3_engine_request_v1",
        "observation_fingerprint": observation.fingerprint(),
        "selected_action_key": selected_key,
        "selected_selection_ev": selection[selected_index],
        "selected_evaluation_ev": evaluation[selected_index],
        "selection_gap": selection_gap,
        "evaluation_sample_regret": best - evaluation[selected_index],
        "legal_action_set_digest": decision["legal_action_set_digest"],
        "legal_action_order_digest": decision["legal_action_order_digest"],
        "action_values": [
            [
                row["action_key"],
                row["original_index"],
                row["rank"],
                row["selection_ev"],
                row["evaluation_ev"],
            ]
            for row in values
        ],
        "candidate_belief_digest": candidate_belief,
        "evaluation_belief_digest": evaluation_belief,
        "candidate_rng_digest": candidate_rng,
        "evaluation_rng_digest": evaluation_rng,
        "search_contract": search_contract,
        "search_contract_digest": search_contract_digest,
        "downstream_t4_native_semantics_id": decision[
            "downstream_t4_native_semantics_id"
        ],
        "child_information_set_count": child_information_set_count,
        "solver_id": decision["solver_id"],
        "engine_version": decision["engine_version"],
        "native_library_sha256": decision["native_library_sha256"],
    }
    semantic_payload = {
        "schema": decision["semantic_result_digest_schema"],
        "runtime_id": decision["runtime_id"],
        "request_schema": "hu_m3_engine_request_v1",
        "action_key_schema": ACTION_KEY_SCHEMA,
        "seat": observation.seat,
        "value_scope": decision["value_scope"],
        "observation_fingerprint": observation.fingerprint(),
        "selected_action_key": selected_key,
        "selected_selection_ev": selection[selected_index],
        "selected_evaluation_ev": evaluation[selected_index],
        "selection_gap": selection_gap,
        "evaluation_sample_regret": best - evaluation[selected_index],
        "legal_action_set_digest": decision["legal_action_set_digest"],
        "action_values": [
            [
                row["action_key"],
                row["rank"],
                row["selection_ev"],
                row["evaluation_ev"],
                row["evaluation_regret"],
            ]
            for row in values
        ],
        "candidate_belief_digest": candidate_belief,
        "evaluation_belief_digest": evaluation_belief,
        "candidate_rng_digest": candidate_rng,
        "evaluation_rng_digest": evaluation_rng,
        "search_contract": search_contract,
        "search_contract_digest": search_contract_digest,
        "sample_independence": "disjoint_particle_rng_keys",
        "downstream_t4_native_semantics_id": decision[
            "downstream_t4_native_semantics_id"
        ],
        "teacher_value_status": "diagnostic_not_match_EV",
        "child_information_set_count": child_information_set_count,
        "solver_id": decision["solver_id"],
        "engine_version": decision["engine_version"],
        "native_library_sha256": decision["native_library_sha256"],
    }
    decision["result_digest"] = subject._runtime_digest(mapping_payload)
    decision["semantic_result_digest"] = subject._runtime_digest(semantic_payload)
    return decision


def _root_and_task(
    index: int,
    *,
    confirmation_required: bool,
    selected_regret: float = 0.5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seeds = train_seed_values(index)
    observations = (_observation(index, "first"), _observation(index, "second"))
    root = {
        "schema": subject.STEP6C_ROOT_TASK_SCHEMA,
        "contract_digest": "c" * 64,
        "global_hand_index": index,
        "profile": subject.behavior_profile_for_train_index(index),
        "seeds": seeds,
        "confirmation_required": confirmation_required,
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
    for offset, observation in enumerate(observations):
        primary = _decision(observation, seeds=seeds, confirmation=False)
        confirmation_payload = None
        confirmation_seconds = None
        if confirmation_required:
            confirmation = _decision(
                observation,
                seeds=seeds,
                confirmation=True,
                selected_regret=selected_regret,
            )
            confirmation_payload = subject._confirmation_payload(primary, confirmation)
            confirmation_seconds = 2.0
        rows.append(
            {
                "root_index": index * 2 + offset,
                "global_hand_index": index,
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "legal_action_count": len(primary["action_values"]),
                "primary_wall_seconds": 1.0,
                "confirmation_wall_seconds": confirmation_seconds,
                "wall_seconds": 1.0 + (confirmation_seconds or 0.0),
                "decision": primary,
                "confirmation": confirmation_payload,
                "integrity": {key: True for key in subject._ROW_INTEGRITY_KEYS},
            }
        )
    task = {
        "schema": subject.STEP6C_SEARCH_TASK_SCHEMA,
        "contract_digest": root["contract_digest"],
        "global_hand_index": index,
        "profile": root["profile"],
        "seeds": seeds,
        "confirmation_required": confirmation_required,
        "process_id": 123,
        "rayon_threads": str(subject.RAYON_THREADS),
        "wall_seconds": sum(row["wall_seconds"] for row in rows),
        "memory": {"peak_rss_bytes": 1024},
        "engine": {
            "version": "ofc_hu_m3_engine/0.1.0",
            "library_sha256": subject.EXPECTED_NATIVE_LIBRARY_SHA256,
            "build_or_fallback": False,
        },
        "rows": rows,
        "all_gates_passed": True,
        "teacher_value_status": "diagnostic_not_match_EV",
        "training_eligible": False,
        "current_profile_changed": False,
        "production_fanout_authorized": False,
    }
    return root, task


def _manifest_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    schedule_path = tmp_path / "schedule.jsonl"
    schedule_path.write_bytes(
        b"".join(_json_bytes(row) for row in schedule_rows(PILOT_HAND_INDICES))
    )
    source_sha = "a" * 64
    manifest = {
        "schema": subject.STEP6C_PACKAGE_SCHEMA,
        "status": "packaged_local_no_gcloud",
        "run_name": "step6c-test",
        "source_sha256": source_sha,
        "schedule_sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
        "total_shards": 2,
        "authorized_shards": [0, 1],
        "paired_hands_per_shard": 25,
        "roots_per_shard": 50,
        "step5_contract_canonical_sha256": subject.STEP5_CONTRACT_CANONICAL_SHA256,
        "step6c_run_id": STEP6C_RUN_ID,
        "production_label_budget": subject._PRIMARY_BUDGET,
        "confirmation_budget": subject._CONFIRMATION_BUDGET,
        "pilot_hand_indices": list(PILOT_HAND_INDICES),
        "confirmation_hand_indices": [5, 16, 29, 39, 45],
        "behavior_schedule_schema": STEP6C_BEHAVIOR_SCHEDULE_SCHEMA,
        "teacher_schedule_canonical_sha256": canonical_sha256(
            schedule_rows(PILOT_HAND_INDICES)
        ),
        "quality_pilot_authorized": True,
        "pilot_rows_training_eligible": False,
        "production_fanout_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "m31_complete": False,
        "parity_golden_sha256": "b" * 64,
        "step6b_status_sha256": "c" * 64,
        "step6b_validation_sha256": "d" * 64,
        "step6b_receive_receipt_sha256": "e" * 64,
        "step6c_contract_byte_sha256": EXPECTED_STEP6C_CONTRACT_BYTE_SHA256,
        "step6c_contract_canonical_sha256": (EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256),
        "step6c_validation_sha256": "f" * 64,
        "native_library": {
            "path": "native/release/libofc_hu_m3_engine.so",
            "sha256": subject.EXPECTED_NATIVE_LIBRARY_SHA256,
            "engine_version": "ofc_hu_m3_engine/0.1.0",
        },
        "feature_encoder_library": {
            "path": "target/release/libofc_stage3_feature_encoder.so",
            "sha256": subject.EXPECTED_FEATURE_ENCODER_SHA256,
        },
        "models": subject.EXPECTED_MODELS,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    return manifest_path, schedule_path, source_sha


def test_step6c_shards_are_exactly_zero_and_one():
    assert subject.ShardSpec("run", 0, tuple(range(25))).root_count == 50
    assert subject.ShardSpec("run", 1, tuple(range(25, 50))).root_count == 50
    with pytest.raises(ValueError, match="shards 0 and 1 only"):
        subject.ShardSpec("run", 2, tuple(range(50, 75)))
    with pytest.raises(ValueError, match="schedule changed"):
        subject.ShardSpec("run", 1, tuple(range(24, 49)))


def test_step6c_manifest_binds_science_schedule_and_two_shards(tmp_path: Path):
    manifest, schedule, source_sha = _manifest_fixture(tmp_path)
    payload, spec, digest = subject._load_manifest_and_spec(
        manifest_path=manifest,
        schedule_path=schedule,
        source_package_sha256=source_sha,
        shard=1,
    )
    assert spec.hand_indices == tuple(range(25, 50))
    assert payload["teacher_schedule_canonical_sha256"] == canonical_sha256(
        schedule_rows(PILOT_HAND_INDICES)
    )
    assert len(digest) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("production_label_budget", {"candidate_samples": 4}),
        ("confirmation_hand_indices", [5, 16]),
        ("quality_pilot_authorized", False),
        ("pilot_rows_training_eligible", True),
        ("production_fanout_authorized", True),
    ],
)
def test_step6c_manifest_mutations_fail_closed(tmp_path: Path, field: str, value: Any):
    manifest, schedule, source_sha = _manifest_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload[field] = value
    manifest.write_bytes(_json_bytes(payload))
    with pytest.raises(ValueError, match="manifest or schedule boundary changed"):
        subject._load_manifest_and_spec(
            manifest_path=manifest,
            schedule_path=schedule,
            source_package_sha256=source_sha,
            shard=0,
        )


def test_primary_and_confirmation_share_candidate_but_lock_primary_action():
    index = 5
    seeds = train_seed_values(index)
    observation = _observation(index, "first")
    primary = _decision(observation, seeds=seeds, confirmation=False)
    confirmation = _decision(
        observation, seeds=seeds, confirmation=True, selected_regret=0.5
    )
    primary_evidence = subject._validate_decision_payload(
        primary,
        observation=observation,
        seeds=seeds,
        budget=subject._PRIMARY_BUDGET,
        evaluation_seed_key="evaluation",
        native_library_sha256=subject.EXPECTED_NATIVE_LIBRARY_SHA256,
    )
    confirmation_evidence = subject._validate_decision_payload(
        confirmation,
        observation=observation,
        seeds=seeds,
        budget=subject._CONFIRMATION_BUDGET,
        evaluation_seed_key="confirmation",
        native_library_sha256=subject.EXPECTED_NATIVE_LIBRARY_SHA256,
    )
    payload = subject._confirmation_payload(primary, confirmation)
    subject._validate_confirmation_pair(primary, confirmation, payload)
    assert primary_evidence.candidate_keys == confirmation_evidence.candidate_keys
    assert not (primary_evidence.candidate_keys & confirmation_evidence.evaluation_keys)
    assert payload["locked_selected_action_key"] == primary["selected_action_key"]
    assert payload["selected_regret"] == pytest.approx(0.5)


def test_confirmation_may_not_replace_primary_action_or_candidate_values():
    index = 5
    seeds = train_seed_values(index)
    observation = _observation(index, "first")
    primary = _decision(observation, seeds=seeds, confirmation=False)
    confirmation = _decision(observation, seeds=seeds, confirmation=True)
    payload = subject._confirmation_payload(primary, confirmation)
    mutated = deepcopy(confirmation)
    mutated["action_values"][0]["selection_ev"] += 0.25
    payload["decision"] = mutated
    with pytest.raises(ValueError, match="changed primary selection semantics"):
        subject._validate_confirmation_pair(primary, mutated, payload)


@pytest.mark.parametrize("confirmation_required", [False, True])
def test_recovered_task_revalidates_all_actions_rng_and_confirmation(
    confirmation_required: bool,
):
    index = 5 if confirmation_required else 0
    root, task = _root_and_task(index, confirmation_required=confirmation_required)
    rows = subject._validate_search_task(
        task,
        root_task=root,
        library_sha256=subject.EXPECTED_NATIVE_LIBRARY_SHA256,
    )
    assert len(rows) == 2
    assert all(len(row.candidate_keys) == 8 for row in rows)
    assert all(len(row.evaluation_keys) == 32 for row in rows)
    assert all(
        len(row.confirmation_keys) == (128 if confirmation_required else 0)
        for row in rows
    )


def test_recovered_task_rejects_hidden_field_and_rng_tampering():
    root, task = _root_and_task(5, confirmation_required=True)
    hidden = deepcopy(task)
    hidden["rows"][0]["decision"]["opponent_private_discards"] = ["As"]
    with pytest.raises(ValueError, match="forbidden hidden-information"):
        subject._validate_search_task(
            hidden,
            root_task=root,
            library_sha256=subject.EXPECTED_NATIVE_LIBRARY_SHA256,
        )
    rng = deepcopy(task)
    rng["rows"][0]["confirmation"]["decision"]["evaluation_rng_digest"] = "0" * 64
    with pytest.raises(ValueError, match="result certificate changed"):
        subject._validate_search_task(
            rng,
            root_task=root,
            library_sha256=subject.EXPECTED_NATIVE_LIBRARY_SHA256,
        )


def test_recovered_task_rejects_unknown_decision_field_and_stale_q_certificate():
    root, task = _root_and_task(0, confirmation_required=False)
    unknown = deepcopy(task)
    unknown["rows"][0]["decision"]["unbound_note"] = "not certified"
    with pytest.raises(ValueError, match="decision is missing"):
        subject._validate_search_task(
            unknown,
            root_task=root,
            library_sha256=subject.EXPECTED_NATIVE_LIBRARY_SHA256,
        )

    shifted = deepcopy(task)
    decision = shifted["rows"][0]["decision"]
    for row in decision["action_values"]:
        row["selection_ev"] += 1_000.0
        row["evaluation_ev"] += 1_000.0
    decision["selected_selection_ev"] += 1_000.0
    decision["selected_evaluation_ev"] += 1_000.0
    with pytest.raises(ValueError, match="result certificate changed"):
        subject._validate_search_task(
            shifted,
            root_task=root,
            library_sha256=subject.EXPECTED_NATIVE_LIBRARY_SHA256,
        )


def test_recovered_task_rejects_unbound_belief_prior_tampering():
    root, task = _root_and_task(0, confirmation_required=False)
    task["rows"][0]["decision"]["belief_prior"] = "tampered-unbound-prior"
    with pytest.raises(ValueError, match="decision contract changed"):
        subject._validate_search_task(
            task,
            root_task=root,
            library_sha256=subject.EXPECTED_NATIVE_LIBRARY_SHA256,
        )


def test_recovered_task_rejects_candidate_rank_tampering():
    root, task = _root_and_task(0, confirmation_required=False)
    values = task["rows"][0]["decision"]["action_values"]
    values[0]["rank"], values[1]["rank"] = values[1]["rank"], values[0]["rank"]
    with pytest.raises(ValueError, match="rank/tie-break mapping changed"):
        subject._validate_search_task(
            task,
            root_task=root,
            library_sha256=subject.EXPECTED_NATIVE_LIBRARY_SHA256,
        )


def test_stop_drill_prefers_bounded_nonconfirmation_but_full_run_keeps_priority():
    tasks = [{"global_hand_index": index} for index in (5, 0, 16, 1)]
    assert [
        row["global_hand_index"]
        for row in subject._ordered_missing_tasks(tasks, resume_drill=True)
    ] == [0, 1, 5, 16]
    assert [
        row["global_hand_index"]
        for row in subject._ordered_missing_tasks(tasks, resume_drill=False)
    ] == [5, 16, 0, 1]
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert 'multiprocessing.get_context("spawn")' in source


def test_recovered_root_is_bound_to_regenerated_seed_observations():
    root, _task = _root_and_task(0, confirmation_required=False)
    expected = tuple(
        ActorObservation.from_dict(row["observation"]) for row in root["observations"]
    )
    subject._validate_root_task(
        root,
        digest=str(root["contract_digest"]),
        global_hand_index=0,
        expected_observations=expected,
    )
    replacement = _observation(999, "first")
    tampered = deepcopy(root)
    tampered["observations"][0] = {
        "seat": "first",
        "observation_fingerprint": replacement.fingerprint(),
        "observation": replacement.to_dict(),
    }
    with pytest.raises(ValueError, match="root observation changed"):
        subject._validate_root_task(
            tampered,
            digest=str(root["contract_digest"]),
            global_hand_index=0,
            expected_observations=expected,
        )


def test_final_summary_is_reused_exactly_and_tampering_fails(tmp_path: Path):
    path = tmp_path / "summary.json"
    summary = {"schema": subject.STEP6C_SUMMARY_SCHEMA, "status": "pass"}
    committed = subject._commit_or_validate_final_summary(path, summary, None)
    assert committed == summary
    assert subject._commit_or_validate_final_summary(path, summary, summary) == summary
    with pytest.raises(ValueError, match="final summary content changed"):
        subject._commit_or_validate_final_summary(
            path,
            {**summary, "status": "no_go"},
            summary,
        )


def test_cli_can_write_a_canonical_resume_report():
    parsed = subject._parser().parse_args(
        [
            "--manifest",
            "manifest.json",
            "--schedule",
            "schedule.jsonl",
            "--source-package-sha256",
            "a" * 64,
            "--repository-root",
            ".",
            "--shard",
            "0",
            "--output-dir",
            "result",
            "--parity-golden",
            "golden.json",
            "--report-output",
            "resume-report.json",
        ]
    )
    assert parsed.report_output == Path("resume-report.json")
