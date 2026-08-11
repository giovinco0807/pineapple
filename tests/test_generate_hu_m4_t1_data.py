from __future__ import annotations

import json

import pytest

import ofc_regular.generate_hu_m4_t1_data as data
from ofc_regular.action_key import action_key
from ofc_regular.action_key import legal_action_set_digest, ordered_action_mapping_digest
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m4_t1_teacher import M4T1TeacherConfig
from ofc_regular.policy import RegularAiPolicy


def _policies():
    return {
        "first": RegularAiPolicy(seat="first", seed=1),
        "second": RegularAiPolicy(seat="second", seed=2),
    }


def _paired_summary(delta: float, *, count: int = 1) -> dict:
    negative = float(delta < 0.0)
    return {
        "schema": "hu_m4_paired_delta_summary_v1",
        "count": count,
        "mean": float(delta),
        "standard_error": 0.0,
        "std": 0.0,
        "min": float(delta),
        "p01": float(delta),
        "p05": float(delta),
        "p25": float(delta),
        "p50": float(delta),
        "p75": float(delta),
        "p95": float(delta),
        "p99": float(delta),
        "max": float(delta),
        "lt0_rate": negative,
        "le_neg6_rate": float(delta <= -6.0),
        "le_neg12_rate": float(delta <= -12.0),
        "le_neg20_rate": float(delta <= -20.0),
    }


def test_generate_second_root_is_live_geometry_and_hidden_safe() -> None:
    root = data.generate_t1_second_root(101, root_policies=_policies())
    assert (root.street, root.seat, root.to_act_order) == ("T1", "second", "second")
    assert root.hero_board.card_count() == 5
    assert root.opponent_public_board.card_count() == 7
    assert root.hero_private_discards == ()
    assert "opponent_private_discards" not in root.to_dict()


def test_config_rejects_current_and_overlapping_rng() -> None:
    with pytest.raises(ValueError, match="current"):
        data.M4T1DataConfig(root_profile="current")
    with pytest.raises(ValueError, match="distinct"):
        data.M4T1DataConfig(candidate_seed=7, evaluation_seed=7)


def test_root_population_schedule_is_deterministic_and_quota_balanced() -> None:
    config = data.M4T1DataConfig(
        roots=20,
        seed_start=551,
        root_profiles=(
            "stage19_p0",
            "stage9f_p2",
            "stage7_m5_r10",
            "stage3_baseline",
            "random_exact_final",
        ),
        root_profile_weights=(4.0, 3.0, 2.0, 1.0, 0.0 + 2.0),
    )
    schedule = config.root_profile_schedule()
    assert schedule == config.root_profile_schedule()
    assert len(schedule) == 20
    counts = {profile: schedule.count(profile) for profile in set(schedule)}
    assert counts == {
        "stage19_p0": 7,
        "stage9f_p2": 5,
        "stage7_m5_r10": 3,
        "stage3_baseline": 2,
        "random_exact_final": 3,
    }
    changed_seed = data.M4T1DataConfig(
        roots=20,
        seed_start=552,
        root_profiles=config.root_profiles,
        root_profile_weights=config.root_profile_weights,
    )
    assert changed_seed.root_profile_schedule() != schedule
    manifest = data._root_population_manifest(config)
    assert sum(row["target_count"] for row in manifest["profiles"]) == 20
    assert all(row["target_count"] == row["completed_count"] for row in manifest["profiles"])


def test_root_population_rejects_current_bad_weights_and_cli_parses_population() -> None:
    with pytest.raises(ValueError, match="current"):
        data.M4T1DataConfig(root_profiles=("stage3_baseline", "current"))
    with pytest.raises(ValueError, match="match"):
        data.M4T1DataConfig(
            root_profiles=("stage3_baseline", "random_exact_final"),
            root_profile_weights=(1.0,),
        )
    with pytest.raises(ValueError, match="positive"):
        data.M4T1DataConfig(
            root_profiles=("stage3_baseline",), root_profile_weights=(0.0,)
        )
    args = data.parse_args(
        [
            "--output", "out.jsonl",
            "--checkpoint", "checkpoint.json",
            "--heartbeat", "heartbeat.json",
            "--root-profiles", "stage19_p0,stage3_baseline", "random_exact_final",
            "--root-weights", "2,1", "1",
        ]
    )
    assert data._split_cli_values(args.root_profiles) == (
        "stage19_p0",
        "stage3_baseline",
        "random_exact_final",
    )
    assert tuple(float(value) for value in data._split_cli_values(args.root_profile_weights)) == (
        2.0,
        1.0,
        1.0,
    )


def test_teacher_result_conversion_uses_locked_eval_and_no_hidden_truth(monkeypatch) -> None:
    root = data.generate_t1_second_root(102, root_policies=_policies())
    actions = generate_turn_actions(root.hero_board, root.dealt_cards)
    baseline = actions[-1]
    baseline_index = len(actions) - 1
    rows = []
    for index, action in enumerate(actions):
        rows.append(
            {
                "original_index": index,
                "action_key": action_key(action).to_token(),
                "evaluation_score": float(index),
                "evaluation_standard_error": 0.25,
                "selection_score": float(len(actions) - index),
                "selection_standard_error": 0.5,
                "selected_by_candidate_plan": index == 0,
                "evaluation_delta_vs_baseline": _paired_summary(
                    float(index - baseline_index)
                ),
            }
        )
    result = {
        "status": "ok",
        "schema": "hu_m4_t1_second_teacher_v1",
        "observation_fingerprint": root.fingerprint(),
        "street": "T1",
        "seat": "second",
        "to_act_order": "second",
        "legal_action_count": len(actions),
        "legal_action_set_digest": legal_action_set_digest(actions),
        "legal_action_order_digest": ordered_action_mapping_digest(actions),
        "actions": rows,
        "selected_action_key": rows[0]["action_key"],
        "selected_action_evaluation_score": 0.0,
        "selected_action_evaluation_standard_error": 0.25,
        "evaluation_sample_regret_of_locked_selection": float(len(actions) - 1),
        "action_key_schema": "regular_action_key_v1",
        "candidate_belief_digest": "candidate",
        "evaluation_belief_digest": "evaluation",
        "candidate_rng_key_digests": ["a"],
        "evaluation_rng_key_digests": ["b"],
        "sample_independence": "disjoint_particle_rng_keys",
        "root_selection_lock": "candidate_action_key_locked_before_evaluation",
        "continuation_policy": {},
        "search_config": {"candidate_samples": 1, "evaluation_samples": 1},
        "paired_delta_baseline_action_key": action_key(baseline).to_token(),
        "paired_delta_common_futures": True,
    }
    sample = data.teacher_result_to_sample(
        root,
        result,
        hand_seed=102,
        root_index=0,
        split="train",
        baseline_action=baseline,
        provenance={"current_profile_resolved": False},
    )
    assert sample["actions"][0]["score"] == float(len(actions) - 1)
    assert sample["best_action"] == 0
    assert sample["teacher_value_status"] == "diagnostic_not_match_EV"
    assert sample["schema"] == "hu_m4_t1_second_training_sample_v2"
    assert sample["actions"][sample["baseline_action_row_index"]]["action_key"] == action_key(baseline).to_token()
    baseline_row = sample["actions"][sample["baseline_action_row_index"]]
    assert baseline_row["delta_vs_baseline"] == 0.0
    assert baseline_row["delta_se_vs_baseline"] == 0.0
    assert baseline_row["paired_delta_vs_baseline"]["count"] == 1
    assert sample["paired_delta_contract"] == {
        "schema": "hu_m4_paired_delta_summary_v1",
        "baseline_action_key": action_key(baseline).to_token(),
        "common_evaluation_futures": True,
        "evaluation_samples": 1,
    }
    serialized = json.dumps(sample)
    assert "opponent_private_discards" not in serialized

    wrong = dict(result)
    wrong["observation_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="different observation"):
        data.teacher_result_to_sample(
            root,
            wrong,
            hand_seed=102,
            root_index=0,
            split="train",
            baseline_action=baseline,
            provenance={"current_profile_resolved": False},
        )


def test_resume_truncates_uncheckpointed_torn_tail_and_rejects_config_change(tmp_path):
    import hashlib

    config = data.M4T1DataConfig(roots=2, split="train", seed_start=100)
    row = {
        "schema": data.M4_T1_DATA_SCHEMA,
        "split": "train",
        "root_index": 0,
        "hand_seed": 100,
        "observation_fingerprint": "fingerprint",
        "provenance": {
            "root_profile": config.root_profile_for(0),
            "root_policy_family": data._root_policy_family(
                config.root_profile_for(0)
            ),
            "root_population_schedule": data.ROOT_POPULATION_SCHEDULE,
            "root_population_schedule_sha256": data._root_population_manifest(
                config
            )["schedule_sha256"],
            "baseline_profile": config.baseline_profile,
            "t2_profile": config.t2_profile,
            "native_batch_threads": config.native_batch_threads,
            "current_profile_resolved": False,
        },
        "search_config": {
            "candidate_samples": config.candidate_samples,
            "evaluation_samples": config.evaluation_samples,
            "candidate_seed": config.candidate_seed,
            "evaluation_seed": config.evaluation_seed,
            "child_policy_seed": config.child_policy_seed,
            "batch_child_selectors": config.batch_child_selectors,
            "run_id": (
                f"{config.run_id}:split=train:root=0:seed=100:obs=fingerprint"
            ),
        },
    }
    committed = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    partial = tmp_path / "shard.jsonl.partial"
    partial.write_bytes(committed + b'{"root_index":1')
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema": data.M4_T1_CHECKPOINT_SCHEMA,
                "config_sha256": data._config_sha256(config),
                "completed_roots": 1,
                "partial_sha256": hashlib.sha256(committed).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    assert data._resume_completed_roots(
        partial,
        checkpoint,
        config=config,
        config_sha256=data._config_sha256(config),
    ) == 1
    assert partial.read_bytes() == committed

    changed = data.M4T1DataConfig(roots=2, split="train", seed_start=101)
    with pytest.raises(ValueError, match="configuration"):
        data._resume_completed_roots(
            partial,
            checkpoint,
            config=changed,
            config_sha256=data._config_sha256(changed),
        )


def test_resume_rejects_legacy_v1_rows_even_with_matching_config_hash(
    tmp_path,
) -> None:
    import hashlib

    config = data.M4T1DataConfig(roots=1)
    raw = b'{"schema":"hu_m4_t1_second_training_sample_v1"}\n'
    partial = tmp_path / "legacy.jsonl.partial"
    partial.write_bytes(raw)
    checkpoint = tmp_path / "legacy.checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema": data.M4_T1_CHECKPOINT_SCHEMA,
                "config_sha256": data._config_sha256(config),
                "completed_roots": 1,
                "partial_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        data._resume_completed_roots(
            partial,
            checkpoint,
            config=config,
            config_sha256=data._config_sha256(config),
        )


def test_resume_rejects_checkpointed_seed_or_index_tampering(tmp_path):
    import hashlib

    config = data.M4T1DataConfig(roots=1, split="train", seed_start=100)
    row = {
        "schema": data.M4_T1_DATA_SCHEMA,
        "split": "train",
        "root_index": 9,
        "hand_seed": 999,
        "observation_fingerprint": "fingerprint",
        "provenance": {
            "root_profile": config.root_profile_for(0),
            "root_policy_family": data._root_policy_family(
                config.root_profile_for(0)
            ),
            "root_population_schedule": data.ROOT_POPULATION_SCHEDULE,
            "root_population_schedule_sha256": data._root_population_manifest(
                config
            )["schedule_sha256"],
            "baseline_profile": config.baseline_profile,
            "t2_profile": config.t2_profile,
            "native_batch_threads": config.native_batch_threads,
            "current_profile_resolved": False,
        },
        "search_config": {},
    }
    raw = (json.dumps(row) + "\n").encode()
    partial = tmp_path / "shard.partial"
    partial.write_bytes(raw)
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema": data.M4_T1_CHECKPOINT_SCHEMA,
                "config_sha256": data._config_sha256(config),
                "completed_roots": 1,
                "partial_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="root_index"):
        data._resume_completed_roots(
            partial,
            checkpoint,
            config=config,
            config_sha256=data._config_sha256(config),
        )


def test_resume_rejects_mixed_root_population_provenance(tmp_path) -> None:
    import hashlib

    config = data.M4T1DataConfig(
        roots=2,
        split="train",
        seed_start=900,
        root_profiles=("stage3_baseline", "random_exact_final"),
        root_profile_weights=(1.0, 1.0),
    )
    expected = config.root_profile_for(0)
    wrong = next(profile for profile in config.root_profiles if profile != expected)
    row = {
        "schema": data.M4_T1_DATA_SCHEMA,
        "split": "train",
        "root_index": 0,
        "hand_seed": 900,
        "observation_fingerprint": "fingerprint",
        "provenance": {
            "root_profile": wrong,
            "root_policy_family": data._root_policy_family(wrong),
            "root_population_schedule": data.ROOT_POPULATION_SCHEDULE,
            "root_population_schedule_sha256": data._root_population_manifest(
                config
            )["schedule_sha256"],
            "baseline_profile": config.baseline_profile,
            "t2_profile": config.t2_profile,
            "native_batch_threads": config.native_batch_threads,
            "current_profile_resolved": False,
        },
        "search_config": {
            "candidate_samples": config.candidate_samples,
            "evaluation_samples": config.evaluation_samples,
            "candidate_seed": config.candidate_seed,
            "evaluation_seed": config.evaluation_seed,
            "child_policy_seed": config.child_policy_seed,
            "run_id": (
                f"{config.run_id}:split=train:root=0:seed=900:obs=fingerprint"
            ),
        },
    }
    raw = (json.dumps(row) + "\n").encode()
    partial = tmp_path / "mixed.jsonl.partial"
    partial.write_bytes(raw)
    checkpoint = tmp_path / "mixed-checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema": data.M4_T1_CHECKPOINT_SCHEMA,
                "config_sha256": data._config_sha256(config),
                "completed_roots": 1,
                "partial_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="policy provenance"):
        data._resume_completed_roots(
            partial,
            checkpoint,
            config=config,
            config_sha256=data._config_sha256(config),
        )


def test_teacher_api_does_not_accept_external_particle_truth() -> None:
    root = data.generate_t1_second_root(103, root_policies=_policies())
    with pytest.raises(TypeError, match="unexpected keyword"):
        from ofc_regular.hu_m4_t1_teacher import evaluate_t1_second_actions

        evaluate_t1_second_actions(
            root,
            t2_policies=_policies(),
            config=M4T1TeacherConfig(candidate_samples=1, evaluation_samples=1),
            opponent_private_discards=("As",),  # type: ignore[call-arg]
        )
