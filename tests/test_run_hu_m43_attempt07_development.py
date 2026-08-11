from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import ofc_regular.run_hu_m43_attempt07_development as runner
from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt06_teacher import ATTEMPT06_FROZEN_MODEL_SHA256
from ofc_regular.hu_m43_attempt07_contract import (
    M43_ATTEMPT07_PLAN_SHA256,
    M43_ATTEMPT07_PROFILES,
    enumerate_attempt07_seed_schedules,
    load_and_validate_attempt07_plan,
)
from ofc_regular.hu_m43_attempt07_teacher import ATTEMPT07_TEACHER_SCHEMA
from ofc_regular.state import Board


def _root() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("9h",), middle=("Th", "Jh"), bottom=("Qh", "Kh")
        ),
        opponent_public_board=Board.from_rows(
            top=("2h",),
            middle=("3h", "4h", "5h"),
            bottom=("6h", "7h", "8h"),
        ),
        dealt_cards=("Ah", "2d", "3d"),
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )


class _Policy:
    def __init__(self, profile: str, seat: str, seed: int) -> None:
        self.profile = profile
        self.seat = seat
        self.seed = seed
        if profile == "stage9f_p2":
            self.topk_context = {
                "runtime_profile": "stage9f_p2",
                "runtime_status": "p2_fixed",
            }


class _Ranker:
    artifact_sha256 = ATTEMPT06_FROZEN_MODEL_SHA256
    model_id = "fake-frozen-attempt06-lambda"


def _paths(tmp_path: Path, prefix: str = "result") -> dict[str, Path]:
    model = tmp_path / "frozen-lambda.pkl"
    model.touch(exist_ok=True)
    return {
        "output": tmp_path / f"{prefix}.jsonl",
        "checkpoint": tmp_path / f"{prefix}.checkpoint.json",
        "heartbeat": tmp_path / f"{prefix}.heartbeat.json",
        "model": model,
    }


def _mock_heavy(monkeypatch, *, teacher_extra=None, captures=None) -> None:
    captures = captures if captures is not None else {}

    def load_bundle(_paths, *, profiles):
        captures.setdefault("loaded_profiles", []).append(set(profiles))
        assert "current" not in profiles
        return object()

    def build(profile, _bundle, *, seed, seat, opening_lookahead_samples):
        assert opening_lookahead_samples == 0
        captures.setdefault("builds", []).append((profile, seat, seed))
        assert profile != "current"
        return _Policy(profile, seat, seed)

    def generate(hand_seed, *, root_policies):
        captures.setdefault("generated", []).append(
            (
                hand_seed,
                {
                    seat: policy.profile
                    for seat, policy in root_policies.items()
                },
            )
        )
        return _root()

    def choose(policy, observation, **kwargs):
        assert policy.profile == "stage18_p1"
        assert observation == _root()
        captures.setdefault("baseline_kwargs", []).append(kwargs)
        return generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )[-1]

    def evaluate(
        observation,
        *,
        baseline_action_key,
        ranker,
        t2_policies,
        config,
    ):
        assert observation == _root()
        assert ranker.artifact_sha256 == ATTEMPT06_FROZEN_MODEL_SHA256
        assert action_key(
            generate_turn_actions(observation.hero_board, observation.dealt_cards)[-1]
        ).to_token() == baseline_action_key
        assert {policy.profile for policy in t2_policies.values()} == {"stage9f_p2"}
        captures.setdefault("teacher_configs", []).append(config)
        payload = {
            "status": "ok",
            "schema": ATTEMPT07_TEACHER_SCHEMA,
            "observation_fingerprint": observation.fingerprint(),
            "street": "T1",
            "seat": "second",
            "to_act_order": "second",
            "baseline_action_key": baseline_action_key,
            "teacher_value_status": "diagnostic_not_match_EV",
            "runtime_gate_allowed": False,
            "profile_activation_allowed": False,
            "current_profile_resolved": False,
            "development_only": True,
        }
        payload.update(teacher_extra or {})
        return payload

    monkeypatch.setattr(runner, "load_model_bundle", load_bundle)
    monkeypatch.setattr(runner, "build_policy", build)
    monkeypatch.setattr(runner, "generate_t1_second_root", generate)
    monkeypatch.setattr(runner, "_choose_from_observation", choose)
    monkeypatch.setattr(runner, "evaluate_attempt07_t1_second", evaluate)
    monkeypatch.setattr(
        runner, "_require_concrete_stage9f_p2_policies", lambda policies: None
    )
    monkeypatch.setattr(
        runner.FrozenAttempt06LambdaRanker,
        "load",
        classmethod(lambda cls, path, *, expected_sha256: _Ranker()),
    )


def _run(tmp_path: Path, *, root_index=7, prefix="result", **overrides):
    paths = _paths(tmp_path, prefix)
    values = {
        **paths,
        "root_index": root_index,
        "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
        "run_id": "attempt07-development-test",
        "batch_child_selectors": True,
        "native_batch_threads": 4,
    }
    values.update(overrides)
    summary = runner.run_development_shard(**values)
    return paths, summary


def test_runner_derives_seed_profile_and_explicit_policy_mapping(
    tmp_path, monkeypatch
) -> None:
    captures: dict[str, list] = {}
    _mock_heavy(monkeypatch, captures=captures)
    paths, summary = _run(
        tmp_path,
        root_index=7,
        batch_child_selectors=True,
        native_batch_threads=4,
    )
    row = json.loads(paths["output"].read_text(encoding="utf-8"))
    plan = load_and_validate_attempt07_plan(runner.DEFAULT_PLAN_PATH)
    schedules = enumerate_attempt07_seed_schedules(plan, population="development")
    expected_seeds = {domain: values[7] for domain, values in schedules.items()}

    assert row["root_index"] == 7
    assert row["hand_seed"] == expected_seeds["hand"]
    assert row["root_profile"] == M43_ATTEMPT07_PROFILES[7 % 5]
    assert row["provenance"]["seeds"] == expected_seeds
    config = captures["teacher_configs"][0]
    assert (
        config.screen_seed,
        config.rerank_seed,
        config.veto_seed,
        config.assessment_seed,
        config.child_policy_seed,
    ) == tuple(
        expected_seeds[name]
        for name in ("screen", "rerank", "veto", "assessment", "child")
    )
    assert captures["loaded_profiles"] == [
        {row["root_profile"], "stage18_p1", "stage9f_p2"}
    ]
    assert captures["generated"] == [
        (
            expected_seeds["hand"],
            {"first": row["root_profile"], "second": row["root_profile"]},
        )
    ]
    assert summary["status"] == "complete"
    assert not paths["output"].with_name(paths["output"].name + ".partial").exists()
    checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
    heartbeat = json.loads(paths["heartbeat"].read_text(encoding="utf-8"))
    assert checkpoint["completed_roots"] == 1
    assert heartbeat["status"] == "complete"


def test_provenance_is_frozen_never_current_and_output_is_hidden_safe(
    tmp_path, monkeypatch
) -> None:
    _mock_heavy(monkeypatch)
    paths, _ = _run(tmp_path, root_index=99)
    raw = paths["output"].read_bytes()
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    row = json.loads(raw)
    provenance = row["provenance"]
    assert provenance["plan_sha256"] == M43_ATTEMPT07_PLAN_SHA256
    assert provenance["ai_profiles_sha256"] == runner.AI_PROFILES_SHA256
    assert provenance["model_sha256"] == ATTEMPT06_FROZEN_MODEL_SHA256
    assert provenance["profile_assignment"] == (
        "root_index_mod_5_in_frozen_profile_order"
    )
    assert provenance["baseline_profile"] == "stage18_p1"
    assert provenance["continuation_profile"] == "stage9f_p2"
    assert provenance["root_generation_policy"] == (
        runner.ATTEMPT07_ROOT_GENERATION_POLICY
    )
    assert "current" not in provenance["explicit_loaded_profiles"]
    assert provenance["current_profile_resolved"] is False
    assert provenance["opponent_private_discard_input_allowed"] is False
    assert provenance["teacher_values_are_realized_match_ev"] is False
    assert provenance["runtime_activation_allowed"] is False
    assert row["policy_observation"] == _root().to_dict()
    assert b"opponent_private_discards" not in raw


def test_completed_partial_resumes_without_reopening_heavy_work(
    tmp_path, monkeypatch
) -> None:
    _mock_heavy(monkeypatch)
    original, _ = _run(tmp_path, prefix="original")
    resumed = _paths(tmp_path, "resumed")
    partial = resumed["output"].with_name(resumed["output"].name + ".partial")
    shutil.copyfile(original["output"], partial)
    shutil.copyfile(original["checkpoint"], resumed["checkpoint"])
    expected = original["output"].read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("completed resume reopened heavy work")

    monkeypatch.setattr(runner, "load_model_bundle", forbidden)
    summary = runner.run_development_shard(
        root_index=7,
        **resumed,
        model_sha256=ATTEMPT06_FROZEN_MODEL_SHA256,
        run_id="attempt07-development-test",
        batch_child_selectors=True,
        native_batch_threads=4,
    )
    assert resumed["output"].read_bytes() == expected
    assert summary["output_sha256"] == hashlib.sha256(expected).hexdigest()


def test_resume_rejects_config_or_hash_mismatch(tmp_path) -> None:
    partial = tmp_path / "result.jsonl.partial"
    partial.write_bytes(b"")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema": runner.ATTEMPT07_CHECKPOINT_SCHEMA,
                "config_sha256": "a" * 64,
                "plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
                "ai_profiles_sha256": runner.AI_PROFILES_SHA256,
                "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
                "target_roots": 1,
                "root_index": 0,
                "completed_roots": 0,
                "partial_sha256": hashlib.sha256(b"").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="configuration or hash mismatch"):
        runner._resume_one_root(
            partial,
            checkpoint,
            config_sha256="b" * 64,
            plan_sha256=M43_ATTEMPT07_PLAN_SHA256,
            ai_profiles_sha256=runner.AI_PROFILES_SHA256,
            model_sha256=ATTEMPT06_FROZEN_MODEL_SHA256,
            expected_root_index=0,
        )


def test_paths_are_distinct_and_plan_byte_hash_is_bound(tmp_path, monkeypatch) -> None:
    model = tmp_path / "model.pkl"
    model.touch()
    output = tmp_path / "aliased.jsonl"
    with pytest.raises(ValueError, match="paths must be distinct"):
        runner.run_development_shard(
            root_index=0,
            output=output,
            checkpoint=tmp_path / "checkpoint.json",
            heartbeat=output,
            model=model,
            model_sha256=ATTEMPT06_FROZEN_MODEL_SHA256,
            run_id="path-test",
        )
    assert not output.with_name(output.name + ".lock").exists()

    _mock_heavy(monkeypatch)
    tampered = tmp_path / "plan.json"
    tampered.write_bytes(runner.DEFAULT_PLAN_PATH.read_bytes() + b"\n")
    paths = _paths(tmp_path, "tampered")
    with pytest.raises(ValueError, match="plan byte SHA-256"):
        runner.run_development_shard(
            root_index=0,
            **paths,
            model_sha256=ATTEMPT06_FROZEN_MODEL_SHA256,
            run_id="tampered-plan-test",
            plan=tampered,
            batch_child_selectors=True,
            native_batch_threads=4,
        )


def test_output_is_byte_deterministic_and_batch_env_is_restored(
    tmp_path, monkeypatch
) -> None:
    captures: dict[str, list] = {}
    monkeypatch.setenv("OFC_HU_M3_BATCH_THREADS", "9")
    _mock_heavy(monkeypatch, captures=captures)

    original_evaluate = runner.evaluate_attempt07_t1_second

    def inspect_env(*args, **kwargs):
        assert runner.os.environ["OFC_HU_M3_BATCH_THREADS"] == "4"
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(runner, "evaluate_attempt07_t1_second", inspect_env)
    first, _ = _run(
        tmp_path,
        prefix="first",
        batch_child_selectors=True,
        native_batch_threads=4,
    )
    assert runner.os.environ["OFC_HU_M3_BATCH_THREADS"] == "9"
    second, _ = _run(
        tmp_path,
        prefix="second",
        batch_child_selectors=True,
        native_batch_threads=4,
    )
    assert first["output"].read_bytes() == second["output"].read_bytes()
    assert runner.os.environ["OFC_HU_M3_BATCH_THREADS"] == "9"


def test_hidden_discard_field_from_a_dependency_fails_closed(
    tmp_path, monkeypatch
) -> None:
    _mock_heavy(
        monkeypatch,
        teacher_extra={"opponent_private_discards": ["As"]},
    )
    with pytest.raises(ValueError, match="exposes opponent private discards"):
        _run(tmp_path)
    assert not (tmp_path / "result.jsonl").exists()


def test_cli_exposes_one_root_shard_and_batch_controls(tmp_path) -> None:
    args = runner.parse_args(
        [
            "--root-index",
            "12",
            "--output",
            str(tmp_path / "out.jsonl"),
            "--checkpoint",
            str(tmp_path / "checkpoint.json"),
            "--heartbeat",
            str(tmp_path / "heartbeat.json"),
            "--model",
            str(tmp_path / "model.pkl"),
            "--model-sha256",
            ATTEMPT06_FROZEN_MODEL_SHA256,
            "--run-id",
            "cli-test",
            "--batch-child-selectors",
            "--native-batch-threads",
            "4",
        ]
    )
    assert args.root_index == 12
    assert args.batch_child_selectors is True
    assert args.native_batch_threads == 4


def test_development_runner_rejects_nonfrozen_execution_mode(
    tmp_path, monkeypatch
) -> None:
    _mock_heavy(monkeypatch)
    with pytest.raises(ValueError, match="requires batched"):
        _run(tmp_path, prefix="scalar", batch_child_selectors=False)
    with pytest.raises(ValueError, match="fixed at 4"):
        _run(tmp_path, prefix="native8", native_batch_threads=8)
