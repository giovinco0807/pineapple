from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import ofc_regular.run_hu_m43_attempt08_development as runner
import ofc_regular.finalize_hu_m43_attempt08_preflight as finalizer
from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt08_contract import (
    ATTEMPT08_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT08_PLAN_SHA256,
    M43_ATTEMPT08_PROFILES,
    enumerate_attempt08_seed_schedules,
    load_and_validate_attempt08_plan,
)
from ofc_regular.hu_m43_attempt08_teacher import ATTEMPT08_TEACHER_SCHEMA
from ofc_regular.run_hu_m43_attempt08_preflight import ATTEMPT08_PREFLIGHT_SLOTS
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
    artifact_sha256 = ATTEMPT08_LAMBDA_MODEL_SHA256
    model_id = "fake-attempt08-ranker"


def _paths(tmp_path: Path, prefix: str = "result") -> dict[str, Path]:
    model = tmp_path / "frozen-lambda.pkl"
    model.touch(exist_ok=True)
    authorization = tmp_path / f"{prefix}.development-open.json"
    proof_hashes = {
        slot: hashlib.sha256(f"proof:{slot}".encode("utf-8")).hexdigest()
        for slot in ATTEMPT08_PREFLIGHT_SLOTS
    }
    payload = finalizer._authorization_payload(
        aggregate={
            "proof_file_sha256": proof_hashes,
            "proof_evidence_sha256": finalizer._sha256_value(proof_hashes),
            "proof_gates": {"all": True},
            "operational_gates": {"all": True},
            "spot_operational_evidence_sha256": "f" * 64,
            "runtime_fingerprint_sha256": (
                finalizer.ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256
            ),
        },
        aggregate_sha256="c" * 64,
    )
    authorization.write_bytes(finalizer.canonical_json_bytes(payload))
    return {
        "output": tmp_path / f"{prefix}.jsonl",
        "checkpoint": tmp_path / f"{prefix}.checkpoint.json",
        "heartbeat": tmp_path / f"{prefix}.heartbeat.json",
        "model": model,
        "development_open_authorization": authorization,
        "preflight_plan": runner.DEFAULT_PREFLIGHT_PLAN_PATH,
    }


def _mock_heavy(monkeypatch, *, captures=None, teacher_extra=None) -> None:
    captures = captures if captures is not None else {}

    def load_bundle(_paths, *, profiles):
        captures.setdefault("loaded_profiles", []).append(set(profiles))
        assert "current" not in profiles
        return object()

    def build(profile, _bundle, *, seed, seat, opening_lookahead_samples):
        assert opening_lookahead_samples == 0
        assert profile != "current"
        captures.setdefault("builds", []).append((profile, seat, seed))
        return _Policy(profile, seat, seed)

    def generate(hand_seed, *, root_policies):
        captures.setdefault("generated", []).append(
            (
                hand_seed,
                {seat: policy.profile for seat, policy in root_policies.items()},
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
        assert ranker.artifact_sha256 == ATTEMPT08_LAMBDA_MODEL_SHA256
        assert {policy.profile for policy in t2_policies.values()} == {"stage9f_p2"}
        captures.setdefault("teacher_configs", []).append(config)
        payload = {
            "status": "ok",
            "schema": ATTEMPT08_TEACHER_SCHEMA,
            "observation_fingerprint": observation.fingerprint(),
            "policy_observation": observation.to_dict(),
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

    def validate(observation, *, baseline_action_key, payload, config):
        assert observation == _root()
        assert payload["baseline_action_key"] == baseline_action_key
        assert config is captures["teacher_configs"][-1]
        captures.setdefault("validated", []).append(payload)
        return {
            "selected_action_key": baseline_action_key,
            "override_fired": False,
            "exact_baseline_fallback": True,
            "assessment_raw_paired_deltas_vs_baseline": [0.0] * 256,
            "rng_key_digests": {"assessment_a256": ["a" * 64]},
            "belief_digests": {"assessment_a256": "b" * 64},
        }

    monkeypatch.setattr(runner, "load_model_bundle", load_bundle)
    monkeypatch.setattr(runner, "build_policy", build)
    monkeypatch.setattr(runner, "generate_t1_second_root", generate)
    monkeypatch.setattr(runner, "_choose_from_observation", choose)
    monkeypatch.setattr(runner, "evaluate_attempt08_t1_second", evaluate)
    monkeypatch.setattr(runner, "validate_attempt08_teacher_output", validate)
    monkeypatch.setattr(
        runner, "_require_concrete_stage9f_p2_policies", lambda policies: None
    )
    monkeypatch.setattr(
        runner.FrozenAttempt08LambdaRanker,
        "load",
        classmethod(lambda cls, path, *, expected_sha256: _Ranker()),
    )


def _run(tmp_path: Path, *, root_index=7, prefix="result", **overrides):
    paths = _paths(tmp_path, prefix)
    values = {
        **paths,
        "root_index": root_index,
        "model_sha256": ATTEMPT08_LAMBDA_MODEL_SHA256,
        "run_id": "attempt08-development-test",
        "batch_child_selectors": True,
        "native_batch_threads": 4,
    }
    values.update(overrides)
    summary = runner.run_development_shard(**values)
    return paths, summary


def test_runner_derives_profile_all_six_seeds_and_explicit_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captures: dict[str, list] = {}
    _mock_heavy(monkeypatch, captures=captures)
    paths, summary = _run(tmp_path, root_index=7)
    row = json.loads(paths["output"].read_text(encoding="utf-8"))
    plan = load_and_validate_attempt08_plan(runner.DEFAULT_PLAN_PATH)
    schedules = enumerate_attempt08_seed_schedules(plan, population="development")
    expected_seeds = {domain: values[7] for domain, values in schedules.items()}

    assert row["root_index"] == 7
    assert row["hand_seed"] == expected_seeds["hand"]
    assert row["root_profile"] == M43_ATTEMPT08_PROFILES[7 % 5]
    assert row["provenance"]["seeds"] == expected_seeds
    config = captures["teacher_configs"][0]
    assert (
        config.hand_seed,
        config.rerank_seed,
        config.veto_seed,
        config.stress_seed,
        config.assessment_seed,
        config.child_policy_seed,
    ) == tuple(
        expected_seeds[name]
        for name in ("hand", "rerank", "veto", "stress", "assessment", "child")
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
    assert len(captures["validated"]) == 1
    assert summary["status"] == "complete"
    assert not paths["output"].with_name(paths["output"].name + ".partial").exists()
    assert json.loads(paths["checkpoint"].read_text())["completed_roots"] == 1
    assert json.loads(paths["heartbeat"].read_text())["status"] == "complete"


def test_provenance_is_development_only_and_never_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_heavy(monkeypatch)
    paths, _ = _run(tmp_path, root_index=199)
    raw = paths["output"].read_bytes()
    row = json.loads(raw)
    provenance = row["provenance"]

    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    assert provenance["plan_sha256"] == M43_ATTEMPT08_PLAN_SHA256
    assert provenance["development_open_authorization_sha256"] == hashlib.sha256(
        paths["development_open_authorization"].read_bytes()
    ).hexdigest()
    assert provenance["preflight_plan_sha256"] == finalizer.ATTEMPT08_PREFLIGHT_PLAN_SHA256
    assert provenance["profile_assignment"] == (
        "root_index_mod_5_in_frozen_profile_order"
    )
    assert provenance["baseline_profile"] == "stage18_p1"
    assert provenance["continuation_profile"] == "stage9f_p2"
    assert "current" not in provenance["explicit_loaded_profiles"]
    assert provenance["current_profile_resolved"] is False
    assert provenance["opponent_private_discard_input_allowed"] is False
    assert provenance["teacher_values_are_realized_match_ev"] is False
    assert provenance["search_freeze_authorization_allowed"] is False
    assert provenance["future_audit_allowed"] is False
    assert provenance["fit_allowed"] is False
    assert provenance["threshold_selection_allowed"] is False
    assert provenance["runtime_activation_allowed"] is False
    assert b"opponent_private_discards" not in raw


def test_completed_partial_resumes_without_reopening_heavy_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_heavy(monkeypatch)
    original, original_summary = _run(tmp_path, prefix="original")
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
        model_sha256=ATTEMPT08_LAMBDA_MODEL_SHA256,
        run_id="attempt08-development-test",
        batch_child_selectors=True,
        native_batch_threads=4,
    )
    assert resumed["output"].read_bytes() == expected
    assert summary["output_sha256"] == hashlib.sha256(expected).hexdigest()
    assert summary["elapsed_seconds"] == original_summary["elapsed_seconds"]
    assert json.loads(resumed["checkpoint"].read_text())[
        "generator_elapsed_seconds"
    ] == original_summary["elapsed_seconds"]
    assert summary["generator_peak_rss_bytes"] == original_summary[
        "generator_peak_rss_bytes"
    ]
    assert json.loads(resumed["checkpoint"].read_text())[
        "generator_peak_rss_bytes"
    ] == original_summary["generator_peak_rss_bytes"]


def _stage_output_commit_crash(
    original: dict[str, Path], recovered: dict[str, Path], *, keep_partial: bool
) -> None:
    partial = recovered["output"].with_name(recovered["output"].name + ".partial")
    shutil.copyfile(original["output"], partial)
    if keep_partial:
        recovered["output"].parent.mkdir(parents=True, exist_ok=True)
        recovered["output"].hardlink_to(partial)
    else:
        shutil.copyfile(partial, recovered["output"])
        partial.unlink()
    shutil.copyfile(original["checkpoint"], recovered["checkpoint"])
    checkpoint = json.loads(recovered["checkpoint"].read_text(encoding="utf-8"))
    running = {
        "schema": runner.ATTEMPT08_HEARTBEAT_SCHEMA,
        "status": "running",
        "config_sha256": checkpoint["config_sha256"],
        "root_index": checkpoint["root_index"],
    }
    recovered["heartbeat"].write_bytes(runner._canonical_json_bytes(running))


@pytest.mark.parametrize("keep_partial", [True, False])
def test_existing_output_after_commit_crash_recovers_without_heavy_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keep_partial: bool,
) -> None:
    _mock_heavy(monkeypatch)
    original, original_summary = _run(tmp_path, prefix="commit-original")
    recovered = _paths(tmp_path, f"commit-recovered-{int(keep_partial)}")
    _stage_output_commit_crash(original, recovered, keep_partial=keep_partial)
    expected = original["output"].read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("commit recovery reopened heavy work")

    monkeypatch.setattr(runner, "load_model_bundle", forbidden)
    summary = runner.run_development_shard(
        root_index=7,
        **recovered,
        model_sha256=ATTEMPT08_LAMBDA_MODEL_SHA256,
        run_id="attempt08-development-test",
        batch_child_selectors=True,
        native_batch_threads=4,
    )
    assert recovered["output"].read_bytes() == expected
    assert summary == original_summary
    assert not recovered["output"].with_name(
        recovered["output"].name + ".partial"
    ).exists()
    assert json.loads(recovered["heartbeat"].read_text(encoding="utf-8"))[
        "status"
    ] == "complete"


@pytest.mark.parametrize("mutation", ["output", "partial", "row_contract"])
def test_existing_output_recovery_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _mock_heavy(monkeypatch)
    original, _ = _run(tmp_path, prefix=f"mismatch-original-{mutation}")
    recovered = _paths(tmp_path, f"mismatch-recovered-{mutation}")
    _stage_output_commit_crash(original, recovered, keep_partial=True)
    partial = recovered["output"].with_name(recovered["output"].name + ".partial")
    if mutation == "output":
        partial.unlink()
        recovered["output"].write_bytes(recovered["output"].read_bytes() + b" ")
    elif mutation == "partial":
        partial.unlink()
        partial.write_bytes(recovered["output"].read_bytes() + b" ")
    else:
        partial.unlink()
        row = json.loads(recovered["output"].read_text(encoding="utf-8"))
        row["root_index"] = 8
        raw = runner._canonical_json_bytes(row)
        recovered["output"].write_bytes(raw)
        partial.write_bytes(raw)
        checkpoint = json.loads(recovered["checkpoint"].read_text(encoding="utf-8"))
        checkpoint["partial_sha256"] = hashlib.sha256(raw).hexdigest()
        recovered["checkpoint"].write_bytes(runner._canonical_json_bytes(checkpoint))
    before_output = recovered["output"].read_bytes()
    before_partial = partial.read_bytes() if partial.exists() else None

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid recovery reopened heavy work")

    monkeypatch.setattr(runner, "load_model_bundle", forbidden)
    with pytest.raises(ValueError, match="output|partial|row identity"):
        runner.run_development_shard(
            root_index=7,
            **recovered,
            model_sha256=ATTEMPT08_LAMBDA_MODEL_SHA256,
            run_id="attempt08-development-test",
            batch_child_selectors=True,
            native_batch_threads=4,
        )
    assert recovered["output"].read_bytes() == before_output
    if before_partial is None:
        assert not partial.exists()
    else:
        assert partial.read_bytes() == before_partial
    assert json.loads(recovered["heartbeat"].read_text(encoding="utf-8"))[
        "status"
    ] == "running"


@pytest.mark.parametrize("root_index", [-1, 200, True])
def test_runner_rejects_root_outside_exact_development200(
    tmp_path: Path, root_index: object
) -> None:
    paths = _paths(tmp_path)
    with pytest.raises((TypeError, ValueError), match="root_index"):
        runner.run_development_shard(
            root_index=root_index,
            **paths,
            model_sha256=ATTEMPT08_LAMBDA_MODEL_SHA256,
            run_id="attempt08-development-test",
            batch_child_selectors=True,
            native_batch_threads=4,
        )


@pytest.mark.parametrize(
    ("batch", "threads"), [(False, 4), (True, 1), (True, True)]
)
def test_runner_requires_production_batch_and_four_threads(
    tmp_path: Path, batch: bool, threads: object
) -> None:
    paths = _paths(tmp_path)
    with pytest.raises(ValueError, match="batched|fixed at 4"):
        runner.run_development_shard(
            root_index=0,
            **paths,
            model_sha256=ATTEMPT08_LAMBDA_MODEL_SHA256,
            run_id="attempt08-development-test",
            batch_child_selectors=batch,
            native_batch_threads=threads,
        )


def test_runner_is_no_clobber(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_heavy(monkeypatch)
    paths, first_summary = _run(tmp_path)
    before = paths["output"].read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("idempotent completion reopened heavy work")

    monkeypatch.setattr(runner, "load_model_bundle", forbidden)
    second_summary = runner.run_development_shard(
        root_index=7,
        **paths,
        model_sha256=ATTEMPT08_LAMBDA_MODEL_SHA256,
        run_id="attempt08-development-test",
        batch_child_selectors=True,
        native_batch_threads=4,
    )
    assert paths["output"].read_bytes() == before
    assert second_summary == first_summary


@pytest.mark.parametrize("mutation", ("missing", "no_go", "plan", "evidence"))
def test_development_open_authorization_fails_before_any_root_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    paths = _paths(tmp_path, mutation)
    authorization = paths["development_open_authorization"]
    if mutation == "missing":
        authorization.unlink()
    else:
        payload = json.loads(authorization.read_text(encoding="utf-8"))
        if mutation == "no_go":
            payload["status"] = "complete_preflight_no_go_development_not_authorized"
        elif mutation == "plan":
            payload["target_plan"]["sha256"] = "0" * 64
        elif mutation == "evidence":
            payload["evidence"]["proof_evidence_sha256"] = "0" * 64
        authorization.write_bytes(finalizer.canonical_json_bytes(payload))

    def forbidden(*args, **kwargs):
        raise AssertionError("authorization failure opened heavy root work")

    monkeypatch.setattr(runner, "load_model_bundle", forbidden)
    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        runner.run_development_shard(
            root_index=0,
            **paths,
            model_sha256=ATTEMPT08_LAMBDA_MODEL_SHA256,
            run_id="attempt08-development-test",
            batch_child_selectors=True,
            native_batch_threads=4,
        )
    assert not paths["output"].exists()
    assert not paths["checkpoint"].exists()
    assert not paths["heartbeat"].exists()
    assert not paths["output"].with_name(paths["output"].name + ".lock").exists()
