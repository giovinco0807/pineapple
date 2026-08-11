from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

import pytest

import ofc_regular.run_hu_m43_attempt09 as runner
from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt09_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT09_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT09_PLAN_SHA256,
    enumerate_attempt09_seed_schedules,
)
from ofc_regular.state import Board


SOURCE_PACKAGE_SHA256 = "a" * 64


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


def _plan() -> dict:
    return json.loads(runner.DEFAULT_PLAN.read_text(encoding="utf-8-sig"))


def _canonical(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner._canonical_json_bytes(value))


def _authorization(tmp_path: Path, *, mode: str, package_hash=SOURCE_PACKAGE_SHA256):
    gate = tmp_path / f"{mode}-gate.json"
    _canonical(gate, {"schema": "test_gate_v1", "status": "pass"})
    gate_sha = hashlib.sha256(gate.read_bytes()).hexdigest()
    first, last = ((0, 199) if mode == "development" else (200, 249))
    auth = tmp_path / f"{mode}-authorization.json"
    _canonical(
        auth,
        {
            "schema": runner.ATTEMPT09_AUTHORIZATION_SCHEMA,
            "status": "authorized",
            "mode": mode,
            "plan_sha256": M43_ATTEMPT09_PLAN_SHA256,
            "source_package_sha256": package_hash,
            "preceding_gate_artifact": str(gate),
            "preceding_gate_sha256": gate_sha,
            "root_index_first": first,
            "root_index_last": last,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
    )
    return auth, gate


@pytest.fixture
def mocked_runner(monkeypatch, tmp_path):
    plan_path = tmp_path / "configs" / "attempt09.json"
    ai_profiles_path = tmp_path / "src" / "ai_profiles.py"
    model_path = tmp_path / "model.pkl"
    plan_path.parent.mkdir(parents=True)
    ai_profiles_path.parent.mkdir(parents=True)
    plan_path.write_text("plan", encoding="utf-8")
    ai_profiles_path.write_text("profiles", encoding="utf-8")
    model_path.write_bytes(b"model")
    plan_payload = _plan()
    calls = {"loads": [], "builds": [], "evaluations": 0, "teacher_batch": []}

    monkeypatch.setattr(runner, "load_and_validate_attempt09_plan", lambda _: plan_payload)
    monkeypatch.setattr(runner, "validate_attempt09_artifact_bindings", lambda *a, **k: None)

    def sha(path):
        target = Path(path)
        if target == plan_path:
            return M43_ATTEMPT09_PLAN_SHA256
        if target == ai_profiles_path:
            return AI_PROFILES_SHA256
        if target == model_path:
            return ATTEMPT09_LAMBDA_MODEL_SHA256
        return hashlib.sha256(target.read_bytes()).hexdigest()

    monkeypatch.setattr(runner, "_sha256_file", sha)

    def load_bundle(_paths, *, profiles):
        calls["loads"].append(set(profiles))
        return object()

    monkeypatch.setattr(runner, "load_model_bundle", load_bundle)

    class Policy:
        def __init__(self, profile, seat, seed):
            self.profile = profile
            self.seat = seat
            self.seed = seed

    def build(profile, _bundle, *, seed, seat, opening_lookahead_samples):
        assert opening_lookahead_samples == 0
        calls["builds"].append((profile, seat, seed))
        return Policy(profile, seat, seed)

    monkeypatch.setattr(runner, "build_policy", build)
    monkeypatch.setattr(
        runner,
        "generate_t1_second_root",
        lambda seed, *, root_policies: _root(),
    )
    baseline_action = generate_turn_actions(_root().hero_board, _root().dealt_cards)[-1]
    monkeypatch.setattr(
        runner,
        "_choose_from_observation",
        lambda *args, **kwargs: baseline_action,
    )
    monkeypatch.setattr(
        runner,
        "_require_concrete_stage9f_p2_policies",
        lambda policies: None,
    )
    monkeypatch.setattr(
        runner.FrozenAttempt09LambdaRanker,
        "load",
        lambda *args, **kwargs: object(),
    )

    def evaluate(observation, *, baseline_action_key, ranker, t2_policies, config):
        calls["evaluations"] += 1
        calls["teacher_batch"].append(config.batch_child_selectors)
        return {
            "status": "ok",
            "schema": runner.ATTEMPT09_TEACHER_SCHEMA,
            "observation_fingerprint": observation.fingerprint(),
            "policy_observation": observation.to_dict(),
            "baseline_action_key": baseline_action_key,
            "runtime_gate_allowed": False,
            "profile_activation_allowed": False,
            "current_profile_resolved": False,
            "development_only": True,
        }

    monkeypatch.setattr(runner, "evaluate_attempt09_t1_second", evaluate)
    monkeypatch.setattr(runner, "validate_attempt09_teacher_output", lambda *a, **k: {})
    monkeypatch.setattr(
        runner, "_native_batch_threads", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(runner, "_process_peak_rss_bytes", lambda: 123456)

    class Pump:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(runner, "_HeartbeatPump", Pump)

    def run(
        directory: str,
        *,
        mode="preflight",
        root_index=0,
        authorization=None,
        source_package_sha256=None,
        run_id="attempt09-test",
        batch_child_selectors=True,
    ):
        target = tmp_path / directory
        target.mkdir(parents=True, exist_ok=True)
        return runner.run_attempt09_root(
            mode=mode,
            root_index=root_index,
            output=target / "teacher.jsonl",
            checkpoint=target / "checkpoint.json",
            heartbeat=target / "heartbeat.json",
            model=model_path,
            model_sha256=ATTEMPT09_LAMBDA_MODEL_SHA256,
            run_id=run_id,
            plan=plan_path,
            ai_profiles=ai_profiles_path,
            authorization=authorization,
            source_package_sha256=source_package_sha256,
            batch_child_selectors=batch_child_selectors,
        )

    return {
        "run": run,
        "calls": calls,
        "tmp": tmp_path,
        "plan": plan_payload,
    }


def test_seed_offsets_use_declared_preflight_development_and_audit_indices() -> None:
    plan = _plan()
    cases = (
        ("preflight", 0, 0),
        ("preflight", 2, 2),
        ("development", 0, 0),
        ("development", 199, 199),
        ("future_audit", 200, 0),
        ("future_audit", 249, 49),
    )
    for mode, root_index, offset in cases:
        schedules = enumerate_attempt09_seed_schedules(plan, population=mode)
        expected = {domain: values[offset] for domain, values in schedules.items()}
        assert runner._seed_values(plan, mode=mode, root_index=root_index) == expected
    with pytest.raises(ValueError, match="0..199"):
        runner._seed_values(plan, mode="development", root_index=200)
    with pytest.raises(ValueError, match="200..249"):
        runner._seed_values(plan, mode="future_audit", root_index=199)


def test_scalar_is_allowed_only_for_preflight_and_propagates_to_teacher(mocked_runner) -> None:
    summary = mocked_runner["run"](
        "scalar-preflight", root_index=0, batch_child_selectors=False
    )
    row = json.loads(
        (mocked_runner["tmp"] / "scalar-preflight" / "teacher.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "complete"
    assert row["provenance"]["batch_child_selectors"] is False
    assert mocked_runner["calls"]["teacher_batch"] == [False]
    with pytest.raises(ValueError, match="development/audit roots require batched"):
        mocked_runner["run"](
            "scalar-development",
            mode="development",
            root_index=0,
            batch_child_selectors=False,
        )


def test_authorization_is_mode_package_gate_and_range_bound(mocked_runner) -> None:
    run = mocked_runner["run"]
    tmp = mocked_runner["tmp"]
    with pytest.raises(ValueError, match="requires explicit authorization"):
        run("missing", mode="development", root_index=0)
    dev_auth, gate = _authorization(tmp, mode="development")
    with pytest.raises(ValueError, match="must not consume later authorization"):
        run("preflight-auth", authorization=dev_auth)
    with pytest.raises(ValueError, match="source package authorization"):
        run(
            "package-mismatch",
            mode="development",
            root_index=0,
            authorization=dev_auth,
            source_package_sha256="b" * 64,
        )
    gate.write_bytes(b"changed")
    with pytest.raises(ValueError, match="gate artifact hash changed"):
        run(
            "gate-mismatch",
            mode="development",
            root_index=0,
            authorization=dev_auth,
            source_package_sha256=SOURCE_PACKAGE_SHA256,
        )
    assert mocked_runner["calls"]["evaluations"] == 0


def test_explicit_profiles_no_current_and_canonical_output(mocked_runner) -> None:
    summary = mocked_runner["run"]("canonical", root_index=2)
    target = mocked_runner["tmp"] / "canonical"
    raw = (target / "teacher.jsonl").read_bytes()
    row = json.loads(raw.decode("utf-8"))
    assert raw == runner._canonical_json_bytes(row)
    assert row["root_profile"] == "stage7_m5_r10"
    assert row["provenance"]["explicit_loaded_profiles"] == [
        "stage18_p1",
        "stage7_m5_r10",
        "stage9f_p2",
    ]
    assert mocked_runner["calls"]["loads"] == [
        {"stage18_p1", "stage7_m5_r10", "stage9f_p2"}
    ]
    assert all(profile != "current" for profile, _, _ in mocked_runner["calls"]["builds"])
    assert row["hand_seed"] == runner._seed_values(
        mocked_runner["plan"], mode="preflight", root_index=2
    )["hand"]
    assert summary["output_sha256"] == hashlib.sha256(raw).hexdigest()
    checkpoint_raw = (target / "checkpoint.json").read_bytes()
    assert checkpoint_raw == runner._canonical_json_bytes(
        json.loads(checkpoint_raw.decode("utf-8"))
    )


def test_completed_output_resume_is_hash_bound_and_does_not_recompute(mocked_runner) -> None:
    run = mocked_runner["run"]
    first = run("resume", root_index=1)
    target = mocked_runner["tmp"] / "resume"
    before = (target / "teacher.jsonl").read_bytes()
    assert mocked_runner["calls"]["evaluations"] == 1
    second = run("resume", root_index=1)
    assert mocked_runner["calls"]["evaluations"] == 1
    assert (target / "teacher.jsonl").read_bytes() == before
    assert first["config_sha256"] == second["config_sha256"]
    assert first["output_sha256"] == second["output_sha256"]


def test_checkpoint_contract_and_completed_output_hash_fail_closed(mocked_runner) -> None:
    run = mocked_runner["run"]
    run("checkpoint", root_index=0)
    target = mocked_runner["tmp"] / "checkpoint"
    checkpoint_path = target / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    original = dict(checkpoint)
    checkpoint["config_sha256"] = "0" * 64
    _canonical(checkpoint_path, checkpoint)
    with pytest.raises(ValueError, match="another root contract"):
        run("checkpoint", root_index=0)
    _canonical(checkpoint_path, original)
    output_path = target / "teacher.jsonl"
    row = json.loads(output_path.read_text(encoding="utf-8"))
    row["teacher"]["extra"] = True
    output_path.write_bytes(runner._canonical_json_bytes(row))
    with pytest.raises(ValueError, match="hash disagrees"):
        run("checkpoint", root_index=0)


def test_hidden_opponent_discard_is_rejected_before_output(monkeypatch, mocked_runner) -> None:
    def hidden(observation, *, baseline_action_key, ranker, t2_policies, config):
        return {
            "schema": runner.ATTEMPT09_TEACHER_SCHEMA,
            "policy_observation": observation.to_dict(),
            "baseline_action_key": baseline_action_key,
            "opponent_hidden_discard": ["As"],
            "runtime_gate_allowed": False,
            "profile_activation_allowed": False,
            "current_profile_resolved": False,
            "development_only": True,
        }

    monkeypatch.setattr(runner, "evaluate_attempt09_t1_second", hidden)
    with pytest.raises(ValueError, match="hidden opponent discard"):
        mocked_runner["run"]("hidden", root_index=0)
    assert not (mocked_runner["tmp"] / "hidden" / "teacher.jsonl").exists()


def test_valid_development_and_future_audit_authorizations_run(mocked_runner) -> None:
    tmp = mocked_runner["tmp"]
    dev_auth, _ = _authorization(tmp / "dev-auth", mode="development")
    dev = mocked_runner["run"](
        "development",
        mode="development",
        root_index=199,
        authorization=dev_auth,
        source_package_sha256=SOURCE_PACKAGE_SHA256,
    )
    audit_auth, _ = _authorization(tmp / "audit-auth", mode="future_audit")
    audit = mocked_runner["run"](
        "audit",
        mode="future_audit",
        root_index=249,
        authorization=audit_auth,
        source_package_sha256=SOURCE_PACKAGE_SHA256,
    )
    assert (dev["root_index"], dev["root_profile"]) == (199, "random_exact_final")
    assert (audit["root_index"], audit["root_profile"]) == (249, "random_exact_final")
