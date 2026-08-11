from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import ofc_regular.hu_m43_attempt09_spot as spot
from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt09_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT09_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT09_PLAN_SHA256,
    enumerate_attempt09_seed_schedules,
)
from ofc_regular.hu_m43_attempt09_teacher import ATTEMPT09_TEACHER_SCHEMA
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


def _canonical(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(spot.canonical_json_bytes(payload))


@pytest.fixture
def package_factory(monkeypatch, tmp_path):
    real_plan = json.loads(spot.DEFAULT_PLAN.read_text(encoding="utf-8-sig"))
    repository = tmp_path / "repository"
    template = tmp_path / "template"
    template.mkdir()
    _canonical(
        template / "source_closure_manifest.json",
        {"schema": "minimal_template_v1", "status": "frozen"},
    )
    (template / "template-marker.txt").write_text("template\n", encoding="utf-8")
    plan_path = repository / "configs" / "hu_joint_policy_m43_attempt09.json"
    plan_path.parent.mkdir(parents=True)
    # The checked-in plan is intentionally human-readable JSON, not a receipt.
    # Receive must use the validated plan loader rather than receipt canonicality.
    plan_path.write_text(
        json.dumps(real_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for relative in (
        "src/ofc_regular/hu_m43_attempt09_contract.py",
        "src/ofc_regular/hu_m43_attempt09_teacher.py",
        "src/ofc_regular/run_hu_m43_attempt09.py",
    ):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    boundary = (
        repository
        / "outputs/hu_joint_policy/m43_attempt08_development"
        / "regular-hu-m43-attempt08-development200-finalprop-20260714-215952"
        / "selector"
    )
    _canonical(boundary / "decision.json", {"status": "no_go", "decision": "no_go"})
    _canonical(
        boundary / "decision_receipt.json",
        {"status": "single_frozen_gate_evaluation_complete", "decision": "no_go"},
    )
    startup = repository / "scripts" / spot.STARTUP_NAME
    startup.parent.mkdir(parents=True)
    startup.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
    monkeypatch.setattr(
        spot, "load_and_validate_attempt09_plan", lambda _path: real_plan
    )
    monkeypatch.setattr(
        spot, "validate_attempt09_artifact_bindings", lambda *args, **kwargs: None
    )

    counter = 0

    def make(mode="preflight", *, gate_payload=None):
        nonlocal counter
        counter += 1
        run_name = f"attempt09-{mode.replace('_', '-')}-{counter:02d}"
        run_dir = repository / "outputs" / "gcp_runs" / run_name
        expected_gate = {
            "preflight": ("pass_local_correctness", "authorize_preflight_only"),
            "development": (
                "pass_correctness_preflight",
                "authorize_development200_package_only",
            ),
            "future_audit": ("go_freeze_attempt09_development", "go"),
        }[mode]
        gate = repository / "gates" / f"{run_name}.json"
        _canonical(
            gate,
            gate_payload
            or {
                "schema": "attempt09_gate_v1",
                "status": expected_gate[0],
                "decision": expected_gate[1],
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
            },
        )
        spot.package_attempt09(
            mode=mode,
            run_name=run_name,
            run_dir=run_dir,
            repository_root=repository,
            template_package=template,
            plan=plan_path,
            startup=startup,
            preceding_gate=gate,
        )
        spot.authorize_launch(run_dir=run_dir)
        return run_dir, real_plan

    return make


def test_schedules_are_fixed_balanced_and_share_preflight_rng_identity() -> None:
    preflight = spot.build_schedule("preflight", "attempt09-preflight-test")
    assert [(row["root_index"], row["batch_child_selectors"]) for row in preflight] == [
        (0, True),
        (0, True),
        (0, False),
        (1, True),
        (2, True),
    ]
    assert len({preflight[index]["run_id"] for index in (0, 1, 2)}) == 1
    development = spot.build_schedule("development", "attempt09-development-test")
    audit = spot.build_schedule("future_audit", "attempt09-audit-test")
    assert len(development) == 200 and [row["shard"] for row in development] == list(range(200))
    assert [row["root_index"] for row in audit] == list(range(200, 250))
    assert {profile: [row["root_profile"] for row in development].count(profile) for profile in spot.M43_ATTEMPT09_PROFILES} == {
        profile: 40 for profile in spot.M43_ATTEMPT09_PROFILES
    }
    with pytest.raises(ValueError, match="safe GCP identity"):
        spot.build_schedule("preflight", "UNSAFE")


def test_package_and_authorization_hash_closure_fail_closed(package_factory) -> None:
    preflight, _ = package_factory("preflight")
    manifest, launch = spot.validate_launch(preflight)
    assert manifest["total_shards"] == 5
    assert launch["root_execution_started"] is False
    assert not (preflight / "execution_authorization.json").exists()
    (preflight / spot.SOURCE_NAME).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="frozen package changed"):
        spot.validate_launch(preflight)

    development, _ = package_factory("development")
    spot.validate_launch(development)
    execution = json.loads(
        (development / "execution_authorization.json").read_text(encoding="utf-8")
    )
    execution["root_index_last"] = 198
    _canonical(development / "execution_authorization.json", execution)
    with pytest.raises(ValueError, match="execution authorization changed"):
        spot.validate_launch(development)


def test_launch_rejects_duplicate_and_more_than_25_before_gcloud(
    monkeypatch, package_factory
) -> None:
    run_dir, _ = package_factory("development")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("gcloud must not run for an invalid shard wave")

    monkeypatch.setattr(spot, "_run", forbidden)
    monkeypatch.setattr(spot.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="duplicate shard"):
        spot.launch_wave(
            run_dir=run_dir,
            project="test",
            bucket="test",
            zone="test-zone",
            shards=["0-2", "2"],
        )
    with pytest.raises(ValueError, match="at most 25"):
        spot.launch_wave(
            run_dir=run_dir,
            project="test",
            bucket="test",
            zone="test-zone",
            shards=["0-25"],
        )
    assert calls == []


def test_subprocess_run_resolves_windows_gcloud_cmd_after_not_found(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if len(calls) == 1:
            raise FileNotFoundError("gcloud shim requires PATHEXT resolution")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(spot.subprocess, "run", fake_run)
    monkeypatch.setattr(spot.shutil, "which", lambda name: r"C:\sdk\gcloud.CMD")

    completed = spot._subprocess_run(
        ["gcloud", "version"], capture_output=True, text=True
    )

    assert completed.returncode == 0
    assert calls == [
        ["gcloud", "version"],
        [r"C:\sdk\gcloud.CMD", "version"],
    ]


def test_launch_uses_mocked_subprocess_and_creates_only_selected_shards(
    monkeypatch, package_factory
) -> None:
    run_dir, _ = package_factory("development")
    created = []

    def fake_run(command, *, timeout=300):
        if command[:4] == ["gcloud", "compute", "images", "describe"]:
            payload = {"id": spot.EXPECTED_IMAGE_ID, "selfLink": spot.EXPECTED_IMAGE_SELF_LINK}
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if command[:4] == ["gcloud", "compute", "instances", "create"]:
            created.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(command)

    def fake_subprocess(command, **kwargs):
        if command[:3] == ["gcloud", "storage", "cp"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:4] in (
            ["gcloud", "storage", "objects", "describe"],
            ["gcloud", "compute", "instances", "describe"],
        ):
            return subprocess.CompletedProcess(command, 1, "", "not found")
        raise AssertionError(command)

    monkeypatch.setattr(spot, "_run", fake_run)
    monkeypatch.setattr(spot.subprocess, "run", fake_subprocess)
    result = spot.launch_wave(
        run_dir=run_dir,
        project="test",
        bucket="test",
        zone="test-zone",
        shards=["0-1"],
        no_self_delete=True,
    )
    assert [item["shard"] for item in result["created"]] == [0, 1]
    assert len(created) == 2
    assert all("--provisioning-model=SPOT" in command for command in created)


def _build_remote(run_dir: Path, plan: dict, remote: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    schedule = [
        json.loads(line)
        for line in (run_dir / spot.SCHEDULE_NAME).read_text(encoding="utf-8").splitlines()
    ]
    seed_schedules = enumerate_attempt09_seed_schedules(
        plan, population=manifest["mode"]
    )
    preflight_indices = plan["preflight_seed_contract"]["source_root_indices"]
    manifest_sha = spot.sha256_file(run_dir / "manifest.json")
    schedule_sha = spot.sha256_file(run_dir / spot.SCHEDULE_NAME)
    execution = run_dir / "execution_authorization.json"
    authorization_sha = spot.sha256_file(execution) if execution.is_file() else "none"
    observation = _root()
    baseline = action_key(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)[-1]
    ).to_token()
    for spec in schedule:
        if manifest["mode"] == "preflight":
            offset = preflight_indices.index(spec["root_index"])
        elif manifest["mode"] == "development":
            offset = spec["root_index"]
        else:
            offset = spec["root_index"] - 200
        seeds = {domain: values[offset] for domain, values in seed_schedules.items()}
        teacher = {
            "status": "ok",
            "schema": ATTEMPT09_TEACHER_SCHEMA,
            "policy_observation": observation.to_dict(),
            "baseline_action_key": baseline,
            "observation_fingerprint": observation.fingerprint(),
            "search_config": {"batch_child_selectors": spec["batch_child_selectors"]},
            "current_profile_resolved": False,
            "runtime_gate_allowed": False,
            "profile_activation_allowed": False,
            "development_only": True,
        }
        row = {
            "schema": spot.ATTEMPT09_ROW_SCHEMA,
            "root_index": spec["root_index"],
            "hand_seed": seeds["hand"],
            "root_profile": spec["root_profile"],
            "policy_observation": observation.to_dict(),
            "baseline_action_key": baseline,
            "provenance": {
                "mode": manifest["mode"],
                "run_id": spec["run_id"],
                "root_index": spec["root_index"],
                "root_profile": spec["root_profile"],
                "seeds": seeds,
                "plan_sha256": M43_ATTEMPT09_PLAN_SHA256,
                "ai_profiles_sha256": AI_PROFILES_SHA256,
                "model_sha256": ATTEMPT09_LAMBDA_MODEL_SHA256,
                "batch_child_selectors": spec["batch_child_selectors"],
                "native_batch_threads": 4,
                "authorization_sha256": None if authorization_sha == "none" else authorization_sha,
                "source_package_sha256": None if authorization_sha == "none" else manifest["source_sha256"],
                "current_profile_resolved": False,
                "opponent_private_discard_input_allowed": False,
                "runtime_activation_allowed": False,
                "elapsed_seconds": 1.0 + spec["shard"],
            },
            "teacher": teacher,
        }
        directory = remote / spec["output_prefix"]
        directory.mkdir(parents=True)
        (directory / "teacher.jsonl").write_bytes(spot.canonical_json_bytes(row))
        _canonical(directory / "checkpoint.json", {"status": "complete"})
        _canonical(directory / "heartbeat.json", {"status": "complete"})
        _canonical(
            directory / "generator_summary.json",
            {
                "status": "complete",
                "root_index": spec["root_index"],
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
            },
        )
        (directory / "run.log").write_bytes(b"ok\n")
        (directory / "time.txt").write_bytes(b"time\n")
        names = (
            "teacher.jsonl", "checkpoint.json", "heartbeat.json",
            "generator_summary.json", "run.log", "time.txt",
        )
        files = {
            name: {
                "sha256": spot.sha256_file(directory / name),
                "bytes": (directory / name).stat().st_size,
            }
            for name in names
        }
        _canonical(
            directory / "DONE.json",
            {
                "schema": spot.DONE_SCHEMA,
                "status": "complete",
                "run_name": manifest["run_name"],
                "mode": manifest["mode"],
                "shard": spec["shard"],
                "root_index": spec["root_index"],
                "output_prefix": spec["output_prefix"],
                "source_sha256": manifest["source_sha256"],
                "manifest_sha256": manifest_sha,
                "schedule_sha256": schedule_sha,
                "authorization_sha256": authorization_sha,
                "files": files,
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
                "completed_unix_seconds": 1.0,
            },
        )


def _receive(monkeypatch, run_dir, plan, remote, output, semantic_calls):
    original_validate = spot.validate_launch
    boundary_calls = []

    def validate_once(path):
        boundary_calls.append(Path(path))
        return original_validate(path)

    def rsync(command, *, timeout=300):
        assert command[:4] == ["gcloud", "storage", "rsync", "--recursive"]
        shutil.copytree(remote, Path(command[5]), dirs_exist_ok=True)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(spot, "validate_launch", validate_once)
    monkeypatch.setattr(spot, "_run", rsync)
    monkeypatch.setattr(
        spot,
        "validate_attempt09_teacher_output",
        lambda observation, **kwargs: semantic_calls.append(
            (observation.fingerprint(), kwargs["config"].batch_child_selectors)
        ),
    )
    result = spot.receive_run(
        run_dir=run_dir,
        project="test",
        bucket="test",
        output_dir=output,
    )
    return result, boundary_calls


def test_receive_is_exact_o_n_semantic_and_canonical(monkeypatch, package_factory, tmp_path) -> None:
    run_dir, plan = package_factory("preflight")
    remote = tmp_path / "remote"
    _build_remote(run_dir, plan, remote)
    semantic_calls = []
    output = tmp_path / "received"
    receipt, boundary_calls = _receive(
        monkeypatch, run_dir, plan, remote, output, semantic_calls
    )
    assert len(boundary_calls) == 1
    assert len(semantic_calls) == 5
    assert [batch for _, batch in semantic_calls] == [True, True, False, True, True]
    assert receipt["batch_boundary_validation_count"] == 1
    assert receipt["per_shard_boundary_revalidation_count"] == 0
    assert receipt["root_indices"] == [0, 0, 0, 1, 2]
    merged = (output / "merged" / "teacher.jsonl").read_bytes()
    assert hashlib.sha256(merged).hexdigest() == receipt["merged_sha256"]
    assert (output / "merged" / "receive_receipt.json").read_bytes() == spot.canonical_json_bytes(receipt)


@pytest.mark.parametrize("mutation,match", [
    ("missing_shard", "received shard set changed"),
    ("extra_root", "received shard set changed"),
    ("missing_file", "files changed"),
    ("extra_file", "files changed"),
    ("tamper_hash", "received content changed"),
    ("noncanonical", "row is not canonical"),
])
def test_receive_rejects_missing_extra_tamper_and_noncanonical(
    monkeypatch, package_factory, tmp_path, mutation, match
) -> None:
    run_dir, plan = package_factory("preflight")
    remote = tmp_path / f"remote-{mutation}"
    _build_remote(run_dir, plan, remote)
    first = sorted(remote.iterdir())[0]
    if mutation == "missing_shard":
        shutil.rmtree(first)
    elif mutation == "extra_root":
        (remote / "unexpected.txt").write_text("extra", encoding="utf-8")
    elif mutation == "missing_file":
        (first / "time.txt").unlink()
    elif mutation == "extra_file":
        (first / "extra.txt").write_text("extra", encoding="utf-8")
    elif mutation == "tamper_hash":
        (first / "teacher.jsonl").write_bytes(b"tampered\n")
    else:
        row = json.loads((first / "teacher.jsonl").read_text(encoding="utf-8"))
        raw = (
            json.dumps(row, sort_keys=True, separators=(", ", ": ")) + "\n"
        ).encode("utf-8")
        (first / "teacher.jsonl").write_bytes(raw)
        done = json.loads((first / "DONE.json").read_text(encoding="utf-8"))
        done["files"]["teacher.jsonl"] = {
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)
        }
        _canonical(first / "DONE.json", done)
    with pytest.raises(ValueError, match=match):
        _receive(
            monkeypatch,
            run_dir,
            plan,
            remote,
            tmp_path / f"received-{mutation}",
            [],
        )


def test_finalize_preflight_normalizes_only_scalar_batch_and_checks_repeat(
    monkeypatch, package_factory, tmp_path
) -> None:
    run_dir, plan = package_factory("preflight")
    remote = tmp_path / "remote-finalize"
    _build_remote(run_dir, plan, remote)
    received = tmp_path / "received-finalize"
    _receive(monkeypatch, run_dir, plan, remote, received, [])
    evidence_root = tmp_path / "evidence-repo"
    inherited = (
        evidence_root
        / "outputs/hu_joint_policy/m43_attempt08_preflight"
        / "regular-hu-m43-attempt08-preflight-finalprop-20260714-213558"
        / "finalization.json"
    )
    _canonical(
        inherited,
        {"status": "pass_correctness_preflight_and_authorize_development200_only"},
    )
    real_sha = spot.sha256_file

    def evidence_sha(path):
        if Path(path).resolve() == inherited.resolve():
            return "681854a8cd37bec1abf6f0ff72e69a8e09e2ad257fe2ecfbea513d3f127dd1b3"
        return real_sha(path)

    monkeypatch.setattr(spot, "_REPO_ROOT", evidence_root)
    monkeypatch.setattr(spot, "sha256_file", evidence_sha)
    result = spot.finalize_preflight(
        received_dir=received, output=tmp_path / "preflight-result.json"
    )
    assert result["deterministic_batch_repeat"] is True
    assert result["scalar_batch_teacher_parity"] is True
    assert result["decision"] == "authorize_development200_package_only"


def test_startup_script_has_valid_bash_syntax() -> None:
    repository = Path(__file__).resolve().parents[1]
    script = Path("scripts") / spot.STARTUP_NAME
    completed = subprocess.run(
        ["bash", "-n", script.as_posix()],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    source = script.read_text(encoding="utf-8")
    assert "scalar child selectors are allowed only in Attempt09 preflight" in source
    assert "--if-generation-match=0" in source
