from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ofc_regular import hu_m43_attempt09_spot as base
from ofc_regular import hu_m43_attempt10_spot as spot
from ofc_regular.hu_m43_attempt10_contract import (
    M43_ATTEMPT10_PLAN_SHA256,
    M43_ATTEMPT10_PROFILES,
)


def _canonical(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(spot.canonical_json_bytes(value))


def _assert_bound() -> None:
    assert base.PACKAGE_SCHEMA == spot.PACKAGE_SCHEMA
    assert base.SHARD_SCHEMA == spot.SHARD_SCHEMA
    assert base.DONE_SCHEMA == spot.DONE_SCHEMA
    assert base.RECEIVE_SCHEMA == spot.RECEIVE_SCHEMA
    assert base.PLAN_RELATIVE == spot.PLAN_RELATIVE
    assert base.GATE_RELATIVE == spot.GATE_RELATIVE
    assert base.OVERLAY_RELATIVES == spot.OVERLAY_RELATIVES
    assert base.M43_ATTEMPT09_PLAN_SHA256 == M43_ATTEMPT10_PLAN_SHA256
    assert base.M43_ATTEMPT09_PROFILES == M43_ATTEMPT10_PROFILES


def test_every_lifecycle_wrapper_binds_attempt10_and_restores_attempt09(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = {
        name: getattr(base, name) for name in spot._ATTEMPT10_BINDINGS
    }
    calls: list[str] = []

    def stub(name, result):
        def invoke(*args, **kwargs):
            _assert_bound()
            calls.append(name)
            return result

        return invoke

    monkeypatch.setattr(base, "build_schedule", stub("schedule", []))
    monkeypatch.setattr(base, "package_attempt09", stub("package", {}))
    monkeypatch.setattr(base, "validate_package", stub("validate_package", {}))
    monkeypatch.setattr(base, "authorize_launch", stub("authorize", {}))
    monkeypatch.setattr(base, "validate_launch", stub("validate_launch", ({}, {})))
    monkeypatch.setattr(base, "launch_wave", stub("launch", {}))
    monkeypatch.setattr(base, "run_status", stub("status", {}))
    monkeypatch.setattr(base, "receive_run", stub("receive", {}))
    monkeypatch.setattr(base, "finalize_preflight", stub("finalize", {}))

    spot.build_schedule("preflight", "attempt10-preflight-test")
    spot.package_attempt10(
        mode="preflight",
        run_name="attempt10-preflight-test",
        run_dir="unused",
        preceding_gate="unused",
    )
    spot.validate_package("unused")
    spot.authorize_launch(run_dir="unused")
    spot.validate_launch("unused")
    spot.launch_wave(
        run_dir="unused", project="p", bucket="b", zone="z", shards=["0"]
    )
    spot.run_status(run_dir="unused", project="p", bucket="b", zone="z")
    spot.receive_run(
        run_dir="unused", project="p", bucket="b", output_dir="unused-output"
    )
    spot.finalize_preflight(received_dir="unused", output="unused-output")

    assert calls == [
        "schedule",
        "package",
        "validate_package",
        "authorize",
        "validate_launch",
        "launch",
        "status",
        "receive",
        "finalize",
    ]
    assert {name: getattr(base, name) for name in original} == original


def test_bindings_restore_even_when_shared_lifecycle_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = {
        name: getattr(base, name) for name in spot._ATTEMPT10_BINDINGS
    }

    def fail(*args, **kwargs):
        _assert_bound()
        raise RuntimeError("expected")

    monkeypatch.setattr(base, "validate_package", fail)
    with pytest.raises(RuntimeError, match="expected"):
        spot.validate_package("unused")
    assert {name: getattr(base, name) for name in original} == original


@pytest.fixture
def package_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repository = tmp_path / "repository"
    template = tmp_path / "template"
    template.mkdir()
    _canonical(
        template / "source_closure_manifest.json",
        {"schema": "minimal_template_v1", "status": "frozen"},
    )
    (template / "template-marker.txt").write_text("template\n", encoding="utf-8")

    for relative in spot.OVERLAY_RELATIVES:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == spot.PLAN_RELATIVE:
            destination.write_bytes(spot.DEFAULT_PLAN.read_bytes())
        else:
            destination.write_text(f"# {relative}\n", encoding="utf-8")
    startup = repository / "scripts" / spot.STARTUP_NAME
    startup.parent.mkdir(parents=True, exist_ok=True)
    startup.write_bytes(spot.DEFAULT_STARTUP.read_bytes())

    monkeypatch.setitem(
        spot._ATTEMPT10_BINDINGS,
        "validate_attempt09_artifact_bindings",
        lambda *args, **kwargs: None,
    )

    def make(mode: str = "preflight") -> Path:
        run_name = f"attempt10-{mode.replace('_', '-')}-test"
        run_dir = repository / "outputs" / "gcp_runs" / run_name
        status, decision = spot.EXPECTED_GATES[mode]
        gate = repository / "gates" / f"{mode}.json"
        _canonical(
            gate,
            {
                "schema": "attempt10_gate_v1",
                "status": status,
                "decision": decision,
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
            },
        )
        spot.package_attempt10(
            mode=mode,
            run_name=run_name,
            run_dir=run_dir,
            repository_root=repository,
            template_package=template,
            plan=repository / spot.PLAN_RELATIVE,
            startup=startup,
            preceding_gate=gate,
        )
        return run_dir

    return make


@pytest.mark.parametrize("mode,total", [("preflight", 5), ("development", 200), ("future_audit", 50)])
def test_package_validate_and_authorize_are_attempt10_closed(
    package_factory, mode: str, total: int
) -> None:
    run_dir = package_factory(mode)
    manifest = spot.validate_package(run_dir)
    authorization = spot.authorize_launch(run_dir=run_dir)
    validated, validated_authorization = spot.validate_launch(run_dir)

    assert manifest == validated
    assert authorization == validated_authorization
    assert manifest["schema"] == spot.PACKAGE_SCHEMA
    assert manifest["total_shards"] == total
    assert manifest["source_name"] == spot.SOURCE_NAME
    assert manifest["plan_sha256"] == M43_ATTEMPT10_PLAN_SHA256
    assert {row["path"] for row in manifest["overlays"]} == set(
        spot.OVERLAY_RELATIVES
    )
    assert manifest["current_profile_mutated"] is False
    assert manifest["runtime_policy_activated"] is False
    if mode == "preflight":
        assert not (run_dir / "execution_authorization.json").exists()
    else:
        execution = json.loads(
            (run_dir / "execution_authorization.json").read_text(encoding="utf-8")
        )
        assert execution["schema"] == spot.ATTEMPT10_AUTHORIZATION_SCHEMA
        assert execution["preceding_gate_artifact"] == spot.GATE_RELATIVE


def test_launch_and_status_use_only_mocked_gcloud(
    monkeypatch: pytest.MonkeyPatch, package_factory
) -> None:
    run_dir = package_factory("preflight")
    spot.authorize_launch(run_dir=run_dir)
    created: list[list[str]] = []

    def fake_run(command, *, timeout=300):
        if command[:4] == ["gcloud", "compute", "images", "describe"]:
            payload = {
                "id": base.EXPECTED_IMAGE_ID,
                "selfLink": base.EXPECTED_IMAGE_SELF_LINK,
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if command[:4] == ["gcloud", "compute", "instances", "create"]:
            created.append(list(command))
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:4] == ["gcloud", "compute", "instances", "list"]:
            return subprocess.CompletedProcess(command, 0, "[]", "")
        raise AssertionError(command)

    def fake_subprocess(command, **kwargs):
        if command[:3] == ["gcloud", "storage", "cp"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:4] in (
            ["gcloud", "storage", "objects", "describe"],
            ["gcloud", "compute", "instances", "describe"],
        ):
            return subprocess.CompletedProcess(command, 1, "", "not found")
        if command[:3] == ["gcloud", "storage", "ls"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(command)

    monkeypatch.setattr(base, "_run", fake_run)
    monkeypatch.setattr(base, "_subprocess_run", fake_subprocess)
    launched = spot.launch_wave(
        run_dir=run_dir,
        project="test",
        bucket="test",
        zone="test-zone",
        shards=["0"],
        no_self_delete=True,
    )
    status = spot.run_status(
        run_dir=run_dir, project="test", bucket="test", zone="test-zone"
    )

    assert launched["schema"] == spot.LAUNCH_WAVE_SCHEMA
    assert launched["created"][0]["shard"] == 0
    assert len(created) == 1
    assert "--provisioning-model=SPOT" in created[0]
    assert status["schema"] == spot.STATUS_SCHEMA
    assert status["done"] == 0 and status["remaining"] == 5


def test_startup_script_has_attempt10_identity_and_valid_bash() -> None:
    repository = Path(__file__).resolve().parents[1]
    relative_script = Path("scripts") / spot.STARTUP_NAME
    script = repository / relative_script
    completed = subprocess.run(
        ["bash", "-n", relative_script.as_posix()],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    source = script.read_text(encoding="utf-8")
    assert "ofc_regular.run_hu_m43_attempt10" in source
    assert "configs/hu_joint_policy_m43_attempt10.json" in source
    assert "hu_m43_attempt10_done_v1" in source
    assert "scalar child selectors are allowed only in Attempt10 preflight" in source
    assert "--if-generation-match=0" in source
    assert "run_hu_m43_attempt09 \"${RUN_ARGS[@]}\"" not in source
