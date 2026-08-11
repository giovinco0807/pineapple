from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

import ofc_regular.hu_m43_attempt08_preflight_spot as spot
from ofc_regular.run_hu_m43_attempt08_preflight import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_START = ROOT / "scripts/Start-GcpHuM43Attempt08PreflightRun.ps1"
PREFLIGHT_STATUS = ROOT / "scripts/Get-GcpHuM43Attempt08PreflightRunStatus.ps1"
PREFLIGHT_RECEIVE = ROOT / "scripts/Receive-GcpHuM43Attempt08PreflightRun.ps1"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _received_bundle(tmp_path: Path, monkeypatch) -> dict:
    run_dir = tmp_path / "run"
    jobs_root = tmp_path / "jobs"
    run_dir.mkdir(parents=True)
    jobs_root.mkdir()
    (run_dir / "manifest.json").write_bytes(b"manifest\n")
    authorization = run_dir / "spot_authorization.json"
    local_evidence = run_dir / "local_evidence.json"
    authorization.write_bytes(b"authorization\n")
    local_evidence.write_bytes(b"evidence\n")
    run_name = "attempt08-test-run"
    done_by_index: dict[int, dict] = {}
    for spec in spot.build_preflight_schedule():
        directory = jobs_root / spec["output_prefix"]
        directory.mkdir()
        proof = directory / "proof.json"
        proof.write_bytes(canonical_json_bytes({"slot": spec["slot"]}))
        proof_sha = spot.sha256_file(proof)
        checkpoint = {
            "schema": spot.CHECKPOINT_SCHEMA,
            "run_name": run_name,
            "job_index": spec["job_index"],
            "slot": spec["slot"],
            "status": "complete",
            "proof_sha256": proof_sha,
            "resume_mode": "complete_no_recompute_needed",
            "new_root_generated": False,
        }
        heartbeat = {
            "schema": spot.HEARTBEAT_SCHEMA,
            "run_name": run_name,
            "job_index": spec["job_index"],
            "slot": spec["slot"],
            "status": "complete",
            "updated_unix_seconds": 0.0,
            "deterministic_recompute_on_preemption": True,
            "proof_sha256": proof_sha,
            "process_alive": False,
            "new_root_generated": False,
            "current_profile_resolved": False,
        }
        elapsed = 10.0 + spec["job_index"]
        rss = 1_000_000 + spec["job_index"]
        summary = {
            "schema": spot.SUMMARY_SCHEMA,
            "run_name": run_name,
            "job_index": spec["job_index"],
            "slot": spec["slot"],
            "proof_sha256": proof_sha,
            "teacher_elapsed_seconds": elapsed,
            "process_peak_rss_bytes": rss,
            "teacher_action_or_value_details_exported": False,
        }
        boot = {
            "schema": "hu_m43_attempt08_boot_image_evidence_v1",
            "run_name": run_name,
            "job_index": spec["job_index"],
            "instance_name": f"attempt08-pf{spec['job_index']}",
            "disk_name": f"attempt08-pf{spec['job_index']}",
            "source_image": (
                "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
                "global/images/debian-12-bookworm-v20260609"
            ),
            "source_image_id": spot.ATTEMPT08_GCP_IMAGE_ID,
        }
        _write(directory / "checkpoint.json", checkpoint)
        _write(directory / "heartbeat.json", heartbeat)
        _write(directory / "summary.json", summary)
        _write(directory / "boot_image_evidence.json", boot)
        (directory / "run.log").write_text(
            f"complete {spec['slot']} {proof_sha}\n", encoding="utf-8"
        )
        done = {
            "run_name": run_name,
            "job_index": spec["job_index"],
            "slot": spec["slot"],
            "proof_sha256": proof_sha,
            "checkpoint_sha256": spot.sha256_file(directory / "checkpoint.json"),
            "heartbeat_sha256": spot.sha256_file(directory / "heartbeat.json"),
            "summary_sha256": spot.sha256_file(directory / "summary.json"),
            "run_log_sha256": spot.sha256_file(directory / "run.log"),
            "boot_image_evidence_sha256": spot.sha256_file(
                directory / "boot_image_evidence.json"
            ),
            "teacher_elapsed_seconds": elapsed,
            "process_peak_rss_bytes": rss,
        }
        done_by_index[spec["job_index"]] = done
        _write(directory / "DONE.json", {"job_index": spec["job_index"]})

    monkeypatch.setattr(
        spot,
        "done_only_status",
        lambda **_kwargs: {
            "all_five_done": True,
            "proof_payloads_opened": False,
        },
    )
    monkeypatch.setattr(
        spot,
        "validate_done_metadata",
        lambda *, job_index, **_kwargs: copy.deepcopy(done_by_index[job_index]),
    )
    monkeypatch.setattr(
        spot,
        "validate_package_artifacts",
        lambda _path: {"run_name": run_name},
    )
    return {
        "run_dir": run_dir,
        "authorization_path": authorization,
        "local_evidence_path": local_evidence,
        "jobs_root": jobs_root,
        "done_by_index": done_by_index,
    }


def test_schedule_is_exact_five_bounded_c4_jobs() -> None:
    schedule = spot.build_preflight_schedule()
    assert [row["job_index"] for row in schedule] == list(range(5))
    assert [row["slot"] for row in schedule] == [
        "root0_batch_a",
        "root0_batch_b",
        "root0_scalar",
        "root1_batch",
        "root2_batch",
    ]
    assert all(row["machine_type"] == "c4-highmem-4" for row in schedule)
    assert all(row["native_batch_threads"] == 4 for row in schedule)
    spot.validate_preflight_schedule(schedule)


def test_actual_received_bundle_is_reopened_and_evidence_is_byte_bound(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _received_bundle(tmp_path, monkeypatch)
    result = spot.validate_received_execution_bundle(**{
        key: value for key, value in paths.items() if key != "done_by_index"
    })
    evidence_path = tmp_path / "execution-evidence.json"
    evidence_path.write_bytes(canonical_json_bytes(result["execution_evidence"]))
    replay = spot.validate_preflight_receive_bundle(
        **{key: value for key, value in paths.items() if key != "done_by_index"},
        execution_evidence_path=evidence_path,
    )
    assert replay["execution_evidence"] == result["execution_evidence"]
    assert set(replay["proof_paths"]) == set(spot.ATTEMPT08_PREFLIGHT_SLOTS)

    forged = json.loads(evidence_path.read_bytes())
    forged["done_sha256"]["root2_batch"] = "0" * 64
    evidence_path.write_bytes(canonical_json_bytes(forged))
    with pytest.raises(ValueError, match="does not match received bundle"):
        spot.validate_preflight_receive_bundle(
            **{key: value for key, value in paths.items() if key != "done_by_index"},
            execution_evidence_path=evidence_path,
        )


def test_relocated_immutable_bundle_skips_only_absolute_command_replay(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _received_bundle(tmp_path, monkeypatch)
    observed_status: list[bool] = []
    observed_done: list[bool] = []

    def _status(**kwargs):
        observed_status.append(bool(kwargs["revalidate_local_commands"]))
        return {"all_five_done": True, "proof_payloads_opened": False}

    def _done(*, job_index, revalidate_local_commands=True, **_kwargs):
        observed_done.append(bool(revalidate_local_commands))
        return copy.deepcopy(paths["done_by_index"][job_index])

    monkeypatch.setattr(spot, "done_only_status", _status)
    monkeypatch.setattr(spot, "validate_done_metadata", _done)
    result = spot.validate_received_execution_bundle(
        **{key: value for key, value in paths.items() if key != "done_by_index"},
        revalidate_local_commands=False,
    )
    assert result["execution_evidence"]["all_five_done_before_payloads_opened"] is True
    assert observed_status == [False]
    assert observed_done == [False] * spot.ATTEMPT08_PREFLIGHT_JOB_COUNT


def test_received_bundle_rejects_extra_job_or_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _received_bundle(tmp_path, monkeypatch)
    (paths["jobs_root"] / "job_999_injected").mkdir()
    with pytest.raises(ValueError, match="job directory set"):
        spot.validate_received_execution_bundle(**{
            key: value for key, value in paths.items() if key != "done_by_index"
        })
    (paths["jobs_root"] / "job_999_injected").rmdir()
    first = paths["jobs_root"] / spot.build_preflight_schedule()[0]["output_prefix"]
    (first / "injected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifacts changed"):
        spot.validate_received_execution_bundle(**{
            key: value for key, value in paths.items() if key != "done_by_index"
        })


@pytest.mark.parametrize("bad_time", [-1.0])
def test_received_bundle_rejects_invalid_complete_heartbeat_time(
    tmp_path: Path, monkeypatch, bad_time: float
) -> None:
    paths = _received_bundle(tmp_path, monkeypatch)
    spec = spot.build_preflight_schedule()[0]
    heartbeat_path = paths["jobs_root"] / spec["output_prefix"] / "heartbeat.json"
    heartbeat = json.loads(heartbeat_path.read_bytes())
    heartbeat["updated_unix_seconds"] = bad_time
    _write(heartbeat_path, heartbeat)
    paths["done_by_index"][0]["heartbeat_sha256"] = spot.sha256_file(heartbeat_path)
    with pytest.raises(ValueError, match="heartbeat semantics"):
        spot.validate_received_execution_bundle(**{
            key: value for key, value in paths.items() if key != "done_by_index"
        })


def test_received_bundle_rejects_support_and_boot_image_tampering(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _received_bundle(tmp_path, monkeypatch)
    spec = spot.build_preflight_schedule()[2]
    directory = paths["jobs_root"] / spec["output_prefix"]
    (directory / "run.log").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="support artifact changed"):
        spot.validate_received_execution_bundle(**{
            key: value for key, value in paths.items() if key != "done_by_index"
        })

    paths = _received_bundle(tmp_path / "boot", monkeypatch)
    spec = spot.build_preflight_schedule()[1]
    boot_path = paths["jobs_root"] / spec["output_prefix"] / "boot_image_evidence.json"
    boot = json.loads(boot_path.read_bytes())
    boot["source_image_id"] = "123"
    _write(boot_path, boot)
    paths["done_by_index"][1]["boot_image_evidence_sha256"] = spot.sha256_file(
        boot_path
    )
    with pytest.raises(ValueError, match="boot image evidence"):
        spot.validate_received_execution_bundle(**{
            key: value for key, value in paths.items() if key != "done_by_index"
        })


def test_startup_and_powershell_enforce_immutable_race_safe_lifecycle() -> None:
    startup = (ROOT / "scripts/startup_hu_m43_attempt08_preflight.sh").read_text(
        encoding="utf-8"
    )
    assert "debian-12-bookworm-v20260609" in startup
    assert "--image-family" not in startup
    assert "timeout --signal=TERM --kill-after=60s 2500s" in startup
    assert 'python3 -m venv "$RUNMETA/.venv"' in startup
    assert "python3 -m venv .venv" not in startup
    assert "PYTHONDONTWRITEBYTECODE=1" in startup
    assert 'then exit 0; fi' not in startup
    assert "Remote DONE exists; reopening the complete immutable job bundle" in startup
    assert "validate_received_job_artifacts" in startup
    assert "REMOTE_DONE_INVALID" in startup
    assert 'objects describe "$RESULT_URI/proof.json"' in startup
    assert 'upload_once_or_verify "$RESULT/proof.json"' in startup
    assert 'upload_once_or_verify "$RESULT/$name" "$RESULT_URI/$name"' in startup
    assert 'gcloud storage cp "$RESULT/$name" "$RESULT_URI/$name"' not in startup
    assert startup.find('upload_once_or_verify "$RESULT/proof.json"') < startup.find(
        'upload_once_or_verify "$RESULT/$name"'
    )
    assert startup.rfind('upload_once_or_verify "$RESULT/DONE.json"') > startup.rfind(
        'upload_once_or_verify "$RESULT/$name"'
    )

    start = (ROOT / "scripts/Start-GcpHuM43Attempt08PreflightRun.ps1").read_text(
        encoding="utf-8"
    )
    assert "SkipExistingInstances" not in start
    assert "'--provisioning-model','SPOT'" in start
    assert "'--maintenance-policy','TERMINATE','--no-restart-on-failure'" in start
    assert "& gcloud" not in start
    assert "Invoke-M43A4GcloudProcess" in start
    assert "Test-M43A4GcsObject" in start
    assert "Publish-M43A8ImmutableObject" in start
    assert "$FrozenSourceRoot = Join-Path $RunDir 'package_src/src'" in start
    assert "-SourceRoot $FrozenSourceRoot" in start
    assert "$env:PYTHONDONTWRITEBYTECODE = '1'" in start
    assert "Start-GcpHuM43Attempt08PreflightRun.ps1 differs from its frozen packaged copy" in start
    assert "Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256" in start
    assert ". $FrozenCommon08" in start
    assert "label = 'HuM43Attempt04Spot.Common.ps1'" in start
    assert "$modes = @(" in start
    for name in (
        "Get-GcpHuM43Attempt08PreflightRunStatus.ps1",
        "Receive-GcpHuM43Attempt08PreflightRun.ps1",
    ):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "validate-launch" in text
        assert "Resolve-JobMirror" in text
        assert "Unsafe output_prefix" in text
        assert "$FrozenSourceRoot=Join-Path $RunDir 'package_src/src'" in text
        assert "$env:PYTHONPATH=$FrozenSourceRoot" in text
        assert "$env:PYTHONDONTWRITEBYTECODE='1'" in text
        assert "$env:PYTHONPATH=Join-Path $RepoRoot 'src'" not in text
        assert "differs from its frozen packaged copy" in text
        assert "Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256" in text
        assert ". $FrozenCommon08" in text
        assert "Copy-M43A4RemoteFileExact" in text
        assert "& gcloud" not in text
        if name.startswith("Get-"):
            assert "Test-M43A4GcsObject" in text


def test_start_exactly_one_phase_survives_windows_powershell_strict_mode() -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:  # pragma: no cover - Windows is the operational host
        pytest.skip("PowerShell is unavailable")
    run_name = f"attempt08-mode-{uuid.uuid4().hex}"
    result = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts/Start-GcpHuM43Attempt08PreflightRun.ps1"),
            "-RunName",
            run_name,
            "-CreateAuthorization",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Package manifest is missing; run PackageOnly first" in combined
    assert "property 'Count' cannot be found" not in combined


def test_fixed_startup_syntax_gate_uses_portable_relative_path(
    tmp_path: Path,
) -> None:
    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover
        pytest.skip("bash is unavailable")
    run_dir = tmp_path / "run"
    package_src = run_dir / "package_src"
    package_src.mkdir(parents=True)
    (run_dir / "startup_hu_m43_attempt08_preflight.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8"
    )
    command = spot._fixed_local_gate_commands(run_dir / "manifest.json")[
        "startup_shell_syntax"
    ]
    assert command == [bash, "-n", "../startup_hu_m43_attempt08_preflight.sh"]
    result = subprocess.run(
        command,
        cwd=package_src,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_create_instances_treats_expected_404_as_absent_under_strict_mode(
    tmp_path: Path,
) -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None or os.name != "nt":  # pragma: no cover
        pytest.skip("Windows PowerShell is required")

    run_name = f"attempt08-gcloud-404-{uuid.uuid4().hex}"
    run_dir = ROOT / "outputs" / "gcp_runs" / run_name
    package_src = run_dir / "package_src"
    frozen_scripts = package_src / "scripts"
    fake_bin = tmp_path / "bin"
    gcloud_log = tmp_path / "gcloud.log"
    try:
        (package_src / "src").mkdir(parents=True)
        packaged_source = package_src / "preflight_source" / "teacher.jsonl"
        packaged_source.parent.mkdir(parents=True)
        packaged_source.write_bytes(b"frozen preflight source\n")
        frozen_scripts.mkdir(parents=True)
        fake_bin.mkdir()
        for path in (
            PREFLIGHT_START,
            ROOT / "scripts/HuM43Attempt08Spot.Common.ps1",
            ROOT / "scripts/HuM43Attempt04Spot.Common.ps1",
        ):
            shutil.copy2(path, frozen_scripts / path.name)

        startup = ROOT / "scripts/startup_hu_m43_attempt08_preflight.sh"
        shutil.copy2(startup, run_dir / startup.name)
        (run_dir / "source.zip").write_bytes(b"synthetic source\n")
        (run_dir / "local_evidence.json").write_text("{}\n", encoding="utf-8")
        output_prefixes = (
            "job_000_root0_batch_a",
            "job_001_root0_batch_b",
            "job_002_root0_scalar",
            "job_003_root1_batch",
            "job_004_root2_batch",
        )
        schedule = "".join(
            json.dumps(
                {"job_index": index, "output_prefix": output_prefix},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for index, output_prefix in enumerate(output_prefixes)
        )
        (run_dir / "shards_manifest.jsonl").write_text(schedule, encoding="utf-8")
        manifest = {
            "run_name": run_name,
            "machine_type": "c4-highmem-4",
            "gcp_image_name": "debian-12-bookworm-v20260609",
            "gcp_image_id": "1449487925682397051",
            "source_zip_sha256": hashlib.sha256(b"synthetic source\n").hexdigest(),
            "startup_sha256": hashlib.sha256(startup.read_bytes()).hexdigest(),
        }
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        authorization = {
            "spot_authorized": True,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
        (run_dir / "spot_authorization.json").write_text(
            json.dumps(authorization, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )

        (fake_bin / "python.cmd").write_text(
            "@echo off\r\nexit /b 0\r\n", encoding="utf-8"
        )
        (fake_bin / "gcloud.ps1").write_text(
            "\n".join(
                (
                    "$line = ($args -join ' ')",
                    "[IO.File]::AppendAllText($env:M43A8_FAKE_GCLOUD_LOG, $line + [Environment]::NewLine)",
                    "if ($args[0] -eq 'compute' -and $args[1] -eq 'images' -and $args[2] -eq 'describe') {",
                    "  Write-Output '{\"name\":\"debian-12-bookworm-v20260609\",\"id\":\"1449487925682397051\",\"selfLink\":\"https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/debian-12-bookworm-v20260609\"}'",
                    "  exit 0",
                    "}",
                    "if ($args[0] -eq 'storage' -and $args[1] -eq 'cp') { exit 0 }",
                    "if ($args[0] -eq 'storage' -and $args[1] -eq 'objects' -and $args[2] -eq 'describe') {",
                    "  [Console]::Error.WriteLine('ERROR: object was not found (404)')",
                    "  exit 1",
                    "}",
                    "if ($args[0] -eq 'compute' -and $args[1] -eq 'instances' -and $args[2] -eq 'describe') {",
                    "  [Console]::Error.WriteLine('ERROR: instance was not found (404)')",
                    "  exit 1",
                    "}",
                    "if ($args[0] -eq 'compute' -and $args[1] -eq 'instances' -and $args[2] -eq 'create') { exit 0 }",
                    "[Console]::Error.WriteLine('unexpected fake gcloud invocation: ' + $line)",
                    "exit 99",
                    "",
                )
            ),
            encoding="utf-8",
        )
        environment = dict(os.environ)
        environment["PATH"] = str(fake_bin) + os.pathsep + environment.get("PATH", "")
        environment["M43A8_FAKE_GCLOUD_LOG"] = str(gcloud_log)
        result = subprocess.run(
            [
                powershell,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(PREFLIGHT_START),
                "-RunName",
                run_name,
                "-CreateInstances",
                "-Jobs",
                "0",
                "-NoSelfDelete",
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, combined
        assert "Started 1 bounded c4-highmem-4 Spot jobs." in combined
        log = gcloud_log.read_text(encoding="utf-8")
        assert "storage objects describe" in log
        assert "compute instances describe" in log
        assert "compute instances create" in log
        assert "property 'Count' cannot be found" not in combined
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_status_treats_missing_done_as_incomplete_under_strict_mode(
    tmp_path: Path,
) -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None or os.name != "nt":  # pragma: no cover
        pytest.skip("Windows PowerShell is required")

    run_name = f"attempt08-status-404-{uuid.uuid4().hex}"
    run_dir = ROOT / "outputs" / "gcp_runs" / run_name
    package_src = run_dir / "package_src"
    frozen_scripts = package_src / "scripts"
    fake_bin = tmp_path / "bin"
    gcloud_log = tmp_path / "gcloud.log"
    try:
        (package_src / "src").mkdir(parents=True)
        frozen_scripts.mkdir(parents=True)
        fake_bin.mkdir()
        for path in (
            PREFLIGHT_STATUS,
            ROOT / "scripts/HuM43Attempt08Spot.Common.ps1",
            ROOT / "scripts/HuM43Attempt04Spot.Common.ps1",
        ):
            shutil.copy2(path, frozen_scripts / path.name)
        (run_dir / "manifest.json").write_text(
            json.dumps({"run_name": run_name}, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "spot_authorization.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "local_evidence.json").write_text("{}\n", encoding="utf-8")
        output_prefixes = (
            "job_000_root0_batch_a",
            "job_001_root0_batch_b",
            "job_002_root0_scalar",
            "job_003_root1_batch",
            "job_004_root2_batch",
        )
        (run_dir / "shards_manifest.jsonl").write_text(
            "".join(
                json.dumps(
                    {"job_index": index, "output_prefix": output_prefix},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for index, output_prefix in enumerate(output_prefixes)
            ),
            encoding="utf-8",
        )
        (fake_bin / "python.cmd").write_text(
            "@echo off\r\nexit /b 0\r\n", encoding="utf-8"
        )
        (fake_bin / "gcloud.ps1").write_text(
            "\n".join(
                (
                    "$line = ($args -join ' ')",
                    "[IO.File]::AppendAllText($env:M43A8_FAKE_GCLOUD_LOG, $line + [Environment]::NewLine)",
                    "if ($args[0] -eq 'storage' -and $args[1] -eq 'objects' -and $args[2] -eq 'describe') {",
                    "  [Console]::Error.WriteLine('ERROR: DONE was not found (404)')",
                    "  exit 1",
                    "}",
                    "[Console]::Error.WriteLine('unexpected fake gcloud invocation: ' + $line)",
                    "exit 99",
                    "",
                )
            ),
            encoding="utf-8",
        )
        environment = dict(os.environ)
        environment["PATH"] = str(fake_bin) + os.pathsep + environment.get("PATH", "")
        environment["M43A8_FAKE_GCLOUD_LOG"] = str(gcloud_log)
        result = subprocess.run(
            [
                powershell,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(PREFLIGHT_STATUS),
                "-RunName",
                run_name,
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, combined
        log_lines = gcloud_log.read_text(encoding="utf-8").splitlines()
        assert len([line for line in log_lines if "storage objects describe" in line]) == 5
        assert not any("storage cp" in line for line in log_lines)
        assert "NativeCommandError" not in combined
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_receive_default_output_is_external_and_protected_from_run_root(
    tmp_path: Path,
) -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None or os.name != "nt":  # pragma: no cover
        pytest.skip("Windows PowerShell is required")

    run_name = f"attempt08-receive-root-{uuid.uuid4().hex}"
    run_dir = ROOT / "outputs" / "gcp_runs" / run_name
    expected_output = (
        ROOT / "outputs" / "hu_joint_policy" / "m43_attempt08_preflight" / run_name
    )
    package_src = run_dir / "package_src"
    frozen_scripts = package_src / "scripts"
    fake_bin = tmp_path / "bin"
    gcloud_log = tmp_path / "gcloud.log"
    python_log = tmp_path / "python.log"
    try:
        (package_src / "src").mkdir(parents=True)
        packaged_source = package_src / "preflight_source" / "teacher.jsonl"
        packaged_source.parent.mkdir(parents=True)
        packaged_source.write_bytes(b"frozen preflight source\n")
        frozen_scripts.mkdir(parents=True)
        fake_bin.mkdir()
        for path in (
            PREFLIGHT_RECEIVE,
            ROOT / "scripts/HuM43Attempt08Spot.Common.ps1",
            ROOT / "scripts/HuM43Attempt04Spot.Common.ps1",
        ):
            shutil.copy2(path, frozen_scripts / path.name)
        (run_dir / "manifest.json").write_text(
            json.dumps({"run_name": run_name}, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "spot_authorization.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "local_evidence.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "shards_manifest.jsonl").write_text(
            json.dumps(
                {"job_index": 0, "output_prefix": "job_000_root0_batch_a"},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        (fake_bin / "python.cmd").write_text(
            "@echo %*>>\"%M43A8_FAKE_PYTHON_LOG%\"\r\n@exit /b 0\r\n",
            encoding="utf-8",
        )
        (fake_bin / "gcloud.ps1").write_text(
            "\n".join(
                (
                    "$line = ($args -join ' ')",
                    "[IO.File]::AppendAllText($env:M43A8_FAKE_GCLOUD_LOG, $line + [Environment]::NewLine)",
                    "if ($args[0] -eq 'storage' -and $args[1] -eq 'cp') { exit 0 }",
                    "[Console]::Error.WriteLine('unexpected fake gcloud invocation: ' + $line)",
                    "exit 99",
                    "",
                )
            ),
            encoding="utf-8",
        )
        environment = dict(os.environ)
        environment["PATH"] = str(fake_bin) + os.pathsep + environment.get("PATH", "")
        environment["M43A8_FAKE_GCLOUD_LOG"] = str(gcloud_log)
        environment["M43A8_FAKE_PYTHON_LOG"] = str(python_log)
        base_command = [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PREFLIGHT_RECEIVE),
            "-RunName",
            run_name,
        ]
        result = subprocess.run(
            base_command,
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, combined
        python_lines = python_log.read_text(encoding="utf-8").splitlines()
        receive_lines = [line for line in python_lines if " receive " in f" {line} "]
        assert len(receive_lines) == 1
        assert f"--output-dir {expected_output}" in receive_lines[0]
        assert f"--source {packaged_source}" in receive_lines[0]
        assert not str(expected_output).startswith(str(run_dir) + os.sep)

        rejected = subprocess.run(
            base_command + ["-OutputDir", str(run_dir / "received")],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        rejected_text = rejected.stdout + rejected.stderr
        assert rejected.returncode != 0
        assert "OutputDir must be the exact external preflight result directory" in rejected_text
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(expected_output, ignore_errors=True)


def test_caller_supplied_local_evidence_api_is_disabled() -> None:
    with pytest.raises(ValueError, match="caller-supplied"):
        spot.create_local_evidence_receipt(evidence={"status": "pass"})


def test_receive_requires_exact_frozen_packaged_preflight_source(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    packaged_source = run_dir / "package_src" / "preflight_source" / "teacher.jsonl"
    packaged_source.parent.mkdir(parents=True)
    packaged_source.write_bytes(b"not opened for this path-identity regression\n")
    outside = tmp_path / "teacher.jsonl"
    outside.write_bytes(packaged_source.read_bytes())

    with pytest.raises(ValueError, match="exact frozen packaged"):
        spot.receive_and_finalize(
            run_dir=run_dir,
            authorization_path=tmp_path / "authorization.json",
            local_evidence_path=tmp_path / "evidence.json",
            jobs_root=tmp_path / "jobs",
            output_dir=tmp_path / "output",
            source=outside,
        )


def test_receive_rejects_symlinked_packaged_preflight_source(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    target = tmp_path / "source.jsonl"
    target.write_bytes(b"same bytes are not sufficient\n")
    packaged_source = run_dir / "package_src" / "preflight_source" / "teacher.jsonl"
    packaged_source.parent.mkdir(parents=True)
    try:
        packaged_source.symlink_to(target)
    except OSError:  # pragma: no cover - Windows may deny symlink creation
        pytest.skip("symlink creation is unavailable")
    monkeypatch.setattr(
        spot,
        "validate_package_artifacts",
        lambda _run_dir: {
            "source_merged_sha256": spot.ATTEMPT07_PREFLIGHT_SOURCE_SHA256
        },
    )
    with pytest.raises(ValueError, match="packaged preflight source changed"):
        spot.receive_and_finalize(
            run_dir=run_dir,
            authorization_path=tmp_path / "authorization.json",
            local_evidence_path=tmp_path / "evidence.json",
            jobs_root=tmp_path / "jobs",
            output_dir=tmp_path / "output",
            source=packaged_source,
        )
