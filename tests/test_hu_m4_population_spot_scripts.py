from __future__ import annotations

import hashlib
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ofc_regular.hu_m43_pilot_contract import canonical_manifest_sha256


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts" / "Start-GcpHuM4PopulationRun.ps1"
STATUS = ROOT / "scripts" / "Get-GcpHuM4PopulationRunStatus.ps1"
RECEIVE = ROOT / "scripts" / "Receive-GcpHuM4PopulationRun.ps1"
PLAN = ROOT / "configs" / "hu_joint_policy_m43_population.json"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _embedded_launch_preflight() -> str:
    match = re.search(
        r"\$lifecyclePreflight = @'\r?\n(?P<script>.*?)\r?\n'@",
        _read(START),
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group("script")


def _launch_preflight_fixture(
    tmp_path: Path, *, mutation: str | None = None
) -> list[Path]:
    fixtures = runpy.run_path(str(ROOT / "tests" / "test_validate_hu_m4_acceptance.py"))
    (
        training,
        contract,
        freeze,
        receipt,
        marker,
        plan,
        _,
    ) = fixtures["_m43_inputs"]()

    model_path = tmp_path / "m43_model.pkl"
    model_path.write_bytes(b"m43 launch preflight model bytes\n")
    model_sha = _sha256(model_path)
    training["runtime_lock"]["candidate_model_sha256"] = model_sha
    training["runtime_lock"]["safety_model_sha256"] = model_sha
    training["runtime_lock"]["safety_threshold"] = freeze["frozen_threshold"]

    plan.update(
        {
            "paired_seeds_per_opponent": 2,
            "shards": 1,
            "paired_seeds_per_shard": 2,
            "activation_guards": {
                "current_profile_changed": False,
                "runtime_policy_activated": False,
                "full_replacement_enabled": False,
                "threshold_changed_after_lock": False,
            },
            "sizing_rule": {
                "fixed_before_realized_population_labels": True,
                "optional_stopping_or_posthoc_extension_allowed": False,
            },
            "freshness": {
                "exclude_all_m4_m41_m42_m43_teacher_hand_seeds": True,
                "exclude_prior_population_smoke_seeds": True,
                "planned_overlap_count_at_freeze": 0,
                "excluded_population_schedules": [
                    {"seed": 20_000, "seed_stride": 101, "paired_seeds": 2}
                ],
            },
        }
    )
    if mutation == "teacher_overlap":
        plan["seed"] = 1
    elif mutation == "prior_overlap":
        plan["freshness"]["excluded_population_schedules"][0]["seed"] = plan["seed"]

    training_path = tmp_path / "training.json"
    _write_json(training_path, training)
    freeze["training_manifest_sha256"] = _sha256(training_path)
    freeze["model_sha256"] = model_sha
    receipt["model_sha256"] = model_sha
    marker["model_sha256"] = model_sha
    freeze_sha = canonical_manifest_sha256(freeze)
    receipt["freeze_manifest_sha256"] = freeze_sha
    marker["freeze_manifest_sha256"] = freeze_sha
    if mutation == "marker_content":
        marker["freeze_manifest_sha256"] = "f" * 64
    elif mutation == "receipt_freeze_binding":
        receipt["freeze_manifest_sha256"] = "f" * 64

    marker_path = tmp_path / "M43_LOCKED_CONSUMED.json"
    _write_json(marker_path, marker)
    receipt["consumption_marker_sha256"] = _sha256(marker_path)

    paths_and_payloads = (
        (tmp_path / "data_contract.json", contract),
        (tmp_path / "freeze.json", freeze),
        (tmp_path / "receipt.json", receipt),
        (tmp_path / "population_plan.json", plan),
    )
    for path, payload in paths_and_payloads:
        _write_json(path, payload)
    if mutation == "model_bytes":
        model_path.write_bytes(model_path.read_bytes() + b"tampered")
    elif mutation == "marker_bytes":
        marker_path.write_bytes(marker_path.read_bytes() + b"\n")
    return [
        model_path,
        training_path,
        paths_and_payloads[0][0],
        paths_and_payloads[1][0],
        paths_and_payloads[2][0],
        marker_path,
        paths_and_payloads[3][0],
    ]


def _run_embedded_launch_preflight(paths: list[Path]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-", *(str(path) for path in paths)],
        cwd=ROOT,
        env=environment,
        input=_embedded_launch_preflight() + "\n",
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )


def _resume_fixture(
    tmp_path: Path, *, omit_model_identity: bool = False
) -> tuple[Path, str, str, str]:
    repo = tmp_path / "resume_repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    copied_start = scripts / START.name
    shutil.copy2(START, copied_start)
    run_name = "resume-contract-test"
    project = "test-project"
    bucket = "test-bucket"
    run_dir = repo / "outputs" / "gcp_runs" / run_name
    run_dir.mkdir(parents=True)
    local_files = {
        "source": run_dir / "ofc_regular_hu_m4_population_source.zip",
        "startup": run_dir / "startup_hu_m4_population.sh",
        "shards": run_dir / "population_shards.jsonl",
    }
    local_files["source"].write_bytes(b"source archive")
    local_files["startup"].write_bytes(b"startup")
    local_files["shards"].write_bytes(b'{"shard":0}\n')
    model_sha = "1" * 64
    training_sha = "2" * 64
    freeze_file_sha = "3" * 64
    receipt_file_sha = "4" * 64
    marker_file_sha = "5" * 64
    plan_file_sha = "6" * 64
    manifest: dict[str, object] = {
        "schema": "hu_m4_population_spot_manifest_v1",
        "run_name": run_name,
        "project_id": project,
        "bucket": bucket,
        "source": {"sha256": _sha256(local_files["source"])},
        "startup": {"sha256": _sha256(local_files["startup"])},
        "shards": {"sha256": _sha256(local_files["shards"]), "count": 1},
        "population_plan": {
            "uri": f"gs://{bucket}/runs/{run_name}/source/population_plan.json",
            "sha256": plan_file_sha,
            "paired_seeds": 2,
        },
        "runtime": {
            "model_sha256": model_sha,
            "training_manifest_sha256": training_sha,
            "data_contract_sha256": "7" * 64,
            "freeze_manifest_sha256": freeze_file_sha,
            "locked_receipt_sha256": receipt_file_sha,
            "consumption_marker_sha256": marker_file_sha,
        },
        "launch_preflight": {
            "schema": "hu_m43_population_launch_preflight_v1",
            "status": "pass",
            "model_sha256": model_sha,
            "training_manifest_sha256": training_sha,
            "data_contract_file_sha256": "8" * 64,
            "population_plan_file_sha256": plan_file_sha,
            "freeze_manifest_canonical_sha256": "9" * 64,
            "locked_receipt_canonical_sha256": "a" * 64,
            "consumption_marker_canonical_sha256": "b" * 64,
            "freeze_manifest_file_sha256": freeze_file_sha,
            "locked_receipt_file_sha256": receipt_file_sha,
            "consumption_marker_sha256": marker_file_sha,
            "teacher_hand_seeds_checked": 6,
            "population_hand_seeds_checked": 2,
            "teacher_overlap_count": 0,
            "prior_population_overlap_count": 0,
        },
        "no_runtime_activation": True,
        "current_profile_mutated": False,
    }
    if omit_model_identity:
        del manifest["runtime"]["model_sha256"]
        del manifest["launch_preflight"]["model_sha256"]
    _write_json(run_dir / "population_run_manifest.json", manifest)
    return copied_start, run_name, project, bucket


def _run_resume_dry_run(
    shell: str, copied_start: Path, run_name: str, project: str, bucket: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            shell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(copied_start),
            "-ModelPath",
            "unused-model",
            "-TrainingManifestPath",
            "unused-training",
            "-DataContractPath",
            "unused-contract",
            "-FreezeManifestPath",
            "unused-freeze",
            "-LockedReceiptPath",
            "unused-receipt",
            "-ConsumptionMarkerPath",
            "unused-marker",
            "-RunName",
            run_name,
            "-ProjectId",
            project,
            "-Bucket",
            bucket,
            "-ResumeExisting",
            "-DryRun",
        ],
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )


def _status_fixture(
    tmp_path: Path,
    *,
    uri_shard: int = 0,
    row_shard: object = 0,
    duplicate_uri: bool = False,
    wrong_prefix: bool = False,
) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "status_repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    copied_status = scripts / STATUS.name
    shutil.copy2(STATUS, copied_status)
    run_name, project, bucket = "status-contract-test", "test-project", "test-bucket"
    run_dir = repo / "outputs" / "gcp_runs" / run_name
    run_dir.mkdir(parents=True)
    source_sha, model_sha = "1" * 64, "2" * 64
    manifest = {
        "schema": "hu_m4_population_spot_manifest_v1",
        "run_name": run_name,
        "project_id": project,
        "bucket": bucket,
        "shards": {"count": 1},
        "source": {"sha256": source_sha},
        "runtime": {"model_sha256": model_sha},
        "no_runtime_activation": True,
        "current_profile_mutated": False,
    }
    manifest_path = run_dir / "population_run_manifest.json"
    _write_json(manifest_path, manifest)
    canonical_prefix = f"gs://{bucket}/runs/{run_name}"
    listed_prefix = (
        f"gs://{bucket}/runs/another-run" if wrong_prefix else canonical_prefix
    )
    done_uri = f"{listed_prefix}/results/shard-{uri_shard:04d}/DONE"
    done_row = {
        "schema": "hu_m4_population_spot_done_v1",
        "status": "complete",
        "run_name": run_name,
        "shard": row_shard,
        "manifest_sha256": _sha256(manifest_path),
        "source_sha256": source_sha,
        "model_sha256": model_sha,
        "evaluation_sha256": "3" * 64,
        "records_sha256": "4" * 64,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }
    state = {
        "done_uris": [done_uri, done_uri] if duplicate_uri else [done_uri],
        "done_rows": {done_uri: done_row},
    }
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    state_path = fake_bin / "state.json"
    _write_json(state_path, state)
    fake_python = fake_bin / "fake_gcloud.py"
    fake_python.write_text(
        """import json, os, sys
state = json.load(open(os.environ['FAKE_GCLOUD_STATE'], encoding='utf-8'))
args = sys.argv[1:]
if args[:2] == ['storage', 'ls']:
    target = args[2]
    if '/results/*/DONE' in target:
        print('\\n'.join(state['done_uris']))
        raise SystemExit(0)
    if '/status/*.json' in target:
        print('No URLs matched', file=sys.stderr)
        raise SystemExit(1)
if args[:2] == ['storage', 'cat']:
    print(json.dumps(state['done_rows'][args[2]], sort_keys=True))
    raise SystemExit(0)
if args[:3] == ['compute', 'instances', 'list']:
    print('[]')
    raise SystemExit(0)
print('unexpected fake gcloud call: ' + repr(args), file=sys.stderr)
raise SystemExit(2)
""",
        encoding="utf-8",
    )
    (fake_bin / "gcloud.cmd").write_text(
        f'@echo off\r\n"{sys.executable}" "%~dp0fake_gcloud.py" %*\r\n',
        encoding="utf-8",
    )
    return copied_status, {
        "RUN_NAME": run_name,
        "PROJECT": project,
        "BUCKET": bucket,
        "FAKE_GCLOUD_STATE": str(state_path),
        "FAKE_BIN": str(fake_bin),
    }


def _run_status(
    shell: str, copied_status: Path, values: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["FAKE_GCLOUD_STATE"] = values["FAKE_GCLOUD_STATE"]
    environment["PATH"] = values["FAKE_BIN"] + os.pathsep + environment["PATH"]
    return subprocess.run(
        [
            shell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(copied_status),
            "-RunName",
            values["RUN_NAME"],
            "-ProjectId",
            values["PROJECT"],
            "-Bucket",
            values["BUCKET"],
        ],
        env=environment,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )


def test_frozen_population_plan_has_power_and_all_final_gates() -> None:
    text = _read(PLAN)
    for token in (
        '"paired_seeds_per_opponent": 1000',
        '"shards": 20',
        '"paired_seeds_per_shard": 50',
        '"second_seat_override_opportunities": 4000',
        '"minimum_valid_overrides": 300',
        '"minimum_population_fire_rate_needed": 0.075',
        '"optional_stopping_or_posthoc_extension_allowed": false',
        '"paired_seat_swap": true',
        '"current_profile_changed": false',
        '"threshold_changed_after_lock": false',
    ):
        assert token in text


def test_start_binds_lifecycle_and_uses_resumable_spot_shards() -> None:
    text = _read(START)
    for token in (
        "TrainingManifestPath",
        "DataContractPath",
        "FreezeManifestPath",
        "LockedReceiptPath",
        "ConsumptionMarkerPath",
        "model_and_threshold_frozen_locked_unopened",
        "evaluated_once_diagnostic_only_no_activation",
        "consumption_marker_sha256",
        "requires_fresh_population_acceptance",
        "minimum_population_valid_overrides",
        "--provisioning-model",
        '"SPOT"',
        "--instance-termination-action",
        'checkpoint = [ordered]@{ unit = "completed_shard"',
        "resume_missing_shards_only = $true",
        "hu_m4_population_spot_status_v1",
        "hu_m4_population_spot_done_v1",
        "current_profile_mutated = $false",
        "no_runtime_activation = $true",
        "validate_model_threshold_freeze",
        "validate_locked_holdout_receipt",
        "canonical_manifest_sha256",
        "consumption marker key set changed",
        "population/teacher seed overlap",
        "population/prior schedule seed overlap",
        "M4.3 lifecycle artifacts changed while the immutable package was assembled",
        "hu_m43_population_launch_preflight_v1",
        "Frozen remote population manifest disagrees with the local resume manifest",
        "$manifest.population_plan.uri",
        "Assert-Sha256Text",
        "$frozenPlanPath",
    ):
        assert token in text
    done_upload = 'gcloud storage cp "$OUT/DONE" "$RESULT/DONE" --if-generation-match=0'
    assert text.count(done_upload) == 1
    assert text.index(done_upload) > text.index(
        'gcloud storage cp "$OUT/evaluation.json" "$RESULT/evaluation.json"'
    )
    assert text.index(done_upload) > text.index("write_status complete 0")


def test_worker_runs_fixed_four_opponent_paired_evaluator_without_current() -> None:
    text = _read(START)
    command = (
        "python -m ofc_regular.evaluate_hu_m4_population --model "
        "artifacts/m43_model.pkl"
    )
    assert command in text
    assert (
        "--opponents stage19_p0 stage9f_p2 stage7_m5_r10 "
        "random_exact_final"
    ) in text
    assert "--seed-stride" in text
    assert "r['current_profile_used'] is False" in text
    assert "r['promotion_artifact_contract'] is True" in text
    assert "build_policy(\"current\"" not in text


def test_receiver_verifies_every_done_and_recomputes_merged_confidence_intervals() -> None:
    text = _read(RECEIVE)
    for token in (
        "outputs/hu_joint_policy/m43_population",
        "refusing profile/current overwrite",
        "hu_m4_population_spot_done_v1",
        "evaluation_sha256",
        "records_sha256",
        "merge_manifest.json",
        "$receipt.merge_manifest_sha256",
        "ofc_regular.merge_hu_m4_population_shards",
        "population_plan_sha256",
        "complete_content_verified",
        "verified_and_merged",
        "current_profile_mutated=$false",
        "no_runtime_activation=$true",
        "Move-Item -LiteralPath $stage -Destination $OutputDir",
    ):
        assert token in text


def test_status_tracks_done_commit_and_missing_shards() -> None:
    text = _read(STATUS)
    for token in (
        "hu_m4_population_spot_done_v1",
        "shards_complete",
        "missing_indices",
        "active_instances",
        "no_runtime_activation=$true",
        "seenDoneShards.Add($uriShard)",
        "$rowShard -ne $uriShard",
        "Population DONE URI shard is outside frozen range",
        "Population DONE URI is outside the frozen run prefix",
        "evaluation_sha256 -notmatch '^[0-9a-f]{64}$'",
        "records_sha256 -notmatch '^[0-9a-f]{64}$'",
        "Convert-RequiredJsonInteger $row.shard",
    ):
        assert token in text


def test_embedded_launch_preflight_accepts_complete_hash_and_seed_chain(
    tmp_path: Path,
) -> None:
    completed = _run_embedded_launch_preflight(_launch_preflight_fixture(tmp_path))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["schema"] == "hu_m43_population_launch_preflight_v1"
    assert receipt["status"] == "pass"
    assert receipt["teacher_overlap_count"] == 0
    assert receipt["prior_population_overlap_count"] == 0
    for key in (
        "freeze_manifest_canonical_sha256",
        "locked_receipt_canonical_sha256",
        "consumption_marker_canonical_sha256",
        "freeze_manifest_file_sha256",
        "locked_receipt_file_sha256",
        "consumption_marker_sha256",
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", receipt[key])


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("marker_content", "consumption marker content mismatch"),
        ("receipt_freeze_binding", "locked receipt is not bound to the freeze"),
        ("model_bytes", "model bytes disagree with freeze"),
        ("marker_bytes", "marker bytes disagree with locked receipt"),
        ("teacher_overlap", "population/teacher seed overlap"),
        ("prior_overlap", "population/prior schedule seed overlap"),
    ),
)
def test_embedded_launch_preflight_fails_closed_on_lifecycle_or_seed_overlap(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    completed = _run_embedded_launch_preflight(
        _launch_preflight_fixture(tmp_path, mutation=mutation)
    )
    assert completed.returncode != 0
    assert expected_error in completed.stderr


@pytest.mark.skipif(_powershell() is None, reason="PowerShell is unavailable")
def test_resume_manifest_requires_complete_non_null_identity_chain(
    tmp_path: Path,
) -> None:
    shell = _powershell()
    assert shell is not None
    valid = _run_resume_dry_run(shell, *_resume_fixture(tmp_path / "valid"))
    assert valid.returncode == 0, valid.stdout + valid.stderr
    assert json.loads(valid.stdout)["mode"] == "dry_run"

    missing = _run_resume_dry_run(
        shell, *_resume_fixture(tmp_path / "missing", omit_model_identity=True)
    )
    assert missing.returncode != 0
    assert "runtime.model_sha256 must be a lowercase SHA-256 digest" in (
        missing.stdout + missing.stderr
    )


@pytest.mark.skipif(_powershell() is None, reason="PowerShell is unavailable")
def test_status_accepts_one_canonical_unique_done_row(tmp_path: Path) -> None:
    shell = _powershell()
    assert shell is not None
    completed = _run_status(shell, *_status_fixture(tmp_path))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["state"] == "complete"
    assert report["shards_complete"] == 1
    assert report["completed_indices"] == [0]


@pytest.mark.skipif(_powershell() is None, reason="PowerShell is unavailable")
@pytest.mark.parametrize(
    ("fixture_kwargs", "expected_error"),
    (
        ({"row_shard": 1}, "Invalid population DONE object"),
        ({"row_shard": "0"}, "DONE.shard must be a JSON integer"),
        ({"uri_shard": 1}, "Population DONE URI shard is outside frozen range"),
        ({"duplicate_uri": True}, "Duplicate population DONE shard"),
        ({"wrong_prefix": True}, "Population DONE URI is outside the frozen run prefix"),
    ),
)
def test_status_rejects_mismatched_noninteger_out_of_range_or_duplicate_done(
    tmp_path: Path, fixture_kwargs: dict[str, object], expected_error: str
) -> None:
    shell = _powershell()
    assert shell is not None
    completed = _run_status(shell, *_status_fixture(tmp_path, **fixture_kwargs))
    assert completed.returncode != 0
    assert expected_error in completed.stdout + completed.stderr


@pytest.mark.skipif(_powershell() is None, reason="PowerShell is unavailable")
def test_population_spot_scripts_parse_in_powershell() -> None:
    shell = _powershell()
    assert shell is not None
    paths = ",".join(f"'{path}'" for path in (START, STATUS, RECEIVE))
    command = (
        f"$bad=@();foreach($f in @({paths})){{"
        "$t=$null;$e=$null;"
        "[System.Management.Automation.Language.Parser]::ParseFile($f,[ref]$t,[ref]$e)|Out-Null;"
        "if($e.Count){$bad+=$e}};"
        "if($bad.Count){$bad|ForEach-Object{$_.Message};exit 1}"
    )
    completed = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
