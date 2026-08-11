from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "scripts" / "HuM43Attempt04Spot.Common.ps1"
START = ROOT / "scripts" / "Start-GcpHuM43Attempt04TeacherRun.ps1"
STATUS = ROOT / "scripts" / "Get-GcpHuM43Attempt04TeacherRunStatus.ps1"
RECEIVE = ROOT / "scripts" / "Receive-GcpHuM43Attempt04TeacherRun.ps1"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _powershell() -> str | None:
    # Windows PowerShell is the production compatibility floor.
    return shutil.which("powershell") or shutil.which("pwsh")


def _function(text: str, name: str, next_name: str) -> str:
    start = text.index(f"function {name}")
    end = text.index(f"function {next_name}", start)
    return text[start:end]


def test_attempt04_teacher_scripts_keep_current_profile_untouched() -> None:
    assert hashlib.sha256(
        (ROOT / "src" / "ofc_regular" / "ai_profiles.py").read_bytes()
    ).hexdigest() == "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
    for path in (COMMON, START, STATUS, RECEIVE):
        text = _read(path)
        assert "ai_profiles.py" not in text
        assert "current_profile_mutated = $false" in text or path == COMMON


def test_start_freezes_exact_four_role_seed_domains_and_70_shards() -> None:
    text = _read(START)
    expected = {
        "precal_holdout": (
            300,
            13106071901,
            20106071901,
            21106071901,
            22106071901,
        ),
        "calibration_safety_fit": (
            100,
            13506071901,
            20506071901,
            21506071901,
            22506071901,
        ),
        "calibration_threshold_lock": (
            100,
            13606071901,
            20606071901,
            21606071901,
            22606071901,
        ),
        "locked_holdout": (
            200,
            13806071901,
            20806071901,
            21806071901,
            22806071901,
        ),
    }
    for role, values in expected.items():
        assert text.count(f"'{role}'") >= 6
        for value in values:
            assert str(value) in text
    for token in (
        "$SeedStride = 1000003L",
        "$RootsPerShard = 10",
        "$TotalRoots = 700",
        "$TotalShards = 70",
        "candidate_samples = 2",
        "evaluation_samples = 128",
        "logical_role = $role",
        "split = [string]$roleSpec.record_split",
        "Attempt04 schedule is not exact 70 shards",
    ):
        assert token in text
    assert "'train.precal_holdout'" not in text
    assert "'calibration.safety_fit'" not in text
    assert "'calibration.threshold_lock'" not in text


def test_start_preflight_binds_both_freezes_and_archive_rechecks_schedule() -> None:
    text = _read(START)
    for token in (
        "'--model-freeze', $ModelFreezePath",
        "'--training-freeze', $TrainingFreezePath",
        "$payload.model_freeze_file_sha256",
        "$payload.training_freeze_file_sha256",
        "Attempt04 teacher preflight lost its plan/model/training freeze binding",
        "Attempt04 packaged schedule does not match the frozen schedule",
        'names.count("shards_manifest.jsonl") != 1',
        'z.read("shards_manifest.jsonl") != schedule',
        "stale or duplicate schedule",
    ):
        assert token in text
    copy_template = text.index(
        "Copy-Item -LiteralPath $templatePackage -Destination $packageRoot -Recurse"
    )
    replace_schedule = text.index(
        "Copy-Item -LiteralPath $schedulePath -Destination $packagedSchedulePath -Force"
    )
    build_archive = text.index("$zipBuilder = @'")
    assert copy_template < replace_schedule < build_archive


def test_start_rewrites_inherited_worker_to_exact_result_object_access() -> None:
    text = _read(START)
    for token in (
        'gcloud storage objects describe \"$DONE_URI\"',
        'gcloud storage objects describe \"$RESUME_URI/checkpoint.json\"',
        'gcloud storage cp \"$SYNC/teacher.jsonl.partial\"',
        'gcloud storage cp \"$SYNC/checkpoint.json\"',
        'gcloud storage cp \"$SYNC/heartbeat.json\"',
        "Attempt04 startup retained wildcard or list-based result access",
    ):
        assert token in text
    assert "$startupText.Contains('gcloud storage ls ')" in text
    assert "$startupText.Contains('\"$SYNC/\"*')" in text


def test_start_and_receive_bind_each_role_identity_manifest() -> None:
    start = _read(START)
    for token in (
        "hu_m43_attempt04_data_contract_v1",
        "hu_m43_attempt04_ordered_teacher_roles_v1",
        "teacher_role_manifests/",
        "role_identity_sha256",
        '"$prefix/source/teacher_role_manifests/$role.json"',
        "Attempt04 contract role manifest set changed",
        "Attempt04 role manifest file/hash changed",
    ):
        assert token in start
    receive = _read(RECEIVE)
    assert '"$prefix/source/teacher_role_manifests/$Role.json"' in receive
    assert "'--role-manifest'" in receive
    assert "authorization role identity does not match the run manifest" in receive


def test_start_removes_arbitrary_template_parameter_and_pins_byte_closure() -> None:
    text = _read(START)
    param_block = text[: text.index("$ErrorActionPreference")]
    assert "$TemplateRunDir" not in param_block
    for token in (
        "$PinnedTemplateRunName = 'regular-hu-m43-attempt03-c2e64-pilot700-r2-20260714-0228'",
        "1e9cc09b3968322bb5b2eeb137a9ddc7aa067e9b8f3f8a2efd424daea2441aea",
        "f42977efa91a1b3c543ed2cdc864b9f1712bc96047633b2217c46b80387d2316",
        "e9b9d9748bb2b2f31a4baa0d928b60c05f254ab5d7915a7461fb8402e1e7ddb8",
        "ec295d070ac7bb21a82aeb2f432b8f6f191e61b63027bc446e8bd5f39f51569f",
        "57e90d89d439ccfe67df78bd6787b075d32ba8b827ffef7dae88cc323664d923",
        "e16764d716b2c749900499667fd13b29bf9cd3472e85d62add6eb9703cf798e4",
        "Assert-M43A4PinnedTeacherTemplate -TemplateDirectory $TemplateRunDir",
        "Pinned Attempt03 teacher template byte closure changed",
        "Pinned Attempt03 teacher template semantic closure changed",
    ):
        assert token in text


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_template_tree_digest_changes_after_one_byte_mutation(tmp_path: Path) -> None:
    shell = _powershell()
    assert shell is not None
    tree = tmp_path / "package_src"
    (tree / "src").mkdir(parents=True)
    (tree / "src" / "worker.py").write_bytes(b"abc")
    (tree / "manifest.json").write_bytes(b"{}")
    start_text = _read(START)
    tree_function = _function(
        start_text,
        "Get-M43A4PackageTreeSha256",
        "Assert-M43A4PinnedTeacherTemplate",
    )
    common = str(COMMON).replace("'", "''")
    tree_value = str(tree).replace("'", "''")

    def digest() -> str:
        command = (
            f". '{common}';"
            + tree_function
            + f"Get-M43A4PackageTreeSha256 -Root '{tree_value}'"
        )
        result = subprocess.run(
            [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
            cwd=ROOT,
            text=True,
            encoding="utf-8-sig",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        value = result.stdout.strip()
        assert len(value) == 64
        return value

    before = digest()
    (tree / "src" / "worker.py").write_bytes(b"abd")
    after = digest()
    assert before != after


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_alternate_template_directory_parameter_is_rejected(tmp_path: Path) -> None:
    shell = _powershell()
    assert shell is not None
    result = subprocess.run(
        [
            shell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(START),
            "-ModelFreezePath",
            str(tmp_path / "model.json"),
            "-TrainingFreezePath",
            str(tmp_path / "training.json"),
            "-TemplateRunDir",
            str(tmp_path / "alternate"),
            "-PackageOnly",
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "TemplateRunDir" in result.stderr


def test_start_resume_compares_remote_manifest_bytes_not_stdout_text() -> None:
    text = _read(START)
    start = text.index("if ($ResumeExisting) {")
    resume = text[start : text.index("else {", start)]
    assert "Copy-M43A4RemoteFileExact" in resume
    assert "-ExpectedSha256 (Get-M43A4Sha256 $manifestPath)" in resume
    assert "storage', 'cat'" not in resume


def test_status_addresses_only_selected_role_exact_done_objects() -> None:
    text = _read(STATUS)
    assert "[Parameter(Mandatory = $true)]" in text
    assert "[ValidateSet(" in text
    filter_index = text.index("$roleSpecs = @(")
    result_index = text.index('$doneUri = "$prefix/results/')
    assert filter_index < result_index
    result_tail = text[result_index:]
    assert "foreach ($spec in $roleSpecs)" in text[filter_index:result_index]
    assert "$allSpecs" not in result_tail
    assert "teacher.jsonl" not in text
    assert "storage', 'ls'" not in text
    assert "/results/**" not in text
    assert "other_role_result_objects_addressed = $false" in text


def test_receive_claims_one_role_before_any_result_object_access() -> None:
    text = _read(RECEIVE)
    assert "[Parameter(Mandatory = $true)][string]$RoleOpenAuthorizationPath" in text
    assert "[Parameter(Mandatory = $true)][string]$RoleConsumptionMarkerPath" in text
    assert "RoleConsumptionMarkerPath must be canonical" in text
    assert "use ResumeClaim for this exact role only" in text
    assert "ResumeClaim requires the exact existing canonical role claim" in text
    assert "hu_m43_attempt04_role_consumption_v1" in text
    assert "consumed_before_any_role_stat_hash_download_or_read" in text
    role_filter = text.index("$roleSpecs = @(")
    claim = text.index("Write-M43A4Utf8CreateNew", role_filter)
    result = text.index('$roleResultPrefix = "$prefix/results/', claim)
    assert role_filter < claim < result
    assert "/results/" not in text[:claim]
    result_tail = text[result:]
    assert "foreach ($spec in $roleSpecs)" in text[claim:result]
    assert "$allSpecs" not in result_tail
    assert "storage', 'ls'" not in text
    assert "/results/**" not in text


def test_receive_uses_six_exact_objects_per_shard_and_max_eight_parallel() -> None:
    text = _read(RECEIVE)
    for name in (
        "teacher.jsonl",
        "DONE.json",
        "checkpoint.json",
        "heartbeat.json",
        "generator_summary.json",
        "run.log",
    ):
        assert f'"$roleResultPrefix/{name}"' in text
    assert "[ValidateRange(1, 8)][int]$MaxParallel = 8" in text
    assert "Invoke-M43A4ParallelExactGcsCopies" in text
    assert "-MaxParallel $MaxParallel" in text
    common = _read(COMMON)
    assert "Assert-M43A4ExactGcsUri" in common
    assert "@('storage', 'cp') +" in common
    assert "[ValidateRange(1, 8)][int]$MaxParallel = 8" in common


def test_receive_publishes_without_clobber_then_reaudits_final_paths() -> None:
    text = _read(RECEIVE)
    sync = text.index("Sync-M43A4TreeBestEffort -Path $stagingRoot")
    publish = text.index(
        "Publish-M43A4DirectoryAtomic -Source $stagingRoot -Destination $OutputDir"
    )
    final_sync = text.index("Sync-M43A4TreeBestEffort -Path $OutputDir")
    finalize = text.index("Invoke-M43A4PostPublishFinalization", final_sync)
    success = text.index("hu_m43_attempt04_teacher_role_receive_result_v1", finalize)
    assert sync < publish < final_sync < finalize < success
    wrapper = _function(
        text,
        "Invoke-M43A4PostPublishFinalization",
        "Assert-M43A4TeacherManifest",
    ) if text.index("function Invoke-M43A4PostPublishFinalization") < text.index("function Assert-M43A4TeacherManifest") else text[
        text.index("function Invoke-M43A4PostPublishFinalization") : text.index("$RepoRoot =")
    ]
    assert "TEST_ONLY_ATTEMPT04_FAILURE_AFTER_PUBLISH" in wrapper
    assert "[IO.Directory]::Move($sourceFull, $destinationFull)" in text
    assert "Atomic publication destination already exists" in text
    assert "Move-Item -LiteralPath $stagingRoot" not in text


def test_existing_output_resume_is_local_only_before_any_gcs_prefix() -> None:
    text = _read(RECEIVE)
    branch = text.index("if ($outputExists) {")
    cloud_prefix = text.index('$prefix = "gs://$Bucket/runs/$RunName"')
    assert branch < cloud_prefix
    recovery = text[branch:cloud_prefix]
    assert "if (-not $ResumeClaim)" in recovery
    assert "Invoke-M43A4PostPublishFinalization" in recovery
    assert "recovered_local_finalization_without_cloud_access" in recovery
    assert "cloud_result_objects_addressed_during_recovery = $false" in recovery
    assert "exit 0" in recovery
    assert "Copy-M43A4RemoteFileExact" not in recovery
    assert "Invoke-M43A4ParallelExactGcsCopies" not in recovery
    helper = _function(
        text,
        "Complete-M43A4PublishedRoleLocal",
        "Read-M43A4ExistingClaim",
    ) if text.index("function Complete-M43A4PublishedRoleLocal") < text.index("function Read-M43A4ExistingClaim") else text[
        text.index("function Complete-M43A4PublishedRoleLocal") : text.index("$RepoRoot =")
    ]
    assert "validate-published-teacher-role" in helper
    assert "canonical published audit differs from fresh local re-audit" in helper
    assert "hu_m43_attempt04_teacher_role_finalization_v1" in helper
    assert "cloud_result_objects_addressed_during_finalization = $false" in helper
    assert "gs://" not in helper
    assert "/results/" not in helper


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
@pytest.mark.parametrize("failure_point", ("after_publish", "after_reaudit"))
def test_same_claim_recovers_locally_after_post_publish_failures(
    tmp_path: Path, failure_point: str
) -> None:
    shell = _powershell()
    assert shell is not None
    output = tmp_path / "published-role"
    run_dir = tmp_path / "run"
    (output / "closure").mkdir(parents=True)
    (output / "merged").mkdir()
    run_dir.mkdir()
    closure_payloads = {
        "manifest.json": b"manifest",
        "shards_manifest.jsonl": b"schedule",
        "teacher_role_manifest.json": b"role-manifest",
        "teacher_contract.json": b"contract",
        "teacher_preflight_receipt.json": b"preflight",
        "teacher_source.zip": b"source",
        "startup_teacher.sh": b"startup",
        "source_model_manifest.json": b"models",
        "source_native_manifest.json": b"native",
    }
    for name, payload in closure_payloads.items():
        (output / "closure" / name).write_bytes(payload)
    (output / "merged" / "teacher.jsonl").write_text("{}\n", encoding="utf-8")
    authorization = tmp_path / "authorization.json"
    claim = tmp_path / "claim.json"
    plan = tmp_path / "plan.json"
    authorization.write_text("{}", encoding="utf-8")
    claim.write_text("{}", encoding="utf-8")
    plan.write_text("{}", encoding="utf-8")
    claim_sha = hashlib.sha256(claim.read_bytes()).hexdigest()
    role_identity = "1" * 64
    receipt = {
        "schema": "hu_m43_attempt04_teacher_role_receive_receipt_v1",
        "status": "verified_role_after_frozen_claim",
        "logical_role": "precal_holdout",
        "role_identity_sha256": role_identity,
        "consumption_marker_file_sha256": claim_sha,
        "receipt_sha256": "2" * 64,
    }
    (output / "merged" / "receive_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    hashes = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in closure_payloads.items()
    }
    receive_text = _read(RECEIVE)
    helpers = receive_text[
        receive_text.index("function Complete-M43A4PublishedRoleLocal") :
        receive_text.index("$RepoRoot =")
    ]

    def ps(value: Path | str) -> str:
        return str(value).replace("'", "''")

    manifest = {
        "teacher_contract_file_sha256": hashes["teacher_contract.json"],
        "preflight_file_sha256": hashes["teacher_preflight_receipt.json"],
        "source_sha256": hashes["teacher_source.zip"],
        "startup_sha256": hashes["startup_teacher.sh"],
        "model_manifest_sha256": hashes["source_model_manifest.json"],
        "native_manifest_sha256": hashes["source_native_manifest.json"],
    }
    manifest_json = json.dumps(manifest, separators=(",", ":")).replace("'", "''")
    common = ps(COMMON)
    command = f"""
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
. '{common}'
$Role='precal_holdout'
$RunName='regular-hu-m43-attempt04-test'
$RepoRoot='{ps(tmp_path)}'
$RunDir='{ps(run_dir)}'
$PlanPath='{ps(plan)}'
$RoleOpenAuthorizationPath='{ps(authorization)}'
$RoleConsumptionMarkerPath='{ps(claim)}'
$RoleIdentitySha256='{role_identity}'
$AuthorizationFileSha256='{hashlib.sha256(authorization.read_bytes()).hexdigest()}'
$PlanFileSha256='{hashlib.sha256(plan.read_bytes()).hexdigest()}'
$ManifestFileSha256='{hashes['manifest.json']}'
$ScheduleFileSha256='{hashes['shards_manifest.jsonl']}'
$Manifest=('{manifest_json}'|ConvertFrom-Json)
$SelectedRoleManifest=[pscustomobject]@{{file_sha256='{hashes['teacher_role_manifest.json']}'}}
function Invoke-M43A4TrainingCli {{
    param([string[]]$Arguments,[string]$Label,[int]$TimeoutSeconds=300)
    $index=[Array]::IndexOf($Arguments,'--output')
    if($index -lt 0){{throw 'mock output missing'}}
    $audit=[ordered]@{{
        schema='hu_m43_attempt04_published_teacher_role_audit_v1'
        status='pass'
        logical_role=$Role
        role_identity_sha256=$RoleIdentitySha256
        final_path_reaudit=$true
    }}
    Write-M43A4Utf8CreateNew -Path $Arguments[$index+1] -Text (($audit|ConvertTo-Json -Depth 6)+"`n")
}}
function Copy-M43A4RemoteFileExact {{ throw 'GCS MUST NOT BE CALLED' }}
function Invoke-M43A4ParallelExactGcsCopies {{ throw 'GCS MUST NOT BE CALLED' }}
{helpers}
$TestOnlyFailureInjection='{failure_point}'
$failed=$false
try {{
    [void](Invoke-M43A4PostPublishFinalization -OutputDirectory '{ps(output)}' -ClaimFileSha256 '{claim_sha}' -RecoveredExistingOutput $false)
}}
catch {{
    if($_.Exception.Message -notmatch 'TEST_ONLY_ATTEMPT04_FAILURE'){{throw}}
    $failed=$true
}}
if(-not $failed){{throw 'injected failure did not fire'}}
$auditPath='{ps(str(output) + '.published_role_audit.json')}'
$finalPath='{ps(str(output) + '.finalized.json')}'
if('{failure_point}' -eq 'after_publish' -and (Test-Path -LiteralPath $auditPath)){{throw 'audit created before after_publish failure'}}
if('{failure_point}' -eq 'after_reaudit' -and -not (Test-Path -LiteralPath $auditPath -PathType Leaf)){{throw 'audit missing after after_reaudit failure'}}
if(Test-Path -LiteralPath $finalPath){{throw 'failure unexpectedly finalized'}}
$TestOnlyFailureInjection='none'
$result=Invoke-M43A4PostPublishFinalization -OutputDirectory '{ps(output)}' -ClaimFileSha256 '{claim_sha}' -RecoveredExistingOutput $true
if(-not (Test-Path -LiteralPath $result.finalization -PathType Leaf)){{throw 'recovery did not finalize'}}
$result|ConvertTo-Json -Compress
"""
    result = subprocess.run(
        [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["recovered_existing_output"] is True
    assert Path(payload["finalization"]).is_file()


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_resume_after_claim_accepts_only_the_exact_frozen_role(tmp_path: Path) -> None:
    shell = _powershell()
    assert shell is not None
    text = _read(RECEIVE)
    function = _function(
        text,
        "Read-M43A4ExistingClaim",
        "Assert-M43A4TeacherManifest",
    )
    digests = {
        "role": "1" * 64,
        "authorization": "2" * 64,
        "plan": "3" * 64,
        "manifest": "4" * 64,
        "schedule": "5" * 64,
    }
    claim_path = tmp_path / "M43_ATTEMPT04_ROLE_CONSUMED.json"
    claim_path.write_text(
        json.dumps(
            {
                "schema": "hu_m43_attempt04_role_consumption_v1",
                "status": "consumed_before_any_role_stat_hash_download_or_read",
                "logical_role": "precal_holdout",
                "role_identity_sha256": digests["role"],
                "run_name": "regular-hu-m43-attempt04-test",
                "authorization_file_sha256": digests["authorization"],
                "plan_file_sha256": digests["plan"],
                "manifest_file_sha256": digests["manifest"],
                "schedule_file_sha256": digests["schedule"],
                "result_objects_addressed_when_claimed": False,
                "other_role_result_objects_addressed": False,
                "current_profile_mutated": False,
                "runtime_policy_activated": False,
            }
        ),
        encoding="utf-8",
    )
    path_value = str(claim_path).replace("'", "''")
    command = (
        "$ErrorActionPreference='Stop';Set-StrictMode -Version Latest;"
        "$Role='precal_holdout';$RunName='regular-hu-m43-attempt04-test';"
        + function
        + f"$claim=Read-M43A4ExistingClaim -Path '{path_value}' "
        + f"-RoleIdentitySha256 '{digests['role']}' "
        + f"-AuthorizationFileSha256 '{digests['authorization']}' "
        + f"-PlanFileSha256 '{digests['plan']}' "
        + f"-ManifestFileSha256 '{digests['manifest']}' "
        + f"-ScheduleFileSha256 '{digests['schedule']}';"
        + "$claim|ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["logical_role"] == "precal_holdout"

    wrong_role = command.replace(
        "$Role='precal_holdout'", "$Role='calibration_safety_fit'", 1
    )
    rejected = subprocess.run(
        [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", wrong_role],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert rejected.returncode != 0
    assert "existing role claim does not match" in rejected.stderr

    wrong_run = command.replace(
        "$RunName='regular-hu-m43-attempt04-test'",
        "$RunName='regular-hu-m43-attempt04-other'",
        1,
    )
    rejected_run = subprocess.run(
        [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", wrong_run],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert rejected_run.returncode != 0
    assert "existing role claim does not match" in rejected_run.stderr

    wrong_hash = command.replace(
        f"-ManifestFileSha256 '{digests['manifest']}'",
        f"-ManifestFileSha256 '{'9' * 64}'",
        1,
    )
    rejected_hash = subprocess.run(
        [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", wrong_hash],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert rejected_hash.returncode != 0
    assert "existing role claim does not match" in rejected_hash.stderr


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_attempt04_teacher_scripts_parse_in_windows_powershell() -> None:
    shell = _powershell()
    assert shell is not None
    paths = ",".join(
        f"'{str(path).replace(chr(39), chr(39) * 2)}'"
        for path in (COMMON, START, STATUS, RECEIVE)
    )
    command = (
        f"$files=@({paths});"
        "foreach($f in $files){$tokens=$null;$errors=$null;"
        "[void][Management.Automation.Language.Parser]::ParseFile("
        "$f,[ref]$tokens,[ref]$errors);"
        "if(@($errors).Count){throw (($errors|ForEach-Object{$_.ToString()})"
        "-join [Environment]::NewLine)}}"
    )
    result = subprocess.run(
        [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=ROOT,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
def test_atomic_role_publication_has_exactly_one_winner(tmp_path: Path) -> None:
    shell = _powershell()
    assert shell is not None
    text = _read(RECEIVE)
    function = _function(
        text,
        "Publish-M43A4DirectoryAtomic",
        "Assert-M43A4TeacherManifest",
    )
    sources = [tmp_path / "source-a", tmp_path / "source-b"]
    destination = tmp_path / "published"
    for index, source in enumerate(sources):
        source.mkdir()
        (source / "winner.json").write_text(
            json.dumps({"source": index}), encoding="utf-8"
        )
    processes: list[subprocess.Popen[str]] = []
    for source in sources:
        source_value = str(source).replace("'", "''")
        destination_value = str(destination).replace("'", "''")
        command = (
            "$ErrorActionPreference='Stop';Set-StrictMode -Version Latest;"
            + function
            + f"Publish-M43A4DirectoryAtomic -Source '{source_value}' "
            + f"-Destination '{destination_value}'"
        )
        processes.append(
            subprocess.Popen(
                [
                    shell,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    command,
                ],
                cwd=ROOT,
                text=True,
                encoding="utf-8-sig",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    outcomes = [process.communicate(timeout=30) for process in processes]
    codes = [process.returncode for process in processes]
    assert sorted(codes) == [0, 1], outcomes
    assert destination.is_dir()
    published = json.loads((destination / "winner.json").read_text(encoding="utf-8"))
    assert published["source"] in (0, 1)
    remaining = [source for source in sources if source.exists()]
    assert len(remaining) == 1
