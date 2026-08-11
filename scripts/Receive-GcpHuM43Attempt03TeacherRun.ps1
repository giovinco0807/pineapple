param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$PlanPath = "configs/hu_joint_policy_m43_attempt03.json",
    [string]$PreflightReceiptPath,
    [string]$RunDir,
    [string]$DownloadDir,
    [string]$OutputDir,
    [switch]$CanaryOnly,
    [switch]$OpenPrecalibration,
    [string]$ModelFreezePath = "configs/hu_joint_policy_m43_attempt03_model_freeze.json",
    [string]$PrecalOpenAuthorizationPath,
    [string]$FitReceiptPath,
    [string]$PrecalDownloadDir,
    [string]$PrecalOutputDir
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw "RunName is not path-safe"
}

$cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (Test-Path -LiteralPath $cloudSdkGcloud) {
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}
elseif (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud not found in PATH or at $cloudSdkGcloud"
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-LineCount([string]$Path) {
    $reader = [IO.File]::OpenText((Resolve-Path -LiteralPath $Path))
    try {
        $count = 0
        while ($null -ne $reader.ReadLine()) { $count += 1 }
        return [int]$count
    }
    finally { $reader.Dispose() }
}

function Write-Utf8NoBom([string]$Path, [string]$Text) {
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    [IO.File]::WriteAllText($Path, $Text, [Text.UTF8Encoding]::new($false))
}

function Write-Utf8NoBomCreateNew([string]$Path, [string]$Text) {
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Text)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    }
    finally { $stream.Dispose() }
}

function Assert-UnderRoot([string]$Path, [string]$Root, [string]$Name) {
    $full = [IO.Path]::GetFullPath($Path)
    $fullRoot = [IO.Path]::GetFullPath($Root)
    if (-not ($full + [IO.Path]::DirectorySeparatorChar).StartsWith(
            $fullRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase
    )) { throw "$Name must remain under $fullRoot" }
}

function Assert-CanonicalSelfHash([string]$Path, [string]$Field) {
    $script = @'
import hashlib,json,sys
path,field=sys.argv[1:]
obj=json.load(open(path,encoding="utf-8-sig"))
claimed=obj.pop(field,None)
actual=hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()
if claimed != actual:
    raise SystemExit(f"{field} mismatch: {claimed} != {actual}")
'@
    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $result = @($script | & python - $Path $Field 2>&1)
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $saved }
    if ($code -ne 0) {
        throw "Canonical self-hash validation failed for $Path`: $($result -join "`n")"
    }
}

function Add-CanonicalSelfHash([string]$Path, [string]$Field) {
    $script = @'
import hashlib,json,os,sys
path,field=sys.argv[1:]
with open(path,encoding="utf-8-sig") as handle:
    obj=json.load(handle)
if field in obj:
    raise SystemExit(f"{field} already exists")
obj[field]=hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()
temporary=path+".selfhash.tmp"
with open(temporary,"x",encoding="utf-8",newline="\n") as handle:
    json.dump(obj,handle,indent=2,sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary,path)
'@
    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $result = @($script | & python - $Path $Field 2>&1)
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $saved }
    if ($code -ne 0) {
        throw "Canonical self-hash creation failed for $Path`: $($result -join "`n")"
    }
}

function Invoke-GcloudExactCopy([string]$Uri, [string]$Destination) {
    if ($Uri -match '[*?\[\]]') { throw "Wildcard GCS URI is forbidden: $Uri" }
    if (Test-Path -LiteralPath $Destination) {
        throw "Refusing to overwrite downloaded artifact: $Destination"
    }
    $parent = Split-Path -Parent $Destination
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @(& gcloud storage cp $Uri $Destination --project $ProjectId 2>&1)
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $saved }
    if ($code -ne 0) {
        throw "Exact GCS copy failed for $Uri`: $($output -join "`n")"
    }
}

function Invoke-PythonJson([string[]]$Arguments) {
    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @(& python @Arguments 2>&1)
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $saved }
    if ($code -ne 0) { throw "Attempt03 Python validation failed: $($output -join "`n")" }
    try { return (($output -join "`n") | ConvertFrom-Json) }
    catch { throw "Attempt03 Python validator returned invalid JSON: $($output -join "`n")" }
}

function Merge-Shards([object[]]$Records, [string]$Destination) {
    if (Test-Path -LiteralPath $Destination) { throw "Refusing to overwrite merged output: $Destination" }
    $temporary = "$Destination.tmp"
    if (Test-Path -LiteralPath $temporary) { throw "Temporary merge output already exists: $temporary" }
    $writer = [IO.StreamWriter]::new($temporary, $false, [Text.UTF8Encoding]::new($false))
    try {
        foreach ($record in @($Records | Sort-Object shard)) {
            foreach ($line in [IO.File]::ReadLines((Resolve-Path -LiteralPath $record.path))) {
                $writer.WriteLine($line)
            }
        }
    }
    finally { $writer.Dispose() }
    Move-Item -LiteralPath $temporary -Destination $Destination
}

function Receive-FrozenClosure([string]$StageDownloadDir) {
    $sourceDir = Join-Path $StageDownloadDir "source"
    New-Item -ItemType Directory -Path $sourceDir | Out-Null
    $paths = [ordered]@{
        manifest = Join-Path $StageDownloadDir "manifest.json"
        schedule = Join-Path $sourceDir "shards_manifest.jsonl"
        model_manifest = Join-Path $sourceDir "source_model_manifest.json"
        native_manifest = Join-Path $sourceDir "source_native_manifest.json"
        source = Join-Path $sourceDir "ofc_regular_hu_m43_attempt03_teacher_source.zip"
        startup = Join-Path $sourceDir "startup_hu_m43_attempt03_teacher.sh"
    }
    foreach ($pair in @(
        @("$script:prefix/manifest.json", $paths.manifest),
        @("$script:prefix/source/shards_manifest.jsonl", $paths.schedule),
        @("$script:prefix/source/source_model_manifest.json", $paths.model_manifest),
        @("$script:prefix/source/source_native_manifest.json", $paths.native_manifest),
        @("$script:prefix/source/ofc_regular_hu_m43_attempt03_teacher_source.zip", $paths.source),
        @("$script:prefix/source/startup_hu_m43_attempt03_teacher.sh", $paths.startup)
    )) { Invoke-GcloudExactCopy -Uri $pair[0] -Destination $pair[1] }

    $localManifestPath = Join-Path $script:RunDir "manifest.json"
    if ((Get-Sha256 $paths.manifest) -ne (Get-Sha256 $localManifestPath)) {
        throw "Remote Attempt03 manifest bytes do not match the local immutable manifest"
    }
    $closure = Invoke-PythonJson @(
        "-B", "-m", "ofc_regular.validate_hu_m43_attempt03_teacher_receive", "validate-closure",
        "--plan", $script:PlanPath, "--manifest", $paths.manifest,
        "--schedule", $paths.schedule, "--source", $paths.source,
        "--startup", $paths.startup, "--model-manifest", $paths.model_manifest,
        "--native-manifest", $paths.native_manifest, "--run-name", $RunName,
        "--project-id", $ProjectId, "--bucket", $Bucket
    )
    if ($closure.schema -ne "hu_m43_attempt03_teacher_closure_audit_v1" -or $closure.status -ne "pass") {
        throw "Attempt03 immutable closure audit did not pass"
    }
    return [pscustomobject]@{ paths = [pscustomobject]$paths; audit = $closure }
}

function Receive-VerifiedShard([object]$Spec, [string]$StageDownloadDir, [object]$Closure) {
    $resultUri = "$script:prefix/results/$([string]$Spec.output_prefix)"
    $resultPath = Join-Path (Join-Path $StageDownloadDir "results") ([string]$Spec.output_prefix)
    New-Item -ItemType Directory -Path $resultPath | Out-Null
    $files = [ordered]@{
        done = Join-Path $resultPath "DONE.json"
        teacher = Join-Path $resultPath "teacher.jsonl"
        checkpoint = Join-Path $resultPath "checkpoint.json"
        heartbeat = Join-Path $resultPath "heartbeat.json"
        generator_summary = Join-Path $resultPath "generator_summary.json"
        run_log = Join-Path $resultPath "run.log"
    }
    # DONE is fetched first, but no content is trusted until the complete hash
    # chain and all three independently uploaded completion summaries pass.
    Invoke-GcloudExactCopy -Uri "$resultUri/DONE.json" -Destination $files.done
    foreach ($name in @("teacher", "checkpoint", "heartbeat", "generator_summary", "run_log")) {
        $remoteName = if ($name -eq "generator_summary") { "generator_summary.json" } elseif ($name -eq "run_log") { "run.log" } elseif ($name -eq "teacher") { "teacher.jsonl" } else { "$name.json" }
        Invoke-GcloudExactCopy -Uri "$resultUri/$remoteName" -Destination $files[$name]
    }
    $audit = Invoke-PythonJson @(
        "-B", "-m", "ofc_regular.validate_hu_m43_attempt03_teacher_receive", "validate-shard",
        "--plan", $script:PlanPath, "--manifest", $Closure.paths.manifest,
        "--schedule", $Closure.paths.schedule, "--shard", ([string][int]$Spec.shard),
        "--teacher", $files.teacher, "--done", $files.done,
        "--checkpoint", $files.checkpoint, "--heartbeat", $files.heartbeat,
        "--generator-summary", $files.generator_summary, "--run-name", $RunName
    )
    if ($audit.schema -ne "hu_m43_attempt03_teacher_shard_audit_v1" -or
        $audit.status -ne "pass" -or [int]$audit.shard -ne [int]$Spec.shard) {
        throw "Attempt03 shard audit did not pass: $($Spec.shard)"
    }
    $auditPath = Join-Path $resultPath "receive_audit.json"
    Write-Utf8NoBom -Path $auditPath -Text (($audit | ConvertTo-Json -Depth 12) + "`n")
    return [pscustomobject]@{
        shard = [int]$Spec.shard
        logical_split = [string]$Spec.logical_split
        split_shard = [int]$Spec.split_shard
        roots = [int]$Spec.roots
        output_prefix = [string]$Spec.output_prefix
        path = $files.teacher
        sha256 = [string]$audit.output_sha256
        audit = $audit
    }
}

function Assert-PreflightBinding([object]$Manifest) {
    $preflight = Get-Content -LiteralPath $script:PreflightReceiptPath -Raw | ConvertFrom-Json
    if ($preflight.schema -ne "hu_m43_attempt03_preflight_receipt_v1" -or
        $preflight.status -ne "pass_frozen_before_fresh_generation") {
        throw "Attempt03 preflight schema/status mismatch"
    }
    Assert-CanonicalSelfHash -Path $script:PreflightReceiptPath -Field "receipt_sha256"
    if ((Get-Sha256 $script:PreflightReceiptPath) -ne [string]$Manifest.preflight_file_sha256 -or
        (Get-Sha256 $script:PlanPath) -ne [string]$Manifest.plan_file_sha256) {
        throw "Attempt03 local plan/preflight does not match the cloud run binding"
    }
    return $preflight
}

function Assert-ModelFreezeBinding([string]$Path, [string]$ManifestPath, [object]$Preflight) {
    $expectedFreezeSha256 = "8aed10b143172c9d3b2a98fca199e4fbe4324f5956406fea51c2d17fcb614b9d"
    if ((Get-Sha256 $Path) -ne $expectedFreezeSha256) {
        throw "Attempt03 executable model-freeze file changed"
    }
    $freeze = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($freeze.schema -ne "hu_m43_attempt03_model_freeze_v1" -or
        $freeze.status -ne "frozen_before_any_attempt03_teacher_row_was_received" -or
        [string]$freeze.teacher_run.run_name -ne $RunName -or
        [string]$freeze.teacher_run.manifest_sha256 -ne (Get-Sha256 $ManifestPath) -or
        [int]$freeze.teacher_run.fresh_teacher_rows_received_when_frozen -ne 0 -or
        [int]$freeze.teacher_run.precalibration_rows_opened_when_frozen -ne 0 -or
        [int]$freeze.teacher_run.sealed_calibration_rows_opened_when_frozen -ne 0 -or
        [int]$freeze.teacher_run.inherited_locked_rows_opened_when_frozen -ne 0) {
        throw "Attempt03 model freeze is not bound before teacher/pre-cal access"
    }
    if ([string]$freeze.parent_plan.file_sha256 -ne (Get-Sha256 $script:PlanPath) -or
        [string]$freeze.parent_plan.preflight_receipt_sha256 -ne [string]$Preflight.receipt_sha256) {
        throw "Attempt03 model freeze plan/preflight binding changed"
    }
    $modulePath = [IO.Path]::GetFullPath((Join-Path $script:repoRoot ([string]$freeze.implementation.module)))
    $stage18Path = [IO.Path]::GetFullPath((Join-Path $script:repoRoot ([string]$freeze.implementation.stage18_model.path)))
    if ((Get-Sha256 $modulePath) -ne [string]$freeze.implementation.file_sha256 -or
        (Get-Sha256 $stage18Path) -ne [string]$freeze.implementation.stage18_model.file_sha256 -or
        [bool]$freeze.implementation.runtime_teacher_inputs -or
        -not [bool]$freeze.implementation.canonical_action_key_tie_break) {
        throw "Attempt03 frozen executable/model binding changed"
    }
    foreach ($property in $freeze.activation_guards.psobject.Properties) {
        if ([bool]$property.Value) { throw "Attempt03 model freeze activation guard is true: $($property.Name)" }
    }
    return [pscustomobject]@{ payload = $freeze; file_sha256 = $expectedFreezeSha256 }
}

function Rebase-FrozenClosureAfterMove(
    [object]$Closure,
    [string]$FinalDownloadDir,
    [System.Collections.IDictionary]$PreMoveClosureHashes
) {
    $rebased = [ordered]@{
        manifest = Join-Path $FinalDownloadDir "manifest.json"
        schedule = Join-Path $FinalDownloadDir "source/shards_manifest.jsonl"
        model_manifest = Join-Path $FinalDownloadDir "source/source_model_manifest.json"
        native_manifest = Join-Path $FinalDownloadDir "source/source_native_manifest.json"
        source = Join-Path $FinalDownloadDir "source/ofc_regular_hu_m43_attempt03_teacher_source.zip"
        startup = Join-Path $FinalDownloadDir "source/startup_hu_m43_attempt03_teacher.sh"
    }
    # Windows PowerShell 5 exposes OrderedDictionary infrastructure properties
    # through .psobject.Properties.  Enumerate the six data keys explicitly.
    foreach ($name in $rebased.Keys) {
        $path = [string]$rebased[$name]
        if (-not (Test-Path -LiteralPath $path) -or
            (Get-Sha256 $path) -ne [string]$PreMoveClosureHashes[$name]) {
            throw "Attempt03 immutable closure changed while rebasing after atomic move: $name"
        }
    }
    return [pscustomobject]@{
        paths = [pscustomobject]$rebased
        audit = $Closure.audit
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PlanPath = if ([IO.Path]::IsPathRooted($PlanPath)) {
    [IO.Path]::GetFullPath($PlanPath)
} else { [IO.Path]::GetFullPath((Join-Path $repoRoot $PlanPath)) }
$PlanPath = (Resolve-Path -LiteralPath $PlanPath).Path
$planPayload = Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json
if (-not $RunDir) { $RunDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName" }
$RunDir = (Resolve-Path -LiteralPath $RunDir).Path
if (-not $PreflightReceiptPath) { $PreflightReceiptPath = Join-Path $RunDir "m43_attempt03_preflight_receipt.json" }
$PreflightReceiptPath = (Resolve-Path -LiteralPath $PreflightReceiptPath).Path
$prefix = "gs://$Bucket/runs/$RunName"
$envPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $repoRoot "src"

$fitOutputRoot = Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt03_teacher"
$precalOutputRoot = Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt03_precal"
$gcpRunRoot = Join-Path $repoRoot "outputs/gcp_runs"
New-Item -ItemType Directory -Force -Path $fitOutputRoot, $precalOutputRoot | Out-Null

try {
    if ($CanaryOnly -and $OpenPrecalibration) {
        throw "CanaryOnly cannot be combined with OpenPrecalibration"
    }
    if (-not $OpenPrecalibration) {
        if ($CanaryOnly) {
            if ($OutputDir) { throw "CanaryOnly does not publish an OutputDir" }
            if (-not $DownloadDir) { $DownloadDir = Join-Path $RunDir "rc" }
            $DownloadDir = [IO.Path]::GetFullPath($DownloadDir)
            Assert-UnderRoot -Path $DownloadDir -Root $gcpRunRoot -Name "DownloadDir"
            if (Test-Path -LiteralPath $DownloadDir) {
                throw "Attempt03 canary validation is immutable; DownloadDir already exists"
            }
            $canaryStage = Join-Path (Split-Path -Parent $DownloadDir) (".rc-{0}" -f [guid]::NewGuid().ToString("N").Substring(0, 8))
            New-Item -ItemType Directory -Path $canaryStage | Out-Null
            $Closure = Receive-FrozenClosure -StageDownloadDir $canaryStage
            $manifest = Get-Content -LiteralPath $Closure.paths.manifest -Raw | ConvertFrom-Json
            $preflight = Assert-PreflightBinding -Manifest $manifest
            $specs = @([IO.File]::ReadLines($Closure.paths.schedule) | ForEach-Object { $_ | ConvertFrom-Json })
            $fitSpecs = @($specs | Where-Object logical_split -eq "train.fit" | Sort-Object shard)
            if ($fitSpecs.Count -ne 50 -or (($fitSpecs.shard -join ',') -ne ((0..49) -join ','))) {
                throw "Attempt03 fit schedule must be shards 0-49"
            }
            # This is the only result object addressed in CanaryOnly mode.
            $canaryRecord = Receive-VerifiedShard -Spec $fitSpecs[0] -StageDownloadDir $canaryStage -Closure $Closure
            if ([int]$canaryRecord.shard -ne 0 -or [int]$canaryRecord.roots -ne 10) {
                throw "Attempt03 canary shard zero did not pass"
            }
            $canaryReceipt = [ordered]@{
                schema = "hu_m43_attempt03_teacher_canary_receive_receipt_v1"
                status = "verified_shard0_only_no_fit_publication"
                run_name = $RunName
                manifest_sha256 = Get-Sha256 $Closure.paths.manifest
                schedule_sha256 = Get-Sha256 $Closure.paths.schedule
                preflight_receipt_sha256 = [string]$preflight.receipt_sha256
                shard = 0
                roots = 10
                output_sha256 = [string]$canaryRecord.sha256
                receive_audit = $canaryRecord.audit
                shards_1_through_69_addressed = $false
                fit_output_published = $false
                precal_result_prefix_listed = $false
                precal_result_downloaded = $false
                precal_result_opened = $false
                current_profile_mutated = $false
                runtime_policy_activated = $false
                validated_at = (Get-Date).ToUniversalTime().ToString("o")
            }
            Write-Utf8NoBom -Path (Join-Path $canaryStage "canary_validation_receipt.json") -Text (($canaryReceipt | ConvertTo-Json -Depth 20) + "`n")
            if (Test-Path -LiteralPath $DownloadDir) {
                throw "Attempt03 canary destination appeared during validation"
            }
            Move-Item -LiteralPath $canaryStage -Destination $DownloadDir
            [pscustomobject]@{
                schema = "hu_m43_attempt03_teacher_canary_receive_result_v1"
                status = "verified_shard0_only_no_fit_publication"
                run_name = $RunName
                shard = 0
                roots = 10
                download_dir = $DownloadDir
                receipt = Join-Path $DownloadDir "canary_validation_receipt.json"
                shards_1_through_69_addressed = $false
                fit_output_published = $false
                current_profile_mutated = $false
            } | ConvertTo-Json -Depth 8
            return
        }
        # Keep result paths below legacy MAX_PATH for gcloud/Python on Windows.
        if (-not $DownloadDir) { $DownloadDir = Join-Path $RunDir "rf" }
        if (-not $OutputDir) { $OutputDir = Join-Path $fitOutputRoot $RunName }
        $DownloadDir = [IO.Path]::GetFullPath($DownloadDir)
        $OutputDir = [IO.Path]::GetFullPath($OutputDir)
        Assert-UnderRoot -Path $DownloadDir -Root $gcpRunRoot -Name "DownloadDir"
        Assert-UnderRoot -Path $OutputDir -Root $fitOutputRoot -Name "OutputDir"
        if ((Test-Path -LiteralPath $DownloadDir) -or (Test-Path -LiteralPath $OutputDir)) {
            throw "Attempt03 fit receive is immutable; DownloadDir/OutputDir already exists"
        }
        $downloadStage = Join-Path (Split-Path -Parent $DownloadDir) (".rf-{0}" -f [guid]::NewGuid().ToString("N").Substring(0, 8))
        $outputStage = Join-Path $fitOutputRoot (".recv-{0}" -f [guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Path $downloadStage, $outputStage | Out-Null
        $Closure = Receive-FrozenClosure -StageDownloadDir $downloadStage
        $manifest = Get-Content -LiteralPath $Closure.paths.manifest -Raw | ConvertFrom-Json
        $preflight = Assert-PreflightBinding -Manifest $manifest
        $specs = @([IO.File]::ReadLines($Closure.paths.schedule) | ForEach-Object { $_ | ConvertFrom-Json })
        $fitSpecs = @($specs | Where-Object logical_split -eq "train.fit" | Sort-Object shard)
        if ($fitSpecs.Count -ne 50 -or (($fitSpecs.shard -join ',') -ne ((0..49) -join ','))) {
            throw "Attempt03 fit schedule must be shards 0-49"
        }
        # Canary shard zero is fully hash-, generator-, row-, and schedule-
        # validated before any other fit result object is addressed.
        $fitRecords = @((Receive-VerifiedShard -Spec $fitSpecs[0] -StageDownloadDir $downloadStage -Closure $Closure))
        if ([int]$fitRecords[0].shard -ne 0) { throw "Attempt03 canary shard zero did not pass" }
        foreach ($spec in @($fitSpecs | Select-Object -Skip 1)) {
            $fitRecords += Receive-VerifiedShard -Spec $spec -StageDownloadDir $downloadStage -Closure $Closure
        }
        if ($fitRecords.Count -ne 50 -or [int](($fitRecords | Measure-Object roots -Sum).Sum) -ne 500) {
            throw "Attempt03 fit receive did not verify exactly 50 shards / 500 roots"
        }
        $mergedStagePath = Join-Path $outputStage "fresh_train_fit.jsonl"
        Merge-Shards -Records $fitRecords -Destination $mergedStagePath
        if ((Get-LineCount $mergedStagePath) -ne 500) { throw "Attempt03 merged fresh fit is not 500 lines" }

        $finalRecords = @($fitRecords | ForEach-Object {
            [ordered]@{
                shard = [int]$_.shard; logical_split = "train.fit"; split_shard = [int]$_.split_shard
                rows = [int]$_.roots; sha256 = [string]$_.sha256
                path = Join-Path (Join-Path $DownloadDir "results") (Join-Path ([string]$_.output_prefix) "teacher.jsonl")
            }
        })
        $attempt02Train = [IO.Path]::GetFullPath((Join-Path $repoRoot ([string]$planPayload.attempt02_provenance.train_fit.path)))
        $fitReceipt = [ordered]@{
            schema = "hu_m43_attempt03_teacher_fit_receive_receipt_v1"
            status = "verified_fresh_train_fit_only_precal_unopened"
            run_name = $RunName
            manifest_sha256 = Get-Sha256 $Closure.paths.manifest
            remote_manifest_equals_local = $true
            schedule_sha256 = Get-Sha256 $Closure.paths.schedule
            source_sha256 = Get-Sha256 $Closure.paths.source
            startup_sha256 = Get-Sha256 $Closure.paths.startup
            plan_sha256 = Get-Sha256 $PlanPath
            preflight_receipt_sha256 = [string]$preflight.receipt_sha256
            canary_shard = 0; canary_validated_before_fan_in = $true
            verified_shards = 50; verified_roots = 500
            fresh_train_fit = [ordered]@{
                path = Join-Path $OutputDir "fresh_train_fit.jsonl"
                rows = 500; sha256 = Get-Sha256 $mergedStagePath; shards = $finalRecords
            }
            downstream_inputs = [ordered]@{
                schema = "hu_m43_attempt03_fit_downstream_inputs_v1"
                inherited_attempt02_train = $attempt02Train
                fresh_train_fit = Join-Path $OutputDir "fresh_train_fit.jsonl"
                fit_source_paths = @($attempt02Train, (Join-Path $OutputDir "fresh_train_fit.jsonl"))
                original_fresh_shard_paths = @($finalRecords | ForEach-Object { $_.path })
                combined_fit_roots = 700
            }
            download_dir = $DownloadDir
            precal_result_prefix_listed = $false
            precal_result_downloaded = $false
            precal_result_opened = $false
            attempt03_contract_finalized = $false
            sealed_calibration_opened = $false
            inherited_locked_opened = $false
            current_profile_mutated = $false
            runtime_policy_activated = $false
            received_at = (Get-Date).ToUniversalTime().ToString("o")
        }
        Write-Utf8NoBom -Path (Join-Path $outputStage "fit_receipt.json") -Text (($fitReceipt | ConvertTo-Json -Depth 20) + "`n")
        Write-Utf8NoBom -Path (Join-Path $outputStage "downstream_inputs.json") -Text (($fitReceipt.downstream_inputs | ConvertTo-Json -Depth 20) + "`n")
        if ((Test-Path -LiteralPath $DownloadDir) -or (Test-Path -LiteralPath $OutputDir)) {
            throw "Attempt03 fit destination appeared during receive"
        }
        Move-Item -LiteralPath $downloadStage -Destination $DownloadDir
        Move-Item -LiteralPath $outputStage -Destination $OutputDir
        [pscustomobject]@{
            schema = "hu_m43_attempt03_teacher_fit_receive_result_v1"
            status = "verified_fresh_train_fit_only_precal_unopened"
            run_name = $RunName; verified_shards = 50; verified_roots = 500
            output_dir = $OutputDir; download_dir = $DownloadDir
            fit_receipt = Join-Path $OutputDir "fit_receipt.json"
            fresh_train_fit = Join-Path $OutputDir "fresh_train_fit.jsonl"
            precal_result_opened = $false; current_profile_mutated = $false
        } | ConvertTo-Json -Depth 8
        return
    }

    # The branch below is the only code path allowed to address result shards
    # 50-69.  It requires the immutable fit receipt and executable model freeze.
    if (-not $OutputDir) { $OutputDir = Join-Path $fitOutputRoot $RunName }
    if (-not $FitReceiptPath) { $FitReceiptPath = Join-Path $OutputDir "fit_receipt.json" }
    if (-not $PrecalOpenAuthorizationPath) {
        throw "OpenPrecalibration requires PrecalOpenAuthorizationPath"
    }
    if (-not $PrecalDownloadDir) { $PrecalDownloadDir = Join-Path $RunDir "rp" }
    if (-not $PrecalOutputDir) { $PrecalOutputDir = Join-Path $precalOutputRoot $RunName }
    $FitReceiptPath = (Resolve-Path -LiteralPath $FitReceiptPath).Path
    $PrecalOpenAuthorizationPath = (Resolve-Path -LiteralPath $PrecalOpenAuthorizationPath).Path
    $ModelFreezePath = if ([IO.Path]::IsPathRooted($ModelFreezePath)) {
        [IO.Path]::GetFullPath($ModelFreezePath)
    } else { [IO.Path]::GetFullPath((Join-Path $repoRoot $ModelFreezePath)) }
    $ModelFreezePath = (Resolve-Path -LiteralPath $ModelFreezePath).Path
    $PrecalDownloadDir = [IO.Path]::GetFullPath($PrecalDownloadDir)
    $PrecalOutputDir = [IO.Path]::GetFullPath($PrecalOutputDir)
    Assert-UnderRoot -Path $PrecalDownloadDir -Root $gcpRunRoot -Name "PrecalDownloadDir"
    Assert-UnderRoot -Path $PrecalOutputDir -Root $precalOutputRoot -Name "PrecalOutputDir"
    if ((Test-Path -LiteralPath $PrecalDownloadDir) -or (Test-Path -LiteralPath $PrecalOutputDir)) {
        throw "Attempt03 pre-cal receive is immutable; destination already exists"
    }
    $fitReceipt = Get-Content -LiteralPath $FitReceiptPath -Raw | ConvertFrom-Json
    if ($fitReceipt.schema -ne "hu_m43_attempt03_teacher_fit_receive_receipt_v1" -or
        $fitReceipt.status -ne "verified_fresh_train_fit_only_precal_unopened" -or
        [string]$fitReceipt.run_name -ne $RunName -or
        [int]$fitReceipt.verified_shards -ne 50 -or [int]$fitReceipt.verified_roots -ne 500 -or
        [bool]$fitReceipt.precal_result_opened -or [bool]$fitReceipt.current_profile_mutated) {
        throw "Attempt03 fit receipt is not eligible for one-shot pre-cal"
    }
    if ((Get-LineCount ([string]$fitReceipt.fresh_train_fit.path)) -ne 500 -or
        (Get-Sha256 ([string]$fitReceipt.fresh_train_fit.path)) -ne [string]$fitReceipt.fresh_train_fit.sha256) {
        throw "Attempt03 fit receipt merged artifact changed"
    }
    $fitRecords = @($fitReceipt.fresh_train_fit.shards | Sort-Object shard)
    if ($fitRecords.Count -ne 50) { throw "Attempt03 fit receipt lost original shard bindings" }
    foreach ($record in $fitRecords) {
        if ((Get-LineCount ([string]$record.path)) -ne 10 -or
            (Get-Sha256 ([string]$record.path)) -ne [string]$record.sha256) {
            throw "Attempt03 fit shard changed after fit-only receive: $($record.shard)"
        }
    }

    $precalDownloadStage = Join-Path (Split-Path -Parent $PrecalDownloadDir) (".rp-{0}" -f [guid]::NewGuid().ToString("N").Substring(0, 8))
    $precalOutputStage = Join-Path $precalOutputRoot (".recv-{0}" -f [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $precalDownloadStage, $precalOutputStage | Out-Null
    $Closure = Receive-FrozenClosure -StageDownloadDir $precalDownloadStage
    $manifest = Get-Content -LiteralPath $Closure.paths.manifest -Raw | ConvertFrom-Json
    $preflight = Assert-PreflightBinding -Manifest $manifest
    if ([string]$fitReceipt.manifest_sha256 -ne (Get-Sha256 $Closure.paths.manifest) -or
        [string]$fitReceipt.schedule_sha256 -ne (Get-Sha256 $Closure.paths.schedule)) {
        throw "Attempt03 fit/pre-cal immutable closure mismatch"
    }
    $freezeBinding = Assert-ModelFreezeBinding -Path $ModelFreezePath -ManifestPath $Closure.paths.manifest -Preflight $preflight
    $originalFreezePath = Join-Path $repoRoot "configs/hu_joint_policy_m43_attempt03_model_freeze_original_ce47.json"
    $freezeLineagePath = Join-Path $repoRoot "configs/hu_joint_policy_m43_attempt03_model_freeze_lineage.json"
    $authorizationBinding = Invoke-PythonJson @(
        "-B", "-m", "ofc_regular.validate_hu_m43_attempt03_teacher_receive", "validate-precal-open",
        "--repo-root", $repoRoot,
        "--authorization", $PrecalOpenAuthorizationPath,
        "--model-freeze", $ModelFreezePath,
        "--original-freeze", $originalFreezePath,
        "--freeze-lineage", $freezeLineagePath
    )
    if ($authorizationBinding.schema -ne "hu_m43_attempt03_precal_open_authorization_audit_v1" -or
        $authorizationBinding.status -ne "pass_without_precalibration_access" -or
        [bool]$authorizationBinding.precalibration_path_received -or
        [bool]$authorizationBinding.precalibration_content_read) {
        throw "Attempt03 pre-cal open authorization audit did not pass"
    }

    # CreateNew makes this an irreversible, model-freeze- and fitted-candidate-
    # bound open/download claim.  The amended/original freeze lineage and every
    # authorization artifact were re-hashed above.  No pre-cal result URI
    # occurs before this boundary.
    $openClaimRoot = Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt03_precal_open_claims/$RunName"
    $openClaimPath = Join-Path $openClaimRoot "M43_ATTEMPT03_PRECAL_OPEN_CLAIM.json"
    $openClaim = [ordered]@{
        schema = "hu_m43_attempt03_precal_open_claim_v1"
        status = "claimed_before_any_precal_result_download_or_parse"
        run_name = $RunName
        model_freeze_path = $ModelFreezePath
        model_freeze_file_sha256 = [string]$freezeBinding.file_sha256
        original_model_freeze_file_sha256 = [string]$authorizationBinding.freeze_lineage.original_freeze_file_sha256
        model_freeze_lineage_file_sha256 = [string]$authorizationBinding.freeze_lineage.freeze_lineage_file_sha256
        precal_open_authorization_path = $PrecalOpenAuthorizationPath
        precal_open_authorization_file_sha256 = [string]$authorizationBinding.authorization_file_sha256
        precal_open_authorization_sha256 = [string]$authorizationBinding.authorization_sha256
        candidate_model_sha256 = [string]$authorizationBinding.candidate_model_sha256
        fit_bundle_sha256 = [string]$authorizationBinding.fit_bundle_sha256
        fit_manifest_file_sha256 = [string]$authorizationBinding.fit_manifest_file_sha256
        fold_cloud_contract_file_sha256 = [string]$authorizationBinding.fold_cloud_contract_file_sha256
        training_freeze_file_sha256 = [string]$authorizationBinding.training_freeze_file_sha256
        teacher_manifest_sha256 = Get-Sha256 $Closure.paths.manifest
        schedule_sha256 = Get-Sha256 $Closure.paths.schedule
        fit_receipt_file_sha256 = Get-Sha256 $FitReceiptPath
        intended_shards = @(50..69)
        global_model_evaluation_consumption_marker_created = $false
        current_profile_mutated = $false
        runtime_policy_activated = $false
        claimed_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    Write-Utf8NoBomCreateNew -Path $openClaimPath -Text (($openClaim | ConvertTo-Json -Depth 12) + "`n")
    $openClaimSha256 = Get-Sha256 $openClaimPath

    $specs = @([IO.File]::ReadLines($Closure.paths.schedule) | ForEach-Object { $_ | ConvertFrom-Json })
    $precalSpecs = @($specs | Where-Object logical_split -eq "train.precal_holdout" | Sort-Object shard)
    if ($precalSpecs.Count -ne 20 -or (($precalSpecs.shard -join ',') -ne ((50..69) -join ','))) {
        throw "Attempt03 pre-cal schedule must be shards 50-69"
    }
    $precalRecords = @((Receive-VerifiedShard -Spec $precalSpecs[0] -StageDownloadDir $precalDownloadStage -Closure $Closure))
    foreach ($spec in @($precalSpecs | Select-Object -Skip 1)) {
        $precalRecords += Receive-VerifiedShard -Spec $spec -StageDownloadDir $precalDownloadStage -Closure $Closure
    }
    if ($precalRecords.Count -ne 20 -or [int](($precalRecords | Measure-Object roots -Sum).Sum) -ne 200) {
        throw "Attempt03 pre-cal receive did not verify exactly 20 shards / 200 roots"
    }
    $precalFinalRecords = @($precalRecords | ForEach-Object {
        [pscustomobject]@{
            shard = [int]$_.shard; logical_split = "train.precal_holdout"; split_shard = [int]$_.split_shard
            roots = [int]$_.roots; sha256 = [string]$_.sha256; output_prefix = [string]$_.output_prefix
            path = Join-Path (Join-Path $PrecalDownloadDir "results") (Join-Path ([string]$_.output_prefix) "teacher.jsonl")
        }
    })
    if (Test-Path -LiteralPath $PrecalDownloadDir) { throw "Attempt03 pre-cal download destination appeared" }
    $preMoveClosureHashes = [ordered]@{}
    foreach ($property in $Closure.paths.psobject.Properties) {
        $preMoveClosureHashes[$property.Name] = Get-Sha256 ([string]$property.Value)
    }
    # Finalize against the immutable final paths, never transient staging paths.
    Move-Item -LiteralPath $precalDownloadStage -Destination $PrecalDownloadDir
    $Closure = Rebase-FrozenClosureAfterMove `
        -Closure $Closure `
        -FinalDownloadDir $PrecalDownloadDir `
        -PreMoveClosureHashes $preMoveClosureHashes

    # Contract finalization is intentionally absent from fit-only receive and
    # occurs exactly here, after the frozen one-shot open claim.
    $stageContractPath = Join-Path $precalOutputStage "data_contract.json"
    $finalizeArgs = @(
        "-B", "-m", "ofc_regular.hu_m43_attempt03_contract", "finalize-fresh",
        "--plan", $PlanPath, "--repo-root", $repoRoot,
        "--preflight-receipt", $PreflightReceiptPath
    )
    foreach ($record in $fitRecords) { $finalizeArgs += @("--train-fit", ([string]$record.path)) }
    foreach ($record in @($precalFinalRecords | Sort-Object shard)) { $finalizeArgs += @("--train-precal-holdout", ([string]$record.path)) }
    $finalizeArgs += @("--output", $stageContractPath)
    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $finalizeOutput = @(& python @finalizeArgs 2>&1)
        $finalizeCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $saved }
    if ($finalizeCode -ne 0) { throw "Attempt03 one-shot contract finalize failed: $($finalizeOutput -join "`n")" }
    $dataContract = Get-Content -LiteralPath $stageContractPath -Raw | ConvertFrom-Json
    if ($dataContract.schema -ne "hu_m43_attempt03_data_contract_v1" -or
        $dataContract.status -ne "pass_fresh_fit_and_one_shot_precal_sealed_calibration_locked_unopened" -or
        [int]$dataContract.fit.records -ne 700 -or [int]$dataContract.precal_holdout.records -ne 200 -or
        [int]$dataContract.precal_holdout.model_evaluation_count -ne 0 -or
        [int]$dataContract.sealed_calibration.jsonl_content_parse_count -ne 0 -or
        [int]$dataContract.inherited_locked.jsonl_content_parse_count -ne 0) {
        throw "Attempt03 finalized data contract boundary changed"
    }
    if ((Get-Sha256 $openClaimPath) -ne $openClaimSha256) {
        throw "Attempt03 pre-cal open claim changed after exclusive creation"
    }
    Assert-CanonicalSelfHash -Path $stageContractPath -Field "contract_sha256"
    $mergedPrecalStage = Join-Path $precalOutputStage "fresh_precal_holdout.jsonl"
    Merge-Shards -Records $precalFinalRecords -Destination $mergedPrecalStage
    if ((Get-LineCount $mergedPrecalStage) -ne 200) { throw "Attempt03 merged pre-cal is not 200 lines" }

    $finalPrecalRecords = @($precalFinalRecords | ForEach-Object {
        [ordered]@{
            shard = [int]$_.shard; logical_split = "train.precal_holdout"; split_shard = [int]$_.split_shard
            rows = [int]$_.roots; sha256 = [string]$_.sha256
            path = [string]$_.path
        }
    })
    $oneShotReceipt = [ordered]@{
        schema = "hu_m43_attempt03_teacher_precal_receive_receipt_v1"
        status = "verified_structural_precal_open_after_frozen_claim"
        run_name = $RunName
        manifest_sha256 = Get-Sha256 $Closure.paths.manifest
        schedule_sha256 = Get-Sha256 $Closure.paths.schedule
        model_freeze_path = $ModelFreezePath
        model_freeze_file_sha256 = [string]$freezeBinding.file_sha256
        precal_open_authorization_path = $PrecalOpenAuthorizationPath
        precal_open_authorization_file_sha256 = [string]$authorizationBinding.authorization_file_sha256
        precal_open_authorization_sha256 = [string]$authorizationBinding.authorization_sha256
        candidate_model_sha256 = [string]$authorizationBinding.candidate_model_sha256
        fit_bundle_sha256 = [string]$authorizationBinding.fit_bundle_sha256
        fit_manifest_file_sha256 = [string]$authorizationBinding.fit_manifest_file_sha256
        fold_cloud_contract_file_sha256 = [string]$authorizationBinding.fold_cloud_contract_file_sha256
        training_freeze_file_sha256 = [string]$authorizationBinding.training_freeze_file_sha256
        open_claim_path = $openClaimPath
        open_claim_file_sha256 = $openClaimSha256
        verified_shards = 20; verified_roots = 200
        fresh_precal_holdout = [ordered]@{
            path = Join-Path $PrecalOutputDir "fresh_precal_holdout.jsonl"
            rows = 200; sha256 = Get-Sha256 $mergedPrecalStage; shards = $finalPrecalRecords
        }
        data_contract = Join-Path $PrecalOutputDir "data_contract.json"
        data_contract_sha256 = [string]$dataContract.contract_sha256
        downstream_model_evaluation = [ordered]@{
            model_freeze_bound = $true
            model_evaluation_count = 0
            global_consumption_marker = [string]$dataContract.precal_holdout.consumption_marker
            global_consumption_marker_must_be_created_exclusively_before_parse = $true
            ready_for_exactly_one_bound_model_evaluation = $true
        }
        sealed_calibration_opened = $false
        inherited_locked_opened = $false
        current_profile_mutated = $false
        runtime_policy_activated = $false
        received_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    $oneShotReceiptPath = Join-Path $precalOutputStage "one_shot_receipt.json"
    Write-Utf8NoBom -Path $oneShotReceiptPath -Text (($oneShotReceipt | ConvertTo-Json -Depth 20) + "`n")
    Add-CanonicalSelfHash -Path $oneShotReceiptPath -Field "receipt_sha256"
    Assert-CanonicalSelfHash -Path $oneShotReceiptPath -Field "receipt_sha256"
    if (Test-Path -LiteralPath $PrecalOutputDir) { throw "Attempt03 pre-cal output destination appeared during receive" }
    Move-Item -LiteralPath $precalOutputStage -Destination $PrecalOutputDir
    [pscustomobject]@{
        schema = "hu_m43_attempt03_teacher_precal_receive_result_v1"
        status = "verified_structural_precal_open_after_frozen_claim"
        run_name = $RunName; verified_shards = 20; verified_roots = 200
        output_dir = $PrecalOutputDir; download_dir = $PrecalDownloadDir
        data_contract = Join-Path $PrecalOutputDir "data_contract.json"
        one_shot_receipt = Join-Path $PrecalOutputDir "one_shot_receipt.json"
        model_evaluation_count = 0; current_profile_mutated = $false
    } | ConvertTo-Json -Depth 8
}
finally { $env:PYTHONPATH = $envPythonPath }
