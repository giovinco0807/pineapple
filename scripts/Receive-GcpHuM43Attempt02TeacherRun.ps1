param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [Parameter(Mandatory = $true)]
    [string]$PlanPath,
    [Parameter(Mandatory = $true)]
    [string]$Attempt01DataContractPath,
    [Parameter(Mandatory = $true)]
    [string]$Attempt01TrainingManifestPath,
    [Parameter(Mandatory = $true)]
    [string]$Attempt01TrainPath,
    [Parameter(Mandatory = $true)]
    [string]$Attempt01CalibrationPath,
    [Parameter(Mandatory = $true)]
    [string]$InheritedLockedPath,
    [Parameter(Mandatory = $true)]
    [string]$PreflightReceiptPath,
    [string]$DownloadDir,
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"
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

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-LineCount {
    param([string]$Path)
    $reader = [System.IO.File]::OpenText((Resolve-Path -LiteralPath $Path))
    try {
        $count = 0
        while ($null -ne $reader.ReadLine()) { $count += 1 }
        return $count
    }
    finally { $reader.Close() }
}

function Write-Utf8NoBom {
    param([string]$Path, [string]$Text)
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.UTF8Encoding]::new($false))
}

function Merge-Shards {
    param([object[]]$Records, [string]$Destination)
    $temporary = "$Destination.tmp"
    $writer = [System.IO.StreamWriter]::new($temporary, $false, [System.Text.UTF8Encoding]::new($false))
    try {
        foreach ($record in @($Records | Sort-Object shard)) {
            foreach ($line in [System.IO.File]::ReadLines((Resolve-Path -LiteralPath $record.path))) {
                $writer.WriteLine($line)
            }
        }
    }
    finally { $writer.Close() }
    Move-Item -LiteralPath $temporary -Destination $Destination -Force
}

function Assert-CanonicalSelfHash {
    param([string]$Path, [string]$Field)
    $script = @'
import hashlib,json,sys
path,field=sys.argv[1:]
obj=json.load(open(path,encoding="utf-8"))
claimed=obj.pop(field,None)
actual=hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()
if claimed != actual:
    raise SystemExit(f"{field} mismatch: {claimed} != {actual}")
'@
    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @($script | & python - $Path $Field 2>&1)
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $saved }
    if ($code -ne 0) { throw "Canonical self-hash validation failed for $Path`: $($output -join "`n")" }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$localInputs = @(
    $PlanPath, $Attempt01DataContractPath, $Attempt01TrainingManifestPath,
    $Attempt01TrainPath, $Attempt01CalibrationPath, $InheritedLockedPath,
    $PreflightReceiptPath
)
for ($index = 0; $index -lt $localInputs.Count; $index += 1) {
    $localInputs[$index] = (Resolve-Path -LiteralPath $localInputs[$index]).Path
}
$PlanPath, $Attempt01DataContractPath, $Attempt01TrainingManifestPath,
    $Attempt01TrainPath, $Attempt01CalibrationPath, $InheritedLockedPath,
    $PreflightReceiptPath = $localInputs

if (-not $DownloadDir) { $DownloadDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName/received-attempt02-teacher" }
if (-not $OutputDir) { $OutputDir = Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt02_teacher/$RunName" }
$DownloadDir = [System.IO.Path]::GetFullPath($DownloadDir)
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$allowedOutputRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt02_teacher"))
if (-not ($OutputDir + [IO.Path]::DirectorySeparatorChar).StartsWith(
        $allowedOutputRoot + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase
)) {
    throw "OutputDir must remain under outputs/hu_joint_policy/m43_attempt02_teacher"
}
if (Test-Path -LiteralPath $OutputDir) {
    throw "Attempt02 teacher OutputDir is immutable and already exists: $OutputDir"
}

$prefix = "gs://$Bucket/runs/$RunName"
$sourceDir = Join-Path $DownloadDir "source"
$resultsDir = Join-Path $DownloadDir "results"
New-Item -ItemType Directory -Force -Path $DownloadDir, $sourceDir, $resultsDir, $allowedOutputRoot | Out-Null
$manifestPath = Join-Path $DownloadDir "manifest.json"
$shardsPath = Join-Path $sourceDir "shards_manifest.jsonl"
$modelManifestPath = Join-Path $sourceDir "source_model_manifest.json"
$nativeManifestPath = Join-Path $sourceDir "source_native_manifest.json"
$sourcePath = Join-Path $sourceDir "ofc_regular_hu_m43_attempt02_teacher_source.zip"
$startupPath = Join-Path $sourceDir "startup_hu_m43_attempt02_teacher.sh"

foreach ($pair in @(
    @("$prefix/manifest.json", $manifestPath),
    @("$prefix/source/shards_manifest.jsonl", $shardsPath),
    @("$prefix/source/source_model_manifest.json", $modelManifestPath),
    @("$prefix/source/source_native_manifest.json", $nativeManifestPath),
    @("$prefix/source/ofc_regular_hu_m43_attempt02_teacher_source.zip", $sourcePath),
    @("$prefix/source/startup_hu_m43_attempt02_teacher.sh", $startupPath)
)) {
    & gcloud storage cp $pair[0] $pair[1] --project $ProjectId | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Unable to download frozen Attempt02 artifact: $($pair[0])" }
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.schema -ne "hu_m43_attempt02_teacher_spot_manifest_v1" -or
    [string]$manifest.run_name -ne $RunName -or
    [string]$manifest.project_id -ne $ProjectId -or
    [string]$manifest.bucket -ne $Bucket) {
    throw "Attempt02 manifest identity/schema mismatch"
}
$manifestText = $manifest | ConvertTo-Json -Depth 20 -Compress
if ($manifestText -match '(?i)locked|attempt01|inherited[-_ ]holdout') {
    throw "Attempt02 cloud manifest contains sealed/local-only material"
}
$splitKeys = @($manifest.split_roots.psobject.Properties.Name | Sort-Object)
if (($splitKeys -join ',') -ne 'calibration,train') {
    throw "Attempt02 cloud manifest must contain only train/calibration"
}

$manifestSha256 = Get-Sha256 $manifestPath
$shardsSha256 = Get-Sha256 $shardsPath
$sourceSha256 = Get-Sha256 $sourcePath
$startupSha256 = Get-Sha256 $startupPath
$modelManifestSha256 = Get-Sha256 $modelManifestPath
$nativeManifestSha256 = Get-Sha256 $nativeManifestPath
if ($shardsSha256 -ne [string]$manifest.shards_manifest_sha256 -or
    $sourceSha256 -ne [string]$manifest.source_sha256 -or
    $startupSha256 -ne [string]$manifest.startup_sha256 -or
    $modelManifestSha256 -ne [string]$manifest.model_manifest_sha256 -or
    $nativeManifestSha256 -ne [string]$manifest.native_manifest_sha256) {
    throw "Attempt02 frozen source/startup/manifest hash verification failed"
}

$preflight = Get-Content -LiteralPath $PreflightReceiptPath -Raw | ConvertFrom-Json
if ($preflight.schema -ne "hu_m43_attempt02_preflight_receipt_v1" -or
    $preflight.status -ne "pass_frozen_before_fresh_generation") {
    throw "Attempt02 preflight receipt schema/status mismatch"
}
Assert-CanonicalSelfHash -Path $PreflightReceiptPath -Field "receipt_sha256"
if ((Get-Sha256 $PreflightReceiptPath) -ne [string]$manifest.local_preflight_binding.file_sha256 -or
    [string]$preflight.receipt_sha256 -ne [string]$manifest.local_preflight_binding.receipt_sha256) {
    throw "Local Attempt02 preflight receipt does not match the cloud run binding"
}
if ((Get-Sha256 $PlanPath) -ne [string]$preflight.plan.file_sha256) {
    throw "Local Attempt02 plan does not match the frozen preflight binding"
}
$freshGeneration = $preflight.fresh_generation
if ([int]$manifest.total_roots -ne [int]$freshGeneration.fresh_roots -or
    [int]$manifest.total_shards -ne [int]$freshGeneration.fresh_shards -or
    [int]$manifest.roots_per_shard -ne [int]$freshGeneration.roots_per_shard -or
    [int]$manifest.split_roots.train -ne [int]$freshGeneration.splits.train.roots -or
    [int]$manifest.split_roots.calibration -ne [int]$freshGeneration.splits.calibration.roots -or
    [int]$manifest.candidate_samples -ne [int]$freshGeneration.teacher_search.candidate_samples -or
    [int]$manifest.evaluation_samples -ne [int]$freshGeneration.teacher_search.evaluation_samples) {
    throw "Attempt02 cloud manifest disagrees with the frozen preflight generation projection"
}

$specs = @()
foreach ($line in [System.IO.File]::ReadLines((Resolve-Path -LiteralPath $shardsPath))) {
    if ($line.Trim()) { $specs += ($line | ConvertFrom-Json) }
}
if ($specs.Count -ne [int]$manifest.total_shards) { throw "Attempt02 shard manifest count mismatch" }
$uniqueShardIds = @($specs | ForEach-Object { [int]$_.shard } | Sort-Object -Unique)
$uniqueOutputPrefixes = @($specs | ForEach-Object { [string]$_.output_prefix } | Sort-Object -Unique)
$trainSpecs = @($specs | Where-Object split -eq "train")
$calibrationSpecs = @($specs | Where-Object split -eq "calibration")
if ($uniqueShardIds.Count -ne 30 -or (($uniqueShardIds -join ',') -ne ((0..29) -join ',')) -or
    $uniqueOutputPrefixes.Count -ne 30 -or $trainSpecs.Count -ne 20 -or
    $calibrationSpecs.Count -ne 10 -or
    @($specs | Where-Object {
        [int]$_.roots -ne 10 -or [int]$_.candidate_samples -ne 2 -or
        [int]$_.evaluation_samples -ne 64
    }).Count -ne 0) {
    throw "Attempt02 requires the frozen 20 train + 10 calibration original-shard schedule"
}
$verified = @()
$missing = @()
foreach ($spec in $specs) {
    if ([string]$spec.split -notin @("train", "calibration")) {
        throw "Forbidden split in Attempt02 shard manifest"
    }
    $resultUri = "$prefix/results/$($spec.output_prefix)"
    $resultPath = Join-Path $resultsDir ([string]$spec.output_prefix)
    New-Item -ItemType Directory -Force -Path $resultPath | Out-Null
    $donePath = Join-Path $resultPath "DONE.json"
    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & gcloud storage cp "$resultUri/DONE.json" $donePath --project $ProjectId *> $null
        $doneCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $saved }
    if ($doneCode -ne 0) { $missing += [int]$spec.shard; continue }
    foreach ($file in @("teacher.jsonl", "checkpoint.json", "heartbeat.json", "generator_summary.json", "run.log")) {
        & gcloud storage cp "$resultUri/$file" (Join-Path $resultPath $file) --project $ProjectId | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Unable to download Attempt02 result file: $resultUri/$file" }
    }
    $done = Get-Content -LiteralPath $donePath -Raw | ConvertFrom-Json
    $teacherPath = Join-Path $resultPath "teacher.jsonl"
    $checkpointPath = Join-Path $resultPath "checkpoint.json"
    $heartbeatPath = Join-Path $resultPath "heartbeat.json"
    if ($done.schema -ne "hu_m43_attempt02_teacher_done_v1" -or $done.status -ne "complete" -or
        [string]$done.run_name -ne $RunName -or [int]$done.roots -ne [int]$spec.roots -or
        [int]$done.shard -ne [int]$spec.shard -or [string]$done.split -ne [string]$spec.split -or
        [string]$done.output_prefix -ne [string]$spec.output_prefix -or
        [string]$done.manifest_sha256 -ne $manifestSha256 -or
        [string]$done.shards_manifest_sha256 -ne $shardsSha256 -or
        [string]$done.source_sha256 -ne $sourceSha256 -or
        [string]$done.startup_sha256 -ne $startupSha256 -or
        [string]$done.model_manifest_sha256 -ne $modelManifestSha256 -or
        [string]$done.native_manifest_sha256 -ne $nativeManifestSha256 -or
        [string]$done.output_sha256 -ne (Get-Sha256 $teacherPath) -or
        [string]$done.checkpoint_sha256 -ne (Get-Sha256 $checkpointPath) -or
        [string]$done.heartbeat_sha256 -ne (Get-Sha256 $heartbeatPath) -or
        (Get-LineCount $teacherPath) -ne [int]$spec.roots) {
        throw "Downloaded Attempt02 shard hash/identity chain is invalid: $($spec.shard)"
    }
    $checkpoint = Get-Content -LiteralPath $checkpointPath -Raw | ConvertFrom-Json
    $heartbeat = Get-Content -LiteralPath $heartbeatPath -Raw | ConvertFrom-Json
    foreach ($summary in @($checkpoint, $heartbeat)) {
        if ($summary.schema -ne "hu_m4_t1_second_shard_v1" -or $summary.status -ne "complete" -or
            [bool]$summary.current_profile_resolved -or
            [string]$summary.output_sha256 -ne [string]$done.output_sha256 -or
            [string]$summary.config.split -ne [string]$spec.split -or
            [long]$summary.config.seed_start -ne [long]$spec.seed_start -or
            [long]$summary.config.seed_stride -ne [long]$spec.seed_stride -or
            [int]$summary.config.candidate_samples -ne [int]$spec.candidate_samples -or
            [int]$summary.config.evaluation_samples -ne [int]$spec.evaluation_samples) {
            throw "Attempt02 generator completion manifest mismatch: $($spec.shard)"
        }
    }
    $verified += [pscustomobject]@{
        shard = [int]$spec.shard
        split = [string]$spec.split
        roots = [int]$spec.roots
        path = $teacherPath
        sha256 = [string]$done.output_sha256
    }
}
if ($missing.Count -gt 0) {
    throw "Missing Attempt02 DONE shards: $($missing -join ','). Use Get-GcpHuM43Attempt02TeacherRunStatus.ps1."
}
if ($verified.Count -ne $specs.Count) { throw "Attempt02 verified shard count mismatch" }

$train = @($verified | Where-Object split -eq "train" | Sort-Object shard)
$calibration = @($verified | Where-Object split -eq "calibration" | Sort-Object shard)
if ($train.Count -ne 20 -or $calibration.Count -ne 10) {
    throw "Attempt02 receive requires exactly 20 original train and 10 original calibration shards"
}

$stageDir = Join-Path $allowedOutputRoot (".receiving-{0}-{1}" -f $RunName, [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $stageDir | Out-Null
$stageDataContractPath = Join-Path $stageDir "data_contract.json"
$dataContractPath = Join-Path $OutputDir "data_contract.json"
$finalizeArgs = @(
    "-B", "-m", "ofc_regular.hu_m43_attempt02_contract", "finalize-fresh",
    "--plan", $PlanPath, "--repo-root", $repoRoot,
    "--attempt01-data-contract", $Attempt01DataContractPath,
    "--attempt01-training-manifest", $Attempt01TrainingManifestPath,
    "--attempt01-train", $Attempt01TrainPath,
    "--attempt01-calibration", $Attempt01CalibrationPath,
    "--inherited-locked", $InheritedLockedPath,
    "--preflight-receipt", $PreflightReceiptPath
)
foreach ($record in $train) { $finalizeArgs += @("--train", $record.path) }
foreach ($record in $calibration) { $finalizeArgs += @("--calibration", $record.path) }
$finalizeArgs += @("--output", $stageDataContractPath)
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $repoRoot "src"
try {
    & python @finalizeArgs
    if ($LASTEXITCODE -ne 0) { throw "Attempt02 finalize-fresh failed with exit code $LASTEXITCODE" }
}
finally { $env:PYTHONPATH = $previousPythonPath }
$dataContract = Get-Content -LiteralPath $stageDataContractPath -Raw | ConvertFrom-Json
if ($dataContract.schema -ne "hu_m43_attempt02_data_contract_v1" -or
    $dataContract.status -ne "pass_fresh_train_calibration_sealed_inherited_locked_unopened") {
    throw "Attempt02 final data contract schema/status mismatch"
}
Assert-CanonicalSelfHash -Path $stageDataContractPath -Field "contract_sha256"

$merged = [ordered]@{}
foreach ($split in @("train", "calibration")) {
    $records = if ($split -eq "train") { $train } else { $calibration }
    $stageDestination = Join-Path $stageDir "$split.jsonl"
    $destination = Join-Path $OutputDir "$split.jsonl"
    Merge-Shards -Records $records -Destination $stageDestination
    $merged[$split] = [ordered]@{
        path = $destination
        rows = Get-LineCount $stageDestination
        sha256 = Get-Sha256 $stageDestination
    }
}
if ($merged.train.rows -ne [int]$manifest.split_roots.train -or
    $merged.calibration.rows -ne [int]$manifest.split_roots.calibration) {
    throw "Merged Attempt02 row count disagrees with frozen manifest"
}

$originalTrainShards = @($train | ForEach-Object {
    [ordered]@{
        shard = [int]$_.shard
        path = [string]$_.path
        rows = [int]$_.roots
        sha256 = [string]$_.sha256
    }
})
$originalCalibrationShards = @($calibration | ForEach-Object {
    [ordered]@{
        shard = [int]$_.shard
        path = [string]$_.path
        rows = [int]$_.roots
        sha256 = [string]$_.sha256
    }
})
$downstreamInputs = [ordered]@{
    schema = "hu_m43_attempt02_teacher_downstream_inputs_v1"
    model_start = [ordered]@{
        train_shards = $originalTrainShards
        train_paths = @($originalTrainShards | ForEach-Object { $_.path })
        merged_train_is_not_a_model_start_input = $true
    }
    model_receive = [ordered]@{
        train_paths = @($originalTrainShards | ForEach-Object { $_.path })
        calibration_shards = $originalCalibrationShards
        calibration_paths = @($originalCalibrationShards | ForEach-Object { $_.path })
        attempt02_data_contract = $dataContractPath
    }
}

$receiptPath = Join-Path $OutputDir "receipt.json"
$stageReceiptPath = Join-Path $stageDir "receipt.json"
$receipt = [ordered]@{
    schema = "hu_m43_attempt02_teacher_receive_receipt_v1"
    status = "verified_fresh_train_calibration_only"
    run_name = $RunName
    manifest_sha256 = $manifestSha256
    shards_manifest_sha256 = $shardsSha256
    source_sha256 = $sourceSha256
    startup_sha256 = $startupSha256
    preflight_receipt_sha256 = [string]$preflight.receipt_sha256
    data_contract = $dataContractPath
    data_contract_sha256 = [string]$dataContract.contract_sha256
    expected_shards = $specs.Count
    verified_shards = $verified.Count
    verified_roots = [int](($verified | Measure-Object roots -Sum).Sum)
    merged = $merged
    downstream_inputs = $downstreamInputs
    current_profile_mutated = $false
    no_runtime_activation = $true
    received_at = (Get-Date).ToUniversalTime().ToString("o")
}
Write-Utf8NoBom -Path $stageReceiptPath -Text (($receipt | ConvertTo-Json -Depth 12) + "`n")
if (Test-Path -LiteralPath $OutputDir) {
    throw "Attempt02 teacher OutputDir appeared during receive; staging preserved at $stageDir"
}
Move-Item -LiteralPath $stageDir -Destination $OutputDir

[pscustomobject]@{
    schema = "hu_m43_attempt02_teacher_receive_result_v1"
    status = "verified_fresh_train_calibration_only"
    run_name = $RunName
    verified_shards = $verified.Count
    verified_roots = $receipt.verified_roots
    output_dir = $OutputDir
    download_dir = $DownloadDir
    data_contract = $dataContractPath
    receipt = $receiptPath
    merged = $merged
    downstream_inputs = $downstreamInputs
    current_profile_mutated = $false
} | ConvertTo-Json -Depth 12
