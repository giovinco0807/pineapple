param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [string]$DownloadDir,
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"

$cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (Test-Path $cloudSdkGcloud) {
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

function Merge-Shards {
    param([object[]]$Records, [string]$Destination)
    $encoding = [System.Text.UTF8Encoding]::new($false)
    $temporary = "$Destination.tmp"
    $writer = [System.IO.StreamWriter]::new($temporary, $false, $encoding)
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

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $DownloadDir) { $DownloadDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName/received" }
if (-not $OutputDir) { $OutputDir = Join-Path $repoRoot "outputs/hu_joint_policy/m42_spot/$RunName" }
$prefix = "gs://$Bucket/runs/$RunName"
$manifestPath = Join-Path $DownloadDir "manifest.json"
$shardsPath = Join-Path $DownloadDir "shards_manifest.jsonl"
$modelManifestPath = Join-Path $DownloadDir "source_model_manifest.json"
$nativeManifestPath = Join-Path $DownloadDir "source_native_manifest.json"
$sourceDir = Join-Path $DownloadDir "source"
$resultsDir = Join-Path $DownloadDir "results"
New-Item -ItemType Directory -Force -Path $DownloadDir, $sourceDir, $resultsDir, $OutputDir | Out-Null

gcloud storage cp "$prefix/manifest.json" $manifestPath --project $ProjectId | Out-Null
gcloud storage cp "$prefix/source/shards_manifest.jsonl" $shardsPath --project $ProjectId | Out-Null
gcloud storage cp "$prefix/source/source_model_manifest.json" $modelManifestPath --project $ProjectId | Out-Null
gcloud storage cp "$prefix/source/source_native_manifest.json" $nativeManifestPath --project $ProjectId | Out-Null

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.schema -ne "hu_m42_spot_manifest_v1") { throw "M4.2 manifest schema mismatch" }
$sourcePath = Join-Path $sourceDir "ofc_regular_hu_m42_source.zip"
$startupPath = Join-Path $sourceDir "startup_hu_m42_spot.sh"
gcloud storage cp ([string]$manifest.source_uri) $sourcePath --project $ProjectId | Out-Null
gcloud storage cp ([string]$manifest.startup_uri) $startupPath --project $ProjectId | Out-Null
$manifestSha256 = Get-Sha256 $manifestPath
$sourceSha256 = Get-Sha256 $sourcePath
$startupSha256 = Get-Sha256 $startupPath
$shardsSha256 = Get-Sha256 $shardsPath
$modelsSha256 = Get-Sha256 $modelManifestPath
$nativeSha256 = Get-Sha256 $nativeManifestPath
if ($sourceSha256 -ne [string]$manifest.source_sha256 -or
    $startupSha256 -ne [string]$manifest.startup_sha256) {
    throw "M4.2 frozen source/startup SHA256 mismatch"
}
if ($shardsSha256 -ne [string]$manifest.shards_manifest_sha256) {
    throw "M4.2 shard manifest SHA256 mismatch"
}
if ($modelsSha256 -ne [string]$manifest.model_manifest_sha256) {
    throw "M4.2 source model manifest SHA256 mismatch"
}
if ($nativeSha256 -ne [string]$manifest.native_manifest_sha256) {
    throw "M4.2 source native manifest SHA256 mismatch"
}

$modelManifest = Get-Content -LiteralPath $modelManifestPath -Raw | ConvertFrom-Json
if ($modelManifest.schema -ne "hu_m42_source_model_manifest_v1" -or
    [int]$modelManifest.model_count -ne 11 -or
    [int]$manifest.required_model_count -ne 11) {
    throw "M4.2 requires an exact 11-model source manifest"
}
if ([long]$modelManifest.total_bytes -ne [long]$manifest.required_model_bytes) {
    throw "M4.2 model byte total disagrees with the run manifest"
}
$declaredModels = @{}
foreach ($row in $manifest.required_models) { $declaredModels[[string]$row.path] = $row }
foreach ($row in $modelManifest.models) {
    $path = [string]$row.path
    if (-not $declaredModels.ContainsKey($path)) { throw "Model missing from run manifest: $path" }
    $declared = $declaredModels[$path]
    if ([long]$row.bytes -ne [long]$declared.bytes -or [string]$row.sha256 -ne [string]$declared.sha256) {
        throw "Model hash/size mismatch between manifests: $path"
    }
}

$nativeManifest = Get-Content -LiteralPath $nativeManifestPath -Raw | ConvertFrom-Json
if ($nativeManifest.schema -ne "hu_m42_source_native_manifest_v1" -or
    [int]$nativeManifest.binary_count -ne 2 -or
    [int]$manifest.required_native_binary_count -ne 2) {
    throw "M4.2 requires an exact two-native-binary source manifest"
}
$declaredNative = @{}
foreach ($row in $manifest.required_native_binaries) { $declaredNative[[string]$row.path] = $row }
foreach ($row in $nativeManifest.binaries) {
    $path = [string]$row.path
    if (-not $declaredNative.ContainsKey($path)) { throw "Native binary missing from run manifest: $path" }
    $declared = $declaredNative[$path]
    if ([long]$row.bytes -ne [long]$declared.bytes -or [string]$row.sha256 -ne [string]$declared.sha256) {
        throw "Native hash/size mismatch between manifests: $path"
    }
}

$specs = @()
foreach ($line in [System.IO.File]::ReadLines((Resolve-Path -LiteralPath $shardsPath))) {
    if ($line.Trim()) { $specs += ($line | ConvertFrom-Json) }
}
if ($specs.Count -ne [int]$manifest.total_shards) {
    throw "M4.2 shard manifest count mismatch: $($specs.Count) != $($manifest.total_shards)"
}

$missing = @()
$verified = @()
foreach ($spec in $specs) {
    $outputPrefix = [string]$spec.output_prefix
    $resultUri = "$prefix/results/$outputPrefix"
    $doneUri = "$resultUri/DONE"
    $doneExists = @()
    try { $doneExists = @(gcloud storage ls $doneUri --project $ProjectId 2>$null) }
    catch { $doneExists = @() }
    if ($doneExists -notcontains $doneUri) {
        $missing += [int]$spec.shard
        continue
    }

    $resultPath = Join-Path $resultsDir $outputPrefix
    New-Item -ItemType Directory -Force -Path $resultPath | Out-Null
    foreach ($file in @("teacher.jsonl", "checkpoint.json", "heartbeat.json", "generator_summary.json", "run.log", "DONE")) {
        gcloud storage cp "$resultUri/$file" (Join-Path $resultPath $file) --project $ProjectId | Out-Null
    }

    $donePath = Join-Path $resultPath "DONE"
    $teacherPath = Join-Path $resultPath "teacher.jsonl"
    $checkpointPath = Join-Path $resultPath "checkpoint.json"
    $heartbeatPath = Join-Path $resultPath "heartbeat.json"
    $done = Get-Content -LiteralPath $donePath -Raw | ConvertFrom-Json
    if ($done.schema -ne "hu_m42_spot_done_v1" -or $done.status -ne "complete") {
        throw "Invalid DONE schema/status for shard $($spec.shard)"
    }
    if ([int]$done.shard -ne [int]$spec.shard -or
        [string]$done.split -ne [string]$spec.split -or
        [string]$done.output_prefix -ne $outputPrefix -or
        [int]$done.roots -ne [int]$spec.roots) {
        throw "DONE identity mismatch for shard $($spec.shard)"
    }
    if ([string]$done.manifest_sha256 -ne $manifestSha256 -or
        [string]$done.shards_manifest_sha256 -ne $shardsSha256 -or
        [string]$done.model_manifest_sha256 -ne $modelsSha256 -or
        [string]$done.native_manifest_sha256 -ne $nativeSha256 -or
        [string]$done.source_sha256 -ne $sourceSha256 -or
        [string]$done.startup_sha256 -ne $startupSha256) {
        throw "DONE manifest/source hash chain mismatch for shard $($spec.shard)"
    }
    $teacherSha256 = Get-Sha256 $teacherPath
    $checkpointSha256 = Get-Sha256 $checkpointPath
    $heartbeatSha256 = Get-Sha256 $heartbeatPath
    if ($teacherSha256 -ne [string]$done.output_sha256 -or
        $checkpointSha256 -ne [string]$done.checkpoint_sha256 -or
        $heartbeatSha256 -ne [string]$done.heartbeat_sha256) {
        throw "Downloaded result hash mismatch for shard $($spec.shard)"
    }
    $lineCount = Get-LineCount $teacherPath
    if ($lineCount -ne [int]$spec.roots) {
        throw "Teacher row count mismatch for shard $($spec.shard): $lineCount != $($spec.roots)"
    }
    $checkpoint = Get-Content -LiteralPath $checkpointPath -Raw | ConvertFrom-Json
    $heartbeat = Get-Content -LiteralPath $heartbeatPath -Raw | ConvertFrom-Json
    # Partial resume files use checkpoint/heartbeat schemas. On successful
    # completion the generator atomically replaces both with the same frozen
    # shard summary contract.
    if ($checkpoint.schema -ne "hu_m4_t1_second_shard_v1" -or
        $heartbeat.schema -ne "hu_m4_t1_second_shard_v1" -or
        $checkpoint.status -ne "complete" -or
        $heartbeat.status -ne "complete" -or
        [int]$checkpoint.roots -ne [int]$spec.roots -or
        [int]$heartbeat.roots -ne [int]$spec.roots -or
        [string]$checkpoint.output_sha256 -ne $teacherSha256 -or
        [string]$heartbeat.output_sha256 -ne $teacherSha256 -or
        [string]$checkpoint.config_sha256 -ne [string]$heartbeat.config_sha256 -or
        [bool]$checkpoint.current_profile_resolved -or
        [bool]$heartbeat.current_profile_resolved) {
        throw "Generator completion manifest mismatch for shard $($spec.shard)"
    }
    foreach ($summary in @($checkpoint, $heartbeat)) {
        if ([string]$summary.config.split -ne [string]$spec.split -or
            [int]$summary.config.roots -ne [int]$spec.roots -or
            [long]$summary.config.seed_start -ne [long]$spec.seed_start -or
            [long]$summary.config.seed_stride -ne [long]$spec.seed_stride -or
            [long]$summary.config.candidate_seed -ne [long]$spec.candidate_seed -or
            [long]$summary.config.evaluation_seed -ne [long]$spec.evaluation_seed -or
            [long]$summary.config.child_policy_seed -ne [long]$spec.child_policy_seed -or
            [int]$summary.config.candidate_samples -ne [int]$spec.candidate_samples -or
            [int]$summary.config.evaluation_samples -ne [int]$spec.evaluation_samples -or
            [string]$summary.config.baseline_profile -ne [string]$spec.baseline_profile -or
            [string]$summary.config.t2_profile -ne [string]$spec.t2_profile) {
            throw "Generator completion config mismatch for shard $($spec.shard)"
        }
    }
    $verified += [pscustomobject]@{
        shard = [int]$spec.shard
        split = [string]$spec.split
        roots = [int]$spec.roots
        path = $teacherPath
        sha256 = $teacherSha256
        done = $donePath
    }
}
if ($missing.Count -gt 0) {
    throw "Missing M4.2 DONE shards: $($missing -join ','). Relaunch with -StartShards '$($missing -join ',')'."
}
if ($verified.Count -ne $specs.Count) { throw "M4.2 verified shard count mismatch" }

$splitInputs = [ordered]@{
    train = @($verified | Where-Object split -eq "train" | Sort-Object shard)
    calibration = @($verified | Where-Object split -eq "calibration" | Sort-Object shard)
    locked_holdout = @($verified | Where-Object split -eq "locked_holdout" | Sort-Object shard)
}
foreach ($name in $splitInputs.Keys) {
    if ($splitInputs[$name].Count -eq 0) { throw "M4.2 split has no verified shards: $name" }
}

$auditPath = Join-Path $OutputDir "data_audit.json"
$auditArgs = @("-B", "-m", "ofc_regular.audit_hu_m4_t1_data")
foreach ($record in $splitInputs.train) { $auditArgs += @("--train", $record.path) }
foreach ($record in $splitInputs.calibration) { $auditArgs += @("--calibration", $record.path) }
foreach ($record in $splitInputs.locked_holdout) { $auditArgs += @("--locked-holdout", $record.path) }
$auditArgs += @("--output", $auditPath)
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $repoRoot "src"
try {
    & python @auditArgs
    if ($LASTEXITCODE -ne 0) { throw "M4.2 locked-split audit failed with exit code $LASTEXITCODE" }
}
finally { $env:PYTHONPATH = $previousPythonPath }
$audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
if ($audit.status -ne "pass") { throw "M4.2 locked-split audit did not pass" }

$mergedOutputs = [ordered]@{}
foreach ($name in $splitInputs.Keys) {
    $destination = Join-Path $OutputDir ("{0}.jsonl" -f $name)
    Merge-Shards -Records $splitInputs[$name] -Destination $destination
    $mergedOutputs[$name] = [ordered]@{
        path = $destination
        rows = Get-LineCount $destination
        sha256 = Get-Sha256 $destination
    }
}

$receiptPath = Join-Path $OutputDir "receipt.json"
$receipt = [ordered]@{
    schema = "hu_m42_spot_receipt_v1"
    status = "verified_and_audited"
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    manifest_sha256 = $manifestSha256
    shards_manifest_sha256 = $shardsSha256
    model_manifest_sha256 = $modelsSha256
    native_manifest_sha256 = $nativeSha256
    source_sha256 = $sourceSha256
    startup_sha256 = $startupSha256
    expected_shards = $specs.Count
    verified_shards = $verified.Count
    verified_roots = [int](($verified | Measure-Object roots -Sum).Sum)
    audit = $auditPath
    audit_sha256 = Get-Sha256 $auditPath
    merged = $mergedOutputs
    shard_outputs = $verified
    received_at = (Get-Date).ToUniversalTime().ToString("o")
}
$encoding = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText(
    $receiptPath,
    (($receipt | ConvertTo-Json -Depth 10) + "`n"),
    $encoding
)

[pscustomobject]@{
    schema = "hu_m42_spot_receive_result_v1"
    status = "verified_and_audited"
    run_name = $RunName
    expected_shards = $specs.Count
    verified_shards = $verified.Count
    verified_roots = $receipt.verified_roots
    output_dir = $OutputDir
    download_dir = $DownloadDir
    audit = $auditPath
    receipt = $receiptPath
    merged = $mergedOutputs
} | ConvertTo-Json -Depth 8
