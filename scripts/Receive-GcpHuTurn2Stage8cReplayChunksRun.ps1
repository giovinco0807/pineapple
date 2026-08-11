param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [string]$DownloadDir,
    [string]$OutputDir = "outputs/evals/hu_turn2_stage8c_replay_chunks_gcp",
    [string]$InputJsonl = "",
    [switch]$AllowPartial,
    [switch]$SkipLossTargets,
    [switch]$SkipRiskTargetAudit,
    [switch]$SkipRiskTargetGap
)

$ErrorActionPreference = "Stop"

$cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (Test-Path $cloudSdkGcloud) {
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}
elseif (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud not found in PATH or at $cloudSdkGcloud"
}

function Merge-ReplayResults {
    param([object[]]$Specs, [string]$ResultsDir, [string]$MergedDir)
    New-Item -ItemType Directory -Force -Path $MergedDir | Out-Null
    $summaryRows = @()
    $teacherOut = Join-Path $MergedDir "topk_hard_negative_replay_teacher.jsonl"
    if (Test-Path -LiteralPath $teacherOut) { Remove-Item -LiteralPath $teacherOut -Force }
    foreach ($spec in $Specs) {
        $resultPath = Join-Path $ResultsDir $spec.output_prefix
        $summaryPath = Join-Path $resultPath "topk_hard_negative_replay_summary.csv"
        if (Test-Path -LiteralPath $summaryPath) {
            $summaryRows += Import-Csv -LiteralPath $summaryPath
        }
        $teacherPath = Join-Path $resultPath "topk_hard_negative_replay_teacher.jsonl"
        if (Test-Path -LiteralPath $teacherPath) {
            foreach ($line in [System.IO.File]::ReadLines((Resolve-Path -LiteralPath $teacherPath).Path)) {
                if (-not [string]::IsNullOrWhiteSpace($line)) {
                    Add-Content -LiteralPath $teacherOut -Value $line -Encoding UTF8
                }
            }
        }
    }
    $summaryOut = Join-Path $MergedDir "topk_hard_negative_replay_summary.csv"
    if ($summaryRows.Count -gt 0) {
        $summaryRows | Sort-Object {[int]$_.row_index} | Export-Csv -LiteralPath $summaryOut -NoTypeInformation -Encoding UTF8
    } else {
        "" | Set-Content -LiteralPath $summaryOut -Encoding UTF8
    }
    return [pscustomobject]@{
        summary = $summaryOut
        teacher = $teacherOut
        summary_rows = $summaryRows.Count
    }
}

if (-not $DownloadDir) {
    $DownloadDir = "outputs/gcp_runs/$RunName"
}

$prefix = "gs://$Bucket/runs/$RunName"
$manifestPath = Join-Path $DownloadDir "manifest.json"
$shardManifestPath = Join-Path $DownloadDir "shards_manifest.jsonl"
$resultsDir = Join-Path $DownloadDir "results"
$statusDir = Join-Path $DownloadDir "status"
$mergedDir = Join-Path $OutputDir "merged"
$lossTargetsDir = Join-Path $OutputDir "loss_targets_from_pack"
$collectionDir = Join-Path $OutputDir "loss_target_collection"
$riskTargetAuditDir = Join-Path $OutputDir "risk_target_audit"
$riskTargetGapDir = Join-Path $OutputDir "risk_target_gap"
New-Item -ItemType Directory -Force -Path $DownloadDir, $resultsDir, $statusDir, $OutputDir | Out-Null

gcloud storage cp "$prefix/manifest.json" $manifestPath --project $ProjectId | Out-Null
gcloud storage cp "$prefix/source/shards_manifest.jsonl" $shardManifestPath --project $ProjectId | Out-Null
gcloud storage cp --recursive "$prefix/results/*" $resultsDir --project $ProjectId | Out-Null
try {
    gcloud storage cp "$prefix/status/*.json" $statusDir --project $ProjectId | Out-Null
}
catch {
    Write-Warning "No statuses downloaded or status download failed: $_"
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$specs = @()
foreach ($line in [System.IO.File]::ReadLines((Resolve-Path -LiteralPath $shardManifestPath).Path)) {
    if ($line.Trim()) { $specs += ($line | ConvertFrom-Json) }
}
$expectedShards = [int]$manifest.total_shards
if ($specs.Count -ne $expectedShards) {
    throw "Shard manifest count mismatch: $($specs.Count) != $expectedShards"
}

$missing = @()
$completedSpecs = @()
foreach ($spec in $specs) {
    $resultPath = Join-Path $resultsDir $spec.output_prefix
    $donePath = Join-Path $resultPath "DONE"
    if (-not (Test-Path -LiteralPath $donePath)) {
        $missing += [int]$spec.shard
    } else {
        $completedSpecs += $spec
    }
}
if ($missing.Count -gt 0 -and -not $AllowPartial) {
    throw "Missing Stage8c replay chunk result shards: $($missing -join ',')"
}
if ($missing.Count -gt 0) {
    Write-Warning "Merging partial Stage8c replay chunks; missing shards: $($missing -join ',')"
}
if ($completedSpecs.Count -eq 0) {
    throw "No completed Stage8c replay chunk result shards found"
}

$merge = Merge-ReplayResults -Specs $completedSpecs -ResultsDir $resultsDir -MergedDir $mergedDir

$lossManifest = $null
if (-not $SkipLossTargets) {
    $resolvedInput = $InputJsonl
    if (-not $resolvedInput) { $resolvedInput = [string]$manifest.input_jsonl }
    if (-not (Test-Path -LiteralPath $resolvedInput)) {
        throw "InputJsonl for loss target build not found: $resolvedInput"
    }
    python -m ofc_regular.prepare_hu_turn2_stage8b_counterfactual_loss_targets `
        --decision-log $resolvedInput `
        --local-replay-summary $merge.summary `
        --include-non-loss-controls `
        --output-dir $lossTargetsDir
    if ($LASTEXITCODE -ne 0) { throw "Loss target build failed with exit code $LASTEXITCODE" }
    python -m ofc_regular.collect_hu_turn2_stage8b_loss_targets `
        --input-dir $lossTargetsDir `
        --output-dir $collectionDir
    if ($LASTEXITCODE -ne 0) { throw "Loss target collection failed with exit code $LASTEXITCODE" }
    if (-not $SkipRiskTargetAudit) {
        python -m ofc_regular.analyze_hu_turn2_stage8b_risk_targets `
            --collection-dir $collectionDir `
            --output-dir $riskTargetAuditDir
        if ($LASTEXITCODE -ne 0) { throw "Risk target audit failed with exit code $LASTEXITCODE" }
    }
    if (-not $SkipRiskTargetGap) {
        python -m ofc_regular.analyze_hu_turn2_stage8c_risk_target_gap `
            --collection-dir $collectionDir `
            --output-dir $riskTargetGapDir
        if ($LASTEXITCODE -ne 0) { throw "Risk target gap analysis failed with exit code $LASTEXITCODE" }
    }
    $lossManifestPath = Join-Path $collectionDir "topk_loss_target_collection_manifest.json"
    if (Test-Path -LiteralPath $lossManifestPath) {
        $lossManifest = Get-Content -LiteralPath $lossManifestPath -Raw | ConvertFrom-Json
    }
}

[pscustomobject]@{
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    expected_shards = $expectedShards
    completed_shards = $completedSpecs.Count
    missing_shards = $missing
    allow_partial = [bool]$AllowPartial
    output_dir = $OutputDir
    download_dir = $DownloadDir
    merged_summary = $merge.summary
    merged_teacher = $merge.teacher
    merged_summary_rows = $merge.summary_rows
    loss_targets_dir = if ($SkipLossTargets) { "" } else { $lossTargetsDir }
    loss_target_collection_dir = if ($SkipLossTargets) { "" } else { $collectionDir }
    risk_target_audit_dir = if ($SkipLossTargets -or $SkipRiskTargetAudit) { "" } else { $riskTargetAuditDir }
    risk_target_gap_dir = if ($SkipLossTargets -or $SkipRiskTargetGap) { "" } else { $riskTargetGapDir }
    loss_target_collection_manifest = $lossManifest
    production_p2_fixed = "No-Go"
    t1_training = "No-Go"
    teacher_50k = "No-Go"
} | ConvertTo-Json -Depth 8
