param(
    [Parameter(Mandatory = $true)]
    [string]$InputJsonl,
    [string]$OutputRoot = "outputs\evals\hu_turn2_stage8c_replay_chunks",
    [int]$FutureSamples = 512,
    [int]$ChunkSize = 100,
    [int]$StartOffset = 0,
    [int]$TotalRows = 0,
    [int]$ReplaySeed = 2026064101,
    [int]$PredictionThreads = 1,
    [int]$OpeningLookaheadSamples = 32,
    [ValidateSet("stage3_reference_default", "stage7_m5_r10")]
    [string]$T3Continuation = "stage3_reference_default",
    [switch]$SkipExisting,
    [switch]$NoMerge,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Get-LineCount([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    $count = 0
    foreach ($line in [System.IO.File]::ReadLines((Resolve-Path -LiteralPath $Path).Path)) {
        if (-not [string]::IsNullOrWhiteSpace($line)) { $count += 1 }
    }
    return $count
}

function Read-JsonFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Write-JsonFile($Value, [string]$Path) {
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Path -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $InputJsonl)) { throw "InputJsonl not found: $InputJsonl" }
if ($FutureSamples -le 0) { throw "FutureSamples must be positive" }
if ($ChunkSize -le 0) { throw "ChunkSize must be positive" }
if ($StartOffset -lt 0) { throw "StartOffset must be non-negative" }
if ($TotalRows -lt 0) { throw "TotalRows must be non-negative" }
if ($PredictionThreads -lt 0) { throw "PredictionThreads must be non-negative" }
if ($OpeningLookaheadSamples -lt 0) { throw "OpeningLookaheadSamples must be non-negative" }

$resolvedInput = (Resolve-Path -LiteralPath $InputJsonl).Path
$sourceRows = Get-LineCount $resolvedInput
if ($StartOffset -gt $sourceRows) { throw "StartOffset $StartOffset exceeds source rows $sourceRows" }
$rowsToProcess = if ($TotalRows -gt 0) { [math]::Min($TotalRows, $sourceRows - $StartOffset) } else { $sourceRows - $StartOffset }
$chunkCount = if ($rowsToProcess -gt 0) { [int][math]::Ceiling($rowsToProcess / $ChunkSize) } else { 0 }

$chunks = @()
for ($chunk = 0; $chunk -lt $chunkCount; $chunk += 1) {
    $offset = $StartOffset + $chunk * $ChunkSize
    $limit = [math]::Min($ChunkSize, $rowsToProcess - $chunk * $ChunkSize)
    $chunkDir = Join-Path $OutputRoot ("chunk_{0:D4}_offset_{1}_limit_{2}" -f $chunk, $offset, $limit)
    $manifestPath = Join-Path $chunkDir "topk_hard_negative_replay_manifest.json"
    $existingManifest = Read-JsonFile $manifestPath
    $isComplete = $false
    if ($existingManifest -ne $null) {
        $isComplete = ([int]$existingManifest.rows -eq [int]$limit) -and ([int]$existingManifest.samples_written -eq [int]$limit)
    }
    $chunks += [pscustomobject]@{
        chunk = $chunk
        offset = $offset
        limit = $limit
        output_dir = $chunkDir
        manifest = $manifestPath
        complete = $isComplete
    }
}

$dryRunPayload = [pscustomobject]@{
    execution = if ($DryRun) { "dry_run" } else { "run" }
    input_jsonl = $resolvedInput
    output_root = $OutputRoot
    source_rows = $sourceRows
    start_offset = $StartOffset
    rows_to_process = $rowsToProcess
    future_samples = $FutureSamples
    chunk_size = $ChunkSize
    chunk_count = $chunkCount
    replay_seed = $ReplaySeed
    prediction_threads = $PredictionThreads
    opening_lookahead_samples = $OpeningLookaheadSamples
    t3_continuation = $T3Continuation
    skip_existing = [bool]$SkipExisting
    no_merge = [bool]$NoMerge
    chunks = $chunks
    production_p2_fixed = "No-Go"
    t1_training = "No-Go"
    teacher_50k = "No-Go"
}

if ($DryRun) {
    $dryRunPayload | ConvertTo-Json -Depth 8
    exit 0
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
Write-JsonFile $dryRunPayload (Join-Path $OutputRoot "replay_chunks_plan.json")

$completed = @()
foreach ($chunkSpec in $chunks) {
    if ($SkipExisting -and $chunkSpec.complete) {
        $completed += [pscustomobject]@{
            chunk = $chunkSpec.chunk
            offset = $chunkSpec.offset
            limit = $chunkSpec.limit
            output_dir = $chunkSpec.output_dir
            status = "skipped_existing"
        }
        continue
    }

    New-Item -ItemType Directory -Force -Path $chunkSpec.output_dir | Out-Null
    $argsList = @(
        "-m", "ofc_regular.replay_hu_turn2_stage8b_topk_hard_negatives",
        "--input-jsonl", $resolvedInput,
        "--output-dir", $chunkSpec.output_dir,
        "--future-samples", "$FutureSamples",
        "--seed", "$ReplaySeed",
        "--offset", "$($chunkSpec.offset)",
        "--limit", "$($chunkSpec.limit)",
        "--prediction-threads", "$PredictionThreads",
        "--opening-lookahead-samples", "$OpeningLookaheadSamples",
        "--t3-continuation", "$T3Continuation"
    )
    python @argsList
    if ($LASTEXITCODE -ne 0) { throw "Replay chunk $($chunkSpec.chunk) failed with exit code $LASTEXITCODE" }
    $completed += [pscustomobject]@{
        chunk = $chunkSpec.chunk
        offset = $chunkSpec.offset
        limit = $chunkSpec.limit
        output_dir = $chunkSpec.output_dir
        status = "completed"
    }
}

Write-JsonFile ([pscustomobject]@{
    input_jsonl = $resolvedInput
    output_root = $OutputRoot
    source_rows = $sourceRows
    start_offset = $StartOffset
    rows_to_process = $rowsToProcess
    future_samples = $FutureSamples
    chunk_size = $ChunkSize
    chunk_count = $chunkCount
    completed_chunks = $completed
    production_p2_fixed = "No-Go"
    t1_training = "No-Go"
    teacher_50k = "No-Go"
}) (Join-Path $OutputRoot "replay_chunks_manifest.json")

if (-not $NoMerge) {
    $mergedDir = Join-Path $OutputRoot "merged"
    New-Item -ItemType Directory -Force -Path $mergedDir | Out-Null
    $summaryRows = @()
    $teacherOut = Join-Path $mergedDir "topk_hard_negative_replay_teacher.jsonl"
    if (Test-Path -LiteralPath $teacherOut) { Remove-Item -LiteralPath $teacherOut -Force }
    foreach ($chunkSpec in $chunks) {
        $summaryPath = Join-Path $chunkSpec.output_dir "topk_hard_negative_replay_summary.csv"
        if (Test-Path -LiteralPath $summaryPath) {
            $summaryRows += Import-Csv -LiteralPath $summaryPath
        }
        $teacherPath = Join-Path $chunkSpec.output_dir "topk_hard_negative_replay_teacher.jsonl"
        if (Test-Path -LiteralPath $teacherPath) {
            foreach ($line in [System.IO.File]::ReadLines((Resolve-Path -LiteralPath $teacherPath).Path)) {
                if (-not [string]::IsNullOrWhiteSpace($line)) {
                    Add-Content -LiteralPath $teacherOut -Value $line -Encoding UTF8
                }
            }
        }
    }
    $summaryOut = Join-Path $mergedDir "topk_hard_negative_replay_summary.csv"
    if ($summaryRows.Count -gt 0) {
        $summaryRows | Export-Csv -LiteralPath $summaryOut -NoTypeInformation -Encoding UTF8
    } else {
        "" | Set-Content -LiteralPath $summaryOut -Encoding UTF8
    }
    Write-JsonFile ([pscustomobject]@{
        merged_summary = $summaryOut
        merged_teacher = $teacherOut
        merged_summary_rows = $summaryRows.Count
        production_p2_fixed = "No-Go"
        t1_training = "No-Go"
        teacher_50k = "No-Go"
    }) (Join-Path $mergedDir "merged_manifest.json")
}

[pscustomobject]@{
    input_jsonl = $resolvedInput
    output_root = $OutputRoot
    rows_to_process = $rowsToProcess
    chunk_count = $chunkCount
    completed_chunks = $completed.Count
    merged = -not [bool]$NoMerge
    production_p2_fixed = "No-Go"
    t1_training = "No-Go"
    teacher_50k = "No-Go"
} | ConvertTo-Json -Depth 8
