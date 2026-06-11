param(
    [ValidateSet("t2", "turn1", "opening")]
    [string]$Phase,
    [string]$DownstreamModel,
    [int]$TotalSamples = 1000,
    [int]$ChunkSize = 50,
    [int]$FutureSamples = 64,
    [int]$ActionBatchSize = -1,
    [int]$PredictionThreads = 1,
    [int]$Seed = 20260602,
    [double]$MinScoreGap = 0.0,
    [string]$OutputDir,
    [string]$MergedOutput
)

$ErrorActionPreference = "Stop"

if (-not $OutputDir) {
    throw "OutputDir is required"
}
if (-not $MergedOutput) {
    throw "MergedOutput is required"
}
if (-not (Test-Path $DownstreamModel)) {
    throw "DownstreamModel not found: $DownstreamModel"
}
if ($TotalSamples -le 0) {
    throw "TotalSamples must be positive"
}
if ($ChunkSize -le 0) {
    throw "ChunkSize must be positive"
}
if ($FutureSamples -lt 0) {
    throw "FutureSamples must be non-negative"
}
if ($ActionBatchSize -lt -1) {
    throw "ActionBatchSize must be -1 or greater"
}
if ($PredictionThreads -lt 0) {
    throw "PredictionThreads must be non-negative"
}

function Get-LineCount {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return 0
    }

    $reader = [System.IO.File]::OpenText((Resolve-Path $Path))
    try {
        $count = 0
        while ($null -ne $reader.ReadLine()) {
            $count += 1
        }
        return $count
    }
    finally {
        $reader.Close()
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$srcPath = Join-Path (Get-Location) "src"
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$srcPath;$env:PYTHONPATH"
}
else {
    $env:PYTHONPATH = $srcPath
}
if ($PredictionThreads -gt 0) {
    $env:OMP_NUM_THREADS = [string]$PredictionThreads
    $env:MKL_NUM_THREADS = [string]$PredictionThreads
    $env:OPENBLAS_NUM_THREADS = [string]$PredictionThreads
}

$progressPath = Join-Path $OutputDir "progress.jsonl"
$chunkCount = [int][math]::Ceiling($TotalSamples / $ChunkSize)
$chunkSummaries = @()

for ($chunk = 0; $chunk -lt $chunkCount; $chunk += 1) {
    $startSample = $chunk * $ChunkSize
    $samplesThisChunk = [math]::Min($ChunkSize, $TotalSamples - $startSample)
    $chunkPath = Join-Path $OutputDir ("{0}_chunk_{1:D4}.jsonl" -f $Phase, $chunk)
    $existingLines = Get-LineCount $chunkPath

    if ($existingLines -eq $samplesThisChunk) {
        $summary = [pscustomobject]@{
            chunk = $chunk
            samples = $samplesThisChunk
            lines = $existingLines
            seconds = 0.0
            skipped = $true
            output = $chunkPath
        }
        $chunkSummaries += $summary
        continue
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    if ($Phase -eq "t2") {
        $phaseArgs = @(
            "-m", "ofc_regular.t2_teacher_data",
            "--turn3-model", $DownstreamModel,
            "--samples", $samplesThisChunk,
            "--future-samples", $FutureSamples,
            "--seed", ($Seed + $chunk),
            "--output", $chunkPath
        )
        if ($ActionBatchSize -ge 0) {
            $phaseArgs += @("--action-batch-size", $ActionBatchSize)
        }
        python @phaseArgs | Out-Null
    }
    else {
        $phaseArgs = @(
            "-m", "ofc_regular.early_teacher_data",
            "--phase", $Phase,
            "--downstream-model", $DownstreamModel,
            "--samples", $samplesThisChunk,
            "--future-samples", $FutureSamples,
            "--seed", ($Seed + $chunk),
            "--min-score-gap", $MinScoreGap,
            "--output", $chunkPath
        )
        if ($ActionBatchSize -ge 0) {
            $phaseArgs += @("--action-batch-size", $ActionBatchSize)
        }
        python @phaseArgs | Out-Null
    }
    $sw.Stop()

    $lines = Get-LineCount $chunkPath
    if ($lines -ne $samplesThisChunk) {
        throw "Chunk $chunk wrote $lines lines, expected $samplesThisChunk"
    }

    $summary = [pscustomobject]@{
        chunk = $chunk
        samples = $samplesThisChunk
        lines = $lines
        seconds = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        skipped = $false
        output = $chunkPath
    }
    $chunkSummaries += $summary
    Add-Content -Path $progressPath -Value ($summary | ConvertTo-Json -Compress)
}

$mergedParent = Split-Path -Parent $MergedOutput
if ($mergedParent) {
    New-Item -ItemType Directory -Force -Path $mergedParent | Out-Null
}

$tempMerged = "$MergedOutput.tmp"
if (Test-Path $tempMerged) {
    [System.IO.File]::Delete((Resolve-Path $tempMerged))
}

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$writer = [System.IO.StreamWriter]::new($tempMerged, $false, $utf8NoBom)
try {
    for ($chunk = 0; $chunk -lt $chunkCount; $chunk += 1) {
        $chunkPath = Join-Path $OutputDir ("{0}_chunk_{1:D4}.jsonl" -f $Phase, $chunk)
        foreach ($line in [System.IO.File]::ReadLines((Resolve-Path $chunkPath))) {
            $writer.WriteLine($line)
        }
    }
}
finally {
    $writer.Close()
}

Move-Item -Force -Path $tempMerged -Destination $MergedOutput
$totalLines = Get-LineCount $MergedOutput

[pscustomobject]@{
    phase = $Phase
    total_samples = $TotalSamples
    total_lines = $totalLines
    chunk_size = $ChunkSize
    future_samples = $FutureSamples
    action_batch_size = $ActionBatchSize
    prediction_threads = $PredictionThreads
    downstream_model = $DownstreamModel
    output = $MergedOutput
    chunks = $chunkSummaries
} | ConvertTo-Json -Depth 5
