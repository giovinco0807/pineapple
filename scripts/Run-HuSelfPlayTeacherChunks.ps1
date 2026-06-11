param(
    [int]$TotalSamples = 1000,
    [int]$ChunkSize = 50,
    [int]$FutureSamples = 8,
    [int]$Seed = 20260602,
    [int]$OpeningLookaheadSamples = 64,
    [double]$MinScoreGap = 0.0,
    [string]$OpeningModel = "models/opening_stage7_torch_wide.pt",
    [string]$Turn1Model = "models/turn1_stage6_torch_wide.pt",
    [string]$Turn2Model = "models/turn2_stage8.pkl",
    [string]$Turn3Model = "models/turn3_stage6.pkl",
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
if ($TotalSamples -le 0) {
    throw "TotalSamples must be positive"
}
if ($ChunkSize -le 0) {
    throw "ChunkSize must be positive"
}
if ($FutureSamples -lt 0) {
    throw "FutureSamples must be non-negative"
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

$progressPath = Join-Path $OutputDir "progress.jsonl"
$chunkCount = [int][math]::Ceiling($TotalSamples / $ChunkSize)
$chunkSummaries = @()

for ($chunk = 0; $chunk -lt $chunkCount; $chunk += 1) {
    $startSample = $chunk * $ChunkSize
    $samplesThisChunk = [math]::Min($ChunkSize, $TotalSamples - $startSample)
    $chunkPath = Join-Path $OutputDir ("hu_turn3_selfplay_chunk_{0:D4}.jsonl" -f $chunk)
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
    python -m ofc_regular.hu_self_play_teacher_data `
        --samples $samplesThisChunk `
        --future-samples $FutureSamples `
        --seed ($Seed + $chunk) `
        --opening-lookahead-samples $OpeningLookaheadSamples `
        --min-score-gap $MinScoreGap `
        --opening-model $OpeningModel `
        --turn1-model $Turn1Model `
        --turn2-model $Turn2Model `
        --turn3-model $Turn3Model `
        --output $chunkPath | Out-Null
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
        $chunkPath = Join-Path $OutputDir ("hu_turn3_selfplay_chunk_{0:D4}.jsonl" -f $chunk)
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
    total_samples = $TotalSamples
    total_lines = $totalLines
    chunk_size = $ChunkSize
    future_samples = $FutureSamples
    opening_lookahead_samples = $OpeningLookaheadSamples
    output = $MergedOutput
    chunks = $chunkSummaries
} | ConvertTo-Json -Depth 5
