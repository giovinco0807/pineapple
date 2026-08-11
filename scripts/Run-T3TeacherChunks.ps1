param(
    [int]$TotalSamples = 10000,
    [int]$ChunkSize = 500,
    [int]$FutureSamples = 64,
    [int]$Seed = 20260602,
    # configs/fl_ev_regular_v3_direct2.json (armA, 2026-08-03).
    [double]$FlEv = 9.109,
    [double]$MinScoreGap = 0.25,
    [string]$OutputDir = "outputs/t3_stage1_chunks",
    [string]$MergedOutput = "outputs/turn3_teacher_stage1_10k_f64.jsonl"
)

$ErrorActionPreference = "Stop"

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

$exe = Join-Path (Get-Location) "target/release/regular_fl_solver.exe"
if (-not (Test-Path $exe)) {
    cargo build --release
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$progressPath = Join-Path $OutputDir "progress.jsonl"
$chunkCount = [int][math]::Ceiling($TotalSamples / $ChunkSize)
$chunkSummaries = @()

for ($chunk = 0; $chunk -lt $chunkCount; $chunk += 1) {
    $startSample = $chunk * $ChunkSize
    $samplesThisChunk = [math]::Min($ChunkSize, $TotalSamples - $startSample)
    $chunkPath = Join-Path $OutputDir ("turn3_chunk_{0:D4}.jsonl" -f $chunk)
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
    & $exe `
        --teacher-output $chunkPath `
        --teacher-samples $samplesThisChunk `
        --future-samples $FutureSamples `
        --seed ($Seed + $chunk) `
        --teacher-fl-ev $FlEv `
        --teacher-min-score-gap $MinScoreGap | Out-Null
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
        $chunkPath = Join-Path $OutputDir ("turn3_chunk_{0:D4}.jsonl" -f $chunk)
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
    output = $MergedOutput
    chunks = $chunkSummaries
} | ConvertTo-Json -Depth 5
