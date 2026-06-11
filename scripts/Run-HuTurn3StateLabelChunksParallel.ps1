param(
    [string]$InputDir,
    [string]$OutputDir,
    [string]$MergedOutput,
    [int]$FutureSamples = 32,
    [int]$MaxParallel = 2,
    [int]$Seed = 20260638,
    [int]$OpeningLookaheadSamples = 64,
    [double]$MinScoreGap = 0.0,
    [double]$SelfRegretPenaltyWeight = 0.25,
    [double]$SelfRegretFree = 0.0,
    [string]$OpeningModel = "models/opening_stage7_torch_wide.pt",
    [string]$Turn1Model = "models/turn1_stage6_torch_wide.pt",
    [string]$Turn2Model = "models/turn2_stage8.pkl",
    [string]$Turn3Model = "models/turn3_stage6.pkl"
)

$ErrorActionPreference = "Stop"

if (-not $InputDir) {
    throw "InputDir is required"
}
if (-not (Test-Path $InputDir)) {
    throw "InputDir does not exist: $InputDir"
}
if (-not $OutputDir) {
    throw "OutputDir is required"
}
if (-not $MergedOutput) {
    throw "MergedOutput is required"
}
if ($FutureSamples -lt 0) {
    throw "FutureSamples must be non-negative"
}
if ($MaxParallel -le 0) {
    throw "MaxParallel must be positive"
}
if ($SelfRegretPenaltyWeight -lt 0) {
    throw "SelfRegretPenaltyWeight must be non-negative"
}
if ($SelfRegretFree -lt 0) {
    throw "SelfRegretFree must be non-negative"
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

function New-ChunkArgs {
    param($Spec)
    return @(
        "-m", "ofc_regular.label_hu_turn3_states",
        "--input", $Spec.Input,
        "--output", $Spec.Path,
        "--seed", ($Seed + $Spec.Chunk),
        "--future-samples", $FutureSamples,
        "--opening-lookahead-samples", $OpeningLookaheadSamples,
        "--min-score-gap", $MinScoreGap,
        "--self-regret-penalty-weight", $SelfRegretPenaltyWeight,
        "--self-regret-free", $SelfRegretFree,
        "--opening-model", $OpeningModel,
        "--turn1-model", $Turn1Model,
        "--turn2-model", $Turn2Model,
        "--turn3-model", $Turn3Model
    )
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$srcPath = Join-Path (Get-Location) "src"
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$srcPath;$env:PYTHONPATH"
}
else {
    $env:PYTHONPATH = $srcPath
}

$inputChunks = Get-ChildItem -LiteralPath $InputDir -Filter "hu_turn3_mined_states_chunk_*.jsonl" |
    Sort-Object Name
if (-not $inputChunks) {
    throw "No mined state chunks found in $InputDir"
}

$progressPath = Join-Path $OutputDir "progress.jsonl"
$chunkSummaries = @()
$pending = [System.Collections.Generic.Queue[object]]::new()

for ($index = 0; $index -lt $inputChunks.Count; $index += 1) {
    $inputChunk = $inputChunks[$index]
    $expectedLines = Get-LineCount $inputChunk.FullName
    if ($expectedLines -le 0) {
        throw "Input chunk has no lines: $($inputChunk.FullName)"
    }
    $chunkPath = Join-Path $OutputDir ("hu_turn3_labeled_teacher_chunk_{0:D4}.jsonl" -f $index)
    $existingLines = Get-LineCount $chunkPath

    if ($existingLines -eq $expectedLines) {
        $summary = [pscustomobject]@{
            chunk = $index
            states = $expectedLines
            samples = $existingLines
            seconds = 0.0
            skipped = $true
            input = $inputChunk.FullName
            output = $chunkPath
        }
        $chunkSummaries += $summary
        continue
    }

    $pending.Enqueue([pscustomobject]@{
        Chunk = $index
        States = $expectedLines
        Input = $inputChunk.FullName
        Path = $chunkPath
    })
}

$running = @()
while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    while ($pending.Count -gt 0 -and $running.Count -lt $MaxParallel) {
        $spec = $pending.Dequeue()
        $stdout = "$($spec.Path).stdout.log"
        $stderr = "$($spec.Path).stderr.log"
        if (Test-Path $stdout) {
            Remove-Item -LiteralPath $stdout -Force
        }
        if (Test-Path $stderr) {
            Remove-Item -LiteralPath $stderr -Force
        }

        $args = New-ChunkArgs $spec
        $process = Start-Process `
            -FilePath "python" `
            -ArgumentList $args `
            -WorkingDirectory (Get-Location).Path `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -PassThru

        $running += [pscustomobject]@{
            Spec = $spec
            Process = $process
            StartedAt = [DateTime]::UtcNow
            Stdout = $stdout
            Stderr = $stderr
        }
    }

    Start-Sleep -Seconds 1
    $stillRunning = @()
    foreach ($item in $running) {
        $item.Process.Refresh()
        if (-not $item.Process.HasExited) {
            $stillRunning += $item
            continue
        }

        $item.Process.WaitForExit()
        $exitCode = $item.Process.ExitCode
        if ($null -ne $exitCode -and $exitCode -ne 0) {
            $stderrText = ""
            if (Test-Path $item.Stderr) {
                $stderrText = Get-Content -LiteralPath $item.Stderr -Raw
            }
            throw "Chunk $($item.Spec.Chunk) failed with exit code ${exitCode}: $stderrText"
        }

        $lines = Get-LineCount $item.Spec.Path
        if ($lines -ne $item.Spec.States) {
            throw "Chunk $($item.Spec.Chunk) wrote $lines lines, expected $($item.Spec.States). Use MinScoreGap 0 for resumable one-to-one chunk labeling."
        }

        $summary = [pscustomobject]@{
            chunk = $item.Spec.Chunk
            states = $item.Spec.States
            samples = $lines
            seconds = [math]::Round(([DateTime]::UtcNow - $item.StartedAt).TotalSeconds, 3)
            skipped = $false
            input = $item.Spec.Input
            output = $item.Spec.Path
        }
        $chunkSummaries += $summary
        Add-Content -Path $progressPath -Value ($summary | ConvertTo-Json -Compress)
    }
    $running = $stillRunning
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
    for ($index = 0; $index -lt $inputChunks.Count; $index += 1) {
        $chunkPath = Join-Path $OutputDir ("hu_turn3_labeled_teacher_chunk_{0:D4}.jsonl" -f $index)
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
    input_dir = $InputDir
    output_dir = $OutputDir
    total_chunks = $inputChunks.Count
    total_lines = $totalLines
    future_samples = $FutureSamples
    max_parallel = $MaxParallel
    seed = $Seed
    opening_lookahead_samples = $OpeningLookaheadSamples
    min_score_gap = $MinScoreGap
    self_regret_penalty_weight = $SelfRegretPenaltyWeight
    self_regret_free = $SelfRegretFree
    output = $MergedOutput
    chunks = $chunkSummaries | Sort-Object chunk
} | ConvertTo-Json -Depth 5
