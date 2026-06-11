param(
    [int]$TotalStates = 1000,
    [int]$ChunkSize = 25,
    [int]$MaxParallel = 2,
    [int]$Seed = 20260637,
    [int]$MaxHandsPerChunk = 1000000,
    [int]$OpeningLookaheadSamples = 64,
    [int]$PredictionThreads = 1,
    [string]$SelectionHuTurn3Model = "models/hu_turn3_stage1_cycle10_direct_penalty_w025_500_f32_extra_trees.pkl",
    [double]$SelectionMinMargin = [double]::NaN,
    [double]$SelectionMaxMargin = [double]::NaN,
    [switch]$SelectionRequireDisagreement,
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
if ($TotalStates -le 0) {
    throw "TotalStates must be positive"
}
if ($ChunkSize -le 0) {
    throw "ChunkSize must be positive"
}
if ($MaxParallel -le 0) {
    throw "MaxParallel must be positive"
}
if ($MaxHandsPerChunk -le 0) {
    throw "MaxHandsPerChunk must be positive"
}
if (-not $SelectionHuTurn3Model) {
    throw "SelectionHuTurn3Model is required"
}
if (-not [double]::IsNaN($SelectionMinMargin) -and -not [double]::IsNaN($SelectionMaxMargin) -and $SelectionMinMargin -gt $SelectionMaxMargin) {
    throw "SelectionMinMargin must be <= SelectionMaxMargin"
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
    $args = @(
        "-m", "ofc_regular.mine_hu_turn3_states",
        "--states", $Spec.States,
        "--seed", ($Seed + $Spec.Chunk),
        "--selection-hu-turn3-model", $SelectionHuTurn3Model,
        "--max-hands", $MaxHandsPerChunk,
        "--opening-lookahead-samples", $OpeningLookaheadSamples,
        "--prediction-threads", $PredictionThreads,
        "--opening-model", $OpeningModel,
        "--turn1-model", $Turn1Model,
        "--turn2-model", $Turn2Model,
        "--turn3-model", $Turn3Model,
        "--output", $Spec.Path
    )
    if (-not [double]::IsNaN($SelectionMinMargin)) {
        $args += @("--selection-min-margin", $SelectionMinMargin)
    }
    if (-not [double]::IsNaN($SelectionMaxMargin)) {
        $args += @("--selection-max-margin", $SelectionMaxMargin)
    }
    if ($SelectionRequireDisagreement) {
        $args += "--selection-require-disagreement"
    }
    return $args
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
$chunkCount = [int][math]::Ceiling($TotalStates / $ChunkSize)
$chunkSummaries = @()
$pending = [System.Collections.Generic.Queue[object]]::new()

for ($chunk = 0; $chunk -lt $chunkCount; $chunk += 1) {
    $startState = $chunk * $ChunkSize
    $statesThisChunk = [math]::Min($ChunkSize, $TotalStates - $startState)
    $chunkPath = Join-Path $OutputDir ("hu_turn3_mined_states_chunk_{0:D4}.jsonl" -f $chunk)
    $existingLines = Get-LineCount $chunkPath

    if ($existingLines -eq $statesThisChunk) {
        $summary = [pscustomobject]@{
            chunk = $chunk
            states = $statesThisChunk
            lines = $existingLines
            seconds = 0.0
            skipped = $true
            output = $chunkPath
        }
        $chunkSummaries += $summary
        continue
    }

    $pending.Enqueue([pscustomobject]@{
        Chunk = $chunk
        States = $statesThisChunk
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
            throw "Chunk $($item.Spec.Chunk) wrote $lines lines, expected $($item.Spec.States)"
        }

        $summary = [pscustomobject]@{
            chunk = $item.Spec.Chunk
            states = $item.Spec.States
            lines = $lines
            seconds = [math]::Round(([DateTime]::UtcNow - $item.StartedAt).TotalSeconds, 3)
            skipped = $false
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
    for ($chunk = 0; $chunk -lt $chunkCount; $chunk += 1) {
        $chunkPath = Join-Path $OutputDir ("hu_turn3_mined_states_chunk_{0:D4}.jsonl" -f $chunk)
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
    total_states = $TotalStates
    total_lines = $totalLines
    chunk_size = $ChunkSize
    max_parallel = $MaxParallel
    seed = $Seed
    max_hands_per_chunk = $MaxHandsPerChunk
    opening_lookahead_samples = $OpeningLookaheadSamples
    prediction_threads = $PredictionThreads
    selection_hu_turn3_model = $SelectionHuTurn3Model
    selection_min_margin = if ([double]::IsNaN($SelectionMinMargin)) { $null } else { $SelectionMinMargin }
    selection_max_margin = if ([double]::IsNaN($SelectionMaxMargin)) { $null } else { $SelectionMaxMargin }
    selection_require_disagreement = [bool]$SelectionRequireDisagreement
    output = $MergedOutput
    chunks = $chunkSummaries | Sort-Object chunk
} | ConvertTo-Json -Depth 5
