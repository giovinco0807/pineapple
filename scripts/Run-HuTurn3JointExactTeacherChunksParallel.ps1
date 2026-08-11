param(
    [string]$InputDir,
    [string]$OutputDir,
    [string]$MergedOutput,
    [int]$CandidateSamples = 16,
    [int]$EvaluationSamples = 32,
    [int]$DownstreamT3Samples = 4,
    [int]$DownstreamT4Samples = 16,
    [int]$MaxParallel = 2,
    [int]$Seed = 2026071303,
    [int]$CandidateSeed = 2026071301,
    [int]$EvaluationSeed = 2026071302,
    [string]$RunId = "hu-m2-t3-sequential-v1",
    [string]$InputFilter = "hu_turn3_mined_states_chunk_*.jsonl",
    [switch]$DisableFinalTurnCache
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
if ($CandidateSamples -le 0 -or $EvaluationSamples -le 0 -or $DownstreamT3Samples -le 0) {
    throw "CandidateSamples, EvaluationSamples, and DownstreamT3Samples must be positive"
}
if ($DownstreamT4Samples -lt 0) {
    throw "DownstreamT4Samples must be non-negative (0 means exact T4-first marginal)"
}
if ($MaxParallel -le 0) {
    throw "MaxParallel must be positive"
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
        "-m", "ofc_regular.hu_turn3_joint_exact_teacher",
        "--input", $Spec.Input,
        "--output", $Spec.Path,
        "--summary-csv", $Spec.SummaryCsv,
        "--candidate-samples", $CandidateSamples,
        "--evaluation-samples", $EvaluationSamples,
        "--downstream-t3-samples", $DownstreamT3Samples,
        "--downstream-t4-samples", $DownstreamT4Samples,
        "--seed", $Seed,
        "--candidate-seed", $CandidateSeed,
        "--evaluation-seed", $EvaluationSeed,
        "--run-id", $RunId
    )
    if ($DisableFinalTurnCache) {
        $args += "--disable-final-turn-cache"
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

$inputChunks = Get-ChildItem -LiteralPath $InputDir -Filter $InputFilter |
    Sort-Object Name
if (-not $inputChunks) {
    throw "No input chunks matching $InputFilter found in $InputDir"
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
    $chunkPath = Join-Path $OutputDir ("hu_turn3_sequential_belief_v2_chunk_{0:D4}.jsonl" -f $index)
    $summaryCsv = Join-Path $OutputDir ("hu_turn3_sequential_belief_v2_chunk_{0:D4}_summary.csv" -f $index)
    $existingLines = Get-LineCount $chunkPath

    if ($existingLines -eq $expectedLines) {
        $firstRecord = Get-Content -LiteralPath $chunkPath -TotalCount 1 | ConvertFrom-Json
        if ($firstRecord.schema -ne "hu_turn3_sequential_belief_v2") {
            throw "Existing chunk has an unsafe/obsolete schema and will not be resumed: $chunkPath"
        }
        $summary = [pscustomobject]@{
            chunk = $index
            states = $expectedLines
            samples = $existingLines
            seconds = 0.0
            skipped = $true
            input = $inputChunk.FullName
            output = $chunkPath
            summary_csv = $summaryCsv
        }
        $chunkSummaries += $summary
        continue
    }

    $pending.Enqueue([pscustomobject]@{
        Chunk = $index
        States = $expectedLines
        Input = $inputChunk.FullName
        Path = $chunkPath
        SummaryCsv = $summaryCsv
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
            throw "Chunk $($item.Spec.Chunk) wrote $lines lines, expected $($item.Spec.States)."
        }

        $summary = [pscustomobject]@{
            chunk = $item.Spec.Chunk
            states = $item.Spec.States
            samples = $lines
            seconds = [math]::Round(([DateTime]::UtcNow - $item.StartedAt).TotalSeconds, 3)
            skipped = $false
            input = $item.Spec.Input
            output = $item.Spec.Path
            summary_csv = $item.Spec.SummaryCsv
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
        $chunkPath = Join-Path $OutputDir ("hu_turn3_sequential_belief_v2_chunk_{0:D4}.jsonl" -f $index)
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
    input_filter = $InputFilter
    output_dir = $OutputDir
    total_chunks = $inputChunks.Count
    total_lines = $totalLines
    schema = "hu_turn3_sequential_belief_v2"
    candidate_samples = $CandidateSamples
    evaluation_samples = $EvaluationSamples
    downstream_t3_samples = $DownstreamT3Samples
    downstream_t4_samples = $DownstreamT4Samples
    max_parallel = $MaxParallel
    seed = $Seed
    candidate_seed = $CandidateSeed
    evaluation_seed = $EvaluationSeed
    run_id = $RunId
    disable_final_turn_cache = [bool]$DisableFinalTurnCache
    output = $MergedOutput
    chunks = $chunkSummaries | Sort-Object chunk
} | ConvertTo-Json -Depth 5
