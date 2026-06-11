param(
    [ValidateSet("t2", "turn1", "opening")]
    [string]$Phase,
    [string]$DownstreamModel,
    [int]$TotalSamples = 1000,
    [int]$ChunkSize = 50,
    [int]$FutureSamples = 64,
    [int]$ActionBatchSize = -1,
    [int]$MaxParallel = 2,
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
if ($MaxParallel -le 0) {
    throw "MaxParallel must be positive"
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

function New-ChunkArgs {
    param($Spec)

    if ($Phase -eq "t2") {
        $args = @(
            "-m", "ofc_regular.t2_teacher_data",
            "--turn3-model", $DownstreamModel,
            "--samples", $Spec.Samples,
            "--future-samples", $FutureSamples,
            "--seed", ($Seed + $Spec.Chunk),
            "--output", $Spec.Path
        )
    }
    else {
        $args = @(
            "-m", "ofc_regular.early_teacher_data",
            "--phase", $Phase,
            "--downstream-model", $DownstreamModel,
            "--samples", $Spec.Samples,
            "--future-samples", $FutureSamples,
            "--seed", ($Seed + $Spec.Chunk),
            "--min-score-gap", $MinScoreGap,
            "--output", $Spec.Path
        )
    }

    if ($ActionBatchSize -ge 0) {
        $args += @("--action-batch-size", $ActionBatchSize)
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
if ($PredictionThreads -gt 0) {
    $env:OMP_NUM_THREADS = [string]$PredictionThreads
    $env:MKL_NUM_THREADS = [string]$PredictionThreads
    $env:OPENBLAS_NUM_THREADS = [string]$PredictionThreads
}

$progressPath = Join-Path $OutputDir "progress.jsonl"
$chunkCount = [int][math]::Ceiling($TotalSamples / $ChunkSize)
$chunkSummaries = @()
$pending = [System.Collections.Generic.Queue[object]]::new()

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

    $pending.Enqueue([pscustomobject]@{
        Chunk = $chunk
        Samples = $samplesThisChunk
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
        if ($lines -ne $item.Spec.Samples) {
            throw "Chunk $($item.Spec.Chunk) wrote $lines lines, expected $($item.Spec.Samples)"
        }

        $summary = [pscustomobject]@{
            chunk = $item.Spec.Chunk
            samples = $item.Spec.Samples
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
    max_parallel = $MaxParallel
    prediction_threads = $PredictionThreads
    downstream_model = $DownstreamModel
    output = $MergedOutput
    chunks = $chunkSummaries | Sort-Object chunk
} | ConvertTo-Json -Depth 5
