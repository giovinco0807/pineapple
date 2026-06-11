param(
    [int]$TotalGames = 1000,
    [int]$ChunkSize = 50,
    [int]$MaxParallel = 2,
    [int]$Seed = 20260655,
    [double]$HuTurn3MinMargin = 4.0,
    [double]$HuTurn3MaxSelfRegret = [double]::NaN,
    [int]$OpeningLookaheadSamples = 64,
    [int]$PredictionThreads = 1,
    [switch]$RecordAll,
    [string]$HuTurn3Model = "models/hu_turn3_stage1_cycle10_direct_penalty_w025_500_f32_extra_trees.pkl",
    [string]$OpeningModel = "models/opening_stage7_torch_wide.pt",
    [string]$Turn1Model = "models/turn1_stage6_torch_wide.pt",
    [string]$Turn2Model = "models/turn2_stage8.pkl",
    [string]$Turn3Model = "models/turn3_stage6.pkl",
    [string]$OutputDir,
    [string]$MergedDecisionOutput,
    [string]$MergedSummaryOutput
)

$ErrorActionPreference = "Stop"

if (-not $OutputDir) {
    throw "OutputDir is required"
}
if (-not $MergedDecisionOutput) {
    throw "MergedDecisionOutput is required"
}
if (-not $MergedSummaryOutput) {
    throw "MergedSummaryOutput is required"
}
if ($TotalGames -le 0) {
    throw "TotalGames must be positive"
}
if ($ChunkSize -le 0) {
    throw "ChunkSize must be positive"
}
if ($MaxParallel -le 0) {
    throw "MaxParallel must be positive"
}
if ($PredictionThreads -lt 0) {
    throw "PredictionThreads must be non-negative"
}
if ($OpeningLookaheadSamples -lt 0) {
    throw "OpeningLookaheadSamples must be non-negative"
}
if (-not $HuTurn3Model) {
    throw "HuTurn3Model is required"
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

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $null
    }
    $text = Get-Content -LiteralPath $Path -Raw
    if (-not $text.Trim()) {
        return $null
    }
    return $text | ConvertFrom-Json
}

function New-ChunkArgs {
    param($Spec)
    $args = @(
        "-m", "ofc_regular.trace_hu_turn3_overrides",
        "--games", $Spec.Games,
        "--seed", ($Seed + $Spec.StartGame),
        "--hu-turn3-model", $HuTurn3Model,
        "--hu-turn3-min-margin", $HuTurn3MinMargin,
        "--opening-lookahead-samples", $OpeningLookaheadSamples,
        "--prediction-threads", $PredictionThreads,
        "--opening-model", $OpeningModel,
        "--turn1-model", $Turn1Model,
        "--turn2-model", $Turn2Model,
        "--turn3-model", $Turn3Model,
        "--decision-output", $Spec.DecisionPath,
        "--output", $Spec.SummaryPath
    )
    if (-not [double]::IsNaN($HuTurn3MaxSelfRegret)) {
        $args += @("--hu-turn3-max-self-regret", $HuTurn3MaxSelfRegret)
    }
    if ($RecordAll) {
        $args += "--record-all"
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
$chunkCount = [int][math]::Ceiling($TotalGames / $ChunkSize)
$chunkSummaries = @()
$pending = [System.Collections.Generic.Queue[object]]::new()

for ($chunk = 0; $chunk -lt $chunkCount; $chunk += 1) {
    $startGame = $chunk * $ChunkSize
    $gamesThisChunk = [math]::Min($ChunkSize, $TotalGames - $startGame)
    $decisionPath = Join-Path $OutputDir ("hu_turn3_override_trace_chunk_{0:D4}.jsonl" -f $chunk)
    $summaryPath = Join-Path $OutputDir ("hu_turn3_override_trace_chunk_{0:D4}_summary.json" -f $chunk)
    $existingSummary = Read-JsonFile $summaryPath

    if (
        $null -ne $existingSummary -and
        [int]$existingSummary.games -eq $gamesThisChunk -and
        [int]$existingSummary.seed -eq ($Seed + $startGame) -and
        (Test-Path $decisionPath)
    ) {
        $summary = [pscustomobject]@{
            chunk = $chunk
            start_game = $startGame
            games = $gamesThisChunk
            seed = $Seed + $startGame
            decision_lines = Get-LineCount $decisionPath
            override_count = [int]$existingSummary.override_count
            candidate_t3_decision_count = [int]$existingSummary.candidate_t3_decision_count
            paired_score_avg = [double]$existingSummary.paired_score_avg
            seconds = 0.0
            skipped = $true
            decision_output = $decisionPath
            summary_output = $summaryPath
        }
        $chunkSummaries += $summary
        continue
    }

    $pending.Enqueue([pscustomobject]@{
        Chunk = $chunk
        StartGame = $startGame
        Games = $gamesThisChunk
        DecisionPath = $decisionPath
        SummaryPath = $summaryPath
    })
}

$running = @()
while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    while ($pending.Count -gt 0 -and $running.Count -lt $MaxParallel) {
        $spec = $pending.Dequeue()
        $stdout = "$($spec.SummaryPath).stdout.log"
        $stderr = "$($spec.SummaryPath).stderr.log"
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

        if (-not (Test-Path $item.Spec.SummaryPath)) {
            throw "Chunk $($item.Spec.Chunk) did not write summary: $($item.Spec.SummaryPath)"
        }
        if (-not (Test-Path $item.Spec.DecisionPath)) {
            throw "Chunk $($item.Spec.Chunk) did not write decisions: $($item.Spec.DecisionPath)"
        }
        $chunkJson = Read-JsonFile $item.Spec.SummaryPath
        if ([int]$chunkJson.games -ne $item.Spec.Games) {
            throw "Chunk $($item.Spec.Chunk) summary games $($chunkJson.games), expected $($item.Spec.Games)"
        }

        $summary = [pscustomobject]@{
            chunk = $item.Spec.Chunk
            start_game = $item.Spec.StartGame
            games = $item.Spec.Games
            seed = $Seed + $item.Spec.StartGame
            decision_lines = Get-LineCount $item.Spec.DecisionPath
            override_count = [int]$chunkJson.override_count
            candidate_t3_decision_count = [int]$chunkJson.candidate_t3_decision_count
            paired_score_avg = [double]$chunkJson.paired_score_avg
            seconds = [math]::Round(([DateTime]::UtcNow - $item.StartedAt).TotalSeconds, 3)
            skipped = $false
            decision_output = $item.Spec.DecisionPath
            summary_output = $item.Spec.SummaryPath
        }
        $chunkSummaries += $summary
        Add-Content -Path $progressPath -Value ($summary | ConvertTo-Json -Compress)
    }
    $running = $stillRunning
}

foreach ($path in @($MergedDecisionOutput, $MergedSummaryOutput)) {
    $parent = Split-Path -Parent $path
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
}

$tempMergedDecision = "$MergedDecisionOutput.tmp"
if (Test-Path $tempMergedDecision) {
    [System.IO.File]::Delete((Resolve-Path $tempMergedDecision))
}

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$writer = [System.IO.StreamWriter]::new($tempMergedDecision, $false, $utf8NoBom)
try {
    for ($chunk = 0; $chunk -lt $chunkCount; $chunk += 1) {
        $chunkPath = Join-Path $OutputDir ("hu_turn3_override_trace_chunk_{0:D4}.jsonl" -f $chunk)
        if (Test-Path $chunkPath) {
            foreach ($line in [System.IO.File]::ReadLines((Resolve-Path $chunkPath))) {
                $writer.WriteLine($line)
            }
        }
    }
}
finally {
    $writer.Close()
}

Move-Item -Force -Path $tempMergedDecision -Destination $MergedDecisionOutput

$sortedSummaries = @($chunkSummaries | Sort-Object chunk)
$totalOverrideCount = 0
$totalCandidateT3DecisionCount = 0
$weightedPairedScore = 0.0
foreach ($summary in $sortedSummaries) {
    $totalOverrideCount += [int]$summary.override_count
    $totalCandidateT3DecisionCount += [int]$summary.candidate_t3_decision_count
    $weightedPairedScore += [double]$summary.paired_score_avg * [int]$summary.games
}
$totalDecisionLines = Get-LineCount $MergedDecisionOutput
$mergedSummary = [pscustomobject]@{
    total_games = $TotalGames
    total_hands = $TotalGames * 2
    total_decision_lines = $totalDecisionLines
    total_override_count = $totalOverrideCount
    total_candidate_t3_decision_count = $totalCandidateT3DecisionCount
    override_rate = if ($totalCandidateT3DecisionCount -gt 0) { $totalOverrideCount / $totalCandidateT3DecisionCount } else { 0.0 }
    weighted_paired_score_avg = $weightedPairedScore / $TotalGames
    seed = $Seed
    chunk_size = $ChunkSize
    max_parallel = $MaxParallel
    hu_turn3_model = $HuTurn3Model
    hu_turn3_min_margin = $HuTurn3MinMargin
    hu_turn3_max_self_regret = if ([double]::IsNaN($HuTurn3MaxSelfRegret)) { $null } else { $HuTurn3MaxSelfRegret }
    opening_lookahead_samples = $OpeningLookaheadSamples
    prediction_threads = $PredictionThreads
    record_all = [bool]$RecordAll
    decision_output = $MergedDecisionOutput
    chunks = $sortedSummaries
}
$mergedSummary | ConvertTo-Json -Depth 6 | Set-Content -Path $MergedSummaryOutput -Encoding UTF8
$mergedSummary | ConvertTo-Json -Depth 6
