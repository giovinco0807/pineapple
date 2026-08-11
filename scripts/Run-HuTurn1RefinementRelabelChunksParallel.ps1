param(
    [string]$TargetInput = "outputs/hu_turn1_stage1_pilot/refinement_targets/stage1_remaining_targets_1800.jsonl",
    [string]$OutputDir = "outputs/hu_turn1_stage1_pilot/refinement_targets/stage9f_p2_remaining1800_relabel_throttle6",
    [int]$TotalTargets = 1800,
    [int]$ChunkSize = 20,
    [int]$StartPart = 0,
    [int]$EndPart = -1,
    [int]$MaxParallel = 6,
    [int]$Seed = 2026062702,
    [string]$Profile = "stage9f_p2",
    [string]$OpponentProfile = "stage9f_p2",
    [int]$FutureSamples = 1,
    [int]$MaxActions = 0,
    [string]$CandidateModel = "",
    [string[]]$CandidateModels = @(),
    [int]$CandidateTopk = 0,
    [int]$CandidateUnionCap = 0,
    [string]$CandidateUnionMode = "min_rank",
    [int]$OpeningLookaheadSamples = 1,
    [double]$TargetTimeoutSeconds = 0,
    [int]$PartTimeoutSeconds = 0,
    [switch]$ContinueOnTargetError,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"

if ($ChunkSize -le 0) { throw "ChunkSize must be positive" }
if ($MaxParallel -le 0) { throw "MaxParallel must be positive" }
if ($TotalTargets -le 0) { throw "TotalTargets must be positive" }
if ($FutureSamples -le 0) { throw "FutureSamples must be positive" }
if ($MaxActions -lt 0) { throw "MaxActions must be non-negative" }
if ($CandidateTopk -lt 0) { throw "CandidateTopk must be non-negative" }
if ($CandidateUnionCap -lt 0) { throw "CandidateUnionCap must be non-negative" }
if ($CandidateUnionMode -notin @("min_rank", "rank_sum", "reciprocal_rank_sum", "mean_score", "max_score", "mean_z_score", "max_z_score")) {
    throw "CandidateUnionMode is invalid: $CandidateUnionMode"
}
if ($CandidateModel -and $CandidateModels.Count -gt 0) { throw "CandidateModel and CandidateModels are mutually exclusive" }
if ($CandidateModel -and $CandidateTopk -le 0) { throw "CandidateTopk must be positive when CandidateModel is set" }
if ($CandidateModels.Count -gt 0 -and $CandidateTopk -le 0) { throw "CandidateTopk must be positive when CandidateModels is set" }
if ($OpeningLookaheadSamples -lt 0) { throw "OpeningLookaheadSamples must be non-negative" }
if ($TargetTimeoutSeconds -lt 0) { throw "TargetTimeoutSeconds must be non-negative" }
if ($PartTimeoutSeconds -lt 0) { throw "PartTimeoutSeconds must be non-negative" }
if ($PollSeconds -le 0) { throw "PollSeconds must be positive" }

$totalParts = [int][Math]::Ceiling($TotalTargets / [double]$ChunkSize)
if ($EndPart -lt 0 -or $EndPart -gt $totalParts) { $EndPart = $totalParts }
if ($StartPart -lt 0 -or $StartPart -ge $EndPart) { throw "Invalid part range: [$StartPart, $EndPart)" }

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$jobsPath = Join-Path $OutputDir "jobs_parallel.jsonl"
$progressPath = Join-Path $OutputDir "progress_parallel.jsonl"

function New-PartPaths([int]$PartIndex) {
    $partName = "part{0:D3}" -f $PartIndex
    return [pscustomobject]@{
        Part = $partName
        Output = Join-Path $OutputDir ($partName + ".jsonl")
        Skipped = Join-Path $OutputDir ($partName + "_skipped.jsonl")
        Summary = Join-Path $OutputDir ($partName + "_summary.json")
        Stdout = Join-Path $OutputDir ($partName + ".stdout.log")
        Stderr = Join-Path $OutputDir ($partName + ".stderr.log")
    }
}

function Write-JsonLine($Path, $Object) {
    $Object | ConvertTo-Json -Compress -Depth 6 | Add-Content -Encoding UTF8 $Path
}

function Start-Part([int]$PartIndex) {
    $paths = New-PartPaths $PartIndex
    $offset = $PartIndex * $ChunkSize
    $count = [Math]::Min($ChunkSize, $TotalTargets - $offset)
    if ($count -le 0) { return $null }
    if (Test-Path $paths.Summary) {
        return [pscustomobject]@{
            Index = $PartIndex
            Offset = $offset
            Count = $count
            Process = $null
            Paths = $paths
            Status = "already_done"
        }
    }

    Remove-Item -Force -ErrorAction SilentlyContinue $paths.Output, $paths.Skipped, $paths.Stdout, $paths.Stderr
    $argsList = @(
        "-m", "ofc_regular.relabel_hu_turn1_refinement_targets",
        "--input", $TargetInput,
        "--output", $paths.Output,
        "--summary-output", $paths.Summary,
        "--skip-output", $paths.Skipped,
        "--profile", $Profile,
        "--opponent-profile", $OpponentProfile,
        "--future-samples", "$FutureSamples",
        "--max-actions", "$MaxActions",
        "--skip-targets", "$offset",
        "--max-targets", "$count",
        "--seed", "$Seed",
        "--opening-lookahead-samples", "$OpeningLookaheadSamples",
        "--target-timeout-seconds", "$TargetTimeoutSeconds"
    )
    if ($CandidateModel) {
        $argsList += @("--candidate-model", $CandidateModel, "--candidate-topk", "$CandidateTopk")
    }
    if ($CandidateModels.Count -gt 0) {
        $argsList += "--candidate-models"
        foreach ($candidatePath in $CandidateModels) {
            if ($candidatePath) { $argsList += $candidatePath }
        }
        $argsList += @("--candidate-topk", "$CandidateTopk")
    }
    if ($CandidateUnionCap -gt 0) {
        $argsList += @("--candidate-union-cap", "$CandidateUnionCap")
    }
    if ($CandidateUnionMode) {
        $argsList += @("--candidate-union-mode", $CandidateUnionMode)
    }
    if ($ContinueOnTargetError) {
        $argsList += "--continue-on-target-error"
    }
    $safeArgsList = @()
    for ($argIndex = 0; $argIndex -lt $argsList.Count; $argIndex++) {
        $arg = $argsList[$argIndex]
        if ($null -eq $arg -or [string]$arg -eq "") {
            throw "empty relabel argument at index $argIndex for part $PartIndex"
        }
        $safeArgsList += [string]$arg
    }
    $process = Start-Process `
        -FilePath python `
        -ArgumentList $safeArgsList `
        -WorkingDirectory (Get-Location) `
        -WindowStyle Hidden `
        -RedirectStandardOutput $paths.Stdout `
        -RedirectStandardError $paths.Stderr `
        -PassThru

    $row = [pscustomobject]@{
        event = "started"
        part_index = $PartIndex
        offset = $offset
        count = $count
        pid = $process.Id
        output = $paths.Output
        skipped = $paths.Skipped
        summary = $paths.Summary
        stderr = $paths.Stderr
        part_timeout_seconds = $PartTimeoutSeconds
        timestamp = (Get-Date).ToString("o")
    }
    Write-JsonLine $jobsPath $row
    return [pscustomobject]@{
        Index = $PartIndex
        Offset = $offset
        Count = $count
        Process = $process
        Paths = $paths
        StartedAt = Get-Date
        Status = "running"
    }
}

$active = New-Object System.Collections.Generic.List[object]
$nextPart = $StartPart
$completed = 0
$failed = 0
$skipped = 0

while ($nextPart -lt $EndPart -and $active.Count -lt $MaxParallel) {
    $job = Start-Part $nextPart
    $nextPart += 1
    if ($null -eq $job) { continue }
    if ($job.Status -eq "already_done") {
        $skipped += 1
        $completed += 1
    }
    else {
        $active.Add($job)
    }
}

while ($active.Count -gt 0 -or $nextPart -lt $EndPart) {
    Start-Sleep -Seconds $PollSeconds

    for ($i = $active.Count - 1; $i -ge 0; $i--) {
        $job = $active[$i]
        $process = Get-Process -Id $job.Process.Id -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            if ($PartTimeoutSeconds -gt 0) {
                $elapsed = ((Get-Date) - $job.StartedAt).TotalSeconds
                if ($elapsed -ge $PartTimeoutSeconds) {
                    Stop-Process -Id $job.Process.Id -Force -ErrorAction SilentlyContinue
                    $failed += 1
                    Write-JsonLine $progressPath ([pscustomobject]@{
                        event = "part_timeout"
                        part_index = $job.Index
                        offset = $job.Offset
                        count = $job.Count
                        elapsed_seconds = [Math]::Round($elapsed, 3)
                        part_timeout_seconds = $PartTimeoutSeconds
                        timestamp = (Get-Date).ToString("o")
                    })
                    $active.RemoveAt($i)
                }
            }
            continue
        }

        $summaryExists = Test-Path $job.Paths.Summary
        $summary = $null
        if ($summaryExists) {
            $summary = Get-Content -LiteralPath $job.Paths.Summary -Raw | ConvertFrom-Json
        }
        $stderrLength = 0
        if (Test-Path $job.Paths.Stderr) {
            $stderrLength = (Get-Item $job.Paths.Stderr).Length
        }
        $lineCount = 0
        if (Test-Path $job.Paths.Output) {
            $lineCount = (Get-Content $job.Paths.Output | Measure-Object -Line).Lines
        }
        $targetAttempts = if ($summary -and ($summary.PSObject.Properties.Name -contains "target_attempts")) { [int]$summary.target_attempts } else { $lineCount }
        if ($summaryExists -and $targetAttempts -eq $job.Count) {
            $completed += 1
            $status = "completed"
        }
        else {
            $failed += 1
            $status = "failed"
        }
        Write-JsonLine $progressPath ([pscustomobject]@{
            event = $status
            part_index = $job.Index
            offset = $job.Offset
            count = $job.Count
            lines = $lineCount
            target_attempts = $targetAttempts
            skipped_targets = if ($summary -and ($summary.PSObject.Properties.Name -contains "skipped_targets")) { [int]$summary.skipped_targets } else { 0 }
            summary_exists = $summaryExists
            stderr_length = $stderrLength
            timestamp = (Get-Date).ToString("o")
        })
        $active.RemoveAt($i)
    }

    while ($nextPart -lt $EndPart -and $active.Count -lt $MaxParallel) {
        $job = Start-Part $nextPart
        $nextPart += 1
        if ($null -eq $job) { continue }
        if ($job.Status -eq "already_done") {
            $skipped += 1
            $completed += 1
        }
        else {
            $active.Add($job)
        }
    }

    $doneSummaries = (Get-ChildItem $OutputDir -Filter "part*_summary.json" | Measure-Object).Count
    Write-Host ("progress parts={0}/{1} active={2} completed_session={3} failed_session={4} skipped_session={5}" -f $doneSummaries, ($EndPart - $StartPart), $active.Count, $completed, $failed, $skipped)
}

$summary = [pscustomobject]@{
    schema = "hu_turn1_stage1_refinement_relabel_parallel_run_v1"
    input = $TargetInput
    output_dir = $OutputDir
    total_targets = $TotalTargets
    chunk_size = $ChunkSize
    start_part = $StartPart
    end_part = $EndPart
    max_parallel = $MaxParallel
    target_timeout_seconds = $TargetTimeoutSeconds
    part_timeout_seconds = $PartTimeoutSeconds
    continue_on_target_error = [bool]$ContinueOnTargetError
    completed_session = $completed
    failed_session = $failed
    skipped_session = $skipped
    summary_count = (Get-ChildItem $OutputDir -Filter "part*_summary.json" | Measure-Object).Count
    timestamp = (Get-Date).ToString("o")
}
$summaryPath = Join-Path $OutputDir "parallel_run_summary.json"
$summary | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $summaryPath
$summary | ConvertTo-Json -Depth 6
