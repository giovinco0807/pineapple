param(
    [int]$GamesPerSeed = 3500,
    [int]$TargetRealizedOverridesPerSeed = 50,
    [string]$Seeds = "2026063701,2026063702,2026063703",
    [int]$SeedStride = 1000000,
    [string]$Configs = "k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta",
    [string]$Model = "models\hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt",
    [string]$OutputDir = "outputs\evals\hu_turn2_current_fl_ev_stage8c_base_mixed5k_firstseat_c4",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [int]$PredictionThreads = 1,
    [int]$OpeningLookaheadSamples = 64,
    [int]$ProgressEvery = 100,
    [ValidateSet("stage3_reference_default", "stage7_m5_r10")]
    [string]$T3Continuation = "stage3_reference_default",
    [string]$AggregateOutputDir = "",
    [switch]$WaitForGpuClear,
    [int]$GpuWaitTimeoutSeconds = 0,
    [int]$GpuPollSeconds = 60,
    [switch]$AllowConcurrentPython,
    [switch]$NoDecisionLog,
    [switch]$NoAggregate,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ($GamesPerSeed -le 0) { throw "GamesPerSeed must be positive" }
if ($TargetRealizedOverridesPerSeed -le 0) { throw "TargetRealizedOverridesPerSeed must be positive" }
if ($SeedStride -le 0) { throw "SeedStride must be positive" }
if ($PredictionThreads -lt 0) { throw "PredictionThreads must be non-negative" }
if ($OpeningLookaheadSamples -lt 0) { throw "OpeningLookaheadSamples must be non-negative" }
if ($ProgressEvery -le 0) { throw "ProgressEvery must be positive" }
if ($GpuWaitTimeoutSeconds -lt 0) { throw "GpuWaitTimeoutSeconds must be non-negative" }
if ($GpuPollSeconds -le 0) { throw "GpuPollSeconds must be positive" }
if ($NoDecisionLog -and -not $NoAggregate) {
    throw "Aggregate output requires runtime_decisions.jsonl; use -NoAggregate when -NoDecisionLog is set"
}
if (-not $AggregateOutputDir) {
    $AggregateOutputDir = "${OutputDir}_aggregate"
}

function Get-ConcurrentGpuPythonJobs {
    return @(
        Get-CimInstance Win32_Process -Filter "name = 'python.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            $cmd = [string]$_.CommandLine
            $cmd -and
            $cmd -match '(?i)(--device\s+cuda|device=cuda|\bcuda\b)' -and
            $cmd -match '(?i)(ofc_regular|ai\.tutor|ofc-pineapple|regular-ofc-pineapple)'
        } |
        Select-Object ProcessId, CommandLine
    )
}

$usesGpu = $Device -ne "cpu"
$concurrentGpuPython = @(Get-ConcurrentGpuPythonJobs)
if ($concurrentGpuPython.Count -gt 0 -and $WaitForGpuClear -and $usesGpu -and -not $AllowConcurrentPython -and -not $DryRun) {
    $startedWaiting = Get-Date
    while ($concurrentGpuPython.Count -gt 0) {
        $elapsed = [int]((Get-Date) - $startedWaiting).TotalSeconds
        if ($GpuWaitTimeoutSeconds -gt 0 -and $elapsed -ge $GpuWaitTimeoutSeconds) {
            break
        }
        $processSummary = ($concurrentGpuPython | ForEach-Object { "PID $($_.ProcessId)" }) -join ", "
        Write-Host "Waiting for concurrent CUDA Python job(s) to finish: $processSummary elapsed=${elapsed}s"
        Start-Sleep -Seconds $GpuPollSeconds
        $concurrentGpuPython = @(Get-ConcurrentGpuPythonJobs)
    }
}
if ($concurrentGpuPython.Count -gt 0 -and $usesGpu -and -not $AllowConcurrentPython -and -not $DryRun) {
    $processList = ($concurrentGpuPython | ForEach-Object { "PID $($_.ProcessId): $($_.CommandLine)" }) -join "`n"
    throw "Concurrent CUDA Python job detected. Re-run with -WaitForGpuClear to wait, or -AllowConcurrentPython to override.`n$processList"
}

$configList = @()
foreach ($part in ($Configs -split ",")) {
    $trimmed = $part.Trim()
    if ($trimmed) { $configList += $trimmed }
}
if ($configList.Count -eq 0) { throw "At least one config is required" }

foreach ($config in $configList) {
    $lower = $config.ToLowerInvariant()
    if ($lower -notmatch '(^|/)seat=first($|/)') {
        throw "First-seat C4 runner only accepts configs with seat=first: $config"
    }
    if ($lower -match '(^|/)seat=(second|both|all|\*)($|/)' -or $lower -match '(^|/)seats=(second|both|all|\*)($|/)') {
        throw "First-seat C4 runner rejects non-first seat configs: $config"
    }
}

$repoRoot = (Get-Location).Path
$runner = Join-Path $repoRoot "scripts\Run-HuTurn2Stage8cTopkPerFireEval.ps1"
if (-not (Test-Path $runner)) { throw "Missing runner: $runner" }
if (-not (Test-Path (Join-Path $repoRoot $Model))) { throw "Model not found: $Model" }

$arguments = @(
    "-GamesPerSeed", "$GamesPerSeed",
    "-TargetRealizedOverridesPerSeed", "$TargetRealizedOverridesPerSeed",
    "-Seeds", $Seeds,
    "-SeedStride", "$SeedStride",
    "-Configs", ($configList -join ","),
    "-Model", $Model,
    "-OutputDir", $OutputDir,
    "-Device", $Device,
    "-PredictionThreads", "$PredictionThreads",
    "-OpeningLookaheadSamples", "$OpeningLookaheadSamples",
    "-T3Continuation", "$T3Continuation",
    "-ProgressEvery", "$ProgressEvery"
)
$runnerParams = @{
    GamesPerSeed = $GamesPerSeed
    TargetRealizedOverridesPerSeed = $TargetRealizedOverridesPerSeed
    Seeds = $Seeds
    SeedStride = $SeedStride
    Configs = ($configList -join ",")
    Model = $Model
    OutputDir = $OutputDir
    Device = $Device
    PredictionThreads = $PredictionThreads
    OpeningLookaheadSamples = $OpeningLookaheadSamples
    T3Continuation = $T3Continuation
    ProgressEvery = $ProgressEvery
}

if ($NoDecisionLog) {
    $arguments += "-NoDecisionLog"
    $runnerParams["NoDecisionLog"] = $true
}

if ($DryRun) {
    [pscustomobject]@{
        execution = "dry_run"
        runner = $runner
        arguments = $arguments
        aggregate_enabled = -not $NoAggregate
        aggregate_output_dir = $AggregateOutputDir
        uses_gpu = $usesGpu
        wait_for_gpu_clear = [bool]$WaitForGpuClear
        gpu_wait_timeout_seconds = $GpuWaitTimeoutSeconds
        gpu_poll_seconds = $GpuPollSeconds
        allow_concurrent_python = [bool]$AllowConcurrentPython
        concurrent_gpu_python_process_count = $concurrentGpuPython.Count
        concurrent_gpu_python_processes = $concurrentGpuPython
        configs = $configList
        seat_scope = "first"
        t3_continuation = $T3Continuation
        model = $Model
        output_dir = $OutputDir
        production_p2_fixed = "No-Go"
        t1_training = "No-Go"
        teacher_50k = "No-Go"
    } | ConvertTo-Json -Depth 6
    exit 0
}

& $runner @runnerParams
if ($LASTEXITCODE -ne 0) {
    throw "TopK per-fire eval failed with exit code $LASTEXITCODE"
}

if (-not $NoAggregate) {
    python -m ofc_regular.analyze_hu_turn2_stage8c_topk_per_fire `
        --input-dir $OutputDir `
        --output-dir $AggregateOutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "TopK per-fire aggregate failed with exit code $LASTEXITCODE"
    }
}
