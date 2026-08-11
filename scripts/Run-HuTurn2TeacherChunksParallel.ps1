param(
    [int]$TotalSamples = 5000,
    [int]$Chunks = 100,
    [int]$FutureSamples = 16,
    [int]$PrefilterFutureSamples = 4,
    [int]$MaxHands = 1000000,
    [int]$MaxParallel = 4,
    [int]$PredictionThreads = 1,
    [Alias("SeedBase")]
    [int]$Seed = 2026061101,
    [int]$ProgressEvery = 10,
    [string]$Stage3FeatureEncoderMode = "rust_direct",
    [ValidateSet("stage3_reference_default", "stage7_m5_r10")]
    [string]$T3Continuation = "stage3_reference_default",
    [switch]$EnableM2T4Search,
    [string]$OutputDir = "outputs/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16/shards"
)

$ErrorActionPreference = "Stop"

$buckets = @(
    @{ Name = "natural"; Weight = 0.50 },
    @{ Name = "teacher_disagreement"; Weight = 0.15 },
    @{ Name = "high_regret"; Weight = 0.15 },
    @{ Name = "low_margin"; Weight = 0.10 },
    @{ Name = "random_off_policy"; Weight = 0.10 }
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$summaryDir = Join-Path $OutputDir "summaries"
New-Item -ItemType Directory -Force -Path $summaryDir | Out-Null
$jobs = @()
$chunkIndex = 0
$repoRoot = (Get-Location).Path

function Receive-HuTurn2ShardJob {
    param(
        [Parameter(Mandatory = $true)]
        [System.Management.Automation.Job]$Job
    )

    $receiveErrors = @()
    $result = Receive-Job -Job $Job -Wait -ErrorAction SilentlyContinue -ErrorVariable receiveErrors
    if ($null -ne $result) {
        $result
    }

    $childErrors = @()
    foreach ($child in $Job.ChildJobs) {
        foreach ($errorRecord in $child.Error) {
            $childErrors += $errorRecord.ToString()
        }
    }
    foreach ($errorRecord in $receiveErrors) {
        $childErrors += $errorRecord.ToString()
    }

    $failed = ($Job.State -eq "Failed") -or ($childErrors.Count -gt 0)
    $jobName = $Job.Name
    Remove-Job -Job $Job -Force

    if ($failed) {
        $message = ($childErrors | Select-Object -First 5) -join "`n"
        if ($message -eq "") {
            $message = "Job state: $($Job.State)"
        }
        throw "HU T2 teacher shard job failed: $jobName`n$message"
    }
}

foreach ($bucket in $buckets) {
    $bucketSamples = [int][Math]::Round($TotalSamples * [double]$bucket.Weight)
    $bucketChunks = [Math]::Max(1, [int][Math]::Round($Chunks * [double]$bucket.Weight))
    $samplesPerChunk = [Math]::Max(1, [int][Math]::Ceiling($bucketSamples / $bucketChunks))
    for ($i = 0; $i -lt $bucketChunks; $i++) {
        $remaining = $bucketSamples - ($i * $samplesPerChunk)
        if ($remaining -le 0) { continue }
        $samples = [Math]::Min($samplesPerChunk, $remaining)
        $path = Join-Path $OutputDir ("hu_turn2_teacher_{0}_{1:D4}.jsonl" -f $bucket.Name, $i)
        $summaryPath = Join-Path $summaryDir ("hu_turn2_teacher_{0}_{1:D4}.summary.json" -f $bucket.Name, $i)
        $shardSeed = $Seed + $chunkIndex * 100003
        while (($jobs | Where-Object { $_.State -eq "Running" }).Count -ge $MaxParallel) {
            Start-Sleep -Seconds 5
            $finished = $jobs | Where-Object { $_.State -ne "Running" }
            foreach ($job in $finished) {
                Receive-HuTurn2ShardJob -Job $job
            }
            $jobs = @($jobs | Where-Object { $_.State -eq "Running" })
        }
        $jobName = "hu_t2_{0}_{1:D4}" -f $bucket.Name, $i
        $jobArgs = @(
            $repoRoot,
            $bucket.Name,
            $samples,
            $FutureSamples,
            $PrefilterFutureSamples,
            $MaxHands,
            $shardSeed,
            $PredictionThreads,
            $ProgressEvery,
            $Stage3FeatureEncoderMode,
            $T3Continuation,
            $path,
            $summaryPath,
            ([bool]$EnableM2T4Search)
        )
        $jobs += @(Start-Job -ScriptBlock {
            param(
                $repoRoot,
                $sourceBucket,
                $samples,
                $futureSamples,
                $prefilterFutureSamples,
                $maxHands,
                $seed,
                $predictionThreads,
                $progressEvery,
                $stage3FeatureEncoderMode,
                $t3Continuation,
                $output,
                $summaryOutput,
                $enableM2T4Search
            )
            Set-Location $repoRoot
            $m2Args = @{}
            if ($enableM2T4Search) {
                $m2Args["EnableM2T4Search"] = $true
            }
            .\scripts\Run-HuTurn2TeacherShard.ps1 `
                -SourceBucket $sourceBucket `
                -Samples $samples `
                -FutureSamples $futureSamples `
                -PrefilterFutureSamples $prefilterFutureSamples `
                -MaxHands $maxHands `
                -Seed $seed `
                -PredictionThreads $predictionThreads `
                -ProgressEvery $progressEvery `
                -Stage3FeatureEncoderMode $stage3FeatureEncoderMode `
                -T3Continuation $t3Continuation `
                -Output $output `
                -SummaryOutput $summaryOutput `
                @m2Args
        } -ArgumentList $jobArgs -Name $jobName)
        $chunkIndex++
    }
}

if ($chunkIndex -le 0) {
    throw "No shard jobs were scheduled. Increase TotalSamples or adjust bucket weights."
}

foreach ($job in $jobs) {
    Receive-HuTurn2ShardJob -Job $job
}

$missingSummaries = @()
$emptyOutputs = @()
Get-ChildItem -LiteralPath $OutputDir -Filter "hu_turn2_teacher_*.jsonl" -File |
    Where-Object { $_.Name -notlike "*_final_turn_slow_states_top50.jsonl" } |
    ForEach-Object {
        if ($_.Length -le 0) {
            $emptyOutputs += $_.FullName
        }
        $summaryName = $_.BaseName + ".summary.json"
        $summaryPath = Join-Path $summaryDir $summaryName
        if (-not (Test-Path -LiteralPath $summaryPath)) {
            $missingSummaries += $summaryPath
        }
    }
if ($emptyOutputs.Count -gt 0) {
    throw "One or more shard outputs are empty:`n$($emptyOutputs -join "`n")"
}
if ($missingSummaries.Count -gt 0) {
    throw "One or more shard summaries are missing:`n$($missingSummaries -join "`n")"
}
