param(
    [int]$TotalSamples = 50000,
    [int]$Chunks = 100,
    [int]$FutureSamples = 4096,
    [int]$MaxParallel = 4,
    [int]$PredictionThreads = 1,
    [int]$Seed = 2026061101,
    [string]$OutputDir = "outputs/hu_turn2_stage1/chunks"
)

$ErrorActionPreference = "Stop"

$buckets = @(
    @{ Name = "natural"; Weight = 0.40 },
    @{ Name = "teacher_disagreement"; Weight = 0.30 },
    @{ Name = "high_regret"; Weight = 0.10 },
    @{ Name = "low_margin"; Weight = 0.10 },
    @{ Name = "high_margin"; Weight = 0.05 },
    @{ Name = "random_off_policy"; Weight = 0.05 }
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$summaryDir = Join-Path $OutputDir "summaries"
New-Item -ItemType Directory -Force -Path $summaryDir | Out-Null
$jobs = @()
$chunkIndex = 0
$repoRoot = (Get-Location).Path

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
                Receive-Job $job -Wait
                Remove-Job $job
            }
            $jobs = $jobs | Where-Object { $_.State -eq "Running" }
        }
        $jobs += Start-Job -ScriptBlock {
            param($repoRoot, $sourceBucket, $samples, $futureSamples, $seed, $predictionThreads, $output, $summaryOutput)
            Set-Location $repoRoot
            .\scripts\Run-HuTurn2TeacherShard.ps1 `
                -SourceBucket $sourceBucket `
                -Samples $samples `
                -FutureSamples $futureSamples `
                -Seed $seed `
                -PredictionThreads $predictionThreads `
                -Output $output `
                -SummaryOutput $summaryOutput
        } -ArgumentList $repoRoot, $bucket.Name, $samples, $FutureSamples, $shardSeed, $PredictionThreads, $path, $summaryPath
        $chunkIndex++
    }
}

foreach ($job in $jobs) {
    Receive-Job $job -Wait
    Remove-Job $job
}
