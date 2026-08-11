param(
    [int[]]$Samples = @(20, 100),
    [int[]]$FutureSamples = @(64, 128),
    [int]$Seed = 20260602,
    # configs/fl_ev_regular_v3_direct2.json (armA, 2026-08-03).
    [double]$FlEv = 9.109,
    [double]$MinScoreGap = 0.25,
    [string]$OutputDir = "outputs/benchmarks"
)

$ErrorActionPreference = "Stop"

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
$results = @()
$caseIndex = 0

foreach ($future in $FutureSamples) {
    foreach ($sampleCount in $Samples) {
        $caseIndex += 1
        $output = Join-Path $OutputDir ("t3_bench_s{0}_f{1}.jsonl" -f $sampleCount, $future)
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $exe `
            --teacher-output $output `
            --teacher-samples $sampleCount `
            --future-samples $future `
            --seed ($Seed + $caseIndex) `
            --teacher-fl-ev $FlEv `
            --teacher-min-score-gap $MinScoreGap | Out-Null
        $sw.Stop()

        $lines = Get-LineCount $output
        $secondsPerSample = $sw.Elapsed.TotalSeconds / [math]::Max($lines, 1)
        $results += [pscustomobject]@{
            samples = $sampleCount
            future_samples = $future
            seconds = [math]::Round($sw.Elapsed.TotalSeconds, 3)
            seconds_per_sample = [math]::Round($secondsPerSample, 3)
            est_1000_minutes = [math]::Round($secondsPerSample * 1000 / 60, 1)
            est_10000_hours = [math]::Round($secondsPerSample * 10000 / 3600, 2)
            lines = $lines
            output = $output
        }
    }
}

$results | ConvertTo-Json -Depth 4
