# Local benchmark: focus on SAMPLE COUNT variations + alternative nesting
# GCP covers: n321 (s30-s2000), n532 (s30-s500), n853 (s30-s100)
# Local covers: n521, n333, n421 with sample count sweep

$exe = "ai\rust_solver\target\release\cfr_solver.exe"
$outDir = "d:\ofc_data\bench_matrix"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$nHands = 10
$seed = 42

$configs = @(
    # --- n521: shallow+wide (cost = 5*2*1 = 10, moderate) ---
    @("5,2,1",   30,  "n521_s30"),
    @("5,2,1",   50,  "n521_s50"),
    @("5,2,1",  100,  "n521_s100"),
    @("5,2,1",  200,  "n521_s200"),
    @("5,2,1",  500,  "n521_s500"),

    # --- n333: uniform depth (cost = 3*3*3 = 27, heavier) ---
    @("3,3,3",   30,  "n333_s30"),
    @("3,3,3",   50,  "n333_s50"),
    @("3,3,3",  100,  "n333_s100"),

    # --- n421: asymmetric (cost = 4*2*1 = 8) ---
    @("4,2,1",   50,  "n421_s50"),
    @("4,2,1",  100,  "n421_s100"),
    @("4,2,1",  200,  "n421_s200"),
    @("4,2,1",  500,  "n421_s500")
)

Write-Host ("=" * 70)
Write-Host "Local Benchmark: $nHands hands, seed=$seed, $($configs.Count) configs"
Write-Host "Focus: sample count sweep + alternative nesting patterns"
Write-Host ("=" * 70)

$timingFile = "$outDir\timing_local.csv"
if (-not (Test-Path $timingFile)) {
    "config,nesting,samples,hands,total_seconds,sec_per_hand" | Out-File $timingFile -Encoding utf8
}

foreach ($cfg in $configs) {
    $nesting = $cfg[0]
    $samples = $cfg[1]
    $label   = $cfg[2]
    $outFile = "$outDir\$label.jsonl"

    if ((Test-Path $outFile) -and (Get-Content $outFile | Measure-Object -Line).Lines -ge $nHands) {
        Write-Host ""
        Write-Host ("--- SKIP $label (already complete) ---")
        continue
    }

    if (Test-Path $outFile) { Remove-Item $outFile }

    Write-Host ""
    Write-Host ("=" * 60)
    Write-Host ("Config: $label  |  nesting=[$nesting]  samples=$samples")
    Write-Host ("-" * 60)

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $exe t0-batch --hands $nHands --samples $samples --nesting $nesting --output $outFile --seed $seed 2>&1 | ForEach-Object { Write-Host $_ }
    $sw.Stop()
    $elapsed = $sw.Elapsed.TotalSeconds
    $perHand = $elapsed / $nHands

    Write-Host ("  Total: {0:F1}s  |  Per hand: {1:F1}s" -f $elapsed, $perHand)
    "$label,$nesting,$samples,$nHands,$([math]::Round($elapsed,1)),$([math]::Round($perHand,1))" | Out-File $timingFile -Append -Encoding utf8
}

Write-Host ""
Write-Host ("=" * 70)
Write-Host "Local benchmark done!"
Write-Host ("=" * 70)
