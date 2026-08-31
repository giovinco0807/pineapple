# Systematic benchmark: same 5 hands, multiple nesting/samples configs
# Goal: find optimal accuracy/speed trade-off
# "Ground truth" = highest-fidelity config (n321_s2000)

$exe = "ai\rust_solver\target\release\cfr_solver.exe"
$outDir = "d:\ofc_data\bench_matrix"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$nHands = 5
$seed = 42

# Config matrix: [nesting, samples, label]
# Ordered from cheapest to most expensive
$configs = @(
    # --- Cheap configs (for speed) ---
    @("3,2,1",   30,   "n321_s30"),
    @("3,2,1",   50,   "n321_s50"),
    @("3,2,1",  100,   "n321_s100"),
    @("3,2,1",  200,   "n321_s200"),
    @("3,2,1",  500,   "n321_s500"),
    @("3,2,1", 1000,   "n321_s1000"),

    # --- Deeper nesting ---
    @("5,3,2",   30,   "n532_s30"),
    @("5,3,2",   50,   "n532_s50"),
    @("5,3,2",  100,   "n532_s100"),
    @("5,3,2",  200,   "n532_s200"),

    # --- Ground truth (expensive) ---
    @("3,2,1", 2000,   "n321_s2000"),
    @("5,3,2",  500,   "n532_s500")
)

Write-Host ("=" * 70)
Write-Host "Benchmark Matrix: $nHands hands, seed=$seed, $($configs.Count) configs"
Write-Host ("=" * 70)

$timingLog = "$outDir\timing.csv"
"config,nesting,samples,hands,total_seconds,sec_per_hand" | Out-File $timingLog -Encoding utf8

foreach ($cfg in $configs) {
    $nesting = $cfg[0]
    $samples = $cfg[1]
    $label   = $cfg[2]
    $outFile = "$outDir\$label.jsonl"
    
    # Skip if already done
    if ((Test-Path $outFile) -and (Get-Content $outFile | Measure-Object -Line).Lines -ge $nHands) {
        Write-Host ""
        Write-Host ("--- SKIP $label (already complete) ---")
        continue
    }

    # Clean for fresh run
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

    # Log timing
    "$label,$nesting,$samples,$nHands,$([math]::Round($elapsed,1)),$([math]::Round($perHand,1))" | Out-File $timingLog -Append -Encoding utf8
}

Write-Host ""
Write-Host ("=" * 70)
Write-Host "All configs done! Run: python analyze_bench_matrix.py"
Write-Host ("=" * 70)
