# Quick benchmark: 1 hand only, multiple configs
# Compare EV rankings for the same hand

$exe = "ai\rust_solver\target\release\cfr_solver.exe"
$outDir = "d:\ofc_data\bench"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$configs = @(
    @("3,2,1",  50,   "quick_n321_s50"),
    @("3,2,1",  200,  "quick_n321_s200"),
    @("5,3,2",  30,   "quick_n532_s30"),
    @("5,3,2",  100,  "quick_n532_s100")
)

Write-Host ("=" * 70)
Write-Host "Quick Benchmark: 1 hand, seed=42, 4 configs"
Write-Host ("=" * 70)

foreach ($cfg in $configs) {
    $nesting = $cfg[0]
    $samples = $cfg[1]
    $label   = $cfg[2]
    $outFile = "$outDir\$label.jsonl"
    if (Test-Path $outFile) { Remove-Item $outFile }

    Write-Host ""
    Write-Host ("--- nesting=[{0}] samples={1} ---" -f $nesting, $samples)

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $exe t0-batch --hands 1 --samples $samples --nesting $nesting --output $outFile --seed 42 2>&1 | Out-Null
    $sw.Stop()
    $elapsed = $sw.Elapsed.TotalSeconds

    Write-Host ("  Time: {0:F1}s" -f $elapsed)
}

Write-Host ""
Write-Host "Done. Comparing results..."
