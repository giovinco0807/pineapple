# Focused benchmark: same 5 hands, 3 configs (fast ones only)
# [3,2,1]x100, [3,2,1]x500, [5,3,2]x50

$exe = "ai\rust_solver\target\release\cfr_solver.exe"
$outDir = "d:\ofc_data\bench"

# Only run configs we don't have yet
$configs = @(
    @("3,2,1",  500,  "n321_s500"),
    @("5,3,2",  50,   "n532_s50")
)

foreach ($cfg in $configs) {
    $nesting = $cfg[0]
    $samples = $cfg[1]
    $label   = $cfg[2]
    $outFile = "$outDir\bench_$label.jsonl"

    Write-Host ""
    Write-Host ("=" * 60)
    Write-Host ("Config: nesting=[{0}] samples={1}" -f $nesting, $samples)
    Write-Host ("-" * 60)

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $exe t0-batch --hands 5 --samples $samples --nesting $nesting --output $outFile --seed 42 2>&1 | Out-Null
    $sw.Stop()
    $elapsed = $sw.Elapsed.TotalSeconds

    Write-Host ("  Total time: {0:F1}s  ({1:F1}s/hand)" -f $elapsed, ($elapsed / 5))
}

Write-Host ""
Write-Host ("=" * 70)
Write-Host "All configs done. Now comparing..."
Write-Host ("=" * 70)
