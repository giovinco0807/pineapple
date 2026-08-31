# T0 Precision Benchmark: 5 hands, different nesting/sample configs
# Same seed=42 across all configs -> same 5 hands for fair comparison

$exe = "ai\rust_solver\target\release\cfr_solver.exe"
$outDir = "d:\ofc_data\bench"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
# Clean old results
Get-ChildItem "$outDir\bench_*.jsonl" -ErrorAction SilentlyContinue | Remove-Item

$configs = @(
    @("3,2,1",  100,  "n321_s100"),
    @("3,2,1",  500,  "n321_s500"),
    @("5,3,2",  100,  "n532_s100"),
    @("5,3,2",  200,  "n532_s200"),
    @("10,6,3", 50,   "n1063_s50")
)

Write-Host ("=" * 70)
Write-Host "T0 Precision Benchmark - 5 hands, seed=42"
Write-Host ("=" * 70)

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

    if (Test-Path $outFile) {
        $lines = Get-Content $outFile
        foreach ($line in $lines) {
            $json = $line | ConvertFrom-Json
            Write-Host ""
            Write-Host ("  Hand {0}: {1}  ({2}, {3} placements)" -f $json.hand_idx, $json.hand, $json.type, $json.n_placements)
            $top3 = $json.placements | Select-Object -First 3
            for ($i = 0; $i -lt $top3.Count; $i++) {
                $p = $top3[$i]
                $marker = if ($i -eq 0) { " <<<" } else { "" }
                Write-Host ("    #{0}  EV:{1,8:F3}  {2}{3}" -f ($i+1), $p.ev, $p.p, $marker)
            }
        }
    } else {
        Write-Host "  ERROR: no output"
    }
}

Write-Host ""
Write-Host ("=" * 70)
Write-Host "Benchmark complete. Comparing best placements across configs..."
Write-Host ("=" * 70)

# Cross-config comparison
Write-Host ""
Write-Host "Hand | Config       | Best EV  | Best Placement"
Write-Host "-----|--------------|----------|---------------"

foreach ($cfg in $configs) {
    $label = $cfg[2]
    $outFile = "$outDir\bench_$label.jsonl"
    if (Test-Path $outFile) {
        $lines = Get-Content $outFile
        foreach ($line in $lines) {
            $json = $line | ConvertFrom-Json
            $best = $json.placements[0]
            Write-Host ("{0,4} | {1,-12} | {2,8:F3} | {3}" -f $json.hand_idx, $label, $best.ev, $best.p)
        }
    }
}
