# Dispatch Round 2 with FIXED script (chown fix) to all DONE+RUNNING VMs
# Re-dispatches to ALL VMs including those that got the old broken script

$PROJECT = "ofc-solver-485418"

function Get-Zone($idx) {
    if ($idx -lt 24) { "us-east1-c" }
    elseif ($idx -lt 48) { "us-central1-a" }
    elseif ($idx -lt 72) { "us-east4-a" }
    elseif ($idx -lt 96) { "us-west1-a" }
    elseif ($idx -lt 122) { "asia-northeast1-b" }
    else { "asia-east1-a" }
}

# All DONE + RUNNING VMs (excludes TERMINATED: 24,47,75,76,82,84,86,89,93)
# vm8 and vm10 already dispatched with fixed script, but re-dispatch is safe
# (t3_generator is already running on those, nohup will start a new one after it finishes)
$allVMs = @(
    8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
    25,26,27,28,29,30,31,34,37,38,39,40,41,46,
    48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,
    72,73,78,80,81,83,85,90,95,
    96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,
    122,123,124,125,126,127
)

# Skip vm8 and vm10 (already correctly dispatched)
$vmsToDispatch = $allVMs | Where-Object { $_ -notin @(8, 10) }

$total = $vmsToDispatch.Count
Write-Host "=== Re-dispatching Round 2 (fixed) to $total VMs ==="
Write-Host "    (skipping vm8, vm10 - already running correctly)"
Write-Host ""

$BATCH_SIZE = 20
$dispatched = 0
$failed = @()

for ($i = 0; $i -lt $total; $i += $BATCH_SIZE) {
    $batchEnd = [Math]::Min($i + $BATCH_SIZE, $total)
    $batchNum = [Math]::Floor($i / $BATCH_SIZE) + 1
    Write-Host "--- Batch $batchNum ($($batchEnd - $i) VMs) ---"

    $jobs = @()
    for ($j = $i; $j -lt $batchEnd; $j++) {
        $vmIdx = $vmsToDispatch[$j]
        $zone = Get-Zone $vmIdx
        $instance = "ofc-t3fleet-$vmIdx"

        $jobs += Start-Job -ScriptBlock {
            param($inst, $z, $proj, $vmId)
            try {
                # Kill any old broken r2 process first, then start fresh
                $cmd = "pkill -f 'gcp_r2_worker' 2>/dev/null; sleep 1; gsutil cp gs://ofc-solver-results/t3_rust_fleet/gcp_r2_worker.sh /tmp/gcp_r2_worker.sh && chmod +x /tmp/gcp_r2_worker.sh && nohup bash /tmp/gcp_r2_worker.sh $vmId > /tmp/r2_vm${vmId}.log 2>&1 & sleep 2 && pgrep -c t3_generator && echo R2_OK"
                $result = & gcloud compute ssh $inst --zone=$z --project=$proj --command=$cmd 2>&1
                $lastLine = ($result | Select-Object -Last 1).ToString().Trim()
                if ($lastLine -eq "R2_OK") {
                    "OK: $inst (vm$vmId)"
                } elseif ($result -match "R2_OK") {
                    "OK: $inst (vm$vmId)"
                } else {
                    "WARN: $inst (vm$vmId) - $lastLine"
                }
            } catch {
                "FAIL: $inst (vm$vmId) - $_"
            }
        } -ArgumentList $instance, $zone, $PROJECT, $vmIdx
    }

    # Wait with timeout
    $completed = $jobs | Wait-Job -Timeout 120
    foreach ($job in $jobs) {
        if ($job.State -eq 'Completed') {
            $result = Receive-Job $job
            Write-Host "  $result"
            if ($result -match "^OK:") { $dispatched++ }
            else { $failed += $result }
        } else {
            $vmInfo = "Timeout: job for batch $batchNum"
            Write-Host "  $vmInfo"
            $failed += $vmInfo
            Stop-Job $job -ErrorAction SilentlyContinue
        }
    }
    $jobs | Remove-Job -Force -ErrorAction SilentlyContinue
    Write-Host ""
}

Write-Host "========================================="
Write-Host "  Dispatch Complete"
Write-Host "========================================="
Write-Host "  Successfully dispatched: $dispatched / $total"
Write-Host "  + vm8, vm10 (already running) = $($dispatched + 2) total"
if ($failed.Count -gt 0) {
    Write-Host "  Failed ($($failed.Count)):"
    $failed | ForEach-Object { Write-Host "    $_" }
}
Write-Host ""
Write-Host "  Each VM generates 3 x 1000 = 3000 new states"
Write-Host "  Expected total: ~$($($dispatched + 2) * 3000) new states"
Write-Host ""
Write-Host "  Monitor: gsutil ls 'gs://ofc-solver-results/t3_rust_fleet/run_20260509/*/DONE_R2' 2>&1 | Select-String 'DONE_R2' | Measure-Object"
