# Fixed dispatch: sudo chown BEFORE nohup, then start worker
$PROJECT = "ofc-solver-485418"

function Get-Zone($idx) {
    if ($idx -lt 24) { "us-east1-c" }
    elseif ($idx -lt 48) { "us-central1-a" }
    elseif ($idx -lt 72) { "us-east4-a" }
    elseif ($idx -lt 96) { "us-west1-a" }
    elseif ($idx -lt 122) { "asia-northeast1-b" }
    else { "asia-east1-a" }
}

# All DONE + RUNNING VMs (skip vm8 which is already running correctly)
$allVMs = @(
    9,11,12,13,14,15,16,17,18,19,20,21,22,23,
    25,26,27,28,29,30,31,34,37,38,39,40,41,46,
    48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,
    72,73,78,80,81,83,85,90,95,
    96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,
    122,123,124,125,126,127
)

$total = $allVMs.Count
Write-Host "=== Dispatching Round 2 (fixed chown) to $total VMs ==="
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
        $vmIdx = $allVMs[$j]
        $zone = Get-Zone $vmIdx
        $instance = "ofc-t3fleet-$vmIdx"

        $jobs += Start-Job -ScriptBlock {
            param($inst, $z, $proj, $vmId)
            try {
                # Step 1: Fix permissions (sudo runs inline, NOT in nohup)
                # Step 2: Download fresh script
                # Step 3: nohup the worker (no sudo needed)
                $cmd = @"
sudo chown -R Owner:Owner /home/Owner/ofc-pineapple/ai/data/t3_fleet/vm${vmId}/ 2>/dev/null
gsutil cp gs://ofc-solver-results/t3_rust_fleet/gcp_r2_worker.sh /tmp/gcp_r2_worker.sh
chmod +x /tmp/gcp_r2_worker.sh
nohup bash /tmp/gcp_r2_worker.sh ${vmId} > /tmp/r2_vm${vmId}.log 2>&1 &
sleep 2
pgrep -c t3_generator && echo R2_OK || echo R2_FAIL
"@
                $result = & gcloud compute ssh $inst --zone=$z --project=$proj --command=$cmd 2>&1
                $output = ($result | Out-String)
                if ($output -match "R2_OK") {
                    "OK: $inst (vm$vmId)"
                } elseif ($output -match "R2_FAIL") {
                    "FAIL: $inst - no generator process"
                } else {
                    $last = ($result | Select-Object -Last 1).ToString().Trim()
                    "WARN: $inst - $last"
                }
            } catch {
                "ERR: $inst - $_"
            }
        } -ArgumentList $instance, $zone, $PROJECT, $vmIdx
    }

    $completed = $jobs | Wait-Job -Timeout 120
    foreach ($job in $jobs) {
        if ($job.State -eq 'Completed') {
            $result = Receive-Job $job
            Write-Host "  $result"
            if ($result -match "^OK:") { $dispatched++ }
            else { $failed += $result }
        } else {
            Write-Host "  Timeout: $($job.Id)"
            $failed += "Timeout"
            Stop-Job $job -ErrorAction SilentlyContinue
        }
    }
    $jobs | Remove-Job -Force -ErrorAction SilentlyContinue
    Write-Host ""
}

Write-Host "========================================="
Write-Host "  Dispatch Complete"
Write-Host "========================================="
Write-Host "  OK: $dispatched / $total"
Write-Host "  + vm8, vm10 (already running) = $($dispatched + 2)"
if ($failed.Count -gt 0) {
    Write-Host "  Issues ($($failed.Count)):"
    $failed | ForEach-Object { Write-Host "    $_" }
}
