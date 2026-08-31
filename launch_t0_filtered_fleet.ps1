# Launch 20 GCP VMs for T0 filtered batch evaluation
# Each VM processes 25 hands (500 total)

$PROJECT = "ofc-solver-485418"
$ZONE = "us-east1-b"
$MACHINE_TYPE = "e2-highcpu-16"
$IMAGE_FAMILY = "ubuntu-2204-lts"
$IMAGE_PROJECT = "ubuntu-os-cloud"
$STARTUP_SCRIPT = "gcp_t0_filtered_startup.sh"
$NUM_VMS = 20

Write-Host "=== Launching $NUM_VMS VMs for T0 Filtered Batch ==="
Write-Host "Machine type: $MACHINE_TYPE"
Write-Host "Zone: $ZONE"
Write-Host ""

# First, delete old terminated workers
Write-Host "Cleaning up old workers..."
$oldVMs = gcloud compute instances list --filter="name~'t0-worker-' AND status=TERMINATED" --format="value(name)" --project=$PROJECT 2>$null
foreach ($vm in $oldVMs) {
    if ($vm) {
        Write-Host "  Deleting $vm..."
        gcloud compute instances delete $vm --zone=$ZONE --quiet --project=$PROJECT 2>$null
    }
}

# Launch VMs
$jobs = @()
for ($i = 0; $i -lt $NUM_VMS; $i++) {
    $vmName = "t0-filtered-$('{0:D2}' -f $i)"
    $shardId = $i
    
    Write-Host "Launching $vmName (shard $shardId)..."
    
    $job = Start-Job -ScriptBlock {
        param($vmName, $zone, $machineType, $imageFamily, $imageProject, $project, $startupScript, $shardId)
        gcloud compute instances create $vmName `
            --zone=$zone `
            --machine-type=$machineType `
            --image-family=$imageFamily `
            --image-project=$imageProject `
            --project=$project `
            --metadata="shard-id=$shardId" `
            --metadata-from-file="startup-script=$startupScript" `
            --scopes="storage-rw" `
            --boot-disk-size=30GB `
            --no-restart-on-failure `
            2>&1
    } -ArgumentList $vmName, $ZONE, $MACHINE_TYPE, $IMAGE_FAMILY, $IMAGE_PROJECT, $PROJECT, $STARTUP_SCRIPT, $shardId
    
    $jobs += $job
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "Waiting for all VMs to be created..."
$jobs | Wait-Job | Out-Null

$success = 0
$failed = 0
foreach ($job in $jobs) {
    $result = Receive-Job -Job $job
    if ($job.State -eq "Completed") {
        $success++
    } else {
        $failed++
        Write-Host "FAILED: $result"
    }
    Remove-Job -Job $job
}

Write-Host ""
Write-Host "=== Launch Complete ==="
Write-Host "Success: $success / $NUM_VMS"
Write-Host "Failed: $failed"
Write-Host ""
Write-Host "Monitor progress:"
Write-Host "  gcloud compute instances list --filter='name~t0-filtered' --format='table(name,status)'"
Write-Host ""
Write-Host "Check results:"
Write-Host "  gsutil ls gs://ofc-solver-485418/t0_filtered_results/"
Write-Host ""
Write-Host "When all done, merge results:"
Write-Host "  gsutil -m cp gs://ofc-solver-485418/t0_filtered_results/*.jsonl ai/data/t0_results/"
