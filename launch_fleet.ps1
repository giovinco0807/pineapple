# Launch 10 GCP worker VMs for parallel T0 data collection
# Each VM: e2-highcpu-16, Spot, runs 50 hands, self-deletes when done
# Total: 500 hands, nesting=[3,2,1], samples=50

$NUM_WORKERS = 10
$HANDS_PER_WORKER = 50
$SAMPLES = 50
$NESTING = "3,2,1"
$BASE_SEED = 100000  # Each worker gets BASE_SEED + worker_id * 10000
$ZONES = @("us-central1-a", "us-central1-b", "us-east1-b", "us-east1-c", "us-west1-a")
$MACHINE_TYPE = "e2-highcpu-16"
$SCRIPT_PATH = "gcp_worker_startup.sh"

Write-Host "============================================"
Write-Host "GCP Fleet Launcher - T0 Data Collection"
Write-Host "============================================"
Write-Host "Workers: $NUM_WORKERS x $MACHINE_TYPE (Spot)"
Write-Host "Each: $HANDS_PER_WORKER hands, s=$SAMPLES, n=[$NESTING]"
Write-Host "Total: $($NUM_WORKERS * $HANDS_PER_WORKER) hands"
Write-Host "Est. cost: ~`$3 total"
Write-Host "============================================"
Write-Host ""

$jobs = @()
for ($i = 0; $i -lt $NUM_WORKERS; $i++) {
    $workerName = "t0-worker-$i"
    $seed = $BASE_SEED + ($i * 10000)
    $zone = $ZONES[$i % $ZONES.Count]
    
    Write-Host "[$($i+1)/$NUM_WORKERS] Launching $workerName in $zone (seed=$seed)..."
    
    $cmd = "gcloud compute instances create $workerName " +
        "--zone=$zone " +
        "--machine-type=$MACHINE_TYPE " +
        "--provisioning-model=SPOT " +
        "--instance-termination-action=DELETE " +
        "--image-family=debian-12 " +
        "--image-project=debian-cloud " +
        "--boot-disk-size=20GB " +
        "--boot-disk-type=pd-ssd " +
        "--scopes=cloud-platform " +
        """--metadata=^~^worker-id=$i~seed=$seed~hands=$HANDS_PER_WORKER~samples=$SAMPLES~nesting=$NESTING""" + " " +
        "--metadata-from-file=startup-script=$SCRIPT_PATH " +
        "--no-restart-on-failure " +
        "2>&1"
    
    $result = Invoke-Expression $cmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK - $workerName created" -ForegroundColor Green
    } else {
        Write-Host "  FAIL - $workerName" -ForegroundColor Red
        Write-Host "  $result"
    }
    
    # Small delay to avoid quota burst
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "============================================"
Write-Host "All workers launched!"
Write-Host "Monitor: gcloud compute instances list --filter='name~t0-worker'"
Write-Host "Results: gsutil ls gs://ofc-solver-485418/t0_training/"
Write-Host "Workers will self-delete when done."
Write-Host "============================================"
