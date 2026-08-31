# Launch 6 spot VMs for parallel T0 batch evaluation
# Each VM: c2d-highcpu-16, spot, auto-delete on completion

$PROJECT = "ofc-solver-485418"
$ZONE = "asia-northeast1-b"
$BUCKET = "gs://ofc-solver-485418"
$HANDS_PER_VM = 34
$SAMPLES = 1000

# VM configs: name, seed, hands
$VMS = @(
    @{ Name="t0-batch-1"; Seed=1001; Hands=34 },
    @{ Name="t0-batch-2"; Seed=2001; Hands=34 },
    @{ Name="t0-batch-3"; Seed=3001; Hands=34 },
    @{ Name="t0-batch-4"; Seed=4001; Hands=34 },
    @{ Name="t0-batch-5"; Seed=5001; Hands=33 },
    @{ Name="t0-batch-6"; Seed=6001; Hands=33 }
)
# Total: 34*4 + 33*2 = 202 hands

foreach ($vm in $VMS) {
    $name = $vm.Name
    $seed = $vm.Seed
    $hands = $vm.Hands
    $output = "t0_batch_${name}.jsonl"
    
    # Startup script: download binary, run, upload results, self-delete
    $startup = @"
#!/bin/bash
set -e
cd /tmp
gcloud storage cp ${BUCKET}/cfr_solver ./cfr_solver
chmod +x ./cfr_solver
echo "Starting batch: $hands hands, seed $seed" | tee /tmp/batch.log
./cfr_solver t0-batch --hands $hands --samples $SAMPLES --output /tmp/$output --seed $seed 2>&1 | tee -a /tmp/batch.log
echo "Uploading results..."
gcloud storage cp /tmp/$output ${BUCKET}/results/$output
gcloud storage cp /tmp/batch.log ${BUCKET}/results/log_${name}.txt
echo "DONE - shutting down"
shutdown -h now
"@
    
    Write-Host "Creating $name (seed=$seed, hands=$hands)..."
    
    gcloud compute instances create $name `
        --project=$PROJECT `
        --zone=$ZONE `
        --machine-type=c2d-highcpu-16 `
        --provisioning-model=SPOT `
        --instance-termination-action=DELETE `
        --scopes=cloud-platform `
        --image-family=debian-12 `
        --image-project=debian-cloud `
        --boot-disk-size=10GB `
        --metadata=startup-script=$startup `
        --no-address `
        2>&1
    
    Write-Host "$name created!"
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "All 6 VMs launched! Monitor with:"
Write-Host "  gcloud compute instances list --filter='name~t0-batch'"
Write-Host "  gcloud storage ls gs://ofc-solver-485418/results/"
