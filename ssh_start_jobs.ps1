# SSH into each running VM and start the batch job manually
$PROJECT = "ofc-solver-485418"
$ZONE = "us-central1-a"

$vms = @(
    @{Name="t0-hifi-01"; Seed=10001},
    @{Name="t0-hifi-02"; Seed=20001},
    @{Name="t0-hifi-03"; Seed=30001},
    @{Name="t0-hifi-04"; Seed=40001},
    @{Name="t0-hifi-05"; Seed=50001},
    @{Name="t0-hifi-06"; Seed=60001}
)

foreach ($vm in $vms) {
    $n = $vm.Name
    $s = $vm.Seed
    $out = "t0_batch_${n}.jsonl"
    
    $cmd = "cd /tmp && gcloud storage cp gs://ofc-solver-485418/cfr_solver ./cfr_solver 2>/dev/null && chmod +x ./cfr_solver && nohup bash -c './cfr_solver t0-batch --hands 5 --samples 50 --output /tmp/${out} --seed ${s} --nesting 10,6,3 2>&1 | tee /tmp/batch_${n}.log; gcloud storage cp /tmp/${out} gs://ofc-solver-485418/results/hifi/${out}; gcloud storage cp /tmp/batch_${n}.log gs://ofc-solver-485418/results/hifi/log_${n}.txt' > /tmp/nohup_${n}.out 2>&1 &"
    
    Write-Host "Starting $n (seed=$s)..."
    gcloud compute ssh $n --zone=$ZONE --project=$PROJECT --command=$cmd 2>&1
    Write-Host "  $n started"
}

Write-Host ""
Write-Host "All 6 VMs running! Check progress:"
Write-Host "  gcloud storage ls gs://ofc-solver-485418/results/hifi/"
