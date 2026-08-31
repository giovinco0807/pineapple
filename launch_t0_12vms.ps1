# Launch 12 spot VMs for parallel T0 batch evaluation
# Each VM: c2d-highcpu-16, spot, 5 hands, [10,6,3] nesting, 50 samples
# Uses metadata-from-file for startup script + per-VM metadata for seed/hands

$PROJECT = "ofc-solver-485418"
$ZONE = "us-central1-a"

$configs = @(
    @{Name="t0-hifi-01"; Seed=10001},
    @{Name="t0-hifi-02"; Seed=20001},
    @{Name="t0-hifi-03"; Seed=30001},
    @{Name="t0-hifi-04"; Seed=40001},
    @{Name="t0-hifi-05"; Seed=50001},
    @{Name="t0-hifi-06"; Seed=60001},
    @{Name="t0-hifi-07"; Seed=70001},
    @{Name="t0-hifi-08"; Seed=80001},
    @{Name="t0-hifi-09"; Seed=90001},
    @{Name="t0-hifi-10"; Seed=100001},
    @{Name="t0-hifi-11"; Seed=110001},
    @{Name="t0-hifi-12"; Seed=120001}
)

Write-Host "=== Launching 12 VMs x 5 hands = 60 hands ==="
Write-Host ""

foreach ($c in $configs) {
    $n = $c.Name
    $s = $c.Seed
    Write-Host "Creating $n (seed=$s)..."
    
    gcloud compute instances create $n `
        --project=$PROJECT `
        --zone=$ZONE `
        --machine-type=c2d-highcpu-16 `
        --provisioning-model=SPOT `
        --instance-termination-action=STOP `
        --scopes=cloud-platform `
        --image-family=debian-12 `
        --image-project=debian-cloud `
        --boot-disk-size=10GB `
        --metadata=batch-seed=$s,batch-hands=5 `
        --metadata-from-file=startup-script=gcp_t0_startup.sh `
        2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK: $n created"
    } else {
        Write-Host "  FAIL: $n"
    }
    Start-Sleep -Seconds 1
}

Write-Host ""
Write-Host "=== Done! Monitor: ==="
Write-Host "gcloud compute instances list --filter='name~t0-hifi' --project=$PROJECT"
Write-Host "gcloud storage ls gs://ofc-solver-485418/results/hifi/"
