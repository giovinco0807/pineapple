param(
    [int]$Count = 60,
    [int]$StartIndex = 0,
    [string]$Project = "ofc-solver-485418",
    [string]$MachineType = "n2-standard-4",
    [int]$MaxSeconds = 10800,
    [int]$NSamples = 1,
    [int]$ChunkSize = 100,
    [int]$States = 100000000,
    [string]$RunId = "run_20260510",
    [ValidateSet("random", "heuristic", "model")]
    [string]$T0BBPolicy = "heuristic",
    [ValidateSet("random", "heuristic", "model")]
    [string]$T0OppPolicy = "heuristic",
    [ValidateSet("bb", "btn", "mixed")]
    [string]$Position = "bb"
)

$ErrorActionPreference = "Continue"

$zones = @(
    "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f",
    "us-west1-a", "us-west1-b", "us-west1-c",
    "us-east1-b", "us-east1-c", "us-east1-d",
    "us-east4-a", "us-east4-b", "us-east4-c",
    "us-west2-a", "us-west2-b", "us-west2-c",
    "us-west3-a", "us-west3-b", "us-west3-c",
    "us-west4-a", "us-west4-b", "us-west4-c",
    "northamerica-northeast1-a", "northamerica-northeast1-b", "northamerica-northeast1-c",
    "northamerica-northeast2-a", "northamerica-northeast2-b", "northamerica-northeast2-c"
)

for ($i = 0; $i -lt $Count; $i++) {
    $zone = $zones[$i % $zones.Count]
    $workerId = $StartIndex + $i
    $name = "ofc-t0hu-$Position-$workerId"
    Write-Host "Creating $name in $zone ..."
    gcloud compute instances create $name `
        --project=$Project `
        --zone=$zone `
        --machine-type=$MachineType `
        --provisioning-model=SPOT `
        --instance-termination-action=DELETE `
        --boot-disk-size=30GB `
        --boot-disk-type=pd-balanced `
        --image-family=ubuntu-2204-lts `
        --image-project=ubuntu-os-cloud `
        --scopes=https://www.googleapis.com/auth/cloud-platform `
        --metadata="worker-id=$workerId,max-seconds=$MaxSeconds,n-samples=$NSamples,chunk-size=$ChunkSize,states=$States,position=$Position,run-id=$RunId,t0-bb-policy=$T0BBPolicy,t0-opp-policy=$T0OppPolicy" `
        --metadata-from-file="startup-script=gcp_t0_hu_oracle_startup.sh" `
        --quiet
}

Write-Host ""
Write-Host "Monitor:"
Write-Host "  gcloud compute instances list --filter='name~ofc-t0hu' --project=$Project"
Write-Host "  gcloud storage ls -r gs://ofc-solver-485418/t0_hu_oracle/results/$RunId/$Position/"
