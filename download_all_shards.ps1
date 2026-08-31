# Download all MC teacher shards from GCP VMs to D:\ofc_data\mc_teacher
# Run this from PowerShell

$ErrorActionPreference = "Continue"
$dest = "D:\ofc_data\mc_teacher"
New-Item -ItemType Directory -Path $dest -Force | Out-Null

$project = "ofc-solver-485418"
$sims = 50
$numVMs = 10
$workersPerVM = 8

# Zone mapping (same as gcp_mc_teacher.sh)
function Get-Zone($i) {
    if ($i -eq 0 -or $i -ge 8) { return "us-east1-b" }
    else { return "us-central1-a" }
}

$totalRecords = 0

for ($i = 0; $i -lt $numVMs; $i++) {
    $instance = "ofc-teacher-$i"
    $zone = Get-Zone $i
    Write-Host "--- VM${i}: $instance ($zone) ---"

    for ($w = 0; $w -lt $workersPerVM; $w++) {
        $sid = $i * $workersPerVM + $w
        $remoteData = "~/ofc-pineapple/teacher_results/mc_s${sims}_shard${sid}.jsonl"
        $remoteLogs = "~/ofc-pineapple/teacher_results/shard${sid}.log"
        $localData  = "$dest\mc_s${sims}_shard${sid}.jsonl"
        $localLog   = "$dest\shard${sid}.log"

        Write-Host -NoNewline "  shard${sid}: "

        # Download via gcloud scp
        $null = gcloud compute scp "${instance}:${remoteData}" "$localData" `
            --zone="$zone" --project="$project" 2>&1

        if (Test-Path $localData) {
            $lines = (Get-Content $localData -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
            $hands = [int]($lines / 6)
            $totalRecords += $lines
            Write-Host "$lines records (~$hands hands)"
        } else {
            Write-Host "NOT FOUND"
        }

        # Download log
        $null = gcloud compute scp "${instance}:${remoteLogs}" "$localLog" `
            --zone="$zone" --project="$project" 2>&1
    }
    Write-Host ""
}

$totalHands = [int]($totalRecords / 6)
Write-Host "============================================"
Write-Host "Total records: $totalRecords"
Write-Host "Total hands:   $totalHands / 10000 ($([int]($totalHands * 100 / 10000))% complete)"
Write-Host ""
Write-Host "Merging shards..."
$merged = "$dest\mc_s${sims}_merged.jsonl"
Get-Content "$dest\mc_s${sims}_shard*.jsonl" | Set-Content $merged -Encoding UTF8
$mergedLines = (Get-Content $merged | Measure-Object -Line).Lines
Write-Host "Merged: $mergedLines records -> $merged"
Write-Host "============================================"
