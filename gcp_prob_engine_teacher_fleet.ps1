param(
    [Parameter(Position = 0)]
    [ValidateSet("create", "upload", "setup", "run", "status", "download", "delete", "all")]
    [string]$Action = "status",
    [string]$Project = "hiroshimapokerclub-app",
    [string]$Prefix = "ofc-pe-r20-20260518",
    [int]$NumVMs = 4,
    [string]$MachineType = "e2-highcpu-8",
    [string]$DiskSize = "30GB",
    [int]$TotalHands = 2000,
    [int]$Sims = 20,
    [int]$Seed = 20260519,
    [int]$WorkersPerVM = 2,
    [int]$RayonThreads = 4,
    [string]$RunId = "prob-engine-r20-fleet-20260518-vm2000-w2r4"
)

$ErrorActionPreference = "Stop"

$Zones = @(
    "us-central1-a",
    "us-central1-b",
    "us-east1-b",
    "us-east1-c",
    "us-west1-b",
    "us-central1-b",
    "us-east1-b",
    "us-west4-a"
)
$LocalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LocalDest = Join-Path $LocalRoot "ai\models\candidate_runs\$RunId"

function VmName([int]$Index) {
    return "$Prefix-$Index"
}

function VmZone([int]$Index) {
    return $Zones[$Index % $Zones.Count]
}

function Invoke-ForEachVm([scriptblock]$Block) {
    $jobs = @()
    foreach ($i in 0..($NumVMs - 1)) {
        $idx = $i
        $jobs += Start-Job -ScriptBlock $Block -ArgumentList $idx, (VmName $idx), (VmZone $idx), $Project, $LocalRoot, $LocalDest, $TotalHands, $Sims, $Seed, $WorkersPerVM, $RunId, $RayonThreads
    }
    Receive-Job -Job $jobs -Wait -AutoRemoveJob
}

function New-CodeArchive {
    $tarPath = Join-Path $env:TEMP "ofc_pe_teacher_code.tar.gz"
    Push-Location $LocalRoot
    try {
        if (Test-Path $tarPath) { Remove-Item $tarPath -Force }
        tar --exclude="ai/rust_solver/target" -czf $tarPath `
            generate_mc_teacher.py `
            ai/__init__.py `
            ai/prob_engine_wrapper.py `
            ai/rust_solver_wrapper.py `
            ai/engine `
            ai/config `
            ai/rust_solver
    } finally {
        Pop-Location
    }
    return $tarPath
}

function Create-VMs {
    Write-Host "Creating $NumVMs spot VMs: $MachineType, total target $TotalHands hands"
    Invoke-ForEachVm {
        param($i, $name, $zone, $project)
        Write-Host "Creating $name in $zone"
        gcloud compute instances create $name `
            --project=$project `
            --zone=$zone `
            --machine-type=$using:MachineType `
            --image-family="ubuntu-2204-lts" `
            --image-project="ubuntu-os-cloud" `
            --boot-disk-size=$using:DiskSize `
            --boot-disk-type="pd-standard" `
            --provisioning-model=SPOT `
            --instance-termination-action=STOP `
            --scopes="default,storage-rw" `
            --labels="purpose=ofc-pe-teacher,run=$using:RunId" `
            --quiet 2>&1 | ForEach-Object { Write-Host $_ }
    }
}

function Upload-Code {
    $tarPath = New-CodeArchive
    Write-Host "Archive: $tarPath"
    Invoke-ForEachVm {
        param($i, $name, $zone, $project, $root)
        Write-Host "Uploading code to $name"
        gcloud compute ssh $name --project=$project --zone=$zone --command="mkdir -p ~/ofc-pineapple" 2>$null
        gcloud compute scp $using:tarPath "${name}:/tmp/ofc_pe_teacher_code.tar.gz" --project=$project --zone=$zone 2>$null
        gcloud compute ssh $name --project=$project --zone=$zone --command="cd ~/ofc-pineapple && tar xzf /tmp/ofc_pe_teacher_code.tar.gz && rm /tmp/ofc_pe_teacher_code.tar.gz" 2>$null
        Write-Host "$name upload done"
    }
}

function Setup-VMs {
    $setup = @'
#!/bin/bash
set -euo pipefail
echo "--- install packages ---"
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-dev python3-pip curl build-essential
echo "--- python deps ---"
python3 -m venv "$HOME/.venvs/ofc-teacher"
"$HOME/.venvs/ofc-teacher/bin/python" -m pip install --upgrade pip -q
"$HOME/.venvs/ofc-teacher/bin/python" -m pip install numpy -q
echo "--- rust ---"
if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"
echo "--- build prob_engine ---"
cd "$HOME/ofc-pineapple/ai/rust_solver"
cargo build --release -p prob_engine
echo "setup complete"
'@
    $tmp = Join-Path $env:TEMP "gcp_pe_setup.sh"
    Set-Content -Path $tmp -Value $setup -Encoding ascii
    Invoke-ForEachVm {
        param($i, $name, $zone, $project)
        Write-Host "Setting up $name"
        gcloud compute scp $using:tmp "${name}:/tmp/gcp_pe_setup.sh" --project=$project --zone=$zone 2>$null
        $out = gcloud compute ssh $name --project=$project --zone=$zone --command="sed -i 's/\r$//' /tmp/gcp_pe_setup.sh && bash /tmp/gcp_pe_setup.sh" 2>&1
        $code = $LASTEXITCODE
        $out | Select-Object -Last 20
        if ($code -ne 0) { throw "$name setup failed with exit code $code" }
        Write-Host "$name setup done"
    }
}

function Run-Teachers {
    Write-Host "Starting generation: $TotalHands hands, sims=$Sims, $NumVMs VMs x $WorkersPerVM workers"
    Invoke-ForEachVm {
        param($i, $name, $zone, $project, $root, $dest, $totalHands, $sims, $seed, $workers, $runId, $rayonThreads)
        $vmStart = [math]::Floor($i * $totalHands / $using:NumVMs)
        $vmEnd = [math]::Floor(($i + 1) * $totalHands / $using:NumVMs)
        $vmHands = $vmEnd - $vmStart
        $remote = @"
set -e
cd ~/ofc-pineapple
mkdir -p teacher_results
export PYTHONUNBUFFERED=1
export RAYON_NUM_THREADS=$rayonThreads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY="`$HOME/.venvs/ofc-teacher/bin/python"
[ -x "`$PY" ] || PY=python3
for w in `$(seq 0 $($workers - 1)); do
  hand_start=`$(( $vmStart + w * $vmHands / $workers ))
  hand_end=`$(( $vmStart + (w + 1) * $vmHands / $workers ))
  sid=`$(( $i * $workers + w ))
  outfile="teacher_results/pe_s${sims}_shard`${sid}.jsonl"
  log="teacher_results/shard`${sid}.log"
  nohup "`$PY" -u generate_mc_teacher.py --n-hands $totalHands --seed $seed --sims $sims --mc-turns 0,1,2 --hand-start `$hand_start --hand-end `$hand_end --output `$outfile > `$log 2>&1 &
  echo "worker `$w shard `$sid hands [`$hand_start, `$hand_end)"
done
"@
        $tmp = Join-Path $env:TEMP ("gcp_pe_run_{0}.sh" -f $i)
        Set-Content -Path $tmp -Value $remote -Encoding ascii
        gcloud compute scp $tmp "${name}:/tmp/gcp_pe_run.sh" --project=$project --zone=$zone 2>$null
        $out = gcloud compute ssh $name --project=$project --zone=$zone --command="sed -i 's/\r$//' /tmp/gcp_pe_run.sh && bash /tmp/gcp_pe_run.sh" 2>&1
        $code = $LASTEXITCODE
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
        $out | ForEach-Object { Write-Host $_ }
        if ($code -ne 0) { throw "$name run failed with exit code $code" }
        Write-Host "$name started range [$vmStart, $vmEnd)"
    }
}

function Check-Status {
    Invoke-ForEachVm {
        param($i, $name, $zone, $project, $root, $dest, $totalHands, $sims, $seed, $workers)
        $remote = @"
cd ~/ofc-pineapple 2>/dev/null || exit 0
running=`$(pgrep -fc 'generate_mc_teacher.py' || true)
summaries=0
records=0
for f in teacher_results/pe_s${sims}_shard*.jsonl; do
  [ -f "`$f" ] || continue
  r=`$(wc -l < "`$f")
  records=`$((records + r))
  s=`$(grep -Ec '"turn"[[:space:]]*:[[:space:]]*-1' "`$f" || true)
  summaries=`$((summaries + s))
done
echo "running=`$running records=`$records hands=`$summaries"
for l in teacher_results/shard*.log; do
  [ -f "`$l" ] || continue
  echo "--- `$(basename `$l) ---"
  tail -1 "`$l"
done | tail -12
"@
        $tmp = Join-Path $env:TEMP ("gcp_pe_status_{0}.sh" -f $i)
        Set-Content -Path $tmp -Value $remote -Encoding ascii
        gcloud compute scp $tmp "${name}:/tmp/gcp_pe_status.sh" --project=$project --zone=$zone 2>$null
        $result = gcloud compute ssh $name --project=$project --zone=$zone --command="sed -i 's/\r$//' /tmp/gcp_pe_status.sh && bash /tmp/gcp_pe_status.sh" 2>$null
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
        Write-Host "[$name $zone]"
        Write-Host $result
    }
}

function Download-Results {
    New-Item -ItemType Directory -Force -Path $LocalDest | Out-Null
    $raw = Join-Path $LocalDest "raw"
    New-Item -ItemType Directory -Force -Path $raw | Out-Null
    Invoke-ForEachVm {
        param($i, $name, $zone, $project, $root, $dest, $totalHands, $sims, $seed, $workers)
        $rawDir = Join-Path $dest "raw"
        foreach ($w in 0..($workers - 1)) {
            $sid = $i * $workers + $w
            gcloud compute ssh $name --project=$project --zone=$zone --command="cat ~/ofc-pineapple/teacher_results/pe_s${sims}_shard${sid}.jsonl 2>/dev/null" 2>$null | Set-Content -Path (Join-Path $rawDir "pe_s${sims}_shard${sid}.jsonl") -Encoding utf8
            gcloud compute ssh $name --project=$project --zone=$zone --command="cat ~/ofc-pineapple/teacher_results/shard${sid}.log 2>/dev/null" 2>$null | Set-Content -Path (Join-Path $rawDir "shard${sid}.log") -Encoding utf8
        }
        Write-Host "$name downloaded"
    }
    Get-ChildItem $raw -Filter "pe_s${Sims}_shard*.jsonl" | Sort-Object Name | Get-Content | Set-Content (Join-Path $LocalDest "teacher.jsonl") -Encoding utf8
    Write-Host "Merged to $(Join-Path $LocalDest 'teacher.jsonl')"
}

function Delete-VMs {
    Invoke-ForEachVm {
        param($i, $name, $zone, $project)
        Write-Host "Deleting $name"
        gcloud compute instances delete $name --project=$project --zone=$zone --quiet 2>$null
    }
}

switch ($Action) {
    "create" { Create-VMs }
    "upload" { Upload-Code }
    "setup" { Setup-VMs }
    "run" { Run-Teachers }
    "status" { Check-Status }
    "download" { Download-Results }
    "delete" { Delete-VMs }
    "all" {
        Create-VMs
        Upload-Code
        Setup-VMs
        Run-Teachers
        Check-Status
    }
}
