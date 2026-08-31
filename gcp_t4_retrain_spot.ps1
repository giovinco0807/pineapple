param(
    [Parameter(Position = 0)]
    [ValidateSet("create", "upload", "setup", "run", "status", "download", "delete", "all")]
    [string]$Action = "status"
)

$ErrorActionPreference = "Stop"

$env:CLOUDSDK_CORE_ACCOUNT = "giovinco.080807@gmail.com"

$Project = "ofc-solver-485418"
$Zone = "us-east1-c"
$Instance = "ofc-t4train-spot-0"
$MachineType = "n1-standard-4"
$GpuType = "nvidia-tesla-t4"
$GpuCount = 1
$ImageFamily = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
$ImageProject = "deeplearning-platform-release"
$DiskSize = "150GB"

$GcsBucket = "gs://ofc-solver-results"
$GcsDataset = "t4_dataset_v2"
$GcsResults = "t4_oracle_v2_retrain"

$LocalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LocalDataDir = Join-Path $LocalRoot "ai\data\t4_dataset_v2"
$LocalResultDir = Join-Path $LocalRoot "ai\data\t4_oracle_v2_retrain_gcp"

function Require-Command($Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name is not available on PATH. Install Google Cloud SDK or run this from Google Cloud Shell."
    }
}

function Create-VM {
    Write-Host "=== Creating Spot GPU VM: $Instance ==="
    gcloud compute instances create $Instance `
        --project=$Project `
        --zone=$Zone `
        --machine-type=$MachineType `
        --image-family=$ImageFamily `
        --image-project=$ImageProject `
        --boot-disk-size=$DiskSize `
        --boot-disk-type="pd-balanced" `
        --accelerator="type=$GpuType,count=$GpuCount" `
        --maintenance-policy=TERMINATE `
        --provisioning-model=SPOT `
        --instance-termination-action=STOP `
        --scopes="default,storage-rw"
    Write-Host "=== Waiting 60s for boot + GPU driver init ==="
    Start-Sleep -Seconds 60
}

function Upload-CodeAndData {
    Write-Host "=== Packing training code ==="
    Push-Location $LocalRoot
    tar czf ofc_t4_train_code.tar.gz `
        ai/__init__.py `
        ai/engine/__init__.py `
        ai/engine/encoding.py `
        ai/engine/action_space.py `
        ai/training/__init__.py `
        ai/training/train_t4_oracle.py `
        ai/training/evaluate_t4_oracle.py

    Write-Host "=== Uploading code to VM ==="
    gcloud compute ssh $Instance --zone=$Zone --command="mkdir -p ~/ofc-pineapple"
    gcloud compute scp ofc_t4_train_code.tar.gz "${Instance}:/tmp/ofc_t4_train_code.tar.gz" --zone=$Zone
    gcloud compute ssh $Instance --zone=$Zone --command="cd ~/ofc-pineapple && tar xzf /tmp/ofc_t4_train_code.tar.gz && rm /tmp/ofc_t4_train_code.tar.gz"
    Remove-Item -LiteralPath "ofc_t4_train_code.tar.gz" -Force
    Pop-Location

    Write-Host "=== Uploading T4 dataset to GCS ==="
    gsutil -m rsync -r $LocalDataDir "$GcsBucket/$GcsDataset"

    Write-Host "=== Downloading T4 dataset from GCS to VM ==="
    gcloud compute ssh $Instance --zone=$Zone --command="mkdir -p ~/ofc-pineapple/ai/data/t4_dataset_v2 && gsutil -m rsync -r $GcsBucket/$GcsDataset ~/ofc-pineapple/ai/data/t4_dataset_v2"
}

function Setup-VM {
    Write-Host "=== Setting up VM ==="
    $cmd = @'
set -e
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -m pip install -q numpy torch
cd ~/ofc-pineapple
touch ai/__init__.py ai/training/__init__.py
python3 - << 'PY'
import torch, numpy
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
print('numpy', numpy.__version__)
PY
'@
    gcloud compute ssh $Instance --zone=$Zone --command=$cmd
}

function Run-Training {
    Write-Host "=== Starting T4 retraining on $Instance ==="
    $cmd = @'
cd ~/ofc-pineapple
mkdir -p ai/data/t4_oracle_v2_retrain
export PYTHONUNBUFFERED=1
nohup python3 -u ai/training/train_t4_oracle.py \
  --data-dir ai/data/t4_dataset_v2 \
  --save-dir ai/data/t4_oracle_v2_retrain \
  --epochs 240 \
  --batch-size 4096 \
  --lr 3e-4 \
  --temp-start 3.0 \
  --temp-end 1.0 \
  --value-weight 0.2 \
  --eval-every 5 \
  --device cuda \
  > ai/data/t4_oracle_v2_retrain/train.log 2>&1 &
echo "Training PID: $!"
'@
    gcloud compute ssh $Instance --zone=$Zone --command=$cmd
}

function Status-VM {
    Write-Host "=== VM status ==="
    gcloud compute instances describe $Instance --zone=$Zone --format="table(name,status,machineType.basename(),scheduling.provisioningModel)"
    Write-Host ""
    Write-Host "=== Training log tail ==="
    gcloud compute ssh $Instance --zone=$Zone --command="tail -80 ~/ofc-pineapple/ai/data/t4_oracle_v2_retrain/train.log 2>/dev/null || echo 'train.log not found yet'; echo; pgrep -af train_t4_oracle.py || true"
}

function Download-Results {
    Write-Host "=== Uploading results from VM to GCS ==="
    gcloud compute ssh $Instance --zone=$Zone --command="gsutil -m rsync -r ~/ofc-pineapple/ai/data/t4_oracle_v2_retrain $GcsBucket/$GcsResults"

    Write-Host "=== Downloading results to local ==="
    New-Item -ItemType Directory -Force -Path $LocalResultDir | Out-Null
    gsutil -m rsync -r "$GcsBucket/$GcsResults" $LocalResultDir
    Write-Host "Saved to $LocalResultDir"
}

function Delete-VM {
    Write-Host "=== Deleting VM: $Instance ==="
    gcloud compute instances delete $Instance --zone=$Zone --quiet
}

Require-Command gcloud
Require-Command gsutil

switch ($Action) {
    "create" { Create-VM }
    "upload" { Upload-CodeAndData }
    "setup" { Setup-VM }
    "run" { Run-Training }
    "status" { Status-VM }
    "download" { Download-Results }
    "delete" { Delete-VM }
    "all" {
        Create-VM
        Upload-CodeAndData
        Setup-VM
        Run-Training
    }
}
