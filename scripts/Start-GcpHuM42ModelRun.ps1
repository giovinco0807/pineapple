param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$TeacherRunName = "regular-hu-m42-c2e8-gate40-20260713-1335",
    [string]$RunName = ("regular-hu-m42-ranker-i30-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [int]$Iterations = 30,
    [int]$CrossFitFolds = 5,
    [string]$MachineType = "c4-highcpu-16",
    [string[]]$FallbackMachineTypes = @("c4-standard-16", "n2-highcpu-16"),
    [string[]]$Zones = @("asia-northeast1-b", "asia-northeast1-c", "us-central1-a"),
    [int]$BootDiskGb = 30,
    [switch]$CreateInstance,
    [switch]$ResumeExisting,
    [switch]$NoSelfDelete,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ExpectedTeacherSourceSha256 = "902c200873927fb31ffd1204588d4f0492880ab7ff29fb579125742c54e12530"
$ExpectedTrainerSha256 = "bda861cd0f72242e904d7bddbb81ca2bbb633f17d1437e4c7b6e56df8625ac5f"
$Thresholds = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99,1"
$ModelId = "hu-m4-t1-second-m42-gate40-c2e8-ranker-v2-i$Iterations"

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-TextSha256 {
    param([Parameter(Mandatory = $true)][string]$Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($Text)
        return ([System.BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )
    Write-BytesCreateNew -Path $Path -Bytes ([System.Text.UTF8Encoding]::new($false).GetBytes($Text))
}

function Write-BytesCreateNew {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][byte[]]$Bytes
    )
    $stream = [System.IO.File]::Open(
        $Path,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try {
        $stream.Write($Bytes, 0, $Bytes.Length)
    }
    finally {
        $stream.Dispose()
    }
}

function Invoke-Gcloud {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & gcloud @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "gcloud failed ($exitCode): gcloud $($Arguments -join ' ')`n$($output -join [Environment]::NewLine)"
    }
    return @($output)
}

function Test-GcsObject {
    param([Parameter(Mandatory = $true)][string]$Uri)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & gcloud storage objects describe $Uri --format=json 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -eq 0) { return $true }
    $message = @($output) -join [Environment]::NewLine
    if ($message -match '(?i)not found|does not exist|No URLs matched|404') {
        return $false
    }
    throw "Unable to determine whether immutable GCS object exists: $Uri`n$message"
}

function Get-ComputeInstancesByName {
    param(
        [Parameter(Mandatory = $true)][string]$InstanceName,
        [Parameter(Mandatory = $true)][string]$CloudProjectId
    )
    $raw = Invoke-Gcloud @(
        "compute", "instances", "list", "--project", $CloudProjectId,
        "--filter", ("name={0}" -f $InstanceName), "--format=json"
    )
    if (-not $raw) { return @() }
    return @(((@($raw) -join "`n") | ConvertFrom-Json))
}

function Convert-ToVmName {
    param([Parameter(Mandatory = $true)][string]$Value)
    $name = ($Value.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-')
    if (-not $name) { throw "RunName does not produce a valid VM name" }
    if ($name -notmatch '^[a-z]') { $name = "m42-$name" }
    if ($name.Length -gt 63) {
        $suffix = (Get-TextSha256 $Value).Substring(0, 8)
        $name = $name.Substring(0, 54).TrimEnd('-') + "-$suffix"
    }
    return $name
}

if ($Iterations -lt 1) { throw "Iterations must be positive" }
if ($CrossFitFolds -lt 2) { throw "CrossFitFolds must be at least two" }
if ($BootDiskGb -lt 20) { throw "BootDiskGb must be at least 20" }
if (-not $Zones -or $Zones.Count -eq 0) { throw "At least one zone is required" }
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw "RunName may only contain letters, numbers, dot, underscore, and dash"
}
if ($TeacherRunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw "TeacherRunName may only contain letters, numbers, dot, underscore, and dash"
}
if ($ResumeExisting -and $DryRun) {
    throw "ResumeExisting and DryRun are mutually exclusive"
}
if ($ResumeExisting -and -not $CreateInstance) {
    throw "ResumeExisting requires CreateInstance; refusing an ambiguous frozen-run operation"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
$startupPath = Join-Path $runDir "startup_hu_m42_model.sh"
$manifestPath = Join-Path $runDir "model_run_manifest.json"
$gcsPrefix = "gs://$Bucket/runs/$RunName"
$manifestUri = "$gcsPrefix/source/model_run_manifest.json"
$startupUri = "$gcsPrefix/source/startup_hu_m42_model.sh"
$doneUri = "$gcsPrefix/results/DONE"
$vmName = Convert-ToVmName $RunName

$remoteManifestExists = $false
if (-not $DryRun) {
    $remoteManifestExists = Test-GcsObject $manifestUri
    if ($ResumeExisting -and -not $remoteManifestExists) {
        throw "ResumeExisting is relaunch-only, but the frozen remote model run does not exist: $manifestUri"
    }
    if (-not $ResumeExisting -and $remoteManifestExists) {
        throw "Model run already exists and is immutable. Use -ResumeExisting -CreateInstance for an incomplete frozen run: $manifestUri"
    }
    if (-not $ResumeExisting -and
        ((Test-Path -LiteralPath $manifestPath) -or (Test-Path -LiteralPath $startupPath))) {
        throw "Local frozen model run files already exist; refusing to overwrite them: $runDir"
    }
}

if ($ResumeExisting) {
    $resumeTempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("hu-m42-model-resume-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $resumeTempDir -Force | Out-Null
    $remoteManifestPath = Join-Path $resumeTempDir "model_run_manifest.json"
    $remoteStartupPath = Join-Path $resumeTempDir "startup_hu_m42_model.sh"
    Invoke-Gcloud @("storage", "cp", $manifestUri, $remoteManifestPath, "--project", $ProjectId) | Out-Null
    $manifest = Get-Content -LiteralPath $remoteManifestPath -Raw | ConvertFrom-Json
    if ($manifest.schema -ne "hu_m42_model_run_manifest_v1" -or
        [string]$manifest.run_name -ne $RunName -or
        [string]$manifest.project_id -ne $ProjectId -or
        [string]$manifest.bucket -ne $Bucket -or
        [string]$manifest.teacher_run_name -ne $TeacherRunName -or
        $manifest.no_runtime_activation -ne $true) {
        throw "Frozen M4.2 model run identity/schema mismatch"
    }
    if ([string]$manifest.source.sha256 -ne $ExpectedTeacherSourceSha256 -or
        [string]$manifest.source.trainer_sha256 -ne $ExpectedTrainerSha256 -or
        [string]$manifest.training_config.action_score_mode -ne "negative_regret_ranker_v2" -or
        [int]$manifest.training_config.iterations -ne $Iterations -or
        [int]$manifest.training_config.cross_fit_folds -ne $CrossFitFolds) {
        throw "Frozen M4.2 model source or training configuration mismatch"
    }
    if ([string]$manifest.startup.uri -ne $startupUri -or
        -not ([string]$manifest.source.uri).StartsWith("gs://$Bucket/runs/$TeacherRunName/")) {
        throw "Frozen M4.2 model run contains an invalid source/startup URI"
    }
    Invoke-Gcloud @("storage", "cp", $startupUri, $remoteStartupPath, "--project", $ProjectId) | Out-Null
    if ((Get-Sha256 $remoteStartupPath) -ne [string]$manifest.startup.sha256) {
        throw "Frozen M4.2 startup script hash verification failed"
    }
    $manifestSha256 = Get-Sha256 $remoteManifestPath
    if (Test-GcsObject $doneUri) {
        throw "Frozen M4.2 model run already has DONE and must not be relaunched: $doneUri"
    }
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
        if ((Get-Sha256 $manifestPath) -ne $manifestSha256) {
            throw "Local model manifest differs from the frozen remote manifest; refusing to overwrite either"
        }
    }
    else {
        Write-BytesCreateNew -Path $manifestPath -Bytes ([System.IO.File]::ReadAllBytes($remoteManifestPath))
    }
    if (Test-Path -LiteralPath $startupPath -PathType Leaf) {
        if ((Get-Sha256 $startupPath) -ne [string]$manifest.startup.sha256) {
            throw "Local startup script differs from the frozen remote startup; refusing to overwrite either"
        }
    }
    else {
        Write-BytesCreateNew -Path $startupPath -Bytes ([System.IO.File]::ReadAllBytes($remoteStartupPath))
    }
    $inputPlan = $manifest.inputs
    $expectedRows = [ordered]@{
        train = [int]$manifest.inputs.train.rows
        calibration = [int]$manifest.inputs.calibration.rows
        locked_holdout = [int]$manifest.inputs.locked_holdout.rows
    }
    $startupSha256 = [string]$manifest.startup.sha256
    if ($null -ne $manifest.compute.vm_name -and [string]$manifest.compute.vm_name) {
        $vmName = [string]$manifest.compute.vm_name
    }
    $MachineType = [string]$manifest.compute.machine_type
    $FallbackMachineTypes = @($manifest.compute.fallback_machine_types | ForEach-Object { [string]$_ })
    $Zones = @($manifest.compute.zones | ForEach-Object { [string]$_ })
    $BootDiskGb = [int]$manifest.compute.boot_disk_gb
}
else {
$teacherOutputDir = Join-Path $repoRoot "outputs/hu_joint_policy/m42_spot/$TeacherRunName"
$teacherRunDir = Join-Path $repoRoot "outputs/gcp_runs/$TeacherRunName"
$receiptPath = Join-Path $teacherOutputDir "receipt.json"
$auditPath = Join-Path $teacherOutputDir "data_audit.json"
$teacherManifestPath = Join-Path $teacherRunDir "manifest.json"
$sourceArchivePath = Join-Path $teacherRunDir "ofc_regular_hu_m42_source.zip"
$trainerPath = Join-Path $repoRoot "src/ofc_regular/train_hu_m4_joint_model.py"

foreach ($required in @($receiptPath, $auditPath, $teacherManifestPath, $sourceArchivePath, $trainerPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required verified M4.2 input is missing: $required"
    }
}

$receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
$audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
$teacherManifest = Get-Content -LiteralPath $teacherManifestPath -Raw | ConvertFrom-Json
if ($receipt.schema -ne "hu_m42_spot_receipt_v1" -or
    $receipt.status -ne "verified_and_audited" -or
    $audit.schema -ne "hu_m4_t1_second_data_audit_v1" -or
    $audit.status -ne "pass") {
    throw "Teacher receipt and data audit must both be verified before model training"
}
if ($receipt.run_name -ne $TeacherRunName -or
    $teacherManifest.schema -ne "hu_m42_spot_manifest_v1" -or
    $teacherManifest.run_name -ne $TeacherRunName -or
    $teacherManifest.project_id -ne $ProjectId -or
    $teacherManifest.bucket -ne $Bucket -or
    -not ([string]$teacherManifest.source_uri).StartsWith("gs://$Bucket/runs/$TeacherRunName/")) {
    throw "Teacher run identity mismatch"
}
if ($teacherManifest.no_runtime_activation -ne $true) {
    throw "Teacher manifest does not preserve the no-runtime-activation boundary"
}
if ((Get-Sha256 $auditPath) -ne [string]$receipt.audit_sha256) {
    throw "Teacher data audit SHA256 does not match its receipt"
}
if ([long](Get-Item -LiteralPath $sourceArchivePath).Length -ne [long]$teacherManifest.source_bytes -or
    (Get-Sha256 $sourceArchivePath) -ne $ExpectedTeacherSourceSha256 -or
    [string]$receipt.source_sha256 -ne $ExpectedTeacherSourceSha256 -or
    [string]$teacherManifest.source_sha256 -ne $ExpectedTeacherSourceSha256) {
    throw "Frozen teacher source archive SHA256 mismatch"
}
if ((Get-Sha256 $trainerPath) -ne $ExpectedTrainerSha256) {
    throw "Local trainer differs from the trainer frozen in the verified source archive"
}

$splitFiles = [ordered]@{
    train = Join-Path $teacherOutputDir "train.jsonl"
    calibration = Join-Path $teacherOutputDir "calibration.jsonl"
    locked_holdout = Join-Path $teacherOutputDir "locked_holdout.jsonl"
}
$expectedRows = [ordered]@{ train = 20; calibration = 10; locked_holdout = 10 }
$inputPlan = [ordered]@{}
foreach ($split in $splitFiles.Keys) {
    $path = $splitFiles[$split]
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Merged verified split is missing: $path"
    }
    $receiptSplit = $receipt.merged.$split
    $sha = Get-Sha256 $path
    if ($sha -ne [string]$receiptSplit.sha256 -or [int]$receiptSplit.rows -ne [int]$expectedRows[$split]) {
        throw "Merged $split input does not match the verified receipt"
    }
    $inputPlan[$split] = [ordered]@{
        local_path = $path
        rows = [int]$expectedRows[$split]
        bytes = [long](Get-Item -LiteralPath $path).Length
        sha256 = $sha
        uri = "gs://$Bucket/runs/$RunName/inputs/$split.jsonl"
    }
}

$startup = @'
#!/usr/bin/env bash
set -Eeuo pipefail
LOG=/var/log/hu_m42_model_startup.log
exec > >(tee -a "$LOG") 2>&1

META=http://metadata.google.internal/computeMetadata/v1
meta() { curl -fsS -H 'Metadata-Flavor: Google' "$META/instance/attributes/$1"; }
project_meta() { curl -fsS -H 'Metadata-Flavor: Google' "$META/project/project-id"; }
instance_meta() { curl -fsS -H 'Metadata-Flavor: Google' "$META/instance/$1"; }
token() {
  curl -fsS -H 'Metadata-Flavor: Google' "$META/instance/service-accounts/default/token" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
}
urlencode() { python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1],safe=""))' "$1"; }

PROJECT_ID="$(project_meta)"
BUCKET="$(meta BUCKET)"
RUN_NAME="$(meta RUN_NAME)"
PREFIX="runs/${RUN_NAME}"
MANIFEST_OBJECT="$(meta MANIFEST_OBJECT)"
MANIFEST_SHA256="$(meta MANIFEST_SHA256)"
SELF_DELETE="$(meta SELF_DELETE)"
INSTANCE_NAME="$(instance_meta name)"
ZONE="$(instance_meta zone)"; ZONE="${ZONE##*/}"
WORK=/work/hu_m42_model
OUT="$WORK/out"
mkdir -p "$WORK" "$OUT"
STATE=booting
HEARTBEAT_PID=""

gcs_download() {
  local object="$1" destination="$2" encoded access
  encoded="$(urlencode "$object")"; access="$(token)"
  curl --retry 5 --retry-all-errors -fsSL \
    -H "Authorization: Bearer ${access}" \
    "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/${encoded}?alt=media" \
    -o "$destination"
}
gcs_upload() {
  local source="$1" object="$2" encoded access
  [[ -f "$source" ]] || return 0
  encoded="$(urlencode "$object")"; access="$(token)"
  curl --retry 5 --retry-all-errors -fsS -X POST \
    -H "Authorization: Bearer ${access}" \
    -H 'Content-Type: application/octet-stream' \
    --data-binary "@${source}" \
    "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&name=${encoded}" >/dev/null
}
gcs_upload_immutable() {
  local source="$1" object="$2" encoded access
  [[ -f "$source" ]] || return 1
  encoded="$(urlencode "$object")"; access="$(token)"
  curl --retry 5 --retry-all-errors -fsS -X POST \
    -H "Authorization: Bearer ${access}" \
    -H 'Content-Type: application/octet-stream' \
    --data-binary "@${source}" \
    "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&name=${encoded}&ifGenerationMatch=0" >/dev/null
}
write_status() {
  local state="$1" exit_code="$2"
  python3 - "$OUT/status.json" "$RUN_NAME" "$state" "$exit_code" "$INSTANCE_NAME" "$ZONE" "$MANIFEST_SHA256" <<'PY'
import datetime,json,sys
path,run,state,exit_code,instance,zone,manifest_sha=sys.argv[1:]
with open(path,"w",encoding="utf-8") as f:
    json.dump({"schema":"hu_m42_model_status_v1","run_name":run,"state":state,
               "exit_code":int(exit_code),"instance":instance,"zone":zone,
               "run_manifest_sha256":manifest_sha,
               "updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat()},f,sort_keys=True)
PY
}
self_delete() {
  [[ "$SELF_DELETE" == 1 ]] || return 0
  local access
  access="$(token)" || return 0
  curl -fsS -X DELETE -H "Authorization: Bearer ${access}" \
    "https://compute.googleapis.com/compute/v1/projects/${PROJECT_ID}/zones/${ZONE}/instances/${INSTANCE_NAME}" >/dev/null || true
}
on_exit() {
  local code=$?
  [[ -z "$HEARTBEAT_PID" ]] || kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true
  if [[ $code -ne 0 ]]; then
    STATE=failed
    write_status failed "$code" || true
    gcs_upload "$OUT/status.json" "$PREFIX/results/status.json" || true
    gcs_upload "$LOG" "$PREFIX/results/startup.log" || true
  fi
  if [[ "$SELF_DELETE" == 1 ]]; then
    self_delete || true
  fi
}
trap on_exit EXIT

write_status booting 0
gcs_upload "$OUT/status.json" "$PREFIX/results/status.json"

apt-get update -y
apt-get install -y unzip curl ca-certificates python3 python3-venv python3-pip

gcs_download "$MANIFEST_OBJECT" "$WORK/model_run_manifest.json"
echo "${MANIFEST_SHA256}  $WORK/model_run_manifest.json" | sha256sum -c -

eval "$(python3 - "$WORK/model_run_manifest.json" <<'PY'
import json,shlex,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
for key,value in {
 "SOURCE_OBJECT":m["source"]["uri"].split("/",3)[3],
 "SOURCE_SHA256":m["source"]["sha256"],
 "TRAIN_OBJECT":m["inputs"]["train"]["uri"].split("/",3)[3],
 "TRAIN_SHA256":m["inputs"]["train"]["sha256"],
 "CALIBRATION_OBJECT":m["inputs"]["calibration"]["uri"].split("/",3)[3],
 "CALIBRATION_SHA256":m["inputs"]["calibration"]["sha256"],
 "HOLDOUT_OBJECT":m["inputs"]["locked_holdout"]["uri"].split("/",3)[3],
 "HOLDOUT_SHA256":m["inputs"]["locked_holdout"]["sha256"],
 "MODEL_ID":m["training_config"]["model_id"],
 "ITERATIONS":m["training_config"]["iterations"],
 "FOLDS":m["training_config"]["cross_fit_folds"],
 "THRESHOLDS":m["training_config"]["thresholds"],
}.items(): print(f"{key}={shlex.quote(str(value))}")
PY
)"

gcs_download "$SOURCE_OBJECT" "$WORK/source.zip"
gcs_download "$TRAIN_OBJECT" "$WORK/train.jsonl"
gcs_download "$CALIBRATION_OBJECT" "$WORK/calibration.jsonl"
gcs_download "$HOLDOUT_OBJECT" "$WORK/locked_holdout.jsonl"
echo "${SOURCE_SHA256}  $WORK/source.zip" | sha256sum -c -
echo "${TRAIN_SHA256}  $WORK/train.jsonl" | sha256sum -c -
echo "${CALIBRATION_SHA256}  $WORK/calibration.jsonl" | sha256sum -c -
echo "${HOLDOUT_SHA256}  $WORK/locked_holdout.jsonl" | sha256sum -c -

mkdir -p "$WORK/repo"
unzip -q "$WORK/source.zip" -d "$WORK/repo"
cd "$WORK/repo"
echo 'bda861cd0f72242e904d7bddbb81ca2bbb633f17d1437e4c7b6e56df8625ac5f  src/ofc_regular/train_hu_m4_joint_model.py' | sha256sum -c -
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'numpy==2.2.6' 'scikit-learn==1.8.0'

STATE=training
write_status training 0
gcs_upload "$OUT/status.json" "$PREFIX/results/status.json"
(
  while true; do
    sleep 60
    python3 - "$OUT/heartbeat.json" "$RUN_NAME" "$MANIFEST_SHA256" <<'PY'
import datetime,json,sys
with open(sys.argv[1],"w",encoding="utf-8") as f:
    json.dump({"schema":"hu_m42_model_heartbeat_v1","run_name":sys.argv[2],"state":"training",
               "run_manifest_sha256":sys.argv[3],
               "updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat()},f,sort_keys=True)
PY
    gcs_upload "$OUT/heartbeat.json" "$PREFIX/results/heartbeat.json" || true
    gcs_upload "$LOG" "$PREFIX/results/startup.log" || true
  done
) &
HEARTBEAT_PID=$!

PYTHONPATH=src python -m ofc_regular.train_hu_m4_joint_model \
  --train "$WORK/train.jsonl" \
  --calibration "$WORK/calibration.jsonl" \
  --locked-holdout "$WORK/locked_holdout.jsonl" \
  --output-model "$OUT/model.pkl" \
  --manifest-output "$OUT/training_manifest.json" \
  --model-id "$MODEL_ID" \
  --action-score-mode negative_regret_ranker_v2 \
  --cross-fit-folds "$FOLDS" \
  --near-best-margin 0.5 \
  --minimum-safe-teacher-gain 0 \
  --iterations "$ITERATIONS" \
  --max-leaf-nodes 31 \
  --learning-rate 0.05 \
  --l2-regularization 1 \
  --seed 2026071801 \
  --safety-fit-ratio 0.5 \
  --safety-split-seed 2026071802 \
  --minimum-safety-fit-samples 5 \
  --minimum-threshold-lock-samples 5 \
  --thresholds "$THRESHOLDS" \
  --minimum-calibration-fires 2 \
  --maximum-false-positive-rate 0.30 \
  --maximum-p95-loss 25 \
  --maximum-p99-loss 40 > "$OUT/training_stdout.json"

kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true
wait "$HEARTBEAT_PID" >/dev/null 2>&1 || true
HEARTBEAT_PID=""

PYTHONPATH=src python - "$OUT/model.pkl" "$OUT/training_manifest.json" "$OUT/validation.json" "$RUN_NAME" "$MANIFEST_SHA256" "$FOLDS" <<'PY'
import datetime,hashlib,json,math,sys
from ofc_regular.hu_m4_joint_model import HuM4JointActionModel,NEGATIVE_REGRET_ACTION_SCORE_MODE
model_path,manifest_path,out_path,run_name,run_manifest_sha,folds_raw=sys.argv[1:]
folds=int(folds_raw)
m=json.load(open(manifest_path,encoding="utf-8"))
assert m["schema"] == "hu_m4_t1_joint_training_manifest_v2"
assert m["training_config"]["action_score_mode"] == NEGATIVE_REGRET_ACTION_SCORE_MODE
assert m["training_config"]["cross_fit_folds"] == folds
assert m["cross_fit"]["status"] == "pass"
assert m["cross_fit"]["folds"] == folds
assert m["cross_fit"]["fold_assignment"]["identity_leakage_count"] == 0
assert m["cross_fit"]["oof_coverage"]["base_head_prediction_counts"] == [1]
assert m["cross_fit"]["oof_coverage"]["meta_rank_prediction_counts"] == [1]
assert m["cross_fit"]["oof_coverage"]["uncertainty_prediction_counts"] == [1]
assert m["cross_fit"]["predictor_lineage"]["identity_leakage_count"] == 0
assert m["cross_fit"]["predictor_lineage"]["all_outer_validation_identities_excluded"] is True
assert m["cross_fit"]["uncertainty_target"]["paired_delta_schema_v2_samples"] == 20
assert m["split_integrity"]["seed_overlap_count"] == 0
assert m["hidden_discard_safety"]["status"] == "pass"
assert m["calibration_partition"]["status"] == "pass"
assert m["locked_holdout_used_for_threshold_or_training"] is False
assert m["teacher_value_runtime_gate"] is False
assert m["runtime_lock"]["candidate_model_sha256"] == m["runtime_lock"]["safety_model_sha256"]
h=lambda p: hashlib.sha256(open(p,"rb").read()).hexdigest()
assert h(model_path) == m["runtime_lock"]["candidate_model_sha256"]
model=HuM4JointActionModel.load(model_path)
assert model.action_score_mode == NEGATIVE_REGRET_ACTION_SCORE_MODE
assert model.meta_rank_estimator is not None
result={
 "schema":"hu_m42_model_validation_v1","status":"pass","run_name":run_name,
 "run_manifest_sha256":run_manifest_sha,"model_sha256":h(model_path),
 "training_manifest_sha256":h(manifest_path),"promotion_status":m["promotion_status"],
 "selected_threshold":m["calibration"]["selected_threshold"],
 "safety_estimator_class":type(model.safety_estimator).__name__,
 "calibration":m["calibration"],"locked_holdout":m["locked_holdout"],
 "teacher_value_status":m["teacher_value_status"],"no_runtime_activation":True,
 "validated_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
json.dump(result,open(out_path,"w",encoding="utf-8"),indent=2,sort_keys=True)
PY

MODEL_SHA256="$(sha256sum "$OUT/model.pkl" | cut -d' ' -f1)"
TRAINING_MANIFEST_SHA256="$(sha256sum "$OUT/training_manifest.json" | cut -d' ' -f1)"
VALIDATION_SHA256="$(sha256sum "$OUT/validation.json" | cut -d' ' -f1)"
STDOUT_SHA256="$(sha256sum "$OUT/training_stdout.json" | cut -d' ' -f1)"

gcs_upload "$OUT/model.pkl" "$PREFIX/results/model.pkl"
gcs_upload "$OUT/training_manifest.json" "$PREFIX/results/training_manifest.json"
gcs_upload "$OUT/validation.json" "$PREFIX/results/validation.json"
gcs_upload "$OUT/training_stdout.json" "$PREFIX/results/training_stdout.json"
gcs_upload "$LOG" "$PREFIX/results/startup.log"
write_status complete 0
gcs_upload "$OUT/status.json" "$PREFIX/results/status.json"

python3 - "$OUT/DONE" "$RUN_NAME" "$MANIFEST_SHA256" "$SOURCE_SHA256" "$MODEL_SHA256" "$TRAINING_MANIFEST_SHA256" "$VALIDATION_SHA256" "$STDOUT_SHA256" <<'PY'
import datetime,json,sys
path,run,run_manifest,source,model,training_manifest,validation,stdout=sys.argv[1:]
json.dump({"schema":"hu_m42_model_done_v1","status":"complete","run_name":run,
 "run_manifest_sha256":run_manifest,"source_sha256":source,"model_sha256":model,
 "training_manifest_sha256":training_manifest,"validation_sha256":validation,
 "training_stdout_sha256":stdout,"no_runtime_activation":True,
 "completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat()},
 open(path,"w",encoding="utf-8"),indent=2,sort_keys=True)
PY
gcs_upload_immutable "$OUT/DONE" "$PREFIX/results/DONE"
STATE=complete
echo "M4.2 model run complete: $RUN_NAME"
'@
$startup = $startup -replace "`r`n", "`n"
$startupSha256 = Get-TextSha256 $startup
if (-not $DryRun) {
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    Write-Utf8NoBom -Path $startupPath -Text $startup
}

$manifest = [ordered]@{
    schema = "hu_m42_model_run_manifest_v1"
    run_name = $RunName
    teacher_run_name = $TeacherRunName
    project_id = $ProjectId
    bucket = $Bucket
    purpose = "bounded_nested_crossfit_quality_gate_not_runtime_promotion"
    source = [ordered]@{
        uri = [string]$teacherManifest.source_uri
        bytes = [long]$teacherManifest.source_bytes
        sha256 = [string]$teacherManifest.source_sha256
        trainer_sha256 = $ExpectedTrainerSha256
    }
    inputs = [ordered]@{
        train = $inputPlan.train
        calibration = $inputPlan.calibration
        locked_holdout = $inputPlan.locked_holdout
        teacher_receipt_sha256 = Get-Sha256 $receiptPath
        teacher_audit_sha256 = Get-Sha256 $auditPath
        split_seed_or_fingerprint_overlap = 0
    }
    training_config = [ordered]@{
        model_id = $ModelId
        action_score_mode = "negative_regret_ranker_v2"
        iterations = $Iterations
        cross_fit_folds = $CrossFitFolds
        thresholds = $Thresholds
        minimum_safety_fit_samples = 5
        minimum_threshold_lock_samples = 5
        minimum_calibration_fires = 2
        maximum_false_positive_rate = 0.30
        maximum_p95_loss = 25.0
        maximum_p99_loss = 40.0
        seed = 2026071801
        safety_split_seed = 2026071802
    }
    compute = [ordered]@{
        machine_type = $MachineType
        fallback_machine_types = $FallbackMachineTypes
        zones = $Zones
        boot_disk_gb = $BootDiskGb
        spot = $true
        termination_action = "DELETE"
        deterministic_full_job_retry = $true
        fold_checkpoint_resume = $false
        vm_name = $vmName
    }
    startup = [ordered]@{ uri = $startupUri; sha256 = $startupSha256 }
    output_prefix = "$gcsPrefix/results"
    done_uri = $doneUri
    teacher_values_are_realized_match_ev = $false
    teacher_lcb_runtime_gate = $false
    current_profile_mutated = $false
    no_runtime_activation = $true
    scale_decision_uses_locked_threshold_search = $false
    created_at = (Get-Date).ToUniversalTime().ToString("o")
}
$manifestText = ($manifest | ConvertTo-Json -Depth 12) + "`n"
$manifestSha256 = Get-TextSha256 $manifestText
if (-not $DryRun) {
    Write-Utf8NoBom -Path $manifestPath -Text $manifestText
}
}

$plan = [ordered]@{
    schema = "hu_m42_model_start_plan_v1"
    run_name = $RunName
    vm_name = $vmName
    manifest_uri = $manifestUri
    manifest_sha256 = $manifestSha256
    source_sha256 = $ExpectedTeacherSourceSha256
    trainer_sha256 = $ExpectedTrainerSha256
    input_rows = $expectedRows
    input_sha256 = [ordered]@{
        train = $inputPlan.train.sha256
        calibration = $inputPlan.calibration.sha256
        locked_holdout = $inputPlan.locked_holdout.sha256
    }
    iterations = $Iterations
    cross_fit_folds = $CrossFitFolds
    machine_type = $MachineType
    spot = $true
    create_instance = [bool]$CreateInstance
    resume_existing = [bool]$ResumeExisting
    dry_run = [bool]$DryRun
    no_runtime_activation = $true
}

if ($DryRun) {
    $plan | ConvertTo-Json -Depth 10
    exit 0
}

if (-not $ResumeExisting -and (Test-GcsObject $doneUri)) {
    throw "Unexpected DONE exists before model run creation: $doneUri"
}

if (-not $ResumeExisting) {
    foreach ($split in $splitFiles.Keys) {
        Invoke-Gcloud @(
            "storage", "cp", $splitFiles[$split], $inputPlan[$split].uri,
            "--project", $ProjectId, "--if-generation-match=0"
        ) | Out-Null
    }
    Invoke-Gcloud @(
        "storage", "cp", $startupPath, $startupUri,
        "--project", $ProjectId, "--if-generation-match=0"
    ) | Out-Null
    Invoke-Gcloud @(
        "storage", "cp", $manifestPath, $manifestUri,
        "--project", $ProjectId, "--if-generation-match=0"
    ) | Out-Null
}

if (-not $CreateInstance) {
    $plan.uploaded = $true
    $plan | ConvertTo-Json -Depth 10
    exit 0
}

$metadata = @(
    "BUCKET=$Bucket",
    "RUN_NAME=$RunName",
    "MANIFEST_OBJECT=runs/$RunName/source/model_run_manifest.json",
    "MANIFEST_SHA256=$manifestSha256",
    ("SELF_DELETE=" + $(if ($NoSelfDelete) { "0" } else { "1" }))
) -join ","

$attempts = New-Object System.Collections.Generic.List[object]
$created = $false
$chosenZone = $null
$chosenMachine = $null
$preExistingInstances = @(Get-ComputeInstancesByName -InstanceName $vmName -CloudProjectId $ProjectId)
if ($preExistingInstances.Count -ne 0) {
    throw "A model worker with frozen VM name $vmName already exists; refusing a duplicate launch"
}
$machineCandidates = @($MachineType) + @($FallbackMachineTypes)
foreach ($machine in $machineCandidates) {
    $diskType = if ($machine -like "c4-*") { "hyperdisk-balanced" } else { "pd-balanced" }
    foreach ($zone in $Zones) {
        $args = @(
            "compute", "instances", "create", $vmName,
            "--project", $ProjectId,
            "--zone", $zone,
            "--machine-type", $machine,
            "--provisioning-model", "SPOT",
            "--instance-termination-action", "DELETE",
            "--maintenance-policy", "TERMINATE",
            "--boot-disk-size", ("{0}GB" -f $BootDiskGb),
            "--boot-disk-type", $diskType,
            "--image-family", "ubuntu-2404-lts-amd64",
            "--image-project", "ubuntu-os-cloud",
            "--scopes", "cloud-platform",
            "--metadata", $metadata,
            "--metadata-from-file", ("startup-script={0}" -f $startupPath),
            "--quiet"
        )
        $previousPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $output = & gcloud @args 2>&1
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousPreference
        }
        $attempts.Add([ordered]@{ zone = $zone; machine_type = $machine; exit_code = $exitCode; output = @($output) })
        if ($exitCode -eq 0) {
            $created = $true
            $chosenZone = $zone
            $chosenMachine = $machine
            break
        }
        $ambiguousInstances = @(Get-ComputeInstancesByName -InstanceName $vmName -CloudProjectId $ProjectId)
        if ($ambiguousInstances.Count -ne 0) {
            throw "Instance creation returned failure but $vmName now exists; refusing a duplicate fallback launch"
        }
    }
    if ($created) { break }
}
if (-not $created) {
    throw "Unable to create an M4.2 Spot model worker in any frozen machine/zone candidate"
}

$plan.uploaded = $true
$plan.instance_created = $true
$plan.zone = $chosenZone
$plan.chosen_machine_type = $chosenMachine
$plan.attempts = $attempts
$plan | ConvertTo-Json -Depth 12
