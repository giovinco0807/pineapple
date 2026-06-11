param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-t3-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [int]$TotalSamples = 1000000,
    [int]$ShardSamples = 200,
    [int]$FutureSamples = 32,
    [int]$VmCount = 20,
    [int]$LocalWorkers = 16,
    [string]$MachineType = "e2-highcpu-16",
    [string[]]$Zones = @(
        "asia-northeast1-a",
        "asia-northeast1-b",
        "asia-northeast1-c",
        "us-central1-a",
        "us-central1-b",
        "us-central1-c",
        "us-east1-b",
        "us-east1-c",
        "us-west1-b",
        "us-west1-c"
    ),
    [int]$BaseSeed = 2026067901,
    [int]$OpeningLookaheadSamples = 0,
    [double]$MinScoreGap = 0.0,
    [double]$SelfRegretPenaltyWeight = 0.25,
    [double]$SelfRegretFree = 0.0,
    [int]$PredictionThreads = 1,
    [int]$TimeLimitMinutes = 120,
    [int]$MinRemainingToStartShardSeconds = 300,
    [int]$BootDiskGb = 50,
    [string]$OpeningModel = "models/opening_stage7_torch_wide.pt",
    [string]$Turn1Model = "models/turn1_stage6_torch_wide.pt",
    [string]$Turn2Model = "models/turn2_stage8.pkl",
    [string]$Turn3Model = "models/turn3_stage6.pkl",
    [string]$SelectionHuTurn3Model = "",
    [string]$SelectionCompareHuTurn3Model = "",
    [double]$SelectionMinMargin = [double]::NaN,
    [double]$SelectionMaxMargin = [double]::NaN,
    [switch]$SelectionRequireDisagreement,
    [switch]$SelectionRequireCompareDisagreement,
    [string[]]$StartShards = @(),
    [switch]$CreateInstances,
    [switch]$SkipExistingInstances,
    [switch]$NoSelfDelete
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    $cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    if (-not (Test-Path $cloudSdkGcloud)) {
        throw "gcloud not found in PATH or at $cloudSdkGcloud"
    }
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}

if ($TotalSamples -le 0) { throw "TotalSamples must be positive" }
if ($ShardSamples -le 0) { throw "ShardSamples must be positive" }
if ($FutureSamples -lt 0) { throw "FutureSamples must be non-negative" }
if ($VmCount -le 0) { throw "VmCount must be positive" }
if ($LocalWorkers -le 0) { throw "LocalWorkers must be positive" }
if ($PredictionThreads -le 0) { throw "PredictionThreads must be positive" }
if ($TimeLimitMinutes -lt 0) { throw "TimeLimitMinutes must be non-negative" }
if ($MinRemainingToStartShardSeconds -lt 0) { throw "MinRemainingToStartShardSeconds must be non-negative" }
if ($SelfRegretPenaltyWeight -lt 0) { throw "SelfRegretPenaltyWeight must be non-negative" }
if ($SelfRegretFree -lt 0) { throw "SelfRegretFree must be non-negative" }
if (-not $Zones -or $Zones.Count -eq 0) { throw "At least one zone is required" }
if ($RunName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') {
    throw "RunName may only contain letters, numbers, dot, underscore, and dash"
}

$StartShardIndices = @()
foreach ($startShardValue in $StartShards) {
    foreach ($part in ($startShardValue -split ",")) {
        $trimmed = $part.Trim()
        if (-not $trimmed) { continue }
        $StartShardIndices += [int]$trimmed
    }
}
$StartShardIndices = @($StartShardIndices | Sort-Object -Unique)
foreach ($startShard in $StartShardIndices) {
    if ($startShard -lt 0 -or $startShard -ge $VmCount) {
        throw "StartShards values must be in [0, VmCount): $startShard"
    }
}

function Convert-ToVmName {
    param([string]$Name)
    $vmName = $Name.ToLowerInvariant() -replace '[^a-z0-9-]', '-'
    $vmName = $vmName.Trim('-')
    if ($vmName.Length -gt 54) {
        $vmName = $vmName.Substring(0, 54).Trim('-')
    }
    if (-not $vmName) {
        throw "RunName does not produce a valid VM name"
    }
    return $vmName
}

function Write-Utf8NoBom {
    param(
        [string]$Path,
        [string]$Text
    )
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

function New-ZipWithForwardSlashes {
    param(
        [string]$SourceDir,
        [string]$DestinationPath
    )

    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (Test-Path $DestinationPath) {
        Remove-Item -LiteralPath $DestinationPath -Force
    }

    $sourceRoot = (Resolve-Path -LiteralPath $SourceDir).Path.TrimEnd('\', '/')
    $prefixLength = $sourceRoot.Length + 1
    $zip = [System.IO.Compression.ZipFile]::Open($DestinationPath, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | ForEach-Object {
            $relativePath = $_.FullName.Substring($prefixLength)
            $entryName = $relativePath -replace '\\', '/'
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $zip,
                $_.FullName,
                $entryName,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
    finally {
        $zip.Dispose()
    }
}

$repoRoot = (Get-Location).Path
$modelInputs = @($OpeningModel, $Turn1Model, $Turn2Model, $Turn3Model)
if ($SelectionHuTurn3Model) {
    $modelInputs += $SelectionHuTurn3Model
}
if ($SelectionCompareHuTurn3Model) {
    $modelInputs += $SelectionCompareHuTurn3Model
}
foreach ($modelInput in $modelInputs) {
    $modelPath = Join-Path $repoRoot $modelInput
    if (-not (Test-Path $modelPath)) {
        throw "Model not found: $modelInput"
    }
}

$openingModelName = Split-Path -Leaf $OpeningModel
$turn1ModelName = Split-Path -Leaf $Turn1Model
$turn2ModelName = Split-Path -Leaf $Turn2Model
$turn3ModelName = Split-Path -Leaf $Turn3Model
$selectionModelName = if ($SelectionHuTurn3Model) { Split-Path -Leaf $SelectionHuTurn3Model } else { "" }
$selectionCompareModelName = if ($SelectionCompareHuTurn3Model) { Split-Path -Leaf $SelectionCompareHuTurn3Model } else { "" }
$totalShards = [int][math]::Ceiling($TotalSamples / $ShardSamples)
$timeLimitSeconds = $TimeLimitMinutes * 60
$runDir = Join-Path $repoRoot ("outputs/gcp_runs/{0}" -f $RunName)
$packageDir = Join-Path $runDir "package_src"
$packagePath = Join-Path $runDir "ofc_regular_hu_selfplay_source.zip"
$startupPath = Join-Path $runDir "startup_hu_selfplay_spot.sh"
$manifestPath = Join-Path $runDir "manifest.json"
$sourceUri = "gs://$Bucket/runs/$RunName/source/ofc_regular_hu_selfplay_source.zip"
$startupUri = "gs://$Bucket/runs/$RunName/source/startup_hu_selfplay_spot.sh"
$manifestUri = "gs://$Bucket/runs/$RunName/manifest.json"

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
if (Test-Path $packageDir) {
    Remove-Item -LiteralPath $packageDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $packageDir | Out-Null

foreach ($item in @("pyproject.toml", "src")) {
    $source = Join-Path $repoRoot $item
    if (-not (Test-Path $source)) {
        throw "Missing package input: $item"
    }
    Copy-Item -LiteralPath $source -Destination $packageDir -Recurse
}

$modelPackageDir = Join-Path $packageDir "models"
New-Item -ItemType Directory -Force -Path $modelPackageDir | Out-Null
foreach ($modelInput in $modelInputs) {
    $source = Join-Path $repoRoot $modelInput
    Copy-Item -LiteralPath $source -Destination (Join-Path $modelPackageDir (Split-Path -Leaf $modelInput)) -Force
}

New-ZipWithForwardSlashes -SourceDir $packageDir -DestinationPath $packagePath

$startup = @'
#!/usr/bin/env bash
set -euo pipefail
export HOME="${HOME:-/root}"
CURRENT_UID="$(id -u)"
CURRENT_GID="$(id -g)"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
}

meta() {
  curl -fsS -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

instance_meta() {
  curl -fsS -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$1"
}

RUN_NAME="$(meta RUN_NAME)"
BUCKET="$(meta BUCKET)"
TOTAL_SHARDS="$(meta TOTAL_SHARDS)"
SHARD_SAMPLES="$(meta SHARD_SAMPLES)"
FUTURE_SAMPLES="$(meta FUTURE_SAMPLES)"
BASE_SEED="$(meta BASE_SEED)"
VM_COUNT="$(meta VM_COUNT)"
START_SHARD="$(meta START_SHARD)"
LOCAL_WORKERS="$(meta LOCAL_WORKERS)"
SOURCE_URI="$(meta SOURCE_URI)"
SELF_DELETE="$(meta SELF_DELETE)"
PREDICTION_THREADS="$(meta PREDICTION_THREADS)"
TIME_LIMIT_SECONDS="$(meta TIME_LIMIT_SECONDS)"
MIN_REMAINING_TO_START_SHARD_SECONDS="$(meta MIN_REMAINING_TO_START_SHARD_SECONDS)"
OPENING_LOOKAHEAD_SAMPLES="$(meta OPENING_LOOKAHEAD_SAMPLES)"
MIN_SCORE_GAP="$(meta MIN_SCORE_GAP)"
SELF_REGRET_PENALTY_WEIGHT="$(meta SELF_REGRET_PENALTY_WEIGHT)"
SELF_REGRET_FREE="$(meta SELF_REGRET_FREE)"
OPENING_MODEL="$(meta OPENING_MODEL)"
TURN1_MODEL="$(meta TURN1_MODEL)"
TURN2_MODEL="$(meta TURN2_MODEL)"
TURN3_MODEL="$(meta TURN3_MODEL)"
SELECTION_HU_TURN3_MODEL="$(meta SELECTION_HU_TURN3_MODEL)"
SELECTION_COMPARE_HU_TURN3_MODEL="$(meta SELECTION_COMPARE_HU_TURN3_MODEL)"
SELECTION_MIN_MARGIN="$(meta SELECTION_MIN_MARGIN)"
SELECTION_MAX_MARGIN="$(meta SELECTION_MAX_MARGIN)"
SELECTION_REQUIRE_DISAGREEMENT="$(meta SELECTION_REQUIRE_DISAGREEMENT)"
SELECTION_REQUIRE_COMPARE_DISAGREEMENT="$(meta SELECTION_REQUIRE_COMPARE_DISAGREEMENT)"
INSTANCE_NAME="$(instance_meta name)"
ZONE_PATH="$(instance_meta zone)"
ZONE="${ZONE_PATH##*/}"

export DEBIAN_FRONTEND=noninteractive
export OMP_NUM_THREADS="$PREDICTION_THREADS"
export MKL_NUM_THREADS="$PREDICTION_THREADS"
export OPENBLAS_NUM_THREADS="$PREDICTION_THREADS"
export NUMEXPR_NUM_THREADS="$PREDICTION_THREADS"

START_SECONDS="$(date +%s)"
DEADLINE_SECONDS=0
if [[ "$TIME_LIMIT_SECONDS" -gt 0 ]]; then
  DEADLINE_SECONDS=$((START_SECONDS + TIME_LIMIT_SECONDS))
fi

log "install dependencies"
sudo apt-get update -y
sudo apt-get install -y unzip curl ca-certificates python3 python3-venv python3-pip

if ! command -v gcloud >/dev/null 2>&1; then
  log "google cloud sdk not found; installing google-cloud-cli"
  sudo apt-get install -y apt-transport-https gnupg
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y google-cloud-cli
fi

WORK_DIR="/opt/ofc-regular-hu-selfplay"
sudo rm -rf "$WORK_DIR"
sudo mkdir -p "$WORK_DIR"
sudo chown "$CURRENT_UID:$CURRENT_GID" "$WORK_DIR"
log "download source package $SOURCE_URI"
gcloud storage cp "$SOURCE_URI" /tmp/ofc_regular_hu_selfplay_source.zip
unzip -q /tmp/ofc_regular_hu_selfplay_source.zip -d "$WORK_DIR"
cd "$WORK_DIR"

log "create python venv"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy>=1.26" "scikit-learn>=1.8,<1.9"
python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "torch==2.6.0+cpu"
export PYTHONPATH="$WORK_DIR/src"

STATUS_DIR="/tmp/ofc-hu-t3-status"
SHARD_DIR="/tmp/ofc-hu-t3-shards"
mkdir -p "$STATUS_DIR" "$SHARD_DIR"

run_shard() {
  local shard="$1"
  local shard_name
  shard_name="$(printf "hu_t3_%06d.jsonl" "$shard")"
  local remote="gs://${BUCKET}/runs/${RUN_NAME}/shards/${shard_name}"
  local status_remote="gs://${BUCKET}/runs/${RUN_NAME}/status/${shard_name%.jsonl}.json"
  if gcloud storage ls "$remote" >/dev/null 2>&1; then
    log "skip existing shard $shard"
    return 0
  fi

  local local_path="${SHARD_DIR}/${shard_name}"
  local status_path="${STATUS_DIR}/${shard_name%.jsonl}.json"
  rm -f "$local_path" "$status_path"
  local started_at
  local start_seconds
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_seconds="$(date +%s)"
  local seed=$((BASE_SEED + shard))
  log "generate shard=$shard seed=$seed samples=$SHARD_SAMPLES future=$FUTURE_SAMPLES"

  local cmd=(
    python -B -m ofc_regular.hu_self_play_teacher_data
    --samples "$SHARD_SAMPLES"
    --seed "$seed"
    --future-samples "$FUTURE_SAMPLES"
    --opening-lookahead-samples "$OPENING_LOOKAHEAD_SAMPLES"
    --min-score-gap "$MIN_SCORE_GAP"
    --self-regret-penalty-weight "$SELF_REGRET_PENALTY_WEIGHT"
    --self-regret-free "$SELF_REGRET_FREE"
    --opening-model "models/${OPENING_MODEL}"
    --turn1-model "models/${TURN1_MODEL}"
    --turn2-model "models/${TURN2_MODEL}"
    --turn3-model "models/${TURN3_MODEL}"
    --output "$local_path"
  )
  if [[ -n "$SELECTION_HU_TURN3_MODEL" ]]; then
    cmd+=(--selection-hu-turn3-model "models/${SELECTION_HU_TURN3_MODEL}")
  fi
  if [[ -n "$SELECTION_COMPARE_HU_TURN3_MODEL" ]]; then
    cmd+=(--selection-compare-hu-turn3-model "models/${SELECTION_COMPARE_HU_TURN3_MODEL}")
  fi
  if [[ "$SELECTION_MIN_MARGIN" != "NaN" ]]; then
    cmd+=(--selection-min-margin "$SELECTION_MIN_MARGIN")
  fi
  if [[ "$SELECTION_MAX_MARGIN" != "NaN" ]]; then
    cmd+=(--selection-max-margin "$SELECTION_MAX_MARGIN")
  fi
  if [[ "$SELECTION_REQUIRE_DISAGREEMENT" == "1" ]]; then
    cmd+=(--selection-require-disagreement)
  fi
  if [[ "$SELECTION_REQUIRE_COMPARE_DISAGREEMENT" == "1" ]]; then
    cmd+=(--selection-require-compare-disagreement)
  fi

  set +e
  "${cmd[@]}"
  local exit_code=$?
  set -e

  local finished_at
  local end_seconds
  local elapsed
  local lines=0
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end_seconds="$(date +%s)"
  elapsed=$((end_seconds - start_seconds))
  if [[ -f "$local_path" ]]; then
    lines="$(wc -l < "$local_path" | tr -d ' ')"
  fi

  if [[ "$exit_code" -eq 0 && "$lines" -eq "$SHARD_SAMPLES" ]]; then
    local tmp_remote="${remote}.tmp-${INSTANCE_NAME}"
    log "upload shard=$shard lines=$lines elapsed=${elapsed}s"
    gcloud storage cp "$local_path" "$tmp_remote"
    gcloud storage mv "$tmp_remote" "$remote"
    cat > "$status_path" <<EOF
{"run_name":"$RUN_NAME","phase":"hu_turn3_selfplay","shard":$shard,"status":"complete","instance":"$INSTANCE_NAME","zone":"$ZONE","seed":$seed,"samples":$SHARD_SAMPLES,"future_samples":$FUTURE_SAMPLES,"opening_lookahead_samples":$OPENING_LOOKAHEAD_SAMPLES,"self_regret_penalty_weight":$SELF_REGRET_PENALTY_WEIGHT,"self_regret_free":$SELF_REGRET_FREE,"lines":$lines,"elapsed_seconds":$elapsed,"started_at":"$started_at","finished_at":"$finished_at","output":"$remote"}
EOF
    gcloud storage cp "$status_path" "$status_remote" || true
    return 0
  fi

  log "failed shard=$shard exit=$exit_code lines=$lines expected=$SHARD_SAMPLES"
  cat > "$status_path" <<EOF
{"run_name":"$RUN_NAME","phase":"hu_turn3_selfplay","shard":$shard,"status":"failed","instance":"$INSTANCE_NAME","zone":"$ZONE","seed":$seed,"samples":$SHARD_SAMPLES,"future_samples":$FUTURE_SAMPLES,"lines":$lines,"exit_code":$exit_code,"elapsed_seconds":$elapsed,"started_at":"$started_at","finished_at":"$finished_at"}
EOF
  gcloud storage cp "$status_path" "$status_remote" || true
  return "$exit_code"
}

log "start shard loop run=$RUN_NAME start=$START_SHARD stride=$VM_COUNT local_workers=$LOCAL_WORKERS total_shards=$TOTAL_SHARDS shard_samples=$SHARD_SAMPLES future=$FUTURE_SAMPLES time_limit_seconds=$TIME_LIMIT_SECONDS"
next_shard="$START_SHARD"
while [[ "$next_shard" -lt "$TOTAL_SHARDS" ]]; do
  if [[ "$DEADLINE_SECONDS" -gt 0 ]]; then
    now="$(date +%s)"
    remaining=$((DEADLINE_SECONDS - now))
    if [[ "$remaining" -lt "$MIN_REMAINING_TO_START_SHARD_SECONDS" ]]; then
      log "time limit reached for starting new shards remaining=${remaining}s"
      break
    fi
  fi

  pids=()
  shards=()
  for (( worker=0; worker<LOCAL_WORKERS && next_shard<TOTAL_SHARDS; worker++ )); do
    shard="$next_shard"
    next_shard=$((next_shard + VM_COUNT))
    run_shard "$shard" &
    pids+=("$!")
    shards+=("$shard")
  done

  for index in "${!pids[@]}"; do
    pid="${pids[$index]}"
    shard="${shards[$index]}"
    if ! wait "$pid"; then
      log "worker failed for shard=$shard"
      exit 1
    fi
  done
done

log "assigned shards finished or time limit reached"
if [[ "$SELF_DELETE" == "1" ]]; then
  log "self delete instance $INSTANCE_NAME $ZONE"
  gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" --quiet || sudo shutdown -h now
else
  sudo shutdown -h now
fi
'@

Write-Utf8NoBom -Path $startupPath -Text $startup

$manifest = [ordered]@{
    project_id = $ProjectId
    bucket = $Bucket
    run_name = $RunName
    phase = "hu_turn3_selfplay"
    shard_prefix = "hu_t3"
    total_samples = $TotalSamples
    shard_samples = $ShardSamples
    total_shards = $totalShards
    future_samples = $FutureSamples
    vm_count = $VmCount
    local_workers = $LocalWorkers
    machine_type = $MachineType
    zones = $Zones
    base_seed = $BaseSeed
    opening_lookahead_samples = $OpeningLookaheadSamples
    min_score_gap = $MinScoreGap
    self_regret_penalty_weight = $SelfRegretPenaltyWeight
    self_regret_free = $SelfRegretFree
    prediction_threads = $PredictionThreads
    time_limit_minutes = $TimeLimitMinutes
    min_remaining_to_start_shard_seconds = $MinRemainingToStartShardSeconds
    opening_model = $OpeningModel
    turn1_model = $Turn1Model
    turn2_model = $Turn2Model
    turn3_model = $Turn3Model
    selection_hu_turn3_model = $SelectionHuTurn3Model
    selection_compare_hu_turn3_model = $SelectionCompareHuTurn3Model
    selection_min_margin = if ([double]::IsNaN($SelectionMinMargin)) { $null } else { $SelectionMinMargin }
    selection_max_margin = if ([double]::IsNaN($SelectionMaxMargin)) { $null } else { $SelectionMaxMargin }
    selection_require_disagreement = [bool]$SelectionRequireDisagreement
    selection_require_compare_disagreement = [bool]$SelectionRequireCompareDisagreement
    source_uri = $sourceUri
    startup_uri = $startupUri
    self_delete = -not $NoSelfDelete
    created_at = (Get-Date).ToUniversalTime().ToString("o")
}
Write-Utf8NoBom -Path $manifestPath -Text (($manifest | ConvertTo-Json -Depth 5) + "`n")

gcloud storage cp $packagePath $sourceUri --project $ProjectId | Out-Null
gcloud storage cp $startupPath $startupUri --project $ProjectId | Out-Null
gcloud storage cp $manifestPath $manifestUri --project $ProjectId | Out-Null

$vmPrefix = Convert-ToVmName $RunName
$instances = @()
$workerIndices = if ($StartShardIndices.Count -gt 0) { $StartShardIndices } else { @(0..($VmCount - 1)) }
foreach ($i in $workerIndices) {
    $zone = $Zones[$i % $Zones.Count]
    $vmName = ("{0}-{1:d3}" -f $vmPrefix, $i)
    if ($vmName.Length -gt 63) {
        $vmName = $vmName.Substring(0, 63).Trim("-")
    }
    $metadata = @(
        "RUN_NAME=$RunName",
        "BUCKET=$Bucket",
        "TOTAL_SHARDS=$totalShards",
        "SHARD_SAMPLES=$ShardSamples",
        "FUTURE_SAMPLES=$FutureSamples",
        "BASE_SEED=$BaseSeed",
        "VM_COUNT=$VmCount",
        "START_SHARD=$i",
        "LOCAL_WORKERS=$LocalWorkers",
        "SOURCE_URI=$sourceUri",
        "PREDICTION_THREADS=$PredictionThreads",
        "TIME_LIMIT_SECONDS=$timeLimitSeconds",
        "MIN_REMAINING_TO_START_SHARD_SECONDS=$MinRemainingToStartShardSeconds",
        "OPENING_LOOKAHEAD_SAMPLES=$OpeningLookaheadSamples",
        "MIN_SCORE_GAP=$MinScoreGap",
        "SELF_REGRET_PENALTY_WEIGHT=$SelfRegretPenaltyWeight",
        "SELF_REGRET_FREE=$SelfRegretFree",
        "OPENING_MODEL=$openingModelName",
        "TURN1_MODEL=$turn1ModelName",
        "TURN2_MODEL=$turn2ModelName",
        "TURN3_MODEL=$turn3ModelName",
        "SELECTION_HU_TURN3_MODEL=$selectionModelName",
        "SELECTION_COMPARE_HU_TURN3_MODEL=$selectionCompareModelName",
        ("SELECTION_MIN_MARGIN=" + ($(if ([double]::IsNaN($SelectionMinMargin)) { "NaN" } else { [string]$SelectionMinMargin }))),
        ("SELECTION_MAX_MARGIN=" + ($(if ([double]::IsNaN($SelectionMaxMargin)) { "NaN" } else { [string]$SelectionMaxMargin }))),
        ("SELECTION_REQUIRE_DISAGREEMENT=" + ($(if ($SelectionRequireDisagreement) { "1" } else { "0" }))),
        ("SELECTION_REQUIRE_COMPARE_DISAGREEMENT=" + ($(if ($SelectionRequireCompareDisagreement) { "1" } else { "0" }))),
        ("SELF_DELETE=" + ($(if ($NoSelfDelete) { "0" } else { "1" })))
    ) -join ","

    $instances += [pscustomobject]@{
        name = $vmName
        zone = $zone
        start_shard = $i
    }

    if ($CreateInstances) {
        if ($SkipExistingInstances) {
            $existing = gcloud compute instances list `
                --project $ProjectId `
                --filter ("name='" + $vmName + "'") `
                --format "value(name)" 2>$null
            if ($existing) {
                continue
            }
        }
        gcloud compute instances create $vmName `
            --project $ProjectId `
            --zone $zone `
            --machine-type $MachineType `
            --provisioning-model SPOT `
            --instance-termination-action DELETE `
            --maintenance-policy TERMINATE `
            --image-family ubuntu-2404-lts-amd64 `
            --image-project ubuntu-os-cloud `
            --boot-disk-size ("{0}GB" -f $BootDiskGb) `
            --scopes https://www.googleapis.com/auth/cloud-platform `
            --metadata $metadata `
            --metadata-from-file startup-script=$startupPath | Out-Null
    }
}

[pscustomobject]@{
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    total_samples = $TotalSamples
    shard_samples = $ShardSamples
    total_shards = $totalShards
    future_samples = $FutureSamples
    vm_count = $VmCount
    local_workers = $LocalWorkers
    machine_type = $MachineType
    time_limit_minutes = $TimeLimitMinutes
    package = $packagePath
    source_uri = $sourceUri
    manifest_uri = $manifestUri
    create_instances = [bool]$CreateInstances
    instances = $instances
} | ConvertTo-Json -Depth 5
