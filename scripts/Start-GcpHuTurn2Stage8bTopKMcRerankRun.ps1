param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-t2-topk-rerank-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [int]$GamesPerSeed = 300,
    [string]$Seeds = "2026071001,2026071002,2026071003,2026071004,2026071005",
    [string]$Configs = "k3/mc64/d0.5/se0/seat=first,k5/mc64/d0.25/se0/seat=first",
    [string]$HuTurn2Stage8bModel = "models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt",
    [string]$Stage3FeatureEncoderMode = "rust_direct",
    [int]$BatchedContinuationBatchSize = 8192,
    [int]$VmCount = 10,
    [string]$MachineType = "e2-highcpu-4",
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
    [int]$SeedStride = 1000000,
    [int]$PredictionThreads = 1,
    [int]$OpeningLookaheadSamples = 64,
    [int]$BootDiskGb = 50,
    [string[]]$StartShards = @(),
    [switch]$CreateInstances,
    [switch]$SkipExistingInstances,
    [switch]$NoSelfDelete
)

$ErrorActionPreference = "Stop"

$cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (Test-Path $cloudSdkGcloud) {
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}
elseif (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud not found in PATH or at $cloudSdkGcloud"
}

if ($GamesPerSeed -le 0) { throw "GamesPerSeed must be positive" }
if ($VmCount -le 0) { throw "VmCount must be positive" }
if ($PredictionThreads -lt 0) { throw "PredictionThreads must be non-negative" }
if (-not $Zones -or $Zones.Count -eq 0) { throw "At least one zone is required" }
if ($RunName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') {
    throw "RunName may only contain letters, numbers, dot, underscore, and dash"
}

$StartShardIndices = @()
foreach ($startShardValue in $StartShards) {
    foreach ($part in ($startShardValue -split ",")) {
        $trimmed = $part.Trim()
        if ($trimmed) {
            $StartShardIndices += [int]$trimmed
        }
    }
}
$StartShardIndices = @($StartShardIndices | Sort-Object -Unique)

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
    param([string]$Path, [string]$Text)
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

function New-ZipWithForwardSlashes {
    param([string]$SourceDir, [string]$DestinationPath)

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
$huTurn2Stage8bModelRepoPath = ($HuTurn2Stage8bModel -replace '\\', '/').TrimStart('/')
if ([System.IO.Path]::IsPathRooted($HuTurn2Stage8bModel)) {
    throw "HuTurn2Stage8bModel must be a repo-relative path: $HuTurn2Stage8bModel"
}
$requiredModels = @(
    "models/opening_stage7_torch_wide.pt",
    "models/turn1_stage6_torch_wide.pt",
    "models/turn2_stage8.pkl",
    "models/turn3_stage6.pkl",
    "models/hu_turn3_stage7_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt",
    $huTurn2Stage8bModelRepoPath
) | Select-Object -Unique
foreach ($model in $requiredModels) {
    if (-not (Test-Path (Join-Path $repoRoot $model))) {
        throw "Required model not found: $model"
    }
}

$seedList = @()
foreach ($part in ($Seeds -split ",")) {
    $trimmed = $part.Trim()
    if ($trimmed) { $seedList += [int]$trimmed }
}
if ($seedList.Count -eq 0) { throw "At least one seed is required" }

$configList = @()
foreach ($part in ($Configs -split ",")) {
    $trimmed = $part.Trim()
    if ($trimmed) { $configList += $trimmed }
}
if ($configList.Count -eq 0) { throw "At least one config is required" }

$runDir = Join-Path $repoRoot ("outputs/gcp_runs/{0}" -f $RunName)
$packageDir = Join-Path $runDir "package_src"
$packagePath = Join-Path $runDir "ofc_regular_hu_t2_topk_rerank_source.zip"
$startupPath = Join-Path $runDir "startup_hu_t2_topk_rerank_spot.sh"
$manifestPath = Join-Path $runDir "manifest.json"
$shardManifestPath = Join-Path $runDir "shards_manifest.jsonl"
$sourceUri = "gs://$Bucket/runs/$RunName/source/ofc_regular_hu_t2_topk_rerank_source.zip"
$startupUri = "gs://$Bucket/runs/$RunName/source/startup_hu_t2_topk_rerank_spot.sh"
$manifestUri = "gs://$Bucket/runs/$RunName/manifest.json"

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
if (Test-Path $packageDir) {
    Remove-Item -LiteralPath $packageDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $packageDir | Out-Null

foreach ($item in @("pyproject.toml", "Cargo.toml", "Cargo.lock", "src", "rust")) {
    $source = Join-Path $repoRoot $item
    if (-not (Test-Path $source)) {
        throw "Missing package input: $item"
    }
    Copy-Item -LiteralPath $source -Destination $packageDir -Recurse
}

foreach ($model in $requiredModels) {
    $dest = Join-Path $packageDir $model
    New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
    Copy-Item -LiteralPath (Join-Path $repoRoot $model) -Destination $dest
}

$shardLines = New-Object System.Collections.Generic.List[string]
$shardIndex = 0
foreach ($config in $configList) {
    foreach ($seed in $seedList) {
        $configName = ($config -replace '[^0-9A-Za-z]+', '_').Trim('_')
        $spec = [ordered]@{
            shard = $shardIndex
            config = $config
            seed = $seed
            games_per_seed = $GamesPerSeed
            output_prefix = ("shard_{0:D3}_{1}_seed{2}" -f $shardIndex, $configName, $seed)
        }
        $shardLines.Add(($spec | ConvertTo-Json -Compress -Depth 5))
        $shardIndex += 1
    }
}
$totalShards = $shardIndex
foreach ($startShard in $StartShardIndices) {
    if ($startShard -lt 0 -or $startShard -ge $totalShards) {
        throw "StartShards values must be in [0, total_shards): $startShard"
    }
}
Write-Utf8NoBom -Path $shardManifestPath -Text (($shardLines -join "`n") + "`n")
Copy-Item -LiteralPath $shardManifestPath -Destination (Join-Path $packageDir "shards_manifest.jsonl")

New-ZipWithForwardSlashes -SourceDir $packageDir -DestinationPath $packagePath

$startup = @'
#!/usr/bin/env bash
set -euo pipefail
export HOME="${HOME:-/root}"
CURRENT_UID="$(id -u)"
CURRENT_GID="$(id -g)"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
meta() { curl -fsS -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
instance_meta() { curl -fsS -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
json_field() { python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get(sys.argv[2], ""))' "$1" "$2"; }

RUN_NAME="$(meta RUN_NAME)"
BUCKET="$(meta BUCKET)"
TOTAL_SHARDS="$(meta TOTAL_SHARDS)"
VM_COUNT="$(meta VM_COUNT)"
START_SHARD="$(meta START_SHARD)"
SOURCE_URI="$(meta SOURCE_URI)"
SELF_DELETE="$(meta SELF_DELETE)"
PREDICTION_THREADS="$(meta PREDICTION_THREADS)"
SEED_STRIDE="$(meta SEED_STRIDE)"
OPENING_LOOKAHEAD_SAMPLES="$(meta OPENING_LOOKAHEAD_SAMPLES)"
HU_TURN2_STAGE8B_MODEL="$(meta HU_TURN2_STAGE8B_MODEL)"
STAGE3_ENCODER_MODE="$(meta STAGE3_ENCODER_MODE)"
BATCHED_CONTINUATION_BATCH_SIZE="$(meta BATCHED_CONTINUATION_BATCH_SIZE)"
INSTANCE_NAME="$(instance_meta name)"
ZONE_PATH="$(instance_meta zone)"
ZONE="${ZONE_PATH##*/}"

cleanup_on_error() {
  code="$1"
  if [[ "$code" -eq 0 ]]; then return; fi
  log "startup failed exit=$code instance=$INSTANCE_NAME zone=$ZONE"
  if [[ "$SELF_DELETE" == "1" ]] && command -v gcloud >/dev/null 2>&1; then
    gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" --quiet || sudo shutdown -h now
  else
    sudo shutdown -h now || true
  fi
}
trap 'cleanup_on_error $?' EXIT

export DEBIAN_FRONTEND=noninteractive
export OMP_NUM_THREADS="$PREDICTION_THREADS"
export MKL_NUM_THREADS="$PREDICTION_THREADS"
export OPENBLAS_NUM_THREADS="$PREDICTION_THREADS"

log "install system dependencies"
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

WORK_DIR="/opt/ofc-regular-hu-t2-topk-rerank"
sudo rm -rf "$WORK_DIR"
sudo mkdir -p "$WORK_DIR"
sudo chown "$CURRENT_UID:$CURRENT_GID" "$WORK_DIR"
log "download source package $SOURCE_URI"
gcloud storage cp "$SOURCE_URI" /tmp/ofc_regular_hu_t2_topk_rerank_source.zip
unzip -q /tmp/ofc_regular_hu_t2_topk_rerank_source.zip -d "$WORK_DIR"
cd "$WORK_DIR"

log "create python venv"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy==2.2.6" "scikit-learn==1.8.0"
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0"
export PYTHONPATH="$WORK_DIR/src"

if [[ "$STAGE3_ENCODER_MODE" == rust* ]]; then
  log "install rust toolchain and build stage3 feature encoder"
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
  source "$HOME/.cargo/env"
  cargo build --release
  if [[ ! -f target/release/libofc_stage3_feature_encoder.so ]]; then
    log "rust encoder build failed; falling back to scalar_fast"
    STAGE3_ENCODER_MODE="scalar_fast"
  fi
fi

STATUS_DIR="/tmp/ofc-hu-t2-topk-status"
RESULT_DIR="/tmp/ofc-hu-t2-topk-results"
mkdir -p "$STATUS_DIR" "$RESULT_DIR"

log "start TopK rerank shard loop run=$RUN_NAME start=$START_SHARD stride=$VM_COUNT total_shards=$TOTAL_SHARDS"
for (( shard=START_SHARD; shard<TOTAL_SHARDS; shard+=VM_COUNT )); do
  spec="$(sed -n "$((shard + 1))p" shards_manifest.jsonl)"
  if [[ -z "$spec" ]]; then
    log "missing spec for shard=$shard"
    exit 2
  fi

  config="$(json_field "$spec" config)"
  seed="$(json_field "$spec" seed)"
  games_per_seed="$(json_field "$spec" games_per_seed)"
  output_prefix="$(json_field "$spec" output_prefix)"
  remote_marker="gs://${BUCKET}/runs/${RUN_NAME}/results/${output_prefix}/DONE"
  status_remote="gs://${BUCKET}/runs/${RUN_NAME}/status/${output_prefix}.json"
  if gcloud storage ls "$remote_marker" >/dev/null 2>&1; then
    log "skip existing shard=$shard $output_prefix"
    continue
  fi

  local_out="${RESULT_DIR}/${output_prefix}"
  rm -rf "$local_out"
  mkdir -p "$local_out"
  log_path="${local_out}/run.log"
  status_path="${STATUS_DIR}/${output_prefix}.json"
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_seconds="$(date +%s)"
  log "run shard=$shard config=$config seed=$seed games=$games_per_seed"

  set +e
  python -B -m ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank \
    --games-per-seed "$games_per_seed" \
    --seeds "$seed" \
    --seed-stride "$SEED_STRIDE" \
    --configs "$config" \
    --hu-turn2-stage8b-model "$HU_TURN2_STAGE8B_MODEL" \
    --output-dir "$local_out" \
    --device cpu \
    --prediction-threads "$PREDICTION_THREADS" \
    --opening-lookahead-samples "$OPENING_LOOKAHEAD_SAMPLES" \
    --batched-continuation-batch-size "$BATCHED_CONTINUATION_BATCH_SIZE" \
    --stage3-feature-encoder-mode "$STAGE3_ENCODER_MODE" \
    --progress-every 0 \
    --write-decision-log > "$log_path" 2>&1
  exit_code=$?
  set -e

  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end_seconds="$(date +%s)"
  elapsed=$((end_seconds - start_seconds))
  if [[ "$exit_code" -eq 0 ]]; then
    log "upload shard=$shard output_prefix=$output_prefix elapsed=${elapsed}s"
    gcloud storage cp "$local_out"/* "gs://${BUCKET}/runs/${RUN_NAME}/results/${output_prefix}/" >/dev/null
    printf 'complete\n' > "${local_out}/DONE"
    gcloud storage cp "${local_out}/DONE" "$remote_marker" >/dev/null
    cat > "$status_path" <<EOF
{"run_name":"$RUN_NAME","phase":"hu_t2_stage8b_topk_mc_rerank","shard":$shard,"config":"$config","seed":$seed,"games_per_seed":$games_per_seed,"status":"complete","instance":"$INSTANCE_NAME","zone":"$ZONE","elapsed_seconds":$elapsed,"started_at":"$started_at","finished_at":"$finished_at","output":"gs://${BUCKET}/runs/${RUN_NAME}/results/${output_prefix}"}
EOF
    gcloud storage cp "$status_path" "$status_remote" >/dev/null || true
  else
    log "failed shard=$shard exit=$exit_code elapsed=${elapsed}s"
    gcloud storage cp "$log_path" "gs://${BUCKET}/runs/${RUN_NAME}/logs/${output_prefix}.log" >/dev/null || true
    cat > "$status_path" <<EOF
{"run_name":"$RUN_NAME","phase":"hu_t2_stage8b_topk_mc_rerank","shard":$shard,"config":"$config","seed":$seed,"games_per_seed":$games_per_seed,"status":"failed","instance":"$INSTANCE_NAME","zone":"$ZONE","exit_code":$exit_code,"elapsed_seconds":$elapsed,"started_at":"$started_at","finished_at":"$finished_at","log":"gs://${BUCKET}/runs/${RUN_NAME}/logs/${output_prefix}.log"}
EOF
    gcloud storage cp "$status_path" "$status_remote" >/dev/null || true
    exit "$exit_code"
  fi
done

log "all assigned TopK rerank shards finished"
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
    phase = "hu_t2_stage8b_topk_mc_rerank"
    games_per_seed = $GamesPerSeed
    configs = $configList
    seeds = $seedList
    total_shards = $totalShards
    vm_count = $VmCount
    machine_type = $MachineType
    zones = $Zones
    seed_stride = $SeedStride
    prediction_threads = $PredictionThreads
    opening_lookahead_samples = $OpeningLookaheadSamples
    hu_turn2_stage8b_model = $huTurn2Stage8bModelRepoPath
    stage3_feature_encoder_mode = $Stage3FeatureEncoderMode
    batched_continuation_batch_size = $BatchedContinuationBatchSize
    source_uri = $sourceUri
    startup_uri = $startupUri
    self_delete = -not $NoSelfDelete
    created_at = (Get-Date).ToUniversalTime().ToString("o")
}
Write-Utf8NoBom -Path $manifestPath -Text (($manifest | ConvertTo-Json -Depth 8) + "`n")

gcloud storage cp $packagePath $sourceUri --project $ProjectId | Out-Null
gcloud storage cp $startupPath $startupUri --project $ProjectId | Out-Null
gcloud storage cp $manifestPath $manifestUri --project $ProjectId | Out-Null
gcloud storage cp $shardManifestPath "gs://$Bucket/runs/$RunName/source/shards_manifest.jsonl" --project $ProjectId | Out-Null

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
        "VM_COUNT=$VmCount",
        "START_SHARD=$i",
        "SOURCE_URI=$sourceUri",
        "PREDICTION_THREADS=$PredictionThreads",
        "SEED_STRIDE=$SeedStride",
        "OPENING_LOOKAHEAD_SAMPLES=$OpeningLookaheadSamples",
        "HU_TURN2_STAGE8B_MODEL=$huTurn2Stage8bModelRepoPath",
        "STAGE3_ENCODER_MODE=$Stage3FeatureEncoderMode",
        "BATCHED_CONTINUATION_BATCH_SIZE=$BatchedContinuationBatchSize",
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
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create instance $vmName in $zone"
        }
    }
}

[pscustomobject]@{
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    phase = "hu_t2_stage8b_topk_mc_rerank"
    games_per_seed = $GamesPerSeed
    configs = $configList
    seeds = $seedList
    total_shards = $totalShards
    vm_count = $VmCount
    machine_type = $MachineType
    package = $packagePath
    source_uri = $sourceUri
    manifest_uri = $manifestUri
    shard_manifest = $shardManifestPath
    create_instances = [bool]$CreateInstances
    instances = $instances
} | ConvertTo-Json -Depth 6
