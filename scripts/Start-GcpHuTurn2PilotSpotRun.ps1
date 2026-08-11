param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-t2-pilot-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [int]$SamplesPerBucket = 400,
    [int]$ShardSamples = 20,
    [int]$FutureSamples = 512,
    [int]$VmCount = 20,
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
    [int]$BaseSeed = 2026062500,
    [int]$PredictionThreads = 1,
    [int]$BootDiskGb = 50,
    [string]$PoolDir = "outputs/hu_turn2_stage1_batch9_3_pilot_2000_mc512/pools",
    [ValidateSet("stage3_reference_default", "stage7_m5_r10")]
    [string]$T3Continuation = "stage3_reference_default",
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

if ($SamplesPerBucket -le 0) { throw "SamplesPerBucket must be positive" }
if ($ShardSamples -le 0) { throw "ShardSamples must be positive" }
if ($SamplesPerBucket % $ShardSamples -ne 0) { throw "SamplesPerBucket must be divisible by ShardSamples" }
if ($FutureSamples -le 0) { throw "FutureSamples must be positive" }
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
$requiredModels = @(
    "models/opening_stage7_torch_wide.pt",
    "models/turn1_stage6_torch_wide.pt",
    "models/turn2_stage8.pkl",
    "models/turn3_stage6.pkl",
    "models/hu_turn3_stage7_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt"
)
foreach ($model in $requiredModels) {
    if (-not (Test-Path (Join-Path $repoRoot $model))) {
        throw "Required model not found: $model"
    }
}

$requiredPools = @(
    "predicted_high_regret_candidates.jsonl",
    "predicted_low_margin_candidates.jsonl",
    "predicted_teacher_disagreement_candidates.jsonl"
)
foreach ($pool in $requiredPools) {
    if (-not (Test-Path (Join-Path $repoRoot (Join-Path $PoolDir $pool)))) {
        throw "Required pool file not found: $(Join-Path $PoolDir $pool)"
    }
}

$bucketSpecs = @(
    [ordered]@{ name = "natural"; mode = "direct"; source_bucket = "natural"; pool_file = ""; offset_base = 0 },
    [ordered]@{ name = "predicted_teacher_disagreement_from_pool"; mode = "pool"; source_bucket = "from_pool"; pool_file = "pools/predicted_teacher_disagreement_candidates.jsonl"; offset_base = 0 },
    [ordered]@{ name = "predicted_high_regret_from_pool"; mode = "pool"; source_bucket = "from_pool"; pool_file = "pools/predicted_high_regret_candidates.jsonl"; offset_base = 0 },
    [ordered]@{ name = "predicted_low_margin_from_pool"; mode = "pool"; source_bucket = "from_pool"; pool_file = "pools/predicted_low_margin_candidates.jsonl"; offset_base = 0 },
    [ordered]@{ name = "random_off_policy"; mode = "direct"; source_bucket = "random_off_policy"; pool_file = ""; offset_base = 0 }
)
$shardsPerBucket = [int]($SamplesPerBucket / $ShardSamples)
$totalShards = $shardsPerBucket * $bucketSpecs.Count

$runDir = Join-Path $repoRoot ("outputs/gcp_runs/{0}" -f $RunName)
$packageDir = Join-Path $runDir "package_src"
$packagePath = Join-Path $runDir "ofc_regular_hu_t2_pilot_source.zip"
$startupPath = Join-Path $runDir "startup_hu_t2_pilot_spot.sh"
$manifestPath = Join-Path $runDir "manifest.json"
$shardManifestPath = Join-Path $runDir "shards_manifest.jsonl"
$sourceUri = "gs://$Bucket/runs/$RunName/source/ofc_regular_hu_t2_pilot_source.zip"
$startupUri = "gs://$Bucket/runs/$RunName/source/startup_hu_t2_pilot_spot.sh"
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

$modelPackageDir = Join-Path $packageDir "models"
New-Item -ItemType Directory -Force -Path $modelPackageDir | Out-Null
foreach ($model in $requiredModels) {
    Copy-Item -LiteralPath (Join-Path $repoRoot $model) -Destination (Join-Path $packageDir $model)
}

$poolPackageDir = Join-Path $packageDir "pools"
New-Item -ItemType Directory -Force -Path $poolPackageDir | Out-Null
foreach ($pool in $requiredPools) {
    Copy-Item -LiteralPath (Join-Path $repoRoot (Join-Path $PoolDir $pool)) -Destination (Join-Path $poolPackageDir $pool)
}

$shardLines = New-Object System.Collections.Generic.List[string]
$shardIndex = 0
for ($bucketIndex = 0; $bucketIndex -lt $bucketSpecs.Count; $bucketIndex += 1) {
    $bucketSpec = $bucketSpecs[$bucketIndex]
    for ($i = 0; $i -lt $shardsPerBucket; $i += 1) {
        $spec = [ordered]@{
            shard = $shardIndex
            bucket = $bucketSpec.name
            mode = $bucketSpec.mode
            source_bucket = $bucketSpec.source_bucket
            pool_file = $bucketSpec.pool_file
            pool_offset = $i * $ShardSamples
            samples = $ShardSamples
            seed = $BaseSeed + $shardIndex
            output_name = ("{0}_{1:D4}.jsonl" -f $bucketSpec.name, $i)
            summary_name = ("{0}_{1:D4}.summary.json" -f $bucketSpec.name, $i)
        }
        $shardLines.Add(($spec | ConvertTo-Json -Compress -Depth 5))
        $shardIndex += 1
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

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
}

meta() {
  curl -fsS -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

instance_meta() {
  curl -fsS -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$1"
}

json_field() {
  python -c 'import json,sys; print(json.loads(sys.argv[1]).get(sys.argv[2], ""))' "$1" "$2"
}

RUN_NAME="$(meta RUN_NAME)"
BUCKET="$(meta BUCKET)"
TOTAL_SHARDS="$(meta TOTAL_SHARDS)"
FUTURE_SAMPLES="$(meta FUTURE_SAMPLES)"
VM_COUNT="$(meta VM_COUNT)"
START_SHARD="$(meta START_SHARD)"
SOURCE_URI="$(meta SOURCE_URI)"
SELF_DELETE="$(meta SELF_DELETE)"
PREDICTION_THREADS="$(meta PREDICTION_THREADS)"
T3_CONTINUATION="$(meta T3_CONTINUATION)"
INSTANCE_NAME="$(instance_meta name)"
ZONE_PATH="$(instance_meta zone)"
ZONE="${ZONE_PATH##*/}"

cleanup_on_error() {
  code="$1"
  if [[ "$code" -eq 0 ]]; then
    return
  fi
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
sudo apt-get install -y unzip curl ca-certificates python3 python3-venv python3-pip build-essential

if ! command -v gcloud >/dev/null 2>&1; then
  log "google cloud sdk not found; installing google-cloud-cli"
  sudo apt-get install -y apt-transport-https gnupg
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y google-cloud-cli
fi

log "install rust toolchain"
if [[ ! -x "$HOME/.cargo/bin/rustup" ]]; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
fi
source "$HOME/.cargo/env"
rustup default stable
hash -r
rustc --version
cargo --version

WORK_DIR="/opt/ofc-regular-hu-t2-pilot"
sudo rm -rf "$WORK_DIR"
sudo mkdir -p "$WORK_DIR"
sudo chown "$CURRENT_UID:$CURRENT_GID" "$WORK_DIR"
log "download source package $SOURCE_URI"
gcloud storage cp "$SOURCE_URI" /tmp/ofc_regular_hu_t2_pilot_source.zip
unzip -q /tmp/ofc_regular_hu_t2_pilot_source.zip -d "$WORK_DIR"
cd "$WORK_DIR"

log "create python venv"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy==2.2.6" "scikit-learn==1.8.0"
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0"
export PYTHONPATH="$WORK_DIR/src"

log "build rust feature encoder"
source "$HOME/.cargo/env"
cargo build --release
python - <<'PY'
from ofc_regular.hu_turn3_stage3_feature_rust import rust_direct_available
assert rust_direct_available(), "rust_direct encoder unavailable"
print("rust_direct_available=true")
PY

STATUS_DIR="/tmp/ofc-hu-t2-status"
SHARD_DIR="/tmp/ofc-hu-t2-shards"
SUMMARY_DIR="/tmp/ofc-hu-t2-summaries"
mkdir -p "$STATUS_DIR" "$SHARD_DIR" "$SUMMARY_DIR"

log "start shard loop run=$RUN_NAME start=$START_SHARD stride=$VM_COUNT total_shards=$TOTAL_SHARDS future=$FUTURE_SAMPLES"
for (( shard=START_SHARD; shard<TOTAL_SHARDS; shard+=VM_COUNT )); do
  spec="$(sed -n "$((shard + 1))p" shards_manifest.jsonl)"
  if [[ -z "$spec" ]]; then
    log "missing spec for shard=$shard"
    exit 2
  fi

  bucket_name="$(json_field "$spec" bucket)"
  mode="$(json_field "$spec" mode)"
  source_bucket="$(json_field "$spec" source_bucket)"
  pool_file="$(json_field "$spec" pool_file)"
  pool_offset="$(json_field "$spec" pool_offset)"
  samples="$(json_field "$spec" samples)"
  seed="$(json_field "$spec" seed)"
  output_name="$(json_field "$spec" output_name)"
  summary_name="$(json_field "$spec" summary_name)"

  remote="gs://${BUCKET}/runs/${RUN_NAME}/shards/${output_name}"
  summary_remote="gs://${BUCKET}/runs/${RUN_NAME}/summaries/${summary_name}"
  status_remote="gs://${BUCKET}/runs/${RUN_NAME}/status/${output_name%.jsonl}.json"
  if gcloud storage ls "$remote" >/dev/null 2>&1; then
    log "skip existing shard $shard $bucket_name"
    continue
  fi

  local_path="${SHARD_DIR}/${output_name}"
  summary_path="${SUMMARY_DIR}/${summary_name}"
  log_path="${SUMMARY_DIR}/${output_name%.jsonl}.log"
  status_path="${STATUS_DIR}/${output_name%.jsonl}.json"
  rm -f "$local_path" "$summary_path" "$status_path"
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_seconds="$(date +%s)"
  log "generate shard=$shard bucket=$bucket_name mode=$mode seed=$seed samples=$samples"

  extra_args=(--source-bucket "$source_bucket")
  if [[ "$mode" == "pool" ]]; then
    pool_shard="/tmp/pool_${shard}.jsonl"
    start_line=$((pool_offset + 1))
    end_line=$((pool_offset + samples))
    awk -v start="$start_line" -v end="$end_line" 'NR >= start && NR <= end { print }' "$pool_file" > "$pool_shard"
    pool_lines="$(wc -l < "$pool_shard" | tr -d ' ')"
    if [[ "$pool_lines" -ne "$samples" ]]; then
      log "pool shard line mismatch shard=$shard lines=$pool_lines expected=$samples"
      exit 3
    fi
    extra_args=(--candidate-pool-input "$pool_shard" --source-bucket from_pool)
  fi

  set +e
  python -B -m ofc_regular.hu_turn2_teacher_data \
    --samples "$samples" \
    --seed "$seed" \
    --future-samples "$FUTURE_SAMPLES" \
    --prediction-threads "$PREDICTION_THREADS" \
    --t3-continuation "$T3_CONTINUATION" \
    --use-batched-continuation \
    --stage3-feature-encoder-mode rust_direct \
    --output "$local_path" \
    --summary-output "$summary_path" \
    "${extra_args[@]}" > "$log_path" 2>&1
  exit_code=$?
  set -e

  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end_seconds="$(date +%s)"
  elapsed=$((end_seconds - start_seconds))
  lines=0
  if [[ -f "$local_path" ]]; then
    lines="$(wc -l < "$local_path" | tr -d ' ')"
  fi

  if [[ "$exit_code" -eq 0 && "$lines" -eq "$samples" ]]; then
    tmp_remote="${remote}.tmp-${INSTANCE_NAME}"
    tmp_summary_remote="${summary_remote}.tmp-${INSTANCE_NAME}"
    log_remote="gs://${BUCKET}/runs/${RUN_NAME}/logs/${output_name%.jsonl}.log"
    log "upload shard=$shard bucket=$bucket_name lines=$lines elapsed=${elapsed}s"
    gcloud storage cp "$local_path" "$tmp_remote"
    gcloud storage mv "$tmp_remote" "$remote"
    if [[ -f "$summary_path" ]]; then
      gcloud storage cp "$summary_path" "$tmp_summary_remote"
      gcloud storage mv "$tmp_summary_remote" "$summary_remote"
    fi
    if [[ -f "$log_path" ]]; then
      gcloud storage cp "$log_path" "$log_remote" || true
    fi
    cat > "$status_path" <<EOF
{"run_name":"$RUN_NAME","phase":"hu_t2_pilot","shard":$shard,"bucket":"$bucket_name","mode":"$mode","status":"complete","instance":"$INSTANCE_NAME","zone":"$ZONE","seed":$seed,"samples":$samples,"future_samples":$FUTURE_SAMPLES,"t3_continuation":"$T3_CONTINUATION","lines":$lines,"elapsed_seconds":$elapsed,"started_at":"$started_at","finished_at":"$finished_at","output":"$remote","summary":"$summary_remote","log":"$log_remote"}
EOF
    gcloud storage cp "$status_path" "$status_remote" || true
  else
    log_remote="gs://${BUCKET}/runs/${RUN_NAME}/logs/${output_name%.jsonl}.log"
    if [[ -f "$log_path" ]]; then
      gcloud storage cp "$log_path" "$log_remote" || true
    fi
    log "failed shard=$shard exit=$exit_code lines=$lines expected=$samples"
    cat > "$status_path" <<EOF
{"run_name":"$RUN_NAME","phase":"hu_t2_pilot","shard":$shard,"bucket":"$bucket_name","mode":"$mode","status":"failed","instance":"$INSTANCE_NAME","zone":"$ZONE","seed":$seed,"samples":$samples,"future_samples":$FUTURE_SAMPLES,"t3_continuation":"$T3_CONTINUATION","lines":$lines,"exit_code":$exit_code,"elapsed_seconds":$elapsed,"started_at":"$started_at","finished_at":"$finished_at","log":"$log_remote"}
EOF
    gcloud storage cp "$status_path" "$status_remote" || true
    exit "$exit_code"
  fi
done

log "all assigned shards finished"
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
    phase = "hu_t2_pilot"
    samples_per_bucket = $SamplesPerBucket
    shard_samples = $ShardSamples
    total_shards = $totalShards
    future_samples = $FutureSamples
    t3_continuation = $T3Continuation
    vm_count = $VmCount
    machine_type = $MachineType
    zones = $Zones
    base_seed = $BaseSeed
    prediction_threads = $PredictionThreads
    pool_dir = $PoolDir
    source_uri = $sourceUri
    startup_uri = $startupUri
    self_delete = -not $NoSelfDelete
    created_at = (Get-Date).ToUniversalTime().ToString("o")
    buckets = $bucketSpecs
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
        "FUTURE_SAMPLES=$FutureSamples",
        "VM_COUNT=$VmCount",
        "START_SHARD=$i",
        "SOURCE_URI=$sourceUri",
        "PREDICTION_THREADS=$PredictionThreads",
        "T3_CONTINUATION=$T3Continuation",
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
    phase = "hu_t2_pilot"
    samples_per_bucket = $SamplesPerBucket
    shard_samples = $ShardSamples
    total_shards = $totalShards
    future_samples = $FutureSamples
    t3_continuation = $T3Continuation
    vm_count = $VmCount
    machine_type = $MachineType
    package = $packagePath
    source_uri = $sourceUri
    manifest_uri = $manifestUri
    shard_manifest = $shardManifestPath
    create_instances = [bool]$CreateInstances
    instances = $instances
} | ConvertTo-Json -Depth 6
