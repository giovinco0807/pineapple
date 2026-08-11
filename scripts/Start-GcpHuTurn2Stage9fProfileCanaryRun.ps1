param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-t2-stage9f-profile-canary-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [int]$TotalGames = 20000,
    [int]$ShardGames = 1000,
    [int]$BaseSeed = 2026068001,
    [int]$SeedStride = 1000000,
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
    [int]$PredictionThreads = 1,
    [int]$OpeningLookaheadSamples = 1,
    [string]$ProfileA = "stage9f_cse2_csemax2_bothseat",
    [string]$ProfileB = "stage7_m5_r10",
    [string]$Phase = "hu_t2_stage9f_profile_canary",
    [string]$HuTurn1TopkConfig = "",
    [string]$HuTurn1Stage1Model = "models/hu_turn1_stage1_stage9f_p2_full2k_hgb_leaf3_lr01_l2_1.pkl",
    [string[]]$HuTurn1Stage1Models = @(),
    [string]$HuTurn1SafeSelectorModel = "models/hu_turn1_safe_override_selector_teacher_regret_combined296_accept0p25_gray1_delta_plus_meta.pkl",
    [string]$HuTurn2Stage8bModel = "models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt",
    [int]$BootDiskGb = 50,
    [string[]]$StartShards = @(),
    [switch]$CreateInstances,
    [switch]$SkipExistingInstances,
    [switch]$ReuseRunArtifacts,
    [switch]$NoSelfDelete,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (Test-Path $cloudSdkGcloud) {
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}
elseif (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud not found in PATH or at $cloudSdkGcloud"
}

if ($TotalGames -le 0) { throw "TotalGames must be positive" }
if ($ShardGames -le 0) { throw "ShardGames must be positive" }
if ($VmCount -le 0) { throw "VmCount must be positive" }
if ($SeedStride -le 0) { throw "SeedStride must be positive" }
if ($PredictionThreads -lt 0) { throw "PredictionThreads must be non-negative" }
if ($OpeningLookaheadSamples -lt 0) { throw "OpeningLookaheadSamples must be non-negative" }
if (-not $Zones -or $Zones.Count -eq 0) { throw "At least one zone is required" }
if ($RunName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') {
    throw "RunName may only contain letters, numbers, dot, underscore, and dash"
}

function Convert-ToVmName {
    param([string]$Name)
    $vmName = $Name.ToLowerInvariant() -replace '[^a-z0-9-]', '-'
    $vmName = $vmName.Trim('-')
    if ($vmName.Length -gt 54) { $vmName = $vmName.Substring(0, 54).Trim('-') }
    if (-not $vmName) { throw "RunName does not produce a valid VM name" }
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
    if (Test-Path $DestinationPath) { Remove-Item -LiteralPath $DestinationPath -Force }
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
    "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt",
    $HuTurn1SafeSelectorModel,
    $HuTurn2Stage8bModel
) | Select-Object -Unique
if ($HuTurn1Stage1Models.Count -gt 0) {
    $requiredModels += $HuTurn1Stage1Models
}
else {
    $requiredModels += $HuTurn1Stage1Model
}
$requiredModels = @($requiredModels | Select-Object -Unique)
foreach ($model in $requiredModels) {
    if ([System.IO.Path]::IsPathRooted($model)) {
        throw "Model paths must be repo-relative: $model"
    }
    if (-not (Test-Path (Join-Path $repoRoot $model))) {
        throw "Required model not found: $model"
    }
}

$totalShards = [int][Math]::Ceiling($TotalGames / [double]$ShardGames)
$StartShardIndices = @()
foreach ($startShardValue in $StartShards) {
    foreach ($part in ($startShardValue -split ",")) {
        $trimmed = $part.Trim()
        if ($trimmed) { $StartShardIndices += [int]$trimmed }
    }
}
$StartShardIndices = @($StartShardIndices | Sort-Object -Unique)
foreach ($startShard in $StartShardIndices) {
    if ($startShard -lt 0 -or $startShard -ge $totalShards) {
        throw "StartShards values must be in [0, total_shards): $startShard"
    }
}

$shardSpecs = New-Object System.Collections.Generic.List[object]
for ($shardIndex = 0; $shardIndex -lt $totalShards; $shardIndex += 1) {
    $gamesDone = $shardIndex * $ShardGames
    $games = [Math]::Min($ShardGames, $TotalGames - $gamesDone)
    $seed = $BaseSeed + ($shardIndex * $ShardGames * $SeedStride)
    $shardSpecs.Add([ordered]@{
        shard = $shardIndex
        games = $games
        seed = $seed
        output_prefix = ("shard_{0:D3}_seed{1}_games{2:D5}" -f $shardIndex, $seed, $games)
    })
}

if ($DryRun) {
    [pscustomobject]@{
        execution = "dry_run"
        run_name = $RunName
        project_id = $ProjectId
        bucket = $Bucket
        phase = $Phase
        profile_a = $ProfileA
        profile_b = $ProfileB
        hu_turn1_topk_config = $HuTurn1TopkConfig
        hu_turn1_stage1_model = $HuTurn1Stage1Model
        hu_turn1_stage1_models = $HuTurn1Stage1Models
        hu_turn1_safe_selector_model = $HuTurn1SafeSelectorModel
        total_games = $TotalGames
        total_hands = $TotalGames * 2
        total_decisions = $TotalGames * 2
        shard_games = $ShardGames
        total_shards = $totalShards
        base_seed = $BaseSeed
        seed_stride = $SeedStride
        vm_count = $VmCount
        machine_type = $MachineType
        zones = $Zones
        prediction_threads = $PredictionThreads
        opening_lookahead_samples = $OpeningLookaheadSamples
        hu_turn2_stage8b_model = $HuTurn2Stage8bModel
        create_instances = [bool]$CreateInstances
        reuse_run_artifacts = [bool]$ReuseRunArtifacts
        production_default = "No-Go"
        production_p2_fixed = "No-Go"
        t1_training = "No-Go"
        teacher_50k = "No-Go"
        shards = $shardSpecs
    } | ConvertTo-Json -Depth 8
    exit 0
}

$runDir = Join-Path $repoRoot ("outputs/gcp_runs/{0}" -f $RunName)
$packageDir = Join-Path $runDir "package_src"
$packagePath = Join-Path $runDir "ofc_regular_hu_t2_stage9f_profile_canary_source.zip"
$startupPath = Join-Path $runDir "startup_hu_t2_stage9f_profile_canary_spot.sh"
$manifestPath = Join-Path $runDir "manifest.json"
$shardManifestPath = Join-Path $runDir "shards_manifest.jsonl"
$sourceUri = "gs://$Bucket/runs/$RunName/source/ofc_regular_hu_t2_stage9f_profile_canary_source.zip"
$startupUri = "gs://$Bucket/runs/$RunName/source/startup_hu_t2_stage9f_profile_canary_spot.sh"
$manifestUri = "gs://$Bucket/runs/$RunName/manifest.json"

if ($ReuseRunArtifacts) {
    foreach ($artifact in @($packagePath, $startupPath, $manifestPath, $shardManifestPath)) {
        if (-not (Test-Path -LiteralPath $artifact)) {
            throw "Cannot reuse missing run artifact: $artifact"
        }
    }
}
else {
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    if (Test-Path $packageDir) { Remove-Item -LiteralPath $packageDir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $packageDir | Out-Null

foreach ($item in @("pyproject.toml", "Cargo.toml", "Cargo.lock", "src", "rust", "configs")) {
    $source = Join-Path $repoRoot $item
    if (-not (Test-Path $source)) { throw "Missing package input: $item" }
    Copy-Item -LiteralPath $source -Destination $packageDir -Recurse
}
foreach ($model in $requiredModels) {
    $dest = Join-Path $packageDir $model
    New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
    Copy-Item -LiteralPath (Join-Path $repoRoot $model) -Destination $dest
}

$shardLines = @($shardSpecs | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 5 })
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
OPENING_LOOKAHEAD_SAMPLES="$(meta OPENING_LOOKAHEAD_SAMPLES)"
PROFILE_A="$(meta PROFILE_A)"
PROFILE_B="$(meta PROFILE_B)"
SEED_STRIDE="$(meta SEED_STRIDE)"
HU_TURN1_TOPK_CONFIG="$(meta HU_TURN1_TOPK_CONFIG)"
HU_TURN1_STAGE1_MODEL="$(meta HU_TURN1_STAGE1_MODEL)"
HU_TURN1_STAGE1_MODELS="$(meta HU_TURN1_STAGE1_MODELS)"
HU_TURN1_SAFE_SELECTOR_MODEL="$(meta HU_TURN1_SAFE_SELECTOR_MODEL)"
HU_TURN2_STAGE8B_MODEL="$(meta HU_TURN2_STAGE8B_MODEL)"
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

sudo apt-get update -y
sudo apt-get install -y unzip curl ca-certificates python3 python3-venv python3-pip

if ! command -v gcloud >/dev/null 2>&1; then
  sudo apt-get install -y apt-transport-https gnupg
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y google-cloud-cli
fi

if [[ ! -x "$HOME/.cargo/bin/rustup" ]]; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
fi
source "$HOME/.cargo/env"
rustup default stable

WORK_DIR="/opt/ofc-regular-hu-t2-stage9f-profile-canary"
sudo rm -rf "$WORK_DIR"
sudo mkdir -p "$WORK_DIR"
sudo chown "$CURRENT_UID:$CURRENT_GID" "$WORK_DIR"
gcloud storage cp "$SOURCE_URI" /tmp/ofc_regular_hu_t2_stage9f_profile_canary_source.zip
unzip -q /tmp/ofc_regular_hu_t2_stage9f_profile_canary_source.zip -d "$WORK_DIR"
cd "$WORK_DIR"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy==2.2.6" "scikit-learn==1.8.0"
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0"
export PYTHONPATH="$WORK_DIR/src"

source "$HOME/.cargo/env"
cargo build --release
python - <<'PY'
from ofc_regular.hu_turn3_stage3_feature_rust import rust_direct_available
assert rust_direct_available(), "rust_direct encoder unavailable"
print("rust_direct_available=true")
PY

STATUS_DIR="/tmp/ofc-hu-t2-stage9f-profile-canary-status"
RESULT_DIR="/tmp/ofc-hu-t2-stage9f-profile-canary-results"
mkdir -p "$STATUS_DIR" "$RESULT_DIR"

log "start Stage9f profile canary loop run=$RUN_NAME start=$START_SHARD stride=$VM_COUNT total_shards=$TOTAL_SHARDS"
for (( shard=START_SHARD; shard<TOTAL_SHARDS; shard+=VM_COUNT )); do
  spec="$(sed -n "$((shard + 1))p" shards_manifest.jsonl)"
  if [[ -z "$spec" ]]; then
    log "missing spec for shard=$shard"
    exit 2
  fi
  games="$(json_field "$spec" games)"
  seed="$(json_field "$spec" seed)"
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
  log "run shard=$shard seed=$seed games=$games profile_a=$PROFILE_A profile_b=$PROFILE_B"

  set +e
  match_args=(
    -B -m ofc_regular.evaluate_matchups
    --profile-a "$PROFILE_A"
    --profile-b "$PROFILE_B"
    --games "$games"
    --seed "$seed"
    --seed-stride "$SEED_STRIDE"
    --opening-lookahead-samples "$OPENING_LOOKAHEAD_SAMPLES"
    --prediction-threads "$PREDICTION_THREADS"
    --trace-limit 0
    --hu-turn1-safe-selector-model "$HU_TURN1_SAFE_SELECTOR_MODEL"
    --hu-turn2-stage8b-model "$HU_TURN2_STAGE8B_MODEL"
    --topk-decision-output "${local_out}/topk_decisions.jsonl"
    --hu-turn1-decision-output "${local_out}/hu_turn1_decisions.jsonl"
    --output "${local_out}/summary.json"
    --progress-every 0
  )
  if [[ -n "$HU_TURN1_TOPK_CONFIG" ]]; then
    match_args+=(--hu-turn1-topk-config "$HU_TURN1_TOPK_CONFIG")
  fi
  if [[ -n "$HU_TURN1_STAGE1_MODELS" ]]; then
    IFS=';' read -r -a hu_turn1_stage1_model_array <<< "$HU_TURN1_STAGE1_MODELS"
    match_args+=(--hu-turn1-stage1-models "${hu_turn1_stage1_model_array[@]}")
  else
    match_args+=(--hu-turn1-stage1-model "$HU_TURN1_STAGE1_MODEL")
  fi
  python "${match_args[@]}" > "$log_path" 2>&1
  exit_code=$?
  set -e

  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end_seconds="$(date +%s)"
  elapsed=$((end_seconds - start_seconds))
  if [[ "$exit_code" -eq 0 ]]; then
    printf 'complete\n' > "${local_out}/DONE"
    gcloud storage cp "$local_out"/* "gs://${BUCKET}/runs/${RUN_NAME}/results/${output_prefix}/" >/dev/null
    cat > "$status_path" <<EOF
{"run_name":"$RUN_NAME","phase":"hu_t2_stage9f_profile_canary","shard":$shard,"games":$games,"seed":$seed,"profile_a":"$PROFILE_A","profile_b":"$PROFILE_B","status":"complete","instance":"$INSTANCE_NAME","zone":"$ZONE","elapsed_seconds":$elapsed,"started_at":"$started_at","finished_at":"$finished_at","output":"gs://${BUCKET}/runs/${RUN_NAME}/results/${output_prefix}"}
EOF
    gcloud storage cp "$status_path" "$status_remote" >/dev/null || true
  else
    gcloud storage cp "$log_path" "gs://${BUCKET}/runs/${RUN_NAME}/logs/${output_prefix}.log" >/dev/null || true
    cat > "$status_path" <<EOF
{"run_name":"$RUN_NAME","phase":"hu_t2_stage9f_profile_canary","shard":$shard,"games":$games,"seed":$seed,"profile_a":"$PROFILE_A","profile_b":"$PROFILE_B","status":"failed","instance":"$INSTANCE_NAME","zone":"$ZONE","exit_code":$exit_code,"elapsed_seconds":$elapsed,"started_at":"$started_at","finished_at":"$finished_at","log":"gs://${BUCKET}/runs/${RUN_NAME}/logs/${output_prefix}.log"}
EOF
    gcloud storage cp "$status_path" "$status_remote" >/dev/null || true
    exit "$exit_code"
  fi
done

if [[ "$SELF_DELETE" == "1" ]]; then
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
    phase = $Phase
    profile_a = $ProfileA
    profile_b = $ProfileB
    total_games = $TotalGames
    total_hands = $TotalGames * 2
    total_decisions = $TotalGames * 2
    shard_games = $ShardGames
    total_shards = $totalShards
    base_seed = $BaseSeed
    vm_count = $VmCount
    machine_type = $MachineType
    zones = $Zones
    seed_stride = $SeedStride
    prediction_threads = $PredictionThreads
    opening_lookahead_samples = $OpeningLookaheadSamples
    hu_turn1_topk_config = $HuTurn1TopkConfig
    hu_turn2_stage8b_model = $HuTurn2Stage8bModel
    hu_turn1_stage1_model = $HuTurn1Stage1Model
    hu_turn1_stage1_models = $HuTurn1Stage1Models
    hu_turn1_safe_selector_model = $HuTurn1SafeSelectorModel
    source_uri = $sourceUri
    startup_uri = $startupUri
    self_delete = -not $NoSelfDelete
    production_default = "No-Go"
    production_p2_fixed = "No-Go"
    t1_training = "No-Go"
    teacher_50k = "No-Go"
    created_at = (Get-Date).ToUniversalTime().ToString("o")
}
Write-Utf8NoBom -Path $manifestPath -Text (($manifest | ConvertTo-Json -Depth 8) + "`n")

    gcloud storage cp $packagePath $sourceUri --project $ProjectId | Out-Null
    gcloud storage cp $startupPath $startupUri --project $ProjectId | Out-Null
    gcloud storage cp $manifestPath $manifestUri --project $ProjectId | Out-Null
    gcloud storage cp $shardManifestPath "gs://$Bucket/runs/$RunName/source/shards_manifest.jsonl" --project $ProjectId | Out-Null
}

$vmPrefix = Convert-ToVmName $RunName
$instances = @()
$workerIndices = if ($StartShardIndices.Count -gt 0) { $StartShardIndices } else { @(0..($VmCount - 1)) }
foreach ($i in $workerIndices) {
    $preferredZone = $Zones[$i % $Zones.Count]
    $zone = $preferredZone
    $vmName = ("{0}-{1:d3}" -f $vmPrefix, $i)
    if ($vmName.Length -gt 63) { $vmName = $vmName.Substring(0, 63).Trim("-") }
    $metadata = @(
        "RUN_NAME=$RunName",
        "BUCKET=$Bucket",
        "TOTAL_SHARDS=$totalShards",
        "VM_COUNT=$VmCount",
        "START_SHARD=$i",
        "SOURCE_URI=$sourceUri",
        "PREDICTION_THREADS=$PredictionThreads",
        "OPENING_LOOKAHEAD_SAMPLES=$OpeningLookaheadSamples",
        "PROFILE_A=$ProfileA",
        "PROFILE_B=$ProfileB",
        "SEED_STRIDE=$SeedStride",
        "HU_TURN1_TOPK_CONFIG=$HuTurn1TopkConfig",
        "HU_TURN1_STAGE1_MODEL=$HuTurn1Stage1Model",
        "HU_TURN1_STAGE1_MODELS=$($HuTurn1Stage1Models -join ';')",
        "HU_TURN1_SAFE_SELECTOR_MODEL=$HuTurn1SafeSelectorModel",
        "HU_TURN2_STAGE8B_MODEL=$HuTurn2Stage8bModel",
        ("SELF_DELETE=" + ($(if ($NoSelfDelete) { "0" } else { "1" })))
    ) -join ","

    if ($CreateInstances) {
        if ($SkipExistingInstances) {
            $existing = @(gcloud compute instances list `
                --project $ProjectId `
                --filter ("name='" + $vmName + "'") `
                --format "csv[no-heading](name,zone.basename())" 2>$null)
            if ($existing.Count -gt 0) {
                $zone = ($existing[0] -split ",")[-1]
                $instances += [pscustomobject]@{ name = $vmName; zone = $zone; start_shard = $i }
                continue
            }
        }
        $created = $false
        $zoneCandidates = @($preferredZone) + @($Zones | Where-Object { $_ -ne $preferredZone })
        foreach ($candidateZone in $zoneCandidates) {
            $zone = $candidateZone
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
            if ($LASTEXITCODE -eq 0) {
                $created = $true
                break
            }
            Write-Warning "Failed to create $vmName in $zone; trying the next configured zone"
        }
        if (-not $created) {
            throw "Failed to create instance $vmName in all configured zones"
        }
    }
    $instances += [pscustomobject]@{ name = $vmName; zone = $zone; start_shard = $i }
}

[pscustomobject]@{
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    phase = $Phase
    profile_a = $ProfileA
    profile_b = $ProfileB
    total_games = $TotalGames
    total_hands = $TotalGames * 2
    total_decisions = $TotalGames * 2
    shard_games = $ShardGames
    total_shards = $totalShards
    vm_count = $VmCount
    machine_type = $MachineType
    package = $packagePath
    source_uri = $sourceUri
    manifest_uri = $manifestUri
    shard_manifest = $shardManifestPath
    create_instances = [bool]$CreateInstances
    reuse_run_artifacts = [bool]$ReuseRunArtifacts
    instances = $instances
} | ConvertTo-Json -Depth 6
