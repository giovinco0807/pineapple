param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-t0-counterfactual-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [Parameter(Mandatory = $true)]
    [string]$CandidateModel,
    [int]$CandidateTopk = 60,
    [string]$SafeSelectorModel = "",
    [double]$SafeSelectorThresholdFirst = 0.0,
    [double]$SafeSelectorThresholdSecond = 0.0,
    [string[]]$Configs = @("m1/1/1/first+second"),
    [int]$PairedSeedsPerConfig = 2000,
    [int]$GamesPerShard = 20,
    [long]$BaseSeed = 2026107001,
    [long]$SeedStride = 1000003,
    [int]$VmCount = 80,
    [string]$MachineType = "c4-highmem-2",
    [int]$CpuThreads = 1,
    [int]$OpeningLookaheadSamples = 1,
    [string[]]$Zones = @(
        "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f",
        "us-east4-a", "us-east4-b", "us-east4-c",
        "us-west1-a", "us-west1-b", "us-west1-c"
    ),
    [int]$BootDiskGb = 50,
    [string[]]$StartShards = @(),
    [switch]$CreateInstances,
    [switch]$SkipExistingInstances,
    [switch]$NoSpot,
    [switch]$NoSelfDelete,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (Test-Path $cloudSdkGcloud) {
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}
elseif (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud not found"
}

if ($CandidateTopk -le 0) { throw "CandidateTopk must be positive" }
if ($SafeSelectorThresholdFirst -lt 0.0 -or $SafeSelectorThresholdFirst -gt 1.0) {
    throw "SafeSelectorThresholdFirst must be in [0, 1]"
}
if ($SafeSelectorThresholdSecond -lt 0.0 -or $SafeSelectorThresholdSecond -gt 1.0) {
    throw "SafeSelectorThresholdSecond must be in [0, 1]"
}
if ($PairedSeedsPerConfig -le 0) { throw "PairedSeedsPerConfig must be positive" }
if ($GamesPerShard -le 0) { throw "GamesPerShard must be positive" }
if ($SeedStride -le 0) { throw "SeedStride must be positive" }
if ($VmCount -le 0) { throw "VmCount must be positive" }
if ($CpuThreads -le 0) { throw "CpuThreads must be positive" }
if (-not $Zones) { throw "At least one zone is required" }
if ($RunName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') { throw "Invalid RunName" }
if ([System.IO.Path]::IsPathRooted($CandidateModel)) {
    throw "CandidateModel must be repo-relative"
}
if ($SafeSelectorModel -and [System.IO.Path]::IsPathRooted($SafeSelectorModel)) {
    throw "SafeSelectorModel must be repo-relative"
}

function Convert-ToVmName {
    param([string]$Name)
    $value = ($Name.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-')
    if ($value.Length -gt 54) { $value = $value.Substring(0, 54).Trim('-') }
    if (-not $value) { throw "RunName does not produce a valid VM name" }
    return $value
}

function Write-Utf8NoBom {
    param([string]$Path, [string]$Text)
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.UTF8Encoding]::new($false))
}

function New-ZipWithForwardSlashes {
    param([string]$SourceDir, [string]$DestinationPath)
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (Test-Path $DestinationPath) { Remove-Item -LiteralPath $DestinationPath -Force }
    $sourceRoot = (Resolve-Path -LiteralPath $SourceDir).Path.TrimEnd('\', '/')
    $prefixLength = $sourceRoot.Length + 1
    $zip = [System.IO.Compression.ZipFile]::Open(
        $DestinationPath,
        [System.IO.Compression.ZipArchiveMode]::Create
    )
    try {
        Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | ForEach-Object {
            $entryName = $_.FullName.Substring($prefixLength) -replace '\\', '/'
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $zip,
                $_.FullName,
                $entryName,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
    finally { $zip.Dispose() }
}

$parsedConfigs = @()
foreach ($rawValue in $Configs) {
    foreach ($raw in ($rawValue -split ',')) {
        if (-not $raw.Trim()) { continue }
        $parts = @($raw.Trim() -split '/')
        if ($parts.Count -ne 4) {
            throw "Config must be id/first_margin/second_margin/first+second: $raw"
        }
        $allowedSeats = @($parts[3] -split '\+' | Where-Object { $_ })
        if (-not $allowedSeats -or @($allowedSeats | Where-Object { $_ -notin @('first', 'second') }).Count) {
            throw "Invalid allowed seats in config: $raw"
        }
        $parsedConfigs += [pscustomobject]@{
            id = $parts[0]
            first_margin = [double]$parts[1]
            second_margin = [double]$parts[2]
            allowed_seats = ($allowedSeats -join ',')
        }
    }
}
if (-not $parsedConfigs) { throw "At least one config is required" }
$duplicateConfigIds = @($parsedConfigs | Group-Object id | Where-Object Count -gt 1)
if ($duplicateConfigIds) { throw "Duplicate config ids: $($duplicateConfigIds.Name -join ',')" }

$repoRoot = (Get-Location).Path
$candidateModelPath = Join-Path $repoRoot (($CandidateModel -replace '\\', '/').TrimStart('/'))
if (-not (Test-Path -LiteralPath $candidateModelPath)) {
    throw "Candidate model not found: $CandidateModel"
}
$requiredModels = @(
    "models/opening_stage7_torch_wide.pt",
    "models/turn1_stage6_torch_wide.pt",
    "models/turn2_stage8.pkl",
    "models/turn3_stage6.pkl",
    "models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl",
    "models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl",
    "models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage7_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt",
    ($CandidateModel -replace '\\', '/')
) | Select-Object -Unique
if ($SafeSelectorModel) {
    $requiredModels += ($SafeSelectorModel -replace '\\', '/')
    $requiredModels = @($requiredModels | Select-Object -Unique)
}
foreach ($model in $requiredModels) {
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $model))) {
        throw "Required model not found: $model"
    }
}

$runDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
$packageDir = Join-Path $runDir "package_src"
$packagePath = Join-Path $runDir "ofc_regular_hu_t0_counterfactual_source.zip"
$startupPath = Join-Path $runDir "startup_hu_t0_counterfactual.sh"
$manifestPath = Join-Path $runDir "manifest.json"
$shardManifestPath = Join-Path $runDir "shards_manifest.jsonl"
$sourceUri = "gs://$Bucket/runs/$RunName/source/ofc_regular_hu_t0_counterfactual_source.zip"
$startupUri = "gs://$Bucket/runs/$RunName/source/startup_hu_t0_counterfactual.sh"
$manifestUri = "gs://$Bucket/runs/$RunName/manifest.json"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
if (Test-Path $packageDir) { Remove-Item -LiteralPath $packageDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $packageDir | Out-Null
foreach ($item in @("pyproject.toml", "Cargo.toml", "Cargo.lock", "src", "rust", "configs")) {
    $source = Join-Path $repoRoot $item
    if (-not (Test-Path $source)) { throw "Missing package input: $item" }
    Copy-Item -LiteralPath $source -Destination $packageDir -Recurse
}
foreach ($model in $requiredModels) {
    $destination = Join-Path $packageDir $model
    New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
    Copy-Item -LiteralPath (Join-Path $repoRoot $model) -Destination $destination
}

$chunksPerConfig = [int][Math]::Ceiling($PairedSeedsPerConfig / [double]$GamesPerShard)
$shardLines = New-Object System.Collections.Generic.List[string]
$shardIndex = 0
foreach ($config in $parsedConfigs) {
    for ($chunk = 0; $chunk -lt $chunksPerConfig; $chunk += 1) {
        $startIndex = $chunk * $GamesPerShard
        $games = [Math]::Min($GamesPerShard, $PairedSeedsPerConfig - $startIndex)
        $seed = $BaseSeed + ($startIndex * $SeedStride)
        $safeId = ($config.id -replace '[^0-9A-Za-z_-]+', '_')
        $spec = [ordered]@{
            shard = $shardIndex
            config_id = $config.id
            min_margin_first = $config.first_margin
            min_margin_second = $config.second_margin
            allowed_seats = $config.allowed_seats
            games = $games
            seed = $seed
            output_prefix = ("shard_{0:D4}_{1}_seed{2}" -f $shardIndex, $safeId, $seed)
        }
        $shardLines.Add(($spec | ConvertTo-Json -Compress -Depth 5))
        $shardIndex += 1
    }
}
$totalShards = $shardIndex
$effectiveVmCount = [Math]::Min($VmCount, $totalShards)
$workerIndices = @()
foreach ($raw in $StartShards) {
    foreach ($part in ($raw -split ',')) {
        if ($part.Trim()) { $workerIndices += [int]$part.Trim() }
    }
}
$workerIndices = @($workerIndices | Sort-Object -Unique)
if (-not $workerIndices) { $workerIndices = @(0..($effectiveVmCount - 1)) }
foreach ($worker in $workerIndices) {
    if ($worker -lt 0 -or $worker -ge $effectiveVmCount) {
        throw "StartShards worker must be in [0, $effectiveVmCount): $worker"
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
CANDIDATE_MODEL="$(meta CANDIDATE_MODEL)"
CANDIDATE_TOPK="$(meta CANDIDATE_TOPK)"
SAFE_SELECTOR_MODEL="$(meta SAFE_SELECTOR_MODEL)"
SAFE_SELECTOR_THRESHOLD_FIRST="$(meta SAFE_SELECTOR_THRESHOLD_FIRST)"
SAFE_SELECTOR_THRESHOLD_SECOND="$(meta SAFE_SELECTOR_THRESHOLD_SECOND)"
SEED_STRIDE="$(meta SEED_STRIDE)"
CPU_THREADS="$(meta CPU_THREADS)"
OPENING_LOOKAHEAD_SAMPLES="$(meta OPENING_LOOKAHEAD_SAMPLES)"
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
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
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

WORK_DIR="/opt/ofc-regular-hu-t0-counterfactual"
sudo rm -rf "$WORK_DIR"
sudo mkdir -p "$WORK_DIR"
sudo chown "$CURRENT_UID:$CURRENT_GID" "$WORK_DIR"
gcloud storage cp "$SOURCE_URI" /tmp/ofc_regular_hu_t0_counterfactual_source.zip
unzip -q /tmp/ofc_regular_hu_t0_counterfactual_source.zip -d "$WORK_DIR"
cd "$WORK_DIR"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy==2.2.6" "scikit-learn==1.8.0"
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0"
export PYTHONPATH="$WORK_DIR/src"
cargo build --release

STATUS_DIR="/tmp/ofc-hu-t0-counterfactual-status"
RESULT_DIR="/tmp/ofc-hu-t0-counterfactual-results"
mkdir -p "$STATUS_DIR" "$RESULT_DIR"
for (( shard=START_SHARD; shard<TOTAL_SHARDS; shard+=VM_COUNT )); do
  spec="$(sed -n "$((shard + 1))p" shards_manifest.jsonl)"
  config_id="$(json_field "$spec" config_id)"
  margin_first="$(json_field "$spec" min_margin_first)"
  margin_second="$(json_field "$spec" min_margin_second)"
  allowed_seats="$(json_field "$spec" allowed_seats)"
  games="$(json_field "$spec" games)"
  seed="$(json_field "$spec" seed)"
  output_prefix="$(json_field "$spec" output_prefix)"
  remote_marker="gs://${BUCKET}/runs/${RUN_NAME}/results/${output_prefix}/DONE"
  if gcloud storage ls "$remote_marker" >/dev/null 2>&1; then
    log "skip existing shard=$shard"
    continue
  fi
  local_out="${RESULT_DIR}/${output_prefix}"
  rm -rf "$local_out"
  mkdir -p "$local_out"
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_seconds="$(date +%s)"
  log "run shard=$shard config=$config_id games=$games seed=$seed"
  set +e
  cmd=(python -B -m ofc_regular.evaluate_hu_turn0_counterfactual \
    --config-id "$config_id" \
    --candidate-model "$CANDIDATE_MODEL" \
    --candidate-topk "$CANDIDATE_TOPK" \
    --min-margin-first "$margin_first" \
    --min-margin-second "$margin_second" \
    --allowed-seats "$allowed_seats" \
    --profile stage18_p1 \
    --games "$games" \
    --seed "$seed" \
    --seed-stride "$SEED_STRIDE" \
    --opening-lookahead-samples "$OPENING_LOOKAHEAD_SAMPLES" \
    --progress-every 0 \
    --events-output "$local_out/events.jsonl" \
    --output "$local_out/summary.json")
  if [[ -n "$SAFE_SELECTOR_MODEL" ]]; then
    cmd+=(
      --safe-selector-model "$SAFE_SELECTOR_MODEL"
      --safe-selector-threshold-first "$SAFE_SELECTOR_THRESHOLD_FIRST"
      --safe-selector-threshold-second "$SAFE_SELECTOR_THRESHOLD_SECOND"
    )
  fi
  "${cmd[@]}" > "$local_out/run.log" 2>&1
  exit_code=$?
  set -e
  elapsed=$(( $(date +%s) - start_seconds ))
  status_path="${STATUS_DIR}/${output_prefix}.json"
  python3 - "$status_path" "$RUN_NAME" "$shard" "$config_id" "$games" "$seed" "$exit_code" "$elapsed" "$started_at" "$INSTANCE_NAME" "$ZONE" <<'PY'
import datetime,json,sys
path,run,shard,config,games,seed,code,elapsed,started,instance,zone=sys.argv[1:]
payload={"run_name":run,"shard":int(shard),"config_id":config,"games":int(games),"seed":int(seed),"exit_code":int(code),"elapsed_seconds":int(elapsed),"started_at":started,"finished_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"instance":instance,"zone":zone,"status":"complete" if int(code)==0 else "failed"}
open(path,"w",encoding="utf-8").write(json.dumps(payload,separators=(",",":"))+"\n")
PY
  if [[ "$exit_code" -ne 0 ]]; then
    gcloud storage cp "$local_out/run.log" "gs://${BUCKET}/runs/${RUN_NAME}/failures/${output_prefix}/run.log" || true
    gcloud storage cp "$status_path" "gs://${BUCKET}/runs/${RUN_NAME}/status/${output_prefix}.json" || true
    exit "$exit_code"
  fi
  touch "$local_out/DONE"
  gcloud storage cp --recursive "$local_out" "gs://${BUCKET}/runs/${RUN_NAME}/results/"
  gcloud storage cp "$status_path" "gs://${BUCKET}/runs/${RUN_NAME}/status/${output_prefix}.json"
done
if [[ "$SELF_DELETE" == "1" ]]; then
  trap - EXIT
  gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" --quiet || sudo shutdown -h now
else
  sudo shutdown -h now
fi
'@
Write-Utf8NoBom -Path $startupPath -Text $startup

$manifest = [ordered]@{
    schema = "hu_turn0_counterfactual_gcp_manifest_v1"
    project_id = $ProjectId
    bucket = $Bucket
    run_name = $RunName
    candidate_model = ($CandidateModel -replace '\\', '/')
    candidate_topk = $CandidateTopk
    safe_selector_model = $(if ($SafeSelectorModel) { ($SafeSelectorModel -replace '\\', '/') } else { $null })
    safe_selector_threshold_first = $SafeSelectorThresholdFirst
    safe_selector_threshold_second = $SafeSelectorThresholdSecond
    configs = $parsedConfigs
    paired_seeds_per_config = $PairedSeedsPerConfig
    games_per_shard = $GamesPerShard
    base_seed = $BaseSeed
    seed_stride = $SeedStride
    total_shards = $totalShards
    vm_count = $effectiveVmCount
    machine_type = $MachineType
    cpu_threads = $CpuThreads
    opening_lookahead_samples = $OpeningLookaheadSamples
    spot = -not [bool]$NoSpot
    source_uri = $sourceUri
    startup_uri = $startupUri
    created_at = (Get-Date).ToUniversalTime().ToString('o')
}
Write-Utf8NoBom -Path $manifestPath -Text (($manifest | ConvertTo-Json -Depth 8) + "`n")

if ($DryRun) {
    [pscustomobject]@{execution="dry_run"; manifest=$manifest; shards=$totalShards} | ConvertTo-Json -Depth 10
    exit 0
}
gcloud storage cp $packagePath $sourceUri --project $ProjectId | Out-Null
gcloud storage cp $startupPath $startupUri --project $ProjectId | Out-Null
gcloud storage cp $manifestPath $manifestUri --project $ProjectId | Out-Null
gcloud storage cp $shardManifestPath "gs://$Bucket/runs/$RunName/source/shards_manifest.jsonl" --project $ProjectId | Out-Null

$vmPrefix = Convert-ToVmName $RunName
$instances = @()
foreach ($worker in $workerIndices) {
    $zone = $Zones[$worker % $Zones.Count]
    $vmName = ("{0}-{1:d3}" -f $vmPrefix, $worker)
    if ($vmName.Length -gt 63) { $vmName = $vmName.Substring(0, 63).Trim('-') }
    $instances += [pscustomobject]@{name=$vmName;zone=$zone;start_shard=$worker}
    if (-not $CreateInstances) { continue }
    if ($SkipExistingInstances) {
        $existing = gcloud compute instances list --project $ProjectId --filter ("name='"+$vmName+"'") --format "value(name)" 2>$null
        if ($existing) { continue }
    }
    $metadata = @(
        "RUN_NAME=$RunName", "BUCKET=$Bucket", "TOTAL_SHARDS=$totalShards",
        "VM_COUNT=$effectiveVmCount", "START_SHARD=$worker", "SOURCE_URI=$sourceUri",
        "CANDIDATE_MODEL=$(($CandidateModel -replace '\\','/'))", "CANDIDATE_TOPK=$CandidateTopk",
        "SAFE_SELECTOR_MODEL=$(($SafeSelectorModel -replace '\\','/'))",
        "SAFE_SELECTOR_THRESHOLD_FIRST=$SafeSelectorThresholdFirst",
        "SAFE_SELECTOR_THRESHOLD_SECOND=$SafeSelectorThresholdSecond",
        "SEED_STRIDE=$SeedStride", "CPU_THREADS=$CpuThreads",
        "OPENING_LOOKAHEAD_SAMPLES=$OpeningLookaheadSamples",
        ("SELF_DELETE=" + $(if ($NoSelfDelete) { '0' } else { '1' }))
    ) -join ','
    $createArgs = @(
        "compute","instances","create",$vmName,"--project",$ProjectId,"--zone",$zone,
        "--machine-type",$MachineType,"--image-family","ubuntu-2404-lts-amd64",
        "--image-project","ubuntu-os-cloud","--boot-disk-size",("{0}GB" -f $BootDiskGb),
        "--scopes","https://www.googleapis.com/auth/cloud-platform","--metadata",$metadata,
        "--metadata-from-file","startup-script=$startupPath"
    )
    if (-not $NoSpot) {
        $createArgs += @("--provisioning-model","SPOT","--instance-termination-action","DELETE","--maintenance-policy","TERMINATE")
    }
    & gcloud @createArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to create $vmName in $zone" }
}

[pscustomobject]@{
    schema = "hu_turn0_counterfactual_gcp_launch_v1"
    run_name = $RunName
    candidate_model = $CandidateModel
    safe_selector_model = $SafeSelectorModel
    safe_selector_threshold_first = $SafeSelectorThresholdFirst
    safe_selector_threshold_second = $SafeSelectorThresholdSecond
    configs = $parsedConfigs
    paired_seeds_per_config = $PairedSeedsPerConfig
    total_shards = $totalShards
    vm_count = $effectiveVmCount
    create_instances = [bool]$CreateInstances
    instances = $instances
    manifest = $manifestPath
} | ConvertTo-Json -Depth 8
