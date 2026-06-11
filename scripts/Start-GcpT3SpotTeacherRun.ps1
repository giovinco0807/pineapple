param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-t3-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [int]$TotalSamples = 100000,
    [int]$ShardSamples = 250,
    [int]$FutureSamples = 256,
    [int]$VmCount = 20,
    [string]$MachineType = "e2-highcpu-8",
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
    [int]$BaseSeed = 2026060300,
    [double]$FlEv = 12.196164,
    [double]$MinScoreGap = 0.25,
    [int]$BootDiskGb = 30,
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

if ($TotalSamples -le 0) {
    throw "TotalSamples must be positive"
}
if ($ShardSamples -le 0) {
    throw "ShardSamples must be positive"
}
if ($FutureSamples -lt 0) {
    throw "FutureSamples must be non-negative"
}
if ($VmCount -le 0) {
    throw "VmCount must be positive"
}
$StartShardIndices = @()
foreach ($startShardValue in $StartShards) {
    foreach ($part in ($startShardValue -split ",")) {
        $trimmed = $part.Trim()
        if (-not $trimmed) {
            continue
        }
        $StartShardIndices += [int]$trimmed
    }
}
$StartShardIndices = @($StartShardIndices | Sort-Object -Unique)
foreach ($startShard in $StartShardIndices) {
    if ($startShard -lt 0 -or $startShard -ge $VmCount) {
        throw "StartShards values must be in [0, VmCount): $startShard"
    }
}
if (-not $Zones -or $Zones.Count -eq 0) {
    throw "At least one zone is required"
}
if ($RunName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') {
    throw "RunName may only contain letters, numbers, dot, underscore, and dash"
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
$totalShards = [int][math]::Ceiling($TotalSamples / $ShardSamples)
$runDir = Join-Path $repoRoot ("outputs/gcp_runs/{0}" -f $RunName)
$packageDir = Join-Path $runDir "package_src"
$packagePath = Join-Path $runDir "ofc_regular_source.zip"
$startupPath = Join-Path $runDir "startup_t3_spot.sh"
$manifestPath = Join-Path $runDir "manifest.json"
$sourceUri = "gs://$Bucket/runs/$RunName/source/ofc_regular_source.zip"
$startupUri = "gs://$Bucket/runs/$RunName/source/startup_t3_spot.sh"
$manifestUri = "gs://$Bucket/runs/$RunName/manifest.json"

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
if (Test-Path $packageDir) {
    Remove-Item -LiteralPath $packageDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $packageDir | Out-Null

foreach ($item in @("Cargo.toml", "Cargo.lock", "rust", "src")) {
    $source = Join-Path $repoRoot $item
    if (-not (Test-Path $source)) {
        throw "Missing package input: $item"
    }
    Copy-Item -LiteralPath $source -Destination $packageDir -Recurse
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
FL_EV="$(meta FL_EV)"
MIN_SCORE_GAP="$(meta MIN_SCORE_GAP)"
VM_COUNT="$(meta VM_COUNT)"
START_SHARD="$(meta START_SHARD)"
SOURCE_URI="$(meta SOURCE_URI)"
SELF_DELETE="$(meta SELF_DELETE)"
INSTANCE_NAME="$(instance_meta name)"
ZONE_PATH="$(instance_meta zone)"
ZONE="${ZONE_PATH##*/}"

export DEBIAN_FRONTEND=noninteractive
log "install dependencies"
sudo apt-get update -y
sudo apt-get install -y unzip build-essential curl ca-certificates

if ! command -v gcloud >/dev/null 2>&1; then
  log "google cloud sdk not found; installing google-cloud-cli"
  sudo apt-get install -y apt-transport-https gnupg
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y google-cloud-cli
fi

if ! command -v cargo >/dev/null 2>&1; then
  log "install rust"
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi

WORK_DIR="/opt/ofc-regular"
sudo rm -rf "$WORK_DIR"
sudo mkdir -p "$WORK_DIR"
sudo chown "$CURRENT_UID:$CURRENT_GID" "$WORK_DIR"
log "download source package $SOURCE_URI"
gcloud storage cp "$SOURCE_URI" /tmp/ofc_regular_source.zip
unzip -q /tmp/ofc_regular_source.zip -d "$WORK_DIR"
cd "$WORK_DIR"

log "build regular_fl_solver"
cargo build --release
EXE="$WORK_DIR/target/release/regular_fl_solver"
if [[ ! -x "$EXE" ]]; then
  echo "solver binary missing: $EXE" >&2
  exit 1
fi

STATUS_DIR="/tmp/ofc-status"
SHARD_DIR="/tmp/ofc-shards"
mkdir -p "$STATUS_DIR" "$SHARD_DIR"

log "start shard loop run=$RUN_NAME start=$START_SHARD stride=$VM_COUNT total_shards=$TOTAL_SHARDS shard_samples=$SHARD_SAMPLES future=$FUTURE_SAMPLES"
for (( shard=START_SHARD; shard<TOTAL_SHARDS; shard+=VM_COUNT )); do
  shard_name="$(printf "t3_%06d.jsonl" "$shard")"
  remote="gs://${BUCKET}/runs/${RUN_NAME}/shards/${shard_name}"
  status_remote="gs://${BUCKET}/runs/${RUN_NAME}/status/${shard_name%.jsonl}.json"
  if gcloud storage ls "$remote" >/dev/null 2>&1; then
    log "skip existing shard $shard"
    continue
  fi

  local_path="${SHARD_DIR}/${shard_name}"
  status_path="${STATUS_DIR}/${shard_name%.jsonl}.json"
  rm -f "$local_path" "$status_path"
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_seconds="$(date +%s)"
  seed=$((BASE_SEED + shard))
  log "generate shard=$shard seed=$seed"

  set +e
  "$EXE" \
    --teacher-output "$local_path" \
    --teacher-samples "$SHARD_SAMPLES" \
    --future-samples "$FUTURE_SAMPLES" \
    --seed "$seed" \
    --teacher-fl-ev "$FL_EV" \
    --teacher-min-score-gap "$MIN_SCORE_GAP"
  exit_code=$?
  set -e

  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end_seconds="$(date +%s)"
  elapsed=$((end_seconds - start_seconds))
  lines=0
  if [[ -f "$local_path" ]]; then
    lines="$(wc -l < "$local_path" | tr -d ' ')"
  fi

  if [[ "$exit_code" -eq 0 && "$lines" -eq "$SHARD_SAMPLES" ]]; then
    tmp_remote="${remote}.tmp-${INSTANCE_NAME}"
    log "upload shard=$shard lines=$lines elapsed=${elapsed}s"
    gcloud storage cp "$local_path" "$tmp_remote"
    gcloud storage mv "$tmp_remote" "$remote"
    cat > "$status_path" <<EOF
{"run_name":"$RUN_NAME","shard":$shard,"status":"complete","instance":"$INSTANCE_NAME","zone":"$ZONE","seed":$seed,"samples":$SHARD_SAMPLES,"future_samples":$FUTURE_SAMPLES,"lines":$lines,"elapsed_seconds":$elapsed,"started_at":"$started_at","finished_at":"$finished_at","output":"$remote"}
EOF
    gcloud storage cp "$status_path" "$status_remote" || true
  else
    log "failed shard=$shard exit=$exit_code lines=$lines expected=$SHARD_SAMPLES"
    cat > "$status_path" <<EOF
{"run_name":"$RUN_NAME","shard":$shard,"status":"failed","instance":"$INSTANCE_NAME","zone":"$ZONE","seed":$seed,"samples":$SHARD_SAMPLES,"future_samples":$FUTURE_SAMPLES,"lines":$lines,"exit_code":$exit_code,"elapsed_seconds":$elapsed,"started_at":"$started_at","finished_at":"$finished_at"}
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
    total_samples = $TotalSamples
    shard_samples = $ShardSamples
    total_shards = $totalShards
    future_samples = $FutureSamples
    vm_count = $VmCount
    machine_type = $MachineType
    zones = $Zones
    base_seed = $BaseSeed
    fl_ev = $FlEv
    min_score_gap = $MinScoreGap
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
$workerIndices = if ($StartShardIndices.Count -gt 0) {
    $StartShardIndices
}
else {
    @(0..($VmCount - 1))
}
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
        "FL_EV=$FlEv",
        "MIN_SCORE_GAP=$MinScoreGap",
        "VM_COUNT=$VmCount",
        "START_SHARD=$i",
        "SOURCE_URI=$sourceUri",
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
            --image-family ubuntu-2204-lts `
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
    machine_type = $MachineType
    package = $packagePath
    source_uri = $sourceUri
    manifest_uri = $manifestUri
    create_instances = [bool]$CreateInstances
    instances = $instances
} | ConvertTo-Json -Depth 5
