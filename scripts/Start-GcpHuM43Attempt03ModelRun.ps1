param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-m43-attempt03-v5-model-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [string]$InheritedTrain,
    [string]$FreshTrainFit,
    [string]$FitReceiveReceipt,
    [Parameter(Mandatory = $true)][string]$ModelFreeze,
    [Parameter(Mandatory = $true)][string]$TrainingFreeze,
    [int[]]$StartShards = @(0),
    [string]$MachineType = "c4-highcpu-8",
    [string[]]$FallbackMachineTypes = @("c4-standard-8", "n2-highcpu-8"),
    [string[]]$Zones = @("asia-northeast1-b", "asia-northeast1-c", "us-central1-a"),
    [ValidateRange(20, 100)][int]$BootDiskGb = 30,
    [switch]$CreateInstances,
    [switch]$ResumeExisting,
    [switch]$PackageOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "HuM43Attempt03ModelSpot.Common.ps1")

$ExpectedJobs = 30
$ExpectedShards = 8
$JobsPerShard = 4
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = (Get-Command python -ErrorAction Stop | Select-Object -First 1).Source
if (-not $pythonPath) { $pythonPath = "python" }

if ($DryRun -and ($CreateInstances -or $ResumeExisting)) { throw "DryRun cannot create or resume cloud state" }
if ($PackageOnly -and -not $DryRun) { throw "PackageOnly requires DryRun" }
if (-not $PackageOnly -and (-not $InheritedTrain -or -not $FreshTrainFit -or -not $FitReceiveReceipt)) {
    throw "A non-package run requires InheritedTrain, FreshTrainFit, and FitReceiveReceipt"
}
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{2,119}$') { throw "RunName is unsafe" }
if ($Bucket -notmatch '^[a-z0-9][a-z0-9._-]{2,221}[a-z0-9]$') { throw "Bucket is unsafe" }
$selectedShards = @($StartShards | Sort-Object -Unique)
if ($selectedShards.Count -eq 0 -or @($selectedShards | Where-Object { $_ -lt 0 -or $_ -ge $ExpectedShards }).Count -ne 0) {
    throw "StartShards must be unique values in 0..7"
}

$resolvedInherited = if ($PackageOnly) { $null } else { Resolve-M43A3Input $InheritedTrain $repoRoot "inherited Attempt02 train200" }
$resolvedFresh = if ($PackageOnly) { $null } else { Resolve-M43A3Input $FreshTrainFit $repoRoot "fresh Attempt03 train.fit500" }
$resolvedFitReceipt = if ($PackageOnly) { $null } else { Resolve-M43A3Input $FitReceiveReceipt $repoRoot "fit receive receipt" }
$resolvedModelFreeze = Resolve-M43A3Input $ModelFreeze $repoRoot "pre-row model freeze"
$resolvedTrainingFreeze = Resolve-M43A3Input $TrainingFreeze $repoRoot "training pipeline freeze"

$runRoot = Join-Path $repoRoot "outputs/gcp_runs"
$runDir = Join-Path $runRoot $RunName
$artifactDir = if ($DryRun) { Join-Path $runDir "dryrun" } else { $runDir }
$manifestPath = Join-Path $artifactDir "m43_attempt03_v5_model_spot_manifest.json"
$contractPath = Join-Path $artifactDir "m43_attempt03_v5_fold_cloud_contract.json"
$sourcePath = Join-Path $artifactDir "m43_attempt03_v5_model_source.zip"
$startupPath = Join-Path $artifactDir "startup_hu_m43_attempt03_v5_model.sh"
$startPlanPath = Join-Path $artifactDir "start_plan.json"
$gcsPrefix = "gs://$Bucket/runs/$RunName"
$manifestUri = "$gcsPrefix/source/m43_attempt03_v5_model_spot_manifest.json"
$contractUri = "$gcsPrefix/source/m43_attempt03_v5_fold_cloud_contract.json"
$sourceUri = "$gcsPrefix/source/m43_attempt03_v5_model_source.zip"
$startupUri = "$gcsPrefix/source/startup_hu_m43_attempt03_v5_model.sh"

function Invoke-LocalPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments, [ValidateRange(1, 600)][int]$TimeoutSeconds = 120)
    $result = Invoke-M43A3ProcessBounded -FilePath $pythonPath -Arguments $Arguments -TimeoutSeconds $TimeoutSeconds `
        -Environment @{ PYTHONPATH = (Join-Path $repoRoot "src"); PYTHONHASHSEED = "0" }
    if ($result.timed_out -or $result.exit_code -ne 0) {
        throw "Attempt03 local Python failed: $(@($result.output) -join [Environment]::NewLine)"
    }
    return @($result.stdout)
}

if ($PackageOnly) {
    if (Test-Path -LiteralPath $artifactDir) { throw "Attempt03 package-only RunName is immutable and already exists: $artifactDir" }
    New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
    $packageDir = Join-Path $artifactDir "package"
    Invoke-LocalPython @(
        "-B", "-m", "ofc_regular.build_hu_m43_attempt03_model_spot_package",
        "--repo-root", $repoRoot,
        "--training-freeze", $resolvedTrainingFreeze,
        "--output-dir", $packageDir
    ) 180 | Out-Null
    $packageManifestPath = Join-Path $packageDir "package_manifest.json"
    $packageManifest = Get-Content -LiteralPath $packageManifestPath -Raw | ConvertFrom-Json
    if ($packageManifest.schema -ne "hu_m43_attempt03_v5_model_spot_package_only_v1" -or
        $packageManifest.status -ne "package_only_dry_run_no_fit_inputs" -or
        $packageManifest.fit_input_count -ne 0 -or $packageManifest.holdout_input_count -ne 0 -or
        $packageManifest.cloud_upload_performed -ne $false -or
        $packageManifest.instance_launch_performed -ne $false -or
        [string]$packageManifest.training_freeze.file_sha256 -ne (Get-M43A3Sha256 $resolvedTrainingFreeze)) {
        throw "Attempt03 package-only manifest changed"
    }
    $packagePlan = [ordered]@{
        schema = "hu_m43_attempt03_v5_model_spot_package_only_plan_v1"
        status = "pass_no_fit_data_no_cloud"
        run_name = $RunName
        package_manifest = $packageManifestPath
        package_manifest_file_sha256 = Get-M43A3Sha256 $packageManifestPath
        source_archive_file_sha256 = [string]$packageManifest.source_archive.file_sha256
        training_freeze_file_sha256 = Get-M43A3Sha256 $resolvedTrainingFreeze
        model_freeze_file_sha256 = Get-M43A3Sha256 $resolvedModelFreeze
        fit_input_count = 0
        holdout_input_count = 0
        cloud_upload_performed = $false
        instance_launch_performed = $false
        current_profile_mutated = $false
        runtime_policy_activated = $false
        full_replacement = $false
    }
    Write-M43A3Utf8CreateNew $startPlanPath (($packagePlan | ConvertTo-Json -Depth 12) + "`n")
    $packagePlan | ConvertTo-Json -Depth 12
    exit 0
}

function Convert-ToVmPrefix {
    param([Parameter(Mandatory = $true)][string]$Value)
    $name = ($Value.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-')
    if (-not $name) { throw "RunName does not produce a VM prefix" }
    if ($name -notmatch '^[a-z]') { $name = "m43a3-$name" }
    if ($name.Length -gt 53) {
        $name = $name.Substring(0, 44).TrimEnd('-') + "-" + (Get-M43A3TextSha256 $Value).Substring(0, 8)
    }
    return $name
}

function Get-InstancesByName {
    param([Parameter(Mandatory = $true)][string]$Name)
    $raw = Invoke-M43A3Gcloud @("compute", "instances", "list", "--project", $ProjectId, "--filter", "name=$Name", "--format=json") 30
    if (-not $raw) { return @() }
    return @(((@($raw) -join "`n") | ConvertFrom-Json))
}

function Wait-InstanceCreate {
    param(
        [Parameter(Mandatory = $true)][string]$Operation,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Zone,
        [Parameter(Mandatory = $true)][string]$Machine
    )
    if ($Operation -notmatch '^operation-[A-Za-z0-9-]+$') { throw "Create operation identity is invalid" }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    while ($watch.Elapsed.TotalSeconds -lt 90) {
        $result = Invoke-M43A3GcloudProcess @("compute", "operations", "describe", $Operation, "--project", $ProjectId, "--zone", $Zone, "--format=json", "--quiet") 10
        if ($result.timed_out) { continue }
        if ($result.exit_code -ne 0) { throw "Unable to poll create operation: $(@($result.output) -join ' ')" }
        $payload = ((@($result.stdout) -join "`n") | ConvertFrom-Json)
        if ([string]$payload.name -ne $Operation) { throw "Create operation identity changed" }
        if ([string]$payload.status -eq "DONE") {
            $errorProperty = $payload.PSObject.Properties["error"]
            $errors = if ($null -ne $errorProperty -and $null -ne $errorProperty.Value) {
                @($errorProperty.Value.errors | Where-Object { $null -ne $_ })
            }
            else { @() }
            if ($errors.Count -gt 0) { return [pscustomobject]@{ created = $false; terminal = $true; errors = $errors } }
            $instances = @(Get-InstancesByName $Name)
            if ($instances.Count -ne 1) { throw "Create succeeded without exactly one observable instance" }
            $instance = $instances[0]
            if (([string]$instance.zone).Split('/')[-1] -ne $Zone -or ([string]$instance.machineType).Split('/')[-1] -ne $Machine) {
                throw "Created instance identity changed"
            }
            return [pscustomobject]@{ created = $true; terminal = $false; errors = @() }
        }
        Start-Sleep -Seconds 2
    }
    $observed = @(Get-InstancesByName $Name)
    if ($observed.Count -eq 1 -and ([string]$observed[0].zone).Split('/')[-1] -eq $Zone -and
        ([string]$observed[0].machineType).Split('/')[-1] -eq $Machine) {
        return [pscustomobject]@{ created = $true; terminal = $false; errors = @("bounded_instance_observed") }
    }
    throw "Create outcome remained ambiguous after bounded polling; no fallback was attempted"
}

function Assert-RemoteShardReceipt {
    param([Parameter(Mandatory = $true)]$Receipt, [Parameter(Mandatory = $true)][int]$Shard)
    $expected = @($manifest.jobs | Where-Object { [int]$_.shard_index -eq $Shard } | ForEach-Object { [int]$_.job_index })
    if ($Receipt.schema -ne "hu_m43_attempt03_v5_model_spot_shard_receipt_v1" -or
        $Receipt.status -ne "complete" -or [string]$Receipt.run_name -ne $RunName -or
        (Get-M43A3StrictInteger $Receipt.shard_index "receipt.shard_index") -ne $Shard -or
        [string]$Receipt.run_manifest_sha256 -ne $manifestSha256 -or
        [string]$Receipt.source_sha256 -ne [string]$manifest.source.sha256 -or
        (@($Receipt.jobs) -join ",") -ne ($expected -join ",") -or
        $Receipt.current_profile_mutated -ne $false -or $Receipt.runtime_policy_activated -ne $false) {
        throw "Attempt03 shard receipt is invalid for shard $Shard"
    }
}

$vmPrefix = Convert-ToVmPrefix $RunName
$manifest = $null
$manifestSha256 = $null
$boundaryAudit = $null

if ($ResumeExisting) {
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "ResumeExisting requires the local immutable manifest" }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $manifestSha256 = Get-M43A3Sha256 $manifestPath
    Assert-M43A3RunManifest $manifest $RunName $ProjectId $Bucket
    foreach ($pair in @(@($contractPath, $manifest.cloud_contract.file_sha256), @($sourcePath, $manifest.source.sha256), @($startupPath, $manifest.startup.sha256))) {
        if (-not (Test-Path -LiteralPath $pair[0] -PathType Leaf) -or (Get-M43A3Sha256 $pair[0]) -ne [string]$pair[1]) {
            throw "ResumeExisting local immutable artifact changed: $($pair[0])"
        }
    }
    $remote = Invoke-M43A3GcloudProcess @("storage", "cat", $manifestUri) 30
    if ($remote.timed_out -or $remote.exit_code -ne 0 -or (Get-M43A3TextSha256 ((@($remote.stdout) -join "`n") + "`n")) -ne $manifestSha256) {
        throw "ResumeExisting remote manifest does not match the local immutable manifest"
    }
}
else {
    if (Test-Path -LiteralPath $artifactDir) { throw "Attempt03 model RunName is immutable and already exists: $artifactDir" }
    New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
    $temporary = Join-Path ([IO.Path]::GetTempPath()) ("m43-attempt03-model-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $temporary | Out-Null
    try {
        $preparedContract = Join-Path $temporary "fold_cloud_contract.json"
        Invoke-LocalPython @(
            "-B", "-m", "ofc_regular.prepare_hu_m43_attempt03_fold_training",
            "--inherited-train", $resolvedInherited,
            "--fresh-train-fit", $resolvedFresh,
            "--fit-receive-receipt", $resolvedFitReceipt,
            "--model-freeze", $resolvedModelFreeze,
            "--training-freeze", $resolvedTrainingFreeze,
            "--repo-root", $repoRoot,
            "--output", $preparedContract
        ) 180 | Out-Null
        $contract = Get-Content -LiteralPath $preparedContract -Raw | ConvertFrom-Json
        if ($contract.schema -ne "hu_m43_attempt03_v5_fold_cloud_contract_v1" -or
            $contract.status -ne "frozen_fit700_only" -or
            $contract.worker_input_boundary.fit_rows -ne 700 -or $contract.worker_input_boundary.holdout_inputs -ne 0 -or
            @($contract.fold_plan.jobs).Count -ne 30 -or
            [string]$contract.model_freeze.file_sha256 -ne (Get-M43A3Sha256 $resolvedModelFreeze) -or
            [string]$contract.training_freeze.file_sha256 -ne (Get-M43A3Sha256 $resolvedTrainingFreeze)) {
            throw "Attempt03 fit700 cloud contract projection changed"
        }

        $packageRoot = Join-Path $temporary "package"
        $closureBuilder = @'
import ast,hashlib,json,pathlib,shutil,sys
repo=pathlib.Path(sys.argv[1]).resolve(); destination=pathlib.Path(sys.argv[2]).resolve(); contract_path=pathlib.Path(sys.argv[3]).resolve(); source=repo/"src"/"ofc_regular"
queue=["ofc_regular","ofc_regular.train_hu_m43_attempt03_fold_job"]; seen=set(); selected=[]
def module_path(name):
 parts=name.split(".")
 if not parts or parts[0]!="ofc_regular": return None
 candidate=source.joinpath(*parts[1:]); path=candidate/"__init__.py" if candidate.is_dir() else candidate.with_suffix(".py")
 return path if path.is_file() else None
def add(name):
 if name.startswith("ofc_regular") and name not in seen and module_path(name) is not None: queue.append(name)
while queue:
 name=queue.pop(0)
 if name in seen: continue
 path=module_path(name)
 if path is None: raise ValueError(f"missing local module: {name}")
 seen.add(name); selected.append((name,path)); tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
 package=name if path.name=="__init__.py" else name.rsplit(".",1)[0]
 for node in ast.walk(tree):
  if isinstance(node,ast.Import):
   for alias in node.names: add(alias.name)
  elif isinstance(node,ast.ImportFrom):
   if node.level:
    base=package.split(".")
    if node.level>1: base=base[:-(node.level-1)]
    target=".".join(base+(([node.module] if node.module else []))); add(target)
    if node.module is None:
     for alias in node.names: add(target+"."+alias.name)
   elif node.module: add(node.module)
entries=[]
for name,path in selected:
 relative=path.relative_to(repo/"src"); token=relative.as_posix().lower()
 if "locked" in token or token.endswith((".json",".jsonl")) or any(part in {"output","outputs","config","configs","data"} for part in relative.parts):
  raise ValueError(f"forbidden dependency entry name: {relative}")
 target=destination/"src"/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target); entries.append("src/"+relative.as_posix())
entries=sorted(entries); digest=hashlib.sha256(json.dumps(entries,separators=(",",":")).encode()).hexdigest()
contract=json.load(open(contract_path,encoding="utf-8-sig")); worker_sources=contract["training_freeze"]["worker_sources"]
v5_path="src/ofc_regular/hu_m43_joint_model_v5.py"; v5_expected=contract["model_freeze"]["v5_source_sha256"]
if worker_sources.get(v5_path)!=v5_expected: raise ValueError("contract v5 source projections disagree")
verified={}
for relative,expected in sorted(worker_sources.items()):
 if not isinstance(expected,str) or len(expected)!=64 or any(c not in "0123456789abcdef" for c in expected): raise ValueError(f"invalid worker source SHA: {relative}")
 packaged=destination/relative
 if not packaged.is_file(): raise ValueError(f"packaged worker source missing: {relative}")
 actual=hashlib.sha256(packaged.read_bytes()).hexdigest()
 if actual!=expected: raise ValueError(f"packaged worker source SHA changed: {relative}: {actual} != {expected}")
 verified[relative]=actual
print(json.dumps({"entries":entries,"entries_sha256":digest,"post_copy_source_audit":"pass","verified_worker_sources":verified,"v5_implementation_sha256":verified[v5_path]},sort_keys=True))
'@
        # Use a bounded temporary script; the script contains code only and no input row values.
        $closureScript = Join-Path $temporary "build_closure.py"
        [IO.File]::WriteAllText($closureScript, $closureBuilder, [Text.UTF8Encoding]::new($false))
        $closureOutput = Invoke-LocalPython @($closureScript, $repoRoot, $packageRoot, $preparedContract) 120
        $closure = ((@($closureOutput) -join "`n") | ConvertFrom-Json)
        if ("src/ofc_regular/train_hu_m43_attempt03_fold_job.py" -notin @($closure.entries) -or
            $closure.post_copy_source_audit -ne "pass" -or
            [string]$closure.v5_implementation_sha256 -ne [string]$contract.model_freeze.v5_source_sha256) {
            throw "Attempt03 source closure/post-copy audit failed"
        }

        $zipBuilder = Join-Path $temporary "build_zip.py"
        [IO.File]::WriteAllText($zipBuilder, @'
import pathlib,sys,zipfile
source=pathlib.Path(sys.argv[1]); destination=pathlib.Path(sys.argv[2])
with zipfile.ZipFile(destination,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
 for path in sorted(source.rglob("*.py")):
  name=path.relative_to(source).as_posix()
  if "\\" in name: raise ValueError("non-portable archive separator")
  archive.write(path,name)
'@, [Text.UTF8Encoding]::new($false))
        $tempSource = Join-Path $temporary "source.zip"
        Invoke-LocalPython @($zipBuilder, $packageRoot, $tempSource) 120 | Out-Null

        $startup = @'
#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=/var/log/hu_m43_attempt03_v5_model.log
exec > >(tee -a "$LOG") 2>&1
META=http://metadata.google.internal/computeMetadata/v1
meta(){ curl --connect-timeout 5 --max-time 20 -fsS -H 'Metadata-Flavor: Google' "$META/instance/attributes/$1"; }
instance_meta(){ curl --connect-timeout 5 --max-time 20 -fsS -H 'Metadata-Flavor: Google' "$META/instance/$1"; }
project_meta(){ curl --connect-timeout 5 --max-time 20 -fsS -H 'Metadata-Flavor: Google' "$META/project/project-id"; }
token(){ curl --connect-timeout 5 --max-time 20 -fsS -H 'Metadata-Flavor: Google' "$META/instance/service-accounts/default/token" | python3 -c 'import json,sys;print(json.load(sys.stdin)["access_token"])'; }
urlencode(){ python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "$1"; }
PROJECT_ID="$(project_meta)"; BUCKET="$(meta BUCKET)"; RUN_NAME="$(meta RUN_NAME)"; SHARD_INDEX="$(meta SHARD_INDEX)"
MANIFEST_OBJECT="$(meta MANIFEST_OBJECT)"; MANIFEST_SHA256="$(meta MANIFEST_SHA256)"; JOB_INDICES="$(meta JOB_INDICES)"; SHARD_EXPECTED_JOBS="$(meta SHARD_EXPECTED_JOBS)"
INSTANCE_NAME="$(instance_meta name)"; ZONE="$(instance_meta zone)"; ZONE="${ZONE##*/}"; PREFIX="runs/${RUN_NAME}"; WORK=/work/m43_attempt03_v5
mkdir -p "$WORK"
gcs_download(){ local o="$1" d="$2" e a; e="$(urlencode "$o")"; a="$(token)"; curl --connect-timeout 10 --max-time 300 --retry 5 --retry-all-errors -fsSL -H "Authorization: Bearer ${a}" "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/${e}?alt=media" -o "$d"; }
gcs_upload(){ local s="$1" o="$2" e a; e="$(urlencode "$o")"; a="$(token)"; curl --connect-timeout 10 --max-time 300 --retry 5 --retry-all-errors -fsS -X POST -H "Authorization: Bearer ${a}" -H 'Content-Type: application/octet-stream' --data-binary "@${s}" "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&name=${e}" >/dev/null; }
gcs_upload_immutable(){ local s="$1" o="$2" e a; e="$(urlencode "$o")"; a="$(token)"; curl --connect-timeout 10 --max-time 300 --retry 5 --retry-all-errors -fsS -X POST -H "Authorization: Bearer ${a}" -H 'Content-Type: application/octet-stream' --data-binary "@${s}" "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&name=${e}&ifGenerationMatch=0" >/dev/null; }
self_delete(){ local a; a="$(token)" || return 0; curl --connect-timeout 5 --max-time 30 -fsS -X DELETE -H "Authorization: Bearer ${a}" "https://compute.googleapis.com/compute/v1/projects/${PROJECT_ID}/zones/${ZONE}/instances/${INSTANCE_NAME}" >/dev/null || true; }
trap self_delete EXIT
timeout 600 apt-get update -y
timeout 600 apt-get install -y unzip curl ca-certificates python3 python3-venv python3-pip
gcs_download "$MANIFEST_OBJECT" "$WORK/manifest.json"; echo "${MANIFEST_SHA256}  $WORK/manifest.json" | sha256sum -c -
eval "$(python3 - "$WORK/manifest.json" <<'PY'
import json,shlex,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
assert m["schema"]=="hu_m43_attempt03_v5_model_spot_run_manifest_v1" and m["status"]=="frozen" and m["job_count"]==30
assert m["cloud_input_boundary"]=="fit700_only_no_holdouts" and [x["role"] for x in m["inputs"]["fit"]]==["inherited_attempt02_train","fresh_train_fit"]
assert [x["rows"] for x in m["inputs"]["fit"]]==[200,500] and m["current_profile_mutated"] is False and m["runtime_policy_activated"] is False
for key,value in {"SOURCE_OBJECT":m["source"]["uri"].split("/",3)[3],"SOURCE_SHA256":m["source"]["sha256"],"CONTRACT_OBJECT":m["cloud_contract"]["uri"].split("/",3)[3],"CONTRACT_SHA256":m["cloud_contract"]["file_sha256"],"INPUT_BUNDLE_SHA256":m["inputs"]["input_bundle_sha256"]}.items(): print(f"{key}={shlex.quote(str(value))}")
PY
)"
gcs_download "$SOURCE_OBJECT" "$WORK/source.zip"; echo "${SOURCE_SHA256}  $WORK/source.zip" | sha256sum -c -
gcs_download "$CONTRACT_OBJECT" "$WORK/contract.json"; echo "${CONTRACT_SHA256}  $WORK/contract.json" | sha256sum -c -
mkdir -p "$WORK/repo" "$WORK/inputs"; unzip -q "$WORK/source.zip" -d "$WORK/repo"
python3 - "$WORK/manifest.json" "$WORK/input_downloads.tsv" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
with open(sys.argv[2],"w",encoding="utf-8") as f:
 for e in m["inputs"]["fit"]: f.write(f'{e["uri"].split("/",3)[3]}\t{e["role"]}.jsonl\t{e["sha256"]}\n')
PY
while IFS=$'\t' read -r object relative sha; do gcs_download "$object" "$WORK/inputs/$relative"; echo "$sha  $WORK/inputs/$relative" | sha256sum -c -; done < "$WORK/input_downloads.tsv"
cd "$WORK/repo"; python3 -m venv .venv; source .venv/bin/activate
timeout 600 python -m pip install --upgrade pip
timeout 600 python -m pip install 'numpy==2.2.6' 'scikit-learn==1.8.0' 'scipy==1.16.3' 'lightgbm==4.6.0'
run_job(){
 local job="$1" out="$WORK/job-$(printf '%02d' "$job")" prefix="$PREFIX/results/job-$(printf '%02d' "$job")" heartbeat_pid=""
 mkdir -p "$out"
 write_status(){ python3 - "$out/status.json" "$RUN_NAME" "$job" "$1" "$MANIFEST_SHA256" "$SOURCE_SHA256" <<'PY'
import datetime,json,sys
p,run,job,state,manifest,source=sys.argv[1:]
json.dump({"schema":"hu_m43_attempt03_v5_model_spot_job_status_v1","run_name":run,"job_index":int(job),"state":state,"run_manifest_sha256":manifest,"source_sha256":source,"current_profile_mutated":False,"runtime_policy_activated":False,"updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat()},open(p,"w",encoding="utf-8"),sort_keys=True)
PY
  gcs_upload "$out/status.json" "$prefix/status.json"; }
 write_status running
 (while true; do sleep 60; python3 - "$out/heartbeat.json" "$RUN_NAME" "$job" "$MANIFEST_SHA256" <<'PY'
import datetime,json,sys
p,run,job,manifest=sys.argv[1:]
json.dump({"schema":"hu_m43_attempt03_v5_model_spot_job_heartbeat_v1","run_name":run,"job_index":int(job),"state":"running","run_manifest_sha256":manifest,"updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat()},open(p,"w",encoding="utf-8"),sort_keys=True)
PY
  gcs_upload "$out/heartbeat.json" "$prefix/heartbeat.json" || true; done) & heartbeat_pid=$!
 eval "$(python3 - "$WORK/manifest.json" "$job" <<'PY'
import json,shlex,sys
m=json.load(open(sys.argv[1],encoding="utf-8")); j=m["jobs"][int(sys.argv[2])]
print("JOB_SPEC_SHA256="+shlex.quote(j["job_spec_sha256"]))
PY
)"
 set +e
 PYTHONPATH=src python -B -m ofc_regular.train_hu_m43_attempt03_fold_job --job-index "$job" --inherited-train "$WORK/inputs/inherited_attempt02_train.jsonl" --fresh-train-fit "$WORK/inputs/fresh_train_fit.jsonl" --fold-cloud-contract "$WORK/contract.json" --output-dir "$out" --run-name "$RUN_NAME" --source-sha256 "$SOURCE_SHA256" --run-manifest-sha256 "$MANIFEST_SHA256" --input-bundle-sha256 "$INPUT_BUNDLE_SHA256" --job-spec-sha256 "$JOB_SPEC_SHA256" >"$out/stdout.log" 2>"$out/stderr.log"
 local code=$?; set -e; kill "$heartbeat_pid" >/dev/null 2>&1 || true; wait "$heartbeat_pid" >/dev/null 2>&1 || true
 if [[ $code -ne 0 ]]; then write_status failed || true; gcs_upload "$out/stdout.log" "$prefix/stdout.log" || true; gcs_upload "$out/stderr.log" "$prefix/stderr.log" || true; return "$code"; fi
 python3 - "$WORK/manifest.json" "$job" "$out" "$MANIFEST_SHA256" <<'PY'
import hashlib,json,os,sys
mp,raw,out,msha=sys.argv[1:]; i=int(raw); m=json.load(open(mp,encoding="utf-8")); expected=m["jobs"][i]
h=lambda p:hashlib.sha256(open(p,"rb").read()).hexdigest(); jm=json.load(open(os.path.join(out,"job_manifest.json"),encoding="utf-8")); done=json.load(open(os.path.join(out,"DONE.json"),encoding="utf-8"))
assert jm["schema"]=="hu_m43_attempt03_v5_fold_job_manifest_v1" and jm["status"]=="pass"
assert done["schema"]=="hu_m43_attempt03_v5_fold_done_v1" and done["status"]=="complete"
for obj in (jm,done):
 assert obj["run_name"]==m["run_name"] and obj["job_index"]==i and obj["job_spec_sha256"]==expected["job_spec_sha256"]
 assert obj["source_sha256"]==m["source"]["sha256"] and obj["run_manifest_sha256"]==msha and obj["cloud_contract_file_sha256"]==m["cloud_contract"]["file_sha256"]
 assert obj["input_bundle_sha256"]==m["inputs"]["input_bundle_sha256"] and obj["training_config_sha256"]==m["training_config_sha256"]
 assert obj["fit700_only"] is True and obj["holdout_input_count"]==0 and obj["current_profile_mutated"] is False and obj["runtime_policy_activated"] is False
assert h(os.path.join(out,"estimator.pkl"))==jm["artifact_sha256"]==done["artifact_sha256"]
assert h(os.path.join(out,"job_manifest.json"))==done["job_manifest_sha256"]
PY
 gcs_upload "$out/estimator.pkl" "$prefix/estimator.pkl"; gcs_upload "$out/job_manifest.json" "$prefix/job_manifest.json"
 gcs_upload "$out/stdout.log" "$prefix/stdout.log"; gcs_upload "$out/stderr.log" "$prefix/stderr.log"
 write_status complete; gcs_upload_immutable "$out/DONE.json" "$prefix/DONE.json"
}
IFS='+' read -r -a JOB_ARRAY <<< "$JOB_INDICES"; pids=(); for job in "${JOB_ARRAY[@]}"; do run_job "$job" & pids+=("$!"); done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; [[ $failed -eq 0 ]] || exit 1
python3 - "$WORK/shard_receipt.json" "$RUN_NAME" "$SHARD_INDEX" "$MANIFEST_SHA256" "$SOURCE_SHA256" "$SHARD_EXPECTED_JOBS" "$JOB_INDICES" <<'PY'
import datetime,json,sys
p,run,shard,manifest,source,expected,executed=sys.argv[1:]
json.dump({"schema":"hu_m43_attempt03_v5_model_spot_shard_receipt_v1","status":"complete","run_name":run,"shard_index":int(shard),"jobs":[int(x) for x in expected.split("+")],"executed_jobs":[int(x) for x in executed.split("+")],"run_manifest_sha256":manifest,"source_sha256":source,"current_profile_mutated":False,"runtime_policy_activated":False,"completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat()},open(p,"w",encoding="utf-8"),sort_keys=True)
PY
gcs_upload_immutable "$WORK/shard_receipt.json" "$PREFIX/results/shard-$(printf '%02d' "$SHARD_INDEX")/receipt.json"
echo "M4.3 Attempt03 v5 model shard complete: $RUN_NAME shard=$SHARD_INDEX jobs=$JOB_INDICES"
'@
        $startup = $startup -replace "`r`n", "`n"

        [IO.File]::WriteAllBytes($sourcePath, [IO.File]::ReadAllBytes($tempSource))
        [IO.File]::WriteAllBytes($contractPath, [IO.File]::ReadAllBytes($preparedContract))
        Write-M43A3Utf8CreateNew $startupPath $startup

        $inputPlan = @(
            [ordered]@{ role = "inherited_attempt02_train"; rows = 200; bytes = [long]$contract.inputs.inherited_attempt02_train.bytes; sha256 = [string]$contract.inputs.inherited_attempt02_train.sha256; uri = "$gcsPrefix/inputs/inherited_attempt02_train.jsonl" },
            [ordered]@{ role = "fresh_train_fit"; rows = 500; bytes = [long]$contract.inputs.fresh_train_fit.bytes; sha256 = [string]$contract.inputs.fresh_train_fit.sha256; uri = "$gcsPrefix/inputs/fresh_train_fit.jsonl" }
        )
        $jobs = @()
        foreach ($entry in @($contract.fold_plan.jobs)) {
            $index = [int]$entry.job_index
            $jobs += [ordered]@{
                job_index = $index; shard_index = [int][Math]::Floor($index / $JobsPerShard)
                job_kind = [string]$entry.kind; outer_fold = [int]$entry.outer_fold; inner_fold = $entry.inner_fold
                job_spec_sha256 = [string]$entry.job_spec_sha256
                result_prefix = "$gcsPrefix/results/job-$($index.ToString('D2'))"
                done_uri = "$gcsPrefix/results/job-$($index.ToString('D2'))/DONE.json"
            }
        }
        $manifest = [ordered]@{
            schema = "hu_m43_attempt03_v5_model_spot_run_manifest_v1"; status = "frozen"; run_name = $RunName
            project_id = $ProjectId; bucket = $Bucket; job_count = $ExpectedJobs
            source = [ordered]@{
                uri = $sourceUri; bytes = [long](Get-Item $sourcePath).Length; sha256 = Get-M43A3Sha256 $sourcePath
                fold_worker_sha256 = Get-M43A3Sha256 (Join-Path $repoRoot "src/ofc_regular/train_hu_m43_attempt03_fold_job.py")
                training_core_sha256 = Get-M43A3Sha256 (Join-Path $repoRoot "src/ofc_regular/hu_m43_attempt03_training.py")
            }
            cloud_contract = [ordered]@{
                uri = $contractUri; bytes = [long](Get-Item $contractPath).Length
                file_sha256 = Get-M43A3Sha256 $contractPath; contract_sha256 = [string]$contract.contract_sha256
            }
            inputs = [ordered]@{ fit = $inputPlan; input_bundle_sha256 = [string]$contract.input_bundle_sha256 }
            jobs = $jobs; training_config_sha256 = [string]$contract.training_config_sha256
            model_freeze_file_sha256 = [string]$contract.model_freeze.file_sha256
            training_freeze_file_sha256 = [string]$contract.training_freeze.file_sha256
            dependencies = $contract.dependencies; process_environment = $contract.process_environment
            source_package_policy = [ordered]@{
                mode = "ast_recursive_local_import_closure_v1"; roots = @("src/ofc_regular/train_hu_m43_attempt03_fold_job.py")
                module_count = @($closure.entries).Count; entries_sha256 = [string]$closure.entries_sha256
                forbidden_entry_patterns = @("(?i)l[o]cked", "(?i)\.jsonl?$", "(?i)(^|/)(outputs?|configs?|data)(/|$)")
                local_input_values_embedded = $false; nonfit_cloud_inputs = 0
            }
            compute = [ordered]@{
                shard_count = 8; jobs_per_shard = 4; max_parallel_processes = 4; canary_shard = 0
                fanout_shards = @(1,2,3,4,5,6,7); canary_required_before_fanout = $true
                machine_type = $MachineType; fallback_machine_types = $FallbackMachineTypes; zones = $Zones
                boot_disk_gb = $BootDiskGb; spot = $true; termination_action = "DELETE"; auto_delete = $true; vm_prefix = $vmPrefix
            }
            startup = [ordered]@{ uri = $startupUri; sha256 = Get-M43A3Sha256 $startupPath }
            cloud_input_boundary = "fit700_only_no_holdouts"
            current_profile_mutated = $false; runtime_policy_activated = $false; full_replacement = $false
            created_at = (Get-Date).ToUniversalTime().ToString("o")
        }
        Write-M43A3Utf8CreateNew $manifestPath (($manifest | ConvertTo-Json -Depth 25) + "`n")
        $manifestSha256 = Get-M43A3Sha256 $manifestPath
        Assert-M43A3RunManifest $manifest $RunName $ProjectId $Bucket

        $boundaryAudit = [ordered]@{
            schema = "hu_m43_attempt03_v5_model_cloud_boundary_audit_v1"; status = "pass"
            fit_inputs = 2; fit_rows = 700; holdout_inputs = 0; source_modules = @($closure.entries).Count
            model_freeze_uploaded = $false; training_freeze_uploaded = $false; fit_receive_receipt_uploaded = $false
            precalibration_uploaded = $false; sealed_calibration_uploaded = $false; inherited_locked_uploaded = $false
            current_profile_mutated = $false; runtime_policy_activated = $false
        }
    }
    finally { Remove-Item -LiteralPath $temporary -Recurse -Force -ErrorAction SilentlyContinue }
}

$plan = [ordered]@{
    schema = "hu_m43_attempt03_v5_model_spot_start_plan_v1"; status = "frozen"
    run_name = $RunName; manifest_uri = $manifestUri; manifest_sha256 = $manifestSha256
    selected_shards = $selectedShards; job_count = 30; shard_count = 8; canary_shard = 0
    canary_required_before_fanout = $true; create_instances = [bool]$CreateInstances
    resume_existing = [bool]$ResumeExisting; dry_run = [bool]$DryRun
    artifact_directory = $artifactDir; cloud_input_boundary = "fit700_only_no_holdouts"
    current_profile_mutated = $false; runtime_policy_activated = $false; full_replacement = $false
}
if ($DryRun) {
    $plan.boundary_audit = $boundaryAudit
    Write-M43A3Utf8CreateNew $startPlanPath (($plan | ConvertTo-Json -Depth 20) + "`n")
    $plan | ConvertTo-Json -Depth 20
    exit 0
}

if (-not $ResumeExisting) {
    if (Test-M43A3GcsObject $manifestUri) { throw "Remote Attempt03 model RunName already exists" }
    $uploads = @(
        @($resolvedInherited, $manifest.inputs.fit[0].uri), @($resolvedFresh, $manifest.inputs.fit[1].uri),
        @($sourcePath, $sourceUri), @($contractPath, $contractUri), @($startupPath, $startupUri), @($manifestPath, $manifestUri)
    )
    foreach ($pair in $uploads) {
        Invoke-M43A3Gcloud @("storage", "cp", $pair[0], $pair[1], "--project", $ProjectId, "--if-generation-match=0") 300 | Out-Null
    }
}
if (-not $CreateInstances) { $plan.uploaded = $true; $plan | ConvertTo-Json -Depth 20; exit 0 }

if (@($selectedShards | Where-Object { $_ -gt 0 }).Count -gt 0) {
    foreach ($job in 0..3) {
        if (-not (Test-M43A3GcsObject $manifest.jobs[$job].done_uri)) { throw "Canary shard 0 must complete before fanout" }
        Assert-M43A3Done (Get-M43A3GcsJson $manifest.jobs[$job].done_uri) $manifest $job $manifestSha256
    }
    $canaryReceiptUri = "$gcsPrefix/results/shard-00/receipt.json"
    if (-not (Test-M43A3GcsObject $canaryReceiptUri)) { throw "Canary shard receipt must complete before fanout" }
    Assert-RemoteShardReceipt (Get-M43A3GcsJson $canaryReceiptUri) 0
}

$createdShards = @(); $attempts = @()
foreach ($shard in $selectedShards) {
    $jobsForShard = @($manifest.jobs | Where-Object { [int]$_.shard_index -eq $shard })
    $missing = @()
    foreach ($job in $jobsForShard) {
        $index = [int]$job.job_index
        if (Test-M43A3GcsObject $job.done_uri) { Assert-M43A3Done (Get-M43A3GcsJson $job.done_uri) $manifest $index $manifestSha256 }
        else { $missing += $index }
    }
    if ($missing.Count -eq 0) { continue }
    $vmName = ("{0}-s{1:D2}" -f $vmPrefix, $shard)
    if (@(Get-InstancesByName $vmName).Count -ne 0) { throw "Attempt03 shard worker already exists: $vmName" }
    $metadata = @(
        "BUCKET=$Bucket", "RUN_NAME=$RunName", "SHARD_INDEX=$shard",
        "MANIFEST_OBJECT=runs/$RunName/source/m43_attempt03_v5_model_spot_manifest.json",
        "MANIFEST_SHA256=$manifestSha256", ("JOB_INDICES=" + ($missing -join "+")),
        ("SHARD_EXPECTED_JOBS=" + (@($jobsForShard | ForEach-Object { [int]$_.job_index }) -join "+"))
    ) -join ","
    $created = $false
    foreach ($machine in @($MachineType) + @($FallbackMachineTypes)) {
        foreach ($zone in $Zones) {
            $diskType = if ($machine -like "c4-*") { "hyperdisk-balanced" } else { "pd-balanced" }
            $arguments = @(
                "compute", "instances", "create", $vmName, "--project", $ProjectId, "--zone", $zone,
                "--machine-type", $machine, "--provisioning-model", "SPOT", "--instance-termination-action", "DELETE",
                "--maintenance-policy", "TERMINATE", "--boot-disk-size", "${BootDiskGb}GB", "--boot-disk-type", $diskType,
                "--image-family", "ubuntu-2404-lts-amd64", "--image-project", "ubuntu-os-cloud", "--scopes", "cloud-platform",
                "--metadata", $metadata, "--metadata-from-file", "startup-script=$startupPath",
                "--async", "--format=value(name)", "--quiet"
            )
            $submission = Invoke-M43A3GcloudProcess $arguments 30
            if ($submission.timed_out) {
                $observed = @(Get-InstancesByName $vmName)
                if ($observed.Count -eq 1) { throw "Create submission timed out after instance observation; reconcile before resume" }
                throw "Create submission timed out ambiguously; no fallback was attempted"
            }
            $output = @($submission.output) -join "`n"
            if ($submission.exit_code -eq 0) {
                $operations = @($submission.stdout | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ -match '^operation-[A-Za-z0-9-]+$' } | Select-Object -Unique)
                if ($operations.Count -ne 1) { throw "Create returned no unique operation identity" }
                $verification = Wait-InstanceCreate $operations[0] $vmName $zone $machine
                $attempts += [ordered]@{ shard_index = $shard; jobs = $missing; machine_type = $machine; zone = $zone; operation = $operations[0]; created = [bool]$verification.created }
                if ($verification.created) { $created = $true; $createdShards += $shard; break }
                continue
            }
            if (@(Get-InstancesByName $vmName).Count -ne 0) { throw "Failed create left an ambiguous instance" }
            if ($output -notmatch '(?i)RESOURCE_EXHAUSTED|ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources|quota|unsupported|not found|\b(?:400|403|404)\b') {
                throw "Create failed ambiguously; no fallback was attempted: $output"
            }
            $attempts += [ordered]@{ shard_index = $shard; jobs = $missing; machine_type = $machine; zone = $zone; created = $false; terminal_failure = $true }
        }
        if ($created) { break }
    }
    if (-not $created) { throw "Unable to create an Attempt03 Spot worker for shard $shard" }
}
$plan.uploaded = $true; $plan.created_shards = $createdShards; $plan.attempts = $attempts
$plan | ConvertTo-Json -Depth 20
