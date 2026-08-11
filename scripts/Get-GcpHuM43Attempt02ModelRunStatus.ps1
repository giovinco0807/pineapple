param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training"
)

$ErrorActionPreference = "Stop"
$ExpectedJobs = 30
$JobsPerShard = 4

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-TextSha256 {
    param([Parameter(Mandatory = $true)][string]$Text)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash([Text.UTF8Encoding]::new($false).GetBytes($Text)))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Invoke-Gcloud {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $output = & gcloud @Arguments 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0) { throw "gcloud failed ($code): gcloud $($Arguments -join ' ')`n$(@($output) -join [Environment]::NewLine)" }
    return @($output)
}

function Sync-GcsResultsSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    if($Uri -ne "gs://$Bucket/runs/$RunName/results") { throw "Attempt02 snapshot URI changed" }
    $old=$ErrorActionPreference;$ErrorActionPreference="Continue"
    try{$tokenOutput=@(& gcloud auth print-access-token --quiet 2>&1);$tokenCode=$LASTEXITCODE}
    finally{$ErrorActionPreference=$old}
    if($tokenCode -ne 0){throw "Unable to acquire a bounded GCS snapshot token: $($tokenOutput-join[Environment]::NewLine)"}
    $tokenCandidates=@($tokenOutput|ForEach-Object{([string]$_).Trim()}|Where-Object{$_ -match '^[A-Za-z0-9._-]{20,}$'})
    if($tokenCandidates.Count -ne 1){throw "GCS access-token output was ambiguous"}
    $snapshotFetcher=@'
import concurrent.futures, json, os, pathlib, sys, time, urllib.error, urllib.parse, urllib.request

bucket,run,destination=sys.argv[1:]
if not bucket or not run or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in run):
    raise ValueError("unsafe Attempt02 snapshot identity")
token=os.environ["M43_ATTEMPT02_GCS_TOKEN"]
root=pathlib.Path(destination).resolve(); root.mkdir(parents=True,exist_ok=True)
prefix=f"runs/{run}/results/"; started=time.monotonic(); deadline=started+55.0
allowed={"DONE.json","estimator.pkl","job_manifest.json","status.json","heartbeat.json","checkpoint.json"}

def remaining():
    value=deadline-time.monotonic()
    if value<=0: raise TimeoutError("Attempt02 GCS snapshot exceeded 55 seconds")
    return value

def request_json(url):
    request=urllib.request.Request(url,headers={"Authorization":f"Bearer {token}"})
    with urllib.request.urlopen(request,timeout=min(10.0,remaining())) as response:
        return json.load(response)

def list_objects():
    result=[]; page=None
    while True:
        query={"prefix":prefix,"fields":"items(name,generation,size),nextPageToken","maxResults":"1000"}
        if page: query["pageToken"]=page
        url=f"https://storage.googleapis.com/storage/v1/b/{urllib.parse.quote(bucket,safe='')}/o?{urllib.parse.urlencode(query)}"
        payload=request_json(url); result.extend(payload.get("items",[])); page=payload.get("nextPageToken")
        if not page: return result

def binding(item):
    name=str(item["name"])
    if not name.startswith(prefix): raise ValueError("object escaped Attempt02 prefix")
    relative=name[len(prefix):]; parts=relative.split("/")
    if len(parts)!=2 or parts[1] not in allowed:
        return None
    if not parts[0].startswith("job-") or len(parts[0])!=6 or not parts[0][4:].isdigit():
        raise ValueError("selected object has invalid Attempt02 job path")
    job=int(parts[0][4:])
    if job<0 or job>=30: raise ValueError("object has out-of-range Attempt02 job")
    generation=str(item.get("generation",""))
    if not generation.isdigit(): raise ValueError("object generation is missing")
    return name,generation,root/parts[0]/parts[1]

def download(entry):
    name,generation,target=entry; target.parent.mkdir(parents=True,exist_ok=True)
    encoded=urllib.parse.quote(name,safe="")
    url=f"https://storage.googleapis.com/storage/v1/b/{urllib.parse.quote(bucket,safe='')}/o/{encoded}?alt=media&generation={generation}"
    temporary=target.with_name(f".{target.name}.{generation}.tmp")
    try:
        request=urllib.request.Request(url,headers={"Authorization":f"Bearer {token}"})
        with urllib.request.urlopen(request,timeout=min(10.0,remaining())) as response,temporary.open("wb") as handle:
            while True:
                chunk=response.read(1024*1024)
                if not chunk: break
                handle.write(chunk)
        os.replace(temporary,target); return None
    except urllib.error.HTTPError as error:
        if error.code in (404,408,409,429,500,502,503,504): return {"name":name,"code":error.code}
        raise
    finally:
        temporary.unlink(missing_ok=True)

passes=[]
for attempt in range(1,4):
    for target in root.glob("job-*/*"):
        if target.is_file() and target.name in allowed:
            target.unlink()
    objects=list_objects(); entries=[entry for entry in (binding(item) for item in objects) if entry is not None]
    failures=[]
    if entries:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for failure in pool.map(download,entries):
                if failure is not None: failures.append(failure)
    passes.append({"attempt":attempt,"listed":len(objects),"selected":len(entries),"transient_failures":len(failures)})
    if not failures:
        print(json.dumps({"schema":"hu_m43_attempt02_v4_gcs_snapshot_v1","status":"pass","attempt_count":attempt,"listed_objects":len(objects),"selected_objects":len(entries),"elapsed_seconds":round(time.monotonic()-started,3),"passes":passes},sort_keys=True)); break
    if attempt==3: raise RuntimeError(f"transient object generations remained unstable: {failures}")
    time.sleep(min(1.0,remaining()))
else:
    raise RuntimeError("unreachable Attempt02 snapshot state")
'@
    $oldToken=[Environment]::GetEnvironmentVariable("M43_ATTEMPT02_GCS_TOKEN","Process")
    [Environment]::SetEnvironmentVariable("M43_ATTEMPT02_GCS_TOKEN",$tokenCandidates[0],"Process")
    $old=$ErrorActionPreference;$ErrorActionPreference="Continue"
    try{$snapshotOutput=@($snapshotFetcher|& python - $Bucket $RunName $Destination 2>&1);$snapshotCode=$LASTEXITCODE}
    finally{
        [Environment]::SetEnvironmentVariable("M43_ATTEMPT02_GCS_TOKEN",$oldToken,"Process")
        $ErrorActionPreference=$old
    }
    if($snapshotCode -ne 0){throw "Attempt02 bounded GCS snapshot failed; refusing to report the run inactive: $($snapshotOutput-join[Environment]::NewLine)"}
    if($snapshotOutput.Count -ne 1){throw "Attempt02 bounded GCS snapshot output was ambiguous"}
    $audit=$snapshotOutput[0]|ConvertFrom-Json
    if($audit.schema -ne "hu_m43_attempt02_v4_gcs_snapshot_v1" -or $audit.status -ne "pass" -or
       (Get-StrictInteger $audit.attempt_count "snapshot.attempt_count") -lt 1 -or (Get-StrictInteger $audit.attempt_count "snapshot.attempt_count") -gt 3 -or
       [double]$audit.elapsed_seconds -gt 55.0){throw "Attempt02 bounded GCS snapshot receipt is invalid"}
    return $audit
}

function Get-StrictInteger {
    param([Parameter(Mandatory = $true)]$Value, [Parameter(Mandatory = $true)][string]$Name)
    $types = @([byte], [sbyte], [int16], [uint16], [int32], [uint32], [int64], [uint64])
    if ($null -eq $Value -or $Value -is [bool] -or $types -notcontains $Value.GetType()) { throw "$Name must be a JSON integer, not bool/string/float" }
    return [int64]$Value
}

function Assert-SourcePackagePolicy {
    param([Parameter(Mandatory = $true)]$Manifest, [Parameter(Mandatory = $true)][string]$ArchivePath)
    $policy=$Manifest.source_package_policy
    $roots=@("src/ofc_regular/train_hu_m43_attempt02_fold_job.py","src/ofc_regular/assemble_hu_m43_attempt02_model.py")
    $patterns=@("(?i)l[o]cked","(?i)\.jsonl?$","(?i)(^|/)(outputs?|configs?|data)(/|$)")
    if ([string]$policy.mode -ne "ast_recursive_local_import_closure_v1" -or (@($policy.roots)-join "|") -ne ($roots-join "|") -or
        (@($policy.forbidden_entry_patterns)-join "|") -ne ($patterns-join "|") -or
        (Get-StrictInteger $policy.module_count "module_count") -lt 3 -or [string]$policy.entries_sha256 -notmatch '^[0-9a-f]{64}$' -or
        $policy.local_input_values_embedded -ne $false -or $policy.nontrain_cloud_inputs -ne 0) { throw "Attempt02 source package policy changed" }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip=[IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        if (@($zip.Entries|Where-Object{$_.FullName -match '\\'}).Count -ne 0) { throw "Frozen source archive contains non-portable entry separators" }
        [string[]]$entries=@($zip.Entries|Where-Object{$_.FullName -match '\.py$'}|ForEach-Object{$_.FullName})
        [Array]::Sort($entries,[StringComparer]::Ordinal)
        if ($entries.Count -ne (Get-StrictInteger $policy.module_count "module_count") -or
            @($entries|Where-Object{$_ -match '(?i)l[o]cked|\.jsonl?$|(^|/)(outputs?|configs?|data)(/|$)'}).Count -ne 0 -or
            (Get-TextSha256 ('["'+($entries -join '","')+'"]')) -ne [string]$policy.entries_sha256) { throw "Attempt02 frozen source archive violates package policy" }
    }
    finally { $zip.Dispose() }
}

function Assert-Manifest {
    param([Parameter(Mandatory = $true)]$Manifest)
    if ($Manifest.schema -ne "hu_m43_attempt02_v4_fold_spot_run_manifest_v1" -or $Manifest.status -ne "frozen" -or
        $Manifest.run_name -ne $RunName -or $Manifest.project_id -ne $ProjectId -or $Manifest.bucket -ne $Bucket -or
        (Get-StrictInteger $Manifest.job_count "job_count") -ne $ExpectedJobs -or
        (Get-StrictInteger $Manifest.compute.shard_count "shard_count") -ne 8 -or
        (Get-StrictInteger $Manifest.compute.jobs_per_shard "jobs_per_shard") -ne 4 -or
        (Get-StrictInteger $Manifest.compute.max_parallel_processes "max_parallel_processes") -ne 4 -or
        $Manifest.compute.auto_delete -ne $true -or $Manifest.current_profile_mutated -ne $false -or $Manifest.runtime_policy_activated -ne $false) {
        throw "Attempt02 local frozen manifest identity/schema is invalid"
    }
    if (@($Manifest.inputs.PSObject.Properties.Name|Where-Object{$_ -notin @("train","input_bundle_sha256")}).Count -ne 0 -or
        @($Manifest.inputs.train).Count -ne 20 -or @($Manifest.jobs).Count -ne 30 -or
        [string]$Manifest.dependencies.numpy -ne "2.2.6" -or [string]$Manifest.dependencies.scikit_learn -ne "1.8.0") {
        throw "Attempt02 train-only/dependency boundary changed"
    }
    for($i=0;$i -lt $ExpectedJobs;$i++) {
        $job=$Manifest.jobs[$i]; $slot=$i%6; $kind=if($slot -eq 0){"outer_runtime"}else{"inner_oof_safety"}; $inner=if($slot -eq 0){$null}else{$slot-1}
        $actualInner=if($null -eq $job.inner_fold){$null}else{Get-StrictInteger $job.inner_fold "jobs[$i].inner_fold"}
        if ((Get-StrictInteger $job.job_index "jobs[$i].job_index") -ne $i -or [string]$job.job_kind -ne $kind -or
            (Get-StrictInteger $job.outer_fold "jobs[$i].outer_fold") -ne [Math]::Floor($i/6) -or
            (($null -eq $inner -and $null -ne $actualInner)-or($null -ne $inner -and $actualInner -ne $inner)) -or
            (Get-StrictInteger $job.shard_index "jobs[$i].shard_index") -ne [Math]::Floor($i/$JobsPerShard) -or
            [string]$job.job_spec_sha256 -notmatch '^[0-9a-f]{64}$' -or [string]$job.done_uri -ne ("gs://$Bucket/runs/$RunName/results/job-{0:D2}/DONE.json" -f $i)) {
            throw "Attempt02 frozen job mapping changed at $i"
        }
    }
}

function Assert-JobObject {
    param([Parameter(Mandatory = $true)]$Object,[Parameter(Mandatory = $true)][string]$Schema,
          [Parameter(Mandatory = $true)][int]$JobIndex,[Parameter(Mandatory = $true)]$Manifest,
          [Parameter(Mandatory = $true)][string]$ManifestSha)
    $expected=$Manifest.jobs[$JobIndex]
    if ($Object.schema -ne $Schema -or $Object.run_name -ne $RunName -or (Get-StrictInteger $Object.job_index "$Schema.job_index") -ne $JobIndex -or
        [string]$Object.job_spec_sha256 -ne [string]$expected.job_spec_sha256 -or [string]$Object.source_sha256 -ne [string]$Manifest.source.sha256 -or
        [string]$Object.run_manifest_sha256 -ne $ManifestSha -or [string]$Object.cloud_contract_file_sha256 -ne [string]$Manifest.cloud_contract.file_sha256 -or
        [string]$Object.input_bundle_sha256 -ne [string]$Manifest.inputs.input_bundle_sha256 -or
        $Object.current_profile_mutated -ne $false -or $Object.runtime_policy_activated -ne $false) { throw "Stale or invalid $Schema for job $JobIndex" }
}

if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw "RunName is not path-safe" }
$repoRoot=(Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDir=Join-Path $repoRoot "outputs/gcp_runs/$RunName"
$manifestPath=Join-Path $runDir "m43_attempt02_v4_fold_run_manifest.json"
$cloudContractPath=Join-Path $runDir "attempt02_fold_cloud_contract.json"
$sourceArchivePath=Join-Path $runDir "ofc_regular_hu_m43_attempt02_v4_source.zip"
foreach($path in @($manifestPath,$cloudContractPath,$sourceArchivePath)){if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "Attempt02 local frozen artifact is missing: $path"}}
$manifest=Get-Content -LiteralPath $manifestPath -Raw|ConvertFrom-Json; $manifestSha=Get-Sha256 $manifestPath
Assert-Manifest $manifest
if((Get-Sha256 $cloudContractPath)-ne[string]$manifest.cloud_contract.file_sha256 -or (Get-Sha256 $sourceArchivePath)-ne[string]$manifest.source.sha256){throw "Attempt02 local frozen hash chain changed"}
Assert-SourcePackagePolicy -Manifest $manifest -ArchivePath $sourceArchivePath

$configBinding=@'
import hashlib,json,sys
m=json.load(open(sys.argv[1],encoding="utf-8")); p=json.load(open(sys.argv[2],encoding="utf-8"))
c=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
unsigned=dict(p); declared=unsigned.pop("contract_sha256"); assert declared==c(unsigned)
assert p["schema"]=="hu_m43_attempt02_v4_fold_cloud_contract_v1" and p["status"]=="frozen_train_only"
assert set(p["inputs"])=={"train"} and set(m["inputs"])=={"train","input_bundle_sha256"}
assert m["training_config"]==p["training_config"] and m["training_config_sha256"]==p["training_config_sha256"]==c(p["training_config"])
assert m["dependencies"]==p["dependencies"]=={"numpy":"2.2.6","scikit_learn":"1.8.0"}
assert m["process_environment"]==p["process_environment"]
assert [{k:v for k,v in e.items() if k!="uri"} for e in m["inputs"]["train"]]==p["inputs"]["train"]
assert m["inputs"]["input_bundle_sha256"]==p["input_bundle_sha256"]
assert len(m["jobs"])==len(p["fold_plan"]["jobs"])==30
for i,(a,b) in enumerate(zip(m["jobs"],p["fold_plan"]["jobs"])):
 assert a["job_index"]==b["job_index"]==i and a["job_spec"]=={k:v for k,v in b.items() if k!="job_spec_sha256"} and a["job_spec_sha256"]==b["job_spec_sha256"]
'@
$old=$ErrorActionPreference;$ErrorActionPreference="Continue"
try{$bindingOutput=@($configBinding|& python - $manifestPath $cloudContractPath 2>&1);$bindingCode=$LASTEXITCODE}finally{$ErrorActionPreference=$old}
if($bindingCode -ne 0){throw "Attempt02 run/contract binding is invalid: $($bindingOutput -join [Environment]::NewLine)"}

$prefix="gs://$Bucket/runs/$RunName/results"
$seenDoneJobs=[Collections.Generic.HashSet[int]]::new();$doneByJob=[ordered]@{}
$observations=[ordered]@{status=[ordered]@{};heartbeat=[ordered]@{};checkpoint=[ordered]@{}}
$snapshotRoot=Join-Path([IO.Path]::GetTempPath())("m43-attempt02-status-"+[guid]::NewGuid().ToString("N"));New-Item -ItemType Directory -Path $snapshotRoot|Out-Null
try{
    $snapshotAudit=Sync-GcsResultsSnapshot -Uri $prefix -Destination $snapshotRoot
    $unexpectedJobDirectories=@(Get-ChildItem -LiteralPath $snapshotRoot -Directory -ErrorAction SilentlyContinue|Where-Object{$_.Name -like 'job-*' -and $_.Name -notmatch '^job-[0-2][0-9]$'})
    if($unexpectedJobDirectories.Count -ne 0){throw "Attempt02 bulk snapshot contains an out-of-range job directory"}
    $allDoneFiles=@(Get-ChildItem -LiteralPath $snapshotRoot -Filter DONE.json -File -Recurse -ErrorAction SilentlyContinue)
    foreach($doneFile in $allDoneFiles){if($doneFile.Directory.Name -notmatch '^job-(\d{2})$'){throw "Unexpected Attempt02 DONE path in bulk snapshot: $($doneFile.FullName)"}}
    for($job=0;$job -lt $ExpectedJobs;$job++){
        $jobDir=Join-Path $snapshotRoot("job-{0:D2}" -f $job)
        foreach($type in @("status","heartbeat","checkpoint")){
            $path=Join-Path $jobDir "$type.json"
            if(Test-Path -LiteralPath $path -PathType Leaf){
                $obj=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json;$expectedSchema="hu_m43_attempt02_v4_fold_job_${type}_v1"
                if($obj.schema -ne $expectedSchema -or $obj.run_name -ne $RunName -or(Get-StrictInteger $obj.job_index "$type.job_index") -ne $job -or
                   [string]$obj.run_manifest_sha256 -ne $manifestSha -or[string]$obj.source_sha256 -ne[string]$manifest.source.sha256 -or
                   $obj.current_profile_mutated -ne $false -or $obj.runtime_policy_activated -ne $false){throw "Stale or invalid Attempt02 ${type} in bulk snapshot for job $job"}
                $observations[$type]["$job"]=$obj
            }
        }
        $donePath=Join-Path $jobDir "DONE.json"
        if(-not(Test-Path -LiteralPath $donePath -PathType Leaf)){continue}
        if(-not$seenDoneJobs.Add($job)){throw "Duplicate DONE job identity: $job"}
        $done=Get-Content -LiteralPath $donePath -Raw|ConvertFrom-Json
        if($done.status -ne "complete"){throw "Attempt02 DONE is incomplete for job $job"}
        Assert-JobObject $done "hu_m43_attempt02_v4_fold_done_v1" $job $manifest $manifestSha
        $estimatorPath=Join-Path $jobDir "estimator.pkl";$jobManifestPath=Join-Path $jobDir "job_manifest.json"
        if(-not(Test-Path -LiteralPath $estimatorPath -PathType Leaf)-or-not(Test-Path -LiteralPath $jobManifestPath -PathType Leaf)){throw "Attempt02 DONE lacks its preceding immutable artifact chain for job $job"}
        $jobManifest=Get-Content -LiteralPath $jobManifestPath -Raw|ConvertFrom-Json
        Assert-JobObject $jobManifest "hu_m43_attempt02_v4_fold_job_manifest_v1" $job $manifest $manifestSha
        if($jobManifest.status -ne "pass" -or(Get-Sha256 $estimatorPath)-ne[string]$done.artifact_sha256 -or
           (Get-Sha256 $estimatorPath)-ne[string]$jobManifest.artifact_sha256 -or(Get-Sha256 $jobManifestPath)-ne[string]$done.job_manifest_sha256){throw "Attempt02 remote estimator/job-manifest/DONE hash chain is invalid for job $job"}
        $done|Add-Member -NotePropertyName remote_artifact_chain_verified -NotePropertyValue $true;$doneByJob["$job"]=$done
    }
    if($allDoneFiles.Count -ne $seenDoneJobs.Count){throw "Attempt02 bulk snapshot DONE coverage is ambiguous"}
}
finally{Remove-Item -LiteralPath $snapshotRoot -Recurse -Force -ErrorAction SilentlyContinue}

$missingJobs=@(0..($ExpectedJobs-1)|Where-Object{-not$seenDoneJobs.Contains($_)})
$resumeShards=@($missingJobs|ForEach-Object{[Math]::Floor($_/$JobsPerShard)}|Sort-Object -Unique)
$vmPrefix=[string]$manifest.compute.vm_prefix
$instanceRaw=Invoke-Gcloud @("compute","instances","list","--project",$ProjectId,"--filter",("name~'^{0}-g[0-9]{{2}}$'" -f [regex]::Escape($vmPrefix)),"--format=json")
$instances=@()
if($instanceRaw){$parsedInstances=(@($instanceRaw)-join "`n")|ConvertFrom-Json;if($null -ne $parsedInstances){$instances=@($parsedInstances)}}

[ordered]@{
    schema="hu_m43_attempt02_v4_fold_spot_status_report_v1";run_name=$RunName
    state=if($seenDoneJobs.Count -eq $ExpectedJobs){"complete"}elseif($instances.Count -gt 0){"running"}else{"incomplete"}
    manifest_sha256=$manifestSha;completed_jobs=$seenDoneJobs.Count;total_jobs=$ExpectedJobs
    completed_job_indices=@($seenDoneJobs|Sort-Object);missing_job_indices=$missingJobs;resume_shards=$resumeShards
    resume_command=if($resumeShards.Count){".\scripts\Start-GcpHuM43Attempt02ModelRun.ps1 -RunName '$RunName' -Train <20-train-shards> -StartShards $($resumeShards -join ',') -ResumeExisting -CreateInstances"}else{$null}
    done=$doneByJob;status=$observations.status;heartbeat=$observations.heartbeat;checkpoint=$observations.checkpoint
    bulk_snapshot=$snapshotAudit
    active_instances=$instances.Count;instances=@($instances|ForEach-Object{[ordered]@{name=$_.name;zone=([string]$_.zone).Split('/')[-1];status=$_.status;machine_type=([string]$_.machineType).Split('/')[-1]}})
    cloud_input_boundary="fresh_train_only";current_profile_mutated=$false;runtime_policy_activated=$false
}|ConvertTo-Json -Depth 15
