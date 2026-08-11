param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [Parameter(Mandatory = $true)][string[]]$Train,
    [Parameter(Mandatory = $true)][string[]]$Calibration,
    [Parameter(Mandatory = $true)][string]$Attempt02DataContract,
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$ExpectedJobs = 30
$ExpectedTrainShards = 20
$ExpectedCalibrationShards = 10

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-TextSha256 {
    param([Parameter(Mandatory = $true)][string]$Text)
    $sha=[Security.Cryptography.SHA256]::Create()
    try{return([BitConverter]::ToString($sha.ComputeHash([Text.UTF8Encoding]::new($false).GetBytes($Text)))).Replace("-","").ToLowerInvariant()}
    finally{$sha.Dispose()}
}

function Resolve-InputPath {
    param([Parameter(Mandatory = $true)][string]$Path,[Parameter(Mandatory = $true)][string]$RepoRoot,[Parameter(Mandatory = $true)][string]$Label)
    $candidate=if([IO.Path]::IsPathRooted($Path)){$Path}else{Join-Path $RepoRoot $Path}
    if(-not(Test-Path -LiteralPath $candidate -PathType Leaf)){throw "Required local $Label is missing: $candidate"}
    return(Resolve-Path -LiteralPath $candidate).Path
}

function Invoke-Gcloud {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $old=$ErrorActionPreference;$ErrorActionPreference="Continue"
    try{$output=& gcloud @Arguments 2>&1;$code=$LASTEXITCODE}finally{$ErrorActionPreference=$old}
    if($code -ne 0){throw "gcloud failed ($code): gcloud $($Arguments -join ' ')`n$(@($output)-join[Environment]::NewLine)"}
    return @($output)
}

function Get-GcsUris {
    param([Parameter(Mandatory = $true)][string]$Pattern)
    $old=$ErrorActionPreference;$ErrorActionPreference="Continue"
    try{$raw=& gcloud storage ls $Pattern 2>&1;$code=$LASTEXITCODE}finally{$ErrorActionPreference=$old}
    if($code -ne 0){throw "All 30 immutable Attempt02 DONE objects are required before receive: $(@($raw)-join[Environment]::NewLine)"}
    return @($raw|ForEach-Object{([string]$_).Trim()}|Where-Object{$_})
}

function Get-StrictInteger {
    param([Parameter(Mandatory = $true)]$Value,[Parameter(Mandatory = $true)][string]$Name)
    $types=@([byte],[sbyte],[int16],[uint16],[int32],[uint32],[int64],[uint64])
    if($null -eq $Value -or $Value -is[bool] -or $types -notcontains $Value.GetType()){throw "$Name must be a JSON integer, not bool/string/float"}
    return [int64]$Value
}

function Write-Utf8CreateNew {
    param([Parameter(Mandatory = $true)][string]$Path,[Parameter(Mandatory = $true)][string]$Text)
    $bytes=[Text.UTF8Encoding]::new($false).GetBytes($Text)
    $stream=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try{$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
}

function Assert-SourcePackagePolicy {
    param([Parameter(Mandatory = $true)]$Manifest,[Parameter(Mandatory = $true)][string]$ArchivePath)
    $policy=$Manifest.source_package_policy
    $roots=@("src/ofc_regular/train_hu_m43_attempt02_fold_job.py","src/ofc_regular/assemble_hu_m43_attempt02_model.py")
    $patterns=@("(?i)l[o]cked","(?i)\.jsonl?$","(?i)(^|/)(outputs?|configs?|data)(/|$)")
    if([string]$policy.mode -ne "ast_recursive_local_import_closure_v1" -or (@($policy.roots)-join"|") -ne($roots-join"|") -or
       (@($policy.forbidden_entry_patterns)-join"|") -ne($patterns-join"|") -or
       (Get-StrictInteger $policy.module_count "module_count") -lt 3 -or [string]$policy.entries_sha256 -notmatch '^[0-9a-f]{64}$' -or
       $policy.local_input_values_embedded -ne $false -or $policy.nontrain_cloud_inputs -ne 0){throw "Attempt02 source package policy changed"}
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip=[IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try{
        if(@($zip.Entries|Where-Object{$_.FullName -match '\\'}).Count -ne 0){throw "Frozen source archive contains non-portable entry separators"}
        [string[]]$entries=@($zip.Entries|Where-Object{$_.FullName -match '\.py$'}|ForEach-Object{$_.FullName})
        [Array]::Sort($entries,[StringComparer]::Ordinal)
        if($entries.Count -ne(Get-StrictInteger $policy.module_count "module_count") -or
           @($entries|Where-Object{$_ -match '(?i)l[o]cked|\.jsonl?$|(^|/)(outputs?|configs?|data)(/|$)'}).Count -ne 0 -or
           (Get-TextSha256 ('["'+($entries-join'","')+'"]')) -ne[string]$policy.entries_sha256){throw "Frozen source archive violates Attempt02 package policy"}
    }finally{$zip.Dispose()}
}

function Assert-Manifest {
    param([Parameter(Mandatory = $true)]$Manifest)
    if($Manifest.schema -ne "hu_m43_attempt02_v4_fold_spot_run_manifest_v1" -or $Manifest.status -ne "frozen" -or
       $Manifest.run_name -ne $RunName -or $Manifest.project_id -ne $ProjectId -or $Manifest.bucket -ne $Bucket -or
       (Get-StrictInteger $Manifest.job_count "job_count") -ne $ExpectedJobs -or
       (Get-StrictInteger $Manifest.compute.shard_count "shard_count") -ne 8 -or
       (Get-StrictInteger $Manifest.compute.jobs_per_shard "jobs_per_shard") -ne 4 -or
       (Get-StrictInteger $Manifest.compute.max_parallel_processes "max_parallel_processes") -ne 4 -or
       $Manifest.compute.auto_delete -ne $true -or $Manifest.current_profile_mutated -ne $false -or $Manifest.runtime_policy_activated -ne $false){throw "Attempt02 frozen manifest is invalid"}
    if(@($Manifest.inputs.PSObject.Properties.Name|Where-Object{$_ -notin@("train","input_bundle_sha256")}).Count -ne 0 -or
       @($Manifest.inputs.train).Count -ne 20 -or @($Manifest.jobs).Count -ne 30 -or
       [string]$Manifest.dependencies.numpy -ne "2.2.6" -or[string]$Manifest.dependencies.scikit_learn -ne "1.8.0"){throw "Attempt02 train-only boundary changed"}
    for($i=0;$i -lt $ExpectedJobs;$i++){
        $job=$Manifest.jobs[$i];$slot=$i%6;$kind=if($slot -eq 0){"outer_runtime"}else{"inner_oof_safety"};$inner=if($slot -eq 0){$null}else{$slot-1}
        $actualInner=if($null -eq $job.inner_fold){$null}else{Get-StrictInteger $job.inner_fold "jobs[$i].inner_fold"}
        if((Get-StrictInteger $job.job_index "jobs[$i].job_index") -ne $i -or[string]$job.job_kind -ne $kind -or
           (Get-StrictInteger $job.outer_fold "jobs[$i].outer_fold") -ne[Math]::Floor($i/6) -or
           (($null -eq $inner -and $null -ne $actualInner)-or($null -ne $inner -and $actualInner -ne $inner)) -or
           (Get-StrictInteger $job.shard_index "jobs[$i].shard_index") -ne[Math]::Floor($i/4) -or
           [string]$job.job_spec_sha256 -notmatch '^[0-9a-f]{64}$' -or[string]$job.done_uri -ne("gs://$Bucket/runs/$RunName/results/job-{0:D2}/DONE.json" -f $i)){throw "Attempt02 frozen job mapping changed at $i"}
    }
}

if($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$'){throw "RunName is not path-safe"}
if($Train.Count -ne $ExpectedTrainShards){throw "Receive requires the original 20 fresh train shards"}
if($Calibration.Count -ne $ExpectedCalibrationShards){throw "Receive requires the 10 fresh calibration shards"}
$repoRoot=(Resolve-Path(Join-Path $PSScriptRoot "..")).Path
$resolvedTrain=@($Train|ForEach-Object{Resolve-InputPath $_ $repoRoot "train shard"})
# Existence/path resolution does not parse either sealed local input.  The frozen
# assembler is the only code allowed to open them, and only after its train-OOF gate.
$resolvedCalibration=@($Calibration|ForEach-Object{Resolve-InputPath $_ $repoRoot "calibration shard"})
$resolvedDataContract=Resolve-InputPath $Attempt02DataContract $repoRoot "Attempt02 data contract"
$runDir=Join-Path $repoRoot "outputs/gcp_runs/$RunName"
$manifestPath=Join-Path $runDir "m43_attempt02_v4_fold_run_manifest.json"
$cloudContractPath=Join-Path $runDir "attempt02_fold_cloud_contract.json"
$localSourcePath=Join-Path $runDir "ofc_regular_hu_m43_attempt02_v4_source.zip"
foreach($path in @($manifestPath,$cloudContractPath)){if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "Attempt02 local frozen run artifact is missing: $path"}}
$manifest=Get-Content -LiteralPath $manifestPath -Raw|ConvertFrom-Json;$manifestSha=Get-Sha256 $manifestPath
Assert-Manifest $manifest
if((Get-Sha256 $cloudContractPath)-ne[string]$manifest.cloud_contract.file_sha256){throw "Attempt02 cloud contract hash changed"}
for($i=0;$i -lt $ExpectedTrainShards;$i++){
    if((Get-Sha256 $resolvedTrain[$i])-ne[string]$manifest.inputs.train[$i].sha256 -or
       [long](Get-Item -LiteralPath $resolvedTrain[$i]).Length -ne(Get-StrictInteger $manifest.inputs.train[$i].bytes "train.bytes") -or
       (Get-StrictInteger $manifest.inputs.train[$i].index "train.index") -ne $i){throw "Original fresh train provenance mismatch at index $i"}
}

$configBinding=@'
import hashlib,json,sys
m=json.load(open(sys.argv[1],encoding="utf-8")); p=json.load(open(sys.argv[2],encoding="utf-8"))
c=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
u=dict(p);d=u.pop("contract_sha256");assert d==c(u)
assert p["schema"]=="hu_m43_attempt02_v4_fold_cloud_contract_v1" and p["status"]=="frozen_train_only"
assert set(p["inputs"])=={"train"} and set(m["inputs"])=={"train","input_bundle_sha256"}
forbidden={"calibration","calibration_path","calibration_paths","data_contract","data_contract_path","inherited_locked","locked_holdout","locked_holdout_path"}
def reject(value):
 if isinstance(value,dict):
  for key,child in value.items(): assert str(key).lower() not in forbidden; reject(child)
 elif isinstance(value,list):
  for child in value: reject(child)
reject(p); reject(m)
assert m["training_config"]==p["training_config"] and m["training_config_sha256"]==p["training_config_sha256"]==c(p["training_config"])
assert m["dependencies"]==p["dependencies"]=={"numpy":"2.2.6","scikit_learn":"1.8.0"} and m["process_environment"]==p["process_environment"]
assert [{k:v for k,v in e.items() if k!="uri"} for e in m["inputs"]["train"]]==p["inputs"]["train"] and m["inputs"]["input_bundle_sha256"]==p["input_bundle_sha256"]
assert len(m["jobs"])==len(p["fold_plan"]["jobs"])==30
for i,(a,b) in enumerate(zip(m["jobs"],p["fold_plan"]["jobs"])):
 assert a["job_index"]==b["job_index"]==i and a["job_spec"]=={k:v for k,v in b.items() if k!="job_spec_sha256"} and a["job_spec_sha256"]==b["job_spec_sha256"]
'@
$old=$ErrorActionPreference;$ErrorActionPreference="Continue"
try{$bindingOutput=@($configBinding|& python - $manifestPath $cloudContractPath 2>&1);$bindingCode=$LASTEXITCODE}finally{$ErrorActionPreference=$old}
if($bindingCode -ne 0){throw "Attempt02 run/contract binding is invalid: $($bindingOutput-join[Environment]::NewLine)"}
$cloudProjection=Get-Content -LiteralPath $cloudContractPath -Raw|ConvertFrom-Json

$modelRunsRoot=[IO.Path]::GetFullPath((Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt02_model_runs"))
New-Item -ItemType Directory -Path $modelRunsRoot -Force|Out-Null
if($OutputDir){if(-not[IO.Path]::IsPathRooted($OutputDir)){$OutputDir=Join-Path $repoRoot $OutputDir};$OutputDir=[IO.Path]::GetFullPath($OutputDir)}else{$OutputDir=[IO.Path]::GetFullPath((Join-Path $modelRunsRoot $RunName))}
$rootPrefix=$modelRunsRoot.TrimEnd([IO.Path]::DirectorySeparatorChar,[IO.Path]::AltDirectorySeparatorChar)+[IO.Path]::DirectorySeparatorChar
if(-not$OutputDir.StartsWith($rootPrefix,[StringComparison]::OrdinalIgnoreCase)){throw "OutputDir must remain under m43_attempt02_model_runs; refusing any runtime profile overwrite"}
if(Test-Path -LiteralPath $OutputDir){
    $receiptPath=Join-Path $OutputDir "receipt.json";$trainingPath=Join-Path $OutputDir "training_manifest.json"
    if(-not(Test-Path $receiptPath -PathType Leaf)-or-not(Test-Path $trainingPath -PathType Leaf)){throw "OutputDir exists without a complete Attempt02 receipt"}
    $existing=Get-Content $receiptPath -Raw|ConvertFrom-Json
    $allowedExistingStatuses=@("no_go_precalibration","verified_candidate_ready_for_freeze","verified_no_go_calibration")
    if($existing.schema -ne "hu_m43_attempt02_v4_model_receive_receipt_v1" -or $existing.run_name -ne $RunName -or
       $existing.run_manifest_sha256 -ne $manifestSha -or(Get-Sha256 $trainingPath)-ne[string]$existing.training_manifest_sha256 -or
       [string]$existing.status -notin $allowedExistingStatuses -or
       (Get-StrictInteger $existing.job_count "existing receipt job_count") -ne $ExpectedJobs -or
       [string]$existing.cloud_input_boundary -ne "fresh_train_only" -or
       $existing.current_profile_mutated -ne $false -or $existing.runtime_policy_activated -ne $false){throw "Existing Attempt02 receive receipt is stale"}
    $existingModelPath=Join-Path $OutputDir "model.pkl"
    if($null -eq $existing.model_sha256){
        if([string]$existing.status -ne "no_go_precalibration" -or(Test-Path -LiteralPath $existingModelPath)){
            throw "Existing Attempt02 pre-calibration No-Go contains an unexpected model artifact"
        }
    }else{
        if([string]$existing.status -eq "no_go_precalibration" -or[string]$existing.model_sha256 -notmatch '^[0-9a-f]{64}$' -or
           -not(Test-Path -LiteralPath $existingModelPath -PathType Leaf)-or(Get-Sha256 $existingModelPath)-ne[string]$existing.model_sha256){
            throw "Existing Attempt02 model hash/status changed"
        }
    }
    $existing|ConvertTo-Json -Depth 20;exit 0
}

$stageDir=Join-Path $modelRunsRoot(".receiving-{0}-{1}" -f $RunName,[guid]::NewGuid().ToString("N"))
$foldArtifactsDir=Join-Path $stageDir "fold_artifacts";New-Item -ItemType Directory -Path $foldArtifactsDir -Force|Out-Null
$frozenSourcePath=Join-Path $stageDir "ofc_regular_hu_m43_attempt02_v4_source.zip"
if(Test-Path -LiteralPath $localSourcePath -PathType Leaf){[IO.File]::WriteAllBytes($frozenSourcePath,[IO.File]::ReadAllBytes($localSourcePath))}
else{Invoke-Gcloud @("storage","cp",$manifest.source.uri,$frozenSourcePath,"--project",$ProjectId)|Out-Null}
if((Get-Sha256 $frozenSourcePath)-ne[string]$manifest.source.sha256 -or[long](Get-Item $frozenSourcePath).Length -ne(Get-StrictInteger $manifest.source.bytes "source.bytes")){throw "Attempt02 frozen source hash/size mismatch"}
Assert-SourcePackagePolicy -Manifest $manifest -ArchivePath $frozenSourcePath
$frozenRepo=Join-Path $stageDir "frozen_repo";Expand-Archive -LiteralPath $frozenSourcePath -DestinationPath $frozenRepo
$frozenWorker=Join-Path $frozenRepo "src/ofc_regular/train_hu_m43_attempt02_fold_job.py";$frozenAssembler=Join-Path $frozenRepo "src/ofc_regular/assemble_hu_m43_attempt02_model.py"
if(-not(Test-Path $frozenWorker -PathType Leaf)-or-not(Test-Path $frozenAssembler -PathType Leaf)-or
   (Get-Sha256 $frozenWorker)-ne[string]$manifest.source.fold_worker_sha256 -or(Get-Sha256 $frozenAssembler)-ne[string]$manifest.source.assembler_sha256){throw "Attempt02 frozen entrypoint hash chain is invalid"}
$frozenPythonPath=Join-Path $frozenRepo "src"

$prefix="gs://$Bucket/runs/$RunName/results";$donePattern="^"+[regex]::Escape($prefix)+"/job-(\d{2})/DONE\.json$"
$doneUris=@(Get-GcsUris "$prefix/**/DONE.json");if($doneUris.Count -ne $ExpectedJobs){throw "Receive requires exactly 30 DONE URIs; found $($doneUris.Count)"}
$seen=[Collections.Generic.HashSet[int]]::new()
foreach($uri in $doneUris){$match=[regex]::Match($uri,$donePattern);if(-not$match.Success){throw "Unexpected Attempt02 DONE URI: $uri"};$job=[int]$match.Groups[1].Value;if($job -lt 0 -or $job -ge 30 -or-not$seen.Add($job)){throw "Out-of-range/duplicate Attempt02 DONE: $uri"};if($uri -ne[string]$manifest.jobs[$job].done_uri){throw "Attempt02 DONE URI differs from manifest"}}
$verifiedJobs=@()
for($i=0;$i -lt $ExpectedJobs;$i++){
    $jobDir=Join-Path $foldArtifactsDir("job-{0:D2}" -f $i);New-Item -ItemType Directory -Path $jobDir|Out-Null;$jobPrefix="$prefix/job-{0:D2}" -f $i
    foreach($name in @("estimator.pkl","job_manifest.json","DONE.json")){Invoke-Gcloud @("storage","cp","$jobPrefix/$name",(Join-Path $jobDir $name),"--project",$ProjectId)|Out-Null}
    $estimator=Join-Path $jobDir "estimator.pkl";$jobManifestPath=Join-Path $jobDir "job_manifest.json";$donePath=Join-Path $jobDir "DONE.json"
    $jobManifest=Get-Content $jobManifestPath -Raw|ConvertFrom-Json;$done=Get-Content $donePath -Raw|ConvertFrom-Json;$expected=$manifest.jobs[$i]
    foreach($check in @(@($jobManifest,"hu_m43_attempt02_v4_fold_job_manifest_v1","pass"),@($done,"hu_m43_attempt02_v4_fold_done_v1","complete"))){
        $object=$check[0]
        if($object.schema -ne $check[1] -or $object.status -ne $check[2] -or $object.run_name -ne $RunName -or
           (Get-StrictInteger $object.job_index "job_index") -ne $i -or[string]$object.job_spec_sha256 -ne[string]$expected.job_spec_sha256 -or
           [string]$object.source_sha256 -ne[string]$manifest.source.sha256 -or[string]$object.run_manifest_sha256 -ne $manifestSha -or
           [string]$object.cloud_contract_file_sha256 -ne[string]$manifest.cloud_contract.file_sha256 -or[string]$object.input_bundle_sha256 -ne[string]$manifest.inputs.input_bundle_sha256 -or
           $object.current_profile_mutated -ne $false -or $object.runtime_policy_activated -ne $false){throw "Downloaded Attempt02 fold object is stale/invalid at job $i"}
    }
    $artifactSha=Get-Sha256 $estimator;$jobManifestSha=Get-Sha256 $jobManifestPath
    if($artifactSha -ne[string]$jobManifest.artifact_sha256 -or $artifactSha -ne[string]$done.artifact_sha256 -or $jobManifestSha -ne[string]$done.job_manifest_sha256){throw "Downloaded Attempt02 job hash chain is invalid at job $i"}
    $verifiedJobs += [ordered]@{job_index=$i;job_spec_sha256=[string]$expected.job_spec_sha256;artifact_sha256=$artifactSha;job_manifest_sha256=$jobManifestSha}
}

$outputModel=Join-Path $stageDir "model.pkl";$trainingManifestOutput=Join-Path $stageDir "training_manifest.json"
$assemblerStdout=Join-Path $stageDir "assembler_stdout.log";$assemblerStderr=Join-Path $stageDir "assembler_stderr.log"
$assemblerArgs=@("-m","ofc_regular.assemble_hu_m43_attempt02_model","--fold-artifacts-dir",$foldArtifactsDir)
foreach($path in $resolvedTrain){$assemblerArgs+=@("--train",$path)}
foreach($path in $resolvedCalibration){$assemblerArgs+=@("--calibration",$path)}
$assemblerArgs+=@("--attempt02-data-contract",$resolvedDataContract,"--repo-root",$repoRoot,"--fold-cloud-contract",$cloudContractPath,
    "--output-model",$outputModel,"--manifest-output",$trainingManifestOutput,"--run-name",$RunName,
    "--source-sha256",[string]$manifest.source.sha256,"--run-manifest-sha256",$manifestSha)
$environment=[ordered]@{PYTHONPATH=$frozenPythonPath;PYTHONHASHSEED="0";OMP_NUM_THREADS="1";OPENBLAS_NUM_THREADS="1";MKL_NUM_THREADS="1";NUMEXPR_NUM_THREADS="1"};$oldEnvironment=@{}
foreach($name in $environment.Keys){$oldEnvironment[$name]=[Environment]::GetEnvironmentVariable($name,"Process");[Environment]::SetEnvironmentVariable($name,[string]$environment[$name],"Process")}
$old=$ErrorActionPreference;$ErrorActionPreference="Continue"
$importPreflight=@'
import pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
import ofc_regular,ofc_regular.train_hu_m43_attempt02_fold_job as worker,ofc_regular.assemble_hu_m43_attempt02_model as assembler
import numpy,sklearn
for module in (ofc_regular,worker,assembler): pathlib.Path(module.__file__).resolve().relative_to(root)
assert numpy.__version__=="2.2.6" and sklearn.__version__=="1.8.0"
'@
try{
    $importOutput=@($importPreflight|& python - $frozenPythonPath 2>&1);if($LASTEXITCODE -ne 0){throw "Attempt02 frozen source import preflight failed: $($importOutput-join[Environment]::NewLine)"}
    & python @assemblerArgs 1>$assemblerStdout 2>$assemblerStderr;$assemblerCode=$LASTEXITCODE
}finally{foreach($name in $oldEnvironment.Keys){[Environment]::SetEnvironmentVariable($name,$oldEnvironment[$name],"Process")};$ErrorActionPreference=$old}
if($assemblerCode -ne 0){throw "Attempt02 local assembler failed once with exit $assemblerCode; staging preserved at $stageDir"}
if(-not(Test-Path $trainingManifestOutput -PathType Leaf)){throw "Attempt02 assembler did not write its decision manifest"}
$training=Get-Content $trainingManifestOutput -Raw|ConvertFrom-Json;$modelSha=$null
$decisionDigestCheck=@'
import hashlib,json,sys
value=json.load(open(sys.argv[1],encoding="utf-8")); key="receipt_sha256" if value.get("schema")=="hu_m43_attempt02_v4_assembly_decision_v1" else "manifest_sha256"
unsigned=dict(value); declared=unsigned.pop(key)
actual=hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(",",":")).encode()).hexdigest()
assert declared==actual
'@
$old=$ErrorActionPreference;$ErrorActionPreference="Continue"
try{$digestOutput=@($decisionDigestCheck|& python - $trainingManifestOutput 2>&1);$digestCode=$LASTEXITCODE}finally{$ErrorActionPreference=$old}
if($digestCode -ne 0){throw "Attempt02 assembler decision digest is invalid: $($digestOutput-join[Environment]::NewLine)"}
if($training.schema -eq "hu_m43_attempt02_v4_assembly_decision_v1"){
    if($training.status -ne "no_go_precalibration" -or $training.model_written -ne $false -or(Test-Path $outputModel) -or
       $training.calibration.data_contract_opened -ne $false -or $training.calibration.fresh_rows_opened -ne $false -or
       $training.inherited_holdout_content_opened -ne $false -or $training.current_profile_mutated -ne $false -or $training.runtime_policy_activated -ne $false){throw "Attempt02 pre-calibration No-Go contract was violated"}
    $receiveStatus="no_go_precalibration"
}elseif($training.schema -eq "hu_m43_t1_training_manifest_v4"){
    if(-not(Test-Path $outputModel -PathType Leaf)){throw "Attempt02 post-gate assembly omitted model"};$modelSha=Get-Sha256 $outputModel
    if([string]$training.model_sha256 -ne $modelSha -or [string]$training.model_artifact.sha256 -ne $modelSha -or
       $training.milestone -ne "M4.3-attempt02" -or $training.fold_assembly.status -ne "verified_consumed_exactly_once" -or
       (Get-StrictInteger $training.fold_assembly.job_count "fold_assembly.job_count") -ne 30 -or @($training.fold_assembly.jobs).Count -ne 30 -or
       $training.calibration_opened_after_precalibration_go -ne $true -or $training.inherited_holdout_content_opened -ne $false -or
       $training.current_profile_mutated -ne $false -or $training.runtime_policy_activated -ne $false){throw "Attempt02 post-gate training manifest violates distributed contract"}
    $receiveStatus=if($training.promotion_status -eq "candidate_ready_for_freeze"){"verified_candidate_ready_for_freeze"}elseif($training.promotion_status -eq "no_go_calibration"){"verified_no_go_calibration"}else{throw "Unexpected Attempt02 promotion status"}
}else{throw "Unexpected Attempt02 assembler decision schema"}
$assembly=$training.fold_assembly
if($assembly.status -ne "verified_consumed_exactly_once" -or(Get-StrictInteger $assembly.job_count "fold_assembly.job_count") -ne 30 -or
   $assembly.run_name -ne $RunName -or[string]$assembly.source_sha256 -ne[string]$manifest.source.sha256 -or
   [string]$assembly.run_manifest_sha256 -ne $manifestSha -or[string]$assembly.cloud_contract_file_sha256 -ne[string]$manifest.cloud_contract.file_sha256 -or
   [string]$assembly.cloud_contract_sha256 -ne[string]$cloudProjection.contract_sha256 -or@($assembly.jobs).Count -ne 30){throw "Attempt02 fold assembly lineage is invalid"}
for($i=0;$i -lt $ExpectedJobs;$i++){
    $actual=$assembly.jobs[$i];$expected=$verifiedJobs[$i]
    if((Get-StrictInteger $actual.job_index "fold_assembly.jobs[$i].job_index") -ne $i -or
       [string]$actual.job_spec_sha256 -ne[string]$expected.job_spec_sha256 -or[string]$actual.artifact_sha256 -ne[string]$expected.artifact_sha256 -or
       [string]$actual.job_manifest_sha256 -ne[string]$expected.job_manifest_sha256){throw "Attempt02 assembler job binding mismatch at job $i"}
}

[IO.File]::WriteAllBytes((Join-Path $stageDir "attempt02_fold_cloud_contract.json"),[IO.File]::ReadAllBytes($cloudContractPath))
[IO.File]::WriteAllBytes((Join-Path $stageDir "m43_attempt02_v4_fold_run_manifest.json"),[IO.File]::ReadAllBytes($manifestPath))
$receipt=[ordered]@{
    schema="hu_m43_attempt02_v4_model_receive_receipt_v1";status=$receiveStatus;run_name=$RunName
    run_manifest_sha256=$manifestSha;source_sha256=[string]$manifest.source.sha256;cloud_contract_sha256=[string]$manifest.cloud_contract.file_sha256
    cloud_input_boundary="fresh_train_only";job_count=$ExpectedJobs;jobs=$verifiedJobs;model_sha256=$modelSha
    training_manifest_sha256=Get-Sha256 $trainingManifestOutput;calibration_and_contract_passed_to_frozen_assembler_only=$true
    precalibration_no_go_opens_local_calibration_or_contract=$false;current_profile_mutated=$false;runtime_policy_activated=$false
    received_at=(Get-Date).ToUniversalTime().ToString("o")
}
Write-Utf8CreateNew -Path(Join-Path $stageDir "receipt.json") -Text(($receipt|ConvertTo-Json -Depth 20)+"`n")
if(Test-Path -LiteralPath $OutputDir){throw "OutputDir appeared during receive; staging preserved at $stageDir"}
Move-Item -LiteralPath $stageDir -Destination $OutputDir
$receipt|ConvertTo-Json -Depth 20
