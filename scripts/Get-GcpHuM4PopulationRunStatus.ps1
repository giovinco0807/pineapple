param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training"
)

$ErrorActionPreference = "Stop"
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw "RunName contains unsafe characters" }

function Get-Sha256([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Convert-RequiredJsonInteger($Value, [string]$Label) {
    if($null -eq $Value -or $Value -is [bool] -or $Value -is [string] -or
       $Value -is [single] -or $Value -is [double] -or $Value -is [decimal]){
        throw "$Label must be a JSON integer"
    }
    try { return [long]$Value } catch { throw "$Label must be a JSON integer" }
}
function Invoke-Gcloud([string[]]$Arguments) {
    $old=$ErrorActionPreference; $ErrorActionPreference="Continue"
    try { $output=& gcloud @Arguments 2>&1; $code=$LASTEXITCODE } finally { $ErrorActionPreference=$old }
    if($code -ne 0){ throw "gcloud failed ($code): $(@($output)-join [Environment]::NewLine)" }
    return @($output)
}

$repoRoot=(Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifestPath=Join-Path $repoRoot "outputs/gcp_runs/$RunName/population_run_manifest.json"
if(-not(Test-Path -LiteralPath $manifestPath -PathType Leaf)){ throw "Frozen population manifest is missing: $manifestPath" }
$manifest=Get-Content -LiteralPath $manifestPath -Raw|ConvertFrom-Json
$manifestSha=Get-Sha256 $manifestPath
if($manifest.schema -ne "hu_m4_population_spot_manifest_v1" -or $manifest.run_name -ne $RunName -or $manifest.project_id -ne $ProjectId -or $manifest.bucket -ne $Bucket -or $manifest.no_runtime_activation -ne $true -or $manifest.current_profile_mutated -ne $false -or $manifest.source.sha256 -notmatch '^[0-9a-f]{64}$' -or $manifest.runtime.model_sha256 -notmatch '^[0-9a-f]{64}$'){ throw "Frozen population manifest identity mismatch" }
$shardCount=Convert-RequiredJsonInteger $manifest.shards.count "manifest.shards.count"
if($shardCount -le 0){ throw "Frozen population manifest has no shards" }
$prefix="gs://$Bucket/runs/$RunName"
$old=$ErrorActionPreference;$ErrorActionPreference="Continue"
try{$rawDone=& gcloud storage ls "$prefix/results/*/DONE" --project $ProjectId 2>&1;$doneCode=$LASTEXITCODE}finally{$ErrorActionPreference=$old}
if($doneCode -eq 0){$doneUris=@($rawDone|Where-Object{$_ -match '/DONE$'})}
elseif((@($rawDone)-join "`n") -match '(?i)not found|does not exist|No URLs matched|404'){$doneUris=@()}
else{throw "Unable to list population DONE objects: $(@($rawDone)-join [Environment]::NewLine)"}
$doneRows=@()
$seenDoneShards=[System.Collections.Generic.HashSet[int]]::new()
foreach($uri in $doneUris){
    if($uri -notmatch '/results/shard-(\d{4})/DONE$'){ throw "Population DONE URI does not encode a canonical shard: $uri" }
    $uriShard=[int]$Matches[1]
    $expectedUri="$prefix/results/shard-{0:D4}/DONE" -f $uriShard
    if([string]$uri -cne $expectedUri){ throw "Population DONE URI is outside the frozen run prefix: $uri" }
    if($uriShard -lt 0 -or $uriShard -ge $shardCount){ throw "Population DONE URI shard is outside frozen range: $uri" }
    if(-not $seenDoneShards.Add($uriShard)){ throw "Duplicate population DONE shard: $uriShard" }
    $raw=@(Invoke-Gcloud @("storage","cat",$uri,"--project",$ProjectId))-join "`n"
    $row=$raw|ConvertFrom-Json
    $rowShard=Convert-RequiredJsonInteger $row.shard "DONE.shard"
    if($row.schema -ne "hu_m4_population_spot_done_v1" -or $row.status -ne "complete" -or $row.run_name -ne $RunName -or $rowShard -ne $uriShard -or $row.manifest_sha256 -ne $manifestSha -or $row.source_sha256 -ne $manifest.source.sha256 -or $row.model_sha256 -ne $manifest.runtime.model_sha256 -or $row.evaluation_sha256 -notmatch '^[0-9a-f]{64}$' -or $row.records_sha256 -notmatch '^[0-9a-f]{64}$' -or $row.current_profile_mutated -ne $false -or $row.no_runtime_activation -ne $true){ throw "Invalid population DONE object: $uri" }
    $doneRows += $row
}
$statusRows=@()
$old=$ErrorActionPreference;$ErrorActionPreference="Continue"
try{$statusUrisRaw=& gcloud storage ls "$prefix/status/*.json" --project $ProjectId 2>&1;$statusCode=$LASTEXITCODE}finally{$ErrorActionPreference=$old}
if($statusCode -eq 0){
    foreach($uri in @($statusUrisRaw|Where-Object{$_ -match '\.json$'})){
        $raw=@(Invoke-Gcloud @("storage","cat",$uri,"--project",$ProjectId))-join "`n"
        $statusRows+=($raw|ConvertFrom-Json)
    }
}elseif((@($statusUrisRaw)-join "`n") -notmatch '(?i)not found|does not exist|No URLs matched|404'){
    throw "Unable to list population status objects: $(@($statusUrisRaw)-join [Environment]::NewLine)"
}
$vmPrefix=(($RunName.ToLowerInvariant()-replace '[^a-z0-9-]','-').Trim('-'))
if($vmPrefix -notmatch '^[a-z]'){$vmPrefix="m4p-$vmPrefix"};if($vmPrefix.Length -gt 50){$vmPrefix=$vmPrefix.Substring(0,50).TrimEnd('-')}
$instancesRaw=Invoke-Gcloud @("compute","instances","list","--project",$ProjectId,"--filter","name~'^$vmPrefix-s'","--format=json")
$instances=if($instancesRaw){@((@($instancesRaw)-join "`n")|ConvertFrom-Json)}else{@()}
[ordered]@{
    schema="hu_m4_population_spot_status_report_v1"
    run_name=$RunName
    state=$(if($doneRows.Count -eq $shardCount){"complete"}elseif(@($statusRows|Where-Object{$_.state -eq 'failed'}).Count){"failed"}else{"running"})
    manifest_sha256=$manifestSha
    shards_total=$shardCount
    shards_complete=$doneRows.Count
    shards_failed=@($statusRows|Where-Object{$_.state -eq 'failed'}).Count
    shards_running=@($statusRows|Where-Object{$_.state -eq 'evaluating'}).Count
    completed_indices=@($doneRows|ForEach-Object{[int]$_.shard}|Sort-Object)
    missing_indices=@(0..($shardCount-1)|Where-Object{$_ -notin @($doneRows|ForEach-Object{[int]$_.shard})})
    active_instances=@($instances).Count
    instances=@($instances|ForEach-Object{[ordered]@{name=$_.name;zone=([string]$_.zone).Split('/')[-1];status=$_.status;machine_type=([string]$_.machineType).Split('/')[-1]}})
    no_runtime_activation=$true
    current_profile_mutated=$false
}|ConvertTo-Json -Depth 10
