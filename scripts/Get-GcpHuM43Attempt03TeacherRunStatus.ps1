param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{0,126}[a-z0-9]$') { throw "RunName is not path-safe" }
if ($ProjectId -notmatch '^[a-z][a-z0-9-]{4,28}[a-z0-9]$') { throw "ProjectId is not valid" }
if ($Bucket -notmatch '^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$' -or $Bucket.Contains('..')) { throw "Bucket is not valid" }

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Quote-NativeArgument([string]$Value) {
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Stop-ProcessTree([Diagnostics.Process]$Process) {
    if ($Process.HasExited) { return }
    if ($env:OS -eq 'Windows_NT') {
        & (Join-Path $env:SystemRoot 'System32/taskkill.exe') /PID $Process.Id /T /F *> $null
    }
    else { try { $Process.Kill($true) } catch { $Process.Kill() } }
}

function Invoke-GcloudBounded([string[]]$Arguments, [int]$TimeoutSeconds = 45, [switch]$AllowFailure) {
    $command = Get-Command gcloud -ErrorAction Stop | Select-Object -First 1
    $source = [string]$command.Source
    if (-not $source) { $source = [string]$command.Definition }
    $info = [Diagnostics.ProcessStartInfo]::new()
    if ([IO.Path]::GetExtension($source).ToLowerInvariant() -eq '.ps1') {
        $info.FileName = [string](Get-Process -Id $PID).Path
        $allArguments = @('-NoLogo','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',$source) + $Arguments
    }
    else { $info.FileName = $source; $allArguments = $Arguments }
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    if ($null -ne $info.PSObject.Properties['ArgumentList']) {
        foreach ($argument in $allArguments) { [void]$info.ArgumentList.Add([string]$argument) }
    }
    else { $info.Arguments = (@($allArguments | ForEach-Object { Quote-NativeArgument ([string]$_) }) -join ' ') }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    try {
        if (-not $process.Start()) { throw "Unable to start gcloud" }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            Stop-ProcessTree $process
            throw "gcloud timed out after $TimeoutSeconds seconds: gcloud $($Arguments -join ' ')"
        }
        $process.WaitForExit()
        $stdout = [string]$stdoutTask.GetAwaiter().GetResult()
        $stderr = [string]$stderrTask.GetAwaiter().GetResult()
        $output = @($stdout -split "`r?`n" | Where-Object { $_ }) + @($stderr -split "`r?`n" | Where-Object { $_ })
        if ($process.ExitCode -ne 0 -and -not $AllowFailure) {
            throw "gcloud failed ($($process.ExitCode)): $($Arguments -join ' ')`n$($output -join [Environment]::NewLine)"
        }
        return [pscustomobject]@{ exit_code=[int]$process.ExitCode; stdout=$stdout; stderr=$stderr; output=$output }
    }
    finally { $process.Dispose() }
}

function Test-IsNoMatch($Result) {
    return (($Result.output -join "`n") -match '(?i)not found|does not exist|no urls matched|matched no objects|404')
}

function Get-GcsUris([string]$Uri) {
    $result = Invoke-GcloudBounded @('storage','ls',$Uri,'--project',$ProjectId) 30 -AllowFailure
    if ($result.exit_code -eq 0) { return @($result.stdout -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }
    if (Test-IsNoMatch $result) { return @() }
    throw "Unable to list Attempt03 objects: $Uri`n$($result.output -join [Environment]::NewLine)"
}

function Get-GcsText([string]$Uri, [switch]$AllowMissing) {
    $result = Invoke-GcloudBounded @('storage','cat',$Uri,'--project',$ProjectId) 30 -AllowFailure
    if ($result.exit_code -eq 0) { return [string]$result.stdout }
    if ($AllowMissing -and (Test-IsNoMatch $result)) { return $null }
    throw "Unable to read Attempt03 object: $Uri`n$($result.output -join [Environment]::NewLine)"
}

function Assert-GcsObjectMatchesFile([string]$Uri, [string]$LocalPath) {
    $tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("hu-m43-a03-status-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tempRoot | Out-Null
    $download = Join-Path $tempRoot 'object.bin'
    try {
        [void](Invoke-GcloudBounded @('storage','cp',$Uri,$download,'--project',$ProjectId) 45)
        if ((Get-Sha256 $download) -ne (Get-Sha256 $LocalPath)) { throw "Remote immutable object bytes changed: $Uri" }
    }
    finally { Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue }
}

function Convert-ToVmPrefix([string]$Value) {
    $clean = ($Value.ToLowerInvariant() -replace '[^a-z0-9-]', '-') -replace '-+', '-'
    $clean = $clean.Trim('-')
    if (-not $clean) { throw "RunName cannot be converted to a VM prefix" }
    if ($clean.Length -gt 54) { $clean = $clean.Substring(0,54).TrimEnd('-') }
    return $clean
}

function Resolve-ShardState([bool]$HasDone, [string]$WorkerStatus, [bool]$HasActiveVm) {
    if ($HasDone) { return 'complete' }
    if ($WorkerStatus -eq 'complete') { return 'inconsistent' }
    if ($WorkerStatus -eq 'running' -and $HasActiveVm) { return 'running' }
    if ($WorkerStatus -eq 'running') { return 'interrupted' }
    if ($WorkerStatus -eq 'failed' -and $HasActiveVm) { return 'starting' }
    if ($WorkerStatus -eq 'failed') { return 'failed' }
    if ($HasActiveVm) { return 'starting' }
    return 'not_started'
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
$manifestPath = Join-Path $runDir 'manifest.json'
$shardsPath = Join-Path $runDir 'shards_manifest.jsonl'
$planPath = Join-Path $repoRoot 'configs/hu_joint_policy_m43_attempt03.json'
$freezePath = Join-Path $repoRoot 'configs/hu_joint_policy_m43_attempt03_model_freeze.json'
foreach ($path in @($manifestPath,$shardsPath,$planPath,$freezePath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Local Attempt03 immutable input is missing: $path" }
}

$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
$freeze = Get-Content -Raw -LiteralPath $freezePath | ConvertFrom-Json
if ($manifest.schema -ne 'hu_m43_attempt03_teacher_spot_manifest_v1' -or
    [string]$manifest.run_name -ne $RunName -or
    [string]$manifest.project_id -ne $ProjectId -or
    [string]$manifest.bucket -ne $Bucket) {
    throw "Attempt03 manifest identity changed"
}
if ($plan.schema -ne 'hu_m43_attempt03_plan_v1' -or $plan.status -ne 'frozen_pre_generation' -or
    (Get-Sha256 $planPath) -ne [string]$freeze.parent_plan.file_sha256 -or
    [string]$manifest.plan_file_sha256 -ne [string]$freeze.parent_plan.file_sha256) {
    throw "Attempt03 plan/freeze binding changed"
}
if ($freeze.schema -ne 'hu_m43_attempt03_model_freeze_v1' -or
    $freeze.status -ne 'frozen_before_any_attempt03_teacher_row_was_received' -or
    [string]$freeze.teacher_run.run_name -ne $RunName -or
    (Get-Sha256 $manifestPath) -ne [string]$freeze.teacher_run.manifest_sha256) {
    throw "Attempt03 teacher run does not match the model freeze"
}
if ((Get-Sha256 $shardsPath) -ne [string]$manifest.schedule_sha256 -or
    [string]$freeze.teacher_run.schedule_sha256 -ne [string]$manifest.schedule_sha256) {
    throw "Attempt03 schedule SHA256 changed"
}

$prefix = "gs://$Bucket/runs/$RunName"
Assert-GcsObjectMatchesFile "$prefix/manifest.json" $manifestPath
Assert-GcsObjectMatchesFile "$prefix/source/shards_manifest.jsonl" $shardsPath

$specs = @(Get-Content -LiteralPath $shardsPath | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
if ($specs.Count -ne 70 -or [int]$manifest.total_shards -ne 70) { throw "Attempt03 schedule is incomplete" }
$expectedProfiles = @('random_exact_final','stage19_p0','stage3_baseline','stage7_m5_r10','stage9f_p2')
$seenSeeds = [Collections.Generic.HashSet[long]]::new()
$globalShard = 0
foreach ($logicalSplit in @('train.fit','train.precal_holdout')) {
    $splitPlan = $plan.fresh_splits.$logicalSplit
    $splitShards = [int]$splitPlan.roots / [int]$plan.budget.roots_per_shard
    for ($splitShard = 0; $splitShard -lt $splitShards; $splitShard++) {
        $spec = $specs[$globalShard]
        $expectedSeedStart = [long]$splitPlan.seed_start + ([long]$splitShard * 10L * [long]$splitPlan.seed_stride)
        $quotaNames = @($spec.profile_quota_per_shard.PSObject.Properties.Name | Sort-Object)
        $quotaValues = @($spec.profile_quota_per_shard.PSObject.Properties.Value)
        if ($spec.schema -ne 'hu_m43_attempt03_teacher_shard_v1' -or
            [int]$spec.shard -ne $globalShard -or [int]$spec.split_shard -ne $splitShard -or
            [string]$spec.logical_split -ne $logicalSplit -or [string]$spec.split -ne [string]$splitPlan.record_split -or
            [int]$spec.roots -ne 10 -or [long]$spec.seed_start -ne $expectedSeedStart -or
            [long]$spec.seed_stride -ne [long]$splitPlan.seed_stride -or
            [long]$spec.candidate_seed -ne ([long]$splitPlan.candidate_seed_start + [long]$splitShard * [long]$splitPlan.seed_stride) -or
            [long]$spec.evaluation_seed -ne ([long]$splitPlan.evaluation_seed_start + [long]$splitShard * [long]$splitPlan.seed_stride) -or
            [long]$spec.child_policy_seed -ne ([long]$splitPlan.child_policy_seed_start + [long]$splitShard * [long]$splitPlan.seed_stride) -or
            [int]$spec.candidate_samples -ne 2 -or [int]$spec.evaluation_samples -ne 64 -or
            ($quotaNames -join ',') -ne ($expectedProfiles -join ',') -or
            @($quotaValues | Where-Object { [int]$_ -ne 2 }).Count -ne 0) {
            throw "Attempt03 schedule mapping changed at shard $globalShard"
        }
        for ($root = 0; $root -lt 10; $root++) {
            $seed = [long]$spec.seed_start + [long]$root * [long]$spec.seed_stride
            if (-not $seenSeeds.Add($seed)) { throw "Attempt03 schedule contains a duplicate hand seed: $seed" }
        }
        $globalShard++
    }
}
if ($globalShard -ne 70) { throw "Attempt03 schedule split mapping is incomplete" }

$specByPrefix = @{}
foreach ($spec in $specs) {
    $key = [string]$spec.output_prefix
    if ($specByPrefix.ContainsKey($key)) { throw "Attempt03 schedule contains a duplicate output prefix: $key" }
    $specByPrefix[$key] = $spec
}

$manifestSha256 = Get-Sha256 $manifestPath
$doneByShard = @{}
foreach ($uri in @(Get-GcsUris "$prefix/results/**/DONE.json")) {
    if ($uri -notmatch '/results/([^/]+)/DONE[.]json$') { throw "Attempt03 DONE URI is not canonical: $uri" }
    $outputPrefix = [string]$Matches[1]
    if (-not $specByPrefix.ContainsKey($outputPrefix)) { throw "Attempt03 DONE does not map to the frozen schedule: $uri" }
    $spec = $specByPrefix[$outputPrefix]
    if ($doneByShard.ContainsKey([int]$spec.shard)) { throw "Duplicate Attempt03 DONE for shard $($spec.shard)" }
    try { $done = (Get-GcsText $uri) | ConvertFrom-Json } catch { throw "Attempt03 DONE JSON is invalid: $uri" }
    if ($done.schema -ne 'hu_m43_attempt02_teacher_done_v1' -or $done.status -ne 'complete' -or
        [string]$done.run_name -ne $RunName -or [int]$done.shard -ne [int]$spec.shard -or
        [string]$done.split -ne [string]$spec.split -or [int]$done.roots -ne [int]$spec.roots -or
        [string]$done.output_prefix -ne $outputPrefix -or [string]$done.manifest_sha256 -ne $manifestSha256 -or
        [string]$done.shards_manifest_sha256 -ne [string]$manifest.schedule_sha256 -or
        [string]$done.source_sha256 -ne [string]$manifest.source_sha256 -or
        [string]$done.startup_sha256 -ne [string]$manifest.startup_sha256 -or
        [string]$done.model_manifest_sha256 -ne [string]$manifest.model_manifest_sha256 -or
        [string]$done.native_manifest_sha256 -ne [string]$manifest.native_manifest_sha256) {
        throw "Attempt03 DONE does not match the frozen closure: $uri"
    }
    $doneByShard[[int]$spec.shard] = $done
}

$statusByShard = @{}
$statusText = Get-GcsText "$prefix/status/*.json" -AllowMissing
if ($null -ne $statusText) {
    foreach ($line in @($statusText -split "`r?`n" | Where-Object { $_.Trim() })) {
        try { $row = $line | ConvertFrom-Json } catch { throw "Attempt03 status JSON is invalid" }
        $shard = [int]$row.shard
        if ($shard -lt 0 -or $shard -ge 70 -or $statusByShard.ContainsKey($shard)) { throw "Attempt03 status shard identity is invalid: $shard" }
        $spec = $specs[$shard]
        if ($row.schema -ne 'hu_m43_attempt02_teacher_status_v1' -or [string]$row.run_name -ne $RunName -or
            [string]$row.split -ne [string]$spec.split -or [string]$row.output_prefix -ne [string]$spec.output_prefix -or
            [string]$row.status -notin @('running','failed','complete') -or
            [int]$row.completed_roots -lt 0 -or [int]$row.completed_roots -gt [int]$spec.roots) {
            throw "Attempt03 status does not match the frozen schedule: shard=$shard"
        }
        $statusByShard[$shard] = $row
    }
}

$vmPrefix = Convert-ToVmPrefix $RunName
$vmResult = Invoke-GcloudBounded @('compute','instances','list','--project',$ProjectId,'--filter',("name~'^$vmPrefix-[0-9]{3}$'"),'--format=json') 45
$vms = if ($vmResult.stdout.Trim()) { @($vmResult.stdout | ConvertFrom-Json) } else { @() }
$activeVmByShard = @{}
$activeVms = @()
$nonActiveVms = @()
$vmPattern = '^' + [regex]::Escape($vmPrefix) + '-([0-9]{3})$'
foreach ($vm in $vms) {
    if ([string]$vm.name -notmatch $vmPattern) { throw "Attempt03 VM name does not map to the frozen run" }
    $shard = [int]$Matches[1]
    $zone = ([string]$vm.zone -split '/')[-1]
    $machineType = ([string]$vm.machineType -split '/')[-1]
    if ($shard -lt 0 -or $shard -ge 70 -or $zone -ne [string]$manifest.zone -or $machineType -ne [string]$manifest.machine_type) {
        throw "Attempt03 VM does not match the frozen compute closure: $($vm.name)"
    }
    $projection = [pscustomobject]@{name=$vm.name;zone=$zone;status=$vm.status;machine_type=$machineType;shard=$shard}
    if ([string]$vm.status -in @('PROVISIONING','STAGING','RUNNING')) {
        if ($activeVmByShard.ContainsKey($shard)) { throw "Duplicate active Attempt03 VM for shard $shard" }
        $activeVmByShard[$shard] = $vm
        $activeVms += $projection
    }
    else { $nonActiveVms += $projection }
}

$rows = foreach ($spec in $specs) {
    $shard = [int]$spec.shard
    $hasDone = $doneByShard.ContainsKey($shard)
    $hasStatus = $statusByShard.ContainsKey($shard)
    $hasActiveVm = $activeVmByShard.ContainsKey($shard)
    $workerStatus = if ($hasStatus) { [string]$statusByShard[$shard].status } else { $null }
    $state = Resolve-ShardState $hasDone $workerStatus $hasActiveVm
    [pscustomobject]@{
        shard=$shard; logical_split=[string]$spec.logical_split; state=$state
        completed_roots=$(if ($hasStatus) { [int]$statusByShard[$shard].completed_roots } elseif ($hasDone) { [int]$spec.roots } else { 0 })
        output_prefix=[string]$spec.output_prefix
    }
}

$counts = @{}
foreach ($state in @('complete','running','starting','failed','interrupted','inconsistent','not_started')) {
    $counts[$state] = @($rows | Where-Object state -eq $state).Count
}
if (($counts.Values | Measure-Object -Sum).Sum -ne 70) { throw "Attempt03 status classification is not exhaustive" }

[pscustomobject]@{
    schema='hu_m43_attempt03_teacher_spot_status_v2';run_name=$RunName
    total_shards=70;complete=$counts.complete;running=$counts.running;starting=$counts.starting
    failed=$counts.failed;interrupted=$counts.interrupted;inconsistent=$counts.inconsistent;not_started=$counts.not_started
    train_fit_complete=@($rows | Where-Object { $_.logical_split -eq 'train.fit' -and $_.state -eq 'complete' }).Count
    precal_holdout_complete=@($rows | Where-Object { $_.logical_split -eq 'train.precal_holdout' -and $_.state -eq 'complete' }).Count
    authoritative_complete_from_verified_done_only=$true
    frozen_manifest_sha256=$manifestSha256
    active_vms=$activeVms;non_active_vms=$nonActiveVms
    shards=$rows;current_profile_mutated=$false;runtime_policy_activated=$false
} | ConvertTo-Json -Depth 8
