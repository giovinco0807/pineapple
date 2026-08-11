param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-m43-attempt02-v4-model-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [Parameter(Mandatory = $true)][string[]]$Train,
    [int[]]$StartShards = @(0),
    [string]$MachineType = "c4-highcpu-4",
    [string[]]$FallbackMachineTypes = @("c4-standard-4", "n2-highcpu-4"),
    [string[]]$Zones = @("asia-northeast1-b", "asia-northeast1-c", "us-central1-a"),
    [int]$BootDiskGb = 30,
    [switch]$CreateInstances,
    [switch]$ResumeExisting,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ExpectedTrainShards = 20
$ExpectedJobs = 30
$JobsPerShard = 4
$ExpectedShards = 8
$ModelId = "hu-m43-t1-v4-attempt02"
$Iterations = 150
$MaxLeafNodes = 31
$LearningRate = 0.05
$PairedSeFloor = 0.50
$HuberAlpha = 0.90

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-TextSha256 {
    param([Parameter(Mandatory = $true)][string]$Text)
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Text)
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally { $sha.Dispose() }
}

function Write-BytesCreateNew {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][byte[]]$Bytes)
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $stream.Write($Bytes, 0, $Bytes.Length) }
    finally { $stream.Dispose() }
}

function Write-Utf8CreateNew {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Text)
    Write-BytesCreateNew -Path $Path -Bytes ([Text.UTF8Encoding]::new($false).GetBytes($Text))
}

function Resolve-InputPath {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$RepoRoot)
    $candidate = if ([IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $RepoRoot $Path }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { throw "Required fresh train shard is missing: $candidate" }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Convert-ToNativeProcessArgument {
    param([AllowEmptyString()][Parameter(Mandatory = $true)][string]$Value)
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
    $builder = [Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq [char]'\') { $slashes += 1; continue }
        if ($character -eq [char]'"') {
            if ($slashes -gt 0) { [void]$builder.Append([char]'\', (2 * $slashes)) }
            [void]$builder.Append('\"'); $slashes = 0; continue
        }
        if ($slashes -gt 0) { [void]$builder.Append([char]'\', $slashes); $slashes = 0 }
        [void]$builder.Append($character)
    }
    if ($slashes -gt 0) { [void]$builder.Append([char]'\', (2 * $slashes)) }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Stop-ProcessTreeBounded {
    param([Parameter(Mandatory = $true)][Diagnostics.Process]$Process, [ValidateRange(1, 10)][int]$TimeoutSeconds = 5)
    if ($Process.HasExited) { return $true }
    if ($env:OS -eq 'Windows_NT') {
        $taskkillPath = Join-Path $env:SystemRoot 'System32\taskkill.exe'
        $info = [Diagnostics.ProcessStartInfo]::new()
        $info.FileName = $taskkillPath
        $info.Arguments = "/PID $($Process.Id) /T /F"
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $killer = [Diagnostics.Process]::new(); $killer.StartInfo = $info
        try {
            if ($killer.Start()) {
                $killerOut = $killer.StandardOutput.ReadToEndAsync(); $killerErr = $killer.StandardError.ReadToEndAsync()
                if (-not $killer.WaitForExit($TimeoutSeconds * 1000)) {
                    try { $killer.Kill() } catch { }
                }
                else {
                    $killer.WaitForExit(); [void]$killerOut.GetAwaiter().GetResult(); [void]$killerErr.GetAwaiter().GetResult()
                }
            }
        }
        finally { $killer.Dispose() }
    }
    else {
        try { $Process.Kill($true) } catch { try { $Process.Kill() } catch { } }
    }
    try { [void]$Process.WaitForExit($TimeoutSeconds * 1000) } catch { }
    try { return [bool]$Process.HasExited } catch { return $false }
}

function Invoke-ProcessBounded {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [ValidateRange(1, 120)][int]$TimeoutSeconds
    )
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $FilePath
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    if ($null -ne $info.PSObject.Properties['ArgumentList']) {
        foreach ($argument in $Arguments) { [void]$info.ArgumentList.Add([string]$argument) }
    }
    else {
        $info.Arguments = (@($Arguments | ForEach-Object { Convert-ToNativeProcessArgument ([string]$_) }) -join ' ')
    }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    try {
        if (-not $process.Start()) { throw "Unable to start bounded process: $FilePath" }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            $treeTerminated = Stop-ProcessTreeBounded -Process $process -TimeoutSeconds 5
            $stdout = ''; $stderr = ''
            if ($treeTerminated) {
                try { if ($stdoutTask.Wait(2000)) { $stdout = [string]$stdoutTask.GetAwaiter().GetResult() } } catch { }
                try { if ($stderrTask.Wait(2000)) { $stderr = [string]$stderrTask.GetAwaiter().GetResult() } } catch { }
            }
            $stdoutLines = @($stdout -split "`r?`n" | Where-Object { $_ -ne '' })
            $stderrLines = @($stderr -split "`r?`n" | Where-Object { $_ -ne '' })
            return [pscustomobject][ordered]@{
                timed_out = $true; exit_code = $null; process_tree_terminated = [bool]$treeTerminated
                stdout = $stdoutLines; stderr = $stderrLines
                output = @("process timed out after $TimeoutSeconds seconds; process_tree_terminated=$treeTerminated") + @($stdoutLines) + @($stderrLines)
            }
        }
        $process.WaitForExit()
        $stdout = [string]$stdoutTask.GetAwaiter().GetResult()
        $stderr = [string]$stderrTask.GetAwaiter().GetResult()
        $stdoutLines = @($stdout -split "`r?`n" | Where-Object { $_ -ne '' })
        $stderrLines = @($stderr -split "`r?`n" | Where-Object { $_ -ne '' })
        return [pscustomobject][ordered]@{
            timed_out = $false; exit_code = [int]$process.ExitCode; process_tree_terminated = $null
            stdout = $stdoutLines; stderr = $stderrLines; output = @($stdoutLines) + @($stderrLines)
        }
    }
    finally { $process.Dispose() }
}

function Invoke-GcloudProcessBounded {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [ValidateRange(1, 120)][int]$TimeoutSeconds = 30
    )
    $command = Get-Command gcloud -ErrorAction Stop | Select-Object -First 1
    $source = [string]$command.Source
    if (-not $source) { $source = [string]$command.Definition }
    if (-not $source) { throw "Unable to resolve gcloud executable" }
    if ([IO.Path]::GetExtension($source).ToLowerInvariant() -eq '.ps1') {
        $hostPath = [string](Get-Process -Id $PID).Path
        $prefix = @('-NoLogo','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',$source)
        return Invoke-ProcessBounded -FilePath $hostPath -Arguments (@($prefix) + @($Arguments)) -TimeoutSeconds $TimeoutSeconds
    }
    return Invoke-ProcessBounded -FilePath $source -Arguments $Arguments -TimeoutSeconds $TimeoutSeconds
}

function Invoke-Gcloud {
    param([Parameter(Mandatory = $true)][string[]]$Arguments, [ValidateRange(1, 120)][int]$TimeoutSeconds = 30)
    $result = Invoke-GcloudProcessBounded -Arguments $Arguments -TimeoutSeconds $TimeoutSeconds
    if ($result.timed_out) { throw "gcloud process timed out after $TimeoutSeconds seconds: gcloud $($Arguments -join ' ')" }
    if ($result.exit_code -ne 0) { throw "gcloud failed ($($result.exit_code)): gcloud $($Arguments -join ' ')`n$(@($result.output) -join [Environment]::NewLine)" }
    return @($result.output)
}

function Test-GcsObject {
    param([Parameter(Mandatory = $true)][string]$Uri)
    $result = Invoke-GcloudProcessBounded -Arguments @('storage','objects','describe',$Uri,'--format=json') -TimeoutSeconds 15
    if ($result.timed_out) { throw "gcloud object lookup timed out: $Uri" }
    if ($result.exit_code -eq 0) { return $true }
    if ((@($result.output) -join "`n") -match '(?i)not found|does not exist|No URLs matched|matched no objects|404') { return $false }
    throw "Unable to determine immutable object state: $Uri"
}

function Get-GcsJson {
    param([Parameter(Mandatory = $true)][string]$Uri)
    $result = Invoke-GcloudProcessBounded -Arguments @('storage','cat',$Uri) -TimeoutSeconds 15
    if ($result.timed_out -or $result.exit_code -ne 0) { throw "Unable to read immutable JSON within the process deadline: $Uri" }
    return (@($result.stdout) -join "`n") | ConvertFrom-Json
}

function Get-InstancesByName {
    param([Parameter(Mandatory = $true)][string]$Name)
    $raw = Invoke-Gcloud @("compute", "instances", "list", "--project", $ProjectId, "--filter", ("name={0}" -f $Name), "--format=json")
    if (-not $raw) { return @() }
    return @(((@($raw) -join "`n") | ConvertFrom-Json))
}

function Get-InstanceInZone {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Zone,
        [ValidateRange(1, 30)][int]$TimeoutSeconds = 10
    )
    $result=Invoke-GcloudProcessBounded -Arguments @('compute','instances','describe',$Name,'--project',$ProjectId,'--zone',$Zone,'--format=json','--quiet') -TimeoutSeconds $TimeoutSeconds
    if($result.timed_out){return $null}
    if($result.exit_code -eq 0){
        try{return ((@($result.stdout)-join"`n")|ConvertFrom-Json)}
        catch{throw "Created instance describe returned invalid JSON for $Name in ${Zone}: $(@($result.output)-join[Environment]::NewLine)"}
    }
    $message=@($result.output)-join[Environment]::NewLine
    if($message -match '(?i)not found|was not found|404'){return $null}
    throw "Unable to verify created instance $Name in ${Zone}: $message"
}

function Wait-AsyncInstanceCreate {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Zone,
        [Parameter(Mandatory = $true)][string]$Machine,
        [Parameter(Mandatory = $true)][string]$OperationName,
        [int]$TimeoutSeconds = 60
    )
    if($OperationName -notmatch '^operation-[A-Za-z0-9-]+$'){throw "Async create operation name is invalid"}
    if($TimeoutSeconds -lt 10 -or $TimeoutSeconds -gt 120){throw "Async create verification timeout is outside 10..120 seconds"}
    $started=[Diagnostics.Stopwatch]::StartNew();$lastStatus=$null;$operation=$null
    while($started.Elapsed.TotalSeconds -lt $TimeoutSeconds){
        $remaining=[Math]::Max(1,[Math]::Ceiling($TimeoutSeconds-$started.Elapsed.TotalSeconds))
        $callTimeout=[Math]::Min(8,$remaining)
        $result=Invoke-GcloudProcessBounded -Arguments @('compute','operations','describe',$OperationName,'--project',$ProjectId,'--zone',$Zone,'--format=json','--quiet') -TimeoutSeconds $callTimeout
        if($result.timed_out){$lastStatus="CLI_TIMEOUT";continue}
        if($result.exit_code -eq 0){
            try{$operation=(@($result.stdout)-join"`n")|ConvertFrom-Json}
            catch{throw "Async create operation returned invalid JSON: $(@($result.output)-join[Environment]::NewLine)"}
            if([string]$operation.name -ne $OperationName){throw "Async create operation identity changed"}
            $lastStatus=[string]$operation.status
            if($lastStatus -notin @("PENDING","RUNNING","DONE")){throw "Async create operation status is invalid: $lastStatus"}
            if($lastStatus -eq "DONE"){
                $errors=@();$errorProperty=$operation.PSObject.Properties['error']
                if($null -ne $errorProperty -and $null -ne $errorProperty.Value){$errors=@($errorProperty.Value.errors|Where-Object{$null -ne $_})}
                if($errors.Count -gt 0){
                    $unexpected=@(Get-InstancesByName $Name)
                    if($unexpected.Count -ne 0){throw "Terminal async create failure left an instance with the requested name"}
                    return [ordered]@{created=$false;terminal_failure=$true;operation_name=$OperationName;operation_status="DONE";instance_status=$null;elapsed_seconds=[Math]::Round($started.Elapsed.TotalSeconds,3);errors=$errors}
                }
                break
            }
        }
        else{
            $message=@($result.output)-join[Environment]::NewLine
            if($message -notmatch '(?i)not found|was not found|404|temporar|timeout|503'){throw "Unable to poll async create operation: $message"}
        }
        Start-Sleep -Seconds 2
    }
    if($null -eq $operation -or $lastStatus -ne "DONE"){
        $observed=Get-InstanceInZone -Name $Name -Zone $Zone
        if($null -eq $observed){throw "Async create outcome remained ambiguous after $TimeoutSeconds seconds; no fallback was attempted"}
        $actualName=[string]$observed.name;$actualZone=([string]$observed.zone).Split('/')[-1];$actualMachine=([string]$observed.machineType).Split('/')[-1];$observedStatus=[string]$observed.status
        if($actualName -ne $Name -or $actualZone -ne $Zone -or $actualMachine -ne $Machine -or $observedStatus -notin @("PROVISIONING","STAGING","RUNNING")){
            throw "Async create timed out with an unexpected instance identity/status; refusing an ambiguous fallback"
        }
        return [ordered]@{created=$true;terminal_failure=$false;operation_name=$OperationName;operation_status=$lastStatus;instance_status=$observedStatus;elapsed_seconds=[Math]::Round($started.Elapsed.TotalSeconds,3);verification="bounded_instance_observed"}
    }
    $instanceDeadline=[Diagnostics.Stopwatch]::StartNew();$instance=$null
    while($instanceDeadline.Elapsed.TotalSeconds -lt 20){
        $remaining=[Math]::Max(1,[Math]::Ceiling(20-$instanceDeadline.Elapsed.TotalSeconds))
        $instance=Get-InstanceInZone -Name $Name -Zone $Zone -TimeoutSeconds ([Math]::Min(8,$remaining))
        if($null -ne $instance){break}
        Start-Sleep -Seconds 2
    }
    if($null -eq $instance){throw "Async create operation succeeded but instance existence was not observed; refusing an ambiguous fallback"}
    $actualName=[string]$instance.name;$actualZone=([string]$instance.zone).Split('/')[-1];$actualMachine=([string]$instance.machineType).Split('/')[-1];$instanceStatus=[string]$instance.status
    if($actualName -ne $Name -or $actualZone -ne $Zone -or $actualMachine -ne $Machine -or $instanceStatus -notin @("PROVISIONING","STAGING","RUNNING")){
        throw "Async create instance identity/status verification failed"
    }
    return [ordered]@{created=$true;terminal_failure=$false;operation_name=$OperationName;operation_status="DONE";instance_status=$instanceStatus;elapsed_seconds=[Math]::Round($started.Elapsed.TotalSeconds,3);verification="operation_done_and_instance_observed"}
}

function Get-StrictInteger {
    param([Parameter(Mandatory = $true)]$Value, [Parameter(Mandatory = $true)][string]$Name)
    $types = @([byte], [sbyte], [int16], [uint16], [int32], [uint32], [int64], [uint64])
    if ($null -eq $Value -or $Value -is [bool] -or $types -notcontains $Value.GetType()) {
        throw "$Name must be a JSON integer, not bool/string/float"
    }
    return [int64]$Value
}

function Get-ObjectPropertyNames {
    param([Parameter(Mandatory = $true)]$Value)
    if ($Value -is [Collections.IDictionary]) { return @($Value.Keys | ForEach-Object { [string]$_ }) }
    return @($Value.PSObject.Properties.Name)
}

function Convert-ToVmPrefix {
    param([Parameter(Mandatory = $true)][string]$Value)
    $name = ($Value.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-')
    if (-not $name) { throw "RunName does not produce a VM prefix" }
    if ($name -notmatch '^[a-z]') { $name = "m43a2-$name" }
    if ($name.Length -gt 53) { $name = $name.Substring(0, 44).TrimEnd('-') + "-" + (Get-TextSha256 $Value).Substring(0, 8) }
    return $name
}

function Assert-SourcePackagePolicy {
    param([Parameter(Mandatory = $true)]$Manifest, [Parameter(Mandatory = $true)][string]$ArchivePath)
    $policy = $Manifest.source_package_policy
    $roots = @("src/ofc_regular/train_hu_m43_attempt02_fold_job.py", "src/ofc_regular/assemble_hu_m43_attempt02_model.py")
    $patterns = @("(?i)l[o]cked", "(?i)\.jsonl?$", "(?i)(^|/)(outputs?|configs?|data)(/|$)")
    if ([string]$policy.mode -ne "ast_recursive_local_import_closure_v1" -or
        (@($policy.roots) -join "|") -ne ($roots -join "|") -or
        (@($policy.forbidden_entry_patterns) -join "|") -ne ($patterns -join "|") -or
        (Get-StrictInteger $policy.module_count "source_package_policy.module_count") -lt 3 -or
        [string]$policy.entries_sha256 -notmatch '^[0-9a-f]{64}$' -or
        $policy.local_input_values_embedded -ne $false -or $policy.nontrain_cloud_inputs -ne 0) {
        throw "Attempt02 frozen source package policy changed"
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        if (@($archive.Entries | Where-Object { $_.FullName -match '\\' }).Count -ne 0) { throw "Frozen source archive contains non-portable entry separators" }
        [string[]]$entries = @($archive.Entries | Where-Object { $_.FullName -match '\.py$' } | ForEach-Object { $_.FullName })
        [Array]::Sort($entries, [StringComparer]::Ordinal)
        if ($entries.Count -ne (Get-StrictInteger $policy.module_count "source_package_policy.module_count") -or
            @($entries | Where-Object { $_ -match '(?i)l[o]cked|\.jsonl?$|(^|/)(outputs?|configs?|data)(/|$)' }).Count -ne 0 -or
            (Get-TextSha256 ('["' + ($entries -join '","') + '"]')) -ne [string]$policy.entries_sha256) {
            throw "Frozen source archive violates Attempt02 package policy"
        }
    }
    finally { $archive.Dispose() }
}

function Assert-Manifest {
    param([Parameter(Mandatory = $true)]$Manifest)
    if ($Manifest.schema -ne "hu_m43_attempt02_v4_fold_spot_run_manifest_v1" -or $Manifest.status -ne "frozen" -or
        $Manifest.run_name -ne $RunName -or $Manifest.project_id -ne $ProjectId -or $Manifest.bucket -ne $Bucket -or
        (Get-StrictInteger $Manifest.job_count "job_count") -ne $ExpectedJobs -or
        (Get-StrictInteger $Manifest.compute.shard_count "compute.shard_count") -ne $ExpectedShards -or
        (Get-StrictInteger $Manifest.compute.jobs_per_shard "compute.jobs_per_shard") -ne $JobsPerShard -or
        $Manifest.compute.max_parallel_processes -ne 4 -or $Manifest.compute.auto_delete -ne $true -or
        $Manifest.current_profile_mutated -ne $false -or $Manifest.runtime_policy_activated -ne $false) {
        throw "Attempt02 frozen run manifest identity/schema is invalid"
    }
    if (@($Manifest.inputs.train).Count -ne $ExpectedTrainShards -or
        @(Get-ObjectPropertyNames $Manifest.inputs | Where-Object { $_ -notin @("train", "input_bundle_sha256") }).Count -ne 0 -or
        [string]$Manifest.source.uri -ne "gs://$Bucket/runs/$RunName/source/ofc_regular_hu_m43_attempt02_v4_source.zip" -or
        [string]$Manifest.cloud_contract.uri -ne "gs://$Bucket/runs/$RunName/inputs/attempt02_fold_cloud_contract.json" -or
        [string]$Manifest.dependencies.numpy -ne "2.2.6" -or [string]$Manifest.dependencies.scikit_learn -ne "1.8.0") {
        throw "Attempt02 source/input/dependency binding is invalid"
    }
    foreach ($name in @("PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")) {
        $expected = if ($name -eq "PYTHONHASHSEED") { "0" } else { "1" }
        if ([string]$Manifest.process_environment.$name -ne $expected) { throw "Attempt02 deterministic environment changed: $name" }
    }
    $jobs = @($Manifest.jobs)
    if ($jobs.Count -ne $ExpectedJobs) { throw "Attempt02 manifest requires exactly 30 jobs" }
    $seen = [Collections.Generic.HashSet[int]]::new()
    for ($i = 0; $i -lt $ExpectedJobs; $i++) {
        $slot = $i % 6; $expectedOuter = [int][Math]::Floor($i / 6)
        $expectedKind = if ($slot -eq 0) { "outer_runtime" } else { "inner_oof_safety" }
        $expectedInner = if ($slot -eq 0) { $null } else { $slot - 1 }
        $job = $jobs[$i]
        $actualInner = if ($null -eq $job.inner_fold) { $null } else { Get-StrictInteger $job.inner_fold "jobs[$i].inner_fold" }
        if (-not $seen.Add([int](Get-StrictInteger $job.job_index "jobs[$i].job_index")) -or
            (Get-StrictInteger $job.job_index "jobs[$i].job_index") -ne $i -or [string]$job.job_kind -ne $expectedKind -or
            (Get-StrictInteger $job.outer_fold "jobs[$i].outer_fold") -ne $expectedOuter -or
            (($null -eq $expectedInner -and $null -ne $actualInner) -or ($null -ne $expectedInner -and $actualInner -ne $expectedInner)) -or
            (Get-StrictInteger $job.shard_index "jobs[$i].shard_index") -ne [int][Math]::Floor($i / $JobsPerShard) -or
            [string]$job.job_spec_sha256 -notmatch '^[0-9a-f]{64}$' -or
            [string]$job.done_uri -ne ("gs://$Bucket/runs/$RunName/results/job-{0:D2}/DONE.json" -f $i)) {
            throw "Attempt02 job mapping changed at index $i"
        }
    }
    for ($i = 0; $i -lt $ExpectedTrainShards; $i++) {
        $entry = $Manifest.inputs.train[$i]
        if ((Get-StrictInteger $entry.index "inputs.train[$i].index") -ne $i -or
            (Get-StrictInteger $entry.rows "inputs.train[$i].rows") -ne 10 -or
            (Get-StrictInteger $entry.bytes "inputs.train[$i].bytes") -lt 1 -or
            [string]$entry.sha256 -notmatch '^[0-9a-f]{64}$') { throw "Attempt02 train binding changed at index $i" }
    }
    if ([string]$Manifest.training_config.schema -ne "hu_m43_attempt02_v4_training_config_v1" -or
        [string]$Manifest.training_config.model_id -ne $ModelId -or
        (Get-StrictInteger $Manifest.training_config.train_states "training_config.train_states") -ne 200 -or
        (Get-StrictInteger $Manifest.training_config.cross_fit_folds "training_config.cross_fit_folds") -ne 5 -or
        (Get-StrictInteger $Manifest.training_config.fold_jobs "training_config.fold_jobs") -ne 30 -or
        $Manifest.training_config.runtime.current_profile_mutated -ne $false -or
        $Manifest.training_config.runtime.policy_activated -ne $false) {
        throw "Attempt02 frozen v4 training config changed"
    }
}

function Assert-CanonicalResumeStartup {
    param([Parameter(Mandatory = $true)][string]$Path)
    $startupText = [IO.File]::ReadAllText($Path, [Text.UTF8Encoding]::new($false, $true))
    $canonicalMarker = 'assert obj["cloud_contract_file_sha256"]==m["cloud_contract"]["file_sha256"]'
    $legacyMarker = 'obj["cloud_contract_sha256"]'
    $canonicalCount = [regex]::Matches($startupText, [regex]::Escape($canonicalMarker)).Count
    if ($canonicalCount -ne 1 -or $startupText.IndexOf($legacyMarker, [StringComparison]::Ordinal) -ge 0) {
        throw "Frozen Attempt02 startup is incompatible with canonical cloud_contract_file_sha256 outputs; create a new RunName instead of resuming"
    }
}

function Assert-FrozenDone {
    param([Parameter(Mandatory = $true)]$Done, [Parameter(Mandatory = $true)]$ExpectedJob,
          [Parameter(Mandatory = $true)][int]$ExpectedIndex, [Parameter(Mandatory = $true)]$Manifest,
          [Parameter(Mandatory = $true)][string]$ManifestSha)
    if ($Done.schema -ne "hu_m43_attempt02_v4_fold_done_v1" -or $Done.status -ne "complete" -or
        $Done.run_name -ne $RunName -or (Get-StrictInteger $Done.job_index "DONE.job_index") -ne $ExpectedIndex -or
        [string]$Done.job_spec_sha256 -ne [string]$ExpectedJob.job_spec_sha256 -or
        [string]$Done.source_sha256 -ne [string]$Manifest.source.sha256 -or
        [string]$Done.run_manifest_sha256 -ne $ManifestSha -or
        [string]$Done.cloud_contract_file_sha256 -ne [string]$Manifest.cloud_contract.file_sha256 -or
        [string]$Done.input_bundle_sha256 -ne [string]$Manifest.inputs.input_bundle_sha256 -or
        [string]$Done.artifact_sha256 -notmatch '^[0-9a-f]{64}$' -or
        [string]$Done.job_manifest_sha256 -notmatch '^[0-9a-f]{64}$' -or
        $Done.current_profile_mutated -ne $false -or $Done.runtime_policy_activated -ne $false) {
        throw "Existing immutable Attempt02 DONE is stale/invalid for job $ExpectedIndex"
    }
}

if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw "RunName is not path-safe" }
if ($BootDiskGb -lt 20) { throw "BootDiskGb must be at least 20" }
if ($Train.Count -ne $ExpectedTrainShards) { throw "Attempt02 model run requires exactly 20 original fresh train shards" }
if (@($Train | Sort-Object -Unique).Count -ne $Train.Count) { throw "Attempt02 train shard paths must be unique" }
if ($ResumeExisting -and -not $CreateInstances) { throw "ResumeExisting requires CreateInstances" }
if ($ResumeExisting -and $DryRun) { throw "ResumeExisting and DryRun are mutually exclusive" }
$uniqueShards = @($StartShards | Sort-Object -Unique)
if ($uniqueShards.Count -ne $StartShards.Count -or @($uniqueShards | Where-Object { $_ -lt 0 -or $_ -ge $ExpectedShards }).Count -ne 0) {
    throw "StartShards must be unique values in 0..7"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedTrain = @($Train | ForEach-Object { Resolve-InputPath $_ $repoRoot })
$runDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
$artifactDir = if ($DryRun) { Join-Path $runDir "dryrun" } else { $runDir }
$manifestPath = Join-Path $artifactDir "m43_attempt02_v4_fold_run_manifest.json"
$startupPath = Join-Path $artifactDir "startup_hu_m43_attempt02_v4_fold_group.sh"
$sourcePath = Join-Path $artifactDir "ofc_regular_hu_m43_attempt02_v4_source.zip"
$cloudContractPath = Join-Path $artifactDir "attempt02_fold_cloud_contract.json"
$dryRunPlanPath = Join-Path $artifactDir "dryrun_plan.json"
$gcsPrefix = "gs://$Bucket/runs/$RunName"
$manifestUri = "$gcsPrefix/source/m43_attempt02_v4_fold_run_manifest.json"
$startupUri = "$gcsPrefix/source/startup_hu_m43_attempt02_v4_fold_group.sh"
$sourceUri = "$gcsPrefix/source/ofc_regular_hu_m43_attempt02_v4_source.zip"
$cloudContractUri = "$gcsPrefix/inputs/attempt02_fold_cloud_contract.json"
$vmPrefix = Convert-ToVmPrefix $RunName

if ($DryRun -and (Test-Path -LiteralPath $runDir)) { throw "DryRun RunName is immutable and already exists: $runDir" }
$remoteExists = $false
if (-not $DryRun) { $remoteExists = Test-GcsObject $manifestUri }
if ($ResumeExisting -and -not $remoteExists) { throw "Frozen Attempt02 model run does not exist: $manifestUri" }
if (-not $ResumeExisting -and $remoteExists) { throw "Attempt02 model run already exists and is immutable; use ResumeExisting" }

if ($ResumeExisting) {
    foreach ($path in @($manifestPath, $startupPath, $sourcePath, $cloudContractPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Resume requires original local frozen artifact: $path" }
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $manifestSha256 = Get-Sha256 $manifestPath
    Assert-Manifest $manifest
    if ((Get-Sha256 $sourcePath) -ne [string]$manifest.source.sha256 -or
        (Get-Sha256 $cloudContractPath) -ne [string]$manifest.cloud_contract.file_sha256 -or
        (Get-Sha256 $startupPath) -ne [string]$manifest.startup.sha256) { throw "Local frozen Attempt02 artifacts changed" }
    Assert-CanonicalResumeStartup -Path $startupPath
    $remoteAudit = Join-Path ([IO.Path]::GetTempPath()) ("m43-attempt02-resume-audit-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $remoteAudit | Out-Null
    try {
        foreach ($binding in @(
            @($manifestUri, $manifestPath, "manifest.json"),
            @($manifest.source.uri, $sourcePath, "source.zip"),
            @($manifest.cloud_contract.uri, $cloudContractPath, "contract.json"),
            @($manifest.startup.uri, $startupPath, "startup.sh")
        )) {
            $download = Join-Path $remoteAudit $binding[2]
            Invoke-Gcloud @("storage", "cp", $binding[0], $download, "--project", $ProjectId) | Out-Null
            if ((Get-Sha256 $download) -ne (Get-Sha256 $binding[1])) { throw "Remote immutable Attempt02 artifact differs from its local frozen copy: $($binding[2])" }
        }
    }
    finally { Remove-Item -LiteralPath $remoteAudit -Recurse -Force -ErrorAction SilentlyContinue }
    for ($i = 0; $i -lt $ExpectedTrainShards; $i++) {
        if ((Get-Sha256 $resolvedTrain[$i]) -ne [string]$manifest.inputs.train[$i].sha256) { throw "Resume fresh train bytes changed at index $i" }
    }
}
else {
    foreach ($path in @($manifestPath, $startupPath, $sourcePath, $cloudContractPath)) {
        if (Test-Path -LiteralPath $path) { throw "Local immutable Attempt02 artifact already exists: $path" }
    }
    $workerPath = Resolve-InputPath "src/ofc_regular/train_hu_m43_attempt02_fold_job.py" $repoRoot
    $assemblerPath = Resolve-InputPath "src/ofc_regular/assemble_hu_m43_attempt02_model.py" $repoRoot
    $temp = Join-Path ([IO.Path]::GetTempPath()) ("m43-attempt02-model-freeze-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $temp | Out-Null
    try {
        $preparedContract = Join-Path $temp "attempt02_fold_cloud_contract.json"
        $prepareArgs = @("-m", "ofc_regular.train_hu_m43_attempt02_fold_job", "prepare")
        foreach ($path in $resolvedTrain) { $prepareArgs += @("--train", $path) }
        $prepareArgs += @(
            "--output-contract", $preparedContract, "--model-id", $ModelId,
            "--iterations", "$Iterations", "--max-leaf-nodes", "$MaxLeafNodes", "--learning-rate", "$LearningRate",
            "--paired-se-floor", "$PairedSeFloor", "--huber-alpha", "$HuberAlpha"
        )
        $oldPythonPath = $env:PYTHONPATH; $env:PYTHONPATH = Join-Path $repoRoot "src"
        $oldPreference = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try { $prepareOutput = @(& python @prepareArgs 2>&1); $prepareCode = $LASTEXITCODE }
        finally { $ErrorActionPreference = $oldPreference; $env:PYTHONPATH = $oldPythonPath }
        if ($prepareCode -ne 0) { throw "Attempt02 train-only cloud contract preparation failed: $($prepareOutput -join [Environment]::NewLine)" }

        $projectionValidation = @'
import hashlib,json,pathlib,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); paths=[pathlib.Path(x) for x in sys.argv[2:]]
h=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
unsigned=dict(p); declared=unsigned.pop("contract_sha256")
canonical=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
assert declared==canonical(unsigned)
assert p["schema"]=="hu_m43_attempt02_v4_fold_cloud_contract_v1" and p["status"]=="frozen_train_only"
assert set(p["inputs"])=={"train"} and len(p["inputs"]["train"])==len(paths)==20
assert p["dependencies"]=={"numpy":"2.2.6","scikit_learn":"1.8.0"}
assert p["process_environment"]=={"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"}
assert p["worker_input_boundary"]=={"fresh_train_only":True,"train_rows":200,"nontrain_input_count":0,"sealed_contract_content_count":0}
for i,(entry,path) in enumerate(zip(p["inputs"]["train"],paths)):
 assert entry["index"]==i and entry["rows"]==10 and entry["sha256"]==h(path) and entry["bytes"]==path.stat().st_size
f=p["fold_plan"]; assert f["outer_folds"]==5 and f["inner_folds_per_outer"]==5 and f["total_jobs"]==30 and len(f["jobs"])==30
for i,j in enumerate(f["jobs"]):
 assert j["job_index"]==i and j["outer_fold"]==i//6 and j["kind"]==("outer_runtime" if i%6==0 else "inner_oof_safety")
 assert j["inner_fold"]==(None if i%6==0 else i%6-1) and len(j["job_spec_sha256"])==64
forbidden={"calibration","calibration_path","calibration_paths","data_contract","data_contract_path","inherited_locked","locked_holdout","locked_holdout_path"}
def reject(value):
 if isinstance(value,dict):
  for key,child in value.items(): assert str(key).lower() not in forbidden; reject(child)
 elif isinstance(value,list):
  for child in value: reject(child)
reject(p)
print(json.dumps({"file_sha256":h(pathlib.Path(sys.argv[1])),"contract_sha256":declared,"input_bundle_sha256":p["input_bundle_sha256"],"fold_plan_sha256":f["fold_plan_sha256"],"training_config_sha256":p["training_config_sha256"],"train_identity_sha256":f["train_identity_sha256"]}))
'@
        $oldPreference = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try { $summaryRaw = @($projectionValidation | & python - $preparedContract @resolvedTrain 2>&1); $summaryCode = $LASTEXITCODE }
        finally { $ErrorActionPreference = $oldPreference }
        if ($summaryCode -ne 0) { throw "Attempt02 redacted projection validation failed: $($summaryRaw -join [Environment]::NewLine)" }
        $projectionSummary = (@($summaryRaw) -join "`n") | ConvertFrom-Json

        $packageRoot = Join-Path $temp "package"
        $sourceClosureBuilder = @'
import ast,hashlib,json,pathlib,shutil,sys
repo=pathlib.Path(sys.argv[1]).resolve(); destination=pathlib.Path(sys.argv[2]).resolve(); source=repo/"src"/"ofc_regular"
queue=["ofc_regular","ofc_regular.train_hu_m43_attempt02_fold_job","ofc_regular.assemble_hu_m43_attempt02_model"]; seen=set(); selected=[]
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
print(json.dumps({"entries":entries,"entries_sha256":digest},sort_keys=True))
'@
        $oldPreference = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try { $closureRaw = @($sourceClosureBuilder | & python - $repoRoot $packageRoot 2>&1); $closureCode = $LASTEXITCODE }
        finally { $ErrorActionPreference = $oldPreference }
        if ($closureCode -ne 0) { throw "Attempt02 AST source closure failed: $($closureRaw -join [Environment]::NewLine)" }
        $closure = (@($closureRaw) -join "`n") | ConvertFrom-Json
        $sourceEntries = @($closure.entries)
        if ("src/ofc_regular/train_hu_m43_attempt02_fold_job.py" -notin $sourceEntries -or
            "src/ofc_regular/assemble_hu_m43_attempt02_model.py" -notin $sourceEntries) { throw "Attempt02 source closure omitted an entrypoint" }

        $tempSource = Join-Path $temp "source.zip"
        $zipBuilder = @'
import pathlib,sys,zipfile
source=pathlib.Path(sys.argv[1]); destination=pathlib.Path(sys.argv[2])
with zipfile.ZipFile(destination,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
 for path in sorted(source.rglob("*.py")):
  arcname=path.relative_to(source).as_posix()
  if "\\" in arcname: raise ValueError("non-portable archive separator")
  archive.write(path,arcname)
'@
        $oldPreference = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try { $zipOutput = @($zipBuilder | & python - $packageRoot $tempSource 2>&1); $zipCode = $LASTEXITCODE }
        finally { $ErrorActionPreference = $oldPreference }
        if ($zipCode -ne 0) { throw "Attempt02 portable source archive creation failed: $($zipOutput -join [Environment]::NewLine)" }

        $projection = Get-Content -LiteralPath $preparedContract -Raw | ConvertFrom-Json
        $inputPlan = @()
        for ($i = 0; $i -lt $ExpectedTrainShards; $i++) {
            $entry = $projection.inputs.train[$i]
            $inputPlan += [ordered]@{
                index = $i; bytes = Get-StrictInteger $entry.bytes "projection.inputs.train[$i].bytes"
                rows = Get-StrictInteger $entry.rows "projection.inputs.train[$i].rows"; sha256 = [string]$entry.sha256
                uri = ("$gcsPrefix/inputs/train-{0:D3}.jsonl" -f $i)
            }
        }

        $startup = @'
#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=/var/log/hu_m43_attempt02_v4_fold.log
exec > >(tee -a "$LOG") 2>&1
META=http://metadata.google.internal/computeMetadata/v1
meta(){ curl -fsS -H 'Metadata-Flavor: Google' "$META/instance/attributes/$1"; }
project_meta(){ curl -fsS -H 'Metadata-Flavor: Google' "$META/project/project-id"; }
instance_meta(){ curl -fsS -H 'Metadata-Flavor: Google' "$META/instance/$1"; }
token(){ curl -fsS -H 'Metadata-Flavor: Google' "$META/instance/service-accounts/default/token" | python3 -c 'import json,sys;print(json.load(sys.stdin)["access_token"])'; }
urlencode(){ python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "$1"; }
PROJECT_ID="$(project_meta)"; BUCKET="$(meta BUCKET)"; RUN_NAME="$(meta RUN_NAME)"
MANIFEST_OBJECT="$(meta MANIFEST_OBJECT)"; MANIFEST_SHA256="$(meta MANIFEST_SHA256)"; JOB_INDICES="$(meta JOB_INDICES)"
INSTANCE_NAME="$(instance_meta name)"; ZONE="$(instance_meta zone)"; ZONE="${ZONE##*/}"; PREFIX="runs/${RUN_NAME}"; WORK=/work/hu_m43_attempt02_v4
mkdir -p "$WORK"
gcs_download(){ local o="$1" d="$2" e a; e="$(urlencode "$o")"; a="$(token)"; curl --retry 5 --retry-all-errors -fsSL -H "Authorization: Bearer ${a}" "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/${e}?alt=media" -o "$d"; }
gcs_upload(){ local s="$1" o="$2" e a; [[ -f "$s" ]] || return 1; e="$(urlencode "$o")"; a="$(token)"; curl --retry 5 --retry-all-errors -fsS -X POST -H "Authorization: Bearer ${a}" -H 'Content-Type: application/octet-stream' --data-binary "@${s}" "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&name=${e}" >/dev/null; }
gcs_upload_immutable(){ local s="$1" o="$2" e a; [[ -f "$s" ]] || return 1; e="$(urlencode "$o")"; a="$(token)"; curl --retry 5 --retry-all-errors -fsS -X POST -H "Authorization: Bearer ${a}" -H 'Content-Type: application/octet-stream' --data-binary "@${s}" "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&name=${e}&ifGenerationMatch=0" >/dev/null; }
self_delete(){ local a; a="$(token)" || return 0; curl -fsS -X DELETE -H "Authorization: Bearer ${a}" "https://compute.googleapis.com/compute/v1/projects/${PROJECT_ID}/zones/${ZONE}/instances/${INSTANCE_NAME}" >/dev/null || true; }
trap 'self_delete' EXIT
apt-get update -y
apt-get install -y unzip curl ca-certificates python3 python3-venv python3-pip
gcs_download "$MANIFEST_OBJECT" "$WORK/run_manifest.json"
echo "${MANIFEST_SHA256}  $WORK/run_manifest.json" | sha256sum -c -
eval "$(python3 - "$WORK/run_manifest.json" <<'PY'
import json,shlex,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
assert m["schema"]=="hu_m43_attempt02_v4_fold_spot_run_manifest_v1" and m["status"]=="frozen" and m["job_count"]==30
assert set(m["inputs"])=={"train","input_bundle_sha256"} and len(m["inputs"]["train"])==20
assert m["dependencies"]=={"numpy":"2.2.6","scikit_learn":"1.8.0"}
assert m["process_environment"]=={"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"}
payload=json.dumps({"inputs":m["inputs"],"cloud_contract":m["cloud_contract"]},sort_keys=True).lower()
for forbidden in ("calibration","data_contract","inherited_locked","locked_holdout"): assert forbidden not in payload
for key,value in {"SOURCE_OBJECT":m["source"]["uri"].split("/",3)[3],"SOURCE_SHA256":m["source"]["sha256"],"CONTRACT_OBJECT":m["cloud_contract"]["uri"].split("/",3)[3],"CONTRACT_SHA256":m["cloud_contract"]["file_sha256"],"INPUT_BUNDLE_SHA256":m["inputs"]["input_bundle_sha256"]}.items():
 print(f"{key}={shlex.quote(str(value))}")
PY
)"
gcs_download "$SOURCE_OBJECT" "$WORK/source.zip"; echo "${SOURCE_SHA256}  $WORK/source.zip" | sha256sum -c -
gcs_download "$CONTRACT_OBJECT" "$WORK/fold_cloud_contract.json"; echo "${CONTRACT_SHA256}  $WORK/fold_cloud_contract.json" | sha256sum -c -
mkdir -p "$WORK/repo" "$WORK/inputs/train"; unzip -q "$WORK/source.zip" -d "$WORK/repo"
python3 - "$WORK/run_manifest.json" "$WORK/input_downloads.tsv" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
with open(sys.argv[2],"w",encoding="utf-8") as f:
 for e in m["inputs"]["train"]: f.write(f'{e["uri"].split("/",3)[3]}\t{e["index"]:03d}.jsonl\t{e["sha256"]}\n')
PY
while IFS=$'\t' read -r object relative sha; do gcs_download "$object" "$WORK/inputs/train/$relative"; echo "$sha  $WORK/inputs/train/$relative" | sha256sum -c -; done < "$WORK/input_downloads.tsv"
cd "$WORK/repo"; python3 -m venv .venv; source .venv/bin/activate; python -m pip install --upgrade pip; python -m pip install 'numpy==2.2.6' 'scikit-learn==1.8.0'
run_job(){
 local job="$1" out="$WORK/job-$(printf '%02d' "$job")" prefix="$PREFIX/results/job-$(printf '%02d' "$job")" heartbeat_pid=""
 mkdir -p "$out"
 write_status(){ python3 - "$out/status.json" "$RUN_NAME" "$job" "$1" "$MANIFEST_SHA256" "$SOURCE_SHA256" <<'PY'
import datetime,json,sys
p,run,job,state,manifest,source=sys.argv[1:]
json.dump({"schema":"hu_m43_attempt02_v4_fold_job_status_v1","run_name":run,"job_index":int(job),"state":state,"run_manifest_sha256":manifest,"source_sha256":source,"current_profile_mutated":False,"runtime_policy_activated":False,"updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat()},open(p,"w",encoding="utf-8"),sort_keys=True)
PY
  gcs_upload "$out/status.json" "$prefix/status.json"; }
 write_status running
 (while true; do sleep 60; python3 - "$out/heartbeat.json" "$RUN_NAME" "$job" "$MANIFEST_SHA256" "$SOURCE_SHA256" <<'PY'
import datetime,json,sys
p,run,job,manifest,source=sys.argv[1:]
json.dump({"schema":"hu_m43_attempt02_v4_fold_job_heartbeat_v1","run_name":run,"job_index":int(job),"state":"running","run_manifest_sha256":manifest,"source_sha256":source,"current_profile_mutated":False,"runtime_policy_activated":False,"updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat()},open(p,"w",encoding="utf-8"),sort_keys=True)
PY
  gcs_upload "$out/heartbeat.json" "$prefix/heartbeat.json" || true; done) & heartbeat_pid=$!
 local train_args=() path
 for path in "$WORK"/inputs/train/*.jsonl; do train_args+=(--train "$path"); done
 eval "$(python3 - "$WORK/run_manifest.json" "$job" <<'PY'
import json,shlex,sys
m=json.load(open(sys.argv[1],encoding="utf-8")); j=m["jobs"][int(sys.argv[2])]
print("JOB_SPEC_SHA256="+shlex.quote(j["job_spec_sha256"]))
PY
)"
 set +e
 PYTHONPATH=src python -m ofc_regular.train_hu_m43_attempt02_fold_job run --job-index "$job" "${train_args[@]}" --fold-cloud-contract "$WORK/fold_cloud_contract.json" --output-dir "$out" --run-name "$RUN_NAME" --source-sha256 "$SOURCE_SHA256" --run-manifest-sha256 "$MANIFEST_SHA256" --input-bundle-sha256 "$INPUT_BUNDLE_SHA256" --job-spec-sha256 "$JOB_SPEC_SHA256" >"$out/stdout.log" 2>"$out/stderr.log"
 local code=$?; set -e; kill "$heartbeat_pid" >/dev/null 2>&1 || true; wait "$heartbeat_pid" >/dev/null 2>&1 || true
 if [[ $code -ne 0 ]]; then write_status failed || true; gcs_upload "$out/stdout.log" "$prefix/stdout.log" || true; gcs_upload "$out/stderr.log" "$prefix/stderr.log" || true; return "$code"; fi
 python3 - "$WORK/run_manifest.json" "$job" "$out" "$MANIFEST_SHA256" <<'PY'
import hashlib,json,os,sys
mp,raw,out,manifest_sha=sys.argv[1:]; i=int(raw); m=json.load(open(mp,encoding="utf-8")); e=m["jobs"][i]
h=lambda p:hashlib.sha256(open(p,"rb").read()).hexdigest(); jm=json.load(open(os.path.join(out,"job_manifest.json"),encoding="utf-8")); done=json.load(open(os.path.join(out,"DONE.json"),encoding="utf-8"))
for obj,schema,status in ((jm,"hu_m43_attempt02_v4_fold_job_manifest_v1","pass"),(done,"hu_m43_attempt02_v4_fold_done_v1","complete")):
 assert obj["schema"]==schema and obj["status"]==status and obj["run_name"]==m["run_name"] and obj["job_index"]==i
 assert obj["job_spec_sha256"]==e["job_spec_sha256"] and obj["source_sha256"]==m["source"]["sha256"] and obj["run_manifest_sha256"]==manifest_sha
 assert obj["cloud_contract_file_sha256"]==m["cloud_contract"]["file_sha256"] and obj["input_bundle_sha256"]==m["inputs"]["input_bundle_sha256"]
 assert obj["current_profile_mutated"] is False and obj["runtime_policy_activated"] is False
assert h(os.path.join(out,"estimator.pkl"))==jm["artifact_sha256"]==done["artifact_sha256"]
assert h(os.path.join(out,"job_manifest.json"))==done["job_manifest_sha256"]
PY
 gcs_upload "$out/estimator.pkl" "$prefix/estimator.pkl"; gcs_upload "$out/job_manifest.json" "$prefix/job_manifest.json"; gcs_upload "$out/stdout.log" "$prefix/stdout.log"; gcs_upload "$out/stderr.log" "$prefix/stderr.log"
 python3 - "$out/checkpoint.json" "$RUN_NAME" "$job" "$MANIFEST_SHA256" "$SOURCE_SHA256" <<'PY'
import datetime,json,sys
p,run,job,manifest,source=sys.argv[1:]
json.dump({"schema":"hu_m43_attempt02_v4_fold_job_checkpoint_v1","run_name":run,"job_index":int(job),"state":"artifacts_uploaded","run_manifest_sha256":manifest,"source_sha256":source,"current_profile_mutated":False,"runtime_policy_activated":False,"updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat()},open(p,"w",encoding="utf-8"),sort_keys=True)
PY
 gcs_upload "$out/checkpoint.json" "$prefix/checkpoint.json"; write_status complete
 gcs_upload_immutable "$out/DONE.json" "$prefix/DONE.json"
}
IFS='+' read -r -a JOB_ARRAY <<< "$JOB_INDICES"; pids=(); for job in "${JOB_ARRAY[@]}"; do run_job "$job" & pids+=("$!"); done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed -eq 0 ]] || exit 1
echo "M4.3 Attempt02 v4 fold shard complete: $RUN_NAME jobs=$JOB_INDICES"
'@
        $startup = $startup -replace "`r`n", "`n"
        New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
        Write-BytesCreateNew -Path $sourcePath -Bytes ([IO.File]::ReadAllBytes($tempSource))
        Write-BytesCreateNew -Path $cloudContractPath -Bytes ([IO.File]::ReadAllBytes($preparedContract))
        Write-BytesCreateNew -Path $startupPath -Bytes ([Text.UTF8Encoding]::new($false).GetBytes($startup))

        $jobs = @(); $projectionJobs = @($projection.fold_plan.jobs)
        for ($i = 0; $i -lt $ExpectedJobs; $i++) {
            $p = $projectionJobs[$i]; $jobSpec = [ordered]@{}
            foreach ($property in $p.PSObject.Properties) { if ($property.Name -ne "job_spec_sha256") { $jobSpec[$property.Name] = $property.Value } }
            $jobs += [ordered]@{
                job_index = $i; job_kind = [string]$p.kind; outer_fold = Get-StrictInteger $p.outer_fold "projection.outer_fold"
                inner_fold = $p.inner_fold; job_spec = $jobSpec; job_spec_sha256 = [string]$p.job_spec_sha256
                shard_index = [int][Math]::Floor($i / $JobsPerShard); result_prefix = ("$gcsPrefix/results/job-{0:D2}" -f $i)
                done_uri = ("$gcsPrefix/results/job-{0:D2}/DONE.json" -f $i)
            }
        }
        $sourceSha = Get-Sha256 $sourcePath
        $manifest = [ordered]@{
            schema = "hu_m43_attempt02_v4_fold_spot_run_manifest_v1"; status = "frozen"; run_name = $RunName
            project_id = $ProjectId; bucket = $Bucket; job_count = $ExpectedJobs
            source = [ordered]@{
                uri = $sourceUri; bytes = [long](Get-Item $sourcePath).Length; sha256 = $sourceSha
                fold_worker_sha256 = Get-Sha256 $workerPath; assembler_sha256 = Get-Sha256 $assemblerPath
            }
            cloud_contract = [ordered]@{
                uri = $cloudContractUri; bytes = [long](Get-Item $cloudContractPath).Length
                file_sha256 = [string]$projectionSummary.file_sha256; contract_sha256 = [string]$projectionSummary.contract_sha256
                fold_plan_sha256 = [string]$projectionSummary.fold_plan_sha256; train_identity_sha256 = [string]$projectionSummary.train_identity_sha256
            }
            inputs = [ordered]@{ train = $inputPlan; input_bundle_sha256 = [string]$projectionSummary.input_bundle_sha256 }
            jobs = $jobs; training_config = $projection.training_config; training_config_sha256 = [string]$projectionSummary.training_config_sha256
            dependencies = $projection.dependencies; process_environment = $projection.process_environment
            source_package_policy = [ordered]@{
                mode = "ast_recursive_local_import_closure_v1"
                roots = @("src/ofc_regular/train_hu_m43_attempt02_fold_job.py", "src/ofc_regular/assemble_hu_m43_attempt02_model.py")
                module_count = [int]$sourceEntries.Count; entries_sha256 = [string]$closure.entries_sha256
                forbidden_entry_patterns = @("(?i)l[o]cked", "(?i)\.jsonl?$", "(?i)(^|/)(outputs?|configs?|data)(/|$)")
                local_input_values_embedded = $false; nontrain_cloud_inputs = 0
            }
            compute = [ordered]@{
                shard_count = $ExpectedShards; jobs_per_shard = $JobsPerShard; max_parallel_processes = 4
                canary_shard = 0; fanout_shards = @(1,2,3,4,5,6,7); canary_required_before_fanout = $true
                machine_type = $MachineType; fallback_machine_types = $FallbackMachineTypes; zones = $Zones
                boot_disk_gb = $BootDiskGb; spot = $true; termination_action = "DELETE"; auto_delete = $true; vm_prefix = $vmPrefix
            }
            startup = [ordered]@{ uri = $startupUri; sha256 = Get-Sha256 $startupPath }
            cloud_input_boundary = "fresh_train_only"
            current_profile_mutated = $false; runtime_policy_activated = $false; created_at = (Get-Date).ToUniversalTime().ToString("o")
        }
        $manifestText = ($manifest | ConvertTo-Json -Depth 20) + "`n"
        Write-Utf8CreateNew -Path $manifestPath -Text $manifestText
        $manifestSha256 = Get-Sha256 $manifestPath

        $boundaryAuditScript = @'
import json,pathlib,sys,zipfile
contract,startup,manifest,archive,repo,*inputs=sys.argv[1:]
c=json.load(open(contract,encoding="utf-8")); m=json.load(open(manifest,encoding="utf-8"))
assert set(c["inputs"])=={"train"} and set(m["inputs"])=={"train","input_bundle_sha256"}
forbidden={"calibration","calibration_path","calibration_paths","data_contract","data_contract_path","inherited_locked","locked_holdout","locked_holdout_path"}
def reject(value):
 if isinstance(value,dict):
  for key,child in value.items(): assert str(key).lower() not in forbidden; reject(child)
 elif isinstance(value,list):
  for child in value: reject(child)
reject(c); reject(m)
targets=[pathlib.Path(contract).read_text(encoding="utf-8"),pathlib.Path(startup).read_text(encoding="utf-8"),pathlib.Path(manifest).read_text(encoding="utf-8")]
with zipfile.ZipFile(archive) as z:
 names=sorted(z.namelist()); assert all("\\" not in n and n.endswith(".py") for n in names)
 targets += [z.read(n).decode("utf-8") for n in names]
sensitive=[str(pathlib.Path(repo).resolve())]+[str(pathlib.Path(x).resolve()) for x in inputs]
matches=[]
for value in sensitive:
 for index,text in enumerate(targets):
  if value.casefold() in text.casefold(): matches.append({"target":index,"value_length":len(value)})
assert not matches
print(json.dumps({"schema":"hu_m43_attempt02_v4_model_cloud_boundary_audit_v1","status":"pass","train_shards":20,"nontrain_cloud_inputs":0,"source_modules":len(names),"exact_local_path_matches":0,"current_profile_mutated":False,"runtime_policy_activated":False},sort_keys=True))
'@
        $oldPreference = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try { $auditRaw = @($boundaryAuditScript | & python - $cloudContractPath $startupPath $manifestPath $sourcePath $repoRoot @resolvedTrain 2>&1); $auditCode = $LASTEXITCODE }
        finally { $ErrorActionPreference = $oldPreference }
        if ($auditCode -ne 0) { throw "Attempt02 source/cloud boundary audit failed: $($auditRaw -join [Environment]::NewLine)" }
        $boundaryAudit = (@($auditRaw) -join "`n") | ConvertFrom-Json
    }
    finally { Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue }
}

Assert-Manifest $manifest
Assert-SourcePackagePolicy -Manifest $manifest -ArchivePath $sourcePath
$plan = [ordered]@{
    schema = "hu_m43_attempt02_v4_fold_spot_start_plan_v1"; run_name = $RunName; manifest_uri = $manifestUri
    manifest_sha256 = $manifestSha256; job_count = $ExpectedJobs; shard_count = $ExpectedShards; selected_shards = $uniqueShards
    canary_required_before_fanout = $true; create_instances = [bool]$CreateInstances; resume_existing = [bool]$ResumeExisting
    dry_run = [bool]$DryRun; artifact_directory = $artifactDir; cloud_input_boundary = "fresh_train_only"
    current_profile_mutated = $false; runtime_policy_activated = $false
}
if ($DryRun) {
    $plan.boundary_audit = $boundaryAudit
    Write-Utf8CreateNew -Path $dryRunPlanPath -Text (($plan | ConvertTo-Json -Depth 15) + "`n")
    $plan | ConvertTo-Json -Depth 15
    exit 0
}

if (-not $ResumeExisting) {
    for ($i = 0; $i -lt $ExpectedTrainShards; $i++) {
        Invoke-Gcloud @("storage", "cp", $resolvedTrain[$i], $manifest.inputs.train[$i].uri, "--project", $ProjectId, "--if-generation-match=0") | Out-Null
    }
    foreach ($pair in @(@($sourcePath,$sourceUri),@($cloudContractPath,$cloudContractUri),@($startupPath,$startupUri),@($manifestPath,$manifestUri))) {
        Invoke-Gcloud @("storage", "cp", $pair[0], $pair[1], "--project", $ProjectId, "--if-generation-match=0") | Out-Null
    }
}
if (-not $CreateInstances) { $plan.uploaded = $true; $plan | ConvertTo-Json -Depth 15; exit 0 }

if (@($uniqueShards | Where-Object { $_ -gt 0 }).Count -gt 0) {
    foreach ($job in 0..3) {
        if (-not (Test-GcsObject $manifest.jobs[$job].done_uri)) { throw "Canary shard 0 must complete before Attempt02 fanout" }
        Assert-FrozenDone (Get-GcsJson $manifest.jobs[$job].done_uri) $manifest.jobs[$job] $job $manifest $manifestSha256
    }
}

$attempts = [Collections.Generic.List[object]]::new(); $createdShards = @()
foreach ($shard in $uniqueShards) {
    $start = $shard * $JobsPerShard; $end = [Math]::Min($ExpectedJobs - 1, $start + $JobsPerShard - 1); $missing = @()
    for ($job = $start; $job -le $end; $job++) {
        if (Test-GcsObject $manifest.jobs[$job].done_uri) { Assert-FrozenDone (Get-GcsJson $manifest.jobs[$job].done_uri) $manifest.jobs[$job] $job $manifest $manifestSha256 }
        else { $missing += $job }
    }
    if ($missing.Count -eq 0) { continue }
    $vmName = ("{0}-g{1:D2}" -f $vmPrefix, $shard)
    if (@(Get-InstancesByName $vmName).Count -ne 0) { throw "Attempt02 worker already exists for shard ${shard}: $vmName" }
    $metadata = @("BUCKET=$Bucket","RUN_NAME=$RunName","MANIFEST_OBJECT=runs/$RunName/source/m43_attempt02_v4_fold_run_manifest.json","MANIFEST_SHA256=$manifestSha256",("JOB_INDICES=" + ($missing -join "+"))) -join ","
    $created = $false
    foreach ($machine in @($MachineType) + @($FallbackMachineTypes)) {
        $diskType = if ($machine -like "c4-*") { "hyperdisk-balanced" } else { "pd-balanced" }
        foreach ($zone in $Zones) {
            $args = @("compute","instances","create",$vmName,"--project",$ProjectId,"--zone",$zone,"--machine-type",$machine,
                "--provisioning-model","SPOT","--instance-termination-action","DELETE","--maintenance-policy","TERMINATE",
                "--boot-disk-size",("{0}GB" -f $BootDiskGb),"--boot-disk-type",$diskType,"--image-family","ubuntu-2404-lts-amd64",
                "--image-project","ubuntu-os-cloud","--scopes","cloud-platform","--metadata",$metadata,
                "--metadata-from-file",("startup-script={0}" -f $startupPath),"--async","--format=value(name)","--quiet")
            $submission=Invoke-GcloudProcessBounded -Arguments $args -TimeoutSeconds 20
            $output=@($submission.output);$code=$submission.exit_code
            $outputText=@($output)-join[Environment]::NewLine
            if($submission.timed_out){
                $observed=@(Get-InstancesByName $vmName)
                if($observed.Count -eq 1){
                    $actual=$observed[0];$actualZone=([string]$actual.zone).Split('/')[-1];$actualMachine=([string]$actual.machineType).Split('/')[-1];$actualStatus=[string]$actual.status
                    if([string]$actual.name -ne $vmName -or $actualZone -ne $zone -or $actualMachine -ne $machine -or $actualStatus -notin @('PROVISIONING','STAGING','RUNNING')){
                        throw "Timed-out async create left an unexpected instance identity/status; no fallback or retry was attempted"
                    }
                    $attempts.Add([ordered]@{shard_index=$shard;jobs=$missing;machine_type=$machine;zone=$zone;exit_code=$null;submission_timed_out=$true;verification="bounded_instance_observed_after_submission_timeout";output=@($output)})
                    $created=$true;$createdShards+=$shard;break
                }
                throw "Async create submission timed out with instances observed=$($observed.Count); no fallback or retry was attempted. Reconcile $vmName before resuming."
            }
            if ($code -eq 0) {
                $operationNames=@($output|ForEach-Object{([string]$_).Trim()}|Where-Object{$_ -match '^operation-[A-Za-z0-9-]+$'}|Select-Object -Unique)
                if($operationNames.Count -ne 1){
                    $observed=@(Get-InstancesByName $vmName)
                    throw "Async create returned no unique operation identity (instances observed=$($observed.Count)); refusing an ambiguous fallback"
                }
                $verification=Wait-AsyncInstanceCreate -Name $vmName -Zone $zone -Machine $machine -OperationName $operationNames[0] -TimeoutSeconds 60
                $attempts.Add([ordered]@{shard_index=$shard;jobs=$missing;machine_type=$machine;zone=$zone;exit_code=$code;operation_name=$operationNames[0];verification=$verification;output=@($output)})
                if([bool]$verification.created){$created=$true;$createdShards+=$shard;break}
                if(-not [bool]$verification.terminal_failure){throw "Async create returned neither success nor terminal failure; refusing an ambiguous fallback"}
                continue
            }
            if (@(Get-InstancesByName $vmName).Count -ne 0) { throw "Ambiguous async create failure left $vmName present; no fallback was attempted" }
            $terminalSubmissionFailure=$outputText -match '(?i)ZONE_RESOURCE_POOL_EXHAUSTED|RESOURCE_EXHAUSTED|QUOTA_EXCEEDED|does not have enough resources|resource pool exhausted|quota|permission denied|forbidden|invalid (value|argument)|unsupported|was not found|not found|\b(?:400|403|404)\b'
            $attempts.Add([ordered]@{shard_index=$shard;jobs=$missing;machine_type=$machine;zone=$zone;exit_code=$code;terminal_submission_failure=[bool]$terminalSubmissionFailure;output=@($output)})
            if(-not $terminalSubmissionFailure){throw "Async create submission failed ambiguously; no fallback was attempted: $outputText"}
        }
        if ($created) { break }
    }
    if (-not $created) { throw "Unable to create Attempt02 Spot worker for shard $shard" }
}
$plan.uploaded=$true; $plan.created_shards=$createdShards; $plan.attempts=$attempts
$plan | ConvertTo-Json -Depth 15
