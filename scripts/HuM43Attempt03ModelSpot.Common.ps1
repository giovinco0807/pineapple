Set-StrictMode -Version Latest

function Get-M43A3Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-M43A3TextSha256 {
    param([Parameter(Mandatory = $true)][string]$Text)
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Text)
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally { $sha.Dispose() }
}

function Resolve-M43A3Input {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $candidate = if ([IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $RepoRoot $Path }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { throw "Missing ${Label}: $candidate" }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Write-M43A3Utf8CreateNew {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Text)
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) }
    finally { $stream.Dispose() }
}

function ConvertTo-M43A3NativeArgument {
    param([AllowEmptyString()][Parameter(Mandatory = $true)][string]$Value)
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Stop-M43A3ProcessTree {
    param([Parameter(Mandatory = $true)][Diagnostics.Process]$Process)
    if ($Process.HasExited) { return $true }
    try {
        if ($env:OS -eq "Windows_NT") {
            $killer = Start-Process -FilePath (Join-Path $env:SystemRoot "System32\taskkill.exe") `
                -ArgumentList @("/PID", "$($Process.Id)", "/T", "/F") -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput ([IO.Path]::GetTempFileName()) `
                -RedirectStandardError ([IO.Path]::GetTempFileName())
            if (-not $killer.WaitForExit(5000)) { $killer.Kill() }
        }
        else { $Process.Kill($true) }
    }
    catch { try { $Process.Kill() } catch { } }
    try { [void]$Process.WaitForExit(5000) } catch { }
    try { return [bool]$Process.HasExited } catch { return $false }
}

function Invoke-M43A3ProcessBounded {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [ValidateRange(1, 600)][int]$TimeoutSeconds = 60,
        [hashtable]$Environment = @{}
    )
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $FilePath
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    if ($null -ne $info.PSObject.Properties["ArgumentList"]) {
        foreach ($argument in $Arguments) { [void]$info.ArgumentList.Add([string]$argument) }
    }
    else { $info.Arguments = (@($Arguments | ForEach-Object { ConvertTo-M43A3NativeArgument ([string]$_) }) -join " ") }
    foreach ($key in $Environment.Keys) {
        if ($null -ne $info.PSObject.Properties["Environment"]) {
            $info.Environment[[string]$key] = [string]$Environment[$key]
        }
        else { $info.EnvironmentVariables[[string]$key] = [string]$Environment[$key] }
    }
    $process = [Diagnostics.Process]::new(); $process.StartInfo = $info
    try {
        if (-not $process.Start()) { throw "Unable to start bounded process: $FilePath" }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync(); $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            $terminated = Stop-M43A3ProcessTree -Process $process
            return [pscustomobject][ordered]@{
                timed_out = $true; exit_code = $null; process_tree_terminated = $terminated
                stdout = @(); stderr = @(); output = @("process timed out after $TimeoutSeconds seconds")
            }
        }
        $process.WaitForExit()
        $stdout = [string]$stdoutTask.GetAwaiter().GetResult(); $stderr = [string]$stderrTask.GetAwaiter().GetResult()
        $stdoutLines = @($stdout -split "`r?`n" | Where-Object { $_ -ne "" })
        $stderrLines = @($stderr -split "`r?`n" | Where-Object { $_ -ne "" })
        return [pscustomobject][ordered]@{
            timed_out = $false; exit_code = [int]$process.ExitCode; process_tree_terminated = $null
            stdout = $stdoutLines; stderr = $stderrLines; output = @($stdoutLines) + @($stderrLines)
        }
    }
    finally { $process.Dispose() }
}

function Invoke-M43A3GcloudProcess {
    param([Parameter(Mandatory = $true)][string[]]$Arguments, [ValidateRange(1, 600)][int]$TimeoutSeconds = 60)
    $command = Get-Command gcloud -ErrorAction Stop | Select-Object -First 1
    $source = [string]$command.Source
    if (-not $source) { $source = [string]$command.Definition }
    if ([IO.Path]::GetExtension($source).ToLowerInvariant() -eq ".ps1") {
        $hostPath = [string](Get-Process -Id $PID).Path
        return Invoke-M43A3ProcessBounded -FilePath $hostPath `
            -Arguments (@("-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", $source) + $Arguments) `
            -TimeoutSeconds $TimeoutSeconds
    }
    return Invoke-M43A3ProcessBounded -FilePath $source -Arguments $Arguments -TimeoutSeconds $TimeoutSeconds
}

function Invoke-M43A3Gcloud {
    param([Parameter(Mandatory = $true)][string[]]$Arguments, [ValidateRange(1, 600)][int]$TimeoutSeconds = 60)
    $result = Invoke-M43A3GcloudProcess -Arguments $Arguments -TimeoutSeconds $TimeoutSeconds
    if ($result.timed_out) { throw "gcloud process timed out after $TimeoutSeconds seconds: gcloud $($Arguments -join ' ')" }
    if ($result.exit_code -ne 0) { throw "gcloud failed ($($result.exit_code)): $(@($result.output) -join [Environment]::NewLine)" }
    return @($result.stdout)
}

function Test-M43A3GcsObject {
    param([Parameter(Mandatory = $true)][string]$Uri)
    $result = Invoke-M43A3GcloudProcess -Arguments @("storage", "objects", "describe", $Uri, "--format=json") -TimeoutSeconds 20
    if ($result.timed_out) { throw "GCS object lookup timed out: $Uri" }
    if ($result.exit_code -eq 0) { return $true }
    if ((@($result.output) -join "`n") -match '(?i)not found|does not exist|No URLs matched|404') { return $false }
    throw "Unable to determine immutable GCS object state: $Uri"
}

function Get-M43A3GcsJson {
    param([Parameter(Mandatory = $true)][string]$Uri)
    $result = Invoke-M43A3GcloudProcess -Arguments @("storage", "cat", $Uri) -TimeoutSeconds 30
    if ($result.timed_out -or $result.exit_code -ne 0) { throw "Unable to read bounded GCS JSON: $Uri" }
    return ((@($result.stdout) -join "`n") | ConvertFrom-Json)
}

function Get-M43A3StrictInteger {
    param([Parameter(Mandatory = $true)]$Value, [Parameter(Mandatory = $true)][string]$Name)
    $types = @([byte], [sbyte], [int16], [uint16], [int32], [uint32], [int64], [uint64])
    if ($null -eq $Value -or $Value -is [bool] -or $types -notcontains $Value.GetType()) {
        throw "$Name must be a JSON integer"
    }
    return [int64]$Value
}

function Assert-M43A3HexDigest {
    param([Parameter(Mandatory = $true)]$Value, [Parameter(Mandatory = $true)][string]$Name)
    if ([string]$Value -notmatch '^[0-9a-f]{64}$') { throw "$Name is not a lowercase SHA-256 digest" }
}

function Assert-M43A3RunManifest {
    param(
        [Parameter(Mandatory = $true)]$Manifest,
        [string]$ExpectedRunName,
        [string]$ExpectedProject,
        [string]$ExpectedBucket
    )
    if ($Manifest.schema -ne "hu_m43_attempt03_v5_model_spot_run_manifest_v1" -or
        $Manifest.status -ne "frozen" -or
        ($ExpectedRunName -and [string]$Manifest.run_name -ne $ExpectedRunName) -or
        ($ExpectedProject -and [string]$Manifest.project_id -ne $ExpectedProject) -or
        ($ExpectedBucket -and [string]$Manifest.bucket -ne $ExpectedBucket) -or
        (Get-M43A3StrictInteger $Manifest.job_count "job_count") -ne 30 -or
        (Get-M43A3StrictInteger $Manifest.compute.shard_count "compute.shard_count") -ne 8 -or
        (Get-M43A3StrictInteger $Manifest.compute.jobs_per_shard "compute.jobs_per_shard") -ne 4 -or
        (Get-M43A3StrictInteger $Manifest.compute.canary_shard "compute.canary_shard") -ne 0 -or
        $Manifest.compute.canary_required_before_fanout -ne $true -or
        $Manifest.cloud_input_boundary -ne "fit700_only_no_holdouts" -or
        $Manifest.current_profile_mutated -ne $false -or $Manifest.runtime_policy_activated -ne $false -or
        $Manifest.full_replacement -ne $false) {
        throw "Attempt03 model Spot manifest identity/lifecycle changed"
    }
    $fit = @($Manifest.inputs.fit)
    if ($fit.Count -ne 2 -or
        (@($fit.role) -join "|") -ne "inherited_attempt02_train|fresh_train_fit" -or
        (Get-M43A3StrictInteger $fit[0].rows "inherited rows") -ne 200 -or
        (Get-M43A3StrictInteger $fit[1].rows "fresh rows") -ne 500) {
        throw "Attempt03 model Spot fit input boundary changed"
    }
    foreach ($digest in @(
        $Manifest.source.sha256, $Manifest.cloud_contract.file_sha256,
        $Manifest.cloud_contract.contract_sha256, $Manifest.inputs.input_bundle_sha256,
        $Manifest.training_config_sha256, $Manifest.model_freeze_file_sha256,
        $Manifest.training_freeze_file_sha256
    )) { Assert-M43A3HexDigest $digest "manifest digest" }
    $jobs = @($Manifest.jobs)
    if ($jobs.Count -ne 30 -or (@($jobs.job_index) -join ",") -ne ((0..29) -join ",")) {
        throw "Attempt03 model Spot job coverage changed"
    }
    for ($index = 0; $index -lt 30; $index++) {
        $job = $jobs[$index]
        if ((Get-M43A3StrictInteger $job.shard_index "job shard") -ne [Math]::Floor($index / 4)) {
            throw "Attempt03 model Spot shard mapping changed"
        }
        Assert-M43A3HexDigest $job.job_spec_sha256 "job spec"
    }
}

function Assert-M43A3Done {
    param(
        [Parameter(Mandatory = $true)]$Done,
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)][int]$JobIndex,
        [Parameter(Mandatory = $true)][string]$ManifestFileSha256
    )
    $job = @($Manifest.jobs)[$JobIndex]
    if ($Done.schema -ne "hu_m43_attempt03_v5_fold_done_v1" -or $Done.status -ne "complete" -or
        (Get-M43A3StrictInteger $Done.job_index "DONE.job_index") -ne $JobIndex -or
        [string]$Done.run_name -ne [string]$Manifest.run_name -or
        [string]$Done.job_spec_sha256 -ne [string]$job.job_spec_sha256 -or
        [string]$Done.source_sha256 -ne [string]$Manifest.source.sha256 -or
        [string]$Done.run_manifest_sha256 -ne $ManifestFileSha256 -or
        [string]$Done.cloud_contract_file_sha256 -ne [string]$Manifest.cloud_contract.file_sha256 -or
        [string]$Done.input_bundle_sha256 -ne [string]$Manifest.inputs.input_bundle_sha256 -or
        [string]$Done.training_config_sha256 -ne [string]$Manifest.training_config_sha256 -or
        $Done.fit700_only -ne $true -or (Get-M43A3StrictInteger $Done.holdout_input_count "DONE.holdout_input_count") -ne 0 -or
        $Done.current_profile_mutated -ne $false -or $Done.runtime_policy_activated -ne $false) {
        throw "Attempt03 remote DONE hash chain is invalid for job $JobIndex"
    }
    Assert-M43A3HexDigest $Done.artifact_sha256 "DONE artifact"
    Assert-M43A3HexDigest $Done.job_manifest_sha256 "DONE manifest"
}
