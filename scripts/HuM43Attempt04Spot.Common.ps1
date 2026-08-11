Set-StrictMode -Version Latest

function Get-M43A4Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-M43A4Sha256 {
    param($Value, [Parameter(Mandatory = $true)][string]$Label)
    if ($Value -isnot [string] -or [string]$Value -cnotmatch '^[0-9a-f]{64}$') {
        throw "$Label must be a lowercase SHA-256 digest"
    }
}

function Write-M43A4Utf8CreateNew {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Text)
    $stream = [IO.File]::Open(
        $Path,
        [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write,
        [IO.FileShare]::None
    )
    try {
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    }
    finally { $stream.Dispose() }
}

function Resolve-M43A4Path {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Label,
        [switch]$RequireFile
    )
    $candidate = $Path
    if (-not [IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path $Root $candidate
    }
    $full = [IO.Path]::GetFullPath($candidate)
    if ($RequireFile -and -not (Test-Path -LiteralPath $full -PathType Leaf)) {
        throw "$Label is missing: $full"
    }
    return $full
}

function Assert-M43A4UnderRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $full = [IO.Path]::GetFullPath($Path)
    $fullRoot = [IO.Path]::GetFullPath($Root).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $prefix = $fullRoot + [IO.Path]::DirectorySeparatorChar
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must remain under $fullRoot"
    }
}

function ConvertTo-M43A4NativeArgument {
    param([AllowEmptyString()][Parameter(Mandatory = $true)][string]$Value)
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Get-M43A4PowerShellPath {
    if ($env:OS -eq 'Windows_NT') {
        $windowsPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
        if (Test-Path -LiteralPath $windowsPowerShell -PathType Leaf) {
            return $windowsPowerShell
        }
    }
    $processPath = [string](Get-Process -Id $PID).Path
    if (-not $processPath) { throw 'Unable to resolve the current PowerShell executable' }
    return $processPath
}

function Get-M43A4GcloudInvocation {
    $command = Get-Command gcloud -ErrorAction Stop | Select-Object -First 1
    $source = [string]$command.Source
    if (-not $source) { $source = [string]$command.Definition }
    if (-not $source) { throw 'Unable to resolve gcloud' }
    $extension = [IO.Path]::GetExtension($source).ToLowerInvariant()
    if ($extension -in @('.cmd', '.bat')) {
        $sibling = [IO.Path]::ChangeExtension($source, '.ps1')
        if (Test-Path -LiteralPath $sibling -PathType Leaf) {
            $source = $sibling
            $extension = '.ps1'
        }
    }
    if ($extension -eq '.ps1') {
        return [pscustomobject][ordered]@{
            file_path = Get-M43A4PowerShellPath
            prefix_arguments = @(
                '-NoLogo', '-NoProfile', '-NonInteractive',
                '-ExecutionPolicy', 'Bypass', '-File', $source
            )
        }
    }
    if ($extension -in @('.cmd', '.bat')) {
        throw 'gcloud.cmd without its gcloud.ps1 sibling is unsupported'
    }
    return [pscustomobject][ordered]@{
        file_path = $source
        prefix_arguments = @()
    }
}

function New-M43A4RunningProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Label,
        [hashtable]$Environment = @{}
    )
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $FilePath
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    if ($null -ne $info.PSObject.Properties['ArgumentList']) {
        foreach ($argument in $Arguments) {
            [void]$info.ArgumentList.Add([string]$argument)
        }
    }
    else {
        $info.Arguments = (@(
            $Arguments | ForEach-Object { ConvertTo-M43A4NativeArgument ([string]$_) }
        ) -join ' ')
    }
    foreach ($key in $Environment.Keys) {
        if ($null -ne $info.PSObject.Properties['Environment']) {
            $info.Environment[[string]$key] = [string]$Environment[$key]
        }
        else {
            $info.EnvironmentVariables[[string]$key] = [string]$Environment[$key]
        }
    }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    if (-not $process.Start()) {
        $process.Dispose()
        throw "Unable to start process: $Label"
    }
    return [pscustomobject][ordered]@{
        label = $Label
        process = $process
        stdout_task = $process.StandardOutput.ReadToEndAsync()
        stderr_task = $process.StandardError.ReadToEndAsync()
        started_at = [DateTime]::UtcNow
    }
}

function Stop-M43A4ProcessTree {
    param([Parameter(Mandatory = $true)][Diagnostics.Process]$Process)
    if ($Process.HasExited) { return $true }
    try {
        if ($env:OS -eq 'Windows_NT') {
            $killer = Start-Process `
                -FilePath (Join-Path $env:SystemRoot 'System32\taskkill.exe') `
                -ArgumentList @('/PID', "$($Process.Id)", '/T', '/F') `
                -WindowStyle Hidden -PassThru
            if (-not $killer.WaitForExit(5000)) { $killer.Kill() }
        }
        else {
            try { $Process.Kill($true) } catch { $Process.Kill() }
        }
    }
    catch { try { $Process.Kill() } catch { } }
    try { [void]$Process.WaitForExit(5000) } catch { }
    try { return [bool]$Process.HasExited } catch { return $false }
}

function Complete-M43A4RunningProcess {
    param(
        [Parameter(Mandatory = $true)]$Running,
        [ValidateRange(1, 3600)][int]$TimeoutSeconds = 300
    )
    $process = [Diagnostics.Process]$Running.process
    try {
        $elapsed = ([DateTime]::UtcNow - [DateTime]$Running.started_at).TotalSeconds
        $remaining = [Math]::Max(0, $TimeoutSeconds - [int][Math]::Floor($elapsed))
        if (-not $process.HasExited -and -not $process.WaitForExit($remaining * 1000)) {
            $terminated = Stop-M43A4ProcessTree -Process $process
            return [pscustomobject][ordered]@{
                label = [string]$Running.label
                timed_out = $true
                process_tree_terminated = [bool]$terminated
                exit_code = $null
                stdout = ''
                stderr = ''
            }
        }
        $process.WaitForExit()
        return [pscustomobject][ordered]@{
            label = [string]$Running.label
            timed_out = $false
            process_tree_terminated = $null
            exit_code = [int]$process.ExitCode
            stdout = [string]$Running.stdout_task.GetAwaiter().GetResult()
            stderr = [string]$Running.stderr_task.GetAwaiter().GetResult()
        }
    }
    finally { $process.Dispose() }
}

function Invoke-M43A4ProcessBounded {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [ValidateRange(1, 3600)][int]$TimeoutSeconds = 300,
        [string]$Label = 'bounded process',
        [hashtable]$Environment = @{}
    )
    $running = New-M43A4RunningProcess `
        -FilePath $FilePath -Arguments $Arguments -Label $Label -Environment $Environment
    return Complete-M43A4RunningProcess -Running $running -TimeoutSeconds $TimeoutSeconds
}

function Invoke-M43A4GcloudProcess {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [ValidateRange(1, 3600)][int]$TimeoutSeconds = 300,
        [string]$Label = 'gcloud'
    )
    $invocation = Get-M43A4GcloudInvocation
    return Invoke-M43A4ProcessBounded `
        -FilePath ([string]$invocation.file_path) `
        -Arguments (@($invocation.prefix_arguments) + $Arguments) `
        -TimeoutSeconds $TimeoutSeconds `
        -Label $Label
}

function Invoke-M43A4Gcloud {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [ValidateRange(1, 3600)][int]$TimeoutSeconds = 300,
        [string]$Label = 'gcloud'
    )
    $result = Invoke-M43A4GcloudProcess `
        -Arguments $Arguments -TimeoutSeconds $TimeoutSeconds -Label $Label
    if ($result.timed_out) { throw "$Label timed out after $TimeoutSeconds seconds" }
    if ($result.exit_code -ne 0) {
        throw "$Label failed ($($result.exit_code)): $($result.stderr)$($result.stdout)"
    }
    return [string]$result.stdout
}

function Test-M43A4GcsObject {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [string]$ProjectId
    )
    Assert-M43A4ExactGcsUri $Uri 'GCS object lookup'
    $arguments = @('storage', 'objects', 'describe', $Uri, '--format=json')
    if ($ProjectId) { $arguments += @('--project', $ProjectId) }
    $result = Invoke-M43A4GcloudProcess -Arguments $arguments -TimeoutSeconds 30 -Label "describe $Uri"
    if (-not $result.timed_out -and $result.exit_code -eq 0) { return $true }
    $message = ([string]$result.stdout) + ([string]$result.stderr)
    if (-not $result.timed_out -and $message -match '(?i)not found|does not exist|No URLs matched|404') {
        return $false
    }
    throw "Unable to prove GCS object state: $Uri"
}

function Assert-M43A4ExactGcsUri {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if ($Uri -notmatch '^gs://[a-z0-9][a-z0-9._-]+/.+$' -or
        $Uri -match '[*?\[\]]' -or $Uri.Contains('..')) {
        throw "$Label is not one exact GCS object URI: $Uri"
    }
}

function Invoke-M43A4ParallelExactGcsCopies {
    param(
        [Parameter(Mandatory = $true)][object[]]$CopyJobs,
        [Parameter(Mandatory = $true)][string]$ProjectId,
        [ValidateRange(1, 8)][int]$MaxParallel = 8,
        [ValidateRange(1, 3600)][int]$TimeoutSecondsPerJob = 600
    )
    if ($CopyJobs.Count -eq 0) { return @() }
    $invocation = Get-M43A4GcloudInvocation
    $pending = [Collections.Generic.Queue[object]]::new()
    foreach ($job in $CopyJobs) {
        $sources = @($job.sources)
        if ($sources.Count -eq 0) { throw 'Parallel copy job has no sources' }
        foreach ($source in $sources) {
            Assert-M43A4ExactGcsUri ([string]$source) ([string]$job.label)
        }
        $destination = [IO.Path]::GetFullPath([string]$job.destination)
        if (-not (Test-Path -LiteralPath $destination -PathType Container)) {
            throw "Parallel copy destination must already exist: $destination"
        }
        $pending.Enqueue([pscustomobject][ordered]@{
            label = [string]$job.label
            sources = $sources
            destination = $destination
        })
    }
    $running = [Collections.ArrayList]::new()
    $completed = [Collections.Generic.List[object]]::new()
    try {
        while ($pending.Count -gt 0 -or $running.Count -gt 0) {
            while ($pending.Count -gt 0 -and $running.Count -lt $MaxParallel) {
                $job = $pending.Dequeue()
                $arguments = @($invocation.prefix_arguments) + @('storage', 'cp') +
                    @($job.sources) + @($job.destination, '--project', $ProjectId)
                $item = New-M43A4RunningProcess `
                    -FilePath ([string]$invocation.file_path) `
                    -Arguments $arguments `
                    -Label ([string]$job.label)
                $item | Add-Member -NotePropertyName copy_job -NotePropertyValue $job
                [void]$running.Add($item)
            }
            $madeProgress = $false
            for ($index = $running.Count - 1; $index -ge 0; $index--) {
                $item = $running[$index]
                $elapsed = ([DateTime]::UtcNow - [DateTime]$item.started_at).TotalSeconds
                if ($item.process.HasExited -or $elapsed -ge $TimeoutSecondsPerJob) {
                    $result = Complete-M43A4RunningProcess `
                        -Running $item -TimeoutSeconds $TimeoutSecondsPerJob
                    $running.RemoveAt($index)
                    $madeProgress = $true
                    if ($result.timed_out -or $result.exit_code -ne 0) {
                        throw "Parallel exact copy failed: $($result.label): $($result.stderr)$($result.stdout)"
                    }
                    $completed.Add([pscustomobject][ordered]@{
                        label = [string]$result.label
                        destination = [string]$item.copy_job.destination
                        source_count = @($item.copy_job.sources).Count
                    })
                }
            }
            if (-not $madeProgress -and $running.Count -gt 0) {
                Start-Sleep -Milliseconds 100
            }
        }
    }
    catch {
        foreach ($item in @($running)) {
            try { [void](Stop-M43A4ProcessTree -Process $item.process) } catch { }
            try { $item.process.Dispose() } catch { }
        }
        throw
    }
    return @($completed)
}

function Copy-M43A4RemoteFileExact {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$ProjectId,
        [string]$ExpectedSha256
    )
    Assert-M43A4ExactGcsUri $Uri 'remote file'
    if (Test-Path -LiteralPath $Destination) {
        throw "Refusing to overwrite exact remote download: $Destination"
    }
    $parent = Split-Path -Parent $Destination
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    [void](Invoke-M43A4Gcloud `
        -Arguments @('storage', 'cp', $Uri, $Destination, '--project', $ProjectId) `
        -TimeoutSeconds 300 `
        -Label "download $Uri")
    if ($ExpectedSha256) {
        Assert-M43A4Sha256 $ExpectedSha256 'expected remote file hash'
        $actual = Get-M43A4Sha256 $Destination
        if ($actual -ne $ExpectedSha256) {
            throw "Remote file byte hash mismatch: expected=$ExpectedSha256 actual=$actual"
        }
    }
}
