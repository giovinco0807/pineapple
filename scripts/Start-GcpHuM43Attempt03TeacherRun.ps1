param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = "regular-hu-m43-attempt03-c2e64-pilot700-r2-20260714-0228",
    [string]$PlanPath = "configs/hu_joint_policy_m43_attempt03.json",
    [string]$TemplateRunDir = "outputs/gcp_runs/regular-hu-m43-attempt02-c2e64-pilot300-20260713-2201",
    [string[]]$InitialShards = @("0"),
    [string[]]$StartShards = @(),
    [string]$Zone = "asia-northeast1-b",
    [string]$MachineType = "c4-standard-4",
    [string]$BootDiskType = "hyperdisk-balanced",
    [int]$BootDiskGb = 50,
    [int]$NativeBatchThreads = 4,
    [int]$SyncIntervalSeconds = 60,
    [switch]$CreateInstances,
    [switch]$SkipExistingInstances,
    [switch]$PackageOnly,
    [switch]$ResumeExisting,
    [switch]$NoSelfDelete
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{0,126}[a-z0-9]$') {
    throw "RunName is not path-safe"
}
if ($ProjectId -notmatch '^[a-z][a-z0-9-]{4,28}[a-z0-9]$') {
    throw "ProjectId is not a valid immutable project identifier"
}
if ($Bucket -notmatch '^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$' -or $Bucket.Contains('..')) {
    throw "Bucket is not a valid immutable bucket identifier"
}
if ($PackageOnly -and $CreateInstances) {
    throw "PackageOnly and CreateInstances are mutually exclusive"
}
if ($PackageOnly -and $ResumeExisting) {
    throw "PackageOnly and ResumeExisting are mutually exclusive"
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Write-Utf8NoBom([string]$Path, [string]$Text) {
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    [IO.File]::WriteAllText($Path, $Text, [Text.UTF8Encoding]::new($false))
}

function Convert-ToVmPrefix([string]$Value) {
    $clean = ($Value.ToLowerInvariant() -replace '[^a-z0-9-]', '-') -replace '-+', '-'
    $clean = $clean.Trim('-')
    if (-not $clean) { throw "RunName cannot be converted to a VM prefix" }
    if ($clean.Length -gt 54) { $clean = $clean.Substring(0, 54).TrimEnd('-') }
    return $clean
}

function Expand-ShardSelection([string[]]$Values, [int]$Count) {
    $selected = [Collections.Generic.SortedSet[int]]::new()
    foreach ($raw in $Values) {
        foreach ($token in ([string]$raw -split ',')) {
            $value = $token.Trim()
            if (-not $value) { continue }
            if ($value -match '^(\d+)-(\d+)$') {
                $first = [int]$Matches[1]; $last = [int]$Matches[2]
                if ($first -gt $last) { throw "Invalid shard range: $value" }
                for ($index = $first; $index -le $last; $index++) { [void]$selected.Add($index) }
            }
            elseif ($value -match '^\d+$') { [void]$selected.Add([int]$value) }
            else { throw "Invalid shard selection: $value" }
        }
    }
    foreach ($index in $selected) {
        if ($index -lt 0 -or $index -ge $Count) { throw "Shard index outside 0..$($Count - 1): $index" }
    }
    return @($selected)
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
    $info.UseShellExecute = $false; $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true
    if ($null -ne $info.PSObject.Properties['ArgumentList']) {
        foreach ($argument in $allArguments) { [void]$info.ArgumentList.Add([string]$argument) }
    }
    else { $info.Arguments = (@($allArguments | ForEach-Object { Quote-NativeArgument ([string]$_) }) -join ' ') }
    $process = [Diagnostics.Process]::new(); $process.StartInfo = $info
    try {
        if (-not $process.Start()) { throw "Unable to start gcloud" }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync(); $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            Stop-ProcessTree $process
            throw "gcloud timed out after $TimeoutSeconds seconds: gcloud $($Arguments -join ' ')"
        }
        $process.WaitForExit()
        $stdout = [string]$stdoutTask.GetAwaiter().GetResult(); $stderr = [string]$stderrTask.GetAwaiter().GetResult()
        $output = @($stdout -split "`r?`n" | Where-Object { $_ }) + @($stderr -split "`r?`n" | Where-Object { $_ })
        if ($process.ExitCode -ne 0 -and -not $AllowFailure) {
            throw "gcloud failed ($($process.ExitCode)): $($Arguments -join ' ')`n$($output -join [Environment]::NewLine)"
        }
        return [pscustomobject]@{ exit_code = [int]$process.ExitCode; stdout = $stdout; stderr = $stderr; output = $output }
    }
    finally { $process.Dispose() }
}

function Test-GcsObject([string]$Uri) {
    $result = Invoke-GcloudBounded @('storage','objects','describe',$Uri,'--project',$ProjectId,'--format=json') 20 -AllowFailure
    if ($result.exit_code -eq 0) { return $true }
    if (($result.output -join "`n") -match '(?i)not found|does not exist|no urls matched|404') { return $false }
    throw "Unable to prove GCS object state: $Uri"
}

function Assert-FileSha256([string]$Path, [string]$Expected, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$Label is missing: $Path" }
    $actual = Get-Sha256 $Path
    if ($actual -ne $Expected) { throw "$Label SHA256 changed: expected=$Expected actual=$actual" }
}

function Assert-GcsObjectMatchesFile([string]$Uri, [string]$LocalPath, [int]$TimeoutSeconds = 180) {
    $tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("hu-m43-a03-gcs-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tempRoot | Out-Null
    $download = Join-Path $tempRoot 'object.bin'
    try {
        [void](Invoke-GcloudBounded @('storage','cp',$Uri,$download,'--project',$ProjectId) $TimeoutSeconds)
        $expected = Get-Sha256 $LocalPath
        $actual = Get-Sha256 $download
        if ($actual -ne $expected) { throw "Remote immutable object bytes changed: $Uri" }
    }
    finally { Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue }
}

function Read-VerifiedRemoteDone([string]$Uri, $Spec, $Manifest, [string]$ManifestSha256) {
    $result = Invoke-GcloudBounded @('storage','cat',$Uri,'--project',$ProjectId) 30
    try { $done = $result.stdout | ConvertFrom-Json }
    catch { throw "Attempt03 DONE JSON is invalid: $Uri" }
    if ($done.schema -ne 'hu_m43_attempt02_teacher_done_v1' -or
        $done.status -ne 'complete' -or
        [string]$done.run_name -ne $RunName -or
        [int]$done.shard -ne [int]$Spec.shard -or
        [string]$done.split -ne [string]$Spec.split -or
        [int]$done.roots -ne [int]$Spec.roots -or
        [string]$done.output_prefix -ne [string]$Spec.output_prefix -or
        [string]$done.manifest_sha256 -ne $ManifestSha256 -or
        [string]$done.shards_manifest_sha256 -ne [string]$Manifest.schedule_sha256 -or
        [string]$done.source_sha256 -ne [string]$Manifest.source_sha256 -or
        [string]$done.startup_sha256 -ne [string]$Manifest.startup_sha256 -or
        [string]$done.model_manifest_sha256 -ne [string]$Manifest.model_manifest_sha256 -or
        [string]$done.native_manifest_sha256 -ne [string]$Manifest.native_manifest_sha256) {
        throw "Attempt03 DONE does not match the frozen closure: $Uri"
    }
    return $done
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Push-Location $repoRoot
try {
    $PlanPath = [IO.Path]::GetFullPath((Join-Path $repoRoot $PlanPath))
    $TemplateRunDir = [IO.Path]::GetFullPath((Join-Path $repoRoot $TemplateRunDir))
    $freezePath = Join-Path $repoRoot 'configs/hu_joint_policy_m43_attempt03_model_freeze.json'
    if (-not (Test-Path -LiteralPath $freezePath -PathType Leaf)) { throw "Attempt03 model freeze is missing" }
    $freeze = Get-Content -Raw -LiteralPath $freezePath | ConvertFrom-Json
    if ($freeze.schema -ne 'hu_m43_attempt03_model_freeze_v1' -or
        $freeze.status -ne 'frozen_before_any_attempt03_teacher_row_was_received') {
        throw "Attempt03 model freeze is not immutable"
    }
    $plan = Get-Content -Raw -LiteralPath $PlanPath | ConvertFrom-Json
    if ($plan.schema -ne 'hu_m43_attempt03_plan_v1' -or $plan.status -ne 'frozen_pre_generation') { throw "Attempt03 plan is not frozen" }
    $planSha256 = Get-Sha256 $PlanPath
    if ($planSha256 -ne [string]$freeze.parent_plan.file_sha256) { throw "Attempt03 plan does not match the model freeze" }
    if ([int]$plan.budget.fresh_roots -ne 700 -or [int]$plan.budget.fresh_shards -ne 70 -or [int]$plan.budget.roots_per_shard -ne 10) { throw "Attempt03 bounded budget changed" }
    if ([int]$plan.teacher_search.candidate_samples -ne 2 -or [int]$plan.teacher_search.evaluation_samples -ne 64) { throw "Attempt03 teacher budget changed" }
    if ($NativeBatchThreads -ne [int]$plan.teacher_search.native_batch_threads) { throw "Attempt03 native thread setting changed" }
    Assert-FileSha256 (Join-Path $repoRoot ([string]$freeze.implementation.module)) ([string]$freeze.implementation.file_sha256) 'Attempt03 v5 implementation'
    Assert-FileSha256 (Join-Path $repoRoot ([string]$freeze.implementation.stage18_model.path)) ([string]$freeze.implementation.stage18_model.file_sha256) 'Attempt03 Stage18 model'
    $frozenRunName = [string]$freeze.teacher_run.run_name
    if ($ResumeExisting -and $RunName -ne $frozenRunName) { throw "ResumeExisting is restricted to the frozen Attempt03 teacher run" }
    if (-not $ResumeExisting -and -not $PackageOnly) { throw "Attempt03 cloud closure is frozen; use -ResumeExisting for the bound r2 run" }

    $runDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
    $manifestPath = Join-Path $runDir 'manifest.json'
    $shardsPath = Join-Path $runDir 'shards_manifest.jsonl'
    $packagePath = Join-Path $runDir 'ofc_regular_hu_m43_attempt03_teacher_source.zip'
    $startupPath = Join-Path $runDir 'startup_hu_m43_attempt03_teacher.sh'
    $preflightPath = Join-Path $runDir 'm43_attempt03_preflight_receipt.json'
    $sourceModelPath = Join-Path $runDir 'source_model_manifest.json'
    $sourceNativePath = Join-Path $runDir 'source_native_manifest.json'
    $prefix = "gs://$Bucket/runs/$RunName"

    $selection = @(if ($StartShards.Count -gt 0) { Expand-ShardSelection $StartShards 70 } else { Expand-ShardSelection $InitialShards 70 })
    if ($ResumeExisting) {
        if (-not (Test-Path -LiteralPath $manifestPath)) { throw "ResumeExisting requires the local immutable manifest" }
        $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
        if ($manifest.schema -ne 'hu_m43_attempt03_teacher_spot_manifest_v1' -or $manifest.run_name -ne $RunName) { throw "Attempt03 resume manifest identity changed" }
        if ([string]$manifest.project_id -ne $ProjectId -or [string]$manifest.bucket -ne $Bucket -or
            [string]$manifest.zone -ne $Zone -or [string]$manifest.machine_type -ne $MachineType -or
            [bool]$manifest.self_delete -eq [bool]$NoSelfDelete) {
            throw "Attempt03 resume parameters conflict with the frozen manifest"
        }
        Assert-FileSha256 $manifestPath ([string]$freeze.teacher_run.manifest_sha256) 'Attempt03 r2 manifest'
        Assert-FileSha256 $PlanPath ([string]$manifest.plan_file_sha256) 'Attempt03 plan'
        Assert-FileSha256 $preflightPath ([string]$manifest.preflight_file_sha256) 'Attempt03 preflight receipt'
        Assert-FileSha256 $shardsPath ([string]$manifest.schedule_sha256) 'Attempt03 shard schedule'
        Assert-FileSha256 $packagePath ([string]$manifest.source_sha256) 'Attempt03 source closure'
        Assert-FileSha256 $startupPath ([string]$manifest.startup_sha256) 'Attempt03 startup script'
        Assert-FileSha256 $sourceModelPath ([string]$manifest.model_manifest_sha256) 'Attempt03 source model manifest'
        Assert-FileSha256 $sourceNativePath ([string]$manifest.native_manifest_sha256) 'Attempt03 source native manifest'
        foreach ($pair in @(
            @($manifestPath,"$prefix/manifest.json"),
            @($packagePath,"$prefix/source/ofc_regular_hu_m43_attempt03_teacher_source.zip"),
            @($startupPath,"$prefix/source/startup_hu_m43_attempt03_teacher.sh"),
            @($shardsPath,"$prefix/source/shards_manifest.jsonl"),
            @($sourceModelPath,"$prefix/source/source_model_manifest.json"),
            @($sourceNativePath,"$prefix/source/source_native_manifest.json")
        )) { Assert-GcsObjectMatchesFile ([string]$pair[1]) ([string]$pair[0]) }
    }
    else {
        if (Test-Path -LiteralPath $runDir) { throw "Attempt03 run directory already exists; use -ResumeExisting" }
        New-Item -ItemType Directory -Path $runDir | Out-Null
        $contractModule = Join-Path $repoRoot 'src/ofc_regular/hu_m43_attempt03_contract.py'
        if (-not (Test-Path -LiteralPath $contractModule)) { throw "Attempt03 contract module is missing" }
        $preflightOutput = @(& python -B -m ofc_regular.hu_m43_attempt03_contract preflight --plan $PlanPath --repo-root $repoRoot --output $preflightPath 2>&1)
        if ($LASTEXITCODE -ne 0) { throw "Attempt03 preflight failed: $($preflightOutput -join "`n")" }
        $preflight = Get-Content -Raw -LiteralPath $preflightPath | ConvertFrom-Json
        if ($preflight.status -notmatch '^pass') { throw "Attempt03 preflight did not pass" }

        $templatePackage = Join-Path $TemplateRunDir 'package_src'
        $templateStartup = Join-Path $TemplateRunDir 'startup_hu_m43_attempt02_teacher.sh'
        if (-not (Test-Path -LiteralPath $templatePackage) -or -not (Test-Path -LiteralPath $templateStartup)) { throw "Pinned Attempt02 teacher closure template is missing" }
        $packageSource = Join-Path $runDir 'package_src'
        Copy-Item -LiteralPath $templatePackage -Destination $packageSource -Recurse
        Copy-Item -LiteralPath $templateStartup -Destination $startupPath

        $specs = [Collections.Generic.List[object]]::new(); $global = 0
        foreach ($logicalSplit in @('train.fit','train.precal_holdout')) {
            $splitSpec = $plan.fresh_splits.$logicalSplit
            $roots = [int]$splitSpec.roots; $shards = $roots / 10
            for ($splitShard = 0; $splitShard -lt $shards; $splitShard++) {
                $offset = $splitShard * 10
                $seedStart = [long]$splitSpec.seed_start + [long]$offset * [long]$splitSpec.seed_stride
                $slug = if ($logicalSplit -eq 'train.fit') { 'train_fit' } else { 'precal_holdout' }
                $specs.Add([ordered]@{
                    schema = 'hu_m43_attempt03_teacher_shard_v1'; shard = $global; logical_split = $logicalSplit
                    split = [string]$splitSpec.record_split; split_shard = $splitShard; roots = 10
                    seed_start = $seedStart; seed_stride = [long]$splitSpec.seed_stride
                    candidate_seed = [long]$splitSpec.candidate_seed_start + [long]$splitShard * [long]$splitSpec.seed_stride
                    evaluation_seed = [long]$splitSpec.evaluation_seed_start + [long]$splitShard * [long]$splitSpec.seed_stride
                    child_policy_seed = [long]$splitSpec.child_policy_seed_start + [long]$splitShard * [long]$splitSpec.seed_stride
                    candidate_samples = 2; evaluation_samples = 64
                    output_prefix = ("{0}_shard_{1:D3}_roots10_seed{2}" -f $slug,$splitShard,$seedStart)
                    profile_quota_per_shard = [ordered]@{ stage19_p0=2; stage9f_p2=2; stage7_m5_r10=2; stage3_baseline=2; random_exact_final=2 }
                })
                $global++
            }
        }
        if ($specs.Count -ne 70) { throw "Attempt03 schedule did not produce 70 shards" }
        $scheduleText = (@($specs | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 8 }) -join "`n") + "`n"
        Write-Utf8NoBom $shardsPath $scheduleText
        Copy-Item -LiteralPath $shardsPath -Destination (Join-Path $packageSource 'shards_manifest.jsonl') -Force
        Copy-Item -LiteralPath (Join-Path $packageSource 'source_model_manifest.json') -Destination $sourceModelPath
        Copy-Item -LiteralPath (Join-Path $packageSource 'source_native_manifest.json') -Destination $sourceNativePath
        # PowerShell's Compress-Archive writes backslash member names on
        # Windows.  Linux unzip treats those as literal characters, so build
        # the immutable cloud closure with POSIX member names explicitly.
        $zipBuilder = @'
import pathlib, sys, zipfile
root = pathlib.Path(sys.argv[1]).resolve()
destination = pathlib.Path(sys.argv[2]).resolve()
with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        archive.write(path, path.relative_to(root).as_posix())
with zipfile.ZipFile(destination) as archive:
    names = archive.namelist()
    assert names and all("\\" not in name and not name.startswith("/") for name in names)
'@
        $zipOutput = @($zipBuilder | & python - $packageSource $packagePath 2>&1)
        if ($LASTEXITCODE -ne 0) { throw "Attempt03 POSIX zip build failed: $($zipOutput -join "`n")" }

        $manifest = [ordered]@{
            schema='hu_m43_attempt03_teacher_spot_manifest_v1'; run_name=$RunName; project_id=$ProjectId; bucket=$Bucket
            milestone='M4.3-attempt03'; purpose='fresh_train_fit_and_sealed_one_shot_precal_teacher'
            plan_file_sha256=(Get-Sha256 $PlanPath); preflight_file_sha256=(Get-Sha256 $preflightPath)
            total_roots=700; total_shards=70; roots_per_shard=10
            logical_split_roots=[ordered]@{'train.fit'=500;'train.precal_holdout'=200}
            record_split='train'; candidate_samples=2; evaluation_samples=64
            schedule_sha256=(Get-Sha256 $shardsPath); source_sha256=(Get-Sha256 $packagePath)
            startup_sha256=(Get-Sha256 $startupPath); model_manifest_sha256=(Get-Sha256 $sourceModelPath)
            native_manifest_sha256=(Get-Sha256 $sourceNativePath)
            inherited_teacher_closure=[ordered]@{source_run='regular-hu-m43-attempt02-c2e64-pilot300-20260713-2201'; algorithm_unchanged=$true}
            machine_type=$MachineType; zone=$Zone; spot=$true; self_delete=(-not $NoSelfDelete)
            current_profile_mutated=$false; runtime_policy_activated=$false; sealed_calibration_uploaded=$false; locked_uploaded=$false
            created_at=(Get-Date).ToUniversalTime().ToString('o')
        }
        Write-Utf8NoBom $manifestPath (($manifest | ConvertTo-Json -Depth 12) + "`n")
        if ($PackageOnly) { [pscustomobject]@{status='packaged';run_name=$RunName;manifest=$manifestPath;source=$packagePath;shards=70} | ConvertTo-Json; return }
        if (Test-GcsObject "$prefix/manifest.json") { throw "Attempt03 remote run already exists and is immutable" }
        foreach ($pair in @(
            @($packagePath,"$prefix/source/ofc_regular_hu_m43_attempt03_teacher_source.zip"),
            @($startupPath,"$prefix/source/startup_hu_m43_attempt03_teacher.sh"),
            @($shardsPath,"$prefix/source/shards_manifest.jsonl"),
            @($sourceModelPath,"$prefix/source/source_model_manifest.json"),
            @($sourceNativePath,"$prefix/source/source_native_manifest.json"),
            @($manifestPath,"$prefix/manifest.json")
        )) { [void](Invoke-GcloudBounded @('storage','cp',$pair[0],$pair[1],'--project',$ProjectId,'--if-generation-match=0') 120) }
    }

    if (-not $CreateInstances) {
        [pscustomobject]@{schema='hu_m43_attempt03_teacher_spot_start_v1';status='prepared';run_name=$RunName;selected_shards=$selection;manifest_sha256=(Get-Sha256 $manifestPath);create_instances=$false} | ConvertTo-Json -Depth 6
        return
    }
    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    $specs = @(Get-Content -LiteralPath $shardsPath | ForEach-Object { $_ | ConvertFrom-Json })
    $vmPrefix = Convert-ToVmPrefix $RunName; $started = @()
    foreach ($shard in $selection) {
        $spec = $specs[$shard]
        $doneUri = "$prefix/results/$($spec.output_prefix)/DONE.json"
        if (Test-GcsObject $doneUri) {
            [void](Read-VerifiedRemoteDone $doneUri $spec $manifest (Get-Sha256 $manifestPath))
            continue
        }
        $vmName = ("{0}-{1:D3}" -f $vmPrefix,$shard)
        $metadata = @(
            "RUN_NAME=$RunName","BUCKET=$Bucket","SHARD_INDEX=$shard",
            "SOURCE_URI=$prefix/source/ofc_regular_hu_m43_attempt03_teacher_source.zip","SOURCE_SHA256=$($manifest.source_sha256)",
            "STARTUP_SHA256=$($manifest.startup_sha256)","MANIFEST_SHA256=$(Get-Sha256 $manifestPath)",
            "SHARDS_SHA256=$($manifest.schedule_sha256)","MODELS_SHA256=$($manifest.model_manifest_sha256)",
            "NATIVE_SHA256=$($manifest.native_manifest_sha256)","SYNC_INTERVAL_SECONDS=$SyncIntervalSeconds",
            "NATIVE_BATCH_THREADS=$NativeBatchThreads",("SELF_DELETE=" + $(if ($NoSelfDelete) {'0'} else {'1'}))
        ) -join ','
        $present = Invoke-GcloudBounded @('compute','instances','describe',$vmName,'--project',$ProjectId,'--zone',$Zone,'--format=json') 30 -AllowFailure
        if ($present.exit_code -eq 0) {
            if (-not $SkipExistingInstances) { throw "Attempt03 worker already exists: $vmName" }
            try { $existing = $present.stdout | ConvertFrom-Json } catch { throw "Existing Attempt03 worker JSON is invalid: $vmName" }
            $existingMetadata = @{}
            foreach ($item in @($existing.metadata.items)) { $existingMetadata[[string]$item.key] = [string]$item.value }
            foreach ($binding in @{
                RUN_NAME=$RunName; BUCKET=$Bucket; SHARD_INDEX=[string]$shard;
                SOURCE_URI="$prefix/source/ofc_regular_hu_m43_attempt03_teacher_source.zip";
                SOURCE_SHA256=[string]$manifest.source_sha256; STARTUP_SHA256=[string]$manifest.startup_sha256;
                MANIFEST_SHA256=(Get-Sha256 $manifestPath);
                SHARDS_SHA256=[string]$manifest.schedule_sha256; MODELS_SHA256=[string]$manifest.model_manifest_sha256;
                NATIVE_SHA256=[string]$manifest.native_manifest_sha256;
                SYNC_INTERVAL_SECONDS=[string]$SyncIntervalSeconds; NATIVE_BATCH_THREADS=[string]$NativeBatchThreads;
                SELF_DELETE=$(if ($NoSelfDelete) {'0'} else {'1'})
            }.GetEnumerator()) {
                if ($existingMetadata[[string]$binding.Key] -ne [string]$binding.Value) { throw "Existing Attempt03 worker metadata changed: $vmName/$($binding.Key)" }
            }
            if (([string]$existing.machineType -split '/')[-1] -ne [string]$manifest.machine_type -or
                [string]$existing.labels.purpose -ne 'hu-m43-a03-teacher' -or
                [string]$existing.labels.milestone -ne 'm43-a03') {
                throw "Existing Attempt03 worker compute closure changed: $vmName"
            }
            if ([string]$existing.status -notin @('PROVISIONING','STAGING','RUNNING')) { throw "Existing Attempt03 worker is not active: $vmName" }
            continue
        }
        if (($present.output -join "`n") -notmatch '(?i)not found|was not found|404') { throw "Unable to prove Attempt03 worker absence: $vmName" }
        $arguments = @('compute','instances','create',$vmName,'--project',$ProjectId,'--zone',$Zone,'--machine-type',$MachineType,
            '--image-family','ubuntu-2404-lts-amd64','--image-project','ubuntu-os-cloud','--boot-disk-size',("${BootDiskGb}GB"),
            '--boot-disk-type',$BootDiskType,'--boot-disk-auto-delete','--provisioning-model','SPOT','--instance-termination-action','DELETE',
            '--maintenance-policy','TERMINATE','--scopes','cloud-platform','--labels','purpose=hu-m43-a03-teacher,milestone=m43-a03',
            '--metadata',$metadata,'--metadata-from-file',("startup-script=$startupPath"),'--async','--format=value(name)')
        $created = Invoke-GcloudBounded $arguments 45 -AllowFailure
        if ($created.exit_code -ne 0) { throw "Unable to submit Attempt03 worker $vmName`: $($created.output -join [Environment]::NewLine)" }
        $started += [pscustomobject]@{shard=$shard;name=$vmName;logical_split=$spec.logical_split;zone=$Zone;machine_type=$MachineType}
    }
    [pscustomobject]@{
        schema='hu_m43_attempt03_teacher_spot_start_v1';status='submitted';run_name=$RunName
        selected_shards=$selection;submitted=$started;manifest_sha256=(Get-Sha256 $manifestPath)
        spot=$true;self_delete=(-not $NoSelfDelete);current_profile_mutated=$false;runtime_policy_activated=$false
    } | ConvertTo-Json -Depth 8
}
finally { Pop-Location }
