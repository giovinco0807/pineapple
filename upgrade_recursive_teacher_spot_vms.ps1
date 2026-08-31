param(
    [string]$Project = "ofc-solver-485418",
    [string]$RunId = "recursive-t0t3-s128-b5-c2-p16-tr32-20260520",
    [string]$Bucket = "gs://pokerhu-ofc-solver-485418-training",
    [int]$TargetHands = 200,
    [switch]$KeepRunningSmall
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
    $global:PSNativeCommandUseErrorActionPreference = $false
}
$Gcloud = "gcloud.cmd"

function Get-Region([string]$Zone) {
    return ($Zone -replace "-[a-z]$", "")
}

function Get-WorkerId([string]$Name) {
    if ($Name -match "-(\d+)$") {
        return [int]$Matches[1]
    }
    throw "Cannot parse worker id from $Name"
}

function Get-FreePreemptibleCpu([string]$Region) {
    $q = & $Gcloud compute regions describe $Region --project=$Project --format="json(quotas)" | ConvertFrom-Json
    $pre = $q.quotas | Where-Object metric -eq "PREEMPTIBLE_CPUS"
    return [double]$pre.limit - [double]$pre.usage
}

function Invoke-GcloudQuiet {
    param([string[]]$ArgsList)
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Gcloud @ArgsList *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "gcloud failed with exit code ${LASTEXITCODE}: $($ArgsList -join ' ')"
        }
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
}

$tmp = Join-Path $env:TEMP ("ofc_summaries_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    gsutil -m cp "$Bucket/runs/$RunId/results/worker_*/worker_*.summary.json" $tmp *> $null
    $ErrorActionPreference = $oldErrorActionPreference
    $hands = @{}
    $requested = @{}
    for ($i = 0; $i -lt 200; $i++) {
        $f = Join-Path $tmp ("worker_{0}.summary.json" -f $i)
        if (Test-Path $f) {
            try {
                $s = Get-Content $f -Raw | ConvertFrom-Json
                $hands[$i] = [int]$s.hands_completed
                $requested[$i] = [int]$s.hands_requested
            } catch {
                Write-Warning "Skipping partial summary for worker_$i"
            }
        }
    }
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

$instances = & $Gcloud compute instances list `
    --project=$Project `
    --filter="labels.run=$RunId" `
    --format="csv[no-heading](name,zone.basename(),machineType.basename(),status)"

$actions = @()
foreach ($line in $instances) {
    if (-not $line) { continue }
    $p = $line -split ","
    $name = $p[0]
    $zone = $p[1]
    $machine = $p[2]
    $status = $p[3]
    $workerId = Get-WorkerId $name
    $done = if ($hands.ContainsKey($workerId)) { [int]$hands[$workerId] } else { 0 }
    $req = if ($requested.ContainsKey($workerId)) { [int]$requested[$workerId] } else { 0 }
    if ($done -ge $TargetHands) {
        continue
    }

    $region = Get-Region $zone
    $free = Get-FreePreemptibleCpu $region

    if ($machine -eq "n2-highcpu-4" -and $status -eq "RUNNING" -and -not $KeepRunningSmall -and ($free + 4) -ge 16) {
        Write-Output "running-upgrade-start $name worker=$workerId done=$done region=$region free_cpu=$free"
        Invoke-GcloudQuiet @(
            "compute", "instances", "stop", $name,
            "--project=$Project",
            "--zone=$zone",
            "--quiet"
        )
        Invoke-GcloudQuiet @(
            "compute", "instances", "set-machine-type", $name,
            "--project=$Project",
            "--zone=$zone",
            "--machine-type=n2-highcpu-16",
            "--quiet"
        )
        Invoke-GcloudQuiet @(
            "compute", "instances", "add-metadata", $name,
            "--project=$Project",
            "--zone=$zone",
            "--metadata=rayon-threads=16",
            "--quiet"
        )
        Invoke-GcloudQuiet @(
            "compute", "instances", "start", $name,
            "--project=$Project",
            "--zone=$zone",
            "--quiet"
        )
        $actions += "stopped-upgraded $name worker=$workerId $done/$TargetHands"
    } elseif ($status -eq "RUNNING" -and $req -gt 0 -and $req -lt $TargetHands -and $done -ge $req) {
        Write-Output "reset-old-target $name worker=$workerId done=$done requested=$req target=$TargetHands"
        Invoke-GcloudQuiet @(
            "compute", "instances", "reset", $name,
            "--project=$Project",
            "--zone=$zone",
            "--quiet"
        )
        $actions += "reset-old-target $name worker=$workerId $done/$TargetHands"
    } elseif ($status -ne "TERMINATED") {
        continue
    } elseif ($machine -eq "n2-highcpu-4" -and $free -ge 16) {
        Write-Output "upgrade-start $name worker=$workerId done=$done region=$region free_cpu=$free"
        Invoke-GcloudQuiet @(
            "compute", "instances", "set-machine-type", $name,
            "--project=$Project",
            "--zone=$zone",
            "--machine-type=n2-highcpu-16",
            "--quiet"
        )
        Invoke-GcloudQuiet @(
            "compute", "instances", "add-metadata", $name,
            "--project=$Project",
            "--zone=$zone",
            "--metadata=rayon-threads=16",
            "--quiet"
        )
        Invoke-GcloudQuiet @(
            "compute", "instances", "start", $name,
            "--project=$Project",
            "--zone=$zone",
            "--quiet"
        )
        $actions += "upgraded $name worker=$workerId $done/$TargetHands"
    } elseif ($free -ge ($(if ($machine -eq "n2-highcpu-4") { 4 } else { 16 }))) {
        Write-Output "resume-start $name worker=$workerId done=$done machine=$machine region=$region free_cpu=$free"
        Invoke-GcloudQuiet @(
            "compute", "instances", "start", $name,
            "--project=$Project",
            "--zone=$zone",
            "--quiet"
        )
        $actions += "resumed $name worker=$workerId $done/$TargetHands"
    } else {
        $actions += "blocked $name worker=$workerId $done/$TargetHands free_cpu=$free"
    }
}

if ($actions.Count -eq 0) {
    Write-Output "No stopped below-target VMs needed action."
} else {
    $actions | ForEach-Object { Write-Output $_ }
}
