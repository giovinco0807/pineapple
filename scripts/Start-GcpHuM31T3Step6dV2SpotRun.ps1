param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$RunDir,
    [string]$CandidateLibrary,
    [string]$CandidateSha256,
    [ValidateSet(
        "candidate01",
        "candidate02_compact_scorer",
        "candidate02_compact_scorer_tail_v2"
    )]
    [string]$ContractVariant = "candidate01",
    [string]$ReferenceLibrary = "outputs/gcp_runs/regular-hu-m31-step6c-quality-20260717-003/package_src/native/release/libofc_hu_m3_engine.so",
    [string]$ReferenceSha256 = "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0",
    [string]$FeatureEncoder = "outputs/gcp_runs/regular-hu-m31-step6c-quality-20260717-003/package_src/target/release/libofc_stage3_feature_encoder.so",
    [string[]]$Jobs = @(),
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string[]]$Zones = @("asia-northeast1-b", "asia-northeast1-c"),
    [switch]$PackageOnly,
    [switch]$AuthorizeOnly,
    [switch]$Launch,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $RunDir) { $RunDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName" }
elseif (-not [IO.Path]::IsPathRooted($RunDir)) { $RunDir = Join-Path $repoRoot $RunDir }
$RunDir = [IO.Path]::GetFullPath($RunDir)
function Resolve-Step6dV2Path([string]$Value) {
    $path = if ([IO.Path]::IsPathRooted($Value)) { $Value } else { Join-Path $repoRoot $Value }
    return (Resolve-Path -LiteralPath $path).Path
}

$selectedModes = @(
    @($PackageOnly.IsPresent, $AuthorizeOnly.IsPresent, $Launch.IsPresent, $Resume.IsPresent) |
        Where-Object { $_ }
)
if ($selectedModes.Count -ne 1) {
    throw "Choose exactly one of -PackageOnly, -AuthorizeOnly, -Launch, or -Resume"
}
if ($RunName -cnotmatch '^[a-z0-9][a-z0-9-]{2,46}[a-z0-9]$') {
    throw "RunName is not a safe bounded Step 6d v2 identity"
}
if ($PackageOnly -and $Jobs.Count -ne 0) {
    throw "PackageOnly does not accept launch jobs"
}
if ($AuthorizeOnly -and $Jobs.Count -ne 0) {
    throw "AuthorizeOnly does not accept launch jobs"
}
if ($Launch -and ($Jobs.Count -ne 1 -or $Jobs[0] -cne "all")) {
    throw "Initial Launch requires exactly -Jobs all"
}
if ($Resume -and $Jobs.Count -eq 0) {
    throw "Resume requires the exact complete set of incomplete jobs proven at preflight"
}
if (($Launch -or $Resume) -and ($ProjectId -cne "ofc-solver-485418" -or
        $Bucket -cne "pokerhu-ofc-solver-485418-training" -or
        $Zones.Count -ne 2 -or $Zones[0] -cne "asia-northeast1-b" -or
        $Zones[1] -cne "asia-northeast1-c")) {
    throw "Launch/resume target differs from the fixed Step 6d v2 authorization"
}

$oldPythonPath = $env:PYTHONPATH
$oldDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
try {
    $arguments = @("-m", "ofc_regular.hu_m31_t3_step6d_spot_v2")
    if ($PackageOnly) {
        if (-not $CandidateLibrary -or -not $CandidateSha256) {
            throw "PackageOnly requires -CandidateLibrary and its accepted -CandidateSha256"
        }
        if ($CandidateSha256 -cnotmatch '^[0-9a-f]{64}$' -or
            $ReferenceSha256 -cnotmatch '^[0-9a-f]{64}$') {
            throw "Candidate/reference SHA-256 values must be explicit lowercase digests"
        }
        $candidate = Resolve-Step6dV2Path $CandidateLibrary
        $reference = Resolve-Step6dV2Path $ReferenceLibrary
        $feature = Resolve-Step6dV2Path $FeatureEncoder
        $arguments += @(
            "package", "--run-name", $RunName, "--output-dir", $RunDir,
            "--repository-root", $repoRoot,
            "--candidate-library", $candidate, "--candidate-sha256", $CandidateSha256,
            "--reference-library", $reference, "--reference-sha256", $ReferenceSha256,
            "--feature-encoder", $feature,
            "--contract-variant", $ContractVariant
        )
    }
    elseif ($AuthorizeOnly) {
        $arguments += @("authorize", "--run-dir", $RunDir)
    }
    elseif ($Launch) {
        $arguments += @(
            "launch", "--run-dir", $RunDir, "--jobs", ($Jobs -join ","),
            "--project", $ProjectId, "--bucket", $Bucket
        )
        foreach ($zone in $Zones) { $arguments += @("--zone", $zone) }
    }
    else {
        $arguments += @(
            "resume", "--run-dir", $RunDir, "--jobs", ($Jobs -join ","),
            "--project", $ProjectId, "--bucket", $Bucket
        )
        foreach ($zone in $Zones) { $arguments += @("--zone", $zone) }
    }
    & python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Step 6d v2 lifecycle command failed ($LASTEXITCODE)" }
}
finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONDONTWRITEBYTECODE = $oldDontWriteBytecode
}
