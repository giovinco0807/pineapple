param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$RunDir,
    [string]$CandidateLibrary,
    [string]$CandidateSha256 = "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d",
    [string]$ReferenceLibrary = "outputs/gcp_runs/regular-hu-m31-step6c-quality-20260717-003/package_src/native/release/libofc_hu_m3_engine.so",
    [string]$ReferenceSha256 = "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0",
    [string]$FeatureEncoder = "outputs/gcp_runs/regular-hu-m31-step6c-quality-20260717-003/package_src/target/release/libofc_stage3_feature_encoder.so",
    [string]$PlanPath = "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/full100_plan_v1.json",
    [string]$RootDir = "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/tail_reselection_v2/roots",
    [string]$TailMergeDir = "outputs/hu_joint_policy/m31_t3_step6d/spot_v2_merge/regular-hu-m31-c02-tail-v2-20260717-001",
    [string[]]$Jobs = @(),
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [switch]$PackageOnly,
    [switch]$AuthorizeOnly,
    [switch]$Launch,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $RunDir) {
    $RunDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
}
elseif (-not [IO.Path]::IsPathRooted($RunDir)) {
    $RunDir = Join-Path $repoRoot $RunDir
}
$RunDir = [IO.Path]::GetFullPath($RunDir)

function Resolve-Full100Path([string]$Value) {
    $path = if ([IO.Path]::IsPathRooted($Value)) {
        $Value
    }
    else {
        Join-Path $repoRoot $Value
    }
    return (Resolve-Path -LiteralPath $path).Path
}

$selectedModes = @(
    @(
        $PackageOnly.IsPresent,
        $AuthorizeOnly.IsPresent,
        $Launch.IsPresent,
        $Resume.IsPresent
    ) | Where-Object { $_ }
)
if ($selectedModes.Count -ne 1) {
    throw "Choose exactly one full100 mode"
}
if ($RunName -cnotmatch '^[a-z0-9][a-z0-9-]{2,46}[a-z0-9]$') {
    throw "RunName is not a safe bounded full100 identity"
}
if (($PackageOnly -or $AuthorizeOnly -or $Launch) -and $Jobs.Count -ne 0) {
    throw "Only full100 Resume accepts -Jobs"
}
if ($Resume -and $Jobs.Count -eq 0) {
    throw "Full100 Resume requires the exact incomplete job set"
}
if (($Launch -or $Resume) -and (
        $ProjectId -cne "ofc-solver-485418" -or
        $Bucket -cne "pokerhu-ofc-solver-485418-training"
    )) {
    throw "Full100 launch target differs from the frozen authorization"
}

$oldPythonPath = $env:PYTHONPATH
$oldDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
try {
    $arguments = @(
        "-m", "ofc_regular.hu_m31_t3_step6d_full100_spot_v1"
    )
    if ($PackageOnly) {
        if (-not $CandidateLibrary) {
            throw "PackageOnly requires -CandidateLibrary"
        }
        if (
            $CandidateSha256 -cnotmatch '^[0-9a-f]{64}$' -or
            $ReferenceSha256 -cnotmatch '^[0-9a-f]{64}$'
        ) {
            throw "Full100 binary SHA-256 values must be lowercase"
        }
        $arguments += @(
            "package",
            "--run-name", $RunName,
            "--output-dir", $RunDir,
            "--repository-root", $repoRoot,
            "--candidate-library", (Resolve-Full100Path $CandidateLibrary),
            "--candidate-sha256", $CandidateSha256,
            "--reference-library", (Resolve-Full100Path $ReferenceLibrary),
            "--reference-sha256", $ReferenceSha256,
            "--feature-encoder", (Resolve-Full100Path $FeatureEncoder),
            "--plan-path", (Resolve-Full100Path $PlanPath),
            "--root-dir", (Resolve-Full100Path $RootDir),
            "--tail-summary", (
                Resolve-Full100Path (Join-Path $TailMergeDir "summary.json")
            ),
            "--tail-validation", (
                Resolve-Full100Path (Join-Path $TailMergeDir "validation.json")
            )
        )
    }
    elseif ($AuthorizeOnly) {
        $arguments += @("authorize", "--run-dir", $RunDir)
    }
    elseif ($Launch) {
        $arguments += @(
            "launch", "--run-dir", $RunDir,
            "--project", $ProjectId, "--bucket", $Bucket
        )
    }
    else {
        $arguments += @(
            "resume", "--run-dir", $RunDir,
            "--jobs", ($Jobs -join ","),
            "--project", $ProjectId, "--bucket", $Bucket
        )
    }
    & python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Full100 lifecycle command failed ($LASTEXITCODE)"
    }
}
finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONDONTWRITEBYTECODE = $oldDontWriteBytecode
}
