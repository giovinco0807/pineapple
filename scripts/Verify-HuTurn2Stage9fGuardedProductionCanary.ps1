param(
    [string]$OutputDir = "outputs/evals/hu_turn2_stage9f_guarded_production_canary",
    [string]$Preset = "configs/hu_turn2_stage9f_cse2_csemax2_bothseat_guarded_production_preset.json",
    [int]$MinimumDecisions = 20000,
    [int]$MinimumRealizedFires = 150,
    [switch]$SkipAuditRegeneration
)

$ErrorActionPreference = "Stop"

$summaryPath = Join-Path $OutputDir "summary.json"
$decisionsPath = Join-Path $OutputDir "topk_decisions.jsonl"
$auditDir = Join-Path $OutputDir "audit"
$verificationDir = Join-Path $OutputDir "guarded_preset_verification"

if (-not (Test-Path -LiteralPath $summaryPath)) {
    throw "Missing matchup summary: $summaryPath"
}
if (-not (Test-Path -LiteralPath $decisionsPath)) {
    throw "Missing topk decisions: $decisionsPath"
}
if (-not (Test-Path -LiteralPath $Preset)) {
    throw "Missing guarded preset: $Preset"
}

if (-not $SkipAuditRegeneration) {
    python -m ofc_regular.analyze_hu_turn2_stage9f_profile_canary `
        --decisions $decisionsPath `
        --matchup-summary $summaryPath `
        --output-dir $auditDir
    if ($LASTEXITCODE -ne 0) {
        throw "Stage9f guarded production canary audit failed with exit code $LASTEXITCODE"
    }
}

python -m ofc_regular.verify_hu_turn2_stage9f_guarded_preset `
    --preset $Preset `
    --audit-dir $auditDir `
    --minimum-decisions $MinimumDecisions `
    --minimum-realized-fires $MinimumRealizedFires `
    --verification-scope guarded_canary `
    --output-dir $verificationDir
if ($LASTEXITCODE -ne 0) {
    throw "Stage9f guarded production canary verification failed with exit code $LASTEXITCODE"
}

[pscustomobject]@{
    output_dir = $OutputDir
    preset = $Preset
    audit_dir = $auditDir
    verification_dir = $verificationDir
    verification_summary = (Join-Path $verificationDir "guarded_preset_verification.md")
    production_default = "No-Go until explicitly enabled"
    p2_fixed = "No-Go until explicitly accepted"
    rollback = "stage9f_off"
} | ConvertTo-Json -Depth 4
