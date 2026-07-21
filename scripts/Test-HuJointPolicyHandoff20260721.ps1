# Test-HuJointPolicyHandoff20260721.ps1
#
# 2026-07-21引継ぎ文書の期待値をread-onlyで自動照合する。
# リポジトリ状態は一切変更しない。
#
# 使い方 (リポジトリルートで):
#   powershell -File scripts\Test-HuJointPolicyHandoff20260721.ps1
#   powershell -File scripts\Test-HuJointPolicyHandoff20260721.ps1 -RunTests
#
# -RunTests を付けるとStep12b/12c回帰(約4分、期待275 passed)も実行する。

[CmdletBinding()]
param(
    [switch]$RunTests
)

$ErrorActionPreference = "Stop"

$expected = @{
    Branch          = "codex/regular-ofc-pineapple"
    Head            = "623e3948cb70b05be2958f0b1c63b6f02c7fb756"
    ProfileSha      = "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
    ReceiptFileSha  = "322eea21eb52bff767539127f29677951a074cf87aa4b0f37eace172ffd4235f"
    ReceiptInnerSha = "31e675593ca168fc899a6dc34083a6f54c8258aabbb62dcd71906d56d77161ef"
    TestsPassed     = 275
}

$receiptPath = "outputs/hu_joint_policy/m31_t3_step6d/step12c_pair_v1_actual/phase2_schema_failure_closeout_receipt.json"
$requiredDocs = @(
    "docs/hu_joint_policy_full_hand_rl_milestones_20260718.md",
    "docs/hu_joint_policy_m31_t3_step12c_phase2_schema_failure_20260721.md",
    "docs/hu_joint_policy_m31_t3_step12d_entrypoint_decision_20260721.md",
    "docs/hu_joint_policy_m30_t4_completion_audit.md",
    "docs/regular_ofc_ai_handoff_status.md",
    "configs/hu_joint_policy_m30_t4_runtime.json",
    $receiptPath
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$failures = 0

function Write-Check {
    param([string]$Name, [bool]$Ok, [string]$Detail)
    if ($Ok) {
        Write-Host ("[PASS] {0}" -f $Name)
    } else {
        Write-Host ("[FAIL] {0} : {1}" -f $Name, $Detail)
        $script:failures++
    }
}

# 1. branch / HEAD
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
Write-Check "branch" ($branch -eq $expected.Branch) `
    ("actual={0} expected={1}" -f $branch, $expected.Branch)

# 引継ぎ時点のbaseline HEADが現在の履歴に含まれることを確認する。
# (引継ぎ後の追加コミットは許容。巻き戻し・書き換えは検出する)
git merge-base --is-ancestor $expected.Head HEAD
Write-Check "baseline HEAD is ancestor of current HEAD" ($LASTEXITCODE -eq 0) `
    ("baseline={0} current={1}" -f $expected.Head, (git rev-parse HEAD).Trim())

# 2. ai_profiles.py hash
$profileSha = (Get-FileHash "src/ofc_regular/ai_profiles.py" -Algorithm SHA256).Hash.ToLower()
Write-Check "ai_profiles.py SHA-256" ($profileSha -eq $expected.ProfileSha) `
    ("actual={0}" -f $profileSha)

# 3. 主要資料の存在
foreach ($doc in $requiredDocs) {
    Write-Check ("exists: {0}" -f $doc) (Test-Path $doc) "missing"
}

# 4. closeout receipt: ファイルSHA-256と内部receipt_sha256は別の値。
#    引継ぎ文書の 31e675... は内部フィールドの値である。
if (Test-Path $receiptPath) {
    $fileSha = (Get-FileHash $receiptPath -Algorithm SHA256).Hash.ToLower()
    Write-Check "closeout receipt file SHA-256" ($fileSha -eq $expected.ReceiptFileSha) `
        ("actual={0}" -f $fileSha)

    $receipt = Get-Content -Encoding utf8 $receiptPath -Raw | ConvertFrom-Json
    $innerSha = $receipt.receipt_sha256
    Write-Check "closeout receipt inner receipt_sha256" ($innerSha -eq $expected.ReceiptInnerSha) `
        ("actual={0}" -f $innerSha)
}

# 5. Step12b/12c回帰 (optional)
if ($RunTests) {
    Write-Host "Running Step12b/12c regression (expect 275 passed, ~4 min)..."
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $tests = @(
        Get-ChildItem tests -Filter "test_hu_m31_t3_step6d_rearm2_diagnostic_step12b_*.py"
        Get-ChildItem tests -Filter "test_hu_m31_t3_step6d_rearm2_diagnostic_step12c_*.py"
    ) | Sort-Object FullName | ForEach-Object { $_.FullName }

    $output = python -m pytest -q -p no:cacheprovider @tests
    $summary = ($output | Select-Object -Last 3) -join " "
    $pattern = ("{0} passed" -f $expected.TestsPassed)
    Write-Check "Step12b/12c regression" ($summary -match $pattern) `
        ("summary: {0}" -f $summary)
} else {
    Write-Host "[SKIP] Step12b/12c regression (-RunTests で実行、期待: 275 passed)"
}

Write-Host ""
if ($failures -eq 0) {
    Write-Host "ALL CHECKS PASSED"
    exit 0
} else {
    Write-Host ("{0} CHECK(S) FAILED" -f $failures)
    exit 1
}
