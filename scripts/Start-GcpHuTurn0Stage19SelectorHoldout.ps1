param(
    [Parameter(Mandatory = $true)]
    [string]$SelectionJson,
    [string]$RunName = ("hu-t0-stage19-selector-holdout-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [string]$CandidateModel = "models/hu_turn0_stage3_top60_mc32_1000_hgb_baseline-delta_sew1_aug3.pkl",
    [int]$CandidateTopk = 60,
    [int]$PairedSeeds = 50000,
    [int]$GamesPerShard = 20,
    [long]$BaseSeed = 2026907001,
    [long]$SeedStride = 1000003,
    [int]$VmCount = 80,
    [string]$MachineType = "c4-highmem-2",
    [ValidateSet("legacy_v1", "tail_audit_v2")]
    [string]$AcceptancePolicy = "legacy_v1",
    [string]$TailAuditSummary = "",
    [switch]$CreateInstances,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath $SelectionJson)) {
        throw "Selection JSON not found: $SelectionJson"
    }
    $selection = Get-Content -LiteralPath $SelectionJson -Raw | ConvertFrom-Json
    if ($selection.decision -ne "Go" -or $null -eq $selection.selected) {
        throw "T0 selector selection is not Go; fresh holdout will not be launched"
    }
    $selected = $selection.selected
    $safeSelectorModel = [string]$selected.model
    $safeThreshold = [double]$selected.threshold
    if (-not (Test-Path -LiteralPath $safeSelectorModel)) {
        throw "Selected safe selector model not found: $safeSelectorModel"
    }
    if ($safeThreshold -lt 0.0 -or $safeThreshold -gt 1.0) {
        throw "Selected safe selector threshold must be in [0, 1]"
    }

    $tailAuditPlan = $null
    if ($AcceptancePolicy -eq "tail_audit_v2") {
        if (-not $TailAuditSummary -or -not (Test-Path -LiteralPath $TailAuditSummary)) {
            throw "Tail-audit v2 requires an existing -TailAuditSummary"
        }
        $tailAudit = Get-Content -LiteralPath $TailAuditSummary -Raw | ConvertFrom-Json
        $tailRecords = [int]$tailAudit.records
        $tailNegativeCount = [int]$tailAudit.label_counts.negative
        $tailNegativeRate = if ($tailRecords -gt 0) {
            $tailNegativeCount / [double]$tailRecords
        }
        else { 1.0 }
        if ($tailRecords -lt 30) { throw "Tail audit must contain at least 30 records" }
        if (@($tailAudit.missing_shards).Count -gt 0) { throw "Tail audit has missing shards" }
        if (-not [bool]$tailAudit.common_random_futures_verified) {
            throw "Tail audit common-random-future verification failed"
        }
        if (-not [bool]$tailAudit.action_mapping_verified) {
            throw "Tail audit action mapping verification failed"
        }
        if ([double]$tailAudit.delta_mean -le 0.0) {
            throw "Tail audit mean delta must be positive"
        }
        if ($tailNegativeRate -gt 0.20) {
            throw "Tail audit negative rate exceeds 0.20"
        }
        $tailAuditPlan = [ordered]@{
            artifact = $TailAuditSummary
            selection = "top30 realized whole-game losses from the prior fixed-candidate holdout"
            future_samples = 512
            minimum_records = 30
            delta_mean_min_exclusive = 0.0
            negative_rate_max = 0.20
            observed_records = $tailRecords
            observed_delta_mean = [double]$tailAudit.delta_mean
            observed_negative_rate = $tailNegativeRate
            common_random_futures_required = $true
            action_mapping_required = $true
        }
    }

    $acceptance = [ordered]@{
        selection_order = "The sole preregistered selector must pass every check."
        aggregate_avg_delta_per_hand_min_exclusive = 0.0
        aggregate_ci95_low_min = 0.0
        avg_delta_per_fire_min_exclusive = 0.0
        per_fire_ci95_low_min = 0.0
        minimum_fires = 50
        non_fired_nonzero_count = 0
        p99_fire_loss_max = 40.0
        enabled_seat_point_estimates_must_be_nonnegative = $true
        duplicate_event_count = 0
        seed_set_mismatch_configs = @()
    }
    if ($AcceptancePolicy -eq "legacy_v1") {
        $acceptance["max_fire_loss_max"] = 50.0
    }
    else {
        $acceptance["max_fire_loss_report_only"] = $true
        $acceptance["max_loss_rationale"] = (
            "Observed single-future maximum loss is sample-size dependent. " +
            "Safety is gated by p99 loss and independent MC512 replay of the worst realized tail."
        )
    }

    $runDir = Join-Path "outputs/gcp_runs" $RunName
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $planPath = Join-Path $runDir "preregistered_holdout.json"
    $planSchema = if ($AcceptancePolicy -eq "tail_audit_v2") {
        "hu_turn0_stage19_safe_selector_holdout_v2"
    }
    else { "hu_turn0_stage19_safe_selector_holdout_v1" }
    $plan = [ordered]@{
        name = "hu_turn0_stage19_safe_selector_fresh_holdout"
        status = "preregistered_before_fresh_holdout"
        schema = $planSchema
        acceptance_policy = $AcceptancePolicy
        fixed_chain = [ordered]@{
            fallback_profile = "stage18_p1"
            t1 = "stage18_p1"
            t2 = "stage9f_p2"
            t3 = "stage7_m5_r10"
            fl_ev = 10.227020614683454
            visibility_model = "hidden_discard"
        }
        candidate_model = $CandidateModel
        candidate_topk = $CandidateTopk
        safe_selector_model = $safeSelectorModel
        selector_selection_artifact = $SelectionJson
        candidates = @(
            [ordered]@{
                id = "stage19_safe_selector"
                priority = 1
                min_margin_by_seat = [ordered]@{first = 0.5; second = 999.0}
                safe_selector_threshold_by_seat = [ordered]@{
                    first = $safeThreshold
                    second = 1.0
                }
                allowed_seats = @("first")
            }
        )
        evaluation = [ordered]@{
            paired_seeds_per_config = $PairedSeeds
            games_per_shard = $GamesPerShard
            base_seed = $BaseSeed
            seed_stride = $SeedStride
            non_overlapping_with_all_prior_t0_runs = $true
            vm_count = $VmCount
            machine_type = $MachineType
            spot = $true
            no_threshold_adaptation_after_holdout = $true
        }
        acceptance = $acceptance
    }
    if ($tailAuditPlan -ne $null) { $plan["tail_audit"] = $tailAuditPlan }
    [System.IO.File]::WriteAllText(
        (Join-Path $repoRoot $planPath),
        (($plan | ConvertTo-Json -Depth 10) + "`n"),
        [System.Text.UTF8Encoding]::new($false)
    )

    $launchArgs = @{
        RunName = $RunName
        CandidateModel = $CandidateModel
        CandidateTopk = $CandidateTopk
        SafeSelectorModel = $safeSelectorModel
        SafeSelectorThresholdFirst = $safeThreshold
        SafeSelectorThresholdSecond = 1.0
        Configs = @("stage19_safe_selector/0.5/999/first")
        PairedSeedsPerConfig = $PairedSeeds
        GamesPerShard = $GamesPerShard
        BaseSeed = $BaseSeed
        SeedStride = $SeedStride
        VmCount = $VmCount
        MachineType = $MachineType
    }
    if ($CreateInstances) { $launchArgs.CreateInstances = $true }
    if ($DryRun) { $launchArgs.DryRun = $true }
    & (Join-Path $PSScriptRoot "Start-GcpHuTurn0CounterfactualRun.ps1") @launchArgs
    if ($LASTEXITCODE -ne 0) { throw "T0 Stage19 selector holdout launch failed" }
}
finally {
    Pop-Location
}
