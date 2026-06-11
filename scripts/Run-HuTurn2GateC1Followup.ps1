param(
    [string]$CacheDir = "D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_gate_c1_5k_mc512",
    [string]$Model = "models/hu_turn2_stage8_pilot_2000_mc512_reference_override_cached_rank_wide.pt",
    [string]$OutputDir = "outputs/hu_turn2_stage1_pilot_training_5000_c1b_false_positive_autopsy",
    [string]$Device = "auto",
    [ValidateSet("val", "test", "holdout")]
    [string]$ThresholdSplit = "test",
    [int]$TargetMinStates = 5000,
    [int]$TargetMaxStates = 10000,
    [int]$HighMcLimit = 120,
    [double]$FatalLossLimit = 1.25
)

$ErrorActionPreference = "Stop"

python -m ofc_regular.analyze_hu_turn2_gate_c1_followup `
    --cache-dir $CacheDir `
    --model $Model `
    --output-dir $OutputDir `
    --device $Device `
    --threshold-split $ThresholdSplit `
    --target-min-states $TargetMinStates `
    --target-max-states $TargetMaxStates `
    --high-mc-limit $HighMcLimit `
    --fatal-loss-limit $FatalLossLimit
