param(
    [string]$CacheDir = "D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_stage1_pilot_2000_mc512",
    [string]$Model = "models/hu_turn2_stage8_pilot_2000_mc512_reference_override_cached_rank_wide.pt",
    [string]$OutputDir = "outputs/hu_turn2_gate_c1_large_calibration_2000_mc512",
    [string]$Device = "auto",
    [ValidateSet("val", "test", "holdout")]
    [string]$ThresholdSplit = "test",
    [int]$TargetMinStates = 5000,
    [int]$TargetMaxStates = 10000
)

$ErrorActionPreference = "Stop"

python -m ofc_regular.analyze_hu_turn2_gate_c1_large_calibration `
    --cache-dir $CacheDir `
    --model $Model `
    --output-dir $OutputDir `
    --device $Device `
    --threshold-split $ThresholdSplit `
    --target-min-states $TargetMinStates `
    --target-max-states $TargetMaxStates
