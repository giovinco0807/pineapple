param(
    [string]$CacheDir = "D:\ofc-pineapple-storage\regular-ofc-pineapple\feature_cache\hu_t2_stage8_20k_mc512",
    [string]$Model = "models\hu_turn2_stage8_broad_20k_mc512_reference_override_cached_rank_wide.pt",
    [string]$C1eDir = "outputs\hu_turn2_stage1_pilot_training_c1e_teacher_ev_feature_cache",
    [string]$OutputDir = "outputs\hu_turn2_stage1_pilot_training_c1f_expanded_calibration",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [int]$BatchSize = 32768
)

$ErrorActionPreference = "Stop"

python -m ofc_regular.analyze_hu_turn2_gate_c1f_expanded_calibration `
    --cache-dir $CacheDir `
    --model $Model `
    --c1e-dir $C1eDir `
    --output-dir $OutputDir `
    --device $Device `
    --batch-size $BatchSize
