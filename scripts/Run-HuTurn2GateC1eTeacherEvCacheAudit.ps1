param(
    [string]$TeacherInput = "outputs\hu_turn2_stage8_20k_mc512\hu_turn2_stage8_20k_mc512.jsonl",
    [string]$BucketSidecarDir = "outputs\hu_turn2_stage8_20k_mc512",
    [string]$CacheDir = "D:\ofc-pineapple-storage\regular-ofc-pineapple\feature_cache\hu_t2_stage8_20k_mc512",
    [string]$Model = "models\hu_turn2_stage8_pilot_2000_mc512_reference_override_cached_rank_wide.pt",
    [string]$C1dDir = "outputs\hu_turn2_stage1_pilot_training_c1d_threshold_repair",
    [string]$OutputDir = "outputs\hu_turn2_stage1_pilot_training_c1e_teacher_ev_feature_cache",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [int]$BatchSize = 32768
)

$ErrorActionPreference = "Stop"

python -m ofc_regular.analyze_hu_turn2_gate_c1e_teacher_ev_cache `
    --teacher-input $TeacherInput `
    --bucket-sidecar-dir $BucketSidecarDir `
    --cache-dir $CacheDir `
    --model $Model `
    --c1d-dir $C1dDir `
    --output-dir $OutputDir `
    --device $Device `
    --batch-size $BatchSize
