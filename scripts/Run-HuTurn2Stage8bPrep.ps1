param(
    [string]$CacheDir = "D:\ofc-pineapple-storage\regular-ofc-pineapple\feature_cache\hu_t2_stage8_20k_mc512",
    [string]$Model = "models\hu_turn2_stage8_broad_20k_mc512_reference_override_cached_rank_wide.pt",
    [string]$C1eDir = "outputs\hu_turn2_stage1_pilot_training_c1e_teacher_ev_feature_cache",
    [string]$C3PostmortemDir = "outputs\evals\hu_turn2_stage8_c3_postmortem",
    [string]$HighMcResultsJsonl = "outputs\evals\hu_turn2_stage8_c4_selected_high_mc_gcp_20260612b\high_mc_audit_results.jsonl",
    [string]$OutputDir = "outputs\hu_turn2_stage8b_prelarge_training",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [int]$BatchSize = 32768,
    [int]$MaxHighMcStates = 200
)

$ErrorActionPreference = "Stop"

python -m ofc_regular.prepare_hu_turn2_stage8b_training `
    --cache-dir $CacheDir `
    --model $Model `
    --c1e-dir $C1eDir `
    --c3-postmortem-dir $C3PostmortemDir `
    --output-dir $OutputDir `
    --device $Device `
    --batch-size $BatchSize `
    --max-high-mc-states $MaxHighMcStates `
    --high-mc-results-jsonl $HighMcResultsJsonl
