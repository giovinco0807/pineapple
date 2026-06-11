param(
  [string]$CacheDir = "D:\ofc-pineapple-storage\regular-ofc-pineapple\feature_cache\hu_t2_gate_c1_5k_mc512",
  [string]$Model = "models\hu_turn2_stage8_pilot_2000_mc512_reference_override_cached_rank_wide.pt",
  [string]$C1cDir = "outputs\hu_turn2_stage1_pilot_training_5000_c1b_false_positive_autopsy",
  [string]$OutputDir = "outputs\hu_turn2_stage1_pilot_training_c1d_threshold_repair",
  [ValidateSet("auto", "cuda", "cpu")]
  [string]$Device = "auto",
  [int]$BatchSize = 32768,
  [int]$HighMcLimit = 120
)

$ErrorActionPreference = "Stop"

python -m ofc_regular.analyze_hu_turn2_gate_c1d_threshold_repair `
  --cache-dir $CacheDir `
  --model $Model `
  --c1c-dir $C1cDir `
  --output-dir $OutputDir `
  --device $Device `
  --batch-size $BatchSize `
  --high-mc-limit $HighMcLimit
