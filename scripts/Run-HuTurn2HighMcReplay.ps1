param(
  [string]$Candidates = "outputs\hu_turn2_stage1_pilot_training_5000_c1b_false_positive_autopsy\high_mc_recheck_candidates.csv",
  [string]$CacheDir = "D:\ofc-pineapple-storage\regular-ofc-pineapple\feature_cache\hu_t2_gate_c1_5k_mc512",
  [string]$OutputDir = "outputs\hu_turn2_stage1_pilot_training_5000_c1b_false_positive_autopsy",
  [int]$McSamples = 2048,
  [int]$Limit = 0,
  [switch]$ReadinessOnly,
  [int]$PredictionThreads = 1,
  [string]$Stage3FeatureEncoderMode = "rust_direct"
)

$ErrorActionPreference = "Stop"

$argsList = @(
  "-m", "ofc_regular.replay_hu_turn2_event_high_mc",
  "--candidates", $Candidates,
  "--cache-dir", $CacheDir,
  "--output-dir", $OutputDir,
  "--mc-samples", "$McSamples",
  "--prediction-threads", "$PredictionThreads",
  "--stage3-feature-encoder-mode", $Stage3FeatureEncoderMode
)

if ($Limit -gt 0) {
  $argsList += @("--limit", "$Limit")
}

if ($ReadinessOnly) {
  $argsList += "--readiness-only"
}

python @argsList
