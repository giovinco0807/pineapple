param(
    [string]$CacheDir = "D:\ofc-pineapple-storage\regular-ofc-pineapple\feature_cache\hu_t2_stage8_20k_mc512",
    [string]$Model = "models\hu_turn2_stage8_broad_20k_mc512_reference_override_cached_rank_wide.pt",
    [string]$C1eDir = "outputs\hu_turn2_stage1_pilot_training_c1e_teacher_ev_feature_cache",
    [string]$C1fDir = "outputs\hu_turn2_stage1_pilot_training_c1f_expanded_calibration",
    [string]$OutputDir = "outputs\evals\hu_turn2_stage8_c2_small",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [int]$BatchSize = 32768,
    [switch]$RunSeatSwap,
    [int]$GamesPerSeed = 300,
    [string]$Seeds = "2026061801,2026061802,2026061803",
    [int]$SeedStride = 1000000,
    [int]$PredictionThreads = 1,
    [int]$OpeningLookaheadSamples = 64,
    [int]$ProgressEvery = 0
)

$ErrorActionPreference = "Stop"

$cmd = @(
    "-m", "ofc_regular.analyze_hu_turn2_stage8_c2_small",
    "--cache-dir", $CacheDir,
    "--model", $Model,
    "--c1e-dir", $C1eDir,
    "--c1f-dir", $C1fDir,
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--batch-size", "$BatchSize",
    "--games-per-seed", "$GamesPerSeed",
    "--seeds", $Seeds,
    "--seed-stride", "$SeedStride",
    "--prediction-threads", "$PredictionThreads",
    "--opening-lookahead-samples", "$OpeningLookaheadSamples",
    "--progress-every", "$ProgressEvery"
)

if ($RunSeatSwap) {
    $cmd += "--run-seat-swap"
}

python @cmd
