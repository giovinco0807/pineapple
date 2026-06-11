param(
    [int]$GamesPerSeed = 1000,
    [int]$PredictionThreads = 1,
    [string]$OutputDir = "outputs/evals/stage7_candidate_A_golden_m5_r10"
)

$ErrorActionPreference = "Stop"

python -m ofc_regular.evaluate_stage7_production_candidate `
    --configs 5/10 `
    --games-per-seed $GamesPerSeed `
    --seeds 2026060904,2026063501,2026068201 `
    --stage7-model models/hu_turn3_stage7_reference_override_cached_rank_wide.pt `
    --stage3-model models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt `
    --turn3-model models/turn3_stage6.pkl `
    --opening-model models/opening_stage7_torch_wide.pt `
    --turn1-model models/turn1_stage6_torch_wide.pt `
    --turn2-model models/turn2_stage8.pkl `
    --prediction-threads $PredictionThreads `
    --progress-every 200 `
    --output-dir $OutputDir
