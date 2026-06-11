param(
    [string]$OutputDir = "outputs\evals\hu_turn2_stage8_c3_larger_seat_swap",
    [int]$GamesPerSeed = 1000,
    [string]$Seeds = "2026061901,2026061902,2026061903,2026061904,2026061905",
    [string]$Configs = "2.5/0/0.9,2.75/0/0.9,2.5/0/0.925,2.5/0/0.7",
    [int]$SeedStride = 1000000,
    [int]$PredictionThreads = 1,
    [int]$OpeningLookaheadSamples = 64,
    [int]$ProgressEvery = 100,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [switch]$WriteDecisionLog
)

$ErrorActionPreference = "Stop"

$evalCmd = @(
    "-m", "ofc_regular.evaluate_hu_turn2_stage8_seat_swap",
    "--games-per-seed", "$GamesPerSeed",
    "--seeds", $Seeds,
    "--seed-stride", "$SeedStride",
    "--configs", $Configs,
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--prediction-threads", "$PredictionThreads",
    "--opening-lookahead-samples", "$OpeningLookaheadSamples",
    "--progress-every", "$ProgressEvery"
)

if ($WriteDecisionLog) {
    $evalCmd += "--write-decision-log"
}

python @evalCmd

python -m ofc_regular.analyze_hu_turn2_stage8_c3_larger_seat_swap `
    --input-dir $OutputDir `
    --output-dir $OutputDir
