param(
    [string]$SourceBucket = "natural",
    [int]$Samples = 100,
    [int]$FutureSamples = 4096,
    [int]$Seed = 2026061101,
    [int]$PredictionThreads = 1,
    [string]$Output = "outputs/hu_turn2_stage1/chunks/hu_turn2_teacher_natural_0000.jsonl",
    [string]$SummaryOutput = ""
)

$ErrorActionPreference = "Stop"

$argsList = @(
    "-m", "ofc_regular.hu_turn2_teacher_data",
    "--samples", "$Samples",
    "--future-samples", "$FutureSamples",
    "--seed", "$Seed",
    "--source-bucket", "$SourceBucket",
    "--prediction-threads", "$PredictionThreads",
    "--progress-every", "10",
    "--opening-model", "models/opening_stage7_torch_wide.pt",
    "--turn1-model", "models/turn1_stage6_torch_wide.pt",
    "--turn2-model", "models/turn2_stage8.pkl",
    "--turn3-model", "models/turn3_stage6.pkl",
    "--stage7-model", "models/hu_turn3_stage7_reference_override_cached_rank_wide.pt",
    "--stage3-reference-model", "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt",
    "--output", "$Output"
)
if ($SummaryOutput -ne "") {
    $argsList += @("--summary-output", "$SummaryOutput")
}

python @argsList
