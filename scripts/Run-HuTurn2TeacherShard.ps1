param(
    [string]$SourceBucket = "natural",
    [int]$Samples = 100,
    [int]$FutureSamples = 16,
    [int]$Seed = 2026061101,
    [int]$MaxHands = 1000000,
    [int]$PrefilterFutureSamples = 0,
    [int]$PredictionThreads = 1,
    [int]$ProgressEvery = 10,
    [string]$Stage3FeatureEncoderMode = "rust_direct",
    [ValidateSet("stage3_reference_default", "stage7_m5_r10")]
    [string]$T3Continuation = "stage3_reference_default",
    [string]$Output = "outputs/hu_turn2_stage1/chunks/hu_turn2_teacher_natural_0000.jsonl",
    [string]$SummaryOutput = "",
    [switch]$EnableM2T4Search,
    [int]$M2T4CandidateSamples = 16,
    [int]$M2T4EvaluationSamples = 32,
    [int]$M2T4Seed = 2026071303,
    [int]$M2T4CandidateSeed = 2026071301,
    [int]$M2T4EvaluationSeed = 2026071302,
    [string]$M2T4RunId = "hu-m2-t2-continuation-v1",
    [switch]$NoBatchedContinuation
)

$ErrorActionPreference = "Stop"

$argsList = @(
    "-m", "ofc_regular.hu_turn2_teacher_data",
    "--samples", "$Samples",
    "--future-samples", "$FutureSamples",
    "--seed", "$Seed",
    "--source-bucket", "$SourceBucket",
    "--max-hands", "$MaxHands",
    "--prediction-threads", "$PredictionThreads",
    "--progress-every", "$ProgressEvery",
    "--t3-continuation", "$T3Continuation",
    "--stage3-feature-encoder-mode", "$Stage3FeatureEncoderMode",
    "--output", "$Output"
)
if (-not $NoBatchedContinuation) {
    $argsList += @("--use-batched-continuation")
}
if ($PrefilterFutureSamples -gt 0) {
    $argsList += @("--prefilter-future-samples", "$PrefilterFutureSamples")
}
if ($SummaryOutput -ne "") {
    $argsList += @("--summary-output", "$SummaryOutput")
}
if ($EnableM2T4Search) {
    $argsList += @(
        "--enable-m2-t4-search",
        "--m2-t4-candidate-samples", "$M2T4CandidateSamples",
        "--m2-t4-evaluation-samples", "$M2T4EvaluationSamples",
        "--m2-t4-seed", "$M2T4Seed",
        "--m2-t4-candidate-seed", "$M2T4CandidateSeed",
        "--m2-t4-evaluation-seed", "$M2T4EvaluationSeed",
        "--m2-t4-run-id", "$M2T4RunId"
    )
}

python @argsList
