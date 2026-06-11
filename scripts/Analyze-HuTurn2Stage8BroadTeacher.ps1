param(
    [string]$Teacher = "outputs/hu_turn2_stage8_20k_mc512/hu_turn2_stage8_20k_mc512.jsonl",
    [string]$OutputDir = "outputs/hu_turn2_stage8_20k_mc512",
    [string]$SummaryDir = "",
    [string]$PoolDir = "outputs/hu_turn2_stage8_20k_mc512/pools",
    [string]$RunName = "",
    [int]$ExpectedTotal = 20000,
    [int]$ExpectedPerBucket = 4000,
    [int]$FutureSamples = 512
)

$ErrorActionPreference = "Stop"

$argsList = @(
    "-m", "ofc_regular.analyze_hu_turn2_stage8_broad_teacher",
    "--teacher", $Teacher,
    "--output-dir", $OutputDir,
    "--expected-total", "$ExpectedTotal",
    "--expected-per-bucket", "$ExpectedPerBucket",
    "--future-samples", "$FutureSamples",
    "--pool-dir", $PoolDir
)
if ($SummaryDir) {
    $argsList += @("--summary-dir", $SummaryDir)
}
if ($RunName) {
    $argsList += @("--run-name", $RunName)
}

python @argsList
