param(
    [int]$Samples = 10,
    [int]$FutureSamples = 1,
    [int]$MaxActions = 0,
    [int]$Seed = 2026062401,
    [string]$Profile = "stage9f_p2",
    [string]$OpponentProfile = "stage9f_p2",
    [int]$OpeningLookaheadSamples = 1,
    [string]$Output = "outputs/hu_turn1_stage1_pilot/chunks/hu_turn1_stage1_pilot_0000.jsonl",
    [string]$SummaryOutput = "",
    [switch]$CollectTopkLog
)

$ErrorActionPreference = "Stop"

if ($Samples -le 0) { throw "Samples must be positive" }
if ($FutureSamples -le 0) { throw "FutureSamples must be positive" }
if ($MaxActions -lt 0) { throw "MaxActions must be non-negative" }
if ($OpeningLookaheadSamples -lt 0) { throw "OpeningLookaheadSamples must be non-negative" }

$argsList = @(
    "-m", "ofc_regular.hu_turn1_teacher_pilot",
    "--samples", "$Samples",
    "--future-samples", "$FutureSamples",
    "--max-actions", "$MaxActions",
    "--seed", "$Seed",
    "--profile", "$Profile",
    "--opponent-profile", "$OpponentProfile",
    "--opening-lookahead-samples", "$OpeningLookaheadSamples",
    "--output", "$Output"
)

if ($SummaryOutput -ne "") {
    $argsList += @("--summary-output", "$SummaryOutput")
}
else {
    $summary = [System.IO.Path]::ChangeExtension($Output, ".summary.json")
    $argsList += @("--summary-output", "$summary")
}

if ($CollectTopkLog) {
    $argsList += @("--collect-topk-log")
}

python @argsList
