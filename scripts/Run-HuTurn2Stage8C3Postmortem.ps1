param(
    [string]$C3Dir = "outputs\evals\hu_turn2_stage8_c3_larger_seat_swap",
    [string]$C2Dir = "outputs\evals\hu_turn2_stage8_c2_small",
    [string]$C1fDir = "outputs\hu_turn2_stage1_pilot_training_c1f_expanded_calibration",
    [string]$OutputDir = "outputs\evals\hu_turn2_stage8_c3_postmortem",
    [string]$GcpRunDir = "",
    [int]$TopLossLimit = 100
)

$ErrorActionPreference = "Stop"

$cmd = @(
    "-m", "ofc_regular.analyze_hu_turn2_stage8_c3_postmortem",
    "--c3-dir", $C3Dir,
    "--c2-dir", $C2Dir,
    "--c1f-dir", $C1fDir,
    "--output-dir", $OutputDir,
    "--top-loss-limit", "$TopLossLimit"
)

if ($GcpRunDir -ne "") {
    $cmd += @("--gcp-run-dir", $GcpRunDir)
}

python @cmd
