param(
    [string]$CacheDir = "D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_stage1_pilot_2000_mc512",
    [string]$OutputDir = "outputs/hu_turn2_stage1_pilot_training_2000_mc512",
    [string]$ModelOutput = "models/hu_turn2_stage8_pilot_2000_mc512_reference_override_cached_rank_wide.pt",
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

python -m ofc_regular.train_hu_turn2_pilot_model `
    --cache-dir $CacheDir `
    --output-dir $OutputDir `
    --model-output $ModelOutput `
    --epochs 30 `
    --batch-action-rows 32768 `
    --learning-rate 0.0005 `
    --weight-decay 0.00005 `
    --hidden-layer-sizes 1024,512,256 `
    --dropout 0.05 `
    --ranking-loss-weight 0.05 `
    --listwise-loss-weight 0.05 `
    --gate-loss-weight 0.2 `
    --seed 2026061702 `
    --device $Device
