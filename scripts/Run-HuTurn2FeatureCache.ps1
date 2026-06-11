param(
    [string]$TeacherInput = "outputs/hu_turn2_stage1_batch9_3_pilot_2000_mc512/hu_turn2_pilot_2000_mc512.jsonl",
    [string]$BucketSidecarDir = "outputs/hu_turn2_stage1_batch9_3_pilot_2000_mc512",
    [string]$CacheDir = "D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_stage1_pilot_2000_mc512",
    [string]$DType = "float32",
    [int]$Seed = 2026061701
)

$ErrorActionPreference = "Stop"

python -m ofc_regular.build_hu_turn2_pilot_feature_cache `
    --input $TeacherInput `
    --bucket-sidecar-dir $BucketSidecarDir `
    --cache-dir $CacheDir `
    --dtype $DType `
    --seed $Seed `
    --force
