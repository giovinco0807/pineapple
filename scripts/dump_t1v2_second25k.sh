#!/usr/bin/env bash
# Encode the merged 25k T1 second-seat corpus (18k + 7k top-up) in one pass.
#
# A script file rather than an inline `wsl.exe -e bash -lc '...'` string, because
# git-bash rewrites absolute /mnt/... paths inside the quoted payload (MSYS path
# mangling) and the mangled command then "executes" the jsonl line by line --
# 25,000 "command not found" and no features.  A file crosses the boundary
# byte-for-byte.
#
# Both label files are passed together: labelgen_feature_dump sorts records by
# offset and the two corpora overlap in offset value (0..17999 and 0..6999) --
# that is fine, offsets are not identity here, each record keeps its own group;
# disjointness of the underlying hands lives in the hand_seed_base blocks
# (952,000,000 vs 952,100,000).
set -euo pipefail
REPO=/mnt/c/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple/regular-ofc-pineapple
OUT=/mnt/d/ofc_data/t1v2/feat_second25k
mkdir -p "$OUT"
RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-14} "$REPO/target/release/labelgen_feature_dump" \
  "$OUT" \
  /mnt/d/ofc_data/t1v2/second_labels.jsonl \
  /mnt/d/ofc_data/t1v2/second_topup_labels.jsonl
