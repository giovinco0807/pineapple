#!/usr/bin/env bash
set -euo pipefail
umask 077

: "${OFC_HU_RL_C4_PACKAGE_ROOT:?missing package root}"
: "${OFC_HU_RL_C4_MACHINE_ATTESTATION:?missing external machine attestation}"
: "${OFC_HU_RL_C4_RESULT:?missing write-once result path}"

[[ -d "$OFC_HU_RL_C4_PACKAGE_ROOT" ]] || {
  echo "package root is not a directory" >&2
  exit 1
}
[[ -f "$OFC_HU_RL_C4_MACHINE_ATTESTATION" ]] || {
  echo "external machine attestation is not a file" >&2
  exit 1
}
[[ ! -e "$OFC_HU_RL_C4_RESULT" ]] || {
  echo "write-once result path already exists" >&2
  exit 1
}

python3 - <<'PY'
import platform
import sys

if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 11):
    raise SystemExit("formal C4 benchmark requires exact CPython 3.11")
PY

cd -- "$OFC_HU_RL_C4_PACKAGE_ROOT"
export PYTHONPATH="$OFC_HU_RL_C4_PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
exec python3 -B -s scripts/run_hu_rl_c4_formal_benchmark.py execute \
  --manifest hu_rl_c4_formal_benchmark_manifest.json \
  --machine-attestation "$OFC_HU_RL_C4_MACHINE_ATTESTATION" \
  --output "$OFC_HU_RL_C4_RESULT"
