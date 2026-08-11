#!/bin/sh
set -eu

python /app/deploy/verify_assembly.py \
  --assembly "${OFC_ASSEMBLY_PATH:-/app/assembly.json}" \
  --root "${OFC_ASSEMBLY_ROOT:-/app}"

if [ "$#" -eq 0 ] || [ "$1" = "serve" ]; then
  exec python -m uvicorn backend.ofc_webapp.api:app \
    --host 0.0.0.0 \
    --port "${PORT:-8080}" \
    --workers 1
fi

exec "$@"
