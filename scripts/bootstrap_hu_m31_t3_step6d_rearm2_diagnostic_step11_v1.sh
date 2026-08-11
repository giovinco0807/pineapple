#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# Stage-0 wrapper for the single-VM Step 11 lifecycle smoke.  The controller
# supplies four immutable initial metadata files.  The signed claim is
# deliberately absent at instance insertion and is released only after the
# controller has observed the provider-issued instance id.

readonly METADATA_HOST="metadata.google.internal"
readonly METADATA_ROOT="/computeMetadata/v1/instance/attributes"
readonly WORK_ROOT="/opt/ofc-step11"
readonly CLAIM_WAIT_SECONDS=300
readonly FAILURE_SHUTDOWN_SECONDS=120
readonly SHUTDOWN_MARKER="/run/ofc-m31-step11-shutdown-requested"

shutdown_attempted=0
phase="stage0_entry"

mark_phase() {
  phase=$1
  printf 'OFC_STEP11_STAGE0 phase=%s\n' "$phase" >&2
}

request_shutdown_once() {
  if [[ "$shutdown_attempted" -eq 1 ]]; then
    return 0
  fi
  shutdown_attempted=1
  if [[ -e "$SHUTDOWN_MARKER" ]]; then
    return 0
  fi
  if ! (
    set -o noclobber
    printf 'shutdown-requested\n' > "$SHUTDOWN_MARKER"
  ) 2>/dev/null; then
    if [[ -e "$SHUTDOWN_MARKER" ]]; then
      return 0
    fi
    return 0
  fi
  if ! /usr/bin/timeout "$FAILURE_SHUTDOWN_SECONDS" /sbin/shutdown -h now \
    >/dev/null 2>&1; then
    if [[ -f "$SHUTDOWN_MARKER" && ! -L "$SHUTDOWN_MARKER" ]] \
      && [[ "$(<"$SHUTDOWN_MARKER")" == "shutdown-requested" ]]; then
      rm -f -- "$SHUTDOWN_MARKER"
    fi
    return 1
  fi
}

finish() {
  local rc=$?
  trap - EXIT ERR HUP INT TERM
  if [[ "$rc" -ne 0 ]]; then
    printf 'OFC_STEP11_STAGE0 phase=%s status=failed rc=%d\n' \
      "$phase" "$rc" >&2
    request_shutdown_once
  fi
  exit "$rc"
}

trap finish EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$(id -u)" -ne 0 ]]; then
  exit 77
fi

mark_phase "apt_update"
/usr/bin/timeout 300 /usr/bin/env DEBIAN_FRONTEND=noninteractive \
  /usr/bin/apt-get update -o Acquire::Retries=2
mark_phase "apt_install"
/usr/bin/timeout 300 /usr/bin/env DEBIAN_FRONTEND=noninteractive \
  /usr/bin/apt-get install -y --no-install-recommends \
  ca-certificates python3 python3-venv

mark_phase "initial_metadata"
/usr/bin/install -d -m 0700 "$WORK_ROOT"

readonly METADATA_FETCHER="$WORK_ROOT/metadata_fetch.py"
/usr/bin/python3 - "$METADATA_FETCHER" <<'PY'
from __future__ import annotations

import http.client
import os
import sys
import urllib.parse
from pathlib import Path

destination = Path(sys.argv[1])
source = """from __future__ import annotations

import http.client
import os
import sys
import urllib.parse
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit(64)
key = sys.argv[1]
destination = Path(sys.argv[2])
if (
    not key
    or key.startswith(".")
    or "/" in key
    or "\\\\" in key
    or destination.exists()
    or destination.is_symlink()
):
    raise SystemExit(65)
path = (
    "/computeMetadata/v1/instance/attributes/"
    + urllib.parse.quote(key, safe="")
)
connection = http.client.HTTPConnection(
    "metadata.google.internal", 80, timeout=5
)
try:
    connection.request("GET", path, headers={"Metadata-Flavor": "Google"})
    response = connection.getresponse()
    body = response.read()
finally:
    connection.close()
if response.status == 404:
    raise SystemExit(75)
if response.status != 200 or response.getheader("Metadata-Flavor") != "Google":
    raise SystemExit(69)
destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
descriptor = os.open(
    destination,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    0o600,
)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(body)
    handle.flush()
    os.fsync(handle.fileno())
"""
descriptor = os.open(
    destination,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    0o600,
)
with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
    handle.write(source)
    handle.flush()
    os.fsync(handle.fileno())
PY

fetch_initial() {
  local key=$1
  local destination=$2
  /usr/bin/python3 "$METADATA_FETCHER" "$key" "$destination"
}

fetch_initial \
  "ofc-step11-transport-contract" \
  "$WORK_ROOT/transport-contract.json"
fetch_initial \
  "ofc-step11-controller-authorization" \
  "$WORK_ROOT/controller-authorization.json"
fetch_initial \
  "ofc-step11-controller-public-key" \
  "$WORK_ROOT/controller-public-key.json"
fetch_initial \
  "ofc-step11-prebootstrap" \
  "$WORK_ROOT/prebootstrap.py"

readonly CLAIM_PATH="$WORK_ROOT/controller-claim.json"
mark_phase "claim_wait"
claim_deadline=$((SECONDS + CLAIM_WAIT_SECONDS))
while (( SECONDS < claim_deadline )); do
  set +e
  /usr/bin/python3 "$METADATA_FETCHER" \
    "ofc-step11-controller-claim" "$CLAIM_PATH"
  claim_rc=$?
  set -e
  if [[ "$claim_rc" -eq 0 ]]; then
    break
  fi
  if [[ "$claim_rc" -ne 75 ]]; then
    exit "$claim_rc"
  fi
  /usr/bin/sleep 2
done

if [[ ! -s "$CLAIM_PATH" ]]; then
  exit 78
fi

export OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER=1
mark_phase "prebootstrap"
/usr/bin/python3 "$WORK_ROOT/prebootstrap.py" \
  --contract "$WORK_ROOT/transport-contract.json" \
  --authorization "$WORK_ROOT/controller-authorization.json" \
  --claim "$CLAIM_PATH" \
  --controller-public-key "$WORK_ROOT/controller-public-key.json" \
  --bootstrap-root "$WORK_ROOT/bootstrap" \
  --fresh-root "$WORK_ROOT/worker"
