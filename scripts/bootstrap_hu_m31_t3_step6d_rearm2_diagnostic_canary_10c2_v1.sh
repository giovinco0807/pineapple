#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# Versioned direct worker entrypoint.  It never writes authorization or claim
# records.  Local-preflight is the default development path; authorized-worker
# requires controller-issued local files and a public trust key before its Python
# runtime can make the fixed metadata/GCS/Compute REST calls.

PYTHON_SHUTDOWN_ATTEMPTED_STATUS=86
SHUTDOWN_MARKER=/run/ofc-m31-step11-shutdown-requested
SHUTDOWN_ATTEMPTED=0

bounded_shutdown_once() {
  if [[ "$SHUTDOWN_ATTEMPTED" == "1" ]]; then
    return
  fi
  SHUTDOWN_ATTEMPTED=1
  if [[ -e "$SHUTDOWN_MARKER" ]]; then
    return
  fi
  if ! ( set -o noclobber; printf 'shutdown-requested\n' > "$SHUTDOWN_MARKER" ) 2>/dev/null; then
    if [[ -e "$SHUTDOWN_MARKER" ]]; then
      return
    fi
    echo "could not create bounded-shutdown marker" >&2
    return
  fi
  if ! /usr/bin/timeout 120 /sbin/shutdown -h now >/dev/null 2>&1; then
    if [[ -f "$SHUTDOWN_MARKER" && ! -L "$SHUTDOWN_MARKER" ]] \
      && [[ "$(<"$SHUTDOWN_MARKER")" == "shutdown-requested" ]]; then
      rm -f -- "$SHUTDOWN_MARKER"
    fi
    return 1
  fi
}

if [[ "$#" -lt 1 ]]; then
  echo "usage: bootstrap-10c2 <local-preflight|authorized-worker> ..." >&2
  if [[ "${OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER:-0}" == "1" ]]; then
    bounded_shutdown_once
  fi
  exit 64
fi

PYTHON_BIN=${OFC_DIAGNOSTIC_10C2_PYTHON:-python3}
MODULE=ofc_regular.hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
OUTER_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)
export PYTHONPATH="$OUTER_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
MODE=$1
shift
case "$MODE" in
  local-preflight)
    if [[ "${OFC_DIAGNOSTIC_10C2_LOCAL_PREFLIGHT:-0}" != "1" || "$#" -ne 3 ]]; then
      echo "local-preflight is not explicitly enabled or has wrong arguments" >&2
      exit 78
    fi
    ARGS=(-m "$MODULE" local-preflight --contract "$1" --package-mirror "$2" --fresh-work-root "$3")
    if [[ -n "${OFC_DIAGNOSTIC_10C2_OFFLINE_WHEEL_MIRROR:-}" ]]; then
      ARGS+=(--offline-wheel-mirror "$OFC_DIAGNOSTIC_10C2_OFFLINE_WHEEL_MIRROR")
    fi
    if [[ "${OFC_DIAGNOSTIC_10C2_OFFLINE_INSTALL_SMOKE:-0}" == "1" ]]; then
      ARGS+=(--offline-install-smoke)
    fi
    ;;
  authorized-worker)
    if [[ "${OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER:-0}" != "1" ]]; then
      echo "authorized-worker is not explicitly enabled" >&2
      bounded_shutdown_once
      exit 78
    fi
    if [[ "$#" -ne 5 ]]; then
      echo "authorized-worker is not explicitly enabled or has wrong arguments" >&2
      bounded_shutdown_once
      exit 78
    fi
    ARGS=(
      -m "$MODULE" authorized-worker
      --contract "$1"
      --authorization "$2"
      --claim "$3"
      --controller-public-key "$4"
      --fresh-root "$5"
    )
    ;;
  *)
    echo "unknown 10c2 mode" >&2
    if [[ "${OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER:-0}" == "1" ]]; then
      bounded_shutdown_once
    fi
    exit 64
    ;;
esac

if [[ "$MODE" == "local-preflight" ]]; then
  exec "$PYTHON_BIN" "${ARGS[@]}"
fi

set +e
"$PYTHON_BIN" "${ARGS[@]}"
STATUS=$?
set -e
if [[ "$STATUS" -ne 0 && "$STATUS" -ne "$PYTHON_SHUTDOWN_ATTEMPTED_STATUS" ]]; then
  bounded_shutdown_once
fi
exit "$STATUS"
