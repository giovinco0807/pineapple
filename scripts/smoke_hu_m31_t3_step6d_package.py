"""Load both Step 6d native engines from an extracted Linux package."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

from ofc_regular.hu_m31_t3_runtime import (
    HuM31T3RuntimeConfig,
    HuM31T3SearchSolver,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--reference-sha256", required=True)
    args = parser.parse_args()

    root = args.package_root.resolve()
    engines: list[dict[str, object]] = []
    for role, digest in (
        ("candidate", args.candidate_sha256),
        ("reference", args.reference_sha256),
    ):
        path = root / "native" / role / "release" / "libofc_hu_m3_engine.so"
        solver = HuM31T3SearchSolver(
            HuM31T3RuntimeConfig(
                expected_library_sha256=digest,
                library_path=path,
            )
        )
        engines.append(
            {
                "role": role,
                "engine_version": solver.engine_version,
                "sha256": solver.library_sha256,
                "release_path": "release" in path.parts,
            }
        )

    feature = root / "target" / "release" / "libofc_stage3_feature_encoder.so"
    ctypes.CDLL(str(feature))
    result = {
        "schema": "hu_m31_t3_step6d_linux_package_smoke_v1",
        "status": "both_engines_and_feature_loaded",
        "engines": engines,
        "feature_bytes": feature.stat().st_size,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
