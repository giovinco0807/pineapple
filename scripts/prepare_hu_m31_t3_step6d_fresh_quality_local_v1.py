#!/usr/bin/env python3
"""Build immutable local staging for the M3.1 fresh-quality 8+7 run."""

from __future__ import annotations

import sys

from ofc_regular.hu_m31_t3_step6d_fresh_quality_local_staging_v1 import main


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"fresh-quality local staging failed closed: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
