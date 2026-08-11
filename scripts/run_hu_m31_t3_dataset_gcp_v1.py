"""CLI entrypoint for the selected M3.1 8-VM dataset controller.

Production preparation is owned by the same-Linux closeout.  Keeping the
legacy raw ``prepare`` command reachable here would allow an operator to omit
the explicit closeout/smoke source binding, so this wrapper fails closed for
that one command.  The controller's read-only and lifecycle commands remain
unchanged.
"""

from __future__ import annotations

import sys
from typing import Sequence

from ofc_regular.hu_m31_t3_dataset_gcp_controller_v1 import (
    main as controller_main,
)


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "prepare":
        print(
            "ERROR: raw dataset prepare is disabled; run the same-Linux "
            "closeout, write-source-binding, and use its prepared controller",
            file=sys.stderr,
        )
        return 2
    return controller_main(values)


if __name__ == "__main__":
    raise SystemExit(main())
