"""Create the immutable fit700-only Attempt03 base-fold contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .hu_m43_attempt03_training import write_attempt03_fold_cloud_contract


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inherited-train", type=Path, required=True)
    parser.add_argument("--fresh-train-fit", type=Path, required=True)
    parser.add_argument("--fit-receive-receipt", type=Path, required=True)
    parser.add_argument("--model-freeze", type=Path, required=True)
    parser.add_argument("--training-freeze", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = write_attempt03_fold_cloud_contract(
        args.output,
        inherited_train_path=args.inherited_train,
        fresh_train_fit_path=args.fresh_train_fit,
        fit_receive_receipt_path=args.fit_receive_receipt,
        model_freeze_path=args.model_freeze,
        training_freeze_path=args.training_freeze,
        repo_root=args.repo_root,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "parse_args"]
