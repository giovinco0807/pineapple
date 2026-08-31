"""Export a boundary net to the `.npz` a fleet worker can read without torch.

The network is a ReLU stack, so nothing is lost in the translation and the
worker image stays `python3-numpy` -- the same image the existing label
fleet boots with.  The export is verified here rather than trusted: both
paths score the same random inputs and the maximum absolute difference is
printed, because a value net that quietly disagrees with itself on the fleet
would put the disagreement into every label it produces and nothing
downstream would notice.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ai.tutor.hu_street_teacher import ValueNet
from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probes", type=int, default=512)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = T4FirstEvaluator(payload["input_dim"], tuple(payload["hidden"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    arrays = {
        "input_mean": np.asarray(payload["input_mean"], dtype=np.float32),
        "input_std": np.asarray(payload["input_std"], dtype=np.float32),
        "input_dim": np.asarray(payload["input_dim"], dtype=np.int64),
    }
    for index, layer in enumerate(
        [layer for layer in model.net if hasattr(layer, "weight")]
    ):
        arrays[f"w{index}"] = layer.weight.detach().numpy().astype(np.float32)
        arrays[f"b{index}"] = layer.bias.detach().numpy().astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)

    # Same inputs, both readers.  The features are standardised inside the
    # net, so raw normal draws exercise the same arithmetic real ones do.
    rng = np.random.default_rng(20260817)
    probe = rng.normal(size=(args.probes, int(payload["input_dim"]))).astype(np.float32)
    reference = ValueNet(args.checkpoint).predict(probe)
    exported = ValueNet(args.out).predict(probe)
    gap = float(np.max(np.abs(reference - exported)))
    print(
        f"{args.out}: {len(arrays) // 2} layers, "
        f"max |torch - numpy| = {gap:.3e} over {args.probes} probes"
    )
    if gap > 1e-3:
        raise SystemExit(f"FATAL: exported net disagrees by {gap}")


if __name__ == "__main__":
    main()
