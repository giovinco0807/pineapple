"""Export the trained T4 first-seat evaluator as a flat little-endian image.

Format mirrors the regular track's `t4_model.rs` loader, whose header explains
why the weights are pinned by digest: "A model paired with a drifted encoder is
silently wrong rather than loudly broken."

    magic       b"T4F1"        4 bytes
    version     u32            = 1
    layers      u32
    input_dim   u32
    mean        f32[input_dim]
    std         f32[input_dim]
    per layer:  inputs u32, outputs u32, weight f32[in*out] (row-major
                [output][input]), bias f32[out]

Usage:
    python -m ai.tutor.export_t4_first_evaluator --model <best.pt> --out <bin>
"""
from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

import torch

from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator

MAGIC = b"T4F1"
VERSION = 1


def export(model_path: Path, out_path: Path) -> dict:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = T4FirstEvaluator(checkpoint["input_dim"], tuple(checkpoint["hidden"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    linears = [layer for layer in model.net if isinstance(layer, torch.nn.Linear)]
    payload = bytearray()
    payload += MAGIC
    payload += struct.pack("<I", VERSION)
    payload += struct.pack("<I", len(linears))
    payload += struct.pack("<I", int(checkpoint["input_dim"]))
    for tensor in (checkpoint["input_mean"], checkpoint["input_std"]):
        payload += tensor.to(torch.float32).contiguous().numpy().tobytes()
    for linear in linears:
        weight = linear.weight.detach().to(torch.float32).contiguous()
        bias = linear.bias.detach().to(torch.float32).contiguous()
        payload += struct.pack("<II", weight.shape[1], weight.shape[0])
        payload += weight.numpy().tobytes()
        payload += bias.numpy().tobytes()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(payload))
    return {
        "path": str(out_path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "input_dim": int(checkpoint["input_dim"]),
        "layers": [(l.weight.shape[1], l.weight.shape[0]) for l in linears],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("D:/ofc_data/t4_first_model_pass1_v2/evaluator_best.pt"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    info = export(args.model, args.out)
    print(info)


if __name__ == "__main__":
    main()
