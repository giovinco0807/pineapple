"""Export a distilled fast evaluator as a flat little-endian image.

Deliberately the same shape as `export_t4_first_evaluator`'s T4F1 so the Rust
loader can be the existing one, with a different magic so a fast net can never
be loaded where a 207-dim champion is expected (and the reverse): the width
alone would not catch it, and a model read by the wrong encoder is silently
wrong rather than loudly broken.

    magic       b"HUF1"        4 bytes
    version     u32            = 1
    layers      u32
    input_dim   u32            = 487
    slot        u32            street * 2 + seat   (0 = bb, 1 = btn)
    per layer:  inputs u32, outputs u32, weight f32[in*out] (row-major
                [output][input]), bias f32[out]

No mean/std block: the features are already bounded (one-hots and ratios), so
the trainer standardises nothing and an exported net that carried an identity
scaler would only invite someone to "fix" it later.

Usage:
    python -m ai.tutor.export_hu_fast --net D:/ofc_data/hu/fast/nets/t1_bb.pt \
        --out D:/ofc_data/hu/fast/bins/t1_bb.bin
"""
from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

import torch

from ai.tutor.train_hu_fast_eval import FastEval, FEATURES

MAGIC = b"HUF1"
VERSION = 1


def slot_code(slot: str) -> int:
    street = int(slot[1])
    seat = 0 if slot.endswith("_bb") else 1
    return street * 2 + seat


def export(net_path: Path, out_path: Path) -> dict:
    checkpoint = torch.load(net_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "hu_fast_eval/v1":
        raise SystemExit(f"{net_path}: unexpected schema {checkpoint.get('schema')!r}")
    model = FastEval()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    linears = [layer for layer in model.net if isinstance(layer, torch.nn.Linear)]
    payload = bytearray()
    payload += MAGIC
    payload += struct.pack("<I", VERSION)
    payload += struct.pack("<I", len(linears))
    payload += struct.pack("<I", FEATURES)
    payload += struct.pack("<I", slot_code(checkpoint["slot"]))
    for linear in linears:
        weight = linear.weight.detach().to(torch.float32).contiguous()
        bias = linear.bias.detach().to(torch.float32).contiguous()
        payload += struct.pack("<II", weight.shape[1], weight.shape[0])
        payload += weight.numpy().tobytes()
        payload += bias.numpy().tobytes()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(payload))
    return {"path": str(out_path), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()[:16],
            "slot": checkpoint["slot"], "slot_code": slot_code(checkpoint["slot"]),
            "layers": [(l.weight.shape[1], l.weight.shape[0]) for l in linears]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.net, args.out))


if __name__ == "__main__":
    main()
