"""Export a trainer checkpoint (.pt) to the engine's T4M1 weight image (.bin).

Layout, read from rust/hu_m3_engine/src/t4_model.rs (all little-endian):

    b"T4M1"                     4 bytes
    version    u32 = 1
    layers     u32
    input_dim  u32
    mean       f32[input_dim]
    std        f32[input_dim]        (the engine inverts it at load)
    per layer:
      inputs   u32
      outputs  u32
      weight   f32[inputs*outputs]   row-major [output][input]
      bias     f32[outputs]
    no trailing bytes; the last layer must produce exactly 1 output

REFUSES a checkpoint trained with a nonzero input clamp. This is the open DEBT
in the M7 record: the image has nowhere to store a clamp and the engine does not
apply one, so a model fitted on clamped standardized inputs would be served
outside the regime it was fitted in -- silently, because the export succeeds and
the digests match. The guard is here rather than in a checklist.

Usage:
    export_t4m1.py <checkpoint.pt> <out.bin> [--sidecar <out.metadata.json>]
                   [--expect-sha256 <hex>]
"""
import argparse
import hashlib
import json
import pathlib
import struct

import numpy as np
import torch


def build_image(payload) -> bytes:
    arch = list(payload["architecture"])
    clamp = float(payload.get("clamp", 0.0) or 0.0)
    if clamp != 0.0:
        raise SystemExit(
            f"refusing to export: checkpoint was trained with clamp={clamp}, "
            "and the T4M1 image cannot carry one. The engine applies no clamp "
            "at inference, so this model would run outside the input regime it "
            "was fitted in. Retrain with clamp disabled."
        )

    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    if mean.shape != std.shape or mean.ndim != 1:
        raise SystemExit("mean and std must be 1-D and the same length")
    if mean.shape[0] != arch[0]:
        raise SystemExit(
            f"standardization is {mean.shape[0]}-D but the architecture takes "
            f"{arch[0]} inputs")
    if not np.all(np.isfinite(std)) or np.any(std == 0.0):
        raise SystemExit("std has a zero or non-finite entry; the engine "
                         "refuses such an image at load")
    if arch[-1] != 1:
        raise SystemExit(f"the final layer must produce 1 value, not {arch[-1]}")

    state = payload["state_dict"]
    # torch.nn.Sequential with ReLU between Linears: weights live at 0, 2, 4, ...
    indices = [2 * i for i in range(len(arch) - 1)]

    out = bytearray()
    out += b"T4M1"
    out += struct.pack("<III", 1, len(indices), arch[0])
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()

    for position, index in enumerate(indices):
        weight = np.asarray(state[f"{index}.weight"], dtype=np.float32)
        bias = np.asarray(state[f"{index}.bias"], dtype=np.float32)
        inputs, outputs = arch[position], arch[position + 1]
        if weight.shape != (outputs, inputs):
            raise SystemExit(
                f"layer {position}: weight is {weight.shape}, expected "
                f"{(outputs, inputs)}")
        if bias.shape != (outputs,):
            raise SystemExit(
                f"layer {position}: bias is {bias.shape}, expected {(outputs,)}")
        out += struct.pack("<II", inputs, outputs)
        # torch stores [out][in], which is already the row-major order the
        # engine reads.
        out += weight.astype("<f4").tobytes()
        out += bias.astype("<f4").tobytes()

    return bytes(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    parser.add_argument("--sidecar", default=None)
    parser.add_argument("--expect-sha256", default=None)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image = build_image(payload)
    digest = hashlib.sha256(image).hexdigest()

    if args.expect_sha256 and digest != args.expect_sha256:
        raise SystemExit(
            f"digest mismatch: produced {digest}, expected {args.expect_sha256}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(image)
    print(f"wrote {out}  {len(image)} bytes  sha256 {digest}")

    if args.sidecar:
        sidecar = pathlib.Path(args.sidecar)
        sidecar.write_text(json.dumps({
            "file": out.name,
            "sha256": digest,
            "bytes": len(image),
            "architecture": list(payload["architecture"]),
            "input_dim": int(payload["architecture"][0]),
            "clamp": float(payload.get("clamp", 0.0) or 0.0),
            "source_checkpoint": str(pathlib.Path(args.checkpoint).name),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
