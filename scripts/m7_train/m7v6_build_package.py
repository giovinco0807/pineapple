"""Build the m7v6 label-generation package: m7v5 plus the T2 v2 pair.

Same thin-driver shape as `m7v4_build_package_t3v7.py` (which built m7v5), one
generation on.  What this adds over the v5 package:

  * `t2first_model_v2.bin` and `t2_model_v2.bin`, the M7 T2 relabel trained on
    25,000 roots at 2,048 particles and adopted 2026-08-08 (gates +0.0171 /
    +0.0295 per hand over 20,004 mirrored deals each).  The generation-1 T2
    images are RETAINED, exactly as v5 retained the T4 v5 and T3 gen-1 images:
    a package that dropped them could not reproduce any plan written before
    today, and a plan picks which it pins.
  * ledger keys `t2_first_model_v2_sha256` / `t2_second_model_v2_sha256`, read
    back from the runtime tree rather than from the source directory, so the
    digest recorded is the digest of what shipped.

Also carried over from the v5 driver: T4 v6 and the T3 v7 pair, the pinned
engine (`e17aa39f…` -- NOT the repository target/release, which has uncommitted
trainer-era changes), and the fl_ev v4 note in the ledger.

The T1 relabel plans this package exists for will pin the v2 T2 pair as the
continuation the labels play through.  That is the whole point of building it.
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

import labelgen_build_package as builder  # noqa: E402

OUT_DIR = "/home/wner/ofc-labelgen-m7v6"
ENGINE = pathlib.Path("/home/wner/ofc-m7/pinned/libofc_hu_m3_engine.so")
EXPECTED_ENGINE_SHA = (
    "e17aa39f797e01e487d0dbcc726073da420ef6fc2c61fb0aa7a47cb5add3670f"
)

T4_V6_DIR = pathlib.Path("/home/wner/ofc-t4/model_v6")
T3V7_SHIP = pathlib.Path("/home/wner/ofc-m7/t3v7/ship/weights")
T2V2_FIRST = pathlib.Path("/home/wner/ofc-m7/t2_2048/first/ship/weights")
T2V2_SECOND = pathlib.Path("/home/wner/ofc-m7/t2_2048/second/ship/weights")

# image filename -> (source dir, ledger key)
PROMOTED = {
    "t4_model_v6.bin": (T4_V6_DIR, "t4_model_v6_sha256"),
    "t3first_model_v2.bin": (T3V7_SHIP, "t3_first_model_v2_sha256"),
    "t3_model_v3.bin": (T3V7_SHIP, "t3_second_model_v3_sha256"),
    "t2first_model_v2.bin": (T2V2_FIRST, "t2_first_model_v2_sha256"),
    "t2_model_v2.bin": (T2V2_SECOND, "t2_second_model_v2_sha256"),
}


def main() -> int:
    if not ENGINE.is_file():
        raise SystemExit(f"missing engine: {ENGINE}")
    if builder.sha256_of(ENGINE) != EXPECTED_ENGINE_SHA:
        raise SystemExit(
            "the pinned engine's digest moved; this driver promotes weights "
            "into the SAME engine generation the m7v5 plans ran on, and a "
            "different .so is a different generation"
        )

    for name, (directory, _) in PROMOTED.items():
        image = directory / name
        sidecar = directory / name.replace(".bin", ".sha256.json")
        for path in (image, sidecar):
            if not path.is_file():
                raise SystemExit(f"missing: {path}")
        builder.WEIGHTS[name] = image
        builder.SIDECAR_DIGESTS[name] = sidecar

    builder.VSFL_FL_EV_CONFIG_RELATIVE = "configs/fl_ev_regular_v4_selfplay.json"
    print(f"weight manifest at {len(builder.WEIGHTS)} images "
          f"(all generation-1 images retained)")

    sys.argv = ["labelgen_build_package", "--out-dir", OUT_DIR,
                "--engine", str(ENGINE)]
    status = builder.main()
    if status != 0:
        return status

    out = pathlib.Path(OUT_DIR)
    ledger_path = out / "package/ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    runtime_weights = out / "build/runtime/weights"
    for name, (_, key) in PROMOTED.items():
        shipped = runtime_weights / name
        if not shipped.is_file():
            raise SystemExit(f"{shipped} did not reach the runtime tree")
        ledger[key] = builder.sha256_of(shipped)
    ledger["fl_ev"] = {
        "config": "configs/fl_ev_regular_v4_selfplay.json",
        "cards": 14,
        "value": 9.6,
        "config_sha256": builder.sha256_of(
            out / "build/runtime/configs/fl_ev_regular_v4_selfplay.json"
        ),
        "note": (
            "The constant this generation's labels bake in. It reaches the "
            "engine through each observation's scoring context, read from this "
            "config by hu_infoset, so the config travelling inside the runtime "
            "is what makes a remote label mean the same thing as a local one."
        ),
    }
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True),
                           encoding="utf-8")
    print("\nledger amended:")
    for _, (_, key) in PROMOTED.items():
        print(f"  {key:28s} {ledger[key]}")
    print(f"  engine_library_sha256        {ledger['engine_library_sha256']}")
    print(f"  weight_files                 {ledger['weight_files']}")
    print(f"\nledger sha256 (for the T1 plan generator's EXPECTED_LEDGER):")
    print(f"  {builder.sha256_of(ledger_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
