"""Assemble the label-generation runtime package and its digest ledger.

The archive must be self-contained: the class of defect that cost the previous
cloud workstream its runs was a payload missing one file that every local run
found via the repository. So the builder ships only an explicit manifest --
the eleven model files the bundle actually opens (found by audit hook, not by
guessing), the two native libraries, the eleven weight images, and the Python
tree -- and the rehearsal harness then runs the worker from the extracted
archive alone, which is the test that matters.

The fifth weight image is the T2 first-seat model, which the T1 second-seat
teacher needs: from a T1 root both T2 decisions are still ahead of it. The
sixth is the T1 second-seat model itself, which only the T1 FIRST-seat teacher
needs: acting first, the opponent's reply on this same street opens every
rollout. The seventh is the T1 FIRST-seat model, which the T0 second-seat
teacher needs: from the opening street the whole T1 street is still ahead, and
acting first at T1 both boards carry five cards where the second seat sees
seven, so that seat has its own model. Those last two live outside the
repository fixtures, in their training runs' own output directories, so they
are copied from there and cross-checked against the sidecars those runs wrote.

The eighth and ninth are the coarse distilled T1 pair from the fast-rollout
run. They do not replace the seventh and sixth: a T0 plan that pins them ships
all nine, because the engine loads the full-precision pair as well and it is the
coarse one it answers T1 replies through. Shipping them unconditionally costs
260 KB and means a plan can opt in without a new package, which is the whole
reason the archive carries an explicit manifest rather than what a plan asked
for. They ship with sidecars from their own training run and are cross-checked
against them like the T1 and T2 images.

The tenth is the T0 second-seat model, which the T0 FIRST-seat teacher needs:
acting first on the opening street the opponent's own T0 reply is still ahead of
it and opens every rollout. It is the eighth and last learned evaluator in the
ladder, and like the T1 images it lives in its training run's output directory
rather than in the engine's fixtures.

The eleventh is that model's coarse distilled twin, from the fast-rollout run
that produced the T1 pair. Same standing as the eighth and ninth: shipped
unconditionally, cross-checked against its sidecar, and REQUIRED rather than
skipped-if-absent. A package built without it would look complete and then fail
on the VM against the first T0 first-seat plan that pins it, which is precisely
the defect class the explicit manifest exists to prevent -- so a build attempted
before that export lands stops here, with the reason.

The fourteenth is the odd one out, and worth reading as such. Every image above
it answers a decision some teacher has AHEAD of it: they are continuations, and
a plan pins them because the label depends on how the rest of the hand is
played. `t0first_model_v1` answers the opening street acting first, which no
teacher ever has ahead of it -- it is the first decision of a hand. It ships
because it is the ROOT POLICY: the chain that decides which positions the next
generation's labels are collected at. Nothing in a label's value depends on it;
everything about which labels exist does.

It is also the only image here that is not loadable by the other arms. Acting
first the opponent's board is empty, which the free-slot outlook refuses, so its
last 46 columns are zero by construction and it was fitted that way. The engine
enforces the pairing by field name -- `learned_t0_first_model_path` reaches only
the arm that produces that geometry -- and the width check catches the rest.

The `--engine` override exists for the same generation. A package is only ever
as good as the .so inside it, and the engine that can answer T0 first seat is
built from the crate rather than taken from wherever a previous build left one.
Defaulting to the repository's `target/release` keeps every existing build
command producing what it produced before.
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

REPO = pathlib.Path(
    "/mnt/c/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple/regular-ofc-pineapple"
)
WHEELHOUSE = pathlib.Path.home() / "ofc-m31-wheelhouse"
OUT = pathlib.Path.home() / "ofc-labelgen"
# The T1 second-seat and T1 first-seat weights are the two images not exported
# into the engine's test fixtures; they ship straight out of the training runs
# that produced them.
T1_MODEL_DIR = pathlib.Path.home() / "ofc-t1/model_v1"
T1FIRST_MODEL_DIR = pathlib.Path.home() / "ofc-t1first/model_v1"
# The T0 second-seat weights, likewise straight out of their training run.
T0_MODEL_DIR = pathlib.Path.home() / "ofc-t0/model_v1"
# The coarse T1 pair, from the fast-rollout distillation run rather than from a
# per-street training run, which is why both seats' images sit in one directory.
FASTROLL_MODEL_DIR = pathlib.Path.home() / "ofc-fastroll/models"
# The coarse T0 second-seat clone came out of a later, separate fast-rollout run
# with its own output directory; it is NOT in the T1 pair's directory.
FASTROLL_T0_MODEL_DIR = pathlib.Path.home() / "ofc-fastroll-t0s/models"
FASTROLL_T2_MODEL_DIR = pathlib.Path.home() / "ofc-fastroll-t2/models"
# The T0 FIRST-seat root policy, out of its own early-stop training run. Not a
# fast-rollout clone and not a per-street continuation: see the module comment.
T0FIRST_MODEL_DIR = pathlib.Path.home() / "ofc-t0first/train_early/curve"

MODEL_FILES = [
    "hu_turn0_stage19_safe_selector_highmc_candidate_delta_plus_meta.pkl",
    "hu_turn0_stage3_top60_mc32_1000_hgb_baseline-delta_sew1_aug3.pkl",
    "hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl",
    "hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl",
    "hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt",
    "hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt",
    "hu_turn3_stage7_reference_override_cached_rank_wide.pt",
    "opening_stage7_torch_wide.pt",
    "turn1_stage6_torch_wide.pt",
    "turn2_stage8.pkl",
    "turn3_stage6.pkl",
]
NATIVE = {
    "libofc_hu_m3_engine.so": REPO / "target/release/libofc_hu_m3_engine.so",
    "libofc_stage3_feature_encoder.so": REPO
    / "outputs/gcp_runs/regular-hu-m31-c02-full100-dev-20260717-001"
    / "package_src/target/release/libofc_stage3_feature_encoder.so",
}
WEIGHTS = {
    "t4_model_v5.bin": REPO / "rust/hu_m3_engine/tests/fixtures/t4_model_v5.bin",
    "t3_model_v2.bin": REPO / "rust/hu_m3_engine/tests/fixtures/t3_model_v2.bin",
    "t3first_model_v1.bin": REPO
    / "rust/hu_m3_engine/tests/fixtures/t3first_model_v1.bin",
    "t2_model_v1.bin": REPO / "rust/hu_m3_engine/tests/fixtures/t2_model_v1.bin",
    "t2first_model_v1.bin": REPO
    / "rust/hu_m3_engine/tests/fixtures/t2first_model_v1.bin",
    "t1_model_v1.bin": T1_MODEL_DIR / "t1_model_v1.bin",
    "t1first_model_v1.bin": T1FIRST_MODEL_DIR / "t1first_model_v1.bin",
    "fast_t1_first_v1.bin": FASTROLL_MODEL_DIR / "fast_t1_first_v1.bin",
    "fast_t1_second_v1.bin": FASTROLL_MODEL_DIR / "fast_t1_second_v1.bin",
    "t0_model_v1.bin": T0_MODEL_DIR / "t0_model_v1.bin",
    "fast_t0_second_v1.bin": FASTROLL_T0_MODEL_DIR / "fast_t0_second_v1.bin",
    "fast_t2_first_v1.bin": FASTROLL_T2_MODEL_DIR / "fast_t2_first_v1.bin",
    "fast_t2_second_v1.bin": FASTROLL_T2_MODEL_DIR / "fast_t2_second_v1.bin",
    "t0first_model_v1.bin": T0FIRST_MODEL_DIR / "t0first_model_v1.bin",
}
SIDECAR_DIGESTS = {
    # The three engine-fixture images declare their digest inside the
    # predictions fixture the Rust tests read it from, rather than in a
    # standalone .sha256.json. Same claim, same key -- so all eleven weights
    # are cross-checked here, not eight.
    "t4_model_v5.bin": REPO
    / "rust/hu_m3_engine/tests/fixtures/t4_model_v5_predictions.json",
    "t3_model_v2.bin": REPO
    / "rust/hu_m3_engine/tests/fixtures/t3_model_v2_predictions.json",
    "t3first_model_v1.bin": REPO
    / "rust/hu_m3_engine/tests/fixtures/t3first_model_v1_predictions.json",
    "t2_model_v1.bin": REPO
    / "rust/hu_m3_engine/tests/fixtures/t2_model_v1.sha256.json",
    "t2first_model_v1.bin": REPO
    / "rust/hu_m3_engine/tests/fixtures/t2first_model_v1.sha256.json",
    "t1_model_v1.bin": T1_MODEL_DIR / "t1_model_v1.sha256.json",
    "t1first_model_v1.bin": T1FIRST_MODEL_DIR / "t1first_model_v1.sha256.json",
    "fast_t1_first_v1.bin": FASTROLL_MODEL_DIR / "fast_t1_first_v1.sha256.json",
    "fast_t1_second_v1.bin": FASTROLL_MODEL_DIR / "fast_t1_second_v1.sha256.json",
    "t0_model_v1.bin": T0_MODEL_DIR / "t0_model_v1.sha256.json",
    "fast_t0_second_v1.bin": FASTROLL_T0_MODEL_DIR / "fast_t0_second_v1.sha256.json",
    "fast_t2_first_v1.bin": FASTROLL_T2_MODEL_DIR / "fast_t2_first_v1.sha256.json",
    "fast_t2_second_v1.bin": FASTROLL_T2_MODEL_DIR / "fast_t2_second_v1.sha256.json",
    "t0first_model_v1.bin": T0FIRST_MODEL_DIR / "t0first_model_v1.sha256.json",
}
# Weights whose export is produced by a workstream other than this one, with the
# reason a build should stop rather than proceed without them. Checked before
# anything is copied, so "the export has not landed yet" is one line at the top
# rather than a FileNotFoundError partway through a build.
LATE_ARRIVALS = {
    "fast_t0_second_v1.bin": (
        "the coarse T0 second-seat model has not been exported yet; every T0 "
        "first-seat plan pins it, so a package built without it would fail on "
        "the VM rather than here"
    ),
    "t0first_model_v1.bin": (
        "the T0 first-seat root policy has not been exported yet; an M7 "
        "generation is defined by which chain collected its positions, so a "
        "package that cannot name that chain cannot start one"
    ),
}


# --- t3_vs_fl payload -------------------------------------------------------
#
# The T3-vs-Fantasyland kind opens neither the engine nor a single learned
# weight. What it needs is a solver binary, the roots file its offsets index
# into, and the FL EV config whose value its labels bake in -- the last of which
# already ships, because `configs/` is copied whole.
#
# Added only when `--with-t3-vs-fl` asks for it. Without the flag this builder
# writes the same archive and the same ledger keys it wrote before the kind
# existed, which is the property that lets the running fleets keep using it.
VSFL_SOLVER = pathlib.Path.home() / "ofc-vsfl/target/release/fl_solver_regular"
VSFL_FL_EV_CONFIG_RELATIVE = "configs/fl_ev_regular_v3_direct2.json"


# --- t1_vs_fl payload -------------------------------------------------------
#
# T1 is the first vs-Fantasyland kind whose labels depend on learned policies:
# its candidates are continued by the trained T2 and T3 rankers rather than
# searched. Those two images are therefore part of what a T1 label MEANS, in
# exactly the way `opponent_mode` is, so they travel inside the package and the
# plan pins them by digest.
#
# They are VFL1 rather than T4M1 because the arms that won need two things the
# engine's image cannot store: an inverse standard deviation of exactly zero
# (which imputes the zeroed opponent tail to its pretrained mean) and an input
# clamp. Writing them as T4M1 would silently drop both and ship a model that is
# not the model that was measured.
VSFL_T1_MODELS = {
    "vsfl_t2_v1.vfl1": pathlib.Path.home() / "ofc-vsfl/models/vsfl_t2_v1.vfl1",
    "vsfl_t3_v1.vfl1": pathlib.Path.home() / "ofc-vsfl/models/vsfl_t3_v1.vfl1",
}
# Each VFL1 ships beside a parity fixture that declares the digest of the image
# it was generated from, under `model_sha256`. Same standing as the weight
# sidecars above: a .vfl1 that no longer matches its fixture is a stale export,
# and shipping one would pin the plan to a model whose parity nobody measured.
VSFL_T1_MODEL_FIXTURES = {
    "vsfl_t2_v1.vfl1": pathlib.Path.home() / "ofc-vsfl/models/vsfl_t2_v1.parity.json",
    "vsfl_t3_v1.vfl1": pathlib.Path.home() / "ofc-vsfl/models/vsfl_t3_v1.parity.json",
}


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default=None,
        help="package root; defaults to ~/ofc-labelgen. Point it elsewhere to "
             "build a package without touching the one a running fleet reads",
    )
    parser.add_argument(
        "--with-t3-vs-fl", action="store_true",
        help="also ship the FL solver binary and a roots file for the "
             "t3_vs_fl plan kind. Omitted, the archive and ledger are "
             "byte-identical to a build from before that kind existed",
    )
    parser.add_argument(
        "--vsfl-roots", default=None,
        help="roots JSONL to ship for t3_vs_fl; required with --with-t3-vs-fl",
    )
    parser.add_argument(
        "--vsfl-t2-roots", default=None,
        help="roots JSONL to ship for the t2_vs_fl kind; shipped under roots/t2_vs_fl_roots.jsonl and recorded in its own ledger section",
    )
    parser.add_argument(
        "--with-t1-vs-fl", action="store_true",
        help="also ship the two VFL1 continuation rankers and a T1 roots file "
             "for the t1_vs_fl plan kind. Ships the solver binary too, so it "
             "does not need --with-t3-vs-fl alongside it",
    )
    parser.add_argument(
        "--vsfl-t1-roots", default=None,
        help="roots JSONL to ship for t1_vs_fl; required with --with-t1-vs-fl",
    )
    parser.add_argument(
        "--engine-features-rev", default=None,
        help="the engine feature generation the T1 continuation models were "
             "encoded under; required with --with-t1-vs-fl. Verified to appear "
             "verbatim inside the shipped solver binary rather than trusted, "
             "because the worker's provenance gate compares the plan's value "
             "against what the solver reports at runtime",
    )
    parser.add_argument(
        "--engine", default=None,
        help="the libofc_hu_m3_engine.so to ship. Defaults to the "
             "repository's target/release build, which is what every earlier "
             "build used; point it at a fresh build to ship an engine the "
             "repository is not carrying",
    )
    args = parser.parse_args()

    global OUT
    if args.out_dir is not None:
        OUT = pathlib.Path(args.out_dir).expanduser()
    if args.engine is not None:
        engine = pathlib.Path(args.engine).expanduser()
        if not engine.is_file():
            raise SystemExit(f"{engine} is not a file")
        NATIVE["libofc_hu_m3_engine.so"] = engine
    if args.with_t3_vs_fl and args.vsfl_roots is None:
        raise SystemExit("--with-t3-vs-fl needs --vsfl-roots")
    if args.with_t1_vs_fl and args.vsfl_t1_roots is None:
        raise SystemExit("--with-t1-vs-fl needs --vsfl-t1-roots")
    if args.with_t1_vs_fl and args.engine_features_rev is None:
        raise SystemExit(
            "--with-t1-vs-fl needs --engine-features-rev: a T1 label's "
            "continuation is only reproducible against the engine generation "
            "its features were computed under, and the worker gates on it"
        )

    began = time.perf_counter()
    for name, reason in LATE_ARRIVALS.items():
        for path in (WEIGHTS[name], SIDECAR_DIGESTS[name]):
            if not path.is_file():
                raise SystemExit(f"{path} is missing: {reason}")
    build = OUT / "build/runtime"
    package = OUT / "package"
    if (OUT / "build").exists():
        shutil.rmtree(OUT / "build")
    package.mkdir(parents=True, exist_ok=True)

    # --- python tree: every .py under src/ofc_regular, nothing compiled -----
    source_root = build / "src/ofc_regular"
    source_root.mkdir(parents=True)
    copied = 0
    for path in sorted((REPO / "src/ofc_regular").rglob("*.py")):
        relative = path.relative_to(REPO / "src/ofc_regular")
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied += 1

    # --- configs: ship the whole directory; it is small and the previous
    # workstream lost a run to one missing config file ----------------------
    if (REPO / "configs").is_dir():
        shutil.copytree(REPO / "configs", build / "configs")

    (build / "models").mkdir()
    for name in MODEL_FILES:
        shutil.copyfile(REPO / "models" / name, build / "models" / name)
    (build / "native").mkdir()
    for name, source in NATIVE.items():
        shutil.copyfile(source, build / "native" / name)
    (build / "weights").mkdir()
    for name, source in WEIGHTS.items():
        shutil.copyfile(source, build / "weights" / name)

    # The solver binary, shared by every vs-Fantasyland kind. Hoisted out of the
    # t3 payload once a second kind needed it, so the two cannot ship different
    # binaries into the same archive.
    solver_target = None
    fl_ev_shipped = None
    if args.with_t3_vs_fl or args.with_t1_vs_fl:
        if not VSFL_SOLVER.is_file():
            raise SystemExit(
                f"{VSFL_SOLVER} is missing; build the fl_solver_regular crate "
                "in release mode before packaging a vs-Fantasyland payload"
            )
        (build / "native").mkdir(exist_ok=True)
        solver_target = build / "native/fl_solver_regular"
        shutil.copyfile(VSFL_SOLVER, solver_target)
        solver_target.chmod(0o755)
        (build / "roots").mkdir(exist_ok=True)
        fl_ev_shipped = build / VSFL_FL_EV_CONFIG_RELATIVE
        if not fl_ev_shipped.is_file():
            raise SystemExit(
                f"{VSFL_FL_EV_CONFIG_RELATIVE} did not reach the runtime tree; "
                "every vs-FL kind pins it and the labels bake its value in"
            )

    # The t3_vs_fl payload, when asked for. Nothing above this point changes,
    # so an archive built without the flag is the same bytes it always was.
    vsfl = None
    if args.with_t3_vs_fl:
        roots_source = pathlib.Path(args.vsfl_roots).expanduser()
        if not roots_source.is_file():
            raise SystemExit(f"{roots_source} is missing")
        roots_target = build / "roots/t3_vs_fl_roots.jsonl"
        shutil.copyfile(roots_source, roots_target)
        roots_count = sum(
            1 for line in roots_target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        vsfl = {
            "fl_solver_binary": "native/fl_solver_regular",
            "fl_solver_binary_sha256": sha256_of(solver_target),
            "roots_file": "roots/t3_vs_fl_roots.jsonl",
            "roots_file_sha256": sha256_of(roots_target),
            "roots_count": roots_count,
            "fl_ev_config": VSFL_FL_EV_CONFIG_RELATIVE,
            "fl_ev_config_sha256": sha256_of(fl_ev_shipped),
            "roots_source": str(roots_source),
        }

    vsfl_t2 = None
    if args.vsfl_t2_roots:
        t2_source = pathlib.Path(args.vsfl_t2_roots).expanduser()
        if not t2_source.is_file():
            raise SystemExit(f"{t2_source} is missing")
        if vsfl is None:
            raise SystemExit(
                "--vsfl-t2-roots needs --with-t3-vs-fl too: the solver binary "
                "and FL EV config are shipped by that payload and the T2 kind "
                "pins the same two"
            )
        (build / "roots").mkdir(exist_ok=True)
        t2_target = build / "roots/t2_vs_fl_roots.jsonl"
        shutil.copyfile(t2_source, t2_target)
        t2_count = sum(
            1 for line in t2_target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        vsfl_t2 = {
            "fl_solver_binary": vsfl["fl_solver_binary"],
            "fl_solver_binary_sha256": vsfl["fl_solver_binary_sha256"],
            "roots_file": "roots/t2_vs_fl_roots.jsonl",
            "roots_file_sha256": sha256_of(t2_target),
            "roots_count": t2_count,
            "fl_ev_config": vsfl["fl_ev_config"],
            "fl_ev_config_sha256": vsfl["fl_ev_config_sha256"],
            "roots_source": str(t2_source),
        }

    # The t1_vs_fl payload: the same solver and FL EV config as above, its own
    # roots file, and -- new at this street -- the two learned continuation
    # rankers, which no earlier kind needed because no earlier kind's labels
    # depended on a policy.
    vsfl_t1 = None
    if args.with_t1_vs_fl:
        t1_roots_source = pathlib.Path(args.vsfl_t1_roots).expanduser()
        if not t1_roots_source.is_file():
            raise SystemExit(f"{t1_roots_source} is missing")

        # The engine generation is stated on the command line and then verified
        # against the binary that will actually compute the features. A value
        # that is merely copied from a sibling plan would pass every check here
        # and fail on the VM, which is the defect class this builder exists to
        # prevent.
        rev = args.engine_features_rev
        if len(rev) != 64 or any(c not in "0123456789abcdef" for c in rev):
            raise SystemExit(
                f"--engine-features-rev must be a 64-character lowercase "
                f"sha256; got {rev!r}"
            )
        if rev.encode("ascii") not in solver_target.read_bytes():
            raise SystemExit(
                f"the shipped solver does not carry engine features rev {rev}; "
                "the binary was built against a different engine generation "
                "than the one this package claims"
            )

        t1_models = {}
        for name, source in VSFL_T1_MODELS.items():
            if not source.is_file():
                raise SystemExit(
                    f"{source} is missing; the t1_vs_fl teacher continues its "
                    "candidates with both rankers and a package carrying one "
                    "of them would fail on the VM"
                )
            fixture = VSFL_T1_MODEL_FIXTURES[name]
            if not fixture.is_file():
                raise SystemExit(f"{fixture} is missing")
            declared = json.loads(fixture.read_text(encoding="utf-8"))["model_sha256"]
            actual = sha256_of(source)
            if declared != actual:
                raise SystemExit(
                    f"{name} digest {actual} does not match its parity "
                    f"fixture's {declared}; the export is stale and its "
                    "measured parity does not describe these bytes"
                )
            shutil.copyfile(source, build / "models" / name)
            t1_models[name] = actual

        t1_roots_target = build / "roots/t1_vs_fl_roots.jsonl"
        shutil.copyfile(t1_roots_source, t1_roots_target)
        t1_count = sum(
            1 for line in t1_roots_target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        vsfl_t1 = {
            "fl_solver_binary": "native/fl_solver_regular",
            "fl_solver_binary_sha256": sha256_of(solver_target),
            "roots_file": "roots/t1_vs_fl_roots.jsonl",
            "roots_file_sha256": sha256_of(t1_roots_target),
            "roots_count": t1_count,
            "fl_ev_config": VSFL_FL_EV_CONFIG_RELATIVE,
            "fl_ev_config_sha256": sha256_of(fl_ev_shipped),
            "roots_source": str(t1_roots_source),
            "t2_model": "models/vsfl_t2_v1.vfl1",
            "t2_model_sha256": t1_models["vsfl_t2_v1.vfl1"],
            "t3_model": "models/vsfl_t3_v1.vfl1",
            "t3_model_sha256": t1_models["vsfl_t3_v1.vfl1"],
            "engine_features_rev": rev,
        }

    # Every weight image ships with a digest sidecar; a .bin that no longer
    # matches it is a stale export, and shipping one would pin the plan to a
    # model nobody measured. Cheaper to fail here than on the VM.
    if set(SIDECAR_DIGESTS) != set(WEIGHTS):
        raise SystemExit(
            "every weight must be cross-checked against a sidecar; missing: "
            f"{sorted(set(WEIGHTS) - set(SIDECAR_DIGESTS))}"
        )
    for name, sidecar in SIDECAR_DIGESTS.items():
        declared = json.loads(sidecar.read_text(encoding="utf-8"))["weights_sha256"]
        actual = sha256_of(WEIGHTS[name])
        if declared != actual:
            raise SystemExit(
                f"{name} digest {actual} does not match its sidecar's {declared}"
            )

    # --- archives -----------------------------------------------------------
    runtime_tar = package / "runtime.tar.gz"
    if runtime_tar.exists():
        runtime_tar.unlink()
    with tarfile.open(runtime_tar, "w:gz") as archive:
        archive.add(build, arcname="runtime", recursive=True)

    wheelhouse_zip = package / "wheelhouse.zip"
    if wheelhouse_zip.exists():
        wheelhouse_zip.unlink()
    wheels = sorted(WHEELHOUSE.glob("*.whl")) + sorted(WHEELHOUSE.glob("*.txt"))
    with zipfile.ZipFile(wheelhouse_zip, "w", zipfile.ZIP_STORED) as archive:
        for wheel in wheels:
            archive.write(wheel, arcname=f"wheelhouse/{wheel.name}")

    ledger = {
        "schema": "hu_m31_label_gen_package_ledger_v1",
        "engine_library_sha256": sha256_of(NATIVE["libofc_hu_m3_engine.so"]),
        "feature_encoder_library_sha256": sha256_of(
            NATIVE["libofc_stage3_feature_encoder.so"]
        ),
        "t4_model_sha256": sha256_of(WEIGHTS["t4_model_v5.bin"]),
        "t3_second_model_sha256": sha256_of(WEIGHTS["t3_model_v2.bin"]),
        "t3_first_model_sha256": sha256_of(WEIGHTS["t3first_model_v1.bin"]),
        "t2_second_model_sha256": sha256_of(WEIGHTS["t2_model_v1.bin"]),
        "t2_first_model_sha256": sha256_of(WEIGHTS["t2first_model_v1.bin"]),
        "t1_second_model_sha256": sha256_of(WEIGHTS["t1_model_v1.bin"]),
        "t1_first_model_sha256": sha256_of(WEIGHTS["t1first_model_v1.bin"]),
        "fast_t1_second_model_sha256": sha256_of(WEIGHTS["fast_t1_second_v1.bin"]),
        "fast_t1_first_model_sha256": sha256_of(WEIGHTS["fast_t1_first_v1.bin"]),
        "t0_second_model_sha256": sha256_of(WEIGHTS["t0_model_v1.bin"]),
        "fast_t0_second_model_sha256": sha256_of(
            WEIGHTS["fast_t0_second_v1.bin"]
        ),
        "fast_t2_first_model_sha256": sha256_of(WEIGHTS["fast_t2_first_v1.bin"]),
        "fast_t2_second_model_sha256": sha256_of(
            WEIGHTS["fast_t2_second_v1.bin"]
        ),
        # The root policy, keyed by the engine field that reads it so a plan
        # generator can copy the name rather than invent one.
        "t0_first_model_sha256": sha256_of(WEIGHTS["t0first_model_v1.bin"]),
        "runtime_archive": {
            "file": runtime_tar.name,
            "bytes": runtime_tar.stat().st_size,
            "sha256": sha256_of(runtime_tar),
        },
        "wheelhouse_archive": {
            "file": wheelhouse_zip.name,
            "bytes": wheelhouse_zip.stat().st_size,
            "sha256": sha256_of(wheelhouse_zip),
        },
        "python_files": copied,
        "model_files": len(MODEL_FILES),
        "weight_files": len(WEIGHTS),
        "wheel_files": len(wheels),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # Present only when the payload is. A consumer that does not know the key
    # reads the same ledger it always read.
    if vsfl is not None:
        ledger["t3_vs_fl"] = vsfl
    if vsfl_t2 is not None:
        ledger["t2_vs_fl"] = vsfl_t2
    if vsfl_t1 is not None:
        ledger["t1_vs_fl"] = vsfl_t1
    (package / "ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"runtime.tar.gz    {runtime_tar.stat().st_size/1e6:8.1f} MB "
          f"({copied} py files, {len(MODEL_FILES)} models, "
          f"{len(WEIGHTS)} weights)")
    print(f"wheelhouse.zip    {wheelhouse_zip.stat().st_size/1e6:8.1f} MB "
          f"({len(wheels)} wheels)")
    print(f"ledger.json written; build took {time.perf_counter()-began:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
