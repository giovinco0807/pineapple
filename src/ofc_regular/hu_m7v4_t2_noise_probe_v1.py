#!/usr/bin/env python3
"""T2 label-noise probe: price the evaluation-sample rung against a deeper reference.

Generated from `t3noise_probe.py` by `m7v4_t2noise_make.py`, which applies a
fixed list of literal substitutions and refuses to write unless every one of
them matched exactly once. What changed: the root generator, the evaluator,
the pin set (T2 needs both T3 evaluators, and the first seat additionally
needs the opponent's T2 reply), and the rung ladder. What did NOT change:
the seed formulas, the independent-particle-stream assertion, the
shared-seed control, the regret statistic and the resume-by-file logic.

Mirrors the production label generator (hu_m31_label_gen_worker_v1.run) for root
construction and JointExactConfig, then scores every root at several
`evaluation_samples` rungs and at one deeper REFERENCE.

The reference draws from an INDEPENDENT seed block. This is deliberate and
load-bearing: a rung that shares the reference's particle stream replays a
PREFIX of the reference's own draws -- the engine starts every evaluation batch
at sample_index 0, so 256 particles are literally the first 256 of 4096 -- and
is then scored against itself, which manufactures a near-zero regret.

The engine's particle key is (base_seed, run_id, street, root fingerprint,
sample_index): see belief.rs::rng_key_digest. run_id is IN the key, so a
differing run_id alone already decorrelates two batches and a "seeds differ"
assertion on its own proves nothing. This probe therefore separates rung from
reference in BOTH base_seed and run_id, and verifies independence by measuring
the engine's own reported `evaluation_rng_key_digests` overlap rather than by
reasoning about seeds. Real rungs must overlap the reference in zero particles
or the root is refused. `--shared-seed-control` deliberately reuses the
reference's seed AND run_id to reproduce the burned bug, which is what gives
the zero-overlap assertion teeth.

Regret, not agreement, is the selection metric:
    regret(rung) = ref_score[argmax ref] - ref_score[argmax rung]   (>= 0 by construction)
Agreement is reported but never selected on.

Writes one JSON per (root offset) into --out, create-only, so a restart resumes
and a kill keeps every root already finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import time


# --------------------------------------------------------------------------
# helpers


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def core_seconds() -> float:
    """User+system CPU of this process and its threads, in seconds."""
    times = os.times()
    return times.user + times.system


def argmax_by_score(rows: list[dict]) -> str:
    """Action key of the highest evaluation score; ties broken by action key.

    Deterministic and independent of the engine's candidate-plan ordering, which
    is driven by candidate_samples (fixed across rungs) rather than by the
    evaluation set the rung actually controls.
    """
    best_key = None
    best_score = -math.inf
    for row in sorted(rows, key=lambda r: r["action_key"]):
        score = float(row["score"])
        if score > best_score:
            best_score = score
            best_key = row["action_key"]
    if best_key is None:
        raise RuntimeError("empty action set")
    return best_key


def digest_summary(digests: list[str]) -> dict:
    ordered = sorted(digests)
    return {
        "count": len(digests),
        "unique": len(set(digests)),
        "set_sha256": hashlib.sha256("".join(ordered).encode()).hexdigest()[:16],
        "first3": digests[:3],
    }


# --------------------------------------------------------------------------
# main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True,
                        help="directory holding native/ and weights/")
    parser.add_argument("--t4-model", required=True,
                        help="T4 model .bin; absolute, or relative to the "
                             "runtime root. NOT defaulted: the probe must be "
                             "re-runnable against a retrained image")
    parser.add_argument("--t3-second-model", default="weights/t3_model_v2.bin",
                        help="T3 second-seat evaluator, used by the FIRST-seat "
                             "root only (the production plan omits it for the "
                             "second seat)")
    parser.add_argument("--t3-first-model", default="weights/t3first_model_v1.bin",
                        help="T3 first-seat evaluator. Required at BOTH "
                             "T2 seats: the T2 teacher plays the "
                             "opponent's T3 first-seat reply through it")
    parser.add_argument("--t2-second-model", default="weights/t2_model_v1.bin",
                        help="T2 second-seat evaluator, the OPPONENT's "
                             "T2 reply, used by the FIRST-seat root only")
    parser.add_argument("--engine-library", default="native/libofc_hu_m3_engine.so")
    parser.add_argument("--feature-encoder",
                        default="native/libofc_stage3_feature_encoder.so")
    parser.add_argument("--roots", type=int, default=20)
    parser.add_argument("--root-start", type=int, default=0,
                        help="first behavior offset; offsets run "
                             "[root_start, root_start + roots)")
    # The shipped T2 corpora are 128-particle labels, so the ladder brackets
    # what shipped rather than starting above it.
    parser.add_argument("--samples", default="128,256,512",
                        help="comma-separated evaluation_samples rungs")
    parser.add_argument("--reference-samples", type=int, default=2048)
    parser.add_argument("--seats", default="first,second")
    # Seed blocks. The rung base mirrors the production plan's eval_seed_base;
    # the reference base is a DIFFERENT block, which is the whole point.
    parser.add_argument("--rung-seed-base", type=int, default=5_000_000)
    parser.add_argument("--reference-seed-base", type=int, default=91_000_000)
    parser.add_argument("--hand-seed-base", type=int, default=960_000_000)
    parser.add_argument("--behavior-seed-offset", type=int, default=500_000)
    parser.add_argument("--shared-seed-control", action="store_true",
                        help="ALSO score the smallest rung on the REFERENCE's "
                             "own particle stream (same base_seed AND same "
                             "run_id). Validation only: this reproduces the "
                             "burned self-comparison bug on purpose, so the "
                             "probe can show it detects it")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="probe")
    parser.add_argument("--stride-index", type=int, default=0)
    parser.add_argument("--stride-count", type=int, default=1)
    args = parser.parse_args()

    runtime = pathlib.Path(args.runtime_root).resolve()

    def under_runtime(text: str) -> pathlib.Path:
        candidate = pathlib.Path(text)
        path = candidate if candidate.is_absolute() else (runtime / text)
        path = path.resolve()
        if not path.is_file():
            raise SystemExit(f"missing file: {path}")
        return path

    engine_path = under_runtime(args.engine_library)
    encoder_path = under_runtime(args.feature_encoder)
    t4_path = under_runtime(args.t4_model)
    t3_second_path = under_runtime(args.t3_second_model)
    t3_first_path = under_runtime(args.t3_first_model)
    t2_second_path = under_runtime(args.t2_second_model)

    t4_sha = sha256_of(t4_path)
    t3_second_sha = sha256_of(t3_second_path)
    t3_first_sha = sha256_of(t3_first_path)
    t2_second_sha = sha256_of(t2_second_path)
    encoder_sha = sha256_of(encoder_path)
    engine_sha = sha256_of(engine_path)

    rungs = [int(x) for x in args.samples.split(",") if x.strip()]
    if not rungs:
        raise SystemExit("--samples must name at least one rung")
    rungs.sort()
    reference_samples = int(args.reference_samples)
    if reference_samples < 4 * rungs[-1]:
        raise SystemExit(
            f"reference {reference_samples} must be at least 4x the largest "
            f"rung {rungs[-1]} (house discipline); refusing to run"
        )
    seats = [s.strip() for s in args.seats.split(",") if s.strip()]
    for seat in seats:
        if seat not in ("first", "second"):
            raise SystemExit(f"unknown seat {seat!r}")

    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    from ofc_regular import hu_m3_rust
    from ofc_regular.ai_profiles import ModelPaths, load_model_bundle
    from ofc_regular.hu_infoset import load_default_fl_ev
    from ofc_regular.hu_m31_t2_behavior_roots import generate_behavior_t2_roots
    from ofc_regular.hu_m31_t3_behavior_roots import behavior_profile_for_index
    from ofc_regular.hu_turn3_joint_exact_teacher import JointExactConfig
    from ofc_regular.hu_turn3_stage3_feature_rust import pinned_feature_encoder_library

    library = hu_m3_rust.load_native_engine(path=engine_path, build_if_missing=False)
    engine_version = hu_m3_rust.engine_version(library=library)
    bundle = load_model_bundle(
        ModelPaths(), {behavior_profile_for_index(i) for i in range(5)}
    )
    fl_ev = load_default_fl_ev()

    def rung_seed(offset: int, rung_index: int) -> int:
        # The production formula, with the rung index in the slot the plan gives
        # the trial index: offset*7 separates roots, the large stride separates
        # rungs so no two rungs replay each other either.
        return args.rung_seed_base + offset * 7 + rung_index * 3_000_017

    def reference_seed(offset: int) -> int:
        return args.reference_seed_base + offset * 7

    def build_config(seat: str, samples: int, seed: int, run_id: str):
        # Both T2 seats reach the terminal through both T3 evaluators, so
        # both are pinned at both seats -- unlike T3, where only the first
        # seat pins a continuation. The first seat additionally pins the
        # opponent's T2 reply, which the second seat has already made.
        extra = {
            "learned_t3_second_model_path": str(t3_second_path),
            "learned_t3_second_model_sha256": t3_second_sha,
            "learned_t3_first_model_path": str(t3_first_path),
            "learned_t3_first_model_sha256": t3_first_sha,
        }
        if seat == "first":
            extra["learned_t2_second_model_path"] = str(t2_second_path)
            extra["learned_t2_second_model_sha256"] = t2_second_sha
        return JointExactConfig(
            candidate_samples=8,
            evaluation_samples=samples,
            downstream_t3_samples=4,
            downstream_t4_samples=0,
            seed=seed,
            candidate_seed=seed,
            evaluation_seed=seed,
            run_id=run_id,
            seat=seat,
            to_act_order=seat,
            learned_t4_model_path=str(t4_path),
            learned_t4_model_sha256=t4_sha,
            **extra,
        )

    def score(root, seat: str, samples: int, seed: int, run_id: str) -> dict:
        wall0 = time.perf_counter()
        cpu0 = core_seconds()
        result = hu_m3_rust.evaluate_t2(
            root, config=build_config(seat, samples, seed, run_id), library=library
        )
        wall = time.perf_counter() - wall0
        cpu = core_seconds() - cpu0
        if result.get("status") != "ok":
            raise SystemExit(f"engine returned {result!r}")
        rows = [
            {"action_key": r["action_key"], "score": float(r["score"])}
            for r in result["actions"]
        ]
        return {
            "samples": samples,
            "seed": seed,
            "run_id": run_id,
            "rows": rows,
            "argmax": argmax_by_score(rows),
            "legal_action_count": result["legal_action_count"],
            "legal_action_set_digest": result["legal_action_set_digest"],
            "engine_locked_selection_key": result["selected_action_key"],
            "engine_in_sample_regret_of_locked_selection": float(
                result["evaluation_sample_regret_of_locked_selection"]
            ),
            "continuation_policy": result["continuation_policy"],
            "evaluation_rng_key_digests": list(
                result.get("evaluation_rng_key_digests", [])
            ),
            "sample_independence": result.get("sample_independence"),
            "wall_seconds": wall,
            "core_seconds": cpu,
        }

    def compare(rung: dict, ref: dict) -> dict:
        if rung["legal_action_set_digest"] != ref["legal_action_set_digest"]:
            raise SystemExit(
                "rung and reference disagree on the legal action set; the two "
                "are not scoring the same root"
            )
        ref_scores = {r["action_key"]: r["score"] for r in ref["rows"]}
        rung_scores = {r["action_key"]: r["score"] for r in rung["rows"]}
        if set(ref_scores) != set(rung_scores):
            raise SystemExit("rung and reference action keys differ")
        ref_best_key = ref["argmax"]
        regret = ref_scores[ref_best_key] - ref_scores[rung["argmax"]]
        if not math.isfinite(regret):
            raise SystemExit(f"non-finite regret {regret}")
        if regret < -1e-9:
            raise SystemExit(
                f"negative regret {regret}: the reference argmax is not the "
                "reference's own maximum, which is impossible unless the "
                "comparison is mis-wired"
            )
        regret = max(regret, 0.0)
        diffs = [rung_scores[k] - ref_scores[k] for k in ref_scores]
        mse = sum(d * d for d in diffs) / len(diffs)
        rmse = math.sqrt(mse)
        # var(rung - ref) = s^2/n_rung + s^2/n_ref, so the rung's own per-label
        # standard error is the RMSE deflated by the reference's share of it.
        # The engine reports no per-action standard error, so this is the
        # probe's estimate rather than an engine number.
        deflate = math.sqrt(1.0 + rung["samples"] / ref["samples"])
        # Particle-set independence, MEASURED on the engine's own per-particle
        # keys rather than inferred from the seeds. run_id is inside that key,
        # so "the seeds differ" would prove nothing on its own.
        rung_digests = set(rung["evaluation_rng_key_digests"])
        ref_digests = set(ref["evaluation_rng_key_digests"])
        if rung_digests and ref_digests:
            shared = len(rung_digests & ref_digests)
            subset = rung_digests <= ref_digests
            basis = "measured_particle_digests"
        else:
            # Not measurable at this street. Fall back to the key
            # construction -- belief.rs builds a particle key from
            # (base_seed, run_id, street, root fingerprint, sample_index),
            # so differing in BOTH base_seed and run_id is what makes the
            # streams disjoint -- and let the shared-seed control show
            # empirically that a replayed stream looks different.
            same = (rung["seed"] == ref["seed"]
                    and rung["run_id"] == ref["run_id"])
            shared = None if not same else rung["samples"]
            subset = None if not same else True
            basis = "argued_from_key_construction"
        return {
            "regret": regret,
            "agree": rung["argmax"] == ref_best_key,
            "rung_argmax": rung["argmax"],
            "ref_argmax": ref_best_key,
            "ref_score_of_ref_argmax": ref_scores[ref_best_key],
            "ref_score_of_rung_argmax": ref_scores[rung["argmax"]],
            "score_rmse_vs_reference": rmse,
            "per_label_se_estimate": rmse / deflate,
            "mean_abs_score_diff": sum(abs(d) for d in diffs) / len(diffs),
            "shared_particle_digests": shared,
            "independence_basis": basis,
            "rung_particle_subset_of_reference": subset,
            "same_seed_as_reference": rung["seed"] == ref["seed"],
            "same_run_id_as_reference": rung["run_id"] == ref["run_id"],
        }

    offsets = [
        o for o in range(args.root_start, args.root_start + args.roots)
        if (o - args.root_start) % args.stride_count == args.stride_index
    ]
    todo = [o for o in offsets if not (out / f"root_{o:05d}.json").exists()]
    print(f"[{args.tag}] stride {args.stride_index}/{args.stride_count}: "
          f"{len(todo)}/{len(offsets)} roots to score; rungs={rungs} "
          f"reference={reference_samples} seats={seats}", flush=True)
    print(f"[{args.tag}] engine {engine_version} fl_ev={fl_ev} "
          f"t4={t4_path} sha={t4_sha[:12]}", flush=True)

    header = {
        "schema": "ofc_m7_t2_noise_probe_v1",
        "tag": args.tag,
        "engine_version": engine_version,
        "engine_library": str(engine_path),
        "engine_library_sha256": engine_sha,
        "feature_encoder": str(encoder_path),
        "feature_encoder_sha256": encoder_sha,
        "t4_model": str(t4_path),
        "t4_model_sha256": t4_sha,
        "t3_second_model": str(t3_second_path),
        "t3_second_model_sha256": t3_second_sha,
        "t3_first_model": str(t3_first_path),
        "t3_first_model_sha256": t3_first_sha,
        "t2_second_model": str(t2_second_path),
        "t2_second_model_sha256": t2_second_sha,
        "fl_ev": {str(k): v for k, v in fl_ev.items()},
        "rungs": rungs,
        "reference_samples": reference_samples,
        "rung_seed_base": args.rung_seed_base,
        "reference_seed_base": args.reference_seed_base,
        "hand_seed_base": args.hand_seed_base,
        "behavior_seed_offset": args.behavior_seed_offset,
        "seats": seats,
        "shared_seed_control": bool(args.shared_seed_control),
    }
    manifest = out / f"manifest_stride{args.stride_index}.json"
    manifest.write_text(json.dumps(header, indent=2, sort_keys=True), encoding="utf-8")

    began = time.perf_counter()
    with pinned_feature_encoder_library(encoder_path, expected_sha256=encoder_sha):
        for done_count, offset in enumerate(todo, start=1):
            profile = behavior_profile_for_index(offset)
            first, second = generate_behavior_t2_roots(
                hand_seed=args.hand_seed_base + offset,
                behavior_seed=args.hand_seed_base + args.behavior_seed_offset + offset,
                profile=profile,
                bundle=bundle,
            )
            roots = {"first": first, "second": second}
            record = {
                "schema": "ofc_m7_t2_noise_probe_root_v1",
                "tag": args.tag,
                "offset": offset,
                "profile": profile,
                "hand_seed": args.hand_seed_base + offset,
                "behavior_seed": args.hand_seed_base + args.behavior_seed_offset + offset,
                "reference_samples": reference_samples,
                "reference_seed": reference_seed(offset),
                "rungs": rungs,
                "seats": {},
            }
            for seat in seats:
                root = roots[seat]
                geometry = {
                    "hero_cards": len(root.hero_board.all_cards()),
                    "opponent_public_cards": len(
                        root.opponent_public_board.all_cards()
                    ),
                    "dealt_cards": len(root.dealt_cards),
                    "hero_private_discards": len(root.hero_private_discards),
                    "observation_seat": root.seat,
                    "street": root.street,
                }
                reference_run_id = f"{args.tag}-{offset}-{seat}-ref"
                ref = score(
                    root, seat, reference_samples, reference_seed(offset),
                    reference_run_id,
                )
                seat_record = {
                    "geometry": geometry,
                    "reference": {
                        "samples": ref["samples"],
                        "seed": ref["seed"],
                        "run_id": ref["run_id"],
                        "argmax": ref["argmax"],
                        "legal_action_count": ref["legal_action_count"],
                        "legal_action_set_digest": ref["legal_action_set_digest"],
                        "engine_locked_selection_key": ref["engine_locked_selection_key"],
                        "engine_in_sample_regret_of_locked_selection":
                            ref["engine_in_sample_regret_of_locked_selection"],
                        "continuation_policy": ref["continuation_policy"],
                        "particles": digest_summary(ref["evaluation_rng_key_digests"]),
                        "wall_seconds": ref["wall_seconds"],
                        "core_seconds": ref["core_seconds"],
                        "scores": {r["action_key"]: r["score"] for r in ref["rows"]},
                    },
                    "rungs": [],
                }
                for rung_index, samples in enumerate(rungs):
                    seed = rung_seed(offset, rung_index)
                    run_id = f"{args.tag}-{offset}-{seat}-r{samples}"
                    if seed == ref["seed"] and run_id == reference_run_id:
                        raise SystemExit(
                            "rung shares the reference's particle key inputs; "
                            "it would be compared against its own draws"
                        )
                    rung = score(root, seat, samples, seed, run_id)
                    entry = {
                        "samples": samples,
                        "seed": seed,
                        "run_id": run_id,
                        "argmax": rung["argmax"],
                        "engine_locked_selection_key": rung["engine_locked_selection_key"],
                        "engine_in_sample_regret_of_locked_selection":
                            rung["engine_in_sample_regret_of_locked_selection"],
                        "particles": digest_summary(rung["evaluation_rng_key_digests"]),
                        "wall_seconds": rung["wall_seconds"],
                        "core_seconds": rung["core_seconds"],
                        "scores": {r["action_key"]: r["score"] for r in rung["rows"]},
                    }
                    entry.update(compare(rung, ref))
                    if entry["independence_basis"] == "argued_from_key_construction":
                        if (entry["same_seed_as_reference"]
                                or entry["same_run_id_as_reference"]):
                            raise SystemExit(
                                "this street does not report per-particle "
                                "keys, so independence rests on the rung "
                                "and the reference differing in BOTH "
                                "base_seed and run_id, and they do not"
                            )
                    if entry["shared_particle_digests"]:
                        raise SystemExit(
                            f"rung {samples} shares "
                            f"{entry['shared_particle_digests']} particles with "
                            "the reference; refusing to write a root whose "
                            "reference is partly a replay of the rung"
                        )
                    seat_record["rungs"].append(entry)
                if args.shared_seed_control:
                    # The burned bug, on purpose: the reference's seed AND its
                    # run_id, which together are what the engine's particle key
                    # is built from. Sharing only the seed would decorrelate
                    # anyway and prove nothing.
                    control = score(
                        root, seat, rungs[0], ref["seed"], reference_run_id,
                    )
                    entry = {
                        "samples": rungs[0],
                        "seed": control["seed"],
                        "run_id": control["run_id"],
                        "argmax": control["argmax"],
                        "particles": digest_summary(
                            control["evaluation_rng_key_digests"]
                        ),
                        "wall_seconds": control["wall_seconds"],
                        "core_seconds": control["core_seconds"],
                        "note": "DELIBERATE self-comparison: same base_seed and "
                                "same run_id as the reference, so these "
                                "particles are a strict PREFIX of the "
                                "reference's. The regret here is fake and "
                                "exists only to show the probe can tell the "
                                "two apart",
                    }
                    entry.update(compare(control, ref))
                    if entry["rung_particle_subset_of_reference"] is None:
                        raise SystemExit(
                            "the shared-seed control did not reproduce the "
                            "reference's key inputs, so it is not the bug "
                            "it exists to reproduce"
                        )
                    if not entry["rung_particle_subset_of_reference"]:
                        raise SystemExit(
                            "the shared-seed control did NOT land inside the "
                            "reference's particle set; the control is not "
                            "reproducing the bug it exists to reproduce, so "
                            "the zero-overlap assertion above has no teeth"
                        )
                    seat_record["shared_seed_control"] = entry
                record["seats"][seat] = seat_record

            path = out / f"root_{offset:05d}.json"
            with path.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=1, sort_keys=True)
            elapsed = time.perf_counter() - began
            print(f"[{args.tag}] root {offset} done "
                  f"({done_count}/{len(todo)}) {elapsed / done_count:.1f}s/root "
                  f"elapsed {elapsed / 60:.1f}m", flush=True)

    print(f"[{args.tag}] stride {args.stride_index} complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
