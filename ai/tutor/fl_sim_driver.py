"""Drive the Fantasyland simulator: calibrate it, then iterate it to a fixed point.

`fl_ev` is self-referential.  The stay decision and the placement both read the
table being measured, so a run under table T does not measure `fl_ev` -- it
measures `fl_ev` *given that the players believed T*.  The measurement is a map
and the answer is its fixed point, so the loop here is: measure under T, write
what came out into T, measure again, stop when the table stops moving.  Width
14's history is the evidence that this contracts: 0 -> 3.17 -> 5.28 -> 6.57,
increments in ratio about 0.63.

The Rust side knows none of this.  One invocation is one pass under one table,
and it stamps that table's SHA-256 into its output so any number here can be
traced back to the table that produced it.

Common random numbers: every iteration and every width uses the same seed.  The
difference between two iterations is then the change in play -- stays,
placements, the opponent's entries -- and not a different deck.  The pairing is
only approximate, because one changed stay shifts every later deal in that
block, but it is far tighter than independent seeds.

Which number goes into the table is a second question, and the answer is not the
run's own mean.  The table is read where a qualification is entered -- the
teacher's leaf credit, the mirror's credit, the stay bonus -- and what an
entering player faces is the distribution of opponent states *at entry*, not the
distribution a chain visits once it is running.  This simulator's hero is pinned
in Fantasyland forever, so its opponent sits in Fantasyland more often than a
real entrant's does, by five points of share at width 16 and three and a half
points of `fl_ev`.  So the loop is driven by `fl_ev_strat`: the same chains,
post-stratified on the opponent's state at the chain's first hand and reweighted
to the reference's entry composition.  Both readings are reported everywhere,
because at width 15 they straddle the T0 placement's crossing point and the
choice between them is the decision, not a footnote to it.

The reference is an anchor, not a target.  It was played by the 2026-08-19
binary and this one is not that binary (`ai/rust_solver/t4_first_exact/src` is
untracked, so the difference is not even expressible as a diff); the calibration
therefore books the residual as a ledger -- session truncation plus engine drift
-- and gates the part that has no name, rather than demanding the two agree.

This driver does not write `ai/config/fl_ev.json`.  Adopting a table is a
generation boundary (`ai/docs/hu_lap2_verdict_20260821.md` §3: a changed table
means a pool header mismatch, a pool regeneration, and a generation note on
every existing label), and that is the owner's call.  The job here ends at the
report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_flsim import LABELS, analyze, mean_se  # noqa: E402

REPO = Path("C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple")
BINARY = REPO / "ai/rust_solver/target/release/t4_first_exact.exe"
OWN = Path("D:/ofc_data/hu/models_union/own_lap4")
BASE_CONFIG = REPO / "ai/config/fl_ev.json"

FLSIM = Path("D:/ofc_data/hu/flsim")
FLEV1_HANDS = Path("D:/ofc_data/hu/flev1/flev1_hands.jsonl")
FLEV1_FINAL = Path("D:/ofc_data/hu/flev1/flev1_final.json")
REF_SHARES = FLSIM / "ref_shares_flev1.json"
CAL700 = FLSIM / "cal700_20260822"
ANCHOR_README = FLSIM / "anchor/README.txt"

WIDTHS = (14, 15, 16, 17)

# The confirmation lap's sizes: enough that the gate's 2*SE_sim is tighter than
# the tolerance it is checked against, which is what makes the gate mean
# something.
CONFIRM_BLOCKS = {14: 800, 15: 800, 16: 1500, 17: 700}
CONFIRM_TOL = {14: 0.3, 15: 0.3, 16: 0.4, 17: 1.0}

# The table the reference data was played under (2026-08-19).  The live config
# has since moved width 14 to 6.57, so calibrating against it would be comparing
# two different games; calibration re-plays the reference game.
CAL_TABLE = {14: 5.28, 15: 16.61, 16: 38.76, 17: 70.07}
CAL_SEED = 88000001
CAL_BLOCKS = 3000
CAL_K = 8

# Reference numbers, read from D:/ofc_data/hu/flev1/flev1_final.json (gen-2
# self-play, 60,000 hands, 2026-08-19, own_lap4 both seats, table CAL_TABLE),
# which is the archived copy of the run the verified analyzer reported on.
# Transcribed here so the gate does not depend on that file staying put; the
# archived file is the authority if the two ever disagree.
REF = {
    "chain_raw": {14: 6.549229738780978, 15: 16.675809199318568,
                  16: 38.80994027303754, 17: 78.24678663239075},
    "chain_paid": {14: 6.569323509711989, 15: 16.65519591141397,
                   16: 38.50959897610922, 17: 75.79434447300771},
    "chain_se": {14: 0.6535709424775188, 15: 0.4585590949335571,
                 16: 0.43281333804021194, 17: 3.7851019777509216},
    "chain_n": {14: 1493, 15: 5870, 16: 14064, 17: 389},
    "stay_rate": {14: 0.3486038394415358, 15: 0.49036291022747003,
                  16: 0.6265732037597579, 17: 0.7483829236739974},
    "mean_chain_length": {14: 1.5351640991292699, 15: 1.9621805792163542,
                          16: 2.6779010238907848, 17: 3.974293059125964},
    "vs_normal_raw": {14: 13.381831610044314, 15: 17.8363117585223,
                      16: 23.407468916917278, 17: 28.826185101580137},
    "vs_normal_se": {14: 0.342865573407394, 15: 0.15993356505584042,
                     16: 0.08715157185308303, 17: 0.4127534662041541},
    "vs_normal_n": {14: 1354, 15: 6659, 16: 22279, 17: 886},
    # hero width -> opponent width, the convolution cells the gate uses.
    "fl_vs_fl_raw": {(15, 16): -5.408784694152519, (16, 16): 0.0},
    "fl_vs_fl_se": {(15, 16): 0.24692567261106244, (16, 16): 0.1520360096471832},
    "fl_vs_fl_n": {(15, 16): 3711, (16, 16): 10484},
    "_source": "D:/ofc_data/hu/flev1/flev1_final.json",
}


def sd_of(width: int) -> float:
    """One chain's standard deviation at this width, from the reference run."""
    return REF["chain_se"][width] * math.sqrt(REF["chain_n"][width])


def write_config(table: dict[int, float], path: Path, tag: str) -> None:
    """A copy of the canonical config with the table swapped.

    The canonical file is read and never written: it prices every label this
    project has made, and a driver that edited it would silently change the
    meaning of work that is already on disk.
    """
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    values = {str(width): float(table[width]) for width in WIDTHS}
    # The engine reads `fl_ev`; `fl_ev_direct` is kept in step only so a human
    # reading this file is not told two different stories.
    config["fl_ev"] = dict(values)
    config["fl_ev_direct"] = dict(values)
    config["source"] = f"{config.get('source', '')} | fl_sim_driver {tag}".strip(" |")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(config, indent=2) + "\n")


def run_sim(
    width: int,
    blocks: int,
    k: int,
    seed: int,
    table: dict[int, float],
    workdir: Path,
    tag: str,
    ref_shares: Path | None = None,
) -> dict:
    """One pass: write the table, play it, analyze it.  Returns the report."""
    config_path = workdir / f"config_{tag}.json"
    run_path = workdir / f"run_{tag}.jsonl"
    report_path = workdir / f"report_{tag}.json"
    write_config(table, config_path, tag)

    argv = [
        str(BINARY),
        "--fl-sim",
        "--fl-sim-width", str(width),
        "--fl-sim-blocks", str(blocks),
        "--fl-sim-chains-per-block", str(k),
        "--self-play-seed", str(seed),
        "--play-t0-model", str(OWN / "t0.bin"),
        "--play-t1-model", str(OWN / "t1.bin"),
        "--play-t2-model", str(OWN / "t2.bin"),
        "--fl-ev-config", str(config_path),
        "--output", str(run_path),
    ]
    print(f"[{tag}] {' '.join(argv[1:])}", flush=True)
    started = time.time()
    done = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True)
    elapsed = time.time() - started
    if done.returncode != 0:
        tail = "\n".join((done.stderr or "").splitlines()[-20:])
        raise SystemExit(
            f"[{tag}] --fl-sim exited {done.returncode}\n{tail}"
        )

    report = analyze(run_path, ref_shares)
    report["wall_seconds"] = elapsed
    report["logical_cores"] = os.cpu_count()
    hands = report["counts"]["hands"]
    chains = report["counts"]["chains"]
    report["seconds_per_hand"] = elapsed / max(hands, 1)
    report["seconds_per_chain"] = elapsed / max(chains, 1)
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")

    mean = report["fl_ev_all"]["mean"]
    se = report["fl_ev_all"]["se"]
    strat = report.get("fl_ev_strat")
    strat_text = (
        f"strat {strat['mean']:+.3f} (sim {strat['se_sim']:.3f}, "
        f"share {strat['se_share']:.3f}), "
        if strat
        else ""
    )
    print(
        f"[{tag}] {strat_text}natural {mean:+.3f} +/- {se:.3f} "
        f"(n={report['fl_ev_all']['n']}), stay {report['stay_rate']:.4f}, "
        f"{elapsed:.1f} s ({report['seconds_per_hand']:.3f} s/hand)",
        flush=True,
    )
    return report


def headline(report: dict) -> tuple[float, float]:
    """The measurement the table takes, and its simulation error.

    The stratified reading when there is one -- the table's consumers price
    entries, and an entry's opponent is drawn from the entry composition -- and
    the natural reading only when no reference composition was supplied.
    """
    strat = report.get("fl_ev_strat")
    if strat:
        return strat["mean"], strat["se_sim"]
    return report["fl_ev_all"]["mean"], report["fl_ev_all"]["se"]


def per_block_series(run: Path, pi: dict[str, float]) -> dict[int, dict]:
    """Per-block statistics whose mean over blocks is the run's estimator.

    Two laps under the same seed deal the same cards until a decision diverges,
    so differencing them block by block cancels most of the deck and resolves a
    lap-to-lap move far below the headline standard error.  The natural series
    is a block's headline chain mean.  The stratified series is the
    linearization

        z_b = B * sum over chains c in block b of  pi[s(c)] / n[s(c)] * y_c

    whose mean over the B blocks is exactly the stratified estimate with the
    stratum counts held fixed -- the counts move between laps too, but their
    contribution is second order and folding it in would need the two files to
    share a stratification, which they do not.
    """
    chains: list[dict] = []
    with run.open(encoding="utf-8") as handle:
        handle.readline()  # metadata
        for line in handle:
            if line.strip():
                chains.append(json.loads(line))
    counts: dict[str, int] = defaultdict(int)
    for chain in chains:
        counts[chain["opp_at_start"]] += 1
    blocks = sorted({chain["block"] for chain in chains})
    natural: dict[int, list[float]] = defaultdict(list)
    weighted: dict[int, float] = defaultdict(float)
    for chain in chains:
        label = chain["opp_at_start"]
        if not chain["burn_in"]:
            natural[chain["block"]].append(chain["sum_settle"])
        share = pi.get(label, 0.0)
        if share > 0.0 and counts[label]:
            weighted[chain["block"]] += share * chain["sum_settle"] / counts[label]
    n_blocks = len(blocks)
    return {
        block: {
            "natural": (
                sum(natural[block]) / len(natural[block]) if natural[block] else None
            ),
            "strat": weighted[block] * n_blocks,
        }
        for block in blocks
    }


def paired_delta(before: dict[int, dict], after: dict[int, dict], key: str) -> dict:
    """The common-random-numbers standard error of one lap-to-lap move."""
    diffs = [
        after[block][key] - before[block][key]
        for block in sorted(set(before) & set(after))
        if before[block][key] is not None and after[block][key] is not None
    ]
    n = len(diffs)
    if n < 2:
        return {"n_blocks": n, "mean": None, "se": None}
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return {"n_blocks": n, "mean": mean, "se": math.sqrt(var / n)}


def gate(name: str, observed: float, reference: float, allowed: float, note: str = "") -> dict:
    delta = observed - reference
    row = {
        "gate": name,
        "reference": reference,
        "observed": observed,
        "delta": delta,
        "allowed_abs_delta": allowed,
        "pass": abs(delta) <= allowed,
    }
    if note:
        row["note"] = note
    return row


def three_sigma(se_ref: float, se_sim: float) -> float:
    return 3.0 * math.sqrt(se_ref * se_ref + se_sim * se_sim)


def v1_gates(reports: dict[int, dict]) -> list[dict]:
    """The eight gates of the v1 calibration, from two width reports.

    Kept as one function so `--recalibrate` recomputes them by running the same
    code rather than by re-deriving it: a transcription that agrees because the
    numbers were copied proves nothing about the numbers.
    """
    rows = []

    # C1 -- the mechanical gate.  Conditioned on the two states, the hand this
    # simulator plays and the hand the reference played are the *same* hand: a
    # fresh 54-card shuffle dealt disjointly, a blind own_lap4 normal side, the
    # same play_fl, the same table.  A disagreement here is a bug, not a
    # structural difference, which is why this gate is tight and C2 is loose.
    for width in (15, 16):
        sim = reports[width]["per_hand"]["fl_vs_normal"]
        rows.append(gate(
            f"C1a.w{width}.fl_vs_normal",
            sim["mean"], REF["vs_normal_raw"][width],
            three_sigma(REF["vs_normal_se"][width], sim["se"]),
            f"n_sim={sim['n']}, n_ref={REF['vs_normal_n'][width]}",
        ))

    # C1b -- two blind players at equal width over exchangeable deals: the true
    # mean is exactly zero.  This is the mirror's zero check.
    sim = reports[16]["per_hand"]["fl_vs_fl"].get("16", {"mean": None, "se": None, "n": 0})
    if sim["n"] < 2:
        rows.append({
            "gate": "C1b.w16.fl_vs_fl16", "pass": False,
            "note": f"only {sim['n']} symmetric hands, cannot test",
        })
    else:
        rows.append(gate(
            "C1b.w16.fl_vs_fl16", sim["mean"], 0.0, 3.0 * sim["se"],
            f"n_sim={sim['n']}; true mean is exactly 0 by symmetry",
        ))

    # C1c -- the asymmetric convolution cell.
    sim = reports[15]["per_hand"]["fl_vs_fl"].get("16", {"mean": None, "se": None, "n": 0})
    if sim["n"] < 2:
        rows.append({
            "gate": "C1c.w15.fl_vs_fl16", "pass": False,
            "note": f"only {sim['n']} hands, cannot test",
        })
    else:
        rows.append(gate(
            "C1c.w15.fl_vs_fl16", sim["mean"], REF["fl_vs_fl_raw"][(15, 16)],
            three_sigma(REF["fl_vs_fl_se"][(15, 16)], sim["se"]),
            f"n_sim={sim['n']}, n_ref={REF['fl_vs_fl_n'][(15, 16)]}",
        ))

    # C2 -- the episodic calibration, loose enough to swallow the known
    # structural differences (no session truncation, no stacks, a blind
    # opponent whose entry rate is a point below the served one).  Every one of
    # those pushes the simulator *up*, so a negative miss has nothing to explain
    # it and is treated as a warning even when it fits inside the tolerance.
    for width in (15, 16):
        row = gate(
            f"C2.w{width}.fl_ev_all",
            reports[width]["fl_ev_all"]["mean"], REF["chain_raw"][width], 2.0,
            f"se_sim={reports[width]['fl_ev_all']['se']:.3f}, "
            f"se_ref={REF['chain_se'][width]:.3f}",
        )
        row["warn_negative"] = row["delta"] < -0.5
        rows.append(row)

    # C3 -- the stay rate.  Same play_fl on both sides, so this moves only with
    # the mix of opponents; more than three points is a table mix-up.
    for width in (15, 16):
        row = gate(
            f"C3.w{width}.stay_rate",
            reports[width]["stay_rate"], REF["stay_rate"][width], 0.03,
        )
        row["stay_rate_headline"] = reports[width]["stay_rate_headline"]
        rows.append(row)

    return rows


def calibrate(args: argparse.Namespace) -> int:
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    blocks = args.blocks
    reports = {}
    for width in (15, 16):
        reports[width] = run_sim(
            width, blocks, CAL_K, CAL_SEED, CAL_TABLE, workdir, f"cal_w{width}"
        )

    rows = v1_gates(reports)
    passed = all(row.get("pass") for row in rows)
    warnings = [row["gate"] for row in rows if row.get("warn_negative")]

    out = {
        "schema": "ofc_flsim_calibration/v1",
        "verdict": "PASS" if passed else "FAIL",
        "warnings": warnings,
        "table": {str(w): CAL_TABLE[w] for w in WIDTHS},
        "seed": CAL_SEED,
        "blocks": blocks,
        "chains_per_block": CAL_K,
        "reference": REF["_source"],
        "gates": rows,
        "diagnostics": {
            str(width): {
                "fl_ev_all": reports[width]["fl_ev_all"],
                "fl_ev_solo": reports[width]["fl_ev_solo"],
                "stay_rate": reports[width]["stay_rate"],
                "mean_chain_length": reports[width]["mean_chain_length"],
                "mean_chain_length_ref": REF["mean_chain_length"][width],
                "geometric_check": reports[width]["geometric_check"],
                "geometric_check_headline": reports[width]["geometric_check_headline"],
                "opp_entry_rate": reports[width]["opp_entry_rate"],
                "opp_entry_by_width": reports[width]["opp_entry_by_width"],
                "opp_fl_at_start_rate": reports[width]["opp_fl_at_start_rate"],
                "per_hand": reports[width]["per_hand"],
                "counts": reports[width]["counts"],
                "wall_seconds": reports[width]["wall_seconds"],
                "seconds_per_hand": reports[width]["seconds_per_hand"],
                "fl_ev_config_sha256": reports[width]["run"]["fl_ev_config_sha256"],
            }
            for width in (15, 16)
        },
        "logical_cores": os.cpu_count(),
    }
    path = workdir / "calibration.json"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(out, indent=2) + "\n")

    print(json.dumps({"verdict": out["verdict"], "gates": rows}, indent=2))
    print(f"\nwrote {path}", flush=True)
    return 0 if passed else 1


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def anchor_sha256() -> str | None:
    """The binary the calibration was measured with, from the anchor's README."""
    if not ANCHOR_README.exists():
        return None
    text = ANCHOR_README.read_text(encoding="utf-8")
    match = re.search(r"exe sha256[^\n]*\n\s*([0-9a-f]{64})", text)
    return match.group(1) if match else None


def git(*arguments: str) -> str:
    done = subprocess.run(
        ["git", *arguments], cwd=str(REPO), capture_output=True, text=True
    )
    return done.stdout if done.returncode == 0 else f"<git {' '.join(arguments)} failed: {done.stderr.strip()}>"


COMMIT_PROPOSAL = """\
To the owner, before the production laps:

  The Fantasyland engine -- self_play.rs, hu_match.rs, fl_sim.rs and the rest of
  t4_first_exact/src -- is untracked, and evaluator.rs carries uncommitted work.
  That is not a tidiness complaint.  The calibration found a real +2.83 at width
  16 between the 2026-08-19 reference and today's binary, present even on hands
  where the opponent had fouled and the hero placed blind, and there is no diff
  that can be shown for it because the two versions were never both in git.  A
  table measured tonight prices tonight's engine; if the engine moves again the
  same silence repeats.

  The proposal is to commit the tree before the laps start, so the measurement
  and the thing measured carry the same generation mark, and to adopt the table
  and the commit together when the time comes.  The driver takes the snapshot
  either way -- exe SHA, HEAD, status, diff stat, and a tar of the sources -- but
  a snapshot is a photograph, not a name.

  This is the owner's call; the driver does not commit anything."""


def snapshot_provenance(workdir: Path) -> dict:
    """Photograph the measuring instrument before it measures.

    Aborts if the binary is not the one the anchor names: a rebuilt exe is a
    different measurement device, and every number taken before it belongs to a
    different engine version -- which is exactly the confusion the calibration
    just spent a night untangling.
    """
    directory = workdir / "provenance"
    directory.mkdir(parents=True, exist_ok=True)
    observed = sha256_of(BINARY)
    expected = anchor_sha256()
    if expected and observed != expected:
        raise SystemExit(
            f"{BINARY} has sha256 {observed}, but the anchor "
            f"({ANCHOR_README}) names {expected}.  The binary was rebuilt: "
            f"rerun the anchor command in that README and `cmp` its output "
            f"against anchor_w16_s999.jsonl before measuring anything"
        )
    head = git("rev-parse", "HEAD").strip()
    status = git("status", "--porcelain")
    diffstat = git("diff", "--stat", "HEAD")
    for name, text in (
        ("git_head.txt", head + "\n"),
        ("git_status_porcelain.txt", status),
        ("git_diff_stat_head.txt", diffstat),
        ("exe_sha256.txt", f"{observed}  {BINARY}\n"),
        ("commit_proposal.txt", COMMIT_PROPOSAL + "\n"),
    ):
        with (directory / name).open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)

    sources = REPO / "ai/rust_solver/t4_first_exact/src"
    archive = directory / "t4_first_exact_src.tar"
    with tarfile.open(archive, "w") as tar:
        for rust in sorted(sources.glob("*.rs")):
            tar.add(rust, arcname=f"t4_first_exact/src/{rust.name}")

    untracked = [
        line[3:] for line in status.splitlines() if line.startswith("?? ")
    ]
    record = {
        "exe": str(BINARY),
        "exe_sha256": observed,
        "exe_sha256_anchor": expected,
        "exe_matches_anchor": expected is not None and observed == expected,
        "git_head": head,
        "git_status_porcelain": status.splitlines(),
        "git_diff_stat_head": diffstat.splitlines(),
        "untracked_paths": untracked,
        "sources_tar": str(archive),
        "sources_tar_sha256": sha256_of(archive),
        "commit_proposal": COMMIT_PROPOSAL,
        "note": (
            "the engine version is part of the measurement (addendum §A4): the "
            "reference and this binary differ by an unrecorded amount because "
            "the sources were never tracked"
        ),
    }
    with (directory / "provenance.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    print(f"provenance: exe {observed[:16]}... matches anchor, HEAD {head[:12]}, "
          f"{len(untracked)} untracked paths -> {directory}", flush=True)
    print(COMMIT_PROPOSAL, flush=True)
    return record


def width_of(state: str) -> int:
    return int(state[2:]) if state.startswith("fl") else 0


def extract_ref_shares(args: argparse.Namespace) -> int:
    """Derive the entry composition from the reference self-play, from scratch.

    The chain reconstruction is `analyze_selfplay_flev.py`'s -- open on a
    Fantasyland hand, close on `not stay or session end`, per (worker, seat) --
    with two things recorded that the reference analyzer had no use for: the
    opponent's state on the chain's first hand, which is the stratifying
    variable, and the hands a zero-stack session end fell on, which is the only
    way the reference truncates a chain.

    `gap` ends do not truncate: `hu_match.rs` requires both seats to be Normal
    *after* the transition, so a hero who stayed cannot be gap-ended.  `zero`
    ends do not check the state, and because the raw log has no stay field --
    the conversion derives it from the next hand's state -- a stay the zero end
    killed is recorded as a non-stay.  The truncation is invisible in the file
    and has to be reconstructed arithmetically, which is what `trunc_bias` is.

    Everything written out is computed from `flev1_hands.jsonl` alone and then
    checked against `flev1_final.json`.  A disagreement is fatal: the whole
    point of re-deriving is that the derivation is the thing being trusted.
    """
    final = json.loads(FLEV1_FINAL.read_text(encoding="utf-8"))

    open_chain: dict[tuple[int, str], dict] = {}
    chains: dict[int, list[dict]] = defaultdict(list)
    fl_hands: dict[int, int] = defaultdict(int)
    fl_stays: dict[int, int] = defaultdict(int)
    fl_vs_fl_hands: dict[int, int] = defaultdict(int)
    zero_hands: dict[int, int] = defaultdict(int)
    gap_hands: dict[int, int] = defaultdict(int)
    # The vs-normal cell, where the stay-rate comparison of C3 lives.
    vn: dict[int, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "stay": 0, "zero": 0, "gap": 0}
    )
    lines = 0

    with FLEV1_HANDS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            lines += 1
            hand = json.loads(line)
            end = hand["end"]
            for me, other, sign in (("a", "b", 1.0), ("b", "a", -1.0)):
                raw = sign * hand["settle_raw"]
                my_width = width_of(hand[f"state_{me}"])
                their = hand[f"state_{other}"]
                label = their if their.startswith("fl") else "normal"
                key = (hand["worker"], me)
                if not my_width:
                    # Defensive, exactly as the reference analyzer: a normal
                    # hand closes any chain bookkeeping left open.
                    chain = open_chain.pop(key, None)
                    if chain is not None:
                        chains[chain["width"]].append(chain)
                    continue
                fl_hands[my_width] += 1
                fl_stays[my_width] += 1 if hand[f"stay_{me}"] else 0
                if label != "normal":
                    fl_vs_fl_hands[my_width] += 1
                else:
                    cell = vn[my_width]
                    cell["n"] += 1
                    cell["stay"] += 1 if hand[f"stay_{me}"] else 0
                if end == "zero":
                    zero_hands[my_width] += 1
                    if label == "normal":
                        vn[my_width]["zero"] += 1
                elif end == "gap":
                    gap_hands[my_width] += 1
                    if label == "normal":
                        vn[my_width]["gap"] += 1
                chain = open_chain.get(key)
                if chain is None:
                    chain = {"width": my_width, "raw": 0.0, "length": 0,
                             "start": label, "opp_fl": 0}
                    open_chain[key] = chain
                chain["raw"] += raw
                chain["length"] += 1
                chain["opp_fl"] += 1 if label != "normal" else 0
                if not hand[f"stay_{me}"] or end:
                    chains[my_width].append(open_chain.pop(key))
    for key in list(open_chain):
        chain = open_chain.pop(key)
        chains[chain["width"]].append(chain)

    widths_out: dict[str, dict] = {}
    checks: list[dict] = []
    for width in WIDTHS:
        rows = chains[width]
        n = len(rows)
        if n == 0:
            raise SystemExit(f"{FLEV1_HANDS}: no width {width} chains")
        totals = [row["raw"] for row in rows]
        stats = mean_se(totals)
        length = sum(row["length"] for row in rows) / n
        stay = fl_stays[width] / fl_hands[width]
        counts = {label: sum(1 for row in rows if row["start"] == label)
                  for label in LABELS}
        shares = {label: counts[label] / n for label in LABELS}
        # The value the reference's own fl_ev is missing because a zero end cut
        # the chain: the rate at which a hero-Fantasyland hand is zero-ended,
        # times the chance that hand was going to continue, times how many hands
        # the continuation was worth, times what a hand of that continuation is
        # worth.  Memorylessness makes the residual value of a cut chain the
        # chain value itself, which is why fl_ev appears on the right.
        trunc = (zero_hands[width] / fl_hands[width]) * stay * length * stats["mean"]
        cell = vn[width]
        # A zero end erases the stay flag, so the recorded stay rate is the
        # stay count over *all* hands including ones whose flag was cleared.
        # Dropping those hands -- identical to imputing them at the rate of the
        # hands that kept their flag -- gives what was recorded had they not
        # been.  gap ends need no such correction: a gap end requires both seats
        # normal after the transition, so its hero genuinely did not stay.
        erasure = (
            cell["stay"] / (cell["n"] - cell["zero"]) - cell["stay"] / cell["n"]
            if cell["n"] and cell["n"] > cell["zero"]
            else None
        )
        reference_row = final["fl_ev_table"][str(width)]
        for name, got, want in (
            ("n_chains", n, reference_row["chains"]),
            ("fl_ev_raw", stats["mean"], reference_row["raw"]["fl_ev"]),
            ("se", stats["se"], reference_row["raw"]["se"]),
            ("mean_chain_length", length, reference_row["mean_chain_length"]),
            ("stay_rate", stay, final["stay_rate"][str(width)]),
        ):
            agree = got == want if isinstance(want, int) else abs(got - want) <= 1e-9
            checks.append({"width": width, "quantity": name, "derived": got,
                           "flev1_final": want, "agree": agree})
        widths_out[str(width)] = {
            "shares": shares,
            "chains_by_start": counts,
            "n_chains": n,
            "hero_fl_hands": fl_hands[width],
            "zero_end_hands": zero_hands[width],
            "gap_end_hands": gap_hands[width],
            "stay_rate": stay,
            "mean_chain_length": length,
            "fl_ev_raw": stats["mean"],
            "fl_ev_raw_se": stats["se"],
            "trunc_bias": trunc,
            "per_hand_oppfl": fl_vs_fl_hands[width] / fl_hands[width],
            # The reference's own stratum means, for comparison against a
            # simulator's: equal shares with unequal stratum means is engine
            # drift, and this is the half of that comparison the reference owns.
            "stratum_means": {
                label: mean_se(
                    [row["raw"] for row in rows if row["start"] == label]
                )
                for label in LABELS
            },
            "vs_normal_cell": {
                "hands": cell["n"], "stays": cell["stay"],
                "zero_end": cell["zero"], "gap_end": cell["gap"],
                "stay_rate_recorded": cell["stay"] / cell["n"] if cell["n"] else None,
                "stay_rate_zero_erasure": erasure,
            },
        }

    failed = [check for check in checks if not check["agree"]]
    if failed:
        for check in failed:
            print(
                f"MISMATCH w{check['width']} {check['quantity']}: "
                f"derived {check['derived']!r} vs flev1_final "
                f"{check['flev1_final']!r}",
                file=sys.stderr,
            )
        raise SystemExit(
            "chain reconstruction disagrees with flev1_final.json; the shares "
            "would be derived from a reconstruction that is not the reference's"
        )

    document = {
        "schema": "ofc_flsim_ref_shares/v1",
        "source": "flev1_hands.jsonl (gen-2 self-play, 60,000 hands, 2026-08-19, "
                  "own_lap4 both seats, table {5.28, 16.61, 38.76, 70.07})",
        "source_path": str(FLEV1_HANDS),
        "validated_against": str(FLEV1_FINAL),
        "hands_read": lines,
        "caveats": [
            "T3-street-disabled environment (hu_gen2_onpolicy_20260819.md §10): "
            "the qualification frequencies that set these shares were produced "
            "with that street off",
            "stay flags derived from the next hand's state "
            "(hu_gen2_onpolicy_20260819.md §8), so a stay killed by a "
            "zero-stack session end is recorded as a non-stay",
            "zero-stack ends also perturb the chain-start distribution itself "
            "at second order: they reset both seats to normal",
            "gen-2 composition.  When a gen-3 or union self-play exists, "
            "re-derive with this same mode and re-price from the stratum means; "
            "no resimulation is needed unless a width moves by more than 1.5",
        ],
        "reconstruction": (
            "analyze_selfplay_flev.py's: per (worker, seat), open on a "
            "Fantasyland hand and close on `not stay or end`; the chain's start "
            "label is the opponent's state on its first hand"
        ),
        "trunc_bias_formula": (
            "(zero_end_hands / hero_fl_hands) * stay_rate * mean_chain_length * "
            "fl_ev_raw -- the value a zero end removed from the reference's "
            "fl_ev, i.e. the amount by which an untruncated simulator reads high"
        ),
        "checks": checks,
        "widths": widths_out,
    }
    out = args.out or REF_SHARES
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(document, indent=2) + "\n")

    for width in WIDTHS:
        row = widths_out[str(width)]
        print(
            f"w{width}: chains={row['n_chains']} fl_ev_raw={row['fl_ev_raw']:.3f} "
            f"stay={row['stay_rate']:.4f} len={row['mean_chain_length']:.3f} "
            f"zero_end={row['zero_end_hands']}/{row['hero_fl_hands']} "
            f"trunc_bias={row['trunc_bias']:+.3f}"
        )
        print("   shares: " + " ".join(
            f"{label}={row['shares'][label]:.4f}" for label in LABELS))
    print(f"\n{len(checks)} checks against flev1_final.json, all agree")
    print(f"wrote {out}", flush=True)
    return 0


def recalibrate(args: argparse.Namespace) -> int:
    """Re-read the calibration runs under the stratified estimator.

    No simulation runs here.  The cal700 files already contain every chain's
    start state, so the composition question is answerable from them, and
    re-playing them would only add noise to a comparison whose whole value is
    that it is the same chains.

    What changes from v1 is the episodic gate.  Comparing the natural mean to
    the reference directly is comparing two different compositions and was only
    ever passing at width 16 because a composition deficit of -3.5 and an engine
    surplus of +3.5 happened to cancel.  The new gate stratifies first, then
    books the remainder as a ledger -- the reference's own zero-end truncation,
    which is arithmetic, plus engine drift, which is not -- and gates the drift.
    """
    ref_shares = args.ref_shares or REF_SHARES
    if not ref_shares.exists():
        raise SystemExit(
            f"missing {ref_shares}; run --extract-ref-shares first"
        )
    shares_doc = json.loads(ref_shares.read_text(encoding="utf-8"))
    final = json.loads(FLEV1_FINAL.read_text(encoding="utf-8"))
    v1_path = CAL700 / "calibration.json"
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v1_rows = {row["gate"]: row for row in v1["gates"]}

    runs = {width: CAL700 / f"run_cal_w{width}.jsonl" for width in (15, 16)}
    for path in runs.values():
        if not path.exists():
            raise SystemExit(f"missing calibration run {path}")
    reports = {
        width: analyze(path, ref_shares) for width, path in runs.items()
    }

    # C1 -- recomputed by the same function that produced v1, then compared
    # against what v1 recorded.  Only the four C1 rows are gates here; the two
    # C2 rows ride along as retired diagnostics and the two C3 rows are read
    # again below with their note.
    recomputed = v1_gates(reports)
    transcribed = []
    for row in recomputed:
        stored = v1_rows.get(row["gate"])
        entry = dict(row)
        entry["v1_observed"] = stored.get("observed") if stored else None
        entry["v1_pass"] = stored.get("pass") if stored else None
        entry["transcription_agrees"] = bool(
            stored is not None
            and stored.get("observed") is not None
            and row.get("observed") is not None
            and abs(stored["observed"] - row["observed"]) <= 1e-12
            and stored.get("pass") == row.get("pass")
        )
        if row["gate"].startswith("C2."):
            entry["status"] = (
                "retired: the natural mean and the reference carry different "
                "compositions, so their difference is not a gate.  Diagnostic "
                "only; C2' below is the gate"
            )
        transcribed.append(entry)

    # C2' -- stratified value against the reference, booked as a ledger.
    c2_prime = []
    for width in (15, 16):
        strat = reports[width]["fl_ev_strat"]
        row_ref = shares_doc["widths"][str(width)]
        ref_raw = row_ref["fl_ev_raw"]
        se_ref = final["fl_ev_table"][str(width)]["raw"]["se"]
        residual = strat["mean"] - ref_raw
        trunc = row_ref["trunc_bias"]
        drift = residual - trunc
        se_cmp = math.sqrt(
            strat["se_sim"] ** 2 + strat["se_share"] ** 2 + se_ref ** 2
        )
        allowed = 2.0 + 2.0 * se_cmp
        c2_prime.append({
            "gate": f"C2'.w{width}.fl_ev_strat",
            "stratified": strat["mean"],
            "se_sim": strat["se_sim"],
            "se_share": strat["se_share"],
            "reference_raw": ref_raw,
            "se_ref": se_ref,
            "residual": residual,
            "ledger": {
                "zero_end_truncation": trunc,
                "engine_drift": drift,
                "identity": "residual = zero_end_truncation + engine_drift",
            },
            "se_cmp": se_cmp,
            "allowed_abs_drift": allowed,
            "pass": abs(drift) <= allowed,
            "natural_diagnostic": {
                "mean": reports[width]["fl_ev_all"]["mean"],
                "se": reports[width]["fl_ev_all"]["se"],
                "minus_reference": reports[width]["fl_ev_all"]["mean"] - ref_raw,
                "note": "composition-confounded; not a gate",
            },
        })

    # C3 -- unchanged, with what the reference's stay rate is missing.
    c3 = []
    for width in (15, 16):
        row = next(r for r in transcribed if r["gate"] == f"C3.w{width}.stay_rate")
        cell = shares_doc["widths"][str(width)]["vs_normal_cell"]
        erasure = cell["stay_rate_zero_erasure"]
        entry = dict(row)
        entry["reference_zero_erasure_pt"] = erasure * 100.0 if erasure else None
        entry["note"] = (
            "the reference's stay rate is recorded low: a zero-stack session "
            "end clears the stay flag, and in the hero-vs-normal cell that is "
            f"{erasure * 100.0:.2f} points at width {width} "
            "(addendum §C3(2) quotes 0.89 / 0.09 for widths 16 / 15).  The "
            "remaining excess is engine drift, not composition: it is present "
            "in equal size on hands where the opponent fouled and the hero "
            "therefore placed blind"
        ) if erasure is not None else "no vs-normal cell"
        c3.append(entry)

    # The sufficiency diagnostic: if conditioning on the start state were not
    # enough -- if the opponent's process inside a chain differed -- the
    # reweighted per-hand mix would not land on the reference's.
    diagnostics = []
    for width in (15, 16):
        observed = reports[width]["per_hand_oppfl_reweighted"]
        expected = shares_doc["widths"][str(width)]["per_hand_oppfl"]
        hands = reports[width]["counts"]["hands"]
        diagnostics.append({
            "gate": f"D.w{width}.per_hand_oppfl_reweighted",
            "observed": observed,
            "reference": expected,
            "delta": observed - expected,
            "allowed_abs_delta": 0.01,
            "pass": abs(observed - expected) <= 0.01,
            "natural": 1.0 - reports[width]["opp_normal_hands"] / hands,
            "note": (
                "the unweighted rate is the simulator's own mix and is expected "
                "to differ; what has to agree is the reweighted one"
            ),
        })

    gates = (
        [row for row in transcribed if row["gate"].startswith(("C1", "C3"))]
        + c2_prime
        + diagnostics
    )
    passed = (
        all(row.get("pass") for row in gates)
        and all(row["transcription_agrees"] for row in transcribed)
    )

    out = {
        "schema": "ofc_flsim_calibration/v2",
        "verdict": "PASS" if passed else "FAIL",
        "note": (
            "no simulation was run: this is the cal700 chains re-read under the "
            "stratified estimator"
        ),
        "runs": {str(width): str(path) for width, path in runs.items()},
        "v1_source": str(v1_path),
        "ref_shares": str(ref_shares),
        "reference": str(FLEV1_FINAL),
        "table": v1["table"],
        "seed": v1["seed"],
        "blocks": v1["blocks"],
        "chains_per_block": v1["chains_per_block"],
        "exe_sha256_expected": anchor_sha256(),
        "c1_transcribed": transcribed,
        "c2_prime": c2_prime,
        "c3": c3,
        "diagnostics": diagnostics,
        "fl_ev_strat": {
            str(width): reports[width]["fl_ev_strat"] for width in (15, 16)
        },
        "per_width": {
            str(width): {
                "fl_ev_all": reports[width]["fl_ev_all"],
                "fl_ev_solo": reports[width]["fl_ev_solo"],
                "per_hand_oppfl_reweighted":
                    reports[width]["per_hand_oppfl_reweighted"],
                "stay_rate": reports[width]["stay_rate"],
                "mean_chain_length": reports[width]["mean_chain_length"],
                "opp_entry_rate": reports[width]["opp_entry_rate"],
                "opp_fl_at_start_rate": reports[width]["opp_fl_at_start_rate"],
                "counts": reports[width]["counts"],
                "fl_ev_config_sha256": reports[width]["run"]["fl_ev_config_sha256"],
            }
            for width in (15, 16)
        },
        "logical_cores": os.cpu_count(),
    }
    path = args.out or (CAL700 / "calibration_v2.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(out, indent=2) + "\n")

    print(f"verdict: {out['verdict']}")
    print("\nC1 (transcribed from v1, recomputed here):")
    for row in transcribed:
        mark = "ok" if row["transcription_agrees"] else "TRANSCRIPTION MISMATCH"
        print(f"  {row['gate']:28s} {row['observed']:+.6f} "
              f"pass={row['pass']}  [{mark}]")
    print("\nC2' (stratified, ledgered):")
    for row in c2_prime:
        ledger = row["ledger"]
        print(
            f"  w{row['gate'][5:7]}: strat {row['stratified']:.3f} "
            f"(se_sim {row['se_sim']:.3f}, se_share {row['se_share']:.3f}) "
            f"vs ref {row['reference_raw']:.3f} -> residual "
            f"{row['residual']:+.3f} = trunc {ledger['zero_end_truncation']:+.3f} "
            f"+ drift {ledger['engine_drift']:+.3f} "
            f"(allowed {row['allowed_abs_drift']:.3f}, pass={row['pass']})"
        )
        print(f"        natural {row['natural_diagnostic']['mean']:.3f} "
              f"({row['natural_diagnostic']['minus_reference']:+.3f} vs ref, "
              f"composition-confounded)")
    print("\nD (reweighted per-hand opponent-FL rate):")
    for row in diagnostics:
        print(f"  {row['gate']:34s} {row['observed']:.4f} vs "
              f"{row['reference']:.4f} ({row['delta']:+.4f}) pass={row['pass']}")
    print(f"\nwrote {path}", flush=True)
    return 0 if passed else 1


def parse_blocks_per_width(text: str) -> dict[int, int]:
    table: dict[int, int] = {}
    for item in text.split(","):
        width, _, blocks = item.strip().partition(":")
        table[int(width)] = int(blocks)
    return table


def require_ref_shares(args: argparse.Namespace) -> Path:
    path = args.ref_shares or REF_SHARES
    if not path.exists():
        raise SystemExit(f"missing {path}; run --extract-ref-shares first")
    return path


def shares_for(ref_shares: Path, width: int) -> dict[str, float]:
    document = json.loads(ref_shares.read_text(encoding="utf-8"))
    return document["widths"][str(width)]["shares"]


def lap_row(report: dict, table_in: float) -> dict:
    """Everything about one width's lap that the next lap or a reader needs."""
    strat = report["fl_ev_strat"]
    natural = report["fl_ev_all"]
    return {
        "table_in": table_in,
        "strat": {
            "mean": strat["mean"],
            "se_sim": strat["se_sim"],
            "se_share": strat["se_share"],
            "strata": strat["strata"],
            "sensitivity": strat["sensitivity"],
            "sim_shares": strat["sim_shares"],
        },
        "natural": natural,
        "solo": report["fl_ev_solo"],
        "delta_signed": strat["mean"] - table_in,
        "stay_rate": report["stay_rate"],
        "mean_chain_length": report["mean_chain_length"],
        "opp_entry_rate": report["opp_entry_rate"],
        "opp_fl_at_start_rate": report["opp_fl_at_start_rate"],
        "per_hand_oppfl_reweighted": report["per_hand_oppfl_reweighted"],
        "geometric_check": report["geometric_check"],
        "wall_seconds": report["wall_seconds"],
        "fl_ev_config_sha256": report["run"]["fl_ev_config_sha256"],
    }


def extrapolate(
    last: float, previous_delta: float, last_delta: float, paired_se: float | None
) -> dict:
    """Continue a geometric increment, but only when there is one to continue.

    Two guards, both learned the expensive way.  The increments have to be
    larger than the noise that measured them -- extrapolating a ratio of two
    numbers that are both zero plus noise invents a limit out of nothing -- and
    the ratio has to look like contraction rather than either a stall or an
    oscillation.  Outside those, the last measurement is the answer and the
    fixed point is simply not resolved yet.
    """
    floor = max(3.0 * paired_se, 0.5) if paired_se else 0.5
    if abs(previous_delta) <= floor or abs(last_delta) <= floor:
        return {"applied": False, "reason": f"increment below floor {floor:.3f}",
                "floor": floor, "target": last}
    ratio = last_delta / previous_delta
    if not 0.2 <= ratio <= 0.85:
        return {"applied": False, "reason": f"ratio {ratio:.3f} outside [0.2, 0.85]",
                "ratio": ratio, "floor": floor, "target": last}
    return {
        "applied": True,
        "ratio": ratio,
        "floor": floor,
        "remaining": last_delta * ratio / (1.0 - ratio),
        "target": last + last_delta * ratio / (1.0 - ratio),
    }


def fixed_point(args: argparse.Namespace) -> int:
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    ref_shares = require_ref_shares(args)
    widths = [int(w) for w in args.widths.split(",")]
    blocks = parse_blocks_per_width(args.blocks_per_width)
    for width in widths:
        if width not in blocks:
            raise SystemExit(f"--blocks-per-width has no entry for width {width}")
    tol = {width: (1.0 if width == 17 else args.tol) for width in widths}
    pi = {width: shares_for(ref_shares, width) for width in widths}

    snapshot = snapshot_provenance(workdir)

    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    table = {width: float(base["fl_ev"][str(width)]) for width in WIDTHS}
    provenance = {
        "start_table": {str(w): table[w] for w in WIDTHS},
        "start_source": base.get("source"),
        "start_config": str(BASE_CONFIG),
        "ref_shares": str(ref_shares),
        "snapshot": snapshot,
    }
    print(f"start table: {provenance['start_table']}", flush=True)

    iterations: list[dict] = []
    series: dict[int, dict[int, dict]] = {}
    converged = False
    for step in range(args.max_iters):
        measured = {}
        rows = {}
        for width in widths:
            tag = f"fp{step}_w{width}"
            report = run_sim(
                width, blocks[width], CAL_K, args.seed, table, workdir, tag,
                ref_shares,
            )
            measured[width] = report
            rows[width] = lap_row(report, table[width])
            # Common random numbers: difference this lap against the previous
            # one block by block, so the move is read against paired noise
            # rather than against the headline standard error.
            current = per_block_series(workdir / f"run_{tag}.jsonl", pi[width])
            if width in series:
                rows[width]["paired_vs_previous"] = {
                    "natural": paired_delta(series[width], current, "natural"),
                    "strat": paired_delta(series[width], current, "strat"),
                }
            series[width] = current

        # The table takes the stratified value: its consumers price entries,
        # and an entry's opponent comes from the entry composition (§B1).
        nxt = dict(table)
        deltas = {}
        for width in widths:
            nxt[width] = rows[width]["strat"]["mean"]
            deltas[width] = abs(nxt[width] - table[width])
        iterations.append({
            "iteration": step,
            "table_in": {str(w): table[w] for w in WIDTHS},
            "measured": {str(w): rows[w] for w in widths},
            "table_out": {str(w): nxt[w] for w in WIDTHS},
            "delta": {str(w): deltas[w] for w in widths},
            "delta_signed": {str(w): rows[w]["delta_signed"] for w in widths},
            "tolerance": {str(w): tol[w] for w in widths},
        })
        print(
            f"iter {step}: "
            + ", ".join(f"w{w} {table[w]:.2f}->{nxt[w]:.2f} (d={deltas[w]:.2f})"
                        for w in widths),
            flush=True,
        )
        table = nxt
        if all(deltas[w] <= tol[w] for w in widths):
            converged = True
            break

    # Geometric extrapolation, guarded (§C3(3)): the increment has to clear the
    # paired noise floor at both ends and the ratio has to look like contraction.
    extrapolated = {}
    target_table = dict(table)
    for width in widths:
        if len(iterations) >= 2:
            last = iterations[-1]["delta_signed"][str(width)]
            previous = iterations[-2]["delta_signed"][str(width)]
            paired = (
                iterations[-1]["measured"][str(width)]
                .get("paired_vs_previous", {})
                .get("strat", {})
                .get("se")
            )
            verdict = extrapolate(table[width], previous, last, paired)
        else:
            verdict = {"applied": False, "reason": "fewer than two laps",
                       "target": table[width]}
        extrapolated[str(width)] = verdict
        target_table[width] = verdict["target"]

    unstable = [
        width for width in widths
        if len(iterations) >= 2
        and iterations[-1]["delta_signed"][str(width)]
        * iterations[-2]["delta_signed"][str(width)] < 0
    ]

    precision = {
        str(width): {
            "chain_sd_reference": sd_of(width),
            "chains_run": (
                iterations[-1]["measured"][str(width)]["natural"]["n"]
                if iterations else 0
            ),
            "se_sim_achieved": (
                iterations[-1]["measured"][str(width)]["strat"]["se_sim"]
                if iterations else None
            ),
            "se_share_floor": (
                iterations[-1]["measured"][str(width)]["strat"]["se_share"]
                if iterations else None
            ),
            "chains_for_se_0_12": (sd_of(width) / 0.12) ** 2,
            "chains_for_se_1_0": (sd_of(width) / 1.0) ** 2,
            "note": (
                "n = (sd/target_se)^2; sd from the reference run's se*sqrt(n).  "
                "se_share is capped by the reference's chain count and no "
                "number of blocks buys it down"
            ),
        }
        for width in widths
    }

    out = {
        "schema": "ofc_flsim_fixedpoint/v2",
        "converged": converged,
        "driven_by": "fl_ev_strat.mean (entry composition, addendum §B4)",
        "seed": args.seed,
        "common_random_numbers": True,
        "widths": widths,
        "blocks_per_width": {str(w): blocks[w] for w in widths},
        "chains_per_block": CAL_K,
        "provenance": provenance,
        "final_table": {str(w): table[w] for w in WIDTHS},
        "target_table": {str(w): target_table.get(w, table[w]) for w in WIDTHS},
        "iterations": iterations,
        "geometric_extrapolation": extrapolated,
        "oscillating_widths": unstable,
        "precision": precision,
        "logical_cores": os.cpu_count(),
        "notes": {
            "adoption": (
                "this driver does not write ai/config/fl_ev.json; adopting a "
                "table is a generation boundary and the owner's call.  Because "
                "the engine that produced these numbers is untracked, the "
                "proposal is to adopt the table and commit the engine together"
            ),
            "stop_rule": (
                "if a width oscillates in sign, or its ratio leaves (0.2, 0.85) "
                "after the third lap, stop and report rather than spending "
                "another night"
            ),
            "width_17": (
                "raw chain totals on three scales.  This file's number is the "
                "simulator's: an untruncated chain value.  In game a zero-stack "
                "session end cuts about 8 points of it (reference arithmetic, "
                "and the hazard is not i.i.d. -- truncation is caused by the "
                "chain being large), and the 200-point stack floor a further "
                "2.45 that the reference measured as paid minus raw"
            ),
        },
    }
    path = workdir / "fixed_point.json"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "converged": converged,
        "final_table": out["final_table"],
        "target_table": out["target_table"],
        "geometric_extrapolation": extrapolated,
        "oscillating_widths": unstable,
    }, indent=2))
    print(f"\nwrote {path}", flush=True)
    return 0


def confirm(args: argparse.Namespace) -> int:
    """One lap at the extrapolated table: does it reproduce itself?

    A fixed point that was extrapolated to rather than reached is a prediction,
    and this is the only place it is tested.  The gate is per width and is the
    looser of twice the simulation error and the lap tolerance -- tight enough
    that a table two points off fails, loose enough that it is not testing the
    noise.
    """
    workdir = args.workdir
    ref_shares = require_ref_shares(args)
    source = workdir / "fixed_point.json"
    if not source.exists():
        raise SystemExit(f"missing {source}; run the iteration laps first")
    fixed = json.loads(source.read_text(encoding="utf-8"))
    target = {int(w): float(v) for w, v in fixed["target_table"].items()}
    widths = [int(w) for w in fixed["widths"]]
    blocks = (
        parse_blocks_per_width(args.blocks_per_width)
        if args.blocks_per_width_given else CONFIRM_BLOCKS
    )
    for width in widths:
        if width not in blocks:
            raise SystemExit(
                f"--blocks-per-width has no entry for width {width}, which the "
                f"iteration laps measured"
            )
    snapshot = snapshot_provenance(workdir)
    print(f"confirming table {fixed['target_table']}", flush=True)

    rows = {}
    gates = []
    for width in widths:
        report = run_sim(
            width, blocks[width], CAL_K, args.seed, target, workdir,
            f"confirm_w{width}", ref_shares,
        )
        rows[width] = lap_row(report, target[width])
        mean, se_sim = headline(report)
        allowed = max(2.0 * se_sim, CONFIRM_TOL.get(width, args.tol))
        gates.append({
            "gate": f"confirm.w{width}",
            "target": target[width],
            "observed": mean,
            "se_sim": se_sim,
            "se_share": report["fl_ev_strat"]["se_share"],
            "delta": mean - target[width],
            "allowed_abs_delta": allowed,
            "pass": abs(mean - target[width]) <= allowed,
            "natural": report["fl_ev_all"]["mean"],
        })

    passed = all(gate["pass"] for gate in gates)
    out = {
        "schema": "ofc_flsim_confirm/v1",
        "verdict": "PASS" if passed else "FAIL",
        "table_used": {str(w): target[w] for w in WIDTHS},
        "fixed_point_source": str(source),
        "seed": args.seed,
        "blocks_per_width": {str(w): blocks[w] for w in widths},
        "chains_per_block": CAL_K,
        "gates": gates,
        "measured": {str(w): rows[w] for w in widths},
        "provenance": snapshot,
        "logical_cores": os.cpu_count(),
    }
    path = workdir / "confirm.json"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"verdict": out["verdict"], "gates": gates}, indent=2))
    print(f"\nwrote {path}", flush=True)
    return 0 if passed else 1


def precision(args: argparse.Namespace) -> int:
    """One large single-width run at the settled table.

    Separate seed: this run is not part of the common-random-numbers column and
    pooling it with one would correlate a precision estimate with the laps that
    chose the table.
    """
    workdir = args.workdir
    ref_shares = require_ref_shares(args)
    width = args.precision_width
    confirmed = workdir / "confirm.json"
    fixed = workdir / "fixed_point.json"
    if confirmed.exists():
        document = json.loads(confirmed.read_text(encoding="utf-8"))
        table_source, table_raw = str(confirmed), document["table_used"]
    elif fixed.exists():
        document = json.loads(fixed.read_text(encoding="utf-8"))
        table_source, table_raw = str(fixed), document["target_table"]
    else:
        raise SystemExit(
            f"neither {confirmed} nor {fixed} exists; there is no settled table "
            f"to measure at"
        )
    table = {int(w): float(v) for w, v in table_raw.items()}
    print(f"precision width {width} at {table_raw} (from {table_source})", flush=True)

    report = run_sim(
        width, args.precision_blocks, CAL_K, args.precision_seed, table, workdir,
        f"precision_w{width}", ref_shares,
    )
    strat = report["fl_ev_strat"]
    combined = math.sqrt(strat["se_sim"] ** 2 + strat["se_share"] ** 2)
    out = {
        "schema": "ofc_flsim_precision/v1",
        "width": width,
        "table_used": table_raw,
        "table_source": table_source,
        "seed": args.precision_seed,
        "blocks": args.precision_blocks,
        "chains_per_block": CAL_K,
        "fl_ev_strat": strat,
        "fl_ev_all": report["fl_ev_all"],
        "fl_ev_solo": report["fl_ev_solo"],
        "combined_se": combined,
        "budget": {
            "se_sim": strat["se_sim"],
            "se_share": strat["se_share"],
            "combined": combined,
            "note": (
                "se_share is the reference composition's own sampling error and "
                "is not buyable here; it shrinks only when a newer self-play "
                "supplies the shares"
            ),
        },
        "measured": lap_row(report, table[width]),
        "logical_cores": os.cpu_count(),
    }
    path = workdir / f"precision_w{width}.json"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "width": width,
        "fl_ev_strat": {"mean": strat["mean"], "se_sim": strat["se_sim"],
                        "se_share": strat["se_share"], "combined": combined},
        "fl_ev_all": report["fl_ev_all"],
    }, indent=2))
    print(f"\nwrote {path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-ref-shares", action="store_true",
                        help="derive the entry composition from "
                             "flev1_hands.jsonl and write ref_shares_flev1.json "
                             "(no simulation)")
    parser.add_argument("--recalibrate", action="store_true",
                        help="re-read the cal700 runs under the stratified "
                             "estimator and write calibration_v2.json (no "
                             "simulation)")
    parser.add_argument("--calibrate", action="store_true",
                        help="play the calibration runs and gate them against "
                             "the reference data")
    parser.add_argument("--confirm", action="store_true",
                        help="one lap at the fixed point's extrapolated table")
    parser.add_argument("--precision-width", type=int,
                        help="one large single-width run at the settled table")
    parser.add_argument("--precision-blocks", type=int, default=11000)
    parser.add_argument("--precision-seed", type=int, default=89200001)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--ref-shares", type=Path,
                        help=f"default {REF_SHARES}")
    parser.add_argument("--out", type=Path,
                        help="--extract-ref-shares / --recalibrate: override "
                             "the output path")
    parser.add_argument("--widths", default="14,15,16,17")
    parser.add_argument("--blocks-per-width", default="14:2500,15:2500,16:2500,17:700")
    parser.add_argument("--seed", type=int, default=89000001)
    parser.add_argument("--tol", type=float, default=0.3,
                        help="convergence tolerance; width 17 always uses 1.0")
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument("--blocks", type=int, default=CAL_BLOCKS,
                        help="--calibrate only: blocks per width (default 3000)")
    args = parser.parse_args()
    args.blocks_per_width_given = any(
        item.startswith("--blocks-per-width") for item in sys.argv[1:]
    )

    modes = [
        args.extract_ref_shares, args.recalibrate, args.calibrate, args.confirm,
        args.precision_width is not None,
    ]
    if sum(bool(mode) for mode in modes) > 1:
        raise SystemExit("pick one mode")

    # The two analysis modes touch no binary and no models: they re-read files
    # that already exist.
    if args.extract_ref_shares:
        return extract_ref_shares(args)
    if args.recalibrate:
        return recalibrate(args)

    if not BINARY.exists():
        raise SystemExit(f"binary not built: {BINARY}")
    for name in ("t0.bin", "t1.bin", "t2.bin"):
        if not (OWN / name).exists():
            raise SystemExit(f"missing chooser: {OWN / name}")
    if args.workdir is None:
        raise SystemExit("--workdir is required for the simulating modes")

    if args.calibrate:
        return calibrate(args)
    if args.confirm:
        return confirm(args)
    if args.precision_width is not None:
        return precision(args)
    return fixed_point(args)


if __name__ == "__main__":
    raise SystemExit(main())
