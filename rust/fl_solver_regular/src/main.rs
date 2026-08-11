//! CLI for the regular-rule Fantasyland solver and the T4-vs-FL teacher.

use fl_solver_regular::behavior::{generate_root, BehaviorConfig, HeroT4Root};
use fl_solver_regular::cards::{cards_to_tokens, parse_cards};
use fl_solver_regular::distribution::{
    sample_deal, DistributionStats, CATEGORY_LABELS, ROYALTY_BIN_EDGES,
};
use fl_solver_regular::objective::{
    find_default_fl_ev_config, load_fl_ev, ObjectiveConfig, ObjectiveKind, ReferenceSet,
};
use fl_solver_regular::solver::{solve_brute_force, FlSolver};
use fl_solver_regular::teacher::{
    label_root, label_t3_root, ExternalRoot, LabelledT3Root, TeacherConfig,
};
use rayon::prelude::*;
use serde_json::json;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// Measured hero (normal player) foul rate against a Fantasyland opponent.
/// Source is recorded alongside the value wherever it is used.
///
/// This deliberately still cites the v3 config even though v4 is now the
/// reader default. What v4 superseded is the Fantasyland EV; it did not
/// re-measure THIS quantity, which is a normal player's foul rate CONDITIONED
/// on facing a Fantasyland opponent. v4's own 0.2176 is an unconditional
/// self-play bust rate and is not the same number. Repointing the citation
/// without a re-measurement would attach a measured value to a source that
/// does not contain it, so the citation stays where the number actually lives.
const HERO_FOUL_PROB_PROVENANCE: &str =
    "configs/fl_ev_regular_v3_direct2.json provenance.opponent_bust_rate=0.2482 \
     (armA direct HU fixed point, 10000 hands, 2026-08-03): the non-FL side's \
     foul rate when the other seat is in Fantasyland.";
const HERO_FOUL_PROB: f64 = 0.2482;

struct CommonArgs {
    fl_ev_config: Option<PathBuf>,
    fl_ev_cards: u8,
    objective_kind: ObjectiveKind,
    reference_path: Option<PathBuf>,
    reference_samples: usize,
    reference_seed: u64,
    hero_foul_prob: f64,
    threads: usize,
}

impl Default for CommonArgs {
    fn default() -> Self {
        Self {
            fl_ev_config: None,
            fl_ev_cards: 14,
            objective_kind: ObjectiveKind::PureV1,
            reference_path: None,
            reference_samples: 64,
            reference_seed: 997_000_900,
            hero_foul_prob: HERO_FOUL_PROB,
            threads: 6,
        }
    }
}

impl CommonArgs {
    fn build_objective(&self) -> Result<ObjectiveConfig, String> {
        let path = match &self.fl_ev_config {
            Some(path) => path.clone(),
            None => {
                let here = std::env::current_dir()
                    .map_err(|error| format!("cannot read the working directory: {error}"))?;
                find_default_fl_ev_config(&here).ok_or_else(|| {
                    "could not find configs/fl_ev_regular_v4_selfplay.json; pass --fl-ev-config"
                        .to_owned()
                })?
            }
        };
        let fl_ev = load_fl_ev(&path, self.fl_ev_cards)?;
        let reference = match self.objective_kind {
            ObjectiveKind::PureV1 => None,
            ObjectiveKind::LineEquityV1 => Some(match &self.reference_path {
                Some(path) => {
                    let text = std::fs::read_to_string(path).map_err(|error| {
                        format!("failed to read reference {}: {error}", path.display())
                    })?;
                    serde_json::from_str::<ReferenceSet>(&text).map_err(|error| {
                        format!("invalid reference {}: {error}", path.display())
                    })?
                }
                None => {
                    ReferenceSet::uniform_random_legal(self.reference_samples, self.reference_seed)
                }
            }),
        };
        let config = ObjectiveConfig {
            kind: self.objective_kind.clone(),
            fl_ev_stay: fl_ev.value,
            fl_ev_config_path: fl_ev.path,
            fl_ev_cards: fl_ev.cards,
            hero_foul_prob: self.hero_foul_prob,
            hero_foul_prob_provenance: HERO_FOUL_PROB_PROVENANCE.to_owned(),
            reference,
        };
        config.validate()?;
        Ok(config)
    }

    fn install_thread_pool(&self) {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .build_global();
    }
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.is_empty() {
        print_help();
        std::process::exit(2);
    }
    let command = arguments[0].clone();
    let rest = &arguments[1..];
    let result = match command.as_str() {
        "solve" => command_solve(rest),
        "verify-exact" => command_verify_exact(rest),
        "bench" => command_bench(rest),
        "dist" => command_dist(rest),
        "build-reference" => command_build_reference(rest),
        "build-library" => command_build_library(rest),
        "label" => command_label(rest),
        "label-t3" => command_label_t3(rest),
        "label-t2" => command_label_t2(rest),
        "label-t1" => command_label_t1(rest),
        "label-t0" => command_label_t0(rest),
        "probe-t1-grid" => command_probe_t1_grid(rest),
        "probe" => command_probe(rest),
        "probe-t3" => command_probe_t3(rest),
        "probe-t3-n" => command_probe_t3_n(rest),
        "frontier-stats" => command_frontier_stats(rest),
        "vfl-parity" => command_vfl_parity(rest),
        "--help" | "-h" | "help" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown command: {other}")),
    };
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

/// Hold the Rust forward pass to the Python model that was measured.
///
/// A continuation policy that disagrees with the model whose regret was
/// reported is a different teacher than the one the plan claims. The fixture
/// carries real feature rows and the scores the training-time module produced
/// for them; this reloads the pinned image and compares.
fn command_vfl_parity(arguments: &[String]) -> Result<(), String> {
    let mut model_path: Option<String> = None;
    let mut fixture_path: Option<String> = None;
    let mut tolerance = 1e-5f64;
    let mut index = 0usize;
    while index < arguments.len() {
        let value = |i: usize| -> Result<String, String> {
            arguments
                .get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", arguments[i]))
        };
        match arguments[index].as_str() {
            "--model" => {
                model_path = Some(value(index)?);
                index += 1;
            }
            "--fixture" => {
                fixture_path = Some(value(index)?);
                index += 1;
            }
            "--tolerance" => {
                tolerance = value(index)?
                    .parse()
                    .map_err(|error| format!("--tolerance: {error}"))?;
                index += 1;
            }
            other => return Err(format!("unknown option {other}")),
        }
        index += 1;
    }
    let model_path = model_path.ok_or("--model is required")?;
    let fixture_path = fixture_path.ok_or("--fixture is required")?;

    let bytes = std::fs::read(&model_path).map_err(|error| format!("{model_path}: {error}"))?;
    let fixture_text =
        std::fs::read_to_string(&fixture_path).map_err(|error| format!("{fixture_path}: {error}"))?;
    let fixture: serde_json::Value =
        serde_json::from_str(&fixture_text).map_err(|error| error.to_string())?;

    let pinned = fixture["model_sha256"]
        .as_str()
        .ok_or("fixture has no model_sha256")?;
    let model = fl_solver_regular::vfl_model::VflModel::load_pinned(&bytes, pinned)?;

    let rows = fixture["rows"].as_array().ok_or("fixture has no rows")?;
    let scores = fixture["scores"].as_array().ok_or("fixture has no scores")?;
    if rows.len() != scores.len() {
        return Err(format!(
            "fixture has {} rows and {} scores",
            rows.len(),
            scores.len()
        ));
    }

    let mut scratch = model.scratch();
    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    for (position, (row, expected)) in rows.iter().zip(scores.iter()).enumerate() {
        let features: Vec<f32> = row
            .as_array()
            .ok_or("a fixture row is not an array")?
            .iter()
            .map(|value| value.as_f64().unwrap_or(f64::NAN) as f32)
            .collect();
        let got = model.predict(&features, &mut scratch)? as f64;
        let want = expected.as_f64().ok_or("a fixture score is not a number")?;
        let delta = (got - want).abs();
        if delta > worst {
            worst = delta;
            worst_at = position;
        }
    }

    println!(
        "{}",
        serde_json::json!({
            "model": model_path,
            "fixture": fixture_path,
            "sha256": pinned,
            "input_dim": model.input_dim(),
            "clamp": model.clamp(),
            "rows": rows.len(),
            "max_abs_delta": worst,
            "worst_row": worst_at,
            "tolerance": tolerance,
            "passed": worst <= tolerance,
        })
    );
    if worst > tolerance {
        return Err(format!(
            "parity failed: max |delta| {worst} over tolerance {tolerance}"
        ));
    }
    Ok(())
}

fn print_help() {
    println!("fl_solver_regular <command> [options]");
    println!();
    println!("commands:");
    println!("  solve            solve one 14-card Fantasyland deal");
    println!("  verify-exact     compare the pruned search against brute force");
    println!("  bench            time the solver over random deals");
    println!("  dist             opponent-Fantasyland distribution statistics");
    println!("  build-reference  write a reference hero-board set to JSON");
    println!("  label            generate and label T4-vs-FL roots to JSONL");
    println!("  probe            opponent-sample convergence probe");
    println!();
    println!("common options:");
    println!("  --fl-ev-config PATH   FL EV config (default: search upward for");
    println!("                        configs/fl_ev_regular_v4_selfplay.json)");
    println!("  --fl-ev-cards N       FL EV key to read (default 14)");
    println!("  --objective NAME      pure_v1 | line_equity_v1 (default pure_v1)");
    println!("  --reference PATH      reference set for line_equity_v1");
    println!("  --reference-samples N synthesise this many reference boards");
    println!("  --reference-seed N    seed for the synthesised reference");
    println!("  --hero-foul-prob X    scales the line term (default 0.2482)");
    println!("  --threads N           worker threads (default 6)");
}

fn parse_common(arguments: &[String]) -> Result<(CommonArgs, HashMap<String, String>), String> {
    let mut common = CommonArgs::default();
    let mut extra = HashMap::new();
    let mut index = 0;
    while index < arguments.len() {
        let key = arguments[index].clone();
        let value = arguments.get(index + 1).cloned();
        let need = |value: Option<String>| -> Result<String, String> {
            value.ok_or_else(|| format!("missing value for {key}"))
        };
        match key.as_str() {
            "--fl-ev-config" => common.fl_ev_config = Some(PathBuf::from(need(value)?)),
            "--fl-ev-cards" => {
                common.fl_ev_cards = need(value)?.parse().map_err(|_| "bad --fl-ev-cards")?
            }
            "--objective" => {
                common.objective_kind = match need(value)?.as_str() {
                    "pure_v1" => ObjectiveKind::PureV1,
                    "line_equity_v1" => ObjectiveKind::LineEquityV1,
                    other => return Err(format!("unknown objective: {other}")),
                }
            }
            "--reference" => common.reference_path = Some(PathBuf::from(need(value)?)),
            "--reference-samples" => {
                common.reference_samples =
                    need(value)?.parse().map_err(|_| "bad --reference-samples")?
            }
            "--reference-seed" => {
                common.reference_seed = need(value)?.parse().map_err(|_| "bad --reference-seed")?
            }
            "--hero-foul-prob" => {
                common.hero_foul_prob = need(value)?.parse().map_err(|_| "bad --hero-foul-prob")?
            }
            "--threads" => common.threads = need(value)?.parse().map_err(|_| "bad --threads")?,
            other if other.starts_with("--") => {
                extra.insert(other.trim_start_matches("--").to_owned(), need(value)?);
            }
            other => return Err(format!("unexpected argument: {other}")),
        }
        index += 2;
    }
    Ok((common, extra))
}

fn extra_usize(extra: &HashMap<String, String>, key: &str, fallback: usize) -> Result<usize, String> {
    match extra.get(key) {
        Some(value) => value.parse().map_err(|_| format!("bad --{key}")),
        None => Ok(fallback),
    }
}

fn extra_u64(extra: &HashMap<String, String>, key: &str, fallback: u64) -> Result<u64, String> {
    match extra.get(key) {
        Some(value) => value.parse().map_err(|_| format!("bad --{key}")),
        None => Ok(fallback),
    }
}

// ---------------------------------------------------------------------------

fn command_solve(arguments: &[String]) -> Result<(), String> {
    let (common, extra) = parse_common(arguments)?;
    let objective = common.build_objective()?;
    let hand_text = extra
        .get("hand")
        .ok_or_else(|| "solve requires --hand with 14 cards".to_owned())?;
    let cards = parse_cards(hand_text)?;
    let hand: [u8; 14] = cards
        .try_into()
        .map_err(|_| "solve requires exactly 14 cards".to_owned())?;
    let mut solver = FlSolver::new(&objective)?;
    let started = Instant::now();
    let solution = solver
        .solve(&hand)
        .ok_or_else(|| "no legal placement".to_owned())?;
    let elapsed = started.elapsed();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "objective": objective.identity(),
            "fl_ev_config_path": objective.fl_ev_config_path,
            "fl_ev_value": objective.fl_ev_stay,
            "hand": cards_to_tokens(&hand),
            "top": cards_to_tokens(&solution.top),
            "middle": cards_to_tokens(&solution.middle),
            "bottom": cards_to_tokens(&solution.bottom),
            "discard": fl_solver_regular::cards::card_to_token(solution.discard),
            "royalty": {
                "top": solution.top_royalty,
                "middle": solution.middle_royalty,
                "bottom": solution.bottom_royalty,
                "total": solution.total_royalty,
            },
            "stay": solution.stay,
            "line_equity": solution.line_equity,
            "score": solution.score,
            "leaves_visited": solver.last_leaves,
            "leaves_total": solver.last_space,
            "micros": elapsed.as_micros() as u64,
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn command_verify_exact(arguments: &[String]) -> Result<(), String> {
    let (common, extra) = parse_common(arguments)?;
    common.install_thread_pool();
    let objective = common.build_objective()?;
    let deals = extra_usize(&extra, "deals", 500)?;
    let seed = extra_u64(&extra, "seed", 997_000_100)?;

    let started = Instant::now();
    let outcomes: Vec<(bool, bool, f64)> = (0..deals as u64)
        .into_par_iter()
        .map(|index| {
            let hand = sample_deal(0, seed, index).expect("fresh deck has 52 cards");
            let mut solver = FlSolver::new(&objective).expect("valid objective");
            let fast = solver.solve(&hand).expect("a legal placement exists");
            let slow = solve_brute_force(&hand, &objective).expect("a legal placement exists");
            let identical = fast == slow;
            let same_score = fast.score.to_bits() == slow.score.to_bits();
            (identical, same_score, (fast.score - slow.score).abs())
        })
        .collect();
    let elapsed = started.elapsed();

    let identical = outcomes.iter().filter(|outcome| outcome.0).count();
    let same_score = outcomes.iter().filter(|outcome| outcome.1).count();
    let worst = outcomes
        .iter()
        .map(|outcome| outcome.2)
        .fold(0.0_f64, f64::max);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "objective": objective.identity(),
            "deals": deals,
            "seed_base": seed,
            "identical_argmax": identical,
            "bit_identical_score": same_score,
            "max_score_difference": worst,
            "exact": identical == deals && same_score == deals,
            "seconds": elapsed.as_secs_f64(),
        }))
        .map_err(|error| error.to_string())?
    );
    if identical != deals || same_score != deals {
        return Err(format!(
            "exactness failed: {identical}/{deals} identical arrangements, \
             {same_score}/{deals} bit-identical scores"
        ));
    }
    Ok(())
}

fn command_bench(arguments: &[String]) -> Result<(), String> {
    let (common, extra) = parse_common(arguments)?;
    let objective = common.build_objective()?;
    let deals = extra_usize(&extra, "deals", 2_000)?;
    let seed = extra_u64(&extra, "seed", 997_000_200)?;
    let mut solver = FlSolver::new(&objective)?;
    let mut timings = Vec::with_capacity(deals);
    let mut leaves_total = 0_u64;
    for index in 0..deals as u64 {
        let hand = sample_deal(0, seed, index)?;
        let started = Instant::now();
        solver.solve(&hand).ok_or("no legal placement")?;
        timings.push(started.elapsed().as_secs_f64() * 1_000.0);
        leaves_total += solver.last_leaves;
    }
    timings.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let mean = timings.iter().sum::<f64>() / timings.len() as f64;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "objective": objective.identity(),
            "deals": deals,
            "mean_ms": mean,
            "p50_ms": timings[timings.len() / 2],
            "p95_ms": timings[timings.len() * 95 / 100],
            "p99_ms": timings[timings.len() * 99 / 100],
            "max_ms": timings[timings.len() - 1],
            "mean_leaves_visited": leaves_total as f64 / deals as f64,
            "leaves_total": 2002 * 126 * 4,
            "solves_per_second_single_thread": 1_000.0 / mean,
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn command_dist(arguments: &[String]) -> Result<(), String> {
    let (common, extra) = parse_common(arguments)?;
    common.install_thread_pool();
    let objective = common.build_objective()?;
    let samples = extra_usize(&extra, "samples", 10_000)?;
    let seed = extra_u64(&extra, "seed", 997_000_300)?;
    let chunk = 250_usize;

    let started = Instant::now();
    let stats = (0..samples.div_ceil(chunk))
        .into_par_iter()
        .map(|block| {
            let mut solver = FlSolver::new(&objective).expect("valid objective");
            let mut local = DistributionStats::new();
            let first = block * chunk;
            let last = ((block + 1) * chunk).min(samples);
            for index in first..last {
                let hand = sample_deal(0, seed, index as u64).expect("fresh deck");
                let solution = solver.solve(&hand).expect("a legal placement exists");
                local.observe(&solution);
            }
            local
        })
        .reduce(DistributionStats::new, |mut left, right| {
            left.merge(&right);
            left
        });
    let elapsed = started.elapsed();

    let royalty_bins: Vec<_> = stats
        .royalty_histogram
        .iter()
        .enumerate()
        .map(|(index, count)| {
            let label = if index < ROYALTY_BIN_EDGES.len() {
                let low = if index == 0 {
                    0
                } else {
                    ROYALTY_BIN_EDGES[index - 1] + 1
                };
                format!("{low}..{}", ROYALTY_BIN_EDGES[index])
            } else {
                format!("{}+", ROYALTY_BIN_EDGES[ROYALTY_BIN_EDGES.len() - 1] + 1)
            };
            json!({"bin": label, "count": count,
                   "share": *count as f64 / stats.solved.max(1) as f64})
        })
        .collect();
    let categories = |counts: &Vec<u64>| -> serde_json::Value {
        json!(counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(index, count)| json!({
                "category": CATEGORY_LABELS[index],
                "count": count,
                "share": *count as f64 / stats.solved.max(1) as f64
            }))
            .collect::<Vec<_>>())
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "objective": objective.identity(),
            "fl_ev_config_path": objective.fl_ev_config_path,
            "fl_ev_value": objective.fl_ev_stay,
            "samples": stats.solved,
            "seed_base": seed,
            "seconds": elapsed.as_secs_f64(),
            "solves_per_second": stats.solved as f64 / elapsed.as_secs_f64(),
            "fouls": stats.fouls,
            "stay_rate": stats.stay_rate(),
            "stay_by_reason": stats.stay_by_reason,
            "qq_plus_top_rate": stats.qq_plus_top as f64 / stats.solved.max(1) as f64,
            "mean_royalty": stats.mean_royalty(),
            "royalty_stddev": stats.royalty_standard_deviation(),
            "royalty_histogram": royalty_bins,
            "top_categories": categories(&stats.top_categories),
            "middle_categories": categories(&stats.middle_categories),
            "bottom_categories": categories(&stats.bottom_categories),
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

/// Build the pre-solved Fantasyland pool the T0 teacher draws its opponents from.
///
/// Two thirds of a T0 position is solving and building frontiers for hands that
/// no opening influences -- 16,384 of each per position, at 5.18 ms and 38.3 ms.
/// A pool drawn once from the whole deck and filtered at each leaf by a bitwise
/// AND samples the same distribution: a uniform 14-card hand conditioned on
/// sharing no card with hero's seventeen IS a uniform hand from the thirty-five
/// hero left.
///
/// Sizing: a hand fits a given seventeen one time in 762, so the pool must be
/// several hundred times the sample count for a leaf to have candidates to
/// choose between rather than exactly the ones that happen to fit.
fn command_build_library(arguments: &[String]) -> Result<(), String> {
    let (common, extra) = parse_common(arguments)?;
    common.install_thread_pool();
    let objective = common.build_objective()?;
    let out = extra
        .get("out")
        .ok_or_else(|| "build-library requires --out".to_owned())?;
    let count = extra_usize(&extra, "count", 200_000)?;
    let seed = extra_u64(&extra, "seed", 999_300_000)?;

    let threads = rayon::current_num_threads();
    let chunk = count.div_ceil(threads.max(1));
    let started = Instant::now();
    let parts: Result<Vec<fl_solver_regular::fl_library::FlLibrary>, String> = (0..threads)
        .into_par_iter()
        .map(|index| {
            let first = index * chunk;
            if first >= count {
                return fl_solver_regular::fl_library::FlLibrary::build(
                    0,
                    seed,
                    &objective,
                    &mut FlSolver::new(&objective).expect("valid objective"),
                );
            }
            let take = chunk.min(count - first);
            let mut solver = FlSolver::new(&objective).expect("valid objective");
            // Each worker owns a disjoint stretch of the seed's stream, so the
            // pool is the same whatever the thread count.
            fl_solver_regular::fl_library::FlLibrary::build_range(
                first,
                take,
                seed,
                &objective,
                &mut solver,
            )
        })
        .collect();
    let parts = parts?;
    let mut entries = Vec::with_capacity(count);
    for part in parts {
        entries.extend(part.entries);
    }
    let library = fl_solver_regular::fl_library::FlLibrary {
        entries,
        fl_ev: objective.fl_ev_stay,
        seed,
    };
    let elapsed = started.elapsed();

    let bytes = library.to_bytes();
    std::fs::write(out, &bytes).map_err(|error| format!("failed to write {out}: {error}"))?;
    let rows: usize = library.entries.iter().map(|e| e.frontier.len()).sum();
    println!(
        "{}",
        serde_json::json!({
            "out": out,
            "count": library.entries.len(),
            "seed": seed,
            "fl_ev": objective.fl_ev_stay,
            "bytes": bytes.len(),
            "mean_frontier_rows": rows as f64 / library.entries.len().max(1) as f64,
            "build_seconds": elapsed.as_secs_f64(),
            "threads": threads,
        })
    );
    Ok(())
}

fn command_build_reference(arguments: &[String]) -> Result<(), String> {
    let (common, extra) = parse_common(arguments)?;
    let out = extra
        .get("out")
        .ok_or_else(|| "build-reference requires --out".to_owned())?;
    let reference =
        ReferenceSet::uniform_random_legal(common.reference_samples, common.reference_seed);
    reference.validate()?;
    let text = serde_json::to_string_pretty(&reference).map_err(|error| error.to_string())?;
    std::fs::write(out, text).map_err(|error| format!("failed to write {out}: {error}"))?;
    println!(
        "{}",
        json!({"wrote": out, "sample_count": reference.sample_count,
               "provenance": reference.provenance})
    );
    Ok(())
}

/// Cost the behavior policy charges itself for fouling, on the scale where a
/// legal board is worth `royalty + fl_ev * entry`.
///
/// The Fantasyland opponent's own value must NOT be added here. Writing the
/// hero's heads-up score out both ways:
///
/// ```text
/// foul:  -6 - opponent_royalty - opponent_fl
/// legal:  line + scoop + hero_royalty + hero_fl - opponent_royalty - opponent_fl
/// ```
///
/// the opponent terms are identical in both branches and cancel out of the
/// comparison. What a foul actually forfeits is the 6-point line swing plus the
/// hero's own royalty, entry and line equity. Folding the opponent's ~13 points
/// into the penalty makes the policy roughly twice as foul-averse as it should
/// be, and it pays for that with its Fantasyland entry rate.
const DEFAULT_FOUL_PENALTY: f64 = 6.0;

fn behavior_config(extra: &HashMap<String, String>, fl_ev: f64) -> Result<BehaviorConfig, String> {
    let rollouts = extra_usize(extra, "rollouts", 24)?;
    let foul_penalty = match extra.get("foul-penalty") {
        Some(value) => value.parse().map_err(|_| "bad --foul-penalty")?,
        None => DEFAULT_FOUL_PENALTY,
    };
    Ok(BehaviorConfig {
        rollouts,
        fl_ev,
        foul_penalty,
        foul_penalty_provenance: format!(
            "foul_penalty={foul_penalty:.4} on the scale where a legal board is worth \
             royalty + fl_ev * entry. The 6.0 floor is the line swing a foul concedes; \
             the opponent's royalty and Fantasyland value are deliberately excluded \
             because they appear identically in the fouling and non-fouling branches \
             of the heads-up score and cancel from the decision. Calibrated against \
             configs/fl_ev_regular_v3_direct2.json provenance: opponent_bust_rate \
             0.2482 and opponent_fl_entry_rate 0.2494 for a normal player facing a \
             Fantasyland opponent."
        ),
    })
}

fn command_label(arguments: &[String]) -> Result<(), String> {
    let (common, extra) = parse_common(arguments)?;
    common.install_thread_pool();
    let objective = common.build_objective()?;
    let behavior = behavior_config(&extra, objective.fl_ev_stay)?;
    let roots = extra_usize(&extra, "roots", 2_000)?;
    let first_root = extra_u64(&extra, "first-root", 0)?;
    let samples = extra_usize(&extra, "samples", 200)?;
    let root_seed = extra_u64(&extra, "root-seed", 997_000_000)?;
    let opponent_seed = extra_u64(&extra, "opponent-seed", 997_500_000)?;
    let out = extra
        .get("out")
        .ok_or_else(|| "label requires --out".to_owned())?;

    let teacher = TeacherConfig {
        opponent_samples: samples,
        root_seed_base: root_seed,
        opponent_seed_base: opponent_seed,
    };

    // Roots either come from this crate's stand-in behavior policy or from an
    // externally generated file -- the learned chain writes one.
    let external = match extra.get("roots-file") {
        Some(path) => {
            let mut loaded = read_roots(path)?;
            loaded.truncate(roots);
            Some(loaded)
        }
        None => None,
    };
    let root_policy = extra
        .get("root-policy")
        .cloned()
        .unwrap_or_else(|| behavior.identity());

    let started = Instant::now();
    let mut labelled: Vec<_> = match &external {
        Some(loaded) => loaded
            .par_iter()
            .map(|record| {
                let root = HeroT4Root {
                    root_index: record.root_index,
                    top: record.top.clone(),
                    middle: record.middle.clone(),
                    bottom: record.bottom.clone(),
                    dealt: record.dealt.clone().try_into().expect("three dealt cards"),
                    discards: record
                        .discards
                        .clone()
                        .try_into()
                        .expect("three prior discards at T4"),
                    seen_mask: record.seen_mask,
                };
                let mut solver = FlSolver::new(&objective).expect("valid objective");
                label_root(&root, &objective, &teacher, &root_policy, &mut solver)
                    .expect("a T4 root always has a legal candidate")
            })
            .collect(),
        None => (first_root..first_root + roots as u64)
            .into_par_iter()
            .map(|root_index| {
                let root = generate_root(root_seed, root_index, &behavior);
                let mut solver = FlSolver::new(&objective).expect("valid objective");
                label_root(&root, &objective, &teacher, &root_policy, &mut solver)
                    .expect("a T4 root always has a legal candidate")
            })
            .collect(),
    };
    labelled.sort_by_key(|record| record.root_index);
    let elapsed = started.elapsed();

    let file = std::fs::File::create(out)
        .map_err(|error| format!("failed to create {out}: {error}"))?;
    let mut writer = std::io::BufWriter::new(file);
    for record in labelled.iter() {
        let line = serde_json::to_string(record).map_err(|error| error.to_string())?;
        writeln!(writer, "{line}").map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;

    println!(
        "{}",
        json!({
            "wrote": out,
            "roots": labelled.len(),
            "opponent_samples": samples,
            "seconds": elapsed.as_secs_f64(),
            "roots_per_second": labelled.len() as f64 / elapsed.as_secs_f64(),
            "objective": objective.identity(),
            "root_policy": root_policy,
            "roots_from_file": external.is_some(),
            "behavior_policy": behavior.identity(),
            "foul_penalty_provenance": behavior.foul_penalty_provenance,
        })
    );
    Ok(())
}

/// Read externally generated roots (one JSON object per line).
fn read_roots(path: &str) -> Result<Vec<ExternalRoot>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read roots {path}: {error}"))?;
    let mut roots = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let root: ExternalRoot = serde_json::from_str(line).map_err(|error| {
            format!("invalid root on line {} of {path}: {error}", line_number + 1)
        })?;
        roots.push(root);
    }
    if roots.is_empty() {
        return Err(format!("{path} holds no roots"));
    }
    Ok(roots)
}

fn label_t3_batch(
    roots: &[ExternalRoot],
    objective: &ObjectiveConfig,
    teacher: &TeacherConfig,
    hero_deals: usize,
    hero_deal_seed: u64,
    root_policy: &str,
    adaptive: bool,
) -> Vec<LabelledT3Root> {
    let mut labelled: Vec<LabelledT3Root> = roots
        .par_iter()
        .map(|root| {
            let mut solver = FlSolver::new(objective).expect("valid objective");
            label_t3_root(
                root,
                objective,
                teacher,
                hero_deals,
                hero_deal_seed,
                root_policy,
                adaptive,
                &mut solver,
            )
            .expect("a T3 root always has a legal candidate")
        })
        .collect();
    labelled.sort_by_key(|record| record.root_index);
    labelled
}

fn command_label_t3(arguments: &[String]) -> Result<(), String> {
    let (common, extra) = parse_common(arguments)?;
    common.install_thread_pool();
    let objective = common.build_objective()?;
    let roots_path = extra
        .get("roots-file")
        .ok_or_else(|| "label-t3 requires --roots-file".to_owned())?;
    let out = extra
        .get("out")
        .ok_or_else(|| "label-t3 requires --out".to_owned())?;
    let samples = extra_usize(&extra, "samples", 800)?;
    let hero_deals = extra_usize(&extra, "hero-deals", 64)?;
    let opponent_seed = extra_u64(&extra, "opponent-seed", 997_600_000)?;
    let hero_deal_seed = extra_u64(&extra, "hero-deal-seed", 997_700_000)?;
    let limit = extra_usize(&extra, "limit", usize::MAX)?;
    let root_policy = extra
        .get("root-policy")
        .cloned()
        .unwrap_or_else(|| "unspecified".to_owned());
    let adaptive = extra.get("opponent-mode").map(String::as_str) == Some("adaptive");

    let mut roots = read_roots(roots_path)?;
    roots.truncate(limit);
    let teacher = TeacherConfig {
        opponent_samples: samples,
        root_seed_base: hero_deal_seed,
        opponent_seed_base: opponent_seed,
    };

    let started = Instant::now();
    let labelled = label_t3_batch(
        &roots,
        &objective,
        &teacher,
        hero_deals,
        hero_deal_seed,
        &root_policy,
        adaptive,
    );
    let elapsed = started.elapsed();

    let file =
        std::fs::File::create(out).map_err(|error| format!("failed to create {out}: {error}"))?;
    let mut writer = std::io::BufWriter::new(file);
    for record in labelled.iter() {
        let line = serde_json::to_string(record).map_err(|error| error.to_string())?;
        writeln!(writer, "{line}").map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;

    let candidate_total: usize = labelled.iter().map(|r| r.candidates.len()).sum();
    let threads = rayon::current_num_threads();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "wrote": out,
            "roots": labelled.len(),
            "opponent_samples": samples,
            // Report what was actually enumerated, not what was requested:
            // `--hero-deals 0` means exhaustive and expands to C(unseen, 3).
            "hero_deal_samples": labelled.first().map(|r| r.hero_deal_samples).unwrap_or(0),
            "deal_mode": labelled
                .first()
                .map(|r| r.deal_mode.clone())
                .unwrap_or_default(),
            "mean_candidates": candidate_total as f64 / labelled.len().max(1) as f64,
            "mean_usable_opponents": labelled.iter().map(|r| r.mean_usable_opponents).sum::<f64>()
                / labelled.len().max(1) as f64,
            "seconds": elapsed.as_secs_f64(),
            "roots_per_second": labelled.len() as f64 / elapsed.as_secs_f64(),
            "threads": threads,
            "core_seconds_per_root": elapsed.as_secs_f64() * threads as f64
                / labelled.len().max(1) as f64,
            "objective": objective.identity(),
            "root_policy": root_policy,
            "seed_bases": {
                "opponent": opponent_seed,
                "hero_deal": hero_deal_seed,
            },
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

/// Convergence probe over the hero-deal sample count `M`.
///
/// `N` (opponent Fantasyland samples) is held fixed and only `M` varies, so the
/// table isolates the outer expectation. The reference is a high-`M` run on the
/// same roots with the same opponent pool.
fn command_probe_t3(arguments: &[String]) -> Result<(), String> {
    let (common, extra) = parse_common(arguments)?;
    common.install_thread_pool();
    let objective = common.build_objective()?;
    let roots_path = extra
        .get("roots-file")
        .ok_or_else(|| "probe-t3 requires --roots-file".to_owned())?;
    let samples = extra_usize(&extra, "samples", 800)?;
    let opponent_seed = extra_u64(&extra, "opponent-seed", 997_600_000)?;
    let hero_deal_seed = extra_u64(&extra, "hero-deal-seed", 997_700_000)?;
    let reference_deals = extra_usize(&extra, "reference-hero-deals", 512)?;
    let limit = extra_usize(&extra, "limit", 120)?;
    let ladder: Vec<usize> = match extra.get("ladder") {
        Some(text) => text
            .split(',')
            .map(|value| value.trim().parse::<usize>().map_err(|_| "bad --ladder"))
            .collect::<Result<Vec<_>, _>>()?,
        None => vec![8, 16, 32, 64, 128, 256],
    };

    let mut roots = read_roots(roots_path)?;
    roots.truncate(limit);
    let teacher = TeacherConfig {
        opponent_samples: samples,
        root_seed_base: hero_deal_seed,
        opponent_seed_base: opponent_seed,
    };

    let reference = label_t3_batch(
        &roots,
        &objective,
        &teacher,
        reference_deals,
        // A disjoint hero-deal stream, still inside the 997-million block.
        hero_deal_seed + 100_000,
        "probe-reference",
        false,
    );

    let mut rows = Vec::new();
    for rung in ladder.iter().copied() {
        let started = Instant::now();
        let labels = label_t3_batch(&roots, &objective, &teacher, rung, hero_deal_seed, "probe", false);
        let elapsed = started.elapsed();

        let mut optimal = 0_usize;
        let mut regret_total = 0.0_f64;
        let mut standard_error_total = 0.0_f64;
        let mut decisive = 0_usize;
        let mut decisive_optimal = 0_usize;
        for (candidate_label, target) in labels.iter().zip(reference.iter()) {
            let picked = candidate_label.best_candidate;
            let target_best = target.candidates[target.best_candidate].expected_value;
            let gap = target_best - target.candidates[picked].expected_value;
            regret_total += gap;
            if gap <= 1e-9 {
                optimal += 1;
            }
            standard_error_total += candidate_label.candidates[picked].standard_error;
            if target.decision_margin >= 1e-9 {
                decisive += 1;
                if gap <= 1e-9 {
                    decisive_optimal += 1;
                }
            }
        }
        let denominator = labels.len().max(1) as f64;
        rows.push(json!({
            "hero_deal_samples": rung,
            "roots": labels.len(),
            "value_optimal_vs_reference": optimal as f64 / denominator,
            "mean_regret": regret_total / denominator,
            "mean_label_standard_error": standard_error_total / denominator,
            "decisive_roots": decisive,
            "value_optimal_on_decisive": decisive_optimal as f64 / decisive.max(1) as f64,
            "seconds": elapsed.as_secs_f64(),
            "roots_per_second": denominator / elapsed.as_secs_f64(),
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "objective": objective.identity(),
            "roots": roots.len(),
            "opponent_samples": samples,
            "reference_hero_deals": reference_deals,
            "candidate_counts": {
                "min": reference.iter().map(|r| r.candidates.len()).min(),
                "max": reference.iter().map(|r| r.candidates.len()).max(),
                "mean": reference.iter().map(|r| r.candidates.len()).sum::<usize>() as f64
                    / reference.len().max(1) as f64,
            },
            "mean_usable_opponents": reference.iter().map(|r| r.mean_usable_opponents).sum::<f64>()
                / reference.len().max(1) as f64,
            "reference_zero_margin_share": reference
                .iter()
                .filter(|record| record.decision_margin < 1e-9)
                .count() as f64
                / reference.len().max(1) as f64,
            "seed_bases": {"opponent": opponent_seed, "hero_deal": hero_deal_seed,
                           "reference_hero_deal": hero_deal_seed + 100_000},
            "ladder": rows,
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

/// Convergence probe over the opponent Fantasyland sample count `N`.
///
/// With the hero-deal expectation taken exhaustively, `N` is the only source of
/// sampling error left in a T3 label -- and it is the only real cost lever,
/// since the solves dominate. The reference draws from a *different* opponent
/// stream so a rung is not scored against a superset of its own samples.
/// Frontier size, construction cost, per-terminal scan cost, and how far below
/// the static optimum the adaptive winner actually sits.
fn command_frontier_stats(arguments: &[String]) -> Result<(), String> {
    use fl_solver_regular::eval::{eval3, eval5, is_foul};
    use fl_solver_regular::frontier::{best_response, build_frontier};

    let (common, extra) = parse_common(arguments)?;
    common.install_thread_pool();
    let objective = common.build_objective()?;
    let deals = extra_usize(&extra, "deals", 2_000)?;
    let heroes = extra_usize(&extra, "heroes-per-deal", 8)?;
    let seed = extra_u64(&extra, "seed", 997_960_000)?;

    #[derive(Default)]
    struct Row {
        size: usize,
        build_micros: u64,
        scan_nanos: u64,
        scans: u64,
        deficits: Vec<f64>,
    }

    let rows: Vec<Row> = (0..deals as u64)
        .into_par_iter()
        .map(|index| {
            let mut rng = fl_solver_regular::rng::SplitMix64::for_stream(seed, index);
            let mut deck: Vec<u8> = (0..52).collect();
            rng.partial_shuffle(&mut deck, 14);
            let hand: [u8; 14] = deck[..14].try_into().expect("14 cards");

            let started = Instant::now();
            let frontier = build_frontier(&hand, objective.fl_ev_stay);
            let build_micros = started.elapsed().as_micros() as u64;

            // The static optimum: the best royalty+stay arrangement, which is
            // what a non-adaptive Fantasyland player would set.
            let static_max = frontier
                .iter()
                .map(|entry| entry.static_value)
                .fold(f64::NEG_INFINITY, f64::max);

            let mut row = Row {
                size: frontier.len(),
                build_micros,
                ..Default::default()
            };
            for hero_index in 0..heroes as u64 {
                let mut hero_rng = fl_solver_regular::rng::SplitMix64::for_stream(
                    seed + 1,
                    index * heroes as u64 + hero_index,
                );
                let mut rest: Vec<u8> = (0..52)
                    .filter(|card| !hand.contains(card))
                    .collect();
                hero_rng.partial_shuffle(&mut rest, 13);
                let top: [u8; 3] = rest[..3].try_into().unwrap();
                let middle: [u8; 5] = rest[3..8].try_into().unwrap();
                let bottom: [u8; 5] = rest[8..13].try_into().unwrap();
                let (hero_top, hero_middle, hero_bottom) =
                    (eval3(&top), eval5(&middle), eval5(&bottom));
                if is_foul(hero_top, hero_middle, hero_bottom) {
                    continue;
                }
                let started = Instant::now();
                let (_, best_index) =
                    best_response(&frontier, hero_top, hero_middle, hero_bottom);
                row.scan_nanos += started.elapsed().as_nanos() as u64;
                row.scans += 1;
                // How much royalty+stay the adaptive winner gave up to win lines.
                row.deficits
                    .push(static_max - frontier[best_index].static_value);
            }
            row
        })
        .collect();

    let mut sizes: Vec<usize> = rows.iter().map(|row| row.size).collect();
    sizes.sort_unstable();
    let mut builds: Vec<u64> = rows.iter().map(|row| row.build_micros).collect();
    builds.sort_unstable();
    let mut deficits: Vec<f64> = rows.iter().flat_map(|row| row.deficits.clone()).collect();
    deficits.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let scan_nanos: u64 = rows.iter().map(|row| row.scan_nanos).sum();
    let scans: u64 = rows.iter().map(|row| row.scans).sum();
    let percentile = |sorted: &[f64], q: f64| -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        sorted[((sorted.len() as f64 - 1.0) * q).round() as usize]
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "objective": objective.identity(),
            "deals": deals,
            "hero_boards_per_deal": heroes,
            "frontier_size": {
                "mean": sizes.iter().sum::<usize>() as f64 / sizes.len() as f64,
                "p50": sizes[sizes.len() / 2],
                "p95": sizes[sizes.len() * 95 / 100],
                "max": sizes[sizes.len() - 1],
                "min": sizes[0],
            },
            "frontier_build_micros": {
                "mean": builds.iter().sum::<u64>() as f64 / builds.len() as f64,
                "p50": builds[builds.len() / 2],
                "p95": builds[builds.len() * 95 / 100],
                "max": builds[builds.len() - 1],
            },
            "scan_nanos_per_terminal": {
                "mean": scan_nanos as f64 / scans.max(1) as f64,
                "terminals_scored": scans,
            },
            "static_value_deficit_of_best_response": {
                "mean": deficits.iter().sum::<f64>() / deficits.len().max(1) as f64,
                "p95": percentile(&deficits, 0.95),
                "max": deficits.last().copied().unwrap_or(0.0),
                "share_over_5": deficits.iter().filter(|d| **d > 5.0).count() as f64
                    / deficits.len().max(1) as f64,
                "share_zero": deficits.iter().filter(|d| **d <= 1e-9).count() as f64
                    / deficits.len().max(1) as f64,
                "theory_bound": 12.0,
                "owner_band_heuristic": 5.0,
                "samples": deficits.len(),
            },
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}


/// Label T2-vs-FL roots. The two hero draws are sampled (`--t3-draws`,
/// `--t4-draws`); T3 and T4 candidate fans are exhaustive.
fn command_label_t2(arguments: &[String]) -> Result<(), String> {
    use fl_solver_regular::t2_teacher::{label_t2_root, LabelledT2Root};

    let (common, extra) = parse_common(arguments)?;
    common.install_thread_pool();
    let objective = common.build_objective()?;
    let roots_path = extra
        .get("roots-file")
        .ok_or_else(|| "label-t2 requires --roots-file".to_owned())?;
    let out = extra
        .get("out")
        .ok_or_else(|| "label-t2 requires --out".to_owned())?;
    let samples = extra_usize(&extra, "samples", 200)?;
    let t3_draws = extra_usize(&extra, "t3-draws", 8)?;
    let t4_draws = extra_usize(&extra, "t4-draws", 4)?;
    let opponent_seed = extra_u64(&extra, "opponent-seed", 997_610_000)?;
    let hero_deal_seed = extra_u64(&extra, "hero-deal-seed", 997_710_000)?;
    let limit = extra_usize(&extra, "limit", usize::MAX)?;
    let root_policy = extra
        .get("root-policy")
        .cloned()
        .unwrap_or_else(|| "unspecified".to_owned());

    let mut roots = read_roots(roots_path)?;
    roots.truncate(limit);
    let teacher = TeacherConfig {
        opponent_samples: samples,
        root_seed_base: hero_deal_seed,
        opponent_seed_base: opponent_seed,
    };
    let started = Instant::now();
    let mut labelled: Vec<LabelledT2Root> = roots
        .par_iter()
        .map(|root| {
            let mut solver = FlSolver::new(&objective).expect("valid objective");
            label_t2_root(
                root,
                &objective,
                &teacher,
                t3_draws,
                t4_draws,
                hero_deal_seed,
                &root_policy,
                &mut solver,
            )
            .expect("a T2 root always has a legal candidate")
        })
        .collect();
    labelled.sort_by_key(|record| record.root_index);
    let elapsed = started.elapsed();

    let file =
        std::fs::File::create(out).map_err(|error| format!("failed to create {out}: {error}"))?;
    let mut writer = std::io::BufWriter::new(file);
    for record in labelled.iter() {
        writeln!(writer, "{}", serde_json::to_string(record).map_err(|e| e.to_string())?)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;

    let threads = rayon::current_num_threads();
    let candidate_total: usize = labelled.iter().map(|r| r.candidates.len()).sum();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "wrote": out,
            "roots": labelled.len(),
            "opponent_samples": samples,
            "t3_draws": t3_draws,
            "t4_draws": t4_draws,
            "opponent_mode": "adaptive_v1",
            "mean_candidates": candidate_total as f64 / labelled.len().max(1) as f64,
            "mean_usable_opponents": labelled.iter().map(|r| r.mean_usable_opponents).sum::<f64>()
                / labelled.len().max(1) as f64,
            "mean_label_standard_error": labelled.iter()
                .map(|r| r.candidates.iter().map(|c| c.standard_error).sum::<f64>()
                     / r.candidates.len() as f64)
                .sum::<f64>() / labelled.len().max(1) as f64,
            "seconds": elapsed.as_secs_f64(),
            "core_seconds_per_root": elapsed.as_secs_f64() * threads as f64
                / labelled.len().max(1) as f64,
            "threads": threads,
            "seed_bases": {"opponent": opponent_seed, "hero_deal": hero_deal_seed},
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

/// Load the two continuation models, both pinned by digest.
///
/// The digest is required rather than computed. A continuation model that is
/// not the measured one produces a label describing a policy nobody evaluated,
/// and that is precisely the failure the rest of this pipeline pins against.
fn load_continuation(
    extra: &std::collections::HashMap<String, String>,
) -> Result<fl_solver_regular::t1_teacher::Continuation, String> {
    use fl_solver_regular::t1_teacher::Continuation;
    let read = |key: &str| -> Result<Vec<u8>, String> {
        let path = extra
            .get(key)
            .ok_or_else(|| format!("--{key} is required"))?;
        std::fs::read(path).map_err(|error| format!("{path}: {error}"))
    };
    let sha = |key: &str| -> Result<String, String> {
        extra
            .get(key)
            .cloned()
            .ok_or_else(|| format!("--{key} is required (continuation models are pinned)"))
    };
    Continuation::load_pinned(
        &read("t2-model")?,
        &sha("t2-model-sha256")?,
        &read("t3-model")?,
        &sha("t3-model-sha256")?,
    )
}

/// Label T0 first-seat roots against a Fantasyland opponent.
///
/// One street deeper than `label-t1` and otherwise its shape, so it takes the
/// same pinned continuation pair plus a T1 ranker of its own. Openings are the
/// 232-wide fan, which is why the draw counts default lower than T1's: the
/// terminal count is the fan times every nested draw, and the fan is 8.6x
/// wider here.
fn command_label_t0(arguments: &[String]) -> Result<(), String> {
    use fl_solver_regular::t0_teacher::{label_t0_root, LabelledT0Root};
    use fl_solver_regular::vfl_model::VflModel;

    let (common, extra) = parse_common(arguments)?;
    common.install_thread_pool();
    let objective = common.build_objective()?;
    let roots_path = extra
        .get("roots-file")
        .ok_or_else(|| "label-t0 requires --roots-file".to_owned())?;
    let out = extra
        .get("out")
        .ok_or_else(|| "label-t0 requires --out".to_owned())?;
    let samples = extra_usize(&extra, "samples", 200)?;
    let t1_draws = extra_usize(&extra, "t1-draws", 8)?;
    let t2_draws = extra_usize(&extra, "t2-draws", 4)?;
    let t3_draws = extra_usize(&extra, "t3-draws", 2)?;
    let t4_draws = extra_usize(&extra, "t4-draws", 2)?;
    let opponent_seed = extra_u64(&extra, "opponent-seed", 997_820_000)?;
    let hero_deal_seed = extra_u64(&extra, "hero-deal-seed", 997_920_000)?;
    let limit = extra_usize(&extra, "limit", usize::MAX)?;
    let root_policy = extra
        .get("root-policy")
        .cloned()
        .unwrap_or_else(|| "unspecified".to_owned());
    let continuation = load_continuation(&extra)?;

    // The T1 ranker is pinned the same way the T2/T3 pair is: a continuation
    // that is not digest-checked is a continuation nobody can reproduce.
    let t1_bytes = std::fs::read(
        extra
            .get("t1-model")
            .ok_or_else(|| "--t1-model is required".to_owned())?,
    )
    .map_err(|error| error.to_string())?;
    let t1_sha = extra
        .get("t1-model-sha256")
        .cloned()
        .ok_or_else(|| "--t1-model-sha256 is required (continuations are pinned)".to_owned())?;
    let t1_model = VflModel::load_pinned(&t1_bytes, &t1_sha)?;

    // Optional fan narrowing. Unlike T1, where the shared draw tree dominates,
    // every T0 opening leaves a different board and so pays for its own
    // Fantasyland solving -- the fan is the multiplier here. The ranker is
    // pinned like every other model, and a plan that names one without a digest
    // is refused rather than trusted.
    let narrow_keep = extra_usize(&extra, "narrow-keep", 0)?;
    let narrow = match extra.get("narrow-model") {
        None => {
            if narrow_keep > 0 {
                return Err("--narrow-keep needs --narrow-model to rank the fan".to_owned());
            }
            None
        }
        Some(path) => {
            let sha = extra.get("narrow-model-sha256").ok_or_else(|| {
                "--narrow-model-sha256 is required (the narrowing model is pinned)".to_owned()
            })?;
            let bytes = std::fs::read(path).map_err(|error| format!("{path}: {error}"))?;
            Some((
                fl_solver_regular::vfl_model::VflModel::load_pinned(&bytes, sha)?,
                sha.clone(),
            ))
        }
    };

    // The pre-solved pool, if this run uses one. It is pinned by digest like
    // every other artifact: a pool is as much a part of a label's identity as
    // the continuation models, since it decides which opponents were averaged.
    let library = match extra.get("fl-library") {
        None => None,
        Some(path) => {
            let sha = extra.get("fl-library-sha256").ok_or_else(|| {
                "--fl-library-sha256 is required (the pool is pinned)".to_owned()
            })?;
            let bytes = std::fs::read(path).map_err(|error| format!("{path}: {error}"))?;
            let actual = {
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(&bytes);
                format!("{:x}", hasher.finalize())
            };
            if &actual != sha {
                return Err(format!(
                    "pool digest mismatch: {path} hashes to {actual}, the plan pins {sha}"
                ));
            }
            Some(fl_solver_regular::fl_library::FlLibrary::from_bytes(
                &bytes,
                objective.fl_ev_stay,
            )?)
        }
    };

    let mut roots = read_roots(roots_path)?;
    roots.truncate(limit);
    let teacher = TeacherConfig {
        opponent_samples: samples,
        root_seed_base: hero_deal_seed,
        opponent_seed_base: opponent_seed,
    };
    let started = Instant::now();
    let labelled: Result<Vec<LabelledT0Root>, String> = roots
        .par_iter()
        .map(|root| {
            let mut solver = FlSolver::new(&objective).expect("valid objective");
            label_t0_root(
                root,
                &objective,
                &teacher,
                &continuation,
                &t1_model,
                t1_draws,
                t2_draws,
                t3_draws,
                t4_draws,
                hero_deal_seed,
                &root_policy,
                narrow.as_ref().map(|(model, _)| model),
                narrow_keep,
                narrow.as_ref().map(|(_, sha)| sha.as_str()),
                library.as_ref(),
                &mut solver,
            )
        })
        .collect();
    let mut labelled = labelled?;
    labelled.sort_by_key(|record| record.root_index);
    let elapsed = started.elapsed();

    let file =
        std::fs::File::create(out).map_err(|error| format!("failed to create {out}: {error}"))?;
    let mut writer = std::io::BufWriter::new(file);
    for record in labelled.iter() {
        writeln!(
            writer,
            "{}",
            serde_json::to_string(record).map_err(|e| e.to_string())?
        )
        .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;

    let candidate_total: usize = labelled.iter().map(|r| r.candidates.len()).sum();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "wrote": out,
            "roots": labelled.len(),
            "opponent_samples": samples,
            "t1_draws": t1_draws,
            "t2_draws": t2_draws,
            "t3_draws": t3_draws,
            "t4_draws": t4_draws,
            "opponent_mode": "adaptive_v1",
            "candidates_total": candidate_total,
            "candidates_per_root": candidate_total as f64 / labelled.len().max(1) as f64,
            "threads": rayon::current_num_threads(),
            "seconds": elapsed.as_secs_f64(),
            "seconds_per_root": elapsed.as_secs_f64() / labelled.len().max(1) as f64,
            "t1_model_sha256": t1_sha,
            "hero_deal_seed_base": hero_deal_seed,
            "opponent_seed_base": opponent_seed,
        }))
        .map_err(|e| e.to_string())?
    );
    Ok(())
}

fn command_label_t1(arguments: &[String]) -> Result<(), String> {
    use fl_solver_regular::t1_teacher::{label_t1_root, LabelledT1Root};

    let (common, extra) = parse_common(arguments)?;
    common.install_thread_pool();
    let objective = common.build_objective()?;
    let roots_path = extra
        .get("roots-file")
        .ok_or_else(|| "label-t1 requires --roots-file".to_owned())?;
    let out = extra
        .get("out")
        .ok_or_else(|| "label-t1 requires --out".to_owned())?;
    let samples = extra_usize(&extra, "samples", 200)?;
    let t2_draws = extra_usize(&extra, "t2-draws", 16)?;
    let t3_draws = extra_usize(&extra, "t3-draws", 8)?;
    let t4_draws = extra_usize(&extra, "t4-draws", 8)?;
    let opponent_seed = extra_u64(&extra, "opponent-seed", 997_620_000)?;
    let hero_deal_seed = extra_u64(&extra, "hero-deal-seed", 997_720_000)?;
    let limit = extra_usize(&extra, "limit", usize::MAX)?;
    let root_policy = extra
        .get("root-policy")
        .cloned()
        .unwrap_or_else(|| "unspecified".to_owned());
    let continuation = load_continuation(&extra)?;

    // Optional fan narrowing. T1's 27 openings each carry a T2 reply, a T3
    // reply, an exhaustive T4 and a Fantasyland solve, so the fan multiplies
    // every other cost; ranking it with the street's own distilled model and
    // keeping the top few converts that into draws, which is where a label's
    // error actually lives. The model is pinned like every other one, and a
    // plan that names it without a digest is refused rather than trusted.
    let narrow_keep = extra_usize(&extra, "narrow-keep", 0)?;
    let narrow = match extra.get("t1-model") {
        None => {
            if narrow_keep > 0 {
                return Err("--narrow-keep needs --t1-model to rank the fan".to_owned());
            }
            None
        }
        Some(path) => {
            let sha = extra.get("t1-model-sha256").ok_or_else(|| {
                "--t1-model-sha256 is required (the narrowing model is pinned)".to_owned()
            })?;
            let bytes = std::fs::read(path).map_err(|error| format!("{path}: {error}"))?;
            Some((
                fl_solver_regular::vfl_model::VflModel::load_pinned(&bytes, sha)?,
                sha.clone(),
            ))
        }
    };

    let mut roots = read_roots(roots_path)?;
    roots.truncate(limit);
    let teacher = TeacherConfig {
        opponent_samples: samples,
        root_seed_base: hero_deal_seed,
        opponent_seed_base: opponent_seed,
    };
    let started = Instant::now();
    let labelled: Result<Vec<LabelledT1Root>, String> = roots
        .par_iter()
        .map(|root| {
            let mut solver = FlSolver::new(&objective).expect("valid objective");
            label_t1_root(
                root,
                &objective,
                &teacher,
                &continuation,
                t2_draws,
                t3_draws,
                t4_draws,
                hero_deal_seed,
                &root_policy,
                narrow.as_ref().map(|(model, _)| model),
                narrow_keep,
                narrow.as_ref().map(|(_, sha)| sha.as_str()),
                &mut solver,
            )
        })
        .collect();
    let mut labelled = labelled?;
    labelled.sort_by_key(|record| record.root_index);
    let elapsed = started.elapsed();

    let file =
        std::fs::File::create(out).map_err(|error| format!("failed to create {out}: {error}"))?;
    let mut writer = std::io::BufWriter::new(file);
    for record in labelled.iter() {
        writeln!(
            writer,
            "{}",
            serde_json::to_string(record).map_err(|e| e.to_string())?
        )
        .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;

    let threads = rayon::current_num_threads();
    let candidate_total: usize = labelled.iter().map(|r| r.candidates.len()).sum();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "wrote": out,
            "roots": labelled.len(),
            "opponent_samples": samples,
            "t2_draws": t2_draws,
            "t3_draws": t3_draws,
            "t4_draws": t4_draws,
            "opponent_mode": "adaptive_v1",
            "continuation": labelled.first().map(|r| r.continuation.clone()),
            "mean_candidates": candidate_total as f64 / labelled.len().max(1) as f64,
            "mean_usable_opponents": labelled.iter().map(|r| r.mean_usable_opponents).sum::<f64>()
                / labelled.len().max(1) as f64,
            "mean_label_standard_error": labelled.iter()
                .map(|r| r.candidates.iter().map(|c| c.standard_error).sum::<f64>()
                     / r.candidates.len() as f64)
                .sum::<f64>() / labelled.len().max(1) as f64,
            "seconds": elapsed.as_secs_f64(),
            "core_seconds_per_root": elapsed.as_secs_f64() * threads as f64
                / labelled.len().max(1) as f64,
            "threads": threads,
            "seed_bases": {"opponent": opponent_seed, "hero_deal": hero_deal_seed},
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

/// The T1 cost/noise grid: every rung against a high-draw reference.
///
/// Same shape as the T2 grid. What a rung is judged on is not its absolute
/// value but how often it picks the reference's argmax and how much expected
/// value it gives up when it does not -- a teacher is a ranker, and a constant
/// offset it shares with the reference costs a student nothing.
fn command_probe_t1_grid(arguments: &[String]) -> Result<(), String> {
    use fl_solver_regular::t1_teacher::label_t1_root;

    let (common, extra) = parse_common(arguments)?;
    common.install_thread_pool();
    let objective = common.build_objective()?;
    let roots_path = extra
        .get("roots-file")
        .ok_or_else(|| "probe-t1-grid requires --roots-file".to_owned())?;
    let samples = extra_usize(&extra, "samples", 200)?;
    // NOT `--reference-samples`: that name is consumed by `parse_common` into
    // `CommonArgs`, so it never reaches `extra` and a value passed under it is
    // silently replaced by this default. That trap already cost one run -- an
    // N-axis probe whose yardstick reported 25,600 on the command line and ran
    // at 400, which is exactly the sample size it was supposed to be judging.
    // `--reference-samples` is consumed by `parse_common` into `CommonArgs`, so
    // it never reaches `extra`. Reading it from `extra` -- as this did, and as
    // `probe-t3-n` still does -- silently substitutes a default, which is how an
    // N-axis probe came to report a 25,600 yardstick while running one of 400:
    // the sample size it existed to judge.
    let reference_samples = common.reference_samples;
    let reference_grid = (
        extra_usize(&extra, "reference-t2-draws", 64)?,
        extra_usize(&extra, "reference-t3-draws", 32)?,
        extra_usize(&extra, "reference-t4-draws", 16)?,
    );
    let opponent_seed = extra_u64(&extra, "opponent-seed", 997_620_000)?;
    let reference_opponent_seed = extra_u64(&extra, "reference-opponent-seed", 997_660_000)?;
    let hero_deal_seed = extra_u64(&extra, "hero-deal-seed", 997_720_000)?;
    // The reference MUST draw from its own seed stream.
    //
    // Deals are addressed by (seed_base, root, t2_index, t3_index, t4_index),
    // so a rung sharing the reference's seed reuses a PREFIX of the reference's
    // own draws -- a 32x16x8 rung against a 32x16x16 reference is then the same
    // estimator on the same deals, and scores a regret near zero that measures
    // nothing. Common random numbers belong between candidates, where they cancel
    // noise; between an estimator and its yardstick they manufacture agreement.
    let reference_hero_deal_seed = extra_u64(&extra, "reference-hero-deal-seed", 997_730_000)?;
    let limit = extra_usize(&extra, "limit", 40)?;
    let root_policy = extra
        .get("root-policy")
        .cloned()
        .unwrap_or_else(|| "probe".to_owned());
    let continuation = load_continuation(&extra)?;

    // Rungs as "t2xt3xt4" or "t2xt3xt4@samples", comma separated. The optional
    // @samples exists so the OPPONENT-sample axis can be probed on its own: at
    // T1 only about 2% of opponent samples survive collision filtering against
    // the hero's nine future draw cards, so the effective sample behind every
    // terminal is N/50, and N is a separate convergence question from the draws.
    let rungs: Vec<(usize, usize, usize, usize)> = match extra.get("grid") {
        Some(text) => text
            .split(',')
            .map(|rung| {
                let (shape, rung_samples) = match rung.trim().split_once('@') {
                    Some((shape, n)) => (
                        shape,
                        n.parse::<usize>().map_err(|e| format!("bad @samples: {e}"))?,
                    ),
                    None => (rung.trim(), samples),
                };
                let parts: Vec<&str> = shape.split('x').collect();
                if parts.len() != 3 {
                    return Err(format!("bad --grid rung {rung:?}, want t2xt3xt4[@N]"));
                }
                Ok((
                    parts[0].parse::<usize>().map_err(|e| e.to_string())?,
                    parts[1].parse::<usize>().map_err(|e| e.to_string())?,
                    parts[2].parse::<usize>().map_err(|e| e.to_string())?,
                    rung_samples,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?,
        None => vec![
            (8, 4, 4, samples),
            (8, 8, 8, samples),
            (16, 8, 8, samples),
            (16, 16, 8, samples),
            (32, 16, 8, samples),
        ],
    };

    let mut roots = read_roots(roots_path)?;
    roots.truncate(limit);
    let threads = rayon::current_num_threads();

    let run = |grid: (usize, usize, usize),
               teacher: &TeacherConfig,
               deal_seed: u64|
     -> Result<Vec<fl_solver_regular::t1_teacher::LabelledT1Root>, String> {
        roots
            .par_iter()
            .map(|root| {
                let mut solver = FlSolver::new(&objective).expect("valid objective");
                label_t1_root(
                    root,
                    &objective,
                    teacher,
                    &continuation,
                    grid.0,
                    grid.1,
                    grid.2,
                    deal_seed,
                    &root_policy,
                    None,
                    0,
                    None,
                    &mut solver,
                )
            })
            .collect()
    };

    // A yardstick has to clear the rungs on the axis being measured. Opponent
    // samples survive collision filtering only ~2% of the time at T1, so what
    // matters is effective sample, and a reference whose N sits at or below a
    // rung's cannot bound that rung's error at all.
    let largest_rung_samples = rungs.iter().map(|rung| rung.3).max().unwrap_or(0);
    if reference_samples < largest_rung_samples.saturating_mul(4) {
        return Err(format!(
            "reference uses {reference_samples} opponent samples but the largest \
             rung uses {largest_rung_samples}; pass --reference-samples at least \
             4x the largest rung so the yardstick is not the thing being measured"
        ));
    }
    let reference_teacher = TeacherConfig {
        opponent_samples: reference_samples,
        root_seed_base: reference_hero_deal_seed,
        opponent_seed_base: reference_opponent_seed,
    };
    let reference_started = Instant::now();
    let reference = run(reference_grid, &reference_teacher, reference_hero_deal_seed)?;
    let reference_seconds = reference_started.elapsed().as_secs_f64();

    let mut rows = Vec::new();
    for (t2, t3, t4, rung_samples) in rungs.iter().copied() {
        let grid = (t2, t3, t4);
        let teacher = TeacherConfig {
            opponent_samples: rung_samples,
            root_seed_base: hero_deal_seed,
            opponent_seed_base: opponent_seed,
        };
        let started = Instant::now();
        let batch = run(grid, &teacher, hero_deal_seed)?;
        let seconds = started.elapsed().as_secs_f64();

        let mut agree = 0usize;
        let mut regret_sum = 0.0f64;
        let mut worst = 0.0f64;
        for (rung_root, reference_root) in batch.iter().zip(reference.iter()) {
            let picked = rung_root.best_candidate;
            let truth = reference_root.best_candidate;
            if picked == truth {
                agree += 1;
            }
            // Regret is priced on the REFERENCE's scale: what the rung's pick is
            // worth there, against the best available there.
            let regret = reference_root.candidates[truth].expected_value
                - reference_root.candidates[picked].expected_value;
            regret_sum += regret;
            if regret > worst {
                worst = regret;
            }
        }
        let count = batch.len().max(1) as f64;
        rows.push(json!({
            "grid": format!("{}x{}x{}@{}", grid.0, grid.1, grid.2, rung_samples),
            "t2_draws": grid.0, "t3_draws": grid.1, "t4_draws": grid.2,
            "opponent_samples": rung_samples,
            "mean_usable_opponents": batch.iter().map(|r| r.mean_usable_opponents).sum::<f64>()
                / count,
            "argmax_agreement": agree as f64 / count,
            "mean_regret_vs_reference": regret_sum / count,
            "max_regret_vs_reference": worst,
            "mean_label_standard_error": batch.iter()
                .map(|r| r.candidates.iter().map(|c| c.standard_error).sum::<f64>()
                     / r.candidates.len() as f64)
                .sum::<f64>() / count,
            "seconds": seconds,
            "core_seconds_per_root": seconds * threads as f64 / count,
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "roots": reference.len(),
            "opponent_samples": samples,
            "reference": {
                "grid": format!("{}x{}x{}", reference_grid.0, reference_grid.1, reference_grid.2),
                "opponent_samples": reference_samples,
                "hero_deal_seed": reference_hero_deal_seed,
                "independent_of_rungs": reference_hero_deal_seed != hero_deal_seed,
                "seconds": reference_seconds,
                "core_seconds_per_root": reference_seconds * threads as f64
                    / reference.len().max(1) as f64,
            },
            "continuation": reference.first().map(|r| r.continuation.clone()),
            "threads": threads,
            "rows": rows,
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn command_probe_t3_n(arguments: &[String]) -> Result<(), String> {
    let (common, extra) = parse_common(arguments)?;
    common.install_thread_pool();
    let objective = common.build_objective()?;
    let roots_path = extra
        .get("roots-file")
        .ok_or_else(|| "probe-t3-n requires --roots-file".to_owned())?;
    let opponent_seed = extra_u64(&extra, "opponent-seed", 997_600_000)?;
    let reference_opponent_seed = extra_u64(&extra, "reference-opponent-seed", 997_650_000)?;
    let hero_deal_seed = extra_u64(&extra, "hero-deal-seed", 997_700_000)?;
    // `--reference-samples` is consumed by `parse_common` into `CommonArgs` and
    // never reaches `extra`, so reading it from there silently substituted this
    // function's default and the flag lied to every caller that passed it. The
    // published T3 N-probe used the default, so its numbers are what they say
    // they are -- but the next caller to set the flag would have been told a
    // sample size the run never used.
    let reference_samples = common.reference_samples;
    let limit = extra_usize(&extra, "limit", 100)?;
    let ladder: Vec<usize> = match extra.get("ladder") {
        Some(text) => text
            .split(',')
            .map(|value| value.trim().parse::<usize>().map_err(|_| "bad --ladder"))
            .collect::<Result<Vec<_>, _>>()?,
        None => vec![100, 200, 400, 800, 1600],
    };

    // Same yardstick rule as the T1 grid: a reference that is not comfortably
    // deeper than the rungs is measuring itself. The ratio is stated in raw
    // opponent samples rather than effective ones because collision survival is
    // a property of the street and cancels -- at T3 it is 23.99% for both.
    let largest_rung = ladder.iter().copied().max().unwrap_or(0);
    if reference_samples < largest_rung.saturating_mul(4) {
        return Err(format!(
            "reference uses {reference_samples} opponent samples but the largest \
             ladder rung uses {largest_rung}; pass --reference-samples at least \
             4x the largest rung so the yardstick is not the thing being measured"
        ));
    }

    let mut roots = read_roots(roots_path)?;
    roots.truncate(limit);

    let reference_teacher = TeacherConfig {
        opponent_samples: reference_samples,
        root_seed_base: hero_deal_seed,
        opponent_seed_base: reference_opponent_seed,
    };
    let reference_started = Instant::now();
    let reference = label_t3_batch(
        &roots,
        &objective,
        &reference_teacher,
        0,
        hero_deal_seed,
        "probe-reference",
        false,
    );
    let reference_seconds = reference_started.elapsed().as_secs_f64();
    let threads = rayon::current_num_threads();

    let mut rows = Vec::new();
    for rung in ladder.iter().copied() {
        let teacher = TeacherConfig {
            opponent_samples: rung,
            root_seed_base: hero_deal_seed,
            opponent_seed_base: opponent_seed,
        };
        let started = Instant::now();
        let labels = label_t3_batch(&roots, &objective, &teacher, 0, hero_deal_seed, "probe", false);
        let elapsed = started.elapsed();

        let mut optimal = 0_usize;
        let mut regret_total = 0.0_f64;
        let mut decisive = 0_usize;
        let mut decisive_optimal = 0_usize;
        let mut decisive_regret = 0.0_f64;
        let mut usable_total = 0.0_f64;
        // Spread of a candidate's own value across hero deals, divided by the
        // usable opponent count: the label's residual standard error now that
        // the deal expectation is exact.
        let mut label_error_total = 0.0_f64;
        for (candidate_label, target) in labels.iter().zip(reference.iter()) {
            let picked = candidate_label.best_candidate;
            let gap = target.candidates[target.best_candidate].expected_value
                - target.candidates[picked].expected_value;
            regret_total += gap;
            if gap <= 1e-9 {
                optimal += 1;
            }
            usable_total += candidate_label.mean_usable_opponents;
            let spread = candidate_label
                .candidates
                .iter()
                .map(|c| (c.expected_value - target.candidates[0].expected_value).abs())
                .fold(0.0_f64, f64::max);
            label_error_total += spread / (candidate_label.mean_usable_opponents.sqrt()).max(1.0);
            if target.decision_margin >= 1e-9 {
                decisive += 1;
                decisive_regret += gap;
                if gap <= 1e-9 {
                    decisive_optimal += 1;
                }
            }
        }
        let denominator = labels.len().max(1) as f64;
        rows.push(json!({
            "opponent_samples": rung,
            "roots": labels.len(),
            "mean_usable_opponents": usable_total / denominator,
            "value_optimal_vs_reference": optimal as f64 / denominator,
            "mean_regret": regret_total / denominator,
            "decisive_roots": decisive,
            "value_optimal_on_decisive": decisive_optimal as f64 / decisive.max(1) as f64,
            "mean_regret_on_decisive": decisive_regret / decisive.max(1) as f64,
            "label_error_proxy": label_error_total / denominator,
            "seconds": elapsed.as_secs_f64(),
            "core_seconds_per_root": elapsed.as_secs_f64() * threads as f64 / denominator,
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "objective": objective.identity(),
            "roots": roots.len(),
            "deal_mode": "exhaustive",
            "reference_opponent_samples": reference_samples,
            "reference_core_seconds_per_root":
                reference_seconds * threads as f64 / roots.len().max(1) as f64,
            "reference_zero_margin_share": reference
                .iter()
                .filter(|record| record.decision_margin < 1e-9)
                .count() as f64
                / reference.len().max(1) as f64,
            "threads": threads,
            "seed_bases": {
                "opponent": opponent_seed,
                "reference_opponent": reference_opponent_seed,
                "hero_deal": hero_deal_seed,
            },
            "ladder": rows,
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn command_probe(arguments: &[String]) -> Result<(), String> {
    let (common, extra) = parse_common(arguments)?;
    common.install_thread_pool();
    let objective = common.build_objective()?;
    let behavior = behavior_config(&extra, objective.fl_ev_stay)?;
    let roots = extra_usize(&extra, "roots", 200)?;
    let root_seed = extra_u64(&extra, "root-seed", 997_000_000)?;
    let opponent_seed = extra_u64(&extra, "opponent-seed", 997_500_000)?;
    let reference_samples = extra_usize(&extra, "reference-samples-opp", 4_000)?;
    let ladder: Vec<usize> = match extra.get("ladder") {
        Some(text) => text
            .split(',')
            .map(|value| value.trim().parse::<usize>().map_err(|_| "bad --ladder"))
            .collect::<Result<Vec<_>, _>>()?,
        None => vec![25, 50, 100, 200, 400, 800],
    };

    // Roots are generated once and reused at every ladder rung, so the ladder
    // measures sampling error only.
    let generated: Vec<_> = (0..roots as u64)
        .into_par_iter()
        .map(|root_index| generate_root(root_seed, root_index, &behavior))
        .collect();

    // High-sample reference labels: the argmax these produce is the target the
    // ladder rungs are scored against.
    // The reference must draw different opponent hands than any ladder rung,
    // or a rung would be scored partly against its own samples. The offset is
    // small enough to stay inside this workstream's 997-million seed block.
    let reference_teacher = TeacherConfig {
        opponent_samples: reference_samples,
        root_seed_base: root_seed,
        opponent_seed_base: opponent_seed + 400_000,
    };
    let reference: Vec<_> = generated
        .par_iter()
        .map(|root| {
            let mut solver = FlSolver::new(&objective).expect("valid objective");
            label_root(
                root,
                &objective,
                &reference_teacher,
                &behavior.identity(),
                &mut solver,
            )
            .expect("labelled")
        })
        .collect();

    let mut rows = Vec::new();
    for rung in ladder.iter().copied() {
        let teacher = TeacherConfig {
            opponent_samples: rung,
            root_seed_base: root_seed,
            opponent_seed_base: opponent_seed,
        };
        let started = Instant::now();
        let labels: Vec<_> = generated
            .par_iter()
            .map(|root| {
                let mut solver = FlSolver::new(&objective).expect("valid objective");
                label_root(root, &objective, &teacher, &behavior.identity(), &mut solver)
                    .expect("labelled")
            })
            .collect();
        let elapsed = started.elapsed();

        let mut top1 = 0_usize;
        let mut regret_total = 0.0_f64;
        let mut standard_error_total = 0.0_f64;
        let mut margin_error_total = 0.0_f64;
        let mut multi_candidate = 0_usize;
        let mut zero_margin = 0_usize;
        let mut decisive = 0_usize;
        let mut decisive_top1 = 0_usize;
        for (candidate_label, target) in labels.iter().zip(reference.iter()) {
            if candidate_label.candidates.len() < 2 {
                continue;
            }
            multi_candidate += 1;
            if candidate_label.decision_margin < 1e-9 {
                zero_margin += 1;
            }
            // Roots the reference says are genuinely decidable. A rung that
            // ties everything scores a free top-1 on the rest, so the honest
            // question is whether it agrees where there is something to agree
            // about.
            if target.decision_margin >= 1e-9 {
                decisive += 1;
                if candidate_label.best_candidate == target.best_candidate {
                    decisive_top1 += 1;
                }
            }
            let picked = candidate_label.best_candidate;
            if picked == target.best_candidate {
                top1 += 1;
            }
            // Regret is measured on the reference labels: what the rung's pick
            // actually costs, not what the rung thinks it costs.
            regret_total += target.candidates[target.best_candidate].expected_value
                - target.candidates[picked].expected_value;
            standard_error_total += candidate_label.candidates[picked].standard_error;
            margin_error_total +=
                (candidate_label.decision_margin - target.decision_margin).abs();
        }
        let denominator = multi_candidate.max(1) as f64;
        rows.push(json!({
            "opponent_samples": rung,
            "roots_scored": multi_candidate,
            "top1_vs_reference": top1 as f64 / denominator,
            "mean_regret": regret_total / denominator,
            "mean_label_standard_error": standard_error_total / denominator,
            "mean_margin_error": margin_error_total / denominator,
            "zero_margin_share": zero_margin as f64 / denominator,
            "decisive_roots": decisive,
            "top1_on_decisive_roots": decisive_top1 as f64 / decisive.max(1) as f64,
            "seconds": elapsed.as_secs_f64(),
            "roots_per_second": roots as f64 / elapsed.as_secs_f64(),
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "objective": objective.identity(),
            "behavior_policy": behavior.identity(),
            "roots": roots,
            "reference_opponent_samples": reference_samples,
            "reference_zero_margin_share": reference
                .iter()
                .filter(|record| record.decision_margin < 1e-9)
                .count() as f64
                / reference.len().max(1) as f64,
            "candidate_counts": {
                "min": reference.iter().map(|r| r.candidates.len()).min(),
                "max": reference.iter().map(|r| r.candidates.len()).max(),
                "mean": reference.iter().map(|r| r.candidates.len()).sum::<usize>() as f64
                    / reference.len().max(1) as f64,
            },
            "ladder": rows,
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}
