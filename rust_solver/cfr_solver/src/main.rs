//! OFC Pineapple CFR Solver - Main entry point.
//!
//! Usage:
//!   cfr_solver train --iterations 10000 --threads 8 --output strategy.bin
//!   cfr_solver bench --iterations 100
//!   cfr_solver t0-eval --hand "Ad 8c 4s 3d 2s" --samples 10000

use clap::{Parser, Subcommand};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::time::Instant;

mod action_gen;
mod game_state;
mod info_set;
mod cfr_engine;
mod t0_eval;

use cfr_engine::{MCCFRSolver, MCCFRConfig};

#[derive(Parser)]
#[command(name = "cfr_solver", about = "OFC Pineapple MCCFR Solver")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the CFR solver
    Train {
        /// Number of iterations
        #[arg(short, long, default_value_t = 1000)]
        iterations: u64,

        /// Number of threads (for future parallel support)
        #[arg(short, long, default_value_t = 1)]
        threads: usize,

        /// Output file for strategy
        #[arg(short, long, default_value = "strategy.bin")]
        output: String,

        /// Log interval
        #[arg(long, default_value_t = 100)]
        log_interval: u64,
    },

    /// Run a quick benchmark
    Bench {
        /// Number of iterations
        #[arg(short, long, default_value_t = 10)]
        iterations: u64,
    },

    /// Evaluate T0 placements for a given hand
    T0Eval {
        /// Hand (5 cards), e.g. "Ad 8c 4s 3d 2s"
        #[arg(long)]
        hand: String,

        /// Number of Monte Carlo samples per placement
        #[arg(long, default_value_t = 100)]
        samples: usize,

        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Show top N results (0 = all)
        #[arg(long, default_value_t = 20)]
        top_n: usize,

        /// Nesting depth for nested MC (comma-separated, e.g. "3,2,1")
        #[arg(long, default_value = "3,2,1")]
        nesting: String,

        /// Top-K filtering: compare full vs top-K screening (0 = full only)
        #[arg(long, default_value_t = 0)]
        top_k: usize,
    },

    /// Batch evaluate random T0 hands and save to JSONL
    T0Batch {
        /// Number of random hands to evaluate
        #[arg(long, default_value_t = 200)]
        hands: usize,

        /// Number of Monte Carlo samples per placement
        #[arg(long, default_value_t = 30)]
        samples: usize,

        /// Output JSONL file path
        #[arg(long, default_value = "t0_train.jsonl")]
        output: String,

        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Nesting depth for nested MC (comma-separated, e.g. "5,3,2")
        #[arg(long, default_value = "5,3,2")]
        nesting: String,

        /// Top-K filtering: screen all placements, then deep-eval top K only (0 = all)
        #[arg(long, default_value_t = 0)]
        top_k: usize,
    },

    /// Filtered batch evaluation: read PolicyNet pre-filtered placements and evaluate
    T0BatchFiltered {
        /// Input JSON file from generate_filtered_t0.py
        #[arg(long)]
        input: String,

        /// Number of Monte Carlo samples per placement
        #[arg(long, default_value_t = 30)]
        samples: usize,

        /// Output JSONL file path
        #[arg(long, default_value = "t0_filtered_train.jsonl")]
        output: String,

        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Nesting depth for nested MC (comma-separated, e.g. "10,6,3")
        #[arg(long, default_value = "10,6,3")]
        nesting: String,
    },

    /// Measure FL statistics (R_N, p_FL) for corrected EV model
    FlStats {
        /// Number of random hands to evaluate
        #[arg(long, default_value_t = 30)]
        hands: usize,

        /// Number of Monte Carlo samples per placement
        #[arg(long, default_value_t = 500)]
        samples: usize,

        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Real-time turn evaluator (T1-T4)
    TurnEval {
        /// Current top row cards, e.g. "Ad"
        #[arg(long, default_value = "")]
        top: String,

        /// Current middle row cards, e.g. "4s 3d 2s"
        #[arg(long, default_value = "")]
        mid: String,

        /// Current bottom row cards, e.g. "8c"
        #[arg(long, default_value = "")]
        bot: String,

        /// The 3 cards dealt this turn, e.g. "Kh 9d 5c"
        #[arg(long)]
        hand: String,

        /// Current turn (1-4)
        #[arg(long)]
        turn: usize,

        /// Monte Carlo samples (more = more accurate, slower)
        #[arg(long, default_value_t = 5000)]
        samples: usize,

        /// Show top N results
        #[arg(long, default_value_t = 10)]
        top_n: usize,

        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Batch evaluate T1-T4 turns and save to JSONL
    TurnBatchImperfect {
        /// Input JSONL file from generate_turn_data.py
        #[arg(long, default_value = "t1_inputs.jsonl")]
        input: String,

        /// Number of Monte Carlo samples for future turns
        #[arg(long, default_value_t = 300)]
        samples: usize,

        /// Output JSONL file path
        #[arg(long, default_value = "t1_train.jsonl")]
        output: String,

        /// Nesting levels for future MC
        #[arg(long, default_value = "5,3")]
        nesting: String,

        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train { iterations, threads: _, output: _, log_interval } => {
            run_training(iterations, log_interval);
        }
        Commands::Bench { iterations } => {
            run_benchmark(iterations);
        }
        Commands::T0Eval { hand, samples, seed, top_n, nesting, top_k } => {
            let parts: Vec<usize> = nesting.split(|c| c == ',' || c == '_' || c == '-').filter_map(|s| s.trim().parse().ok()).collect();
            let nest = if parts.len() == 3 { [parts[0], parts[1], parts[2]] } else { [3, 2, 1] };
            run_t0_eval(&hand, samples, seed, top_n, nest, top_k);
        }
        Commands::T0Batch { hands, samples, output, seed, nesting, top_k } => {
            let parts: Vec<usize> = nesting.split(|c| c == ',' || c == '_' || c == '-').filter_map(|s| s.trim().parse().ok()).collect();
            if parts.len() != 3 {
                eprintln!("Error: --nesting must be 3 comma-separated values (e.g. 5,3,2)");
                std::process::exit(1);
            }
            let nest = [parts[0], parts[1], parts[2]];
            t0_eval::run_batch(hands, samples, &output, seed, nest, top_k);
        }
        Commands::T0BatchFiltered { input, samples, output, seed, nesting } => {
            let parts: Vec<usize> = nesting.split(|c| c == ',' || c == '_' || c == '-').filter_map(|s| s.trim().parse().ok()).collect();
            if parts.len() != 3 {
                eprintln!("Error: --nesting must be 3 comma-separated values (e.g. 10,6,3)");
                std::process::exit(1);
            }
            let nest = [parts[0], parts[1], parts[2]];
            t0_eval::run_batch_filtered(&input, samples, &output, seed, nest);
        }
        Commands::FlStats { hands, samples, seed } => {
            t0_eval::measure_fl_stats(hands, samples, seed);
        }
        Commands::TurnEval { top, mid, bot, hand, turn, samples, top_n, seed } => {
            run_turn_eval(&top, &mid, &bot, &hand, turn, samples, top_n, seed);
        }
        Commands::TurnBatchImperfect { input, samples, output, nesting, seed } => {
            let parts: Vec<usize> = nesting.split(|c| c == ',' || c == '_' || c == '-').filter_map(|s| s.trim().parse().ok()).collect();
            let nest = if parts.len() == 2 { [parts[0], parts[1], 1] } else { [5, 2, 1] };
            t0_eval::run_turn_batch_imperfect(&input, samples, &output, seed, &nest);
        }
    }
}

fn run_turn_eval(top: &str, mid: &str, bot: &str, hand_str: &str, turn: usize, samples: usize, top_n: usize, seed: u64) {
    if turn < 1 || turn > 4 {
        eprintln!("Error: --turn must be 1-4");
        std::process::exit(1);
    }

    let board = match t0_eval::parse_board(top, mid, bot) {
        Some(b) => b,
        None => {
            eprintln!("Error: Invalid board. Use e.g. --top \"Ad\" --mid \"4s 3d 2s\" --bot \"8c\"");
            std::process::exit(1);
        }
    };

    let hand_cards: Vec<ofc_core::Card> = hand_str.split_whitespace()
        .filter_map(|s| t0_eval::parse_card(s))
        .collect();

    if hand_cards.len() != 3 {
        eprintln!("Error: --hand must have exactly 3 cards, e.g. \"Kh 9d 5c\"");
        std::process::exit(1);
    }
    let hand = [hand_cards[0], hand_cards[1], hand_cards[2]];

    let turns_after = 4 - turn; // T1→3, T2→2, T3→1, T4→0

    // Collect all known cards
    let mut known = t0_eval::board_cards(&board);
    known.extend_from_slice(&hand);

    println!("=== OFC Pineapple Turn {} Evaluator ===", turn);
    println!("Board: Top[{}] Mid[{}] Bot[{}]", top, mid, bot);
    println!("Hand: {}", hand_str);
    println!("Turns remaining after: {}", turns_after);
    println!("Samples: {}", samples);
    println!();

    let start = Instant::now();

    let results = t0_eval::evaluate_turn(&board, &hand, &known, turns_after, samples, seed);

    let elapsed = start.elapsed().as_secs_f64();

    let show = if top_n == 0 || top_n >= results.len() { results.len() } else { top_n };

    println!("=== Results (top {} of {}) ===", show, results.len());
    println!("{:<4} {:<8} {:<35} {:>10}", "Rank", "Discard", "Placement", "EV");
    println!("{}", "-".repeat(60));

    for (i, r) in results.iter().take(show).enumerate() {
        println!("{:<4} {:<8} {:<35} {:>+10.3}", i + 1, r.discard, r.placement_desc, r.ev);
    }

    println!();
    println!("Best:  Discard {} → {} (EV: {:+.3})", results[0].discard, results[0].placement_desc, results[0].ev);
    println!("Time: {:.2}s", elapsed);
}

fn run_t0_eval(hand_str: &str, samples: usize, seed: u64, top_n: usize, nesting: [usize; 3], top_k: usize) {
    let hand = match t0_eval::parse_hand(hand_str) {
        Some(h) => h,
        None => {
            eprintln!("Error: Invalid hand format. Use e.g. \"Ad 8c 4s 3d 2s\"");
            eprintln!("  Ranks: 2-9, T, J, Q, K, A");
            eprintln!("  Suits: s(spades), h(hearts), d(diamonds), c(clubs)");
            eprintln!("  Joker: Jo");
            std::process::exit(1);
        }
    };

    println!("=== OFC Pineapple T0 EV Evaluator ===");

    // --- Full evaluation (all placements) ---
    println!("\n--- [FULL] All placements, nesting={:?}, samples={} ---", nesting, samples);
    let start_full = Instant::now();
    let results_full = t0_eval::evaluate_t0(&hand, samples, seed, nesting);
    let elapsed_full = start_full.elapsed().as_secs_f64();

    let show = if top_n == 0 || top_n >= results_full.len() { results_full.len() } else { top_n };
    println!("=== FULL Results (top {} of {}) === ({:.1}s)", show, results_full.len(), elapsed_full);
    println!("{:<4} {:<50} {:>10}", "Rank", "Placement", "EV");
    println!("{}", "-".repeat(66));
    for (i, (desc, ev)) in results_full.iter().take(show).enumerate() {
        println!("{:<4} {:<50} {:>+10.3}", i + 1, desc, ev);
    }

    // --- Top-K evaluation (2-pass) ---
    if top_k > 0 {
        println!("\n--- [TOP-K={}] 2-pass screening, nesting={:?}, samples={} ---", top_k, nesting, samples);
        let start_topk = Instant::now();
        let results_topk = t0_eval::evaluate_t0_quiet_topk(&hand, samples, seed, &nesting, top_k);
        let elapsed_topk = start_topk.elapsed().as_secs_f64();

        let show_k = if top_n == 0 || top_n >= results_topk.len() { results_topk.len() } else { top_n };
        println!("=== TOP-K Results (top {} of {}) === ({:.1}s)", show_k, results_topk.len(), elapsed_topk);
        println!("{:<4} {:<50} {:>10}  {:>10}", "Rank", "Placement", "EV", "FullRank");
        println!("{}", "-".repeat(78));

        // Build lookup from full results for comparison
        let full_rank: std::collections::HashMap<&str, usize> = results_full.iter()
            .enumerate()
            .map(|(i, (desc, _))| (desc.as_str(), i + 1))
            .collect();

        for (i, (desc, ev)) in results_topk.iter().take(show_k).enumerate() {
            let fr = full_rank.get(desc.as_str()).copied().unwrap_or(999);
            println!("{:<4} {:<50} {:>+10.3}  {:>10}", i + 1, desc, ev, fr);
        }

        // Summary: how many of full top-N are in topk top-N?
        println!("\n=== Accuracy Summary ===");
        println!("Full eval: {:.1}s | Top-K eval: {:.1}s | Speedup: {:.1}x", elapsed_full, elapsed_topk, elapsed_full / elapsed_topk);
        let full_top_set: std::collections::HashSet<&str> = results_full.iter().take(top_k).map(|(d, _)| d.as_str()).collect();
        let topk_set: std::collections::HashSet<&str> = results_topk.iter().map(|(d, _)| d.as_str()).collect();
        let overlap = full_top_set.intersection(&topk_set).count();
        println!("Full top-{} ∩ TopK result: {}/{} ({:.1}%)", top_k, overlap, top_k.min(results_full.len()), 100.0 * overlap as f64 / top_k.min(results_full.len()) as f64);

        // Check if full #1 is in top-K results
        let best_full = &results_full[0].0;
        let in_topk = results_topk.iter().any(|(d, _)| d == best_full);
        println!("Full #1 ({}) in Top-K: {}", best_full, if in_topk { "YES ✓" } else { "MISSED ✗" });

        for check in [1, 5, 10, 20] {
            if check > results_full.len() { continue; }
            let full_n: std::collections::HashSet<&str> = results_full.iter().take(check).map(|(d, _)| d.as_str()).collect();
            let topk_n: std::collections::HashSet<&str> = results_topk.iter().take(check).map(|(d, _)| d.as_str()).collect();
            let ov = full_n.intersection(&topk_n).count();
            println!("Top-{:>3} overlap: {}/{} ({:.0}%)", check, ov, check, 100.0 * ov as f64 / check as f64);
        }
    } else {
        println!("\nBest:  {} → EV: {:+.3}", results_full[0].0, results_full[0].1);
        println!("Worst: {} → EV: {:+.3}", results_full.last().unwrap().0, results_full.last().unwrap().1);
        println!("Total time: {:.1}s", elapsed_full);
    }
}

fn run_training(iterations: u64, log_interval: u64) {
    println!("=== OFC Pineapple MCCFR Training ===");
    println!("Iterations: {}", iterations);

    let config = MCCFRConfig::default();
    let mut solver = MCCFRSolver::new(config);
    let mut rng = SmallRng::from_entropy();

    let start = Instant::now();
    let mut last_log = Instant::now();

    for i in 0..iterations {
        solver.run_iteration(&mut rng);

        if (i + 1) % log_interval == 0 || i == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let iter_per_sec = (i + 1) as f64 / elapsed;
            let since_last = last_log.elapsed().as_secs_f64();

            println!(
                "[iter {:>6}] {:.1}s elapsed | {:.1} iter/s | {:.1}s/batch | nodes: {} | terminals: {} | infosets: {}",
                i + 1,
                elapsed,
                iter_per_sec,
                since_last,
                solver.nodes_visited,
                solver.terminal_reached,
                solver.store.len(),
            );
            last_log = Instant::now();
        }
    }

    let total_time = start.elapsed().as_secs_f64();
    println!("\n=== Training Complete ===");
    println!("Total time: {:.1}s", total_time);
    println!("Iterations: {}", iterations);
    println!("Avg time/iter: {:.3}s", total_time / iterations as f64);
    println!("Total nodes: {}", solver.nodes_visited);
    println!("Total terminals: {}", solver.terminal_reached);
    println!("Info sets: {}", solver.store.len());
}

fn run_benchmark(iterations: u64) {
    println!("=== Benchmark: {} iterations ===", iterations);

    let config = MCCFRConfig::default();
    let mut solver = MCCFRSolver::new(config);
    let mut rng = SmallRng::seed_from_u64(42);

    let start = Instant::now();

    for i in 0..iterations {
        let iter_start = Instant::now();
        solver.run_iteration(&mut rng);
        let iter_time = iter_start.elapsed().as_secs_f64();
        println!("  iter {}: {:.3}s | nodes: {} | terminals: {}",
            i + 1, iter_time,
            solver.nodes_visited, solver.terminal_reached,
        );
    }

    let total = start.elapsed().as_secs_f64();
    println!("\nTotal: {:.3}s | Avg: {:.3}s/iter | Info sets: {}",
        total, total / iterations as f64, solver.store.len(),
    );
}
