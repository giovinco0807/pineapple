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
        Commands::T0Eval { hand, samples, seed, top_n, nesting } => {
            let parts: Vec<usize> = nesting.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            let nest = if parts.len() == 3 { [parts[0], parts[1], parts[2]] } else { [3, 2, 1] };
            run_t0_eval(&hand, samples, seed, top_n, nest);
        }
        Commands::T0Batch { hands, samples, output, seed, nesting, top_k } => {
            let parts: Vec<usize> = nesting.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            if parts.len() != 3 {
                eprintln!("Error: --nesting must be 3 comma-separated values (e.g. 5,3,2)");
                std::process::exit(1);
            }
            let nest = [parts[0], parts[1], parts[2]];
            t0_eval::run_batch(hands, samples, &output, seed, nest, top_k);
        }
        Commands::T0BatchFiltered { input, samples, output, seed, nesting } => {
            let parts: Vec<usize> = nesting.split(',').filter_map(|s| s.trim().parse().ok()).collect();
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

fn run_t0_eval(hand_str: &str, samples: usize, seed: u64, top_n: usize, nesting: [usize; 3]) {
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
    let start = Instant::now();

    let results = t0_eval::evaluate_t0(&hand, samples, seed, nesting);

    let elapsed = start.elapsed().as_secs_f64();

    let show = if top_n == 0 || top_n >= results.len() { results.len() } else { top_n };

    println!("=== Results (top {} of {}) ===", show, results.len());
    println!("{:<4} {:<40} {:>10}", "Rank", "Placement", "EV");
    println!("{}", "-".repeat(56));

    for (i, (desc, ev)) in results.iter().take(show).enumerate() {
        println!("{:<4} {:<40} {:>+10.3}", i + 1, desc, ev);
    }

    if results.len() > show {
        println!("  ... and {} more placements", results.len() - show);
    }

    println!();
    println!("Best:  {} → EV: {:+.3}", results[0].0, results[0].1);
    println!("Worst: {} → EV: {:+.3}", results.last().unwrap().0, results.last().unwrap().1);
    println!("Total time: {:.1}s", elapsed);
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
