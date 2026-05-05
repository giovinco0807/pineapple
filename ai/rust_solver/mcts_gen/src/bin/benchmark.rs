use clap::Parser;
use mcts_gen::mcts::IsMcts;
use mcts_gen::state::GameState;
use mcts_gen::inference::{Evaluator, PolicyValueSession};
use mcts_gen::action_gen::{get_initial_actions, get_turn_actions};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "1000")]
    num_iterations: usize,

    #[arg(short, long)]
    model_path: Option<String>,
}

struct DummyEvaluator;

impl Evaluator for DummyEvaluator {
    fn evaluate(&mut self, state: &GameState) -> Option<(Vec<f32>, f32)> {
        let legal_actions = if state.turn == 0 {
            get_initial_actions(state.current_hand, if state.is_p1_turn { &state.p1_board } else { &state.p2_board })
        } else {
            get_turn_actions(state.current_hand, if state.is_p1_turn { &state.p1_board } else { &state.p2_board })
        };
        
        if legal_actions.is_empty() {
            return None;
        }
        let prob = 1.0 / legal_actions.len() as f32;
        let policy = vec![prob; legal_actions.len()];
        Some((policy, 0.0))
    }
}

fn run_benchmark<F, E>(evaluator_factory: F, num_iterations: usize, label: &str, threads: usize)
where
    F: Fn() -> E + Sync + Send,
    E: Evaluator,
{
    println!("Benchmarking MCTS node generation ({label}) with {threads} threads...");

    // Warm up
    let mut evaluator_warmup = evaluator_factory();
    let mut mcts_warmup = IsMcts::new();
    let mut state_warmup = GameState::initial();
    state_warmup.deal_cards();
    mcts_warmup.search(&state_warmup, 10, &mut evaluator_warmup);

    let bench_start = Instant::now();
    
    use rayon::prelude::*;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    
    pool.install(|| {
        (0..threads).into_par_iter().for_each(|_| {
            let mut evaluator = evaluator_factory();
            let mut mcts = IsMcts::new();
            let mut state = GameState::initial();
            state.deal_cards();
            mcts.search(&state, num_iterations, &mut evaluator);
        });
    });

    let duration = bench_start.elapsed();

    let total_iterations = num_iterations * threads;
    let iters_per_sec = total_iterations as f64 / duration.as_secs_f64();

    println!("Benchmark completed for {} iterations across {} threads.", total_iterations, threads);
    println!("Time taken: {:?}", duration);
    println!("Throughput: {:.2} iterations/sec", iters_per_sec);
}

fn main() {
    let args = Args::parse();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    
    if let Some(model_path) = args.model_path {
        println!("Loading ONNX model from: {}", model_path);
        // We need a factory that creates a PolicyValueSession
        // Loading the model multiple times might be slow, but it's okay for benchmark setup
        let factory = || PolicyValueSession::new(&model_path).unwrap();
        // Warm up and verify
        if let Err(e) = PolicyValueSession::new(&model_path) {
            eprintln!("Failed to load ONNX model: {:?}", e);
            return;
        }
        run_benchmark(factory, args.num_iterations, "with ONNX model", threads);
    } else {
        let factory = || DummyEvaluator;
        run_benchmark(factory, args.num_iterations, "pure throughput / dummy evaluator", threads);
    }
}
