use clap::Parser;
use mcts_gen::eval::compute_score;
use mcts_gen::inference::{Evaluator, PolicyValueSession};
use mcts_gen::mcts::{Action, Edge, IsMcts};
use mcts_gen::state::GameState;
use rand::Rng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

#[derive(serde::Serialize, Clone)]
struct DumpAction {
    to_top: u64,
    to_middle: u64,
    to_bottom: u64,
    discards: u64,
    visits: u32,
}

#[derive(serde::Serialize, Clone)]
struct DumpState {
    turn: u8,
    is_p1_turn: bool,
    p1_top: u64,
    p1_middle: u64,
    p1_bottom: u64,
    p1_discards: u64,
    p2_top: u64,
    p2_middle: u64,
    p2_bottom: u64,
    p2_discards: u64,
    current_hand: u64,
    actions: Vec<DumpAction>,
    z: f64, // outcome
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "models/model.onnx")]
    model_path: String,

    #[arg(long, default_value_t = 100)]
    games: usize,

    #[arg(long, default_value_t = 400)]
    simulations: usize,

    #[arg(long, default_value_t = 1.5)]
    c_puct: f64,

    #[arg(long, default_value_t = 2.5)]
    pw_c: f64,

    #[arg(long, default_value_t = 0.5)]
    pw_alpha: f64,

    #[arg(long, default_value = "output")]
    output_dir: String,

    #[arg(long, default_value_t = 8)]
    threads: usize,
}

fn select_action_with_temperature(policy: &[(Action, u32)], turn: u8, rng: &mut impl Rng) -> Action {
    if policy.is_empty() {
        panic!("Empty policy!");
    }
    // Turn 0 is initial deal (13 cards). Subsequent are turn 1, 2, 3, 4 (3 cards each).
    // Let's sample proportionally for first 2 decision turns (e.g., turn <= 1).
    if turn <= 1 {
        let total: u32 = policy.iter().map(|(_, v)| *v).sum();
        if total == 0 {
            return policy[0].0;
        }
        let mut sample = rng.gen_range(0..total);
        for (action, visits) in policy {
            if sample < *visits {
                return *action;
            }
            sample -= *visits;
        }
        policy.last().unwrap().0
    } else {
        let mut best_action = policy[0].0;
        let mut max_visits = 0;
        for (action, visits) in policy {
            if *visits > max_visits {
                max_visits = *visits;
                best_action = *action;
            }
        }
        best_action
    }
}

fn main() {
    let args = Args::parse();
    
    // Set up thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap();

    std::fs::create_dir_all(&args.output_dir).expect("Failed to create output dir");
    
    // Create a unique file name using thread rng
    let random_id: u64 = rand::random();
    let file_path = format!("{}/replays_{:x}.jsonl", args.output_dir, random_id);
    let output_file = File::create(&file_path).expect("Failed to create output file");
    let writer = Mutex::new(BufWriter::new(output_file));
    
    let completed_games = AtomicUsize::new(0);
    let start_time = Instant::now();

    println!("Starting self-play generation...");
    println!("Games: {}, Threads: {}, Model: {}", args.games, args.threads, args.model_path);

    (0..args.games).into_par_iter().for_each(|_game_idx| {
        // Create a local ONNX evaluator per thread
        let mut evaluator = match PolicyValueSession::new(&args.model_path) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Failed to load ONNX model: {:?}", e);
                return;
            }
        };

        let mut rng = rand::thread_rng();
        let mut state = GameState::initial();
        state.deal_cards_with_rng(&mut rng); // Initial deal

        let mut history: Vec<DumpState> = Vec::new();

        while !state.is_terminal() {
            if state.current_hand.is_empty() {
                state.deal_cards_with_rng(&mut rng);
                continue;
            }

            let mut mcts = IsMcts::new();
            mcts.c_puct = args.c_puct;
            mcts.progressive_widening_c = args.pw_c;
            mcts.progressive_widening_alpha = args.pw_alpha;

            // Run search
            mcts.search(&state, args.simulations, &mut evaluator);

            // Extract policy
            let mut policy = Vec::new();
            let mut dump_actions = Vec::new();
            for action in &mcts.root.valid_actions {
                let visits = if let Some(child) = mcts.root.children.get(&Edge::Action(*action)) {
                    child.visits
                } else {
                    0
                };
                policy.push((*action, visits));
                dump_actions.push(DumpAction {
                    to_top: action.to_top.0,
                    to_middle: action.to_middle.0,
                    to_bottom: action.to_bottom.0,
                    discards: action.discards.0,
                    visits,
                });
            }

            // Save state and policy
            let dump_state = DumpState {
                turn: state.turn,
                is_p1_turn: state.is_p1_turn,
                p1_top: state.p1_board.top.0,
                p1_middle: state.p1_board.middle.0,
                p1_bottom: state.p1_board.bottom.0,
                p1_discards: state.p1_board.discards.0,
                p2_top: state.p2_board.top.0,
                p2_middle: state.p2_board.middle.0,
                p2_bottom: state.p2_board.bottom.0,
                p2_discards: state.p2_board.discards.0,
                current_hand: state.current_hand.0,
                actions: dump_actions,
                z: 0.0, // to be updated at the end
            };
            history.push(dump_state);

            // Select action
            let selected_action = select_action_with_temperature(&policy, state.turn, &mut rng);
            state.apply_action(selected_action);
        }

        // Terminal, compute score
        let score = compute_score(
            state.p1_board.top, state.p1_board.middle, state.p1_board.bottom,
            state.p2_board.top, state.p2_board.middle, state.p2_board.bottom
        );
        let z_p1 = score as f64;

        // Update z values and write to file
        let mut local_jsonl = String::new();
        for mut dump in history {
            // z from the perspective of the player whose turn it is
            dump.z = if dump.is_p1_turn { z_p1 } else { -z_p1 };
            
            if let Ok(json) = serde_json::to_string(&dump) {
                local_jsonl.push_str(&json);
                local_jsonl.push('\n');
            }
        }

        {
            let mut w = writer.lock().unwrap();
            w.write_all(local_jsonl.as_bytes()).unwrap();
        }

        let completed = completed_games.fetch_add(1, Ordering::Relaxed) + 1;
        if completed % 10 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let games_per_sec = completed as f64 / elapsed;
            println!("Completed: {}/{} games | {:.2} games/sec", completed, args.games, games_per_sec);
        }
    });

    println!("Self-play complete! Total time: {:.2}s", start_time.elapsed().as_secs_f64());
}
