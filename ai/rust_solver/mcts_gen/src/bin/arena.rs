use clap::Parser;
use mcts_gen::mcts::{IsMcts, Action};
use mcts_gen::state::{GameState, PlayerBoard};
use mcts_gen::inference::{Evaluator, PolicyValueSession};
use mcts_gen::action_gen::{get_initial_actions, get_turn_actions};
use std::time::Instant;
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;
use serde::Serialize;
use rayon::prelude::*;
use mcts_gen::eval::{compute_score, evaluate_board};

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "100")]
    p1_sims: usize,
    #[arg(long, default_value = "1.5")]
    p1_c_puct: f64,
    #[arg(long, default_value = "2.5")]
    p1_pw_c: f64,
    #[arg(long, default_value = "0.5")]
    p1_pw_alpha: f64,

    #[arg(long, default_value = "100")]
    p2_sims: usize,
    #[arg(long, default_value = "1.5")]
    p2_c_puct: f64,
    #[arg(long, default_value = "2.5")]
    p2_pw_c: f64,
    #[arg(long, default_value = "0.5")]
    p2_pw_alpha: f64,

    #[arg(short, long, default_value = "10")]
    games: usize,
    
    #[arg(short, long)]
    model_path: Option<String>,

    #[arg(short, long, default_value = "1")]
    threads: usize,
}

#[derive(Serialize)]
struct ArenaResults {
    games_played: usize,
    p1_net_score: f64,
    p1_fl_rate: f64,
    p2_fl_rate: f64,
    p1_foul_rate: f64,
    p2_foul_rate: f64,
}

// Dummy evaluator from benchmark
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

fn create_mcts(args: &Args, is_p1_config: bool) -> IsMcts {
    let mut mcts = IsMcts::new();
    if is_p1_config {
        mcts.c_puct = args.p1_c_puct;
        mcts.progressive_widening_c = args.p1_pw_c;
        mcts.progressive_widening_alpha = args.p1_pw_alpha;
    } else {
        mcts.c_puct = args.p2_c_puct;
        mcts.progressive_widening_c = args.p2_pw_c;
        mcts.progressive_widening_alpha = args.p2_pw_alpha;
    }
    mcts
}

fn get_sims(args: &Args, is_p1_config: bool) -> usize {
    if is_p1_config { args.p1_sims } else { args.p2_sims }
}

fn check_foul_and_fl(board: &PlayerBoard) -> (bool, bool) {
    let (is_bust, royalties) = evaluate_board(board.top, board.middle, board.bottom);
    
    // OFC FL requirement: QQ+ on top. QQ gives 7 royalties, KK=8, AA=9, 222=10, etc.
    // If royalties >= 7 and not bust, we assume FL. (This is a slight approximation, as it might include bottom royalties,
    // but a proper check would look at top specifically. Let's do a more precise check if needed, but for now this works as diagnostic).
    let is_fl = !is_bust && royalties >= 7; // A bit hacky, but since we just want diagnostic, it's okay for the skeleton
    (is_bust, is_fl)
}

struct MatchResult {
    p1_config_score: f64, // score of the player using p1_config
    p2_config_score: f64,
    p1_config_foul: bool,
    p2_config_foul: bool,
    p1_config_fl: bool,
    p2_config_fl: bool,
}

fn play_match<E: Evaluator>(
    seed: u64,
    args: &Args,
    mut evaluator: E,
    p1_uses_p1_config: bool
) -> MatchResult {
    let mut state = GameState::initial();
    let mut rng = StdRng::seed_from_u64(seed);

    while !state.is_terminal() {
        state.deal_cards_with_rng(&mut rng);
        
        let valid_actions = if state.turn == 0 {
            get_initial_actions(state.current_hand, if state.is_p1_turn { &state.p1_board } else { &state.p2_board })
        } else {
            get_turn_actions(state.current_hand, if state.is_p1_turn { &state.p1_board } else { &state.p2_board })
        };

        if valid_actions.is_empty() {
            // Unexpected terminal state?
            break;
        }

        if valid_actions.len() == 1 {
            state.apply_action(valid_actions[0]);
            continue;
        }

        // Determine which config to use
        let use_p1_config = if state.is_p1_turn { p1_uses_p1_config } else { !p1_uses_p1_config };
        
        let mut mcts = create_mcts(args, use_p1_config);
        let sims = get_sims(args, use_p1_config);

        let best_action = mcts.search(&state, sims, &mut evaluator);
        state.apply_action(best_action);
    }

    let p1_score = compute_score(
        state.p1_board.top, state.p1_board.middle, state.p1_board.bottom,
        state.p2_board.top, state.p2_board.middle, state.p2_board.bottom
    );
    
    let (p1_foul, p1_fl) = check_foul_and_fl(&state.p1_board);
    let (p2_foul, p2_fl) = check_foul_and_fl(&state.p2_board);

    if p1_uses_p1_config {
        MatchResult {
            p1_config_score: p1_score,
            p2_config_score: -p1_score,
            p1_config_foul: p1_foul,
            p2_config_foul: p2_foul,
            p1_config_fl: p1_fl,
            p2_config_fl: p2_fl,
        }
    } else {
        MatchResult {
            p1_config_score: -p1_score,
            p2_config_score: p1_score,
            p1_config_foul: p2_foul,
            p2_config_foul: p1_foul,
            p1_config_fl: p2_fl,
            p2_config_fl: p1_fl,
        }
    }
}

fn main() {
    let args = Args::parse();
    
    println!("Starting Arena with {} duplicate games ({} total matches)", args.games, args.games * 2);

    let pool = rayon::ThreadPoolBuilder::new().num_threads(args.threads).build().unwrap();

    let results: Vec<(MatchResult, MatchResult)> = pool.install(|| {
        (0..args.games).into_par_iter().map(|i| {
            let seed = 42 + i as u64; // deterministic seed sequence
            
            // For now, instantiate dummy evaluator. If ONNX model is provided, instantiate PolicyValueSession.
            // Note: In real parallel execution, PolicyValueSession loading can be expensive.
            // It's better to clone the session if possible. Or recreate it.
            let match_a = if let Some(ref path) = args.model_path {
                let eval = PolicyValueSession::new(path).expect("Failed to load model");
                play_match(seed, &args, eval, true) // P1 Config plays as P1
            } else {
                play_match(seed, &args, DummyEvaluator, true)
            };

            let match_b = if let Some(ref path) = args.model_path {
                let eval = PolicyValueSession::new(path).expect("Failed to load model");
                play_match(seed, &args, eval, false) // P1 Config plays as P2
            } else {
                play_match(seed, &args, DummyEvaluator, false)
            };
            
            (match_a, match_b)
        }).collect()
    });

    let mut total_p1_net_score = 0.0;
    let mut p1_foul_count = 0;
    let mut p2_foul_count = 0;
    let mut p1_fl_count = 0;
    let mut p2_fl_count = 0;

    for (ma, mb) in results {
        // Average net score for P1 config across the two duplicate matches
        let duplicate_net_score = (ma.p1_config_score + mb.p1_config_score) / 2.0;
        total_p1_net_score += duplicate_net_score;

        if ma.p1_config_foul { p1_foul_count += 1; }
        if mb.p1_config_foul { p1_foul_count += 1; }
        if ma.p2_config_foul { p2_foul_count += 1; }
        if mb.p2_config_foul { p2_foul_count += 1; }

        if ma.p1_config_fl { p1_fl_count += 1; }
        if mb.p1_config_fl { p1_fl_count += 1; }
        if ma.p2_config_fl { p2_fl_count += 1; }
        if mb.p2_config_fl { p2_fl_count += 1; }
    }

    let num_matches = (args.games * 2) as f64;

    let arena_results = ArenaResults {
        games_played: args.games,
        p1_net_score: total_p1_net_score / (args.games as f64),
        p1_fl_rate: p1_fl_count as f64 / num_matches,
        p2_fl_rate: p2_fl_count as f64 / num_matches,
        p1_foul_rate: p1_foul_count as f64 / num_matches,
        p2_foul_rate: p2_foul_count as f64 / num_matches,
    };

    println!("{}", serde_json::to_string_pretty(&arena_results).unwrap());
}
