use self_play::{GameState, PlayerBoard, Turn, Row};
use self_play::inference::{InferenceClient, InferenceRequest};
use self_play::mcts::{MCTS, Node, PlacementAction, Action as MctsAction};
use ofc_core::Card;
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;
use rand::seq::SliceRandom;
use rand::thread_rng;

#[derive(Serialize)]
struct TrainingSample {
    model: String,
    turn: String,
    position: String,
    board: String,
    hand: String,
    opp_top: String,
    opp_mid: String,
    opp_bot: String,
    dead_cards: String,
    placements: Vec<PlacementData>,
}

#[derive(Serialize)]
struct PlacementData {
    d: String,
    p: String,
    visit_prob: f64,
}

fn card_to_string(c: &Card) -> String {
    if c.rank == 0 {
        if c.suit == 4 { return "X1".to_string(); }
        return "X2".to_string(); // fallback
    }
    let ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A'];
    let suits = ['s','h','d','c'];
    format!("{}{}", ranks[(c.rank - 2) as usize], suits[c.suit as usize])
}

fn board_to_string(b: &PlayerBoard) -> String {
    let top = b.top.iter().map(|c| card_to_string(c)).collect::<Vec<_>>().join(" ");
    let mid = b.mid.iter().map(|c| card_to_string(c)).collect::<Vec<_>>().join(" ");
    let bot = b.bot.iter().map(|c| card_to_string(c)).collect::<Vec<_>>().join(" ");
    format!("Top[{}] Mid[{}] Bot[{}]", top, mid, bot)
}

fn cards_to_string(cards: &[Card]) -> String {
    cards.iter().map(|c| card_to_string(c)).collect::<Vec<_>>().join(" ")
}

fn build_deck() -> Vec<Card> {
    let mut deck = Vec::new();
    for suit in 0..4 {
        for rank in 2..=14 {
            deck.push(Card { rank, suit });
        }
    }
    deck.push(Card { rank: 0, suit: 4 }); // X1
    deck.push(Card { rank: 0, suit: 5 }); // X2
    deck
}

fn run_mcts(
    state: &GameState,
    hand: &[Card],
    client: &mut InferenceClient,
    simulations: usize,
) -> Vec<(PlacementAction, f64)> {
    let mut mcts = MCTS::new(state.clone(), 1.0);
    
    // Evaluate root node to expand it immediately
    expand_node(&mut mcts, 0, hand, client);

    for _ in 0..simulations {
        let mut node_idx = 0;
        let mut path = vec![node_idx];
        
        // Selection
        while mcts.nodes[node_idx].is_expanded() {
            node_idx = mcts.select_child(node_idx);
            path.push(node_idx);
        }
        
        let node_state = mcts.nodes[node_idx].state.clone();
        
        // Check if terminal
        if node_state.turn == Turn::Showdown {
            // Need perspective: calculate_score returns P1's score. 
            // If the node we just reached is P2's turn, it means P1 just acted.
            // MCTS backprop will immediately flip this if parent is P1, so we should return EV from the perspective
            // of the player whose turn it currently is.
            let p1_score = GameState::calculate_reward(&node_state.p1_board, &node_state.p2_board);
            let score = if node_state.current_player == 1 { p1_score } else { -p1_score };
            mcts.backpropagate(node_idx, score);
            continue;
        }

        // Determinization: Sample a random hand for the current player of this node
        let mut deck = build_deck();
        let mut remove_card = |c: &Card| {
            if let Some(pos) = deck.iter().position(|x| x == c) {
                deck.remove(pos);
            }
        };
        for c in &node_state.p1_board.top { remove_card(c); }
        for c in &node_state.p1_board.mid { remove_card(c); }
        for c in &node_state.p1_board.bot { remove_card(c); }
        for c in &node_state.p1_board.discards { remove_card(c); }
        for c in &node_state.p2_board.top { remove_card(c); }
        for c in &node_state.p2_board.mid { remove_card(c); }
        for c in &node_state.p2_board.bot { remove_card(c); }
        for c in &node_state.p2_board.discards { remove_card(c); }
        // Also remove the root hand if we are at root (though if node_idx==0, it's already expanded, so this only matters if we sample root, but we don't)
        // Wait, if node_idx is not 0, the root hand was placed in the board, so it's already removed!
        // But what about the cards currently in the opponent's hand? We don't know them, so they are in the deck!
        // This is correct (Information Set MCTS).
        deck.shuffle(&mut thread_rng());
        
        let n_cards = if node_state.turn == Turn::T0 { 5 } else { 3 };
        let sampled_hand: Vec<Card> = deck.into_iter().take(n_cards).collect();

        // Expansion & Evaluation
        let ev = expand_node(&mut mcts, node_idx, &sampled_hand, client);
        mcts.backpropagate(node_idx, ev);
    }
    
    // Collect root children visit counts
    let root = &mcts.nodes[0];
    let total_visits: u32 = root.children.iter().map(|&c| mcts.nodes[c].visits).sum();
    
    let mut results = Vec::new();
    for &child_idx in &root.children {
        let child = &mcts.nodes[child_idx];
        let prob = if total_visits > 0 { child.visits as f64 / total_visits as f64 } else { 0.0 };
        results.push((child.action_taken.as_ref().unwrap().placement.clone(), prob));
    }
    
    results
}

fn expand_node(mcts: &mut MCTS, node_idx: usize, hand: &[Card], client: &mut InferenceClient) -> f64 {
    let state = &mcts.nodes[node_idx].state;
    if state.turn == Turn::Showdown {
        return GameState::calculate_reward(&state.p1_board, &state.p2_board);
    }
    
    let legal_actions = state.get_legal_actions(hand);
    
    let (my_board, opp_board) = if state.current_player == 1 {
        (&state.p1_board, &state.p2_board)
    } else {
        (&state.p2_board, &state.p1_board)
    };
    
    let model_name = match state.turn {
        Turn::T0 => if state.current_player == 1 { "t0_bb" } else { "t0_btn" },
        Turn::T1 => if state.current_player == 1 { "t1_bb" } else { "t1_btn" },
        Turn::T2 => if state.current_player == 1 { "t2_bb" } else { "t2_btn" },
        Turn::T3 => if state.current_player == 1 { "t3_bb" } else { "t3_btn" },
        Turn::T4 => if state.current_player == 1 { "t4_bb" } else { "t4_btn" },
        _ => "t1_bb",
    };
    
    let req = InferenceRequest {
        model: model_name.to_string(),
        board: board_to_string(my_board),
        hand: cards_to_string(hand),
        opp_top: cards_to_string(&opp_board.top),
        opp_mid: cards_to_string(&opp_board.mid),
        opp_bot: cards_to_string(&opp_board.bot),
        dead_cards: "".to_string(),
    };
    
    let resp = client.predict(&req).unwrap();
    
    let mut actions = Vec::new();
    let mut sum_prob = 0.0;
    
    for action in legal_actions {
        let mut joint_prob = 1.0;
        
        // Calculate joint probability based on NN output
        if hand.len() == resp.probs.len() && (hand.len() == 3 || hand.len() == 5) {
            // Mapping: Top=0, Mid=1, Bot=2, Discard=3
            for (i, &card) in hand.iter().enumerate() {
                let mut dest_class = 3; // default discard
                if action.discard == Some(card) {
                    dest_class = 3;
                } else {
                    for &(placed_card, dest) in &action.cards {
                        if placed_card == card {
                            dest_class = match dest {
                                Row::Top => 0,
                                Row::Middle => 1,
                                Row::Bottom => 2,
                            };
                            break;
                        }
                    }
                }
                joint_prob *= resp.probs[i][dest_class];
            }
        } else {
            // fallback (if unhandled)
            joint_prob = 1.0;
        }
        
        actions.push(MctsAction {
            placement: action,
            prior_prob: joint_prob,
        });
        sum_prob += joint_prob;
    }
    
    // Normalize priors
    if sum_prob > 0.0 {
        for a in &mut actions {
            a.prior_prob /= sum_prob;
        }
    }
    
    mcts.expand(node_idx, actions);
    resp.ev
}

fn format_placement(placement: &PlacementAction) -> (String, String) {
    let d = match placement.discard {
        Some(c) => card_to_string(&c),
        None => "".to_string(),
    };
    let mut p_parts = Vec::new();
    for &(c, dest) in &placement.cards {
        let dest_str = match dest {
            Row::Top => "Top",
            Row::Middle => "Middle",
            Row::Bottom => "Bottom",
        };
        p_parts.push(format!("{}->{}", card_to_string(&c), dest_str));
    }
    (d, p_parts.join(", "))
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let num_games = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(1)
    } else {
        1
    };
    
    println!("Starting Self-Play Worker... (Generating {} games)", num_games);
    let mut client = InferenceClient::new("127.0.0.1:5555")?;
    
    let mut file = OpenOptions::new().create(true).append(true).open("self_play_data.jsonl")?;
    
    let mut total_score_p1 = 0.0;
    let mut p1_fl_count = 0;
    let mut p2_fl_count = 0;
    let mut p1_fl_qq = 0;
    let mut p1_fl_kk = 0;
    let mut p1_fl_aa = 0;
    let mut p1_fl_trips = 0;
    let mut p2_fl_qq = 0;
    let mut p2_fl_kk = 0;
    let mut p2_fl_aa = 0;
    let mut p2_fl_trips = 0;
    let mut p1_bust_count = 0;
    let mut p2_bust_count = 0;
    let mut p1_royalty_total = 0;
    let mut p2_royalty_total = 0;
    
    for game_idx in 0..num_games {
        if game_idx % 10 == 0 {
            println!("Playing game {}/{}", game_idx, num_games);
        }
        
        let mut deck = build_deck();
        deck.shuffle(&mut thread_rng());
        
        let mut state = GameState::new(deck.clone());
        let mut deck_idx = 0;
        
        // Game Loop
        while state.turn != Turn::Showdown {
            let n_cards = if state.turn == Turn::T0 { 5 } else { 3 };
            
            let hand: Vec<Card> = (0..n_cards).map(|_| {
                let c = deck[deck_idx];
                deck_idx += 1;
                c
            }).collect();
            
            let turn_str = format!("{:?}", state.turn);
            let pos_str = if state.current_player == 1 { "BB" } else { "BTN" };
            let model_str = format!("{}_{}", turn_str.to_lowercase(), pos_str.to_lowercase());
            
            let (my_board, opp_board) = if state.current_player == 1 {
                (&state.p1_board, &state.p2_board)
            } else {
                (&state.p2_board, &state.p1_board)
            };
            
            // Run MCTS with actual simulations
            // GCP deployment: 300 simulations for deeper search
            let simulations = 300;
            let mcts_results = run_mcts(&state, &hand, &mut client, simulations);
            
            let mut placements = Vec::new();
            for (action, prob) in &mcts_results {
                let (d, p) = format_placement(action);
                placements.push(PlacementData {
                    d,
                    p,
                    visit_prob: *prob,
                });
            }
            
            let sample = TrainingSample {
                model: model_str,
                turn: turn_str,
                position: pos_str.to_string(),
                board: board_to_string(my_board),
                hand: cards_to_string(&hand),
                opp_top: cards_to_string(&opp_board.top),
                opp_mid: cards_to_string(&opp_board.mid),
                opp_bot: cards_to_string(&opp_board.bot),
                dead_cards: "".to_string(),
                placements,
            };
            
            file.write_all(serde_json::to_string(&sample)?.as_bytes())?;
            file.write_all(b"\n")?;
            
            // Choose the best action based on MCTS visit counts
            let best_action = mcts_results.into_iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
            
            // We need to convert PlacementAction back to Action format expected by apply_placement
            let mut apply_action = self_play::mcts::Action {
                placement: best_action,
                prior_prob: 1.0, // Not strictly needed for apply
            };
            state.apply_placement(&apply_action);
        }
        
        let score = GameState::calculate_reward(&state.p1_board, &state.p2_board);
        total_score_p1 += score;
        
        if state.p1_board.busted { p1_bust_count += 1; }
        if state.p2_board.busted { p2_bust_count += 1; }
        
        let p1_royalty = state.p1_board.total_royalty();
        let p2_royalty = state.p2_board.total_royalty();
        p1_royalty_total += p1_royalty;
        p2_royalty_total += p2_royalty;
        
        // FL tracking
        let p1_top_royalty = ofc_core::get_top_royalty(&state.p1_board.top);
        if !state.p1_board.busted && p1_top_royalty >= 7 { 
            p1_fl_count += 1;
            match p1_top_royalty {
                7 => p1_fl_qq += 1,
                8 => p1_fl_kk += 1,
                9 => p1_fl_aa += 1,
                _ => p1_fl_trips += 1,
            }
        }
        
        let p2_top_royalty = ofc_core::get_top_royalty(&state.p2_board.top);
        if !state.p2_board.busted && p2_top_royalty >= 7 { 
            p2_fl_count += 1; 
            match p2_top_royalty {
                7 => p2_fl_qq += 1,
                8 => p2_fl_kk += 1,
                9 => p2_fl_aa += 1,
                _ => p2_fl_trips += 1,
            }
        }
    }
    
    let n_f64 = num_games as f64;
    println!("Finished generating {} games.", num_games);
    println!("STATS|P1_Score:{:.2}|P1_Bust:{:.1}%|P2_Bust:{:.1}%|P1_FL:{:.1}%|P2_FL:{:.1}%|P1_Royalty:{:.2}|P2_Royalty:{:.2}|P1_QQ:{}|P1_KK:{}|P1_AA:{}|P1_Trips:{}|P2_QQ:{}|P2_KK:{}|P2_AA:{}|P2_Trips:{}",
        total_score_p1 / n_f64,
        (p1_bust_count as f64 / n_f64) * 100.0,
        (p2_bust_count as f64 / n_f64) * 100.0,
        (p1_fl_count as f64 / n_f64) * 100.0,
        (p2_fl_count as f64 / n_f64) * 100.0,
        p1_royalty_total as f64 / n_f64,
        p2_royalty_total as f64 / n_f64,
        p1_fl_qq, p1_fl_kk, p1_fl_aa, p1_fl_trips,
        p2_fl_qq, p2_fl_kk, p2_fl_aa, p2_fl_trips
    );
    
    Ok(())
}
