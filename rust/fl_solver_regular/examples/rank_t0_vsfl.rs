//! Rank a T0 opening fan with a vs-Fantasyland T0 model and print the top rows.
//!
//! This is the narrowing pass out of `t0_teacher::label_t0_root` and nothing
//! else: the same `generate_opening_actions` fan, the same `NodeFeatures`
//! encoding, the same `VflModel::predict`, the same descending sort with the
//! generator index as the tie-break. It runs the model, not the teacher, so it
//! answers "what does this ranker prefer" in milliseconds rather than paying
//! for a Fantasyland solve per opening.
//!
//! Usage:
//!   cargo run --release --example rank_t0_vsfl -- <model.vfl1> <sha256> "8s 4c 2s Ts 5c" [top_n]

use fl_solver_regular::behavior::{
    generate_opening_actions, PartialBoard, ROW_BOTTOM, ROW_MIDDLE, ROW_TOP,
};
use fl_solver_regular::cards::{card_to_token, parse_cards};
use fl_solver_regular::engine_features::NodeFeatures;
use ofc_hu_m3_engine::infoset::Street;
use fl_solver_regular::vfl_model::VflModel;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err(
            "usage: rank_t0_vsfl <model.vfl1> <model-sha256> \"<five cards>\" [top_n]".to_owned(),
        );
    }
    let model_path = &args[1];
    let sha = &args[2];
    let dealt_cards = parse_cards(&args[3])?;
    let top_n: usize = args.get(4).map_or(Ok(5), |v| v.parse::<usize>().map_err(|e| e.to_string()))?;

    let dealt: [u8; 5] = dealt_cards
        .as_slice()
        .try_into()
        .map_err(|_| "T0 needs exactly five dealt cards".to_owned())?;

    let bytes = std::fs::read(model_path).map_err(|e| format!("{model_path}: {e}"))?;
    // Pinned by digest, exactly as every caller in this crate loads a model:
    // a score from unverified bytes is not reproducible.
    let model = VflModel::load_pinned(&bytes, sha)?;

    let mut openings: Vec<[u8; 5]> = Vec::with_capacity(232);
    generate_opening_actions(&mut openings);

    let mut scratch = model.scratch();
    let mut node = NodeFeatures::new(&PartialBoard::default(), &dealt[..], &[], Street::T0)?;

    let mut scored: Vec<(usize, f32, [u8; 5])> = Vec::with_capacity(openings.len());
    for (index, assignment) in openings.iter().enumerate() {
        let mut five = PartialBoard::default();
        let mut legal = true;
        for (slot, row) in assignment.iter().enumerate() {
            if five.open_slots(*row) == 0 {
                legal = false;
                break;
            }
            five.push(dealt[slot], *row);
        }
        if !legal {
            continue;
        }
        let row = node.encode(&five)?;
        scored.push((index, model.predict(&row, &mut scratch)?, *assignment));
    }

    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    println!("model  : {model_path}");
    println!("dealt  : {}", args[3]);
    println!("legal  : {} openings", scored.len());
    for (rank, (index, score, assignment)) in scored.iter().take(top_n).enumerate() {
        let mut rows: [Vec<String>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (slot, row) in assignment.iter().enumerate() {
            rows[*row as usize].push(card_to_token(dealt[slot]));
        }
        let name = |r: usize| -> String {
            if rows[r].is_empty() {
                "-".to_owned()
            } else {
                rows[r].join("")
            }
        };
        println!(
            "{:>3}. score {:+.4}  top:{} middle:{} bottom:{}  (fan index {})",
            rank + 1,
            score,
            name(ROW_TOP as usize),
            name(ROW_MIDDLE as usize),
            name(ROW_BOTTOM as usize),
            index
        );
    }
    Ok(())
}
