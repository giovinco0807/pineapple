//! Batch coarse-feature extraction for the T1 distillation set.
//!
//! The sibling of `labelgen_feature_dump`, and different in the two ways the
//! task is different. The features are [`fast_features`]' rather than
//! [`t3first_features`]', because the model being fitted is the one that will
//! read them inside a rollout. And the target is an action rather than a value:
//! the phase-1 records hold the full-precision teacher's *choice*, not its
//! score for every candidate, so what comes out is one index per position
//! rather than one label per row.
//!
//! Output is three flat little-endian arrays, a key list and a JSON header:
//!
//! * `x.bin` -- `f32`, `FEATURE_SIZE` per row, one row per legal action.
//! * `groups.bin` -- `u32`, one per row, the position each row belongs to.
//! * `chosen.bin` -- `u32`, one per *position*, where the teacher's action sits
//!   in that position's canonical action order.
//! * `offsets.bin` -- `u64`, one per position, so the trainer can split on the
//!   record's own offset rather than on row order.
//! * `keys.txt` -- one action key per row, in row order.
//!
//! The canonical order is the decision function's own -- the order it
//! enumerates and ranks in. `chosen.bin` is an index into that order, so a
//! trainer that scores the rows of a group in file order and takes an argmax is
//! asking exactly the question the engine asks.
//!
//! Which generator produces that order depends on the street, and dispatching
//! on it is the whole of what T0 needed here. Every turn street places two of
//! three dealt cards and discards one; the opening street places all five and
//! discards none, so its legal set comes from `generate_initial_actions` and is
//! 232 wide rather than twenty-seven. `generate_actions` makes that choice by
//! the same board-empty test the engine uses, which is what keeps the row order
//! this binary writes identical to the row order `learned_t0_second_action`
//! scores in. The geometry check below is separate and by name, so a street
//! whose composition has not been validated fails loudly instead of quietly
//! producing a row.
//!
//! One kind per run. The seats present different geometry -- five against five,
//! five against seven, nothing against five -- and get their own model, so a
//! dump that mixed them would fit one set of weights to all of them and report
//! an average that describes none. Mixed input is refused by name.

use ofc_hu_m3_engine::action_key::action_key;
use ofc_hu_m3_engine::fast_features::{
    fast_encode, fast_encode_hidden_opponent, fast_outlook, FastOutlookCache,
};
use ofc_hu_m3_engine::infoset::{ActOrder, ActorObservation, Street};
use ofc_hu_m3_engine::t3_features::unknown_cards;
use ofc_hu_m3_engine::t3first_features::FEATURE_SIZE;
use rayon::prelude::*;
use serde::Deserialize;
use std::io::Write;

#[derive(Deserialize)]
struct Record {
    offset: u64,
    kind: String,
    observation: ActorObservation,
    chosen_key: String,
}

struct PositionRows {
    features: Vec<[f32; FEATURE_SIZE]>,
    keys: Vec<String>,
    chosen: u32,
    offset: u64,
}

/// The board pair each supported street presents, as `(hero, opponent)` card
/// counts.
///
/// Named rather than inferred because the coarse outlook enumerates four, six
/// and eight open slots and nothing else: a board outside those widths is
/// refused by the outlook with a message about slots, which says nothing about
/// which street was wrong. Checking here means the error names the street.
/// Whether the two blocks reading the opponent can be computed at all.
///
/// Three of the arms below cannot compute them, and for two different reasons.
/// The opening street acting first faces an opponent that has not moved yet;
/// the vs-Fantasyland streets face one that never will. Both leave the outlook
/// an empty board -- thirteen open slots, which it enumerates for nothing -- so
/// both dispatch to `fast_encode_hidden_opponent` and both carry a zeroed tail.
/// They are distinguished in the table rather than merged, because the reason
/// is part of what the arm is.
#[derive(Copy, Clone, PartialEq)]
enum Composition {
    /// Structural, hero outlook, opponent outlook, head-to-head. All four real.
    FourBlock,
    /// Structural and hero outlook real, the last 46 columns zero.
    ZeroedTail,
}

fn expected_geometry(
    street: Street,
    order: ActOrder,
    opponent_hidden: bool,
) -> Option<(usize, usize, Composition)> {
    if opponent_hidden {
        // Against a Fantasyland opponent the hero plays its ordinary streets
        // with its ordinary action space; only the information opposite is
        // gone, so the hero counts are the ones the ordinary table already
        // states and the opponent count is zero at every one of them.
        //
        // The hero board AFTER the action is what the outlook sees: five plus
        // two is seven at T1 (six open), nine at T2 (four open) and eleven at
        // T3 (TWO open, which is the width `t3first_features` had to learn).
        // T0 is absent on purpose -- acting first against a hidden opponent is
        // the same 0/0 geometry as the ordinary T0 first seat, and giving it
        // two spellings would let one position be labelled under two kinds.
        return match (street, order) {
            (Street::T1, ActOrder::First) => Some((5, 0, Composition::ZeroedTail)),
            (Street::T2, ActOrder::First) => Some((7, 0, Composition::ZeroedTail)),
            (Street::T3, ActOrder::First) => Some((9, 0, Composition::ZeroedTail)),
            _ => None,
        };
    }
    match (street, order) {
        // Two empty boards: the opening street acting first.
        (Street::T0, ActOrder::First) => Some((0, 0, Composition::ZeroedTail)),
        (Street::T0, ActOrder::Second) => Some((0, 5, Composition::FourBlock)),
        (Street::T1, ActOrder::First) => Some((5, 5, Composition::FourBlock)),
        (Street::T1, ActOrder::Second) => Some((5, 7, Composition::FourBlock)),
        (Street::T2, ActOrder::First) => Some((7, 7, Composition::FourBlock)),
        (Street::T2, ActOrder::Second) => Some((7, 9, Composition::FourBlock)),
        _ => None,
    }
}

fn encode_position(record: &Record) -> Result<PositionRows, String> {
    let observation = &record.observation;
    let Some((hero, opponent, composition)) = expected_geometry(
        observation.street,
        observation.to_act_order,
        observation.opponent_hidden(),
    ) else {
        return Err(format!(
            "offset {}: {:?}/{:?}{} has no validated coarse composition",
            record.offset,
            observation.street,
            observation.to_act_order,
            if observation.opponent_hidden() { " vs-fantasyland" } else { "" }
        ));
    };
    let actual = (
        observation.hero_board.card_count(),
        observation.opponent_public_board.card_count(),
    );
    if actual != (hero, opponent) {
        return Err(format!(
            "offset {}: {:?}/{:?} expects a {hero}-card hero board and a \
             {opponent}-card opponent board, got {actual:?}",
            record.offset, observation.street, observation.to_act_order
        ));
    }
    let unknown = unknown_cards(observation);
    // The dispatcher rather than the turn generator: the opening street places
    // five cards and discards none, and its 232 actions are the order the T0
    // second-seat decider ranks in. For every other street the board-empty test
    // is false and this is the same turn generator it always was.
    let actions = ofc_hu_m3_engine::action::generate_actions(
        &observation.hero_board,
        &observation.dealt_cards,
    )?;
    if actions.is_empty() {
        return Err(format!("offset {}: no legal actions", record.offset));
    }

    // One cache for the position: the opponent block is computed once and the
    // candidate boards share their untouched rows, which is the same sharing
    // the decision function gets at run time.
    //
    // The opening street acting first has no opponent block to compute -- the
    // board it would come from is empty -- so that half is skipped and the two
    // blocks reading it stay zero. Everything else, including the sharing, is
    // unchanged.
    let mut cache = FastOutlookCache::new();
    let opponent = match composition {
        Composition::ZeroedTail => None,
        Composition::FourBlock => Some(fast_outlook(
            &observation.opponent_public_board,
            &unknown,
            &mut cache,
        )?),
    };

    let mut features = Vec::with_capacity(actions.len());
    let mut keys = Vec::with_capacity(actions.len());
    let mut chosen = None;
    for (index, action) in actions.iter().enumerate() {
        let token = action_key(action)?.to_token();
        if token == record.chosen_key {
            if chosen.is_some() {
                return Err(format!(
                    "offset {}: chosen key {} matches two legal actions",
                    record.offset, record.chosen_key
                ));
            }
            chosen = Some(index as u32);
        }
        let board = observation.hero_board.place(&action.placements)?;
        features.push(match opponent.as_ref() {
            Some((opponent_block, opponent_finishes)) => fast_encode(
                observation,
                &board,
                &unknown,
                opponent_block,
                opponent_finishes,
                &mut cache,
            )?,
            None => fast_encode_hidden_opponent(observation, &board, &unknown, &mut cache)?,
        });
        keys.push(token);
    }
    // A record whose stored choice is not in the legal set means the dump and
    // the generator disagree about the action space, which no amount of
    // training would recover from.
    let Some(chosen) = chosen else {
        return Err(format!(
            "offset {}: chosen key {} is not among the {} legal actions",
            record.offset,
            record.chosen_key,
            actions.len()
        ));
    };
    Ok(PositionRows {
        features,
        keys,
        chosen,
        offset: record.offset,
    })
}

fn main() -> Result<(), String> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() < 3 {
        return Err(format!(
            "usage: {} <output-dir> <phase1.jsonl> [more.jsonl ...]",
            arguments[0]
        ));
    }
    let output = std::path::PathBuf::from(&arguments[1]);
    std::fs::create_dir_all(&output).map_err(|error| error.to_string())?;

    let mut records: Vec<Record> = Vec::new();
    for path in &arguments[2..] {
        let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            records.push(serde_json::from_str(line).map_err(|error| format!("{path}: {error}"))?);
        }
    }
    if records.is_empty() {
        return Err("no records to encode".to_owned());
    }
    let kind = records[0].kind.clone();
    if let Some(other) = records.iter().find(|record| record.kind != kind) {
        return Err(format!(
            "one kind per dump: found both {kind:?} and {:?}; the seats have \
             different geometry and their own model",
            other.kind
        ));
    }
    records.sort_by_key(|record| record.offset);
    let duplicated = records.windows(2).any(|pair| pair[0].offset == pair[1].offset);
    if duplicated {
        return Err("two records share an offset; shards overlap".to_owned());
    }
    eprintln!(
        "encoding {} {kind} positions on {} threads",
        records.len(),
        rayon::current_num_threads()
    );

    let began = std::time::Instant::now();
    let encoded: Result<Vec<PositionRows>, String> =
        records.par_iter().map(encode_position).collect();
    let encoded = encoded?;
    eprintln!("encoded in {:.1}s", began.elapsed().as_secs_f64());

    let mut x: Vec<u8> = Vec::new();
    let mut groups: Vec<u8> = Vec::new();
    let mut chosen: Vec<u8> = Vec::new();
    let mut offsets: Vec<u8> = Vec::new();
    let mut key_lines = String::new();
    let mut rows = 0usize;
    for (group, position) in encoded.iter().enumerate() {
        for (row, key) in position.features.iter().zip(&position.keys) {
            key_lines.push_str(key);
            key_lines.push('\n');
            for value in row {
                x.extend_from_slice(&value.to_le_bytes());
            }
            groups.extend_from_slice(&(group as u32).to_le_bytes());
            rows += 1;
        }
        chosen.extend_from_slice(&position.chosen.to_le_bytes());
        offsets.extend_from_slice(&position.offset.to_le_bytes());
    }
    std::fs::write(output.join("keys.txt"), key_lines).map_err(|error| error.to_string())?;
    for (name, bytes) in [
        ("x.bin", &x),
        ("groups.bin", &groups),
        ("chosen.bin", &chosen),
        ("offsets.bin", &offsets),
    ] {
        let mut file =
            std::fs::File::create(output.join(name)).map_err(|error| error.to_string())?;
        file.write_all(bytes).map_err(|error| error.to_string())?;
    }
    let mean_actions = rows as f64 / encoded.len() as f64;
    std::fs::write(
        output.join("meta.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "fast_feature_dump_v1",
            "kind": kind,
            "rows": rows,
            "positions": encoded.len(),
            "dim": FEATURE_SIZE,
            "mean_actions": mean_actions,
            "histogram_cap": ofc_hu_m3_engine::fast_features::FAST_HISTOGRAM_CAP,
            "joint_draws": ofc_hu_m3_engine::fast_features::FAST_JOINT_DRAWS,
            "x": "f32le", "groups": "u32le", "chosen": "u32le", "offsets": "u64le",
        }))
        .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    eprintln!(
        "wrote {rows} rows x {FEATURE_SIZE} features over {} positions ({mean_actions:.1} \
         actions each) to {}",
        encoded.len(),
        output.display()
    );
    Ok(())
}
