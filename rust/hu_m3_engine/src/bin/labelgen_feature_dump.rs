//! Batch feature extraction for label files, at native speed.
//!
//! The Python extractor runs the free-slot outlook once per action in
//! interpreted code and needs hours per label set; every encoder it calls has
//! long since been ported here and parity-checked, so this binary walks the
//! same label records and writes the same 168-dimension rows in minutes.
//!
//! Output is four flat little-endian arrays plus a JSON header; a thin Python
//! shim reshapes them into the trainer's npz. No compression, no cleverness --
//! the interchange format should be too simple to have bugs.
//!
//! Streets are inferred from each record's observation: a T1 or T2 record
//! composes structural + free-outlook(hero) + free-outlook(opponent) +
//! head-to-head, a T3 first-seat record uses the first-seat encoder. Both
//! compositions are byte-for-byte the ones the Python references produce.
//!
//! T1 shares T2's composition rather than needing its own. The free-slot
//! outlook is told nothing about the street -- it reads the room a board has
//! left and enumerates against the unknown set -- so a seven-card board with
//! six open slots is encoded by the same call as a nine-card board with four,
//! and the head-to-head block compares the two finish distributions the same
//! way. Only the cost differs, and that is the outlook's business, not this
//! binary's.
//!
//! T0 second seat joins that arm for the same reason, one street earlier: the
//! hero has placed five cards and has eight slots open, the opponent has opened
//! with five and has eight open too, and the outlook answers an eight-slot
//! board as readily as a six- or four-slot one. The first seat is deliberately
//! not included -- it acts against an empty opponent board with thirteen slots
//! open, which the outlook does not enumerate -- so that an unvalidated
//! composition fails loudly rather than silently producing a row.
//!
//! Action generation dispatches through `generate_actions` rather than going
//! straight to the turn generator, because T0 places five cards and discards
//! none. For every other street the dispatcher's board-empty test is false and
//! it calls the same turn generator this binary always called.

use ofc_hu_m3_engine::action_key::action_key;
use ofc_hu_m3_engine::infoset::{ActOrder, ActorObservation, Street};
use ofc_hu_m3_engine::t3_features::{encode_structural, unknown_cards};
use ofc_hu_m3_engine::t3first_features::{
    encode_first, opponent_outlook_first, FEATURE_SIZE,
};
use rayon::prelude::*;
use serde::Deserialize;
use std::io::Write;

#[derive(Deserialize)]
struct Record {
    offset: u64,
    skeleton: String,
    observation: ActorObservation,
    runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Run {
    scores: std::collections::BTreeMap<String, f64>,
}

struct PositionRows {
    features: Vec<[f32; FEATURE_SIZE]>,
    labels: Vec<f64>,
    keys: Vec<String>,
    code: u32,
}

fn encode_position(record: &Record) -> Result<PositionRows, String> {
    let observation = &record.observation;
    let scores = &record.runs.first().ok_or("record has no runs")?.scores;
    let unknown = unknown_cards(observation);
    let actions = ofc_hu_m3_engine::action::generate_actions(
        &observation.hero_board,
        &observation.dealt_cards,
    )?;

    let (opponent_block, opponent_finishes) =
        opponent_outlook_first(&observation.opponent_public_board, &unknown)?;

    let mut features = Vec::with_capacity(scores.len());
    let mut labels = Vec::with_capacity(scores.len());
    let mut keys = Vec::with_capacity(scores.len());
    let mut matched = 0usize;
    for action in &actions {
        let token = action_key(action)?.to_token();
        let Some(score) = scores.get(&token) else {
            continue;
        };
        let board = observation.hero_board.place(&action.placements)?;
        let row = match (observation.street, observation.to_act_order) {
            (Street::T0, ActOrder::Second) | (Street::T1, _) | (Street::T2, _) => {
                let mut out = [0.0f32; FEATURE_SIZE];
                out[..86].copy_from_slice(&encode_structural(observation, &board));
                let (hero_block, hero_finishes) =
                    opponent_outlook_first(&board, &unknown)?;
                out[86..122].copy_from_slice(&hero_block);
                out[122..158].copy_from_slice(&opponent_block);
                out[158..].copy_from_slice(
                    &ofc_hu_m3_engine::t3_features::head_to_head(
                        &hero_finishes,
                        &opponent_finishes,
                    ),
                );
                out
            }
            (Street::T3, ActOrder::First) => encode_first(
                observation,
                &board,
                &unknown,
                &opponent_block,
                &opponent_finishes,
            ),
            (street, order) => {
                return Err(format!(
                    "unsupported street/order {street:?}/{order:?} at offset {}",
                    record.offset
                ))
            }
        };
        features.push(row);
        labels.push(*score);
        keys.push(token);
        matched += 1;
    }
    if matched != scores.len() {
        return Err(format!(
            "offset {}: {} stored action keys did not match a legal action",
            record.offset,
            scores.len() - matched
        ));
    }
    let code = u32::from_str_radix(&record.skeleton, 16)
        .map_err(|error| format!("bad skeleton at offset {}: {error}", record.offset))?;
    Ok(PositionRows {
        features,
        labels,
        keys,
        code,
    })
}

fn main() -> Result<(), String> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() < 3 {
        return Err(format!(
            "usage: {} <output-dir> <labels.jsonl> [more.jsonl ...]",
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
            records.push(
                serde_json::from_str(line)
                    .map_err(|error| format!("{path}: {error}"))?,
            );
        }
    }
    records.sort_by_key(|record| record.offset);
    eprintln!("encoding {} positions on {} threads",
              records.len(), rayon::current_num_threads());

    let began = std::time::Instant::now();
    let encoded: Result<Vec<PositionRows>, String> =
        records.par_iter().map(encode_position).collect();
    let encoded = encoded?;
    eprintln!("encoded in {:.1}s", began.elapsed().as_secs_f64());

    let mut x: Vec<u8> = Vec::new();
    let mut y: Vec<u8> = Vec::new();
    let mut groups: Vec<u8> = Vec::new();
    let mut codes: Vec<u8> = Vec::new();
    let mut rows = 0usize;
    let mut key_lines = String::new();
    for (group, position) in encoded.iter().enumerate() {
        for ((row, label), key) in position.features.iter().zip(&position.labels).zip(&position.keys) {
            key_lines.push_str(key);
            key_lines.push(0x0a as char);
            for value in row {
                x.extend_from_slice(&value.to_le_bytes());
            }
            y.extend_from_slice(&label.to_le_bytes());
            groups.extend_from_slice(&(group as u32).to_le_bytes());
            codes.extend_from_slice(&position.code.to_le_bytes());
            rows += 1;
        }
    }
    std::fs::write(output.join("keys.txt"), key_lines)
        .map_err(|error| error.to_string())?;
    for (name, bytes) in [("x.bin", &x), ("y.bin", &y),
                          ("groups.bin", &groups), ("codes.bin", &codes)] {
        let mut file = std::fs::File::create(output.join(name))
            .map_err(|error| error.to_string())?;
        file.write_all(bytes).map_err(|error| error.to_string())?;
    }
    std::fs::write(
        output.join("meta.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "labelgen_features_v1",
            "rows": rows,
            "positions": encoded.len(),
            "dim": FEATURE_SIZE,
            "x": "f32le", "y": "f64le", "groups": "u32le", "codes": "u32le",
        }))
        .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    eprintln!("wrote {rows} rows x {FEATURE_SIZE} features to {}", output.display());
    Ok(())
}
