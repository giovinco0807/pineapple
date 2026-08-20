//! Timing harness for the GPU-port investigation (Task 1 groundwork).
//!
//! Two modes:
//!   forward             -- micro-bench Model::predict_with on fixture weights
//!   t0 PS K N FAST      -- one real evaluate_t0 (first seat, stand-in weights):
//!                          prefilter_samples=PS, prefilter_keep=K,
//!                          evaluation_samples=N, FAST=1 pins the fast_* slots.
//!
//! Weights are the repo's stand-in fixtures, digests computed on the fly, so
//! the run is real engine code end to end; only the numeric outputs are
//! meaningless (which changes which action wins, not what a street costs).

use ofc_hu_m3_engine::cards::ALL_CARDS;
use ofc_hu_m3_engine::infoset::{ActOrder, ActorObservation, ScoringContext, Seat, Street};
use ofc_hu_m3_engine::search::{evaluate_t0, model_scores, T3Config};
use ofc_hu_m3_engine::state::Board;
use ofc_hu_m3_engine::t4_model::Model;
use sha2::{Digest, Sha256};
use std::hint::black_box;
use std::time::Instant;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// Sha256 of the serialized value, so before/after runs can assert byte
/// identity of engine output without storing the whole document.
fn value_digest(value: &serde_json::Value) -> String {
    let serialized = serde_json::to_string(value).expect("serializes");
    let mut hex = String::with_capacity(64);
    for byte in Sha256::digest(serialized.as_bytes()) {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// Path plus the sha256 the engine's pinning check will demand.
fn pinned(name: &str) -> (String, String) {
    let path = fixture(name);
    let bytes = std::fs::read(&path).expect("fixture readable");
    let mut hex = String::with_capacity(64);
    for byte in Sha256::digest(&bytes) {
        hex.push_str(&format!("{byte:02x}"));
    }
    (path, hex)
}

fn bench_forward(name: &str) {
    let (path, sha) = pinned(name);
    let bytes = std::fs::read(&path).unwrap();
    let model = Model::load_pinned(&bytes, &sha).expect("model loads");
    let dim = model.input_dim();
    let mut scratch = model.scratch();
    let features: Vec<f32> = (0..dim).map(|i| ((i % 13) as f32) * 0.25 - 1.5).collect();
    for _ in 0..2_000 {
        black_box(model.predict_with(black_box(&features), &mut scratch).unwrap());
    }
    let iters: u32 = 200_000;
    let start = Instant::now();
    for _ in 0..iters {
        black_box(model.predict_with(black_box(&features), &mut scratch).unwrap());
    }
    let elapsed = start.elapsed();
    println!(
        "forward {name}: input_dim={dim} bytes={} -> {:.3} us/pass ({iters} iters)",
        bytes.len(),
        elapsed.as_secs_f64() * 1e6 / f64::from(iters)
    );
}

/// Two empty boards, five dealt cards: the T0 first-seat root.
fn t0_first_observation() -> ActorObservation {
    ActorObservation::new(
        Board::empty(),
        Board::empty(),
        ALL_CARDS[0..5].to_vec(),
        Vec::new(),
        Seat::First,
        Street::T0,
        ActOrder::First,
        ScoringContext::default(),
    )
    .expect("observation")
}

fn evaluation_config(
    prefilter_samples: usize,
    prefilter_keep: usize,
    evaluation_samples: usize,
    fast: bool,
) -> T3Config {
    let (wide_path, wide_sha) = pinned("t3first_model_v1.bin");
    let mut config = T3Config {
        candidate_samples: 1,
        evaluation_samples,
        downstream_t3_samples: 1,
        downstream_t4_samples: 0,
        seed: 994,
        candidate_seed: 995,
        evaluation_seed: 996,
        run_id: "t0-profile".to_owned(),
        use_t4_action_cache: true,
        prefilter_samples,
        prefilter_keep,
        learned_t4_model_path: Some(fixture("t4_model_v5.bin")),
        learned_t4_model_sha256: Some(pinned("t4_model_v5.bin").1),
        learned_t3_second_model_path: Some(fixture("t3_model_v2.bin")),
        learned_t3_second_model_sha256: Some(pinned("t3_model_v2.bin").1),
        learned_t3_first_model_path: Some(wide_path.clone()),
        learned_t3_first_model_sha256: Some(wide_sha.clone()),
        learned_t2_second_model_path: Some(wide_path.clone()),
        learned_t2_second_model_sha256: Some(wide_sha.clone()),
        learned_t2_first_model_path: Some(wide_path.clone()),
        learned_t2_first_model_sha256: Some(wide_sha.clone()),
        learned_t1_second_model_path: Some(wide_path.clone()),
        learned_t1_second_model_sha256: Some(wide_sha.clone()),
        learned_t1_first_model_path: Some(wide_path.clone()),
        learned_t1_first_model_sha256: Some(wide_sha.clone()),
        learned_t0_second_model_path: Some(wide_path),
        learned_t0_second_model_sha256: Some(wide_sha),
        ..Default::default()
    };
    if fast {
        for (path_slot, sha_slot, file) in [
            (
                &mut config.fast_t0_second_model_path,
                &mut config.fast_t0_second_model_sha256,
                "fast_t0_second_v1.bin",
            ),
            (
                &mut config.fast_t1_first_model_path,
                &mut config.fast_t1_first_model_sha256,
                "fast_t1_first_v1.bin",
            ),
            (
                &mut config.fast_t1_second_model_path,
                &mut config.fast_t1_second_model_sha256,
                "fast_t1_second_v1.bin",
            ),
            (
                &mut config.fast_t2_first_model_path,
                &mut config.fast_t2_first_model_sha256,
                "fast_t2_first_v1.bin",
            ),
            (
                &mut config.fast_t2_second_model_path,
                &mut config.fast_t2_second_model_sha256,
                "fast_t2_second_v1.bin",
            ),
        ] {
            let (path, sha) = pinned(file);
            *path_slot = Some(path);
            *sha_slot = Some(sha);
        }
    }
    config
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("forward") => {
            for name in [
                "t3first_model_v1.bin",
                "t3_model_v2.bin",
                "t4_model_v5.bin",
                "fast_t0_second_v1.bin",
                "t0first_model_v1.bin",
            ] {
                bench_forward(name);
            }
        }
        Some("scores") => {
            // The brief's `model_scores` measurement: 232 openings through
            // action generation + fast_encode_hidden_opponent + forward pass.
            let (path, sha) = pinned("t0first_model_v1.bin");
            let config = T3Config {
                run_id: "t0-profile-scores".to_owned(),
                learned_t0_first_model_path: Some(path),
                learned_t0_first_model_sha256: Some(sha),
                ..Default::default()
            };
            let observation = t0_first_observation();
            // Warm-up, then timed repeats.
            let warm = model_scores(&observation, &config).expect("model_scores");
            let rows = warm["actions"].as_array().map(Vec::len).unwrap_or(0);
            let repeats: u32 = 20;
            let start = Instant::now();
            for _ in 0..repeats {
                black_box(model_scores(black_box(&observation), &config).expect("model_scores"));
            }
            let elapsed = start.elapsed();
            let per_call_ms = elapsed.as_secs_f64() * 1e3 / f64::from(repeats);
            println!(
                "model_scores T0 first: {rows} actions, {per_call_ms:.2} ms/call, \
                 {:.1} us/action ({repeats} repeats; includes model reload+sha256 per call)",
                per_call_ms * 1e3 / rows as f64
            );
            println!("model_scores sha256: {}", value_digest(&warm));
        }
        Some("t0bench") => {
            let prefilter_samples: usize = args[1].parse().expect("PS");
            let prefilter_keep: usize = args[2].parse().expect("K");
            let evaluation_samples: usize = args[3].parse().expect("N");
            let fast = args[4].as_str() == "1";
            let repeats: usize = args[5].parse().expect("REPS");
            let observation = t0_first_observation();
            let config =
                evaluation_config(prefilter_samples, prefilter_keep, evaluation_samples, fast);
            let mut times = Vec::with_capacity(repeats);
            let mut digest = String::new();
            for _ in 0..repeats {
                let start = Instant::now();
                let value = evaluate_t0(&observation, &config).expect("evaluate_t0");
                times.push(start.elapsed().as_secs_f64());
                let this = value_digest(&value);
                assert!(
                    digest.is_empty() || digest == this,
                    "evaluate_t0 output changed between identical repeats"
                );
                digest = this;
            }
            times.sort_by(f64::total_cmp);
            let median = times[times.len() / 2];
            let per_rep: Vec<String> = times.iter().map(|t| format!("{t:.3}")).collect();
            println!(
                "t0bench ps={prefilter_samples} keep={prefilter_keep} n={evaluation_samples} \
                 fast={fast}: median {median:.3} s over {repeats} (sorted: {})",
                per_rep.join(" ")
            );
            println!("t0bench sha256: {digest}");
        }
        Some("t0") => {
            let prefilter_samples: usize = args[1].parse().expect("PS");
            let prefilter_keep: usize = args[2].parse().expect("K");
            let evaluation_samples: usize = args[3].parse().expect("N");
            let fast = args.get(4).map(String::as_str) == Some("1");
            let observation = t0_first_observation();
            let config =
                evaluation_config(prefilter_samples, prefilter_keep, evaluation_samples, fast);
            let start = Instant::now();
            let result = evaluate_t0(&observation, &config);
            let elapsed = start.elapsed();
            match result {
                Ok(value) => {
                    let keys: Vec<&str> = value
                        .as_object()
                        .map(|map| map.keys().map(String::as_str).collect())
                        .unwrap_or_default();
                    println!(
                        "evaluate_t0 ps={prefilter_samples} keep={prefilter_keep} \
                         n={evaluation_samples} fast={fast}: {:.3} s (keys: {})",
                        elapsed.as_secs_f64(),
                        keys.join(",")
                    );
                    println!("evaluate_t0 sha256: {}", value_digest(&value));
                }
                Err(error) => println!("evaluate_t0 failed after {elapsed:?}: {error}"),
            }
        }
        _ => {
            eprintln!(
                "usage: t0_profile forward | scores | t0 PS K N [FAST] | t0bench PS K N FAST REPS"
            );
            std::process::exit(2);
        }
    }
}
