//! Local-only benchmark for the actor-visible Rust batch mechanics path.
//!
//! This binary intentionally avoids PyO3 and JSON conversion.  Its receipt is
//! diagnostic evidence only: it cannot satisfy the frozen c4-standard-16
//! performance gate by itself.

use ofc_hu_rl_engine::{
    BatchExecutionConfig, BatchHuRlEnv, Card, HuRlActorViewV1, ALL_CARDS, DECISION_COUNT,
    MAX_BATCH_LANES, MAX_BATCH_THREADS,
};
use serde_json::json;
use std::{env, process::ExitCode, time::Instant};

const SCHEMA: &str = "regular_ofc_hu_rl_batch_core_benchmark_v1";
const MAX_EPISODES: usize = 1_000;

#[derive(Copy, Clone)]
struct Config {
    lanes: usize,
    chunk_width: usize,
    thread_count: usize,
    episodes: usize,
}

fn main() -> ExitCode {
    match run() {
        Ok(receipt) => match serde_json::to_string(&receipt) {
            Ok(encoded) => {
                println!("{encoded}");
                ExitCode::SUCCESS
            }
            Err(_) => {
                eprintln!("HU RL batch benchmark failed to encode its receipt");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("HU RL batch benchmark failed closed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<serde_json::Value, String> {
    let config = parse_args(env::args().skip(1))?;
    let decks = benchmark_decks(config.lanes);
    let execution = BatchExecutionConfig::new(config.chunk_width, config.thread_count)
        .map_err(|_| "invalid execution configuration".to_owned())?;
    let mut batch = BatchHuRlEnv::with_execution_config(&decks, execution)
        .map_err(|_| "batch construction failed".to_owned())?;

    // One untimed episode ensures lazy allocator and worker-pool setup cannot
    // be mistaken for steady-state mechanics throughput.
    play_episode(&mut batch, &decks, 0)?;

    let started = Instant::now();
    for episode in 0..config.episodes {
        play_episode(&mut batch, &decks, episode + 1)?;
    }
    let elapsed_seconds = started.elapsed().as_secs_f64();
    if !elapsed_seconds.is_finite() || elapsed_seconds <= 0.0 {
        return Err("benchmark clock returned an invalid duration".to_owned());
    }
    let actor_decisions = config
        .lanes
        .checked_mul(DECISION_COUNT)
        .and_then(|value| value.checked_mul(config.episodes))
        .ok_or_else(|| "actor decision count overflowed".to_owned())?;
    let actor_decisions_per_second = actor_decisions as f64 / elapsed_seconds;
    if !actor_decisions_per_second.is_finite() {
        return Err("benchmark throughput is not finite".to_owned());
    }

    Ok(json!({
        "schema": SCHEMA,
        "status": "diagnostic_only",
        "artifact_role": "local_batch_core_performance_diagnostic",
        "cloud_performance_gate_eligible": false,
        "promotion_eligible": false,
        "policy_input_eligible": false,
        "training_eligible": false,
        "lanes": config.lanes,
        "chunk_width": config.chunk_width,
        "thread_count": config.thread_count,
        "episodes": config.episodes,
        "decisions_per_episode": DECISION_COUNT,
        "actor_decisions": actor_decisions,
        "elapsed_seconds": elapsed_seconds,
        "actor_decisions_per_second": actor_decisions_per_second,
        "all_done": batch.all_done(),
    }))
}

fn play_episode(
    batch: &mut BatchHuRlEnv,
    decks: &[Vec<Card>],
    episode: usize,
) -> Result<(), String> {
    batch
        .reset_from_explicit_decks(decks)
        .map_err(|_| "batch reset failed".to_owned())?;
    for decision in 0..DECISION_COUNT {
        let views = batch
            .observe_batch()
            .map_err(|_| "batch observation failed".to_owned())?;
        let selected = select_actions(&views, episode, decision)?;
        batch
            .step_batch(&selected)
            .map_err(|_| "batch step failed".to_owned())?;
    }
    if !batch.all_done() {
        return Err("full episode did not reach terminal state".to_owned());
    }
    Ok(())
}

fn select_actions(
    views: &[HuRlActorViewV1],
    episode: usize,
    decision: usize,
) -> Result<Vec<ofc_hu_rl_engine::ActionKey>, String> {
    views
        .iter()
        .enumerate()
        .map(|(lane, view)| {
            let mapping = view.legal_action_mapping();
            let index = lane
                .wrapping_mul(17)
                .wrapping_add(episode.wrapping_mul(31))
                .wrapping_add(decision.wrapping_mul(13))
                % mapping.action_count();
            mapping
                .key_at(index)
                .map_err(|_| "deterministic action selection failed".to_owned())
        })
        .collect()
}

fn benchmark_decks(lanes: usize) -> Vec<Vec<Card>> {
    (0..lanes)
        .map(|lane| {
            let mut deck = ALL_CARDS.to_vec();
            let deck_len = deck.len();
            deck.rotate_left(lane % deck_len);
            if (lane / deck_len) % 2 == 1 {
                deck.reverse();
            }
            deck
        })
        .collect()
}

fn parse_args(arguments: impl Iterator<Item = String>) -> Result<Config, String> {
    let values = arguments.collect::<Vec<_>>();
    if values.len() % 2 != 0 {
        return Err("arguments must be flag/value pairs".to_owned());
    }
    let mut lanes = None;
    let mut chunk_width = None;
    let mut thread_count = None;
    let mut episodes = None;
    for pair in values.chunks_exact(2) {
        let slot = match pair[0].as_str() {
            "--lanes" => &mut lanes,
            "--chunk-width" => &mut chunk_width,
            "--threads" => &mut thread_count,
            "--episodes" => &mut episodes,
            _ => return Err("unknown benchmark argument".to_owned()),
        };
        if slot.is_some() {
            return Err("benchmark argument was supplied twice".to_owned());
        }
        *slot = Some(
            pair[1]
                .parse::<usize>()
                .map_err(|_| "benchmark argument is not a positive integer".to_owned())?,
        );
    }
    let config = Config {
        lanes: lanes.ok_or_else(|| "--lanes is required".to_owned())?,
        chunk_width: chunk_width.ok_or_else(|| "--chunk-width is required".to_owned())?,
        thread_count: thread_count.ok_or_else(|| "--threads is required".to_owned())?,
        episodes: episodes.ok_or_else(|| "--episodes is required".to_owned())?,
    };
    if !(1..=MAX_BATCH_LANES).contains(&config.lanes) {
        return Err("--lanes is outside the supported range".to_owned());
    }
    if config.chunk_width == 0 {
        return Err("--chunk-width must be greater than zero".to_owned());
    }
    if !(1..=MAX_BATCH_THREADS).contains(&config.thread_count) {
        return Err("--threads is outside the supported range".to_owned());
    }
    if !(1..=MAX_EPISODES).contains(&config.episodes) {
        return Err("--episodes is outside the supported range".to_owned());
    }
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> impl Iterator<Item = String> {
        values
            .iter()
            .map(|value| (*value).to_owned())
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn strict_cli_accepts_one_complete_bounded_configuration() {
        let config = parse_args(arguments(&[
            "--lanes",
            "512",
            "--chunk-width",
            "32",
            "--threads",
            "16",
            "--episodes",
            "2",
        ]))
        .unwrap();
        assert_eq!(config.lanes, 512);
        assert_eq!(config.chunk_width, 32);
        assert_eq!(config.thread_count, 16);
        assert_eq!(config.episodes, 2);
    }

    #[test]
    fn strict_cli_rejects_unknown_duplicate_missing_and_unbounded_values() {
        for values in [
            vec!["--unknown", "1"],
            vec![
                "--lanes",
                "1",
                "--lanes",
                "2",
                "--chunk-width",
                "1",
                "--threads",
                "1",
                "--episodes",
                "1",
            ],
            vec![
                "--lanes",
                "0",
                "--chunk-width",
                "1",
                "--threads",
                "1",
                "--episodes",
                "1",
            ],
            vec![
                "--lanes",
                "1",
                "--chunk-width",
                "1",
                "--threads",
                "65",
                "--episodes",
                "1",
            ],
            vec![
                "--lanes",
                "1",
                "--chunk-width",
                "1",
                "--threads",
                "1",
                "--episodes",
                "1001",
            ],
            vec!["--lanes", "1"],
        ] {
            assert!(parse_args(arguments(&values)).is_err());
        }
    }

    #[test]
    fn generated_lanes_are_complete_regular_decks() {
        let decks = benchmark_decks(MAX_BATCH_LANES);
        assert_eq!(decks.len(), MAX_BATCH_LANES);
        for deck in decks {
            assert_eq!(deck.len(), ALL_CARDS.len());
            let mut sorted = deck;
            sorted.sort_unstable_by_key(|card| card.index());
            assert_eq!(sorted, ALL_CARDS);
        }
    }
}
