//! What one nested T2 decision costs, and that making it cheaper did not move it.
//!
//! `evaluate_t1` spends nearly all of its time inside nested T2 decisions -- two
//! per rollout, thousands of rollouts per position -- and each of those is
//! dominated by the free outlook, computed once per candidate action. The
//! outlook now shares its per-row and per-draw work across a decision's
//! candidate boards, so what a decision chooses has to be checked against the
//! composition that recomputed everything, which is what `uncached_decision`
//! below is: `search::learned_t2_action` transcribed onto the public per-board
//! `opponent_outlook_first`.
//!
//! The costs are reported by tests ignored by default; run them with
//! `cargo test --release --test t2_decision_cost -- --ignored --nocapture`.

use ofc_hu_m3_engine::action::generate_turn_actions;
use ofc_hu_m3_engine::cards::Card;
use ofc_hu_m3_engine::infoset::{ActOrder, ActorObservation, Seat, Street};
use ofc_hu_m3_engine::search::{decide, T3Config};
use ofc_hu_m3_engine::state::{Board, Row};
use ofc_hu_m3_engine::t3_features::{encode_structural, head_to_head, unknown_cards};
use ofc_hu_m3_engine::t3first_features::{opponent_outlook_first, FEATURE_SIZE};
use ofc_hu_m3_engine::t4_model::Model;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    observation: ActorObservation,
}

fn fixture_path(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// The observations a parity fixture carries, whichever street it pins.
fn positions_from(name: &str) -> Vec<ActorObservation> {
    let raw = std::fs::read_to_string(fixture_path(name)).expect("fixture is present");
    let fixture: Fixture = serde_json::from_str(&raw).expect("fixture parses");
    fixture
        .cases
        .into_iter()
        .map(|case| case.observation)
        .collect()
}

/// The fixture's own T2 first-seat positions: seven-card hero board, seven-card
/// opponent board, six open slots on each.
fn first_seat_positions() -> Vec<ActorObservation> {
    positions_from("t2first_features_parity.json")
}

/// A T2 second-seat position derived from a first-seat one: the opponent has
/// answered, so its board carries nine cards and four open slots instead of
/// seven and six. The two extra cards are taken from the unknown set in deck
/// order, which keeps this deterministic.
fn to_second_seat(observation: &ActorObservation) -> ActorObservation {
    let unknown = unknown_cards(observation);
    let opponent = &observation.opponent_public_board;
    let mut placements: Vec<(Card, Row)> = Vec::new();
    for (row, capacity) in [(Row::Top, 3usize), (Row::Middle, 5), (Row::Bottom, 5)] {
        let mut room = capacity - opponent.cards(row).len();
        while room > 0 && placements.len() < 2 {
            placements.push((unknown[placements.len()], row));
            room -= 1;
        }
    }
    let board = opponent
        .place(&placements)
        .expect("legal opponent placement");
    ActorObservation::new(
        observation.hero_board.clone(),
        board,
        observation.dealt_cards.clone(),
        observation.hero_private_discards.clone(),
        Seat::Second,
        Street::T2,
        ActOrder::Second,
        observation.scoring.clone(),
    )
    .expect("derived T2 second-seat observation")
}

fn digest(name: &str) -> String {
    let parsed: Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_path(&format!("{name}.sha256.json"))).expect("digest"),
    )
    .expect("digest parses");
    parsed["weights_sha256"]
        .as_str()
        .expect("digest present")
        .to_owned()
}

fn model(name: &str) -> Model {
    let image = std::fs::read(fixture_path(&format!("{name}.bin"))).expect("weights");
    Model::load_pinned(&image, &digest(name)).expect("load")
}

/// A config that pins both T2 seats to the fixture weights, so `decide` answers
/// through exactly the evaluators the nested rollout replies use.
fn config() -> T3Config {
    T3Config {
        learned_t2_first_model_path: Some(fixture_path("t2first_model_v1.bin")),
        learned_t2_first_model_sha256: Some(digest("t2first_model_v1")),
        learned_t2_second_model_path: Some(fixture_path("t2_model_v1.bin")),
        learned_t2_second_model_sha256: Some(digest("t2_model_v1")),
        run_id: "t2-decision-cost".to_owned(),
        ..Default::default()
    }
}

fn model_name(observation: &ActorObservation) -> &'static str {
    match observation.to_act_order {
        ActOrder::First => "t2first_model_v1",
        ActOrder::Second => "t2_model_v1",
    }
}

/// `search::learned_t2_action` with the outlook recomputed per candidate board,
/// which is what it did before the outlook learned to share row work. Returns
/// the chosen action in `decide`'s own JSON shape so the two can be compared.
fn uncached_decision(observation: &ActorObservation, model: &Model) -> Value {
    let actions = generate_turn_actions(&observation.hero_board, &observation.dealt_cards)
        .expect("legal actions");
    assert!(!actions.is_empty());
    let unknown = unknown_cards(observation);
    let (opponent_block, opponent_finishes) =
        opponent_outlook_first(&observation.opponent_public_board, &unknown)
            .expect("opponent outlook");
    let mut scratch = model.scratch();
    let mut values = Vec::with_capacity(actions.len());
    for action in &actions {
        let board = action.apply(&observation.hero_board).expect("legal");
        let mut features = [0.0f32; FEATURE_SIZE];
        features[..86].copy_from_slice(&encode_structural(observation, &board));
        let (hero_block, hero_finishes) =
            opponent_outlook_first(&board, &unknown).expect("hero outlook");
        features[86..122].copy_from_slice(&hero_block);
        features[122..158].copy_from_slice(&opponent_block);
        features[158..].copy_from_slice(&head_to_head(&hero_finishes, &opponent_finishes));
        values.push(model.predict_with(&features, &mut scratch).expect("predict") as f64);
    }
    // `canonical_descending_indices`' rule for this shape: strict maximum, and a
    // tie goes to the earlier action in the order `generate_turn_actions` emits.
    let mut best = 0usize;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    let action = &actions[best];
    json!({
        "placements": action
            .placements
            .iter()
            .map(|(card, row)| json!([card, row]))
            .collect::<Vec<_>>(),
        "discards": action.discards,
    })
}

/// The gate on the sharing: the T2 teacher must still choose what it chose.
///
/// Feature-level parity against the Python reference is pinned elsewhere, one
/// board at a time. This pins the thing that actually reaches a label -- the
/// selected action -- for both seats, over every fixture position, through the
/// production entry point rather than a transcription of it.
#[test]
fn the_shared_outlook_chooses_what_the_per_board_one_chose() {
    let first_seat = first_seat_positions();
    let second_seat: Vec<ActorObservation> = first_seat.iter().map(to_second_seat).collect();
    let config = config();
    let mut compared = 0usize;
    for observations in [&first_seat, &second_seat] {
        let loaded = model(model_name(&observations[0]));
        for (index, observation) in observations.iter().enumerate() {
            let produced = decide(observation, &config).expect("decide");
            let expected = uncached_decision(observation, &loaded);
            assert_eq!(
                produced["placements"], expected["placements"],
                "position {index} {:?}: the shared outlook changed the choice",
                observation.to_act_order
            );
            assert_eq!(produced["discards"], expected["discards"]);
            compared += 1;
        }
    }
    assert_eq!(compared, 2 * first_seat.len());
}

#[test]
#[ignore]
fn nested_t2_decision_cost_is_reported() {
    let first_seat = first_seat_positions();
    let second_seat: Vec<ActorObservation> = first_seat.iter().map(to_second_seat).collect();
    let config = config();
    let rounds = 3usize;

    println!("nested T2 decision cost");
    let mut shared_total = 0.0f64;
    let mut plain_total = 0.0f64;
    for observations in [&first_seat, &second_seat] {
        let label = match observations[0].to_act_order {
            ActOrder::First => "T2 first seat",
            ActOrder::Second => "T2 second seat",
        };
        let loaded = model(model_name(&observations[0]));
        let mut sink = 0usize;

        let began = std::time::Instant::now();
        for _ in 0..rounds {
            for observation in observations.iter() {
                sink += uncached_decision(observation, &loaded).to_string().len();
            }
        }
        let plain_ms = began.elapsed().as_secs_f64() * 1e3 / (rounds * observations.len()) as f64;

        let began = std::time::Instant::now();
        for _ in 0..rounds {
            for observation in observations.iter() {
                sink += decide(observation, &config)
                    .expect("decide")
                    .to_string()
                    .len();
            }
        }
        let shared_ms = began.elapsed().as_secs_f64() * 1e3 / (rounds * observations.len()) as f64;

        // `decide` re-reads and re-digests the weights on every call; the nested
        // rollout path holds one loaded model, so that is not part of what a
        // nested decision costs and is measured out.
        let began = std::time::Instant::now();
        for _ in 0..rounds * observations.len() {
            sink += model(model_name(&observations[0])).input_dim();
        }
        let load_ms = began.elapsed().as_secs_f64() * 1e3 / (rounds * observations.len()) as f64;
        let shared_ms = shared_ms - load_ms;

        let actions =
            generate_turn_actions(&observations[0].hero_board, &observations[0].dealt_cards)
                .expect("actions")
                .len();
        println!(
            "  {label:<15} per board {plain_ms:7.2} ms -> shared {shared_ms:7.2} ms  \
             ({:4.2}x, {actions} actions, {} positions)",
            plain_ms / shared_ms,
            observations.len()
        );
        plain_total += plain_ms;
        shared_total += shared_ms;
        assert!(sink > 0);
    }

    // One T2 first-seat and one T2 second-seat decision per T1 rollout.
    for particles in [128.0f64, 256.0] {
        let rollouts = 24.0 * particles;
        println!(
            "  projected evaluate_t1 nested-T2 cost at {particles:.0} particles: \
             {:7.1} s -> {:7.1} s  ({rollouts:.0} rollouts x (first + second))",
            rollouts * plain_total / 1e3,
            rollouts * shared_total / 1e3,
        );
    }
}

/// A whole T1 evaluation through the cached outlook, and what the fingerprint
/// caches above the nested decisions are worth.
///
/// The other candidate for making `evaluate_t1` cheaper was reuse across
/// particles: `locked_t2_first_action` and its siblings already memo by
/// observation fingerprint, so if rollouts repeated observations the nested
/// decisions would be nearly free. They do not. Every particle deals different
/// cards, so every nested observation is new, and the count of distinct child
/// information sets is exactly the number of nested decisions -- which is what
/// this reports, at two particle counts, so the reader can see it scale
/// linearly rather than take the claim on faith.
#[test]
#[ignore]
fn nested_observation_reuse_is_reported() {
    fn sha(file: &str) -> String {
        let parsed: Value =
            serde_json::from_str(&std::fs::read_to_string(fixture_path(file)).expect("digest"))
                .expect("parses");
        parsed["weights_sha256"]
            .as_str()
            .expect("digest present")
            .to_owned()
    }

    // A T1 second-seat position: the opponent has answered T1, so its board
    // carries seven cards to the hero's five.
    let base = &first_seat_positions()[0];
    let raw = std::fs::read_to_string(fixture_path("t1first_features_parity.json"))
        .expect("T1 fixture");
    let fixture: Fixture = serde_json::from_str(&raw).expect("parses");
    let seed = &fixture.cases[0].observation;
    let unknown = unknown_cards(seed);
    let opponent = seed
        .opponent_public_board
        .place(&[(unknown[0], Row::Top), (unknown[1], Row::Middle)])
        .expect("legal");
    let observation = ActorObservation::new(
        seed.hero_board.clone(),
        opponent,
        seed.dealt_cards.clone(),
        Vec::new(),
        Seat::Second,
        Street::T1,
        ActOrder::Second,
        base.scoring.clone(),
    )
    .expect("T1 second-seat observation");

    for particles in [1usize, 2] {
        let config = T3Config {
            candidate_samples: particles,
            evaluation_samples: particles,
            downstream_t3_samples: 1,
            downstream_t4_samples: 0,
            seed: 4242,
            candidate_seed: 4243,
            evaluation_seed: 4244,
            run_id: format!("t1-reuse-{particles}"),
            learned_t4_model_path: Some(fixture_path("t4_model_v5.bin")),
            learned_t4_model_sha256: Some(sha("t4_model_v5_predictions.json")),
            learned_t3_second_model_path: Some(fixture_path("t3_model_v2.bin")),
            learned_t3_second_model_sha256: Some(sha("t3_model_v2_predictions.json")),
            learned_t3_first_model_path: Some(fixture_path("t3first_model_v1.bin")),
            learned_t3_first_model_sha256: Some(sha("t3first_model_v1_predictions.json")),
            learned_t2_second_model_path: Some(fixture_path("t2_model_v1.bin")),
            learned_t2_second_model_sha256: Some(digest("t2_model_v1")),
            learned_t2_first_model_path: Some(fixture_path("t2first_model_v1.bin")),
            learned_t2_first_model_sha256: Some(digest("t2first_model_v1")),
            ..Default::default()
        };
        let began = std::time::Instant::now();
        let result = ofc_hu_m3_engine::search::evaluate_t1(&observation, &config).expect("t1");
        let elapsed = began.elapsed().as_secs_f64();
        let actions = result["legal_action_count"].as_u64().expect("count") as f64;
        let rollouts = actions * 2.0 * particles as f64;
        let distinct = result["child_information_set_count"].as_u64().expect("sets") as f64;
        println!(
            "  {particles} + {particles} particles: {rollouts:.0} rollouts, \
             {:.0} nested decisions, {distinct:.0} distinct observations \
             (reuse {:.1}%), {elapsed:.1} s",
            rollouts * 6.0,
            100.0 * (1.0 - distinct / (rollouts * 6.0)),
        );
    }
}

/// Where a decision's time goes, measured on the per-board path so the blocks
/// are separable: the shared opponent block, the per-action hero outlook, the
/// structural block and the head-to-head.
#[test]
#[ignore]
fn nested_t2_decision_breakdown_is_reported() {
    let first_seat = first_seat_positions();
    let second_seat: Vec<ActorObservation> = first_seat.iter().map(to_second_seat).collect();

    for observations in [&first_seat, &second_seat] {
        let label = match observations[0].to_act_order {
            ActOrder::First => "first seat",
            ActOrder::Second => "second seat",
        };
        let rounds = 3usize;
        let mut sink = 0.0f64;

        let began = std::time::Instant::now();
        for _ in 0..rounds {
            for observation in observations.iter() {
                let unknown = unknown_cards(observation);
                let (block, _finishes) =
                    opponent_outlook_first(&observation.opponent_public_board, &unknown)
                        .expect("opponent");
                sink += block[0] as f64;
            }
        }
        let opponent_ms =
            began.elapsed().as_secs_f64() * 1e3 / (rounds * observations.len()) as f64;

        let prepared: Vec<(Vec<Card>, Vec<Board>)> = observations
            .iter()
            .map(|observation| {
                let unknown = unknown_cards(observation);
                let boards =
                    generate_turn_actions(&observation.hero_board, &observation.dealt_cards)
                        .expect("actions")
                        .iter()
                        .map(|action| action.apply(&observation.hero_board).expect("legal"))
                        .collect();
                (unknown, boards)
            })
            .collect();

        let mut rows = 0usize;
        let began = std::time::Instant::now();
        for _ in 0..rounds {
            for (unknown, boards) in &prepared {
                for board in boards {
                    sink += opponent_outlook_first(board, unknown).expect("hero").0[0] as f64;
                    rows += 1;
                }
            }
        }
        let hero_us = began.elapsed().as_secs_f64() * 1e6 / rows as f64;

        let mut rows = 0usize;
        let began = std::time::Instant::now();
        for _ in 0..rounds {
            for (index, observation) in observations.iter().enumerate() {
                for board in &prepared[index].1 {
                    sink += encode_structural(observation, board)[0] as f64;
                    rows += 1;
                }
            }
        }
        let structural_us = began.elapsed().as_secs_f64() * 1e6 / rows as f64;

        let finishes: Vec<_> = observations
            .iter()
            .enumerate()
            .map(|(index, observation)| {
                let (unknown, boards) = &prepared[index];
                let (_block, opponent) =
                    opponent_outlook_first(&observation.opponent_public_board, unknown)
                        .expect("opponent");
                let hero: Vec<_> = boards
                    .iter()
                    .map(|board| opponent_outlook_first(board, unknown).expect("hero").1)
                    .collect();
                (opponent, hero)
            })
            .collect();

        let mut rows = 0usize;
        let began = std::time::Instant::now();
        for _ in 0..rounds {
            for (opponent, hero) in &finishes {
                for ours in hero {
                    sink += head_to_head(ours, opponent)[0] as f64;
                    rows += 1;
                }
            }
        }
        let h2h_us = began.elapsed().as_secs_f64() * 1e6 / rows as f64;

        let actions = prepared[0].1.len();
        println!("{label}: {actions} actions per decision");
        println!("  opponent outlook   {opponent_ms:8.2} ms once");
        println!("  hero outlook       {hero_us:8.1} us x {actions}");
        println!("  structural         {structural_us:8.1} us x {actions}");
        println!("  head to head       {h2h_us:8.1} us x {actions}");
        assert!(sink.is_finite());
    }
}

/// One free outlook at each of the three widths it is asked for.
///
/// The breakdown above reports the outlook only at the widths a T2 decision
/// reaches. The cost is not linear in the open slots: the draw list is a fixed
/// 512 at every width, but the arrangement count the joint block walks per draw
/// goes 12 at four slots, 90 at six and 560 at eight, and the finishes it builds
/// are the same 512 either way. Eight slots is therefore where both halves of
/// the joint block are worst, and it is the width a T1 first-seat node pays for
/// its opponent block and a T0 second-seat node pays once per candidate action,
/// so it gets its own line rather than being inferred from the narrower two.
///
/// Boards come from the parity fixtures so the widths are the real ones: the T1
/// first-seat opponent shows five cards and eight open slots, the T2 first-seat
/// opponent seven and six, and the derived T2 second-seat opponent nine and four.
///
/// What this was written to settle, and did: the joint block is not where an
/// eight-slot outlook's time goes. Timed by phase, an eight-slot board spends
/// about three fifths of itself building the per-row completion tables, a fifth
/// on the per-row histograms, and under a sixth on the arrangement loop and the
/// finishes together -- and a whole T1 first-seat decision, which amortises the
/// tables and histograms across its candidate boards, still only spends about a
/// fifth there. Making the arrangement loop and the `Finish` representation free
/// outright would be worth about 1.2x and no more; the row tables are the next
/// thing worth opening, not the joint block.
#[test]
#[ignore]
fn free_outlook_cost_by_width_is_reported() {
    let six = first_seat_positions();
    let four: Vec<ActorObservation> = six.iter().map(to_second_seat).collect();
    let eight = positions_from("t1first_features_parity.json");

    println!("free outlook cost by open-slot width");
    for (slots, observations, cards, rounds) in
        [(4usize, &four, 9usize, 20usize), (6, &six, 7, 10), (8, &eight, 5, 5)]
    {
        let boards: Vec<(Board, Vec<Card>)> = observations
            .iter()
            .map(|observation| {
                assert_eq!(
                    observation.opponent_public_board.card_count(),
                    cards,
                    "the {slots}-slot source is not the width it claims"
                );
                (
                    observation.opponent_public_board.clone(),
                    unknown_cards(observation),
                )
            })
            .collect();

        // The best of `rounds`, per board, rather than the mean of them. This
        // box shares its cores with unrelated jobs, and a mean over a contended
        // run moves by more than any change to this code would: the same board
        // has come back 2.6x apart between two runs of this test. A minimum is
        // the least contended sample taken, which is the one that describes the
        // work rather than the machine.
        let mut sink = 0.0f64;
        let mut survivors = 0usize;
        let mut best = 0.0f64;
        for (board, unknown) in &boards {
            let mut fastest = f64::INFINITY;
            for _ in 0..rounds {
                let began = std::time::Instant::now();
                let (block, finishes) =
                    opponent_outlook_first(board, unknown).expect("outlook");
                fastest = fastest.min(began.elapsed().as_secs_f64());
                sink += block[0] as f64;
                survivors += finishes.len();
            }
            best += fastest;
        }
        println!(
            "  {slots}-slot outlook  {:9.1} us per board  \
             ({} boards, {:.0} survivors each)",
            best * 1e6 / boards.len() as f64,
            boards.len(),
            survivors as f64 / (rounds * boards.len()) as f64,
        );
        assert!(sink.is_finite());
    }
}

/// A whole T1 first-seat decision, which is where the eight-slot outlook lands.
///
/// A T0 teacher evaluation spends nearly all of its time in nested T1 decisions,
/// and a T1 first-seat decision is the one that pays an eight-slot opponent
/// block once and a six-slot hero block per action. `decide` is the production
/// entry point for it; the fixture's T1 first-seat positions are the input, and
/// the T2 first-seat weights stand in for the T1 first-seat ones, which do not
/// exist yet as a fixture. The two share the 168-wide layout, so the arithmetic
/// per decision is identical and only the numbers the model returns differ --
/// and this test times the decision rather than reading its answer.
#[test]
#[ignore]
fn nested_t1_first_decision_cost_is_reported() {
    let observations = positions_from("t1first_features_parity.json");
    assert!(!observations.is_empty());
    let config = T3Config {
        learned_t1_first_model_path: Some(fixture_path("t2first_model_v1.bin")),
        learned_t1_first_model_sha256: Some(digest("t2first_model_v1")),
        run_id: "t1-first-decision-cost".to_owned(),
        ..Default::default()
    };
    let rounds = 5usize;
    let mut sink = 0usize;
    let mut decide_seconds = 0.0f64;
    let mut load_seconds = 0.0f64;

    // `decide` re-reads and re-digests the weights on every call; the nested
    // rollout path holds one loaded model, so that is measured out, for the
    // same reason the T2 report measures it out.
    //
    // Both halves are the best of `rounds` rather than the mean. On a box whose
    // fixtures live behind a Windows mount the file read dominates and its
    // spread is enormous -- wide enough that a mean load came back larger than
    // the mean of the decisions that each contain one, which subtracts to a
    // negative decision.
    //
    // The load probe is `load_learned_model`'s body rather than the `model`
    // helper above, and for the same reason: the helper reads the digest file
    // too, which is a second open on that mount and costs more than the decision
    // being measured. `decide` is handed the digest as a string, so measuring a
    // load that reads one is measuring the wrong thing.
    let sha = digest("t2first_model_v1");
    let weights = fixture_path("t2first_model_v1.bin");
    for observation in &observations {
        let mut fastest_decide = f64::INFINITY;
        let mut fastest_load = f64::INFINITY;
        for _ in 0..rounds {
            let began = std::time::Instant::now();
            sink += decide(observation, &config).expect("decide").to_string().len();
            fastest_decide = fastest_decide.min(began.elapsed().as_secs_f64());

            let began = std::time::Instant::now();
            let image = std::fs::read(&weights).expect("weights");
            sink += Model::load_pinned(&image, &sha).expect("load").input_dim();
            fastest_load = fastest_load.min(began.elapsed().as_secs_f64());
        }
        decide_seconds += fastest_decide;
        load_seconds += fastest_load;
    }
    let positions = observations.len() as f64;
    let decide_ms = decide_seconds * 1e3 / positions;
    let load_ms = load_seconds * 1e3 / positions;
    let decision_ms = decide_ms - load_ms;

    let actions: usize = observations
        .iter()
        .map(|observation| {
            generate_turn_actions(&observation.hero_board, &observation.dealt_cards)
                .expect("actions")
                .len()
        })
        .sum::<usize>()
        / observations.len();
    println!("nested T1 first-seat decision cost");
    println!(
        "  decision  {decision_ms:8.2} ms  ({actions} actions, {} positions, \
         load {load_ms:.2} ms measured out)",
        observations.len()
    );
    assert!(sink > 0);
}
