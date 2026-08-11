//! Playing against a hand nobody can see.
//!
//! An opponent in Fantasyland takes fourteen cards face down and sets thirteen
//! of them without showing one. The hero plays its ordinary five streets with
//! its ordinary action space; what changes is that the board opposite is empty
//! at T1, empty at T2, empty at T3, and empty when the hand is scored.
//!
//! Two things in this engine refused that, and both were refusing it for a
//! reason that no longer applies once the situation is named.
//!
//! GATE B, the observation. `regular_decision_geometry` demanded an exact
//! opponent card count per street -- five at T1, seven at T2, nine at T3 -- so
//! a vs-Fantasyland decision could not be CONSTRUCTED, let alone encoded. The
//! fix is a second table rather than a looser bound on the first: with a hidden
//! opponent the count is zero at every street, exactly, and a partial board is
//! still wrong.
//!
//! GATE A, the outlook. `t3first_features`' free-slot outlook enumerated boards
//! with four, six or eight open slots. Against Fantasyland the hero still plays
//! T3, and its board after that action holds eleven cards -- TWO open slots --
//! a width no ordinary hand ever presents to an outlook, because in an ordinary
//! hand nobody asks for an outlook after T3.
//!
//! The encoding contract is the T0 first-seat one, inherited whole: hero blocks
//! computed normally, the opponent-outlook and head-to-head blocks left at zero
//! because with a hidden opponent they are uncomputable rather than expensive.
//! The debt is the same debt and is recorded in the same place --
//! `fast_encode_hidden_opponent`'s own documentation.
//!
//! What is NOT tested here is that any of this plays well. These are geometry
//! claims: that the situation can be represented, that the vector can be built,
//! and that turning the flag off leaves every existing payload exactly where it
//! was.

use ofc_hu_m3_engine::cards::{Card, ALL_CARDS};
use ofc_hu_m3_engine::fast_features::{
    fast_encode_hidden_opponent, fast_outlook, FastOutlookCache, HERO_BLOCK_END,
};
use ofc_hu_m3_engine::infoset::{ActOrder, ActorObservation, ScoringContext, Seat, Street};
use ofc_hu_m3_engine::state::{Board, Row};
use ofc_hu_m3_engine::t3_features::unknown_cards;
use ofc_hu_m3_engine::t3first_features::FEATURE_SIZE;

/// Cards off the top of the deck, without repeats.
struct Deck {
    next: usize,
}

impl Deck {
    fn new() -> Self {
        Self { next: 0 }
    }
    fn take(&mut self, count: usize) -> Vec<Card> {
        let out: Vec<Card> = (0..count).map(|i| ALL_CARDS[self.next + i]).collect();
        self.next += count;
        out
    }
}

/// The hero board a street presents BEFORE its action, as `[top, middle,
/// bottom]` counts, together with how many cards the hero has already thrown.
fn hero_shape(street: Street) -> ([usize; 3], usize) {
    match street {
        Street::T1 => ([1, 2, 2], 0), // 5 placed
        Street::T2 => ([1, 3, 3], 1), // 7 placed
        Street::T3 => ([2, 4, 3], 2), // 9 placed
        other => panic!("{other:?} is not one of the vs-Fantasyland turn streets"),
    }
}

fn vs_fantasyland(street: Street) -> Result<ActorObservation, String> {
    let (rows, discards) = hero_shape(street);
    let mut deck = Deck::new();
    let board = Board::new(deck.take(rows[0]), deck.take(rows[1]), deck.take(rows[2]))
        .expect("legal hero board");
    let dealt = deck.take(3);
    let thrown = deck.take(discards);
    ActorObservation::new_vs_fantasyland(
        board,
        dealt,
        thrown,
        Seat::First,
        street,
        ActOrder::First,
        ScoringContext::default(),
    )
}

/// The same hero cards with a visible opponent, which is what the ordinary
/// table demands.
fn with_visible_opponent(street: Street, opponent_cards: usize) -> Result<ActorObservation, String> {
    let (rows, discards) = hero_shape(street);
    let mut deck = Deck::new();
    let board = Board::new(deck.take(rows[0]), deck.take(rows[1]), deck.take(rows[2]))
        .expect("legal hero board");
    let dealt = deck.take(3);
    let thrown = deck.take(discards);
    // Spread the opponent's cards legally: three to the top at most, then
    // middle, then bottom.
    let mut top = Vec::new();
    let mut middle = Vec::new();
    let mut bottom = Vec::new();
    for (index, card) in deck.take(opponent_cards).into_iter().enumerate() {
        match index % 3 {
            0 if top.len() < 3 => top.push(card),
            1 if middle.len() < 5 => middle.push(card),
            _ if bottom.len() < 5 => bottom.push(card),
            _ => middle.push(card),
        }
    }
    let opponent = Board::new(top, middle, bottom).expect("legal opponent board");
    ActorObservation::new(
        board,
        opponent,
        dealt,
        thrown,
        Seat::First,
        street,
        ActOrder::First,
        ScoringContext::default(),
    )
}

/// GATE B. The three vs-Fantasyland decisions exist.
///
/// Stated as a table rather than three assertions, because the table IS the
/// claim: the hero's own counts are untouched at every street and only the
/// opponent's column goes to zero. A future street added to one table and not
/// the other fails here.
#[test]
fn the_three_vs_fantasyland_turn_decisions_can_be_constructed() {
    for (street, hero, dealt, discards) in [
        (Street::T1, 5usize, 3usize, 0usize),
        (Street::T2, 7, 3, 1),
        (Street::T3, 9, 3, 2),
    ] {
        let observation = vs_fantasyland(street).unwrap_or_else(|error| {
            panic!("{street:?} vs Fantasyland was refused: {error}")
        });
        assert_eq!(observation.hero_board.card_count(), hero);
        assert_eq!(observation.opponent_public_board.card_count(), 0);
        assert_eq!(observation.dealt_cards.len(), dealt);
        assert_eq!(observation.hero_private_discards.len(), discards);
        assert!(observation.opponent_hidden());
        observation.validate().expect("validates");
    }
}

/// And the ordinary decisions still demand their opponent board.
///
/// The half of Gate B that is easy to lose: a second table is only a second
/// table while the first one still bites. If the widening had been done by
/// relaxing the opponent column to "zero or the usual count", this would pass
/// with a zero-card opponent under the flag OFF, and a genuinely malformed
/// observation would be accepted as a vs-Fantasyland one.
#[test]
fn a_visible_opponent_is_still_required_when_the_flag_is_off() {
    for street in [Street::T1, Street::T2, Street::T3] {
        let (rows, discards) = hero_shape(street);
        let mut deck = Deck::new();
        let board = Board::new(deck.take(rows[0]), deck.take(rows[1]), deck.take(rows[2]))
            .expect("legal hero board");
        let dealt = deck.take(3);
        let thrown = deck.take(discards);
        let error = ActorObservation::new(
            board,
            Board::empty(),
            dealt,
            thrown,
            Seat::First,
            street,
            ActOrder::First,
            ScoringContext::default(),
        )
        .expect_err("an empty opponent board is not an ordinary decision");
        assert!(error.contains("geometry"), "{street:?}: got {error}");
    }
}

/// The ordinary geometries are unreachable through the hidden-opponent
/// constructor, and the ordinary constructor still accepts them.
#[test]
fn a_hidden_opponent_cannot_be_carrying_a_board() {
    for (street, opponent) in [(Street::T1, 5usize), (Street::T2, 7), (Street::T3, 9)] {
        // The ordinary path is untouched.
        with_visible_opponent(street, opponent)
            .unwrap_or_else(|error| panic!("{street:?} ordinary was refused: {error}"));

        // And the flag with a board present is refused. Built by hand rather
        // than through `new_vs_fantasyland`, which cannot express it -- this is
        // the shape a hand-written JSON payload could still present.
        let visible = with_visible_opponent(street, opponent).expect("ordinary");
        let mut hidden = visible.clone();
        hidden.opponent_in_fantasyland = true;
        let error = hidden
            .validate()
            .expect_err("a hidden opponent showing cards is not a situation");
        assert!(error.contains("vs-fantasyland"), "{street:?}: got {error}");
    }
}

/// The hero's own Fantasyland is still refused, and by name.
///
/// It is a different game, not a different view: the hero takes fourteen cards
/// and sets thirteen in one action, so the action space this observation
/// describes is the wrong one. Widening the opponent's side must not quietly
/// widen the hero's.
#[test]
fn the_heros_own_fantasyland_is_still_a_different_game() {
    let mut observation = vs_fantasyland(Street::T2).expect("vs FL");
    observation.hero_in_fantasyland = true;
    let error = observation.validate().expect_err("must refuse");
    assert!(error.contains("hero_in_fantasyland"), "got {error}");
    assert!(error.contains("thirteen of fourteen"), "got {error}");
}

/// The flag enters the fingerprint, so the two situations are never confused.
///
/// It has always been in the fingerprint payload -- what changed is that one
/// side of the comparison can now be built. Two observations with the SAME hero
/// cards and the same street, one against a hidden opponent and one against an
/// empty-but-not-hidden... cannot both exist, since the second is refused. So
/// the comparison that can be made is against the ordinary observation with the
/// opponent board present, which must differ for two reasons at once, and
/// against a re-read of the same hidden observation, which must not differ at
/// all.
#[test]
fn the_hidden_opponent_flag_is_part_of_the_fingerprint() {
    let hidden = vs_fantasyland(Street::T2).expect("vs FL");
    let again = vs_fantasyland(Street::T2).expect("vs FL");
    assert_eq!(hidden.fingerprint(), again.fingerprint());

    let visible = with_visible_opponent(Street::T2, 7).expect("ordinary");
    assert_ne!(hidden.fingerprint(), visible.fingerprint());

    // And the flag is visible in the emitted JSON, so a payload that carried it
    // can be told from one that did not without re-deriving the geometry.
    let text = serde_json::to_string(&hidden.to_json()).expect("serializes");
    assert!(text.contains("\"opponent_in_fantasyland\":true"), "{text}");
    let ordinary = serde_json::to_string(&visible.to_json()).expect("serializes");
    assert!(
        ordinary.contains("\"opponent_in_fantasyland\":false"),
        "{ordinary}"
    );
}

/// GATE A. The hero board a T3 vs-Fantasyland action produces has two open
/// slots, and the outlook now answers for it.
///
/// Eleven cards is a width no ordinary hand ever hands an outlook: in an
/// ordinary hand the T3 decision is the last one an evaluator is asked about,
/// and it is asked about the NINE-card board before the action. Against
/// Fantasyland the same decision has to be scored on its own prospects alone,
/// so the board after the action is exactly what has to be described.
#[test]
fn the_outlook_answers_for_a_two_slot_board() {
    let observation = vs_fantasyland(Street::T3).expect("vs FL");
    let unknown = unknown_cards(&observation);
    let actions = ofc_hu_m3_engine::action::generate_turn_actions(
        &observation.hero_board,
        &observation.dealt_cards,
    )
    .expect("legal actions");
    assert!(!actions.is_empty());

    let mut cache = FastOutlookCache::new();
    for (index, action) in actions.iter().enumerate() {
        let board = observation
            .hero_board
            .place(&action.placements)
            .expect("legal placement");
        assert_eq!(board.card_count(), 11, "action {index} did not place two");
        let open: usize = [Row::Top, Row::Middle, Row::Bottom]
            .iter()
            .map(|row| row.capacity() - board.cards(*row).len())
            .sum();
        assert_eq!(open, 2, "action {index} left {open} open slots");
        let (block, _finishes) = fast_outlook(&board, &unknown, &mut cache)
            .unwrap_or_else(|error| panic!("action {index}: {error}"));
        assert!(block.iter().any(|value| *value != 0.0));
        for (slot, value) in block.iter().enumerate() {
            assert!(value.is_finite(), "action {index} slot {slot} is {value}");
        }
    }
}

/// A board width nobody validated is still refused, by the widened message.
///
/// The guard was widened to a named list, not removed. An odd width is not
/// reachable from a legal board -- every street places two cards -- so a board
/// presenting one is malformed, and it should say so rather than produce a
/// vector.
#[test]
fn an_unvalidated_board_width_is_still_refused() {
    let mut deck = Deck::new();
    // Twelve cards: one open slot, which no legal sequence of streets produces.
    let board = Board::new(deck.take(3), deck.take(5), deck.take(4)).expect("board");
    let used = board.all_cards();
    let unknown: Vec<Card> = ALL_CARDS
        .iter()
        .copied()
        .filter(|card| !used.contains(card))
        .collect();
    // `Finish` is deliberately not `Debug`, so the Ok side cannot be unwrapped
    // into a panic message; matched instead.
    let error = match fast_outlook(&board, &unknown, &mut FastOutlookCache::new()) {
        Ok(_) => panic!("one open slot is not a validated width, but was accepted"),
        Err(error) => error,
    };
    assert!(error.contains("two, four, six or eight"), "got {error}");
}

/// The encoding contract: hero half real, opponent half zero, at all three
/// vs-Fantasyland streets.
///
/// The same contract `fast_encode_hidden_opponent` carries for T0 first seat,
/// and the reason it is asserted again here is that the two arms reach it from
/// opposite directions. At T0 the hero board is empty and the OPPONENT is
/// unplayed; here the hero board is well into the hand and the opponent is
/// unplayable. A change that filled the tail in for one would fill it in for
/// the other, and the model on either side expects zeros.
#[test]
fn a_vs_fantasyland_row_is_real_in_the_hero_half_and_zero_in_the_opponent_half() {
    for street in [Street::T1, Street::T2, Street::T3] {
        let observation = vs_fantasyland(street).expect("vs FL");
        let unknown = unknown_cards(&observation);
        let actions = ofc_hu_m3_engine::action::generate_turn_actions(
            &observation.hero_board,
            &observation.dealt_cards,
        )
        .expect("legal actions");
        let mut cache = FastOutlookCache::new();
        let mut hero_half_ever_nonzero = false;
        for (index, action) in actions.iter().enumerate() {
            let board = observation
                .hero_board
                .place(&action.placements)
                .expect("legal placement");
            let features = fast_encode_hidden_opponent(&observation, &board, &unknown, &mut cache)
                .unwrap_or_else(|error| panic!("{street:?} action {index}: {error}"));
            assert_eq!(features.len(), FEATURE_SIZE);
            for (slot, value) in features.iter().enumerate().take(HERO_BLOCK_END) {
                assert!(
                    value.is_finite(),
                    "{street:?} action {index} slot {slot} is {value}"
                );
                if *value != 0.0 {
                    hero_half_ever_nonzero = true;
                }
            }
            for (slot, value) in features.iter().enumerate().skip(HERO_BLOCK_END) {
                assert_eq!(
                    *value, 0.0,
                    "{street:?} action {index} slot {slot} is {value}, and the \
                     hidden-opponent contract is a zeroed tail"
                );
            }
        }
        assert!(
            hero_half_ever_nonzero,
            "{street:?}: every hero column was zero, which is not an encoding"
        );
    }
}

/// Turning the flag off leaves an ordinary observation exactly where it was.
///
/// The whole addition has to be inert for the running fleet: the labels being
/// generated right now are ordinary-opponent labels, and their fingerprints key
/// the resume logic and the position files. Compared as the serialized payload,
/// because a changed key or a changed default is precisely the failure.
#[test]
fn an_ordinary_observation_is_untouched_by_the_addition() {
    let visible = with_visible_opponent(Street::T2, 7).expect("ordinary");
    let payload = serde_json::to_string(&visible.to_json()).expect("serializes");
    // Every field the schema has ever carried, at the values it has always
    // carried them, and nothing else. A new key would fail the round trip
    // below rather than this assertion, so both are here.
    assert!(payload.contains("\"opponent_in_fantasyland\":false"));
    assert!(payload.contains("\"hero_in_fantasyland\":false"));
    let reparsed: ActorObservation =
        serde_json::from_str(&payload).expect("an emitted payload parses back");
    assert_eq!(reparsed, visible);
    assert_eq!(reparsed.fingerprint(), visible.fingerprint());
    assert!(!reparsed.opponent_hidden());
}

/// A vs-Fantasyland observation survives the JSON round trip the fleet uses.
///
/// The worker sends observations to the engine as JSON and stores them in
/// position files as JSON. An arm that only works when the observation was
/// built in-process is an arm the fleet cannot reach.
#[test]
fn a_vs_fantasyland_observation_round_trips_through_json() {
    for street in [Street::T1, Street::T2, Street::T3] {
        let observation = vs_fantasyland(street).expect("vs FL");
        let payload = serde_json::to_string(&observation.to_json()).expect("serializes");
        let reparsed: ActorObservation =
            serde_json::from_str(&payload).expect("parses back");
        assert_eq!(reparsed, observation, "{street:?}");
        assert_eq!(reparsed.fingerprint(), observation.fingerprint());
        assert!(reparsed.opponent_hidden());
    }
}
