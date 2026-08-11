//! Information-set-safe exchangeable hidden-card particles.
//!
//! Sampling accepts only `ActorObservation`. There is deliberately no API that
//! accepts replay truth, a simulator world, an opponent discard list, or a
//! realized deck tail.

use crate::cards::{validate_cards, Card, ALL_CARDS};
use crate::counter_rng::{sha256_hex_json, CounterActor, CounterRngKey, COUNTER_RNG_SCHEMA};
use crate::infoset::{ActOrder, ActorObservation, Street};
use serde::{Serialize, Serializer};
use serde_json::{json, Value};

pub const HIDDEN_CARD_BELIEF_SCHEMA: &str = "regular_ofc_hidden_card_belief_v1";
pub const HIDDEN_CARD_PARTICLE_SCHEMA: &str = "regular_ofc_hidden_card_particle_v1";
pub const HIDDEN_CARD_BATCH_SCHEMA: &str = "regular_ofc_hidden_card_particle_batch_v1";
pub const HIDDEN_CARD_PRIOR: &str = "uniform_exchangeable_v1";

const RNG_PHASE: &str = "hidden_card_belief";
const RNG_STREAM: &str = "unknown_card_permutation";
const RNG_DOMAIN: u64 = 1_u64 << 63;
const ATTEMPT_BITS: u32 = 32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HiddenCardParticle {
    pub observation_fingerprint: String,
    pub street: Street,
    pub sample_index: u64,
    pub opponent_private_discards: Vec<Card>,
    pub unseen_deck: Vec<Card>,
    pub rng_key_digest: String,
    pub prior: String,
}

impl HiddenCardParticle {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        observation_fingerprint: impl Into<String>,
        street: Street,
        sample_index: u64,
        opponent_private_discards: Vec<Card>,
        unseen_deck: Vec<Card>,
        rng_key_digest: impl Into<String>,
    ) -> Result<Self, String> {
        let particle = Self {
            observation_fingerprint: observation_fingerprint.into(),
            street,
            sample_index,
            opponent_private_discards,
            unseen_deck,
            rng_key_digest: rng_key_digest.into(),
            prior: HIDDEN_CARD_PRIOR.to_owned(),
        };
        particle.validate_shape()?;
        Ok(particle)
    }

    pub fn hidden_cards(&self) -> Vec<Card> {
        self.opponent_private_discards
            .iter()
            .chain(&self.unseen_deck)
            .copied()
            .collect()
    }

    pub fn draw(&self, count: usize, offset: usize) -> Result<&[Card], String> {
        let end = offset
            .checked_add(count)
            .ok_or_else(|| "draw offset overflow".to_owned())?;
        if end > self.unseen_deck.len() {
            return Err("draw exceeds unseen deck".to_owned());
        }
        Ok(&self.unseen_deck[offset..end])
    }

    pub fn validate_shape(&self) -> Result<(), String> {
        if !self.street.supports_hidden_belief() {
            return Err("hidden-card particles support only T0-T4".to_owned());
        }
        if !is_sha256(&self.observation_fingerprint) {
            return Err("invalid observation fingerprint".to_owned());
        }
        if !is_sha256(&self.rng_key_digest) {
            return Err("invalid RNG key digest".to_owned());
        }
        if self.prior != HIDDEN_CARD_PRIOR {
            return Err(format!("unsupported hidden-card prior: {:?}", self.prior));
        }
        validate_cards(&self.hidden_cards())?;
        Ok(())
    }

    pub fn validate_against(&self, observation: &ActorObservation) -> Result<(), String> {
        self.validate_shape()?;
        validate_decision_observation(observation)?;
        if self.observation_fingerprint != observation.fingerprint() {
            return Err("particle belongs to a different observation".to_owned());
        }
        if self.street != observation.street {
            return Err("particle street disagrees with observation".to_owned());
        }
        if self.opponent_private_discards.len() != observation.opponent_discard_count() {
            return Err("opponent private discard count is inconsistent".to_owned());
        }
        let known_mask = observation
            .known_unavailable_cards()
            .iter()
            .fold(0_u64, |mask, card| mask | card.bit());
        let hidden = self.hidden_cards();
        if hidden.iter().any(|card| known_mask & card.bit() != 0) {
            return Err("particle overlaps actor-visible cards".to_owned());
        }
        let hidden_mask = hidden.iter().fold(0_u64, |mask, card| mask | card.bit());
        if known_mask | hidden_mask != (1_u64 << 52) - 1 {
            return Err("particle does not cover the regular 52-card deck".to_owned());
        }
        Ok(())
    }

    pub fn to_json(&self) -> Value {
        json!({
            "schema": HIDDEN_CARD_PARTICLE_SCHEMA,
            "prior": self.prior,
            "observation_fingerprint": self.observation_fingerprint,
            "street": self.street,
            "sample_index": self.sample_index,
            "opponent_private_discards": self.opponent_private_discards,
            "unseen_deck": self.unseen_deck,
            "rng_key_digest": self.rng_key_digest,
        })
    }

    pub fn digest(&self) -> String {
        sha256_hex_json(&self.to_json())
    }
}

impl Serialize for HiddenCardParticle {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_json().serialize(serializer)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HiddenCardParticleBatch {
    pub observation_fingerprint: String,
    pub street: Street,
    pub base_seed: i64,
    pub run_id: String,
    pub start_index: u64,
    pub particles: Vec<HiddenCardParticle>,
    pub prior: String,
}

impl HiddenCardParticleBatch {
    pub fn new(
        observation_fingerprint: impl Into<String>,
        street: Street,
        base_seed: i64,
        run_id: impl Into<String>,
        start_index: u64,
        particles: Vec<HiddenCardParticle>,
    ) -> Result<Self, String> {
        let batch = Self {
            observation_fingerprint: observation_fingerprint.into(),
            street,
            base_seed,
            run_id: run_id.into(),
            start_index,
            particles,
            prior: HIDDEN_CARD_PRIOR.to_owned(),
        };
        batch.validate_shape()?;
        Ok(batch)
    }

    pub fn validate_shape(&self) -> Result<(), String> {
        if self.run_id.is_empty() {
            return Err("run_id must not be empty".to_owned());
        }
        if self.particles.is_empty() {
            return Err("a belief batch requires at least one particle".to_owned());
        }
        if self.prior != HIDDEN_CARD_PRIOR {
            return Err("unsupported hidden-card prior".to_owned());
        }
        for (offset, particle) in self.particles.iter().enumerate() {
            let expected = self
                .start_index
                .checked_add(offset as u64)
                .ok_or_else(|| "particle sample index overflow".to_owned())?;
            if particle.sample_index != expected {
                return Err("particle sample indices are not contiguous".to_owned());
            }
            if particle.observation_fingerprint != self.observation_fingerprint
                || particle.street != self.street
                || particle.prior != self.prior
            {
                return Err("batch contains a particle from another root".to_owned());
            }
        }
        Ok(())
    }

    pub fn validate_against(&self, observation: &ActorObservation) -> Result<(), String> {
        self.validate_shape()?;
        if observation.fingerprint() != self.observation_fingerprint {
            return Err("batch belongs to a different observation".to_owned());
        }
        for particle in &self.particles {
            particle.validate_against(observation)?;
        }
        Ok(())
    }

    pub fn particle_digests(&self) -> Vec<String> {
        self.particles
            .iter()
            .map(HiddenCardParticle::digest)
            .collect()
    }

    pub fn to_json(&self, include_particles: bool) -> Value {
        let mut value = json!({
            "schema": HIDDEN_CARD_BATCH_SCHEMA,
            "belief_schema": HIDDEN_CARD_BELIEF_SCHEMA,
            "prior": self.prior,
            "counter_rng_schema": COUNTER_RNG_SCHEMA,
            "observation_fingerprint": self.observation_fingerprint,
            "street": self.street,
            "base_seed": self.base_seed,
            "run_id": self.run_id,
            "start_index": self.start_index,
            "sample_count": self.particles.len(),
            "particle_digests": self.particle_digests(),
        });
        if include_particles {
            value["particles"] = Value::Array(
                self.particles
                    .iter()
                    .map(HiddenCardParticle::to_json)
                    .collect(),
            );
        }
        value
    }

    pub fn digest(&self) -> String {
        sha256_hex_json(&self.to_json(false))
    }
}

impl Serialize for HiddenCardParticleBatch {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_json(false).serialize(serializer)
    }
}

pub fn sample_hidden_card_particle(
    observation: &ActorObservation,
    base_seed: i64,
    run_id: &str,
    sample_index: u64,
) -> Result<HiddenCardParticle, String> {
    validate_decision_observation(observation)?;
    if run_id.is_empty() {
        return Err("run_id must not be empty".to_owned());
    }

    let fingerprint = observation.fingerprint();
    let known_mask = observation
        .known_unavailable_cards()
        .iter()
        .fold(0_u64, |mask, card| mask | card.bit());
    let unknown = ALL_CARDS
        .iter()
        .copied()
        .filter(|card| known_mask & card.bit() == 0)
        .collect::<Vec<_>>();
    let mut shuffled = counter_shuffle(
        unknown,
        base_seed,
        run_id,
        sample_index,
        observation.street,
        &fingerprint,
    )?;
    let opponent_discard_count = observation.opponent_discard_count();
    let mut opponent_discards = shuffled.drain(..opponent_discard_count).collect::<Vec<_>>();
    opponent_discards.sort_unstable_by_key(|card| card.index());
    let particle = HiddenCardParticle::new(
        &fingerprint,
        observation.street,
        sample_index,
        opponent_discards,
        shuffled,
        rng_key_digest(
            base_seed,
            run_id,
            sample_index,
            observation.street,
            &fingerprint,
        )?,
    )?;
    particle.validate_against(observation)?;
    Ok(particle)
}

pub fn sample_hidden_card_particles(
    observation: &ActorObservation,
    base_seed: i64,
    run_id: &str,
    sample_count: usize,
    start_index: u64,
) -> Result<HiddenCardParticleBatch, String> {
    validate_decision_observation(observation)?;
    if sample_count == 0 {
        return Err("sample_count must be positive".to_owned());
    }
    let particles = (0..sample_count)
        .map(|offset| {
            let sample_index = start_index
                .checked_add(offset as u64)
                .ok_or_else(|| "particle sample index overflow".to_owned())?;
            sample_hidden_card_particle(observation, base_seed, run_id, sample_index)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let batch = HiddenCardParticleBatch::new(
        observation.fingerprint(),
        observation.street,
        base_seed,
        run_id,
        start_index,
        particles,
    )?;
    batch.validate_against(observation)?;
    Ok(batch)
}

fn validate_decision_observation(observation: &ActorObservation) -> Result<(), String> {
    observation.validate()?;
    if !observation.street.supports_hidden_belief() {
        return Err("hidden-card belief supports only T0-T4 decisions".to_owned());
    }
    let expected = belief_geometry(observation.street, observation.to_act_order);
    let actual = (
        observation.hero_board.card_count(),
        observation.opponent_public_board.card_count(),
        observation.hero_private_discards.len(),
        observation.opponent_discard_count(),
    );
    // Asked per street rather than accepted as either count. A blanket "three
    // or five" would let a T1 root carrying five cards through, and its
    // geometry arm would then be checked against a deal it never has.
    if observation.dealt_cards.len() != observation.street.dealt_card_count() {
        return Err(match observation.street {
            Street::T0 => "T0 observation must contain exactly five dealt cards".to_owned(),
            _ => "T1-T4 observation must contain exactly three dealt cards".to_owned(),
        });
    }
    if actual != expected {
        return Err(format!(
            "inconsistent {} decision geometry: expected hero/opponent/hero-discards/opponent-discards={expected:?}, got {actual:?}",
            match observation.street {
                Street::T0 => "T0",
                _ => "T1-T4",
            }
        ));
    }
    Ok(())
}

/// Hero board, opponent board, hero discards, opponent discards -- per street.
///
/// The two T0 arms are as strict as every other arm rather than permissive:
/// the opening street places all five dealt cards and discards nothing, so
/// both discard counts are zero on both seats, the hero board is empty because
/// the decision is what to put on it, and the opponent board is empty acting
/// first and holds exactly its five placed cards acting second. Nothing else
/// is an opening root, and anything else is refused here.
fn belief_geometry(street: Street, order: ActOrder) -> (usize, usize, usize, usize) {
    match (street, order) {
        (Street::T0, ActOrder::First) => (0, 0, 0, 0),
        (Street::T0, ActOrder::Second) => (0, 5, 0, 0),
        (Street::T1, ActOrder::First) => (5, 5, 0, 0),
        (Street::T1, ActOrder::Second) => (5, 7, 0, 1),
        (Street::T2, ActOrder::First) => (7, 7, 1, 1),
        (Street::T2, ActOrder::Second) => (7, 9, 1, 2),
        (Street::T3, ActOrder::First) => (9, 9, 2, 2),
        (Street::T3, ActOrder::Second) => (9, 11, 2, 3),
        (Street::T4, ActOrder::First) => (11, 11, 3, 3),
        (Street::T4, ActOrder::Second) => (11, 13, 3, 4),
    }
}

fn counter_shuffle(
    mut cards: Vec<Card>,
    base_seed: i64,
    run_id: &str,
    sample_index: u64,
    street: Street,
    root_fingerprint: &str,
) -> Result<Vec<Card>, String> {
    for step in 0..cards.len().saturating_sub(1) {
        let final_index = cards.len() - 1 - step;
        let swap_index = counter_randbelow(
            final_index + 1,
            base_seed,
            run_id,
            sample_index,
            street,
            root_fingerprint,
            step as u64,
        )?;
        cards.swap(final_index, swap_index);
    }
    Ok(cards)
}

#[allow(clippy::too_many_arguments)]
fn counter_randbelow(
    upper_bound: usize,
    base_seed: i64,
    run_id: &str,
    sample_index: u64,
    street: Street,
    root_fingerprint: &str,
    step: u64,
) -> Result<usize, String> {
    if upper_bound == 0 {
        return Err("upper_bound must be positive".to_owned());
    }
    let upper_bound = upper_bound as u64;
    let limit = RNG_DOMAIN - (RNG_DOMAIN % upper_bound);
    let mut attempt = 0_u64;
    loop {
        let counter = (step << ATTEMPT_BITS) | attempt;
        let value = CounterRngKey::new(
            base_seed,
            run_id,
            RNG_PHASE,
            sample_index,
            CounterActor::Chance,
            street.as_str(),
            RNG_STREAM,
            counter,
            root_fingerprint,
        )?
        .seed();
        if value < limit {
            return Ok((value % upper_bound) as usize);
        }
        attempt += 1;
        if attempt >= (1_u64 << ATTEMPT_BITS) {
            return Err("counter RNG rejection sampling exhausted its domain".to_owned());
        }
    }
}

fn rng_key_digest(
    base_seed: i64,
    run_id: &str,
    sample_index: u64,
    street: Street,
    root_fingerprint: &str,
) -> Result<String, String> {
    let key = CounterRngKey::new(
        base_seed,
        run_id,
        RNG_PHASE,
        sample_index,
        CounterActor::Chance,
        street.as_str(),
        RNG_STREAM,
        0,
        root_fingerprint,
    )?;
    Ok(sha256_hex_json(&key.payload()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|character| character.is_ascii_digit() || (b'a'..=b'f').contains(&character))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infoset::{ScoringContext, Seat};
    use crate::state::Board;
    use std::collections::BTreeMap;

    /// The FL EV the cross-language parity vector below was pinned under.
    ///
    /// It is the superseded June constant (10.227020614683454) on purpose.  The
    /// FL EV reaches the sampler only through the observation fingerprint that
    /// keys the counter-based RNG, so its value here is incidental to what the
    /// vector claims -- which is the Fisher-Yates contract, not the economics.
    /// Freezing it means a re-measured FL EV, of which 9.109 is the first,
    /// cannot invalidate a shuffle golden that never depended on the number.
    ///
    /// The Python side pins the same vector under the same constant:
    /// `tests/test_hu_belief.py::_PARITY_VECTOR_FL_EV_14`.
    const PARITY_VECTOR_FL_EV_14: f64 = 10.227_020_614_683_454;

    fn parity_vector_scoring() -> ScoringContext {
        ScoringContext {
            fl_ev: BTreeMap::from([(14, PARITY_VECTOR_FL_EV_14)]),
            ..ScoringContext::default()
        }
    }

    fn board(cards: &[Card]) -> Board {
        let top_count = cards.len().min(3);
        let middle_count = (cards.len() - top_count).min(5);
        Board::new(
            cards[..top_count].to_vec(),
            cards[top_count..top_count + middle_count].to_vec(),
            cards[top_count + middle_count..].to_vec(),
        )
        .unwrap()
    }

    fn observation(street: Street, order: ActOrder) -> ActorObservation {
        observation_with_scoring(street, order, ScoringContext::default())
    }

    fn observation_with_scoring(
        street: Street,
        order: ActOrder,
        scoring: ScoringContext,
    ) -> ActorObservation {
        let (hero_count, opponent_count, discard_count) = match (street, order) {
            (Street::T1, ActOrder::First) => (5, 5, 0),
            (Street::T1, ActOrder::Second) => (5, 7, 0),
            (Street::T2, ActOrder::First) => (7, 7, 1),
            (Street::T2, ActOrder::Second) => (7, 9, 1),
            (Street::T3, ActOrder::First) => (9, 9, 2),
            (Street::T3, ActOrder::Second) => (9, 11, 2),
            (Street::T4, ActOrder::First) => (11, 11, 3),
            (Street::T4, ActOrder::Second) => (11, 13, 3),
            _ => unreachable!(),
        };
        let mut cursor = 0;
        let hero = board(&ALL_CARDS[cursor..cursor + hero_count]);
        cursor += hero_count;
        let opponent = board(&ALL_CARDS[cursor..cursor + opponent_count]);
        cursor += opponent_count;
        let dealt = ALL_CARDS[cursor..cursor + 3].to_vec();
        cursor += 3;
        let discards = ALL_CARDS[cursor..cursor + discard_count].to_vec();
        ActorObservation::new(
            hero,
            opponent,
            dealt,
            discards,
            match order {
                ActOrder::First => Seat::First,
                ActOrder::Second => Seat::Second,
            },
            street,
            order,
            scoring,
        )
        .unwrap()
    }

    #[test]
    fn python_fisher_yates_particle_and_batch_golden_vectors() {
        // Built under the frozen parity constant, not the production default:
        // see `PARITY_VECTOR_FL_EV_14`.
        let observation =
            observation_with_scoring(Street::T2, ActOrder::Second, parity_vector_scoring());
        let batch =
            sample_hidden_card_particles(&observation, 918273, "belief-determinism", 1, 0).unwrap();
        let golden = &batch.particles[0];
        assert_eq!(
            golden
                .opponent_private_discards
                .iter()
                .map(|c| c.as_str())
                .collect::<Vec<_>>(),
            vec!["Td", "8c"]
        );
        assert_eq!(
            golden.unseen_deck[..8]
                .iter()
                .map(|c| c.as_str())
                .collect::<Vec<_>>(),
            vec!["7s", "2s", "Ts", "8s", "6s", "Kc", "As", "Ad"]
        );
        assert_eq!(
            golden.rng_key_digest,
            "9ebfccec564550201568f4f5b9b1ce7ada5eb8a5aff18fa61370557e746e02ce"
        );
        assert_eq!(
            golden.digest(),
            "2828abbe316c82ad99df2a431899fb7f6bb26055857a3216ef1133518bc0ffc3"
        );
        assert_eq!(
            batch.digest(),
            "d4ece6923ef02c6e7ad18e9baeca7fa60c8563dcd93b266893f404aebcb2cd11"
        );
    }

    #[test]
    fn every_t1_t4_particle_is_a_complete_hidden_partition() {
        for street in [Street::T1, Street::T2, Street::T3, Street::T4] {
            for order in [ActOrder::First, ActOrder::Second] {
                let observation = observation(street, order);
                let particle =
                    sample_hidden_card_particle(&observation, 2026071201, "partition", 2).unwrap();
                particle.validate_against(&observation).unwrap();
                assert_eq!(
                    particle.opponent_private_discards.len(),
                    observation.opponent_discard_count()
                );
                assert_eq!(
                    observation.known_unavailable_cards().len() + particle.hidden_cards().len(),
                    52
                );
            }
        }
    }

    #[test]
    fn sampling_is_repeatable_shard_addressable_and_truth_independent() {
        let observation = observation(Street::T4, ActOrder::First);
        let whole = sample_hidden_card_particles(&observation, 77, "shards", 6, 0).unwrap();
        let repeated = sample_hidden_card_particles(&observation, 77, "shards", 6, 0).unwrap();
        let left = sample_hidden_card_particles(&observation, 77, "shards", 2, 0).unwrap();
        let right = sample_hidden_card_particles(&observation, 77, "shards", 4, 2).unwrap();
        assert_eq!(whole, repeated);
        assert_eq!(
            whole.particle_digests(),
            left.particle_digests()
                .into_iter()
                .chain(right.particle_digests())
                .collect::<Vec<_>>()
        );

        // These stand in for two different offline opponent-discard truths.
        // Neither can be passed to the sampler, so the result remains solely a
        // function of actor-visible observation and explicit RNG coordinates.
        let hidden_truth_a = ALL_CARDS[50];
        let hidden_truth_b = ALL_CARDS[51];
        assert_ne!(hidden_truth_a, hidden_truth_b);
        assert_eq!(
            whole,
            sample_hidden_card_particles(&observation, 77, "shards", 6, 0).unwrap()
        );
    }

    fn t0_observation(order: ActOrder) -> ActorObservation {
        let (opponent, dealt) = match order {
            ActOrder::First => (Board::empty(), ALL_CARDS[0..5].to_vec()),
            ActOrder::Second => (board(&ALL_CARDS[0..5]), ALL_CARDS[5..10].to_vec()),
        };
        ActorObservation::new(
            Board::empty(),
            opponent,
            dealt,
            Vec::new(),
            match order {
                ActOrder::First => Seat::First,
                ActOrder::Second => Seat::Second,
            },
            Street::T0,
            order,
            ScoringContext::default(),
        )
        .unwrap()
    }

    /// The opening street is the one root where nothing has been discarded yet,
    /// so the particle is the entire remainder of the deck and the assertion is
    /// that none of it went missing: 47 cards behind a first-seat root that has
    /// seen only its own five, 42 behind a second-seat root that has also seen
    /// the opponent's five.
    #[test]
    fn t0_roots_sample_particles_covering_the_whole_hidden_deck() {
        for (order, expected_unseen) in [(ActOrder::First, 47), (ActOrder::Second, 42)] {
            let observation = t0_observation(order);
            let particle =
                sample_hidden_card_particle(&observation, 2026072901, "t0-partition", 3).unwrap();
            particle.validate_against(&observation).unwrap();
            assert!(
                particle.opponent_private_discards.is_empty(),
                "the opening street discards nothing, so neither seat has a discard to hide"
            );
            assert_eq!(particle.unseen_deck.len(), expected_unseen);
            assert_eq!(
                observation.known_unavailable_cards().len() + particle.hidden_cards().len(),
                52
            );

            let batch =
                sample_hidden_card_particles(&observation, 2026072901, "t0-batch", 2, 0).unwrap();
            batch.validate_against(&observation).unwrap();
        }
    }

    /// Admitting T0 admits exactly the two opening geometries.
    ///
    /// Each case below is a plausible near-miss rather than nonsense: a root
    /// dealt three cards is what every later street looks like, a nonzero hero
    /// discard list is what T0 would look like if it discarded, and a seven-card
    /// opponent board is the T1 second seat's view. All three are refused.
    #[test]
    fn malformed_t0_roots_are_refused_by_the_belief_boundary() {
        let base = t0_observation(ActOrder::Second);

        let mut dealt_three = base.clone();
        dealt_three.dealt_cards.truncate(3);
        assert!(sample_hidden_card_particle(&dealt_three, 5, "t0-bad-deal", 0).is_err());

        let mut discarding = base.clone();
        discarding.hero_private_discards = vec![ALL_CARDS[20]];
        assert!(sample_hidden_card_particle(&discarding, 5, "t0-bad-discard", 0).is_err());

        let mut opponent_seven = base.clone();
        opponent_seven.opponent_public_board = board(&ALL_CARDS[30..37]);
        assert!(sample_hidden_card_particle(&opponent_seven, 5, "t0-bad-opponent", 0).is_err());

        // And a particle that claims a street it was not drawn for stays refused
        // by shape alone, which is what keeps the two T0 geometries from
        // borrowing another street's continuation.
        let mut mislabelled =
            sample_hidden_card_particle(&base, 5, "t0-mislabel", 0).unwrap();
        mislabelled.street = Street::T1;
        assert!(mislabelled.validate_against(&base).is_err());
    }

    /// The relaxation is per street, not a blanket "three or five".
    ///
    /// If the deal check ever became a disjunction, a T1 root carrying five
    /// cards would validate as though it were an opening and then be scored
    /// against T1's geometry arm. The count is therefore asked of the street.
    #[test]
    fn the_dealt_count_is_asked_per_street_not_accepted_as_either() {
        assert_eq!(Street::T0.dealt_card_count(), 5);
        for street in [Street::T1, Street::T2, Street::T3, Street::T4] {
            assert_eq!(street.dealt_card_count(), 3, "{street:?} deals three");
        }

        let mut t1_dealt_five = observation(Street::T1, ActOrder::First);
        t1_dealt_five.dealt_cards = ALL_CARDS[20..25].to_vec();
        let error = sample_hidden_card_particle(&t1_dealt_five, 5, "t1-five", 0)
            .expect_err("a T1 root dealt five cards is not an opening");
        assert!(error.contains("geometry"), "got {error}");
    }

    #[test]
    fn visible_overlap_and_wrong_root_fail_closed() {
        let root = observation(Street::T3, ActOrder::Second);
        let mut particle = sample_hidden_card_particle(&root, 31, "validation", 0).unwrap();
        particle.unseen_deck[0] = root.hero_board.top[0];
        let error = particle.validate_against(&root).unwrap_err();
        assert!(error.contains("duplicate") || error.contains("overlaps"));

        let other = observation(Street::T3, ActOrder::First);
        assert!(sample_hidden_card_particle(&other, 31, "validation", 0)
            .unwrap()
            .validate_against(&root)
            .is_err());
    }
}
