//! Information-set-safe policy/search roots for heads-up regular OFC.
//!
//! `ActorObservation` deliberately has no field for an opponent's private
//! discards, realized deck tail, or simulator world.  Hidden state may only be
//! represented by particles sampled from this public observation.

use crate::cards::{validate_cards, Card, ALL_CARDS};
use crate::counter_rng::sha256_hex_json;
use crate::state::Board;
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, Map, Number, Value};
use std::collections::BTreeMap;

pub const OBSERVATION_SCHEMA: &str = "regular_ofc_actor_observation_v1";
pub const SCORING_CONTEXT_SCHEMA: &str = "regular_ofc_scoring_context_v1";
/// Mirrors `configs/fl_ev_regular_v4_selfplay.json` (M6 run B, 2026-08-06).
/// Every decode path takes `fl_ev` from the observation JSON, which is
/// required, so this only backs `ScoringContext::default()` -- notably the
/// scalar RL V1 pin in `hu_rl_engine`. It supersedes 9.109, which was measured
/// against a STATIC Fantasyland side and was therefore a floor; v4 is the fixed
/// point of 150,000 self-play hands with Fantasyland played adaptively on both
/// sides. 10.227020614683454 before that.
pub const DEFAULT_FL_EV: f64 = 9.6;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Seat {
    First,
    Second,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActOrder {
    First,
    Second,
}

/// Compatibility spelling for call sites that prefer the longer name.
pub type ActionOrder = ActOrder;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Street {
    T0,
    T1,
    T2,
    T3,
    T4,
}

impl Street {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::T0 => "T0",
            Self::T1 => "T1",
            Self::T2 => "T2",
            Self::T3 => "T3",
            Self::T4 => "T4",
        }
    }

    /// Whether hidden state at this street may be represented by particles.
    ///
    /// T0 was excluded while nothing sampled from an opening root; the T0
    /// teacher does, so it is admitted here. Every street the enum currently
    /// names is now listed, and the match is kept exhaustive rather than
    /// collapsed to `true` so that a street added later -- Fantasy Land is the
    /// obvious candidate, and its geometry is not this one -- has to be
    /// admitted deliberately instead of inheriting support by default.
    pub const fn supports_hidden_belief(self) -> bool {
        matches!(
            self,
            Self::T0 | Self::T1 | Self::T2 | Self::T3 | Self::T4
        )
    }

    /// How many cards this street deals to the actor.
    ///
    /// The opening street deals five and places all of them; every later street
    /// deals three and discards one. Callers that check the deal must ask per
    /// street rather than accept either count, or a T1 root carrying five cards
    /// would validate as though it were an opening.
    pub const fn dealt_card_count(self) -> usize {
        match self {
            Self::T0 => 5,
            Self::T1 | Self::T2 | Self::T3 | Self::T4 => 3,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoringContext {
    pub fl_ev: BTreeMap<u8, f64>,
    pub middle_trips_royalty: i32,
    pub hu_line_points: bool,
    pub scoop_bonus: i32,
    pub foul_enabled: bool,
    pub fantasyland_cards: u8,
}

impl Default for ScoringContext {
    fn default() -> Self {
        Self {
            fl_ev: BTreeMap::from([(14, DEFAULT_FL_EV)]),
            middle_trips_royalty: 2,
            hu_line_points: true,
            scoop_bonus: 3,
            foul_enabled: true,
            fantasyland_cards: 14,
        }
    }
}

impl ScoringContext {
    pub fn validate(&self) -> Result<(), String> {
        if self.fl_ev.is_empty()
            || self
                .fl_ev
                .iter()
                .any(|(&cards, &value)| cards == 0 || !value.is_finite())
        {
            return Err("fl_ev must contain finite values for positive card counts".to_owned());
        }
        if self.middle_trips_royalty != 2 {
            return Err("regular OFC middle trips royalty must be 2".to_owned());
        }
        if self.fantasyland_cards != 14 {
            return Err("regular HU fantasyland uses 14 cards".to_owned());
        }
        if !self.hu_line_points || self.scoop_bonus != 3 || !self.foul_enabled {
            return Err(
                "M3 supports standard HU line points, scoop bonus 3, and foul scoring only"
                    .to_owned(),
            );
        }
        Ok(())
    }

    pub fn to_json(&self) -> Value {
        let fl_ev = self
            .fl_ev
            .iter()
            .map(|(&cards, &value)| {
                (
                    cards.to_string(),
                    Value::Number(Number::from_f64(value).expect("validated finite FL EV")),
                )
            })
            .collect::<Map<_, _>>();
        json!({
            "schema": SCORING_CONTEXT_SCHEMA,
            "fl_ev": Value::Object(fl_ev),
            "middle_trips_royalty": self.middle_trips_royalty,
            "hu_line_points": self.hu_line_points,
            "scoop_bonus": self.scoop_bonus,
            "foul_enabled": self.foul_enabled,
            "fantasyland_cards": self.fantasyland_cards,
        })
    }

    fn from_json(value: Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or_else(|| "scoring context must be a mapping".to_owned())?;
        reject_unknown_keys(
            object,
            &[
                "schema",
                "fl_ev",
                "middle_trips_royalty",
                "hu_line_points",
                "scoop_bonus",
                "foul_enabled",
                "fantasyland_cards",
            ],
            "scoring context",
        )?;
        if object.get("schema").and_then(Value::as_str) != Some(SCORING_CONTEXT_SCHEMA) {
            return Err("unsupported scoring context schema".to_owned());
        }
        let raw_fl_ev = object
            .get("fl_ev")
            .and_then(Value::as_object)
            .ok_or_else(|| "scoring context fl_ev must be a mapping".to_owned())?;
        let mut fl_ev = BTreeMap::new();
        for (cards, value) in raw_fl_ev {
            let cards = cards
                .parse::<u8>()
                .map_err(|_| "invalid scoring context fl_ev card count".to_owned())?;
            let value = value
                .as_f64()
                .ok_or_else(|| "invalid scoring context fl_ev value".to_owned())?;
            fl_ev.insert(cards, value);
        }
        let context = Self {
            fl_ev,
            middle_trips_royalty: integer(object, "middle_trips_royalty", 2)? as i32,
            hu_line_points: boolean(object, "hu_line_points", true)?,
            scoop_bonus: integer(object, "scoop_bonus", 3)? as i32,
            foul_enabled: boolean(object, "foul_enabled", true)?,
            fantasyland_cards: integer(object, "fantasyland_cards", 14)? as u8,
        };
        context.validate()?;
        Ok(context)
    }
}

impl Serialize for ScoringContext {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_json().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ScoringContext {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_json(Value::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

/// The only card-bearing root accepted by M3 policy/search/belief code.
#[derive(Clone, Debug, PartialEq)]
pub struct ActorObservation {
    pub hero_board: Board,
    pub opponent_public_board: Board,
    pub dealt_cards: Vec<Card>,
    pub hero_private_discards: Vec<Card>,
    pub seat: Seat,
    pub street: Street,
    pub to_act_order: ActOrder,
    pub scoring: ScoringContext,
    pub hero_in_fantasyland: bool,
    pub opponent_in_fantasyland: bool,
}

impl ActorObservation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hero_board: Board,
        opponent_public_board: Board,
        dealt_cards: Vec<Card>,
        hero_private_discards: Vec<Card>,
        seat: Seat,
        street: Street,
        to_act_order: ActOrder,
        scoring: ScoringContext,
    ) -> Result<Self, String> {
        let observation = Self {
            hero_board,
            opponent_public_board,
            dealt_cards,
            hero_private_discards,
            seat,
            street,
            to_act_order,
            scoring,
            hero_in_fantasyland: false,
            opponent_in_fantasyland: false,
        };
        observation.validate()?;
        Ok(observation)
    }

    /// The same decision point with a Fantasyland opponent: nothing opposite,
    /// at any street.
    ///
    /// A separate constructor rather than a flag on [`Self::new`], so that the
    /// hundred-odd existing call sites keep the signature they have and a
    /// hidden-opponent observation has to be asked for by name. The flag itself
    /// is not new -- `opponent_in_fantasyland` has ridden along in the schema,
    /// the JSON and the fingerprint since this type was written, and was
    /// refused by the validator. What is new is that it now MEANS something.
    #[allow(clippy::too_many_arguments)]
    pub fn new_vs_fantasyland(
        hero_board: Board,
        dealt_cards: Vec<Card>,
        hero_private_discards: Vec<Card>,
        seat: Seat,
        street: Street,
        to_act_order: ActOrder,
        scoring: ScoringContext,
    ) -> Result<Self, String> {
        let observation = Self {
            hero_board,
            // Not a parameter: there is no opponent board to pass. A caller
            // holding one is describing a different situation.
            opponent_public_board: Board::empty(),
            dealt_cards,
            hero_private_discards,
            seat,
            street,
            to_act_order,
            scoring,
            hero_in_fantasyland: false,
            opponent_in_fantasyland: true,
        };
        observation.validate()?;
        Ok(observation)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.hero_board.validate()?;
        self.opponent_public_board.validate()?;
        self.scoring.validate()?;
        let all_visible = self
            .hero_board
            .all_cards()
            .into_iter()
            .chain(self.opponent_public_board.all_cards())
            .chain(self.dealt_cards.iter().copied())
            .chain(self.hero_private_discards.iter().copied())
            .collect::<Vec<_>>();
        validate_cards(&all_visible)?;

        let expected =
            regular_decision_geometry(self.street, self.to_act_order, self.opponent_in_fantasyland);
        let actual = (
            self.hero_board.card_count(),
            self.opponent_public_board.card_count(),
            self.dealt_cards.len(),
            self.hero_private_discards.len(),
        );
        if actual != expected {
            return Err(format!(
                "inconsistent regular decision geometry: street/order={}/{}{} expected hero/opponent/dealt/hero-discards={expected:?}, got {actual:?}",
                self.street.as_str(),
                match self.to_act_order { ActOrder::First => "first", ActOrder::Second => "second" },
                if self.opponent_in_fantasyland { " vs-fantasyland" } else { "" }
            ));
        }
        if !matches!(
            (self.seat, self.to_act_order),
            (Seat::First, ActOrder::First) | (Seat::Second, ActOrder::Second)
        ) {
            return Err("regular HU seat must match within-street action order".to_owned());
        }
        // The hero's own Fantasyland is a different GAME, not a different view
        // of this one: the hero receives fourteen cards and sets thirteen of
        // them in a single action, so the action space, the street sequence and
        // the scoring all differ from anything this observation can describe.
        // It stays refused.
        //
        // The opponent's Fantasyland is a different VIEW. The hero plays the
        // ordinary five streets with the ordinary action space; what changes is
        // that nothing ever appears opposite. That is representable here, and
        // above it is: the geometry table asks for a zero-card opponent board at
        // every street rather than the usual count.
        if self.hero_in_fantasyland {
            return Err(
                "hero_in_fantasyland requires the FL observation schema: the hero \
                 sets thirteen of fourteen cards in one action, which this action \
                 space cannot describe"
                    .to_owned(),
            );
        }
        Ok(())
    }

    /// True when the opponent's hand is permanently unobservable.
    ///
    /// Named rather than read off the flag at each use, because the two things
    /// the callers care about -- "the opponent block cannot be computed" and
    /// "the opponent board is empty" -- are consequences of this and not of the
    /// board being empty for some other reason. At T0 first seat the opponent's
    /// board is also empty, and that one is going to fill in.
    pub fn opponent_hidden(&self) -> bool {
        self.opponent_in_fantasyland
    }

    pub fn opponent_discard_count(&self) -> usize {
        self.opponent_public_board
            .card_count()
            .saturating_sub(5)
            .checked_div(2)
            .unwrap_or(0)
            .min(4)
    }

    pub fn legacy_dead_cards(&self) -> Vec<Card> {
        self.opponent_public_board
            .all_cards()
            .into_iter()
            .chain(self.hero_private_discards.iter().copied())
            .collect()
    }

    /// Actor-visible unavailable cards in Python `ALL_CARDS` order.
    pub fn known_unavailable_cards(&self) -> Vec<Card> {
        let mut mask = 0_u64;
        for card in self
            .hero_board
            .all_cards()
            .into_iter()
            .chain(self.opponent_public_board.all_cards())
            .chain(self.dealt_cards.iter().copied())
            .chain(self.hero_private_discards.iter().copied())
        {
            mask |= card.bit();
        }
        ALL_CARDS
            .iter()
            .copied()
            .filter(|card| mask & card.bit() != 0)
            .collect()
    }

    /// Order-invariant SHA-256 fingerprint matching Python M2 exactly.
    pub fn fingerprint(&self) -> String {
        self.validate().expect("invalid ActorObservation");
        sha256_hex_json(&self.canonical_fingerprint_payload())
    }

    pub fn to_json(&self) -> Value {
        json!({
            "schema": OBSERVATION_SCHEMA,
            "hero_board": self.hero_board,
            "opponent_public_board": self.opponent_public_board,
            "dealt_cards": self.dealt_cards,
            "hero_private_discards": self.hero_private_discards,
            "seat": self.seat,
            "street": self.street,
            "to_act_order": self.to_act_order,
            "scoring": self.scoring,
            "hero_in_fantasyland": self.hero_in_fantasyland,
            "opponent_in_fantasyland": self.opponent_in_fantasyland,
            "opponent_discard_count": self.opponent_discard_count(),
        })
    }

    fn canonical_fingerprint_payload(&self) -> Value {
        let sorted_board = |board: &Board| {
            let mut top = board.top.clone();
            let mut middle = board.middle.clone();
            let mut bottom = board.bottom.clone();
            top.sort_unstable_by_key(|card| card.index());
            middle.sort_unstable_by_key(|card| card.index());
            bottom.sort_unstable_by_key(|card| card.index());
            json!({"top": top, "middle": middle, "bottom": bottom})
        };
        let mut dealt = self.dealt_cards.clone();
        let mut discards = self.hero_private_discards.clone();
        dealt.sort_unstable_by_key(|card| card.index());
        discards.sort_unstable_by_key(|card| card.index());
        json!({
            "schema": OBSERVATION_SCHEMA,
            "hero_board": sorted_board(&self.hero_board),
            "opponent_public_board": sorted_board(&self.opponent_public_board),
            "dealt_cards": dealt,
            "hero_private_discards": discards,
            "seat": self.seat,
            "street": self.street,
            "to_act_order": self.to_act_order,
            "scoring": self.scoring,
            "hero_in_fantasyland": self.hero_in_fantasyland,
            "opponent_in_fantasyland": self.opponent_in_fantasyland,
        })
    }

    fn from_json(value: Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or_else(|| "actor observation must be a mapping".to_owned())?;
        reject_unknown_keys(
            object,
            &[
                "schema",
                "hero_board",
                "opponent_public_board",
                "dealt_cards",
                "hero_private_discards",
                "seat",
                "street",
                "to_act_order",
                "scoring",
                "hero_in_fantasyland",
                "opponent_in_fantasyland",
                "opponent_discard_count",
            ],
            "actor observation",
        )?;
        if object.get("schema").and_then(Value::as_str) != Some(OBSERVATION_SCHEMA) {
            return Err("unsupported actor observation schema".to_owned());
        }
        let hero_board = decode(object, "hero_board")?;
        let opponent_public_board = decode(object, "opponent_public_board")?;
        let dealt_cards = decode(object, "dealt_cards")?;
        let hero_private_discards = decode(object, "hero_private_discards")?;
        let seat = decode(object, "seat")?;
        let street = decode(object, "street")?;
        let to_act_order = decode(object, "to_act_order")?;
        let scoring = decode(object, "scoring")?;
        let observation = Self {
            hero_board,
            opponent_public_board,
            dealt_cards,
            hero_private_discards,
            seat,
            street,
            to_act_order,
            scoring,
            hero_in_fantasyland: boolean(object, "hero_in_fantasyland", false)?,
            opponent_in_fantasyland: boolean(object, "opponent_in_fantasyland", false)?,
        };
        observation.validate()?;
        let declared_count = object.get("opponent_discard_count").and_then(Value::as_u64);
        if declared_count
            .is_some_and(|value| value as usize != observation.opponent_discard_count())
        {
            return Err("opponent_discard_count disagrees with public board".to_owned());
        }
        Ok(observation)
    }
}

impl Serialize for ActorObservation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_json().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ActorObservation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_json(Value::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

fn reject_unknown_keys(
    object: &Map<String, Value>,
    allowed: &[&str],
    context: &str,
) -> Result<(), String> {
    let mut unknown = object
        .keys()
        .filter(|key| !allowed.contains(&key.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    unknown.sort();
    if unknown.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{context} contains unknown fields: {}",
            unknown.join(", ")
        ))
    }
}

/// Cards each side shows at a decision point: `(hero, opponent, dealt,
/// hero-discards)`.
///
/// `opponent_hidden` selects between two tables rather than relaxing one. A
/// Fantasyland opponent takes its whole hand face down and never places a card
/// where the hero can see it, so at EVERY street its public board holds zero
/// cards -- not "fewer than usual", zero, from the first street to the last.
/// Reading that as a looser bound on the ordinary table would accept a partial
/// board as well, and a partial board against a hidden opponent is a hand
/// nobody is playing.
///
/// The hero's own three columns are untouched. It is the same player making the
/// same decisions with the same cards; only the information opposite it is
/// gone.
pub(crate) fn regular_decision_geometry(
    street: Street,
    order: ActOrder,
    opponent_hidden: bool,
) -> (usize, usize, usize, usize) {
    let (hero, opponent, dealt, discards) = match (street, order) {
        (Street::T0, ActOrder::First) => (0, 0, 5, 0),
        (Street::T0, ActOrder::Second) => (0, 5, 5, 0),
        (Street::T1, ActOrder::First) => (5, 5, 3, 0),
        (Street::T1, ActOrder::Second) => (5, 7, 3, 0),
        (Street::T2, ActOrder::First) => (7, 7, 3, 1),
        (Street::T2, ActOrder::Second) => (7, 9, 3, 1),
        (Street::T3, ActOrder::First) => (9, 9, 3, 2),
        (Street::T3, ActOrder::Second) => (9, 11, 3, 2),
        (Street::T4, ActOrder::First) => (11, 11, 3, 3),
        (Street::T4, ActOrder::Second) => (11, 13, 3, 3),
    };
    if opponent_hidden {
        (hero, 0, dealt, discards)
    } else {
        (hero, opponent, dealt, discards)
    }
}

fn decode<T: for<'de> Deserialize<'de>>(
    object: &Map<String, Value>,
    key: &str,
) -> Result<T, String> {
    serde_json::from_value(
        object
            .get(key)
            .cloned()
            .ok_or_else(|| format!("actor observation lacks {key}"))?,
    )
    .map_err(|error| format!("invalid actor observation {key}: {error}"))
}

fn integer(object: &Map<String, Value>, key: &str, default: i64) -> Result<i64, String> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_i64()
            .ok_or_else(|| format!("{key} must be an integer")),
    }
}

fn boolean(object: &Map<String, Value>, key: &str, default: bool) -> Result<bool, String> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| format!("{key} must be a boolean")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The FL EV the cross-language fingerprint vectors below were pinned
    /// under.
    ///
    /// It is the superseded June constant (10.227020614683454) on purpose.  The
    /// vectors pin the *fingerprint contract* -- which observation fields enter
    /// the digest, in what canonical order, and that the digest is stable
    /// across the two implementations.  The FL EV is one of the values inside
    /// it, so its number is incidental to the claim; freezing it keeps the
    /// vectors valid across an FL EV re-measurement (9.109 being the first)
    /// instead of forcing eight hashes to be re-derived every time the
    /// economics move.
    ///
    /// The production default is exercised elsewhere -- `ScoringContext`'s own
    /// tests and every result fixture that is *supposed* to move with the
    /// economics, notably `tests/t0_pruning_safety.rs`.
    const FINGERPRINT_VECTOR_FL_EV_14: f64 = 10.227_020_614_683_454;

    fn fingerprint_vector_scoring() -> ScoringContext {
        ScoringContext {
            fl_ev: BTreeMap::from([(14, FINGERPRINT_VECTOR_FL_EV_14)]),
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
        // The ordinary table: this helper builds ordinary observations, and a
        // hidden-opponent one has its own constructor and its own tests.
        let (hero_count, opponent_count, _, discard_count) =
            regular_decision_geometry(street, order, false);
        let mut cursor = 0;
        let hero = board(&ALL_CARDS[cursor..cursor + hero_count]);
        cursor += hero_count;
        let opponent = board(&ALL_CARDS[cursor..cursor + opponent_count]);
        cursor += opponent_count;
        let dealt = ALL_CARDS[cursor..cursor + if street == Street::T0 { 5 } else { 3 }].to_vec();
        cursor += dealt.len();
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
    fn python_fingerprints_match_every_t1_t4_geometry() {
        let vectors = [
            (
                Street::T1,
                ActOrder::First,
                "f636cee15cbb7299b43feebebb47554eb71441bd086ec38d124d907a7a2e83aa",
            ),
            (
                Street::T1,
                ActOrder::Second,
                "3f26eb458dc0a2b88e1c3f00ab77d5c23ed3584bbe365ff8f2c472de821179fc",
            ),
            (
                Street::T2,
                ActOrder::First,
                "dc381d5951fb30824b06f07d058f2c11532d9a339fedd311dafc62064ab1c2a9",
            ),
            (
                Street::T2,
                ActOrder::Second,
                "ace03ec568b1d274fbfaea3406ff9d3f6207c9879755b974c7d96adfd3f24dd3",
            ),
            (
                Street::T3,
                ActOrder::First,
                "a788605e53749ce1037d10ff4387619e2d3a4dbaa1367b47a1cc1eb6ff911ecf",
            ),
            (
                Street::T3,
                ActOrder::Second,
                "6fff2986318d2935381904af932baa20ad55bda59cf1a975cde3deb002c636c3",
            ),
            (
                Street::T4,
                ActOrder::First,
                "503fe1d68cf6284c7913f3ea9cb1d6a61764534ea561d8ed1e0e5f93b618dcf8",
            ),
            (
                Street::T4,
                ActOrder::Second,
                "14fa39f51f6a71347d297ea35b6ae7d004892ca02680fb11ef58cb3db5a78af7",
            ),
        ];
        for (street, order, expected) in vectors {
            // Built under the frozen vector constant, not the production
            // default: see `FINGERPRINT_VECTOR_FL_EV_14`.
            assert_eq!(
                observation_with_scoring(street, order, fingerprint_vector_scoring()).fingerprint(),
                expected
            );
        }
    }

    #[test]
    fn fingerprint_is_order_invariant_and_truth_cannot_enter_observation() {
        let original = observation(Street::T3, ActOrder::First);
        let mut reordered = original.clone();
        reordered.hero_board.top.reverse();
        reordered.hero_board.middle.reverse();
        reordered.opponent_public_board.bottom.reverse();
        reordered.dealt_cards.reverse();
        reordered.hero_private_discards.reverse();
        assert_eq!(original.fingerprint(), reordered.fingerprint());

        let json = serde_json::to_string(&original).unwrap();
        for forbidden in [
            "opponent_private_discards",
            "true_dead_cards",
            "remaining_deck",
            "world_state",
        ] {
            assert!(!json.contains(forbidden));
        }
        // Different replay truth values have nowhere to enter the constructor,
        // therefore cannot alter policy-visible state or its fingerprint.
        let first_hidden_truth = ALL_CARDS[50];
        let second_hidden_truth = ALL_CARDS[51];
        assert_ne!(first_hidden_truth, second_hidden_truth);
        assert_eq!(original.fingerprint(), original.clone().fingerprint());
    }

    #[test]
    fn geometry_and_cross_zone_duplicates_fail_closed() {
        let mut invalid = observation(Street::T4, ActOrder::First);
        invalid.dealt_cards[0] = invalid.hero_board.top[0];
        assert!(invalid.validate().unwrap_err().contains("duplicate"));

        let valid = observation(Street::T2, ActOrder::First);
        let mut payload = valid.to_json();
        payload["opponent_discard_count"] = json!(4);
        assert!(serde_json::from_value::<ActorObservation>(payload).is_err());
    }
}
