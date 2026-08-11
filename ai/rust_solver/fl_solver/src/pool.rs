//! A pre-solved pool of Fantasyland best-response frontiers.
//!
//! # Why a pool is exact rather than an approximation
//!
//! A uniform 14-card hand drawn from the 54-card deck, conditioned on sharing
//! no card with the cards hero can see, is distributed exactly as a uniform
//! 14-card hand from what hero left.  That is the definition of conditioning,
//! not a sampling argument, so hands can be solved once, in advance, and drawn
//! from later by a disjointness test.  Nothing about the draw is approximate;
//! the only error is Monte Carlo over how many entries are drawn.
//!
//! # The disjointness test, and why a bitmask is not enough here
//!
//! The regular deck packs 52 cards into a `u64` and the test is one AND.  The
//! joker deck has two more cards and they are **identical**: `Card { rank: 0,
//! suit: 4 }` twice.  Giving each a bit would claim hero's `X1` blocks a pool
//! hand's `X1` but not its `X2`, which is meaningless when the two are the same
//! object.  So an entry carries a 52-bit natural mask **and a joker count**,
//! and is usable by a hero holding `hero_jokers` jokers iff
//!
//! ```text
//! entry.naturals & hero_naturals == 0  &&  entry.jokers + hero_jokers <= 2
//! ```
//!
//! An earlier library in this repo set both joker bits for either joker and
//! then cleared the high one when exactly one was seen.  That happens to give
//! the right answer for one joker and the wrong one for two; the count is what
//! the rule actually is.
//!
//! # What is stored, and what is not
//!
//! Rows are stored **unpriced**: three canonical row values, the royalty they
//! pay, and whether the arrangement keeps Fantasyland.  `static_value` is
//! recomputed at load from the header's table.  That is licensed by
//! `frontier::tests::frontier_membership_is_invariant_across_nonnegative_fl_ev`
//! -- which rows survive does not depend on the constant while it is
//! non-negative -- and it is what makes a re-derived table a **re-run** rather
//! than a rebuild.
//!
//! It stops being licensed below zero, which is why the header pins the whole
//! table and the loader refuses a mismatch rather than re-pricing silently.  A
//! pool built under a superseded table is not a stale pool, it is a wrong one,
//! and nothing downstream would notice.
//!
//! # Format `JFL1`, little-endian throughout
//!
//! ```text
//! header  magic      [u8; 4]  = b"JFL1"
//!         version    u32      = 1
//!         width      u32      Fantasyland card count this pool is for
//!         entries    u64      number of entries
//!         fl_ev      [f64; 4] the WHOLE table, widths 14..17
//!         seed       u64      the deal stream this pool was built from
//!         reserved   [u8; 16] zero
//! entry   naturals   u64      52-bit mask of the hand's natural cards
//!         jokers     u32      0, 1 or 2
//!         rows       u32      frontier row count
//!         row[..]    top u32, mid u32, bot u32, royalty i32, stays u32
//! ```
//!
//! `stays` is a `u32` rather than a byte so every field is four-aligned and the
//! reader can be a plain cast on any platform that cares.

use crate::frontier::{best_response, build_frontier, FrontierEntry};
use crate::Card;

pub const MAGIC: [u8; 4] = *b"JFL1";
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 4 + 4 + 4 + 8 + 32 + 8 + 16;
pub const ROW_BYTES: usize = 4 * 5;

/// The four-width Fantasyland EV table a pool was priced under.
pub type FlEvTable = [f64; 4];

/// One pool entry: a solved hand, ready to be filtered and scanned.
#[derive(Clone, Debug)]
pub struct PoolEntry {
    pub naturals: u64,
    pub jokers: u32,
    pub rows: Vec<FrontierEntry>,
}

#[derive(Debug)]
pub struct Pool {
    pub width: u32,
    pub fl_ev: FlEvTable,
    pub seed: u64,
    pub entries: Vec<PoolEntry>,
}

/// Bit index of a natural card in the 52-bit mask; jokers are counted, not
/// masked.  Pinned to the same rank-major order the rest of the crate uses.
#[inline]
pub fn natural_bit(card: &Card) -> Option<u64> {
    if card.rank == 0 {
        return None;
    }
    Some(1u64 << ((card.rank as u64 - 2) * 4 + card.suit as u64))
}

/// The mask and joker count of a set of cards -- hero's seen cards, or a pool
/// hand's own.
pub fn mask_of(cards: &[Card]) -> (u64, u32) {
    let mut naturals = 0u64;
    let mut jokers = 0u32;
    for card in cards {
        match natural_bit(card) {
            Some(bit) => naturals |= bit,
            None => jokers += 1,
        }
    }
    (naturals, jokers)
}

impl PoolEntry {
    /// Can a hero holding these cards face this hand?
    #[inline]
    pub fn compatible(&self, hero_naturals: u64, hero_jokers: u32) -> bool {
        self.naturals & hero_naturals == 0 && self.jokers + hero_jokers <= 2
    }
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub fn serialize(pool: &Pool) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_BYTES + pool.entries.len() * 256);
    out.extend_from_slice(&MAGIC);
    write_u32(&mut out, VERSION);
    write_u32(&mut out, pool.width);
    write_u64(&mut out, pool.entries.len() as u64);
    for value in pool.fl_ev {
        out.extend_from_slice(&value.to_le_bytes());
    }
    write_u64(&mut out, pool.seed);
    out.extend_from_slice(&[0u8; 16]);
    for entry in &pool.entries {
        write_u64(&mut out, entry.naturals);
        write_u32(&mut out, entry.jokers);
        write_u32(&mut out, entry.rows.len() as u32);
        for row in &entry.rows {
            write_u32(&mut out, row.top);
            write_u32(&mut out, row.mid);
            write_u32(&mut out, row.bot);
            write_u32(&mut out, row.royalty as u32);
            write_u32(&mut out, u32::from(row.stays));
        }
    }
    out
}

#[derive(Debug)]
pub enum LoadError {
    Truncated,
    BadMagic,
    BadVersion(u32),
    /// The pool prices its stay term under a different table.  This is not a
    /// stale pool, it is a wrong one.
    TableMismatch { stored: FlEvTable, wanted: FlEvTable },
    WidthMismatch { stored: u32, wanted: u32 },
    EmptyFrontier { entry: usize },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Truncated => write!(out, "pool is truncated"),
            LoadError::BadMagic => write!(out, "not a JFL1 pool"),
            LoadError::BadVersion(v) => write!(out, "pool version {v} is not {VERSION}"),
            LoadError::TableMismatch { stored, wanted } => write!(
                out,
                "pool was priced under fl_ev {stored:?}, caller wants {wanted:?}; \
                 a pool built under a superseded table is wrong, not stale"
            ),
            LoadError::WidthMismatch { stored, wanted } => {
                write!(out, "pool is width {stored}, caller wants {wanted}")
            }
            LoadError::EmptyFrontier { entry } => {
                write!(out, "entry {entry} has an empty frontier")
            }
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<u32, LoadError> {
        let end = self.cursor + 4;
        let slice = self.bytes.get(self.cursor..end).ok_or(LoadError::Truncated)?;
        self.cursor = end;
        Ok(u32::from_le_bytes(slice.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, LoadError> {
        let end = self.cursor + 8;
        let slice = self.bytes.get(self.cursor..end).ok_or(LoadError::Truncated)?;
        self.cursor = end;
        Ok(u64::from_le_bytes(slice.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, LoadError> {
        Ok(f64::from_bits(self.u64()?))
    }
}

/// Load a pool, refusing anything priced under a different table or built for
/// a different width.  Rows arrive unpriced and are priced here.
pub fn deserialize(bytes: &[u8], wanted_width: u32, wanted: FlEvTable) -> Result<Pool, LoadError> {
    if bytes.len() < HEADER_BYTES {
        return Err(LoadError::Truncated);
    }
    if bytes[0..4] != MAGIC {
        return Err(LoadError::BadMagic);
    }
    let mut reader = Reader { bytes, cursor: 4 };
    let version = reader.u32()?;
    if version != VERSION {
        return Err(LoadError::BadVersion(version));
    }
    let width = reader.u32()?;
    if width != wanted_width {
        return Err(LoadError::WidthMismatch { stored: width, wanted: wanted_width });
    }
    let count = reader.u64()? as usize;
    let mut stored = [0.0f64; 4];
    for slot in &mut stored {
        *slot = reader.f64()?;
    }
    // Bit equality, not approximate: two tables that differ anywhere price a
    // different game, and "close enough" is how a wrong pool loads quietly.
    if stored.iter().zip(&wanted).any(|(a, b)| a.to_bits() != b.to_bits()) {
        return Err(LoadError::TableMismatch { stored, wanted });
    }
    let seed = reader.u64()?;
    reader.cursor += 16;

    let fl_ev = stored[(width as usize).saturating_sub(14).min(3)];
    let mut entries = Vec::with_capacity(count);
    for index in 0..count {
        let naturals = reader.u64()?;
        let jokers = reader.u32()?;
        let row_count = reader.u32()? as usize;
        if row_count == 0 {
            return Err(LoadError::EmptyFrontier { entry: index });
        }
        let mut rows = Vec::with_capacity(row_count);
        for _ in 0..row_count {
            let top = reader.u32()?;
            let mid = reader.u32()?;
            let bot = reader.u32()?;
            let royalty = reader.u32()? as i32;
            let stays = reader.u32()? != 0;
            rows.push(FrontierEntry {
                top,
                mid,
                bot,
                royalty,
                stays,
                static_value: royalty as f64 + if stays { fl_ev } else { 0.0 },
            });
        }
        entries.push(PoolEntry { naturals, jokers, rows });
    }
    Ok(Pool { width, fl_ev: stored, seed, entries })
}

/// Deal the pool's hands.  The stream is a function of `(seed, index)` only, so
/// an entry's hand does not depend on how many threads built it or on where in
/// a batch it fell -- the property the regular pool learned to want the hard
/// way.
pub fn deal(seed: u64, index: u64, width: usize) -> Vec<Card> {
    let mut deck: Vec<Card> = Vec::with_capacity(54);
    for rank in 2..=14u8 {
        for suit in 0..4u8 {
            deck.push(Card { rank, suit });
        }
    }
    deck.push(Card { rank: 0, suit: 4 });
    deck.push(Card { rank: 0, suit: 4 });
    // SplitMix64 over (seed, index): independent per entry, no shared state.
    let mut state = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(index.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        ^ 0xD6E8_FEB8_6659_FD93;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as usize
    };
    for position in (1..deck.len()).rev() {
        deck.swap(position, next() % (position + 1));
    }
    deck.truncate(width);
    deck
}

/// Draw `want` compatible entries, deterministically.
///
/// A short draw is an **error**, not a shrug: a leaf that quietly averages over
/// three opponents when the caller asked for sixteen is a silent failure, and
/// silent failures in a label are the expensive kind.
#[derive(Debug)]
pub struct ShortDraw {
    pub found: usize,
    pub wanted: usize,
}

pub fn draw<'a>(
    pool: &'a Pool,
    hero_naturals: u64,
    hero_jokers: u32,
    want: usize,
    stream: u64,
) -> Result<Vec<&'a PoolEntry>, ShortDraw> {
    let total = pool.entries.len();
    if total == 0 {
        return Err(ShortDraw { found: 0, wanted: want });
    }
    // Walk from a stream-dependent offset with a stride coprime to the pool
    // size, so two draws over the same hero cards see different entries without
    // materialising a shuffle.
    let mut stride = (stream | 1) % total as u64;
    if stride == 0 {
        stride = 1;
    }
    while gcd(stride, total as u64) != 1 {
        stride = (stride + 2) % total as u64 | 1;
    }
    let mut cursor = (stream % total as u64) as usize;
    let mut out = Vec::with_capacity(want);
    for _ in 0..total {
        let entry = &pool.entries[cursor];
        if entry.compatible(hero_naturals, hero_jokers) {
            out.push(entry);
            if out.len() == want {
                return Ok(out);
            }
        }
        cursor = (cursor + stride as usize) % total;
    }
    Err(ShortDraw { found: out.len(), wanted: want })
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Mean best-response value of a hero board over drawn opponents.
pub fn opponent_mean(entries: &[&PoolEntry], hero_top: u32, hero_mid: u32, hero_bot: u32) -> f64 {
    if entries.is_empty() {
        return 0.0;
    }
    let total: f64 = entries
        .iter()
        .map(|entry| best_response(&entry.rows, hero_top, hero_mid, hero_bot))
        .sum();
    total / entries.len() as f64
}

/// Build one entry.  Separated so the builder can parallelise over entries
/// while each frontier stays sequential -- a shared-frontier parallel sweep
/// would change which of two equal rows survives, and with it the pool's bytes.
pub fn build_entry(seed: u64, index: u64, width: usize, fl_ev: f64) -> PoolEntry {
    let hand = deal(seed, index, width);
    let (naturals, jokers) = mask_of(&hand);
    let rows = build_frontier(&hand, fl_ev);
    assert!(
        !rows.is_empty(),
        "entry {index} produced an empty frontier; every 14-card hand has a \
         legal arrangement, so this is a solver defect, not a deal"
    );
    PoolEntry { naturals, jokers, rows }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: FlEvTable = [0.0, 10.7, 29.9, 63.5];

    fn small_pool(entries: usize, width: usize) -> Pool {
        let fl_ev = TABLE[width.saturating_sub(14).min(3)];
        Pool {
            width: width as u32,
            fl_ev: TABLE,
            seed: 0x2026_0811,
            entries: (0..entries as u64)
                .map(|index| build_entry(0x2026_0811, index, width, fl_ev))
                .collect(),
        }
    }

    /// The deal is a function of `(seed, index)` and nothing else -- not of
    /// batch position, not of thread count.  A pool whose entry 7 depends on
    /// how it was scheduled is not reproducible and its digest means nothing.
    #[test]
    fn deals_depend_only_on_seed_and_index() {
        for index in [0u64, 1, 7, 999] {
            let first = deal(0x2026_0811, index, 14);
            let again = deal(0x2026_0811, index, 14);
            assert_eq!(first.len(), 14);
            assert!(first.iter().zip(&again).all(|(a, b)| a.rank == b.rank && a.suit == b.suit));
        }
        // Different indices give different hands; a stream that repeats would
        // make a large pool a small one wearing a big number.
        let a = deal(0x2026_0811, 0, 14);
        let b = deal(0x2026_0811, 1, 14);
        assert!(a.iter().zip(&b).any(|(x, y)| x.rank != y.rank || x.suit != y.suit));
    }

    /// A hand deals 14 distinct cards from a 54-card deck: at most two jokers,
    /// and no natural twice.
    #[test]
    fn deals_are_legal_hands() {
        for index in 0..200u64 {
            let hand = deal(0x2026_0811, index, 14);
            let (naturals, jokers) = mask_of(&hand);
            assert!(jokers <= 2, "index {index} dealt {jokers} jokers");
            assert_eq!(
                naturals.count_ones() + jokers,
                14,
                "index {index} dealt a duplicate card"
            );
        }
    }

    /// A round trip preserves every stored field, and prices rows from the
    /// header rather than from whatever the writer happened to hold.
    #[test]
    fn round_trip_preserves_entries_and_prices_from_the_header() {
        let pool = small_pool(6, 14);
        let image = serialize(&pool);
        let back = deserialize(&image, 14, TABLE).expect("round trip");
        assert_eq!(back.entries.len(), pool.entries.len());
        assert_eq!(back.seed, pool.seed);
        for (before, after) in pool.entries.iter().zip(&back.entries) {
            assert_eq!(before.naturals, after.naturals);
            assert_eq!(before.jokers, after.jokers);
            assert_eq!(before.rows.len(), after.rows.len());
            for (row_before, row_after) in before.rows.iter().zip(&after.rows) {
                assert_eq!(
                    (row_before.top, row_before.mid, row_before.bot, row_before.royalty, row_before.stays),
                    (row_after.top, row_after.mid, row_after.bot, row_after.royalty, row_after.stays)
                );
                assert_eq!(
                    row_after.static_value.to_bits(),
                    (row_after.royalty as f64 + if row_after.stays { TABLE[0] } else { 0.0 }).to_bits()
                );
            }
        }
    }

    /// The loader refuses a pool priced under any other table, on any width.
    /// This is the whole reason the table is in the header.
    #[test]
    fn a_pool_priced_under_another_table_is_refused() {
        let image = serialize(&small_pool(3, 14));
        for wrong in [
            [0.1, 10.7, 29.9, 63.5],
            [0.0, 10.8, 29.9, 63.5],
            [0.0, 10.7, 29.9, 63.4],
            [-4.5, 10.7, 29.9, 63.5],
        ] {
            match deserialize(&image, 14, wrong) {
                Err(LoadError::TableMismatch { .. }) => {}
                other => panic!("loader accepted a pool priced under {wrong:?}: {other:?}"),
            }
        }
        // ... and a pool built for another width.
        match deserialize(&image, 17, TABLE) {
            Err(LoadError::WidthMismatch { .. }) => {}
            other => panic!("loader accepted a width-14 pool as width 17: {other:?}"),
        }
    }

    #[test]
    fn truncation_and_bad_magic_are_errors() {
        let image = serialize(&small_pool(3, 14));
        assert!(matches!(deserialize(&image[..HEADER_BYTES + 4], 14, TABLE), Err(LoadError::Truncated)));
        let mut corrupt = image.clone();
        corrupt[1] = b'X';
        assert!(matches!(deserialize(&corrupt, 14, TABLE), Err(LoadError::BadMagic)));
    }

    /// **The pool's claim.** A pooled entry must answer exactly what solving
    /// its hand on the spot would answer -- the pool is a cache, not a model.
    ///
    /// Tested over the entries directly rather than over a draw: the draw's
    /// job is picking *which* hands, and mixing that in here would make a
    /// failure ambiguous between the cache and the filter.  The filter has its
    /// own tests below.
    #[test]
    fn pooled_entries_answer_as_solving_their_hand_would() {
        let pool = small_pool(20, 14);
        let hero_hand = deal(0xFEED_0001, 0, 13);
        let core = crate::to_core_cards(&hero_hand);
        let eval = ofc_core::evaluate_board_with_joker_constraint(
            &core[0..3], &core[3..8], &core[8..13],
        );
        let hero = (
            ofc_core::evaluate_hand_value(&eval.top, 3),
            ofc_core::evaluate_hand_value(&eval.mid, 5),
            ofc_core::evaluate_hand_value(&eval.bot, 5),
        );
        // Through the wire too: what a consumer actually holds is the reloaded
        // pool, not the one the builder had in memory.
        let reloaded = deserialize(&serialize(&pool), 14, TABLE).expect("round trip");
        for (index, entry) in reloaded.entries.iter().enumerate() {
            let fresh = build_frontier(&deal(pool.seed, index as u64, 14), TABLE[0]);
            assert_eq!(
                best_response(&entry.rows, hero.0, hero.1, hero.2).to_bits(),
                best_response(&fresh, hero.0, hero.1, hero.2).to_bits(),
                "pooled entry {index} disagrees with solving its hand on the spot"
            );
        }
    }

    /// How often a pool entry survives the disjointness filter -- the number
    /// that sets how big a pool has to be.
    ///
    /// It is not a free parameter: hero's seen cards grow street by street, and
    /// the acceptance rate falls with them.  This measures it rather than
    /// asserting a target, and only pins the ordering (later streets are
    /// stricter), because the pool sizing decision belongs to whoever is
    /// building one.  Run with `--nocapture`.
    #[test]
    fn acceptance_rate_by_hero_card_count() {
        let pool = small_pool(30, 14);
        println!("pool acceptance by hero cards seen:");
        let mut previous = 1.01f64;
        for hero_cards in [5usize, 8, 11, 13, 17] {
            let (mut accepted, mut trials) = (0usize, 0usize);
            for hero_index in 0..400u64 {
                let hero_hand = deal(0xACCE_0001, hero_index, hero_cards);
                let (hero_naturals, hero_jokers) = mask_of(&hero_hand);
                for entry in &pool.entries {
                    trials += 1;
                    if entry.compatible(hero_naturals, hero_jokers) {
                        accepted += 1;
                    }
                }
            }
            let rate = accepted as f64 / trials as f64;
            println!(
                "  hero {hero_cards:2} cards: {:.2}%  ({} entries per 10k pool)",
                rate * 100.0,
                (rate * 10_000.0).round() as u64
            );
            assert!(
                rate <= previous,
                "acceptance rose from {previous:.4} to {rate:.4} as hero saw more cards"
            );
            previous = rate;
        }
    }

    /// Every drawn entry is disjoint from hero -- the conditioning the whole
    /// exactness argument rests on.  A pool that hands back a hand containing
    /// one of hero's cards is not sampling the right distribution at all.
    #[test]
    fn drawn_entries_are_disjoint_from_hero() {
        let pool = small_pool(60, 14);
        for hero_index in 0..12u64 {
            // Five cards, so a 60-entry pool actually yields draws; the
            // property under test is disjointness, not pool sizing.
            let hero_hand = deal(0xFEED_0002, hero_index, 5);
            let (hero_naturals, hero_jokers) = mask_of(&hero_hand);
            let Ok(entries) = draw(&pool, hero_naturals, hero_jokers, 8, hero_index) else {
                continue;
            };
            for entry in entries {
                assert_eq!(
                    entry.naturals & hero_naturals,
                    0,
                    "drawn entry shares a natural card with hero"
                );
                assert!(
                    entry.jokers + hero_jokers <= 2,
                    "drawn entry claims {} jokers while hero holds {hero_jokers}",
                    entry.jokers
                );
            }
        }
    }

    /// A short draw is an error carrying both numbers, not a quiet average over
    /// however many turned up.
    #[test]
    fn a_short_draw_is_an_error() {
        let pool = small_pool(4, 14);
        let hero_hand = deal(0xFEED_0003, 0, 13);
        let (hero_naturals, hero_jokers) = mask_of(&hero_hand);
        match draw(&pool, hero_naturals, hero_jokers, 99, 1) {
            Err(ShortDraw { found, wanted }) => {
                assert_eq!(wanted, 99);
                assert!(found <= 4, "found {found} in a 4-entry pool");
            }
            Ok(entries) => panic!("a 4-entry pool returned {} of 99", entries.len()),
        }
    }
}
