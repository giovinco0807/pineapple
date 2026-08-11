//! A pre-solved pool of Fantasyland hands, drawn on once per leaf instead of built.
//!
//! Two thirds of a T0 position is one thing: at each of its 1,024 leaves the
//! teacher draws `samples` Fantasyland hands from the deck hero left, solves
//! each, and builds each one's non-dominated frontier. Measured on this
//! machine, a solve is 5.18 ms and a frontier is 38.3 ms, so a position spends
//! 16,384 of each -- about 700 core-seconds -- before it has scored a single
//! one of its 232 openings.
//!
//! None of that work depends on the opening. All 232 use the same five cards,
//! so the deck is the same, so the opponent's hands are the same. Narrowing the
//! fan therefore cannot touch it, which is why cutting 232 openings to 20 was
//! measured at only 1.42x.
//!
//! # Why a pool is exact rather than an approximation
//!
//! A uniform 14-card hand from the whole deck, *conditioned on sharing no card
//! with hero's seventeen*, is distributed exactly as a uniform 14-card hand
//! from the thirty-five hero left. That is the definition of conditioning, not
//! a modelling assumption. So a pool drawn once from the full deck, filtered at
//! each leaf to the hands that fit, samples the right distribution -- and the
//! filter is a bitwise AND against a 52-bit mask.
//!
//! A hand fits a given seventeen with probability C(35,14)/C(52,14) = 0.13%,
//! one in 762, so a pool of 200,000 leaves about 262 candidates at each leaf.
//!
//! # What it costs
//!
//! Reuse. Independent draws give every leaf its own opponents; a pool gives
//! overlapping ones, so the same hand is scored at many leaves. The marginal
//! distribution is untouched but the estimate carries less independent
//! information than its sample count suggests. That is a variance question and
//! shows up in seed-to-seed self-agreement, which is where it should be
//! measured rather than argued about.
//!
//! Selection within a leaf is by random probe, not by taking the first hands
//! that fit: the pool's order is arbitrary but fixed, and reading it from the
//! front would let its early entries serve far more leaves than its late ones.

use crate::frontier::{build_frontier, FrontierEntry};
use crate::objective::ObjectiveConfig;
use crate::rng::SplitMix64;
use crate::solver::FlSolver;

/// One pre-solved hand: which cards it uses, and its non-dominated frontier.
pub struct LibraryEntry {
    pub mask: u64,
    pub deal: [u8; 14],
    pub frontier: Vec<FrontierEntry>,
}

pub struct FlLibrary {
    pub entries: Vec<LibraryEntry>,
    pub fl_ev: f64,
    pub seed: u64,
}

impl FlLibrary {
    /// Draw and solve `count` hands from the full deck.
    ///
    /// The pool is deck-wide on purpose. Restricting it to a particular root's
    /// unseen set would make it a different pool for every position and give up
    /// the amortisation that is the whole point.
    pub fn build(
        count: usize,
        seed: u64,
        objective: &ObjectiveConfig,
        solver: &mut FlSolver,
    ) -> Result<Self, String> {
        Self::build_range(0, count, seed, objective, solver)
    }

    /// Entries `first..first + count` of the pool `seed` defines.
    ///
    /// The stream index is the entry's position in the pool, not its position
    /// in this call, so splitting the build across threads produces the same
    /// pool as building it in one -- a pool that changed with the thread count
    /// would make a label depend on the machine that produced it.
    pub fn build_range(
        first: usize,
        count: usize,
        seed: u64,
        objective: &ObjectiveConfig,
        solver: &mut FlSolver,
    ) -> Result<Self, String> {
        let mut entries = Vec::with_capacity(count);
        for index in first as u64..(first + count) as u64 {
            let mut rng = SplitMix64::for_stream(seed, index);
            let mut deck: Vec<u8> = (0..52_u8).collect();
            rng.partial_shuffle(&mut deck, 14);
            let mut deal: [u8; 14] = deck[..14].try_into().expect("fourteen cards");
            deal.sort_unstable();
            // Solving is not needed for scoring -- `best_response` reads the
            // frontier -- but it is what proves the hand has a legal placement,
            // and a hand that does not would poison every leaf that drew it.
            solver
                .solve(&deal)
                .ok_or_else(|| "a 14-card Fantasyland deal must have a legal placement".to_owned())?;
            entries.push(LibraryEntry {
                mask: crate::cards::mask_of(&deal),
                deal,
                frontier: build_frontier(&deal, objective.fl_ev_stay),
            });
        }
        Ok(Self {
            entries,
            fl_ev: objective.fl_ev_stay,
            seed,
        })
    }

    /// `want` hands that share no card with `seen_mask`, chosen by random probe.
    ///
    /// Returns fewer than `want` only if the pool cannot supply them, which the
    /// caller must treat as a sizing error rather than absorb: a leaf that
    /// averages over three opponents when the plan said sixteen is the same
    /// silent failure the collision-rejection shape had.
    pub fn draw(&self, seen_mask: u64, want: usize, stream: u64) -> Vec<&LibraryEntry> {
        let mut rng = SplitMix64::for_stream(self.seed ^ 0x9E37_79B9_7F4A_7C15, stream);
        let mut out = Vec::with_capacity(want);
        let mut taken = vec![false; self.entries.len()];
        // One in 762 hands fits, so the expected probe count is 762 per hand
        // wanted; the cap is generous enough that only an undersized pool
        // reaches it.
        let cap = want.saturating_mul(4_000).max(20_000);
        for _ in 0..cap {
            if out.len() == want {
                break;
            }
            let index = (rng.next_u64() % self.entries.len() as u64) as usize;
            if taken[index] {
                continue;
            }
            let entry = &self.entries[index];
            if entry.mask & seen_mask == 0 {
                taken[index] = true;
                out.push(entry);
            }
        }
        out
    }
}

// -- on disk -----------------------------------------------------------------
//
// The pool is built once and read many times, on a machine that did not build
// it, so it travels as bytes rather than as a seed to replay: rebuilding
// 200,000 frontiers costs about eighteen minutes on a worker's eight vCPU, and
// a worker that rebuilds is a worker not labelling. The format is flat and
// little-endian, and the reader checks the magic, the version and the
// Fantasyland constant -- a pool built under a different constant prices its
// stay term differently and is a different pool.

const MAGIC: &[u8; 4] = b"FLL1";
const VERSION: u32 = 1;

impl FlLibrary {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.fl_ev.to_le_bytes());
        out.extend_from_slice(&self.seed.to_le_bytes());
        for entry in self.entries.iter() {
            out.extend_from_slice(&entry.mask.to_le_bytes());
            out.extend_from_slice(&entry.deal);
            out.extend_from_slice(&(entry.frontier.len() as u32).to_le_bytes());
            for row in entry.frontier.iter() {
                out.extend_from_slice(&row.top_key.to_le_bytes());
                out.extend_from_slice(&row.middle_key.to_le_bytes());
                out.extend_from_slice(&row.bottom_key.to_le_bytes());
                out.extend_from_slice(&row.static_value.to_le_bytes());
                out.extend_from_slice(&row.total_royalty.to_le_bytes());
                out.push(row.stays as u8);
            }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8], expected_fl_ev: f64) -> Result<Self, String> {
        if bytes.len() < 28 || &bytes[..4] != MAGIC {
            return Err("library does not start with the expected magic".to_owned());
        }
        let mut at = 4_usize;
        let mut u32_at = |at: &mut usize| -> Result<u32, String> {
            if *at + 4 > bytes.len() {
                return Err("library ends inside a u32".to_owned());
            }
            let value = u32::from_le_bytes(bytes[*at..*at + 4].try_into().unwrap());
            *at += 4;
            Ok(value)
        };
        let version = u32_at(&mut at)?;
        if version != VERSION {
            return Err(format!("library version {version} is not supported"));
        }
        let count = {
            if at + 8 > bytes.len() {
                return Err("library ends inside its count".to_owned());
            }
            let value = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
            at += 8;
            value as usize
        };
        let fl_ev = {
            let value = f64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
            at += 8;
            value
        };
        if (fl_ev - expected_fl_ev).abs() > 1e-9 {
            return Err(format!(
                "library was built at fl_ev {fl_ev}, the run asks for {expected_fl_ev};                  the stay term is priced into every frontier and the two are not                  the same pool"
            ));
        }
        let seed = {
            let value = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
            at += 8;
            value
        };

        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            if at + 8 + 14 + 4 > bytes.len() {
                return Err("library ends inside an entry header".to_owned());
            }
            let mask = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
            at += 8;
            let deal: [u8; 14] = bytes[at..at + 14].try_into().unwrap();
            at += 14;
            let rows = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            at += 4;
            let mut frontier = Vec::with_capacity(rows);
            for _ in 0..rows {
                if at + 4 + 4 + 4 + 8 + 4 + 1 > bytes.len() {
                    return Err("library ends inside a frontier row".to_owned());
                }
                let top_key = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                at += 4;
                let middle_key = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                at += 4;
                let bottom_key = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                at += 4;
                let static_value = f64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
                at += 8;
                let total_royalty = i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                at += 4;
                let stays = bytes[at] != 0;
                at += 1;
                frontier.push(FrontierEntry {
                    top_key,
                    middle_key,
                    bottom_key,
                    static_value,
                    total_royalty,
                    stays,
                });
            }
            entries.push(LibraryEntry { mask, deal, frontier });
        }
        if at != bytes.len() {
            return Err(format!("library has {} trailing bytes", bytes.len() - at));
        }
        Ok(Self { entries, fl_ev, seed })
    }
}
