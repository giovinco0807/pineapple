//! Rust-owned paired deck generation for deterministic HU seat-swap replay.
//!
//! The generator deliberately reproduces the historical Python oracle:
//!
//! `random.Random(splitmix64(pair_seed)).shuffle(list(ALL_CARDS))`
//!
//! where `pair_seed = seed_base + global_pair_index * seed_stride`.  The first
//! lane of every pair is the oracle deck (AB); the second lane (BA) swaps only
//! the ten public deal windows between seats.  Positions 34..52, including the
//! entire unrealized tail, are identical in both legs and never cross the
//! Python boundary.

use crate::{Card, HuRlError, HuRlResult, ALL_CARDS, MAX_BATCH_LANES};

pub const MAX_PAIRED_SEED: u64 = i64::MAX as u64;
pub const MAX_PAIRED_HANDS_PER_BATCH: usize = MAX_BATCH_LANES / 2;

const MT_STATE_WORDS: usize = 624;
const MT_PERIOD_OFFSET: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;

const DEAL_WINDOW_PAIRS: [((usize, usize), (usize, usize)); 5] = [
    ((0, 5), (5, 10)),
    ((10, 13), (13, 16)),
    ((16, 19), (19, 22)),
    ((22, 25), (25, 28)),
    ((28, 31), (31, 34)),
];

/// Validated immutable seed geometry for one paired seat-swap native batch.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PairedSeedRange {
    seed_base: u64,
    global_pair_start: u64,
    pair_count: usize,
    seed_stride: u64,
}

impl PairedSeedRange {
    pub fn new(
        seed_base: u64,
        global_pair_start: u64,
        pair_count: usize,
        seed_stride: u64,
    ) -> HuRlResult<Self> {
        if seed_base > MAX_PAIRED_SEED {
            return Err(HuRlError::new(
                "paired seed base is outside the unsigned 63-bit domain",
            ));
        }
        if global_pair_start > MAX_PAIRED_SEED {
            return Err(HuRlError::new(
                "global pair start is outside the unsigned 63-bit domain",
            ));
        }
        if !(1..=MAX_PAIRED_HANDS_PER_BATCH).contains(&pair_count) {
            return Err(HuRlError::new(
                "paired seed range has an invalid pair count",
            ));
        }
        if seed_stride == 0 || seed_stride > MAX_PAIRED_SEED {
            return Err(HuRlError::new(
                "paired seed stride is outside the unsigned 63-bit domain",
            ));
        }

        let final_pair_index = global_pair_start
            .checked_add((pair_count - 1) as u64)
            .ok_or_else(|| HuRlError::new("paired seed range overflows global pair index"))?;
        if final_pair_index > MAX_PAIRED_SEED {
            return Err(HuRlError::new(
                "paired seed range exceeds the unsigned 63-bit pair domain",
            ));
        }
        let final_offset = final_pair_index
            .checked_mul(seed_stride)
            .ok_or_else(|| HuRlError::new("paired seed range multiplication overflow"))?;
        let final_seed = seed_base
            .checked_add(final_offset)
            .ok_or_else(|| HuRlError::new("paired seed range addition overflow"))?;
        if final_seed > MAX_PAIRED_SEED {
            return Err(HuRlError::new(
                "paired seed range exceeds the unsigned 63-bit seed domain",
            ));
        }

        Ok(Self {
            seed_base,
            global_pair_start,
            pair_count,
            seed_stride,
        })
    }

    pub const fn seed_base(self) -> u64 {
        self.seed_base
    }

    pub const fn global_pair_start(self) -> u64 {
        self.global_pair_start
    }

    pub const fn pair_count(self) -> usize {
        self.pair_count
    }

    pub const fn seed_stride(self) -> u64 {
        self.seed_stride
    }

    pub const fn lane_count(self) -> usize {
        self.pair_count * 2
    }

    fn pair_seed(self, local_pair: usize) -> u64 {
        // Construction proves every operation and the final value are valid.
        self.seed_base + (self.global_pair_start + local_pair as u64) * self.seed_stride
    }
}

/// Generate `[pair0 AB, pair0 BA, pair1 AB, pair1 BA, ...]` entirely in Rust.
pub fn paired_seeded_decks(range: PairedSeedRange) -> Vec<Vec<Card>> {
    let mut decks = Vec::with_capacity(range.lane_count());
    for local_pair in 0..range.pair_count() {
        let ab = cpython_oracle_deck(range.pair_seed(local_pair));
        let ba = seat_swapped_deck(&ab);
        decks.push(ab.to_vec());
        decks.push(ba.to_vec());
    }
    decks
}

/// The explicit oracle deck for one already-derived pair seed.
///
/// This is public only so native correctness tests and explicit-deck callers
/// can compare the seeded path without exposing any deck through PyO3.
pub fn cpython_oracle_deck(pair_seed: u64) -> [Card; 52] {
    let lane_seed = splitmix64(pair_seed);
    let mut rng = CpythonMt19937::from_nonnegative_int(lane_seed);
    let mut deck = ALL_CARDS;
    // CPython random.shuffle: for i in reversed(range(1, len(x))):
    //     j = self._randbelow(i + 1); x[i], x[j] = x[j], x[i]
    for index in (1..deck.len()).rev() {
        let swap_index = rng.rand_below(index + 1);
        deck.swap(index, swap_index);
    }
    deck
}

fn seat_swapped_deck(deck: &[Card; 52]) -> [Card; 52] {
    let mut swapped = *deck;
    for ((first_start, first_end), (second_start, second_end)) in DEAL_WINDOW_PAIRS {
        debug_assert_eq!(first_end - first_start, second_end - second_start);
        let width = first_end - first_start;
        for offset in 0..width {
            swapped[first_start + offset] = deck[second_start + offset];
            swapped[second_start + offset] = deck[first_start + offset];
        }
    }
    swapped
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// CPython `_random.Random` MT19937 state for a non-negative integer seed.
struct CpythonMt19937 {
    state: [u32; MT_STATE_WORDS],
    index: usize,
}

impl CpythonMt19937 {
    fn from_nonnegative_int(seed: u64) -> Self {
        // CPython exports a positive PyLong as little-endian 32-bit limbs.
        // Zero still has one limb; a value <= u32::MAX has exactly one limb.
        let low = seed as u32;
        let high = (seed >> 32) as u32;
        if high == 0 {
            Self::init_by_array(&[low])
        } else {
            Self::init_by_array(&[low, high])
        }
    }

    fn init_by_array(key: &[u32]) -> Self {
        let mut rng = Self {
            state: [0; MT_STATE_WORDS],
            index: MT_STATE_WORDS,
        };
        rng.init_genrand(19_650_218);

        let mut i = 1_usize;
        let mut j = 0_usize;
        for _ in 0..MT_STATE_WORDS.max(key.len()) {
            let previous = rng.state[i - 1];
            rng.state[i] = (rng.state[i] ^ (previous ^ (previous >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= MT_STATE_WORDS {
                rng.state[0] = rng.state[MT_STATE_WORDS - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..MT_STATE_WORDS - 1 {
            let previous = rng.state[i - 1];
            rng.state[i] = (rng.state[i]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= MT_STATE_WORDS {
                rng.state[0] = rng.state[MT_STATE_WORDS - 1];
                i = 1;
            }
        }
        rng.state[0] = 0x8000_0000;
        rng.index = MT_STATE_WORDS;
        rng
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for index in 1..MT_STATE_WORDS {
            let previous = self.state[index - 1];
            self.state[index] = 1_812_433_253_u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(index as u32);
        }
        self.index = MT_STATE_WORDS;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= MT_STATE_WORDS {
            let mag = [0_u32, MT_MATRIX_A];
            for kk in 0..MT_STATE_WORDS - MT_PERIOD_OFFSET {
                let y = (self.state[kk] & MT_UPPER_MASK) | (self.state[kk + 1] & MT_LOWER_MASK);
                self.state[kk] =
                    self.state[kk + MT_PERIOD_OFFSET] ^ (y >> 1) ^ mag[(y & 1) as usize];
            }
            for kk in MT_STATE_WORDS - MT_PERIOD_OFFSET..MT_STATE_WORDS - 1 {
                let y = (self.state[kk] & MT_UPPER_MASK) | (self.state[kk + 1] & MT_LOWER_MASK);
                self.state[kk] = self.state[kk + MT_PERIOD_OFFSET - MT_STATE_WORDS]
                    ^ (y >> 1)
                    ^ mag[(y & 1) as usize];
            }
            let y =
                (self.state[MT_STATE_WORDS - 1] & MT_UPPER_MASK) | (self.state[0] & MT_LOWER_MASK);
            self.state[MT_STATE_WORDS - 1] =
                self.state[MT_PERIOD_OFFSET - 1] ^ (y >> 1) ^ mag[(y & 1) as usize];
            self.index = 0;
        }

        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    fn getrandbits_at_most_32(&mut self, bit_count: u32) -> u32 {
        debug_assert!((1..=32).contains(&bit_count));
        self.next_u32() >> (32 - bit_count)
    }

    fn rand_below(&mut self, exclusive_upper: usize) -> usize {
        debug_assert!(exclusive_upper > 0 && exclusive_upper <= 52);
        // Python int.bit_length(), intentionally including the extra rejection
        // bit for powers of two.
        let bit_count = usize::BITS - exclusive_upper.leading_zeros();
        loop {
            let value = self.getrandbits_at_most_32(bit_count) as usize;
            if value < exclusive_upper {
                return value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn ten_thousand_decks_match_pinned_cpython_oracle_digest() {
        // Generated independently with CPython 3.13:
        // for seed in range(10_000):
        //   deck=list(range(52)); Random(splitmix64(seed)).shuffle(deck)
        //   digest.update(bytes(deck))
        let mut digest = Sha256::new();
        for seed in 0..10_000_u64 {
            for card in cpython_oracle_deck(seed) {
                digest.update([card.index() as u8]);
            }
        }
        assert_eq!(
            format!("{:x}", digest.finalize()),
            "bf60d6c3cbca7ba56262154b487b7c59b665fa0d991e626980749cf4e933637c"
        );
    }

    #[test]
    fn seat_swap_changes_only_ten_deal_windows() {
        let range = PairedSeedRange::new(17, 3, 1, 11).unwrap();
        let decks = paired_seeded_decks(range);
        assert_eq!(decks.len(), 2);
        let ab = &decks[0];
        let ba = &decks[1];
        assert_eq!(&ba[0..5], &ab[5..10]);
        assert_eq!(&ba[5..10], &ab[0..5]);
        assert_eq!(&ba[10..13], &ab[13..16]);
        assert_eq!(&ba[13..16], &ab[10..13]);
        assert_eq!(&ba[16..19], &ab[19..22]);
        assert_eq!(&ba[19..22], &ab[16..19]);
        assert_eq!(&ba[22..25], &ab[25..28]);
        assert_eq!(&ba[25..28], &ab[22..25]);
        assert_eq!(&ba[28..31], &ab[31..34]);
        assert_eq!(&ba[31..34], &ab[28..31]);
        assert_eq!(&ba[34..52], &ab[34..52]);
        let mut sorted = ba.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, ALL_CARDS);
    }

    #[test]
    fn edge_seed_ranges_validate_without_wraparound() {
        for seed in [0, u32::MAX as u64, u32::MAX as u64 + 1, MAX_PAIRED_SEED] {
            let range = PairedSeedRange::new(seed, 0, 1, 1).unwrap();
            assert_eq!(paired_seeded_decks(range).len(), 2);
        }
        assert!(PairedSeedRange::new(MAX_PAIRED_SEED + 1, 0, 1, 1).is_err());
        assert!(PairedSeedRange::new(0, 0, 0, 1).is_err());
        assert!(PairedSeedRange::new(0, 0, MAX_PAIRED_HANDS_PER_BATCH + 1, 1).is_err());
        assert!(PairedSeedRange::new(0, 0, 1, 0).is_err());
        assert!(PairedSeedRange::new(0, MAX_PAIRED_SEED, 2, 1).is_err());
        assert!(PairedSeedRange::new(MAX_PAIRED_SEED, 1, 1, 1).is_err());
    }
}
