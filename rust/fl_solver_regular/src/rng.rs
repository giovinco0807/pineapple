//! Counter-based deterministic RNG.
//!
//! Every stream is addressed by `(seed_base, stream_index)` and seeded by
//! hashing the pair, so sample `i` is reproducible without replaying samples
//! `0..i`. That is what makes common random numbers work across candidates:
//! the same `(seed_base, i)` yields the same opponent deal for every hero
//! candidate being compared.

#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

#[inline(always)]
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut out = z;
    out = (out ^ (out >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    out = (out ^ (out >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    out ^ (out >> 31)
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Deterministic stream for `(seed_base, stream_index)`.
    pub fn for_stream(seed_base: u64, stream_index: u64) -> Self {
        let mixed = mix(seed_base ^ mix(stream_index.wrapping_add(0x1234_5678_9ABC_DEF0)));
        Self { state: mixed }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Unbiased index in `0..bound` via Lemire rejection.
    #[inline(always)]
    pub fn below(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        if bound <= 1 {
            return 0;
        }
        let bound_u64 = bound as u64;
        let threshold = bound_u64.wrapping_neg() % bound_u64;
        loop {
            let draw = self.next_u64();
            let product = (draw as u128) * (bound_u64 as u128);
            if (product as u64) >= threshold {
                return (product >> 64) as usize;
            }
        }
    }

    #[inline(always)]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }

    /// Partial Fisher-Yates: draw `count` distinct items into the prefix.
    pub fn partial_shuffle<T: Copy>(&mut self, items: &mut [T], count: usize) {
        let take = count.min(items.len());
        for slot in 0..take {
            let pick = slot + self.below(items.len() - slot);
            items.swap(slot, pick);
        }
    }

    pub fn shuffle<T: Copy>(&mut self, items: &mut [T]) {
        let len = items.len();
        if len > 1 {
            self.partial_shuffle(items, len - 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_addressable_and_reproducible() {
        let first = SplitMix64::for_stream(997_000_000, 42).next_u64();
        let again = SplitMix64::for_stream(997_000_000, 42).next_u64();
        assert_eq!(first, again);
        assert_ne!(first, SplitMix64::for_stream(997_000_000, 43).next_u64());
        assert_ne!(first, SplitMix64::for_stream(997_000_001, 42).next_u64());
    }

    #[test]
    fn partial_shuffle_draws_distinct_items() {
        let mut rng = SplitMix64::for_stream(1, 1);
        let mut deck: Vec<u8> = (0..52).collect();
        rng.partial_shuffle(&mut deck, 14);
        let mut drawn = deck[..14].to_vec();
        drawn.sort_unstable();
        drawn.dedup();
        assert_eq!(drawn.len(), 14);
    }

    #[test]
    fn below_stays_in_range_and_covers_it() {
        let mut rng = SplitMix64::for_stream(7, 7);
        let mut seen = [false; 5];
        for _ in 0..2_000 {
            let value = rng.below(5);
            assert!(value < 5);
            seen[value] = true;
        }
        assert!(seen.iter().all(|hit| *hit));
    }
}
