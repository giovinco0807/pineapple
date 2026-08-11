"""Training-only symmetry augmentation for HU Turn1 teacher samples."""

from __future__ import annotations

import itertools
import random
from typing import Any, Sequence


RANKS = frozenset("23456789TJQKA")
SUITS = ("c", "d", "h", "s")


def _is_card(value: str) -> bool:
    return len(value) == 2 and value[0] in RANKS and value[1] in SUITS


def _replace_suits(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value[0] + mapping[value[1]] if _is_card(value) else value
    if isinstance(value, list):
        return [_replace_suits(item, mapping) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_suits(item, mapping) for item in value)
    if isinstance(value, dict):
        return {key: _replace_suits(item, mapping) for key, item in value.items()}
    return value


def suit_mappings(*, count: int, seed: int) -> list[dict[str, str]]:
    """Return deterministic non-identity suit permutations."""

    if count < 0:
        raise ValueError("suit augmentation count must be non-negative")
    permutations = [
        permutation
        for permutation in itertools.permutations(SUITS)
        if permutation != SUITS
    ]
    if count > len(permutations):
        raise ValueError(f"suit augmentation count must be <= {len(permutations)}")
    random.Random(seed).shuffle(permutations)
    return [dict(zip(SUITS, permutation, strict=True)) for permutation in permutations[:count]]


def augment_samples_by_suit(
    samples: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Append suit-permuted copies while preserving the original sample first."""

    if count == 0:
        return list(samples)
    mappings = suit_mappings(count=count, seed=seed)
    output: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        output.append(sample)
        for permutation_index, mapping in enumerate(mappings, start=1):
            augmented = _replace_suits(sample, mapping)
            augmented["training_augmentation"] = {
                "kind": "global_suit_permutation",
                "source_sample_index": int(sample_index),
                "permutation_index": int(permutation_index),
                "mapping": mapping,
            }
            output.append(augmented)
    return output
