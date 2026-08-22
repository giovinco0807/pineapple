"""Labelling a street whose tree still contains an opponent decision.

``exact.evaluate_t3`` stops at the BTN because that seat closes T3: both
opponents have already finished their own T3, so everything below the hero's
move is one T4 round and the tree has no interior opponent node.  Every street
above it does.  At (T3, BB) there is exactly one -- the BTN's T3 -- which makes
it the cheapest place in the whole cascade to ask the question the cascade's
cost hinges on:

    how good does the policy standing at an interior node have to be before the
    label above it stops moving?

That question has a real answer and a real bill attached.  Resolving the
interior node with the exact solver costs about a second per call and there are
``|hero actions| x |interior deals|`` of them per root, which is minutes per
root; resolving it with the distilled T3-BTN model costs microseconds.  If the
two produce the same label, the cascade can climb.  If they do not, T3-BTN has
to be made better before T2 can be built on it at all.

So the interior policy is a parameter here, not a constant.  ``INTERIOR_POLICY``
adapters wrap the exact solver, the distilled model, the Monte-Carlo referee,
the frozen heads-up models, and a uniform-random control -- the control first,
because this project has twice built a measurement that could not tell a
constant answer from a good one.

Draws are stratified rather than i.i.d., which is what makes the exact arm
affordable at all.  The interior decision depends on the BTN's own cards, not
on the T4 cards nobody has seen yet, so one stratum fixes (BTN discards, SB
discards, BTN's T3 deal) and varies only the T4 round underneath it.  That
turns ``J x L`` draws into ``J`` interior solves per hero action instead of
``J x L``, and every policy arm reads the same strata, so the comparison runs
under common random numbers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..action_key import action_key
from ..action_space import Action, generate_actions
from ..state import ROWS, Board
from .exact import T3Candidate, _completing_terminals, evaluate_t3, round_hero_value
from .mc import _key, evaluate_actions_mc
from .scoring import DEFAULT_FL_EV_PER_PAIR
from .seating import SEAT_BTN, act_order_of
from .world import ThreeMaxObservation

# The middle seat's own act index at T4: the SB opens the round, the hero
# answers, and the BTN closes it.
MIDDLE_HERO_INDEX = 1

# What one stratum consumes from the hero's unseen set before the T4 round is
# dealt: the BTN's two private discards, the SB's three, and the BTN's T3 deal.
STRATUM_CARDS = 2 + 3 + 3
T4_ROUND_CARDS = 9


@dataclass(frozen=True)
class MiddleDraw:
    """One stratum: an interior information set plus its T4 sub-draws.

    ``btn_discards`` and ``sb_discards`` are cards the hero cannot see and that
    nobody deals again.  They are sampled rather than ignored because the BTN's
    own discards are part of ITS information set -- they are dead cards it
    knows about and the hero does not -- and dropping them would hand the
    interior policy a deck that does not exist.
    """

    btn_discards: tuple[str, ...]
    sb_discards: tuple[str, ...]
    btn_dealt: tuple[str, ...]
    t4_draws: tuple[tuple[str, ...], ...]


def sample_middle_draws(
    unseen: Sequence[str], *, strata: int, per_stratum: int, seed: int
) -> tuple[MiddleDraw, ...]:
    """Stratified continuations for a (T3, BB) root.

    The hero sees 34 cards, so 18 are unseen: the BTN's 2 discards, the SB's 3,
    the 12 still to be dealt, and the 1 the hand never reaches.  A draw assigns
    every one of them, which is why the T4 round is dealt from what a stratum
    leaves rather than from the unseen set directly.
    """
    if strata <= 0 or per_stratum <= 0:
        raise ValueError("strata and per_stratum must be positive")
    pool = list(unseen)
    if len(pool) < STRATUM_CARDS + T4_ROUND_CARDS:
        raise ValueError(
            f"a (T3, BB) continuation needs {STRATUM_CARDS + T4_ROUND_CARDS} "
            f"unseen cards, got {len(pool)}"
        )

    rng = random.Random(seed)
    draws: list[MiddleDraw] = []
    for _ in range(strata):
        head = rng.sample(pool, STRATUM_CARDS)
        rest = [card for card in pool if card not in head]
        draws.append(
            MiddleDraw(
                btn_discards=tuple(head[0:2]),
                sb_discards=tuple(head[2:5]),
                btn_dealt=tuple(head[5:8]),
                t4_draws=tuple(
                    tuple(rng.sample(rest, T4_ROUND_CARDS))
                    for _ in range(per_stratum)
                ),
            )
        )
    return tuple(draws)


# --- Interior policies -------------------------------------------------------


def exact_interior(*, samples: int = 16, seed_salt: int = 0):
    """Resolve the interior node with the exact T3 solver.

    This is the reference arm and the expensive one.  ``samples`` is how many
    T4 draws the BTN averages its own decision over; it is a knob because the
    arm's cost is linear in it and the BTN's CHOICE is far more stable than its
    values are.
    """

    def policy(observation: ThreeMaxObservation, decision_seed: int) -> Action:
        ranked = evaluate_t3(
            observation, samples=samples, seed=decision_seed ^ seed_salt
        )
        return ranked[0].action

    policy.__name__ = f"exact_interior(samples={samples},salt={seed_salt})"
    return policy


def mc_interior(*, sims: int = 32):
    """Resolve the interior node with the Monte-Carlo referee."""

    def policy(observation: ThreeMaxObservation, decision_seed: int) -> Action:
        return evaluate_actions_mc(observation, sims=sims, seed=decision_seed)[0].action

    policy.__name__ = f"mc_interior(sims={sims})"
    return policy


def random_interior():
    """Uniform over legal actions: the no-information control.

    Every measurement in this track gets one of these before it gets a verdict.
    Twice now a probe here has scored a policy that knew nothing as if it knew
    everything -- once because a mask was applied after the tensors were built,
    once because ties resolved toward a corpus sorted by the answer.
    """

    def policy(observation: ThreeMaxObservation, decision_seed: int) -> Action:
        actions = generate_actions(observation.hero_board, observation.dealt_cards)
        return actions[random.Random(decision_seed).randrange(len(actions))]

    policy.__name__ = "random_interior"
    return policy


def _build_ranker(state_dict: Mapping[str, "object"]):
    """Rebuild the trainer's MLP from the checkpoint's own shapes.

    Reading the widths out of the weights rather than importing ``Ranker`` from
    ``scripts/`` keeps this module a library module, and it cannot drift from a
    checkpoint the way a second copy of the architecture could.
    """
    from torch import nn

    indices = sorted(
        int(name.split(".")[1])
        for name in state_dict
        if name.startswith("body.") and name.endswith(".weight")
    )
    layers: list[nn.Module] = []
    for position, index in enumerate(indices):
        weight = state_dict[f"body.{index}.weight"]
        out_features, in_features = tuple(weight.shape)
        if position:
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features, out_features))
    body = nn.Sequential(*layers)
    body.load_state_dict(
        {
            f"{position * 2}.{suffix}": state_dict[f"body.{index}.{suffix}"]
            for position, index in enumerate(indices)
            for suffix in ("weight", "bias")
        }
    )
    body.eval()
    return body


@lru_cache(maxsize=4)
def _load_ranker(path: str):
    import torch

    # Interior policies run inside a per-core worker pool; a 17-row forward
    # pass has nothing to parallelise and every extra thread is contention.
    torch.set_num_threads(1)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    body = _build_ranker(checkpoint["state_dict"])
    input_dim = int(checkpoint["input_dim"])
    return body, checkpoint["mean"], checkpoint["std"], input_dim


def model_interior(model_path: str | Path, *, fl_ev_14: float = 9.6):
    """Resolve the interior node with the distilled T3-BTN ranker.

    The encoder has grown since the shipped checkpoints were trained (225 dims
    to 261), and it grew by APPENDING blocks, so the first 225 dims are still
    bit-identical to the old layout and a narrower model reads a prefix.  That
    is asserted, not assumed: a checkpoint of any other width is rejected
    rather than silently fed a misaligned vector.
    """
    import numpy as np
    import torch

    from . import features as _features

    path = str(Path(model_path))

    def policy(observation: ThreeMaxObservation, decision_seed: int) -> Action:
        body, mean, std, input_dim = _load_ranker(path)
        if input_dim > _features.FEATURE_SIZE:
            raise ValueError(
                f"checkpoint wants {input_dim} features, encoder emits "
                f"{_features.FEATURE_SIZE}"
            )
        actions = generate_actions(observation.hero_board, observation.dealt_cards)
        unseen = observation.unknown_cards()
        rows = []
        for action in actions:
            placed = observation.hero_board.place(action.placements)
            discarded = set(action.discards)
            rows.append(
                _features.encode(
                    hero_board=placed,
                    opponent_boards=observation.opponent_boards,
                    dealt_cards=observation.dealt_cards,
                    hero_private_discards=observation.hero_private_discards,
                    unseen=[card for card in unseen if card not in discarded],
                    fl_ev_14=fl_ev_14,
                ).features[:input_dim]
            )
        matrix = (np.asarray(rows, dtype=np.float32) - mean) / std
        with torch.no_grad():
            scores = body(torch.from_numpy(matrix)).squeeze(-1).numpy()
        return actions[int(np.argmax(scores))]

    policy.__name__ = f"model_interior({Path(model_path).name})"
    return policy


# --- The middle seat's label -------------------------------------------------


def _interior_observation(
    *,
    btn_board: Board,
    sb_board: Board,
    hero_board_after: Board,
    draw: MiddleDraw,
) -> ThreeMaxObservation:
    """The BTN's own view of its T3 decision, once the hero has moved.

    Opponent boards are act-relative, and ``opponent_seats("btn")`` is
    ``(sb, bb)``: the SB answers first at T4, the hero second.
    """
    return ThreeMaxObservation(
        hero_board=btn_board,
        opponent_boards=(sb_board, hero_board_after),
        dealt_cards=draw.btn_dealt,
        hero_private_discards=draw.btn_discards,
        seat=SEAT_BTN,
        street="T3",
    )


def evaluate_t3_middle(
    observation: ThreeMaxObservation,
    *,
    interior_policy: Callable[[ThreeMaxObservation, int], Action],
    draws: Sequence[MiddleDraw],
    seed: int = 0,
    fl_ev_per_pair: Mapping[int, float] | None = None,
) -> list[T3Candidate]:
    """Rank the middle seat's T3 actions under a given interior policy.

    Below the hero's move sit two things: the BTN's T3 decision, taken by
    ``interior_policy``, and then a T4 round resolved exactly by backward
    induction with the hero in the middle of it.  Only the first is an
    approximation, which is the point -- swap it and watch what the ranking
    does.

    The BTN's reply is recomputed for every hero action, because it is a reply:
    the BTN maximises its own total and the hero's finished board is half of
    what it is maximising against.  Whether that dependence is strong enough to
    matter is one of the things this is here to measure.
    """
    if observation.street != "T3":
        raise ValueError(f"middle-seat T3 evaluation needs T3, got {observation.street}")
    if act_order_of(observation.seat) != MIDDLE_HERO_INDEX:
        raise ValueError(
            f"evaluate_t3_middle is for the middle actor, got seat {observation.seat!r}"
        )
    btn_board, sb_board = observation.opponent_boards
    if btn_board.card_count() != 9 or sb_board.card_count() != 11:
        raise ValueError(
            "middle-seat T3 expects the BTN on nine cards and the SB on eleven"
        )
    if not draws:
        raise ValueError("at least one stratum is required")

    table = dict(fl_ev_per_pair or DEFAULT_FL_EV_PER_PAIR)
    fl_ev_14 = float(table.get(14, 0.0))

    hero_actions = generate_actions(observation.hero_board, observation.dealt_cards)
    hero_boards = [
        observation.hero_board.place(action.placements) for action in hero_actions
    ]

    totals = [0.0] * len(hero_actions)
    samples = 0
    for stratum_index, draw in enumerate(draws):
        decision_seed = seed * 1_000_003 + stratum_index
        btn_finals = [
            btn_board.place(
                interior_policy(
                    _interior_observation(
                        btn_board=btn_board,
                        sb_board=sb_board,
                        hero_board_after=hero_after,
                        draw=draw,
                    ),
                    decision_seed,
                ).placements
            )
            for hero_after in hero_boards
        ]

        # The hero's own T4 completions do not depend on the stratum's interior
        # cards, only on its T4 deal, so they are built once per (action, deal).
        for t4 in draw.t4_draws:
            sb_options = _completing_terminals(sb_board, t4[0:3])
            for index, hero_after in enumerate(hero_boards):
                totals[index] += round_hero_value(
                    (
                        sb_options,
                        _completing_terminals(hero_after, t4[3:6]),
                        _completing_terminals(btn_finals[index], t4[6:9]),
                    ),
                    MIDDLE_HERO_INDEX,
                    fl_ev_14,
                )
            samples += 1

    candidates = [
        T3Candidate(
            action=action,
            board=board,
            ev=total / samples,
            samples=samples,
            leaf="exact_t4",
        )
        for action, board, total in zip(hero_actions, hero_boards, totals)
    ]
    candidates.sort(
        key=lambda item: (
            -item.ev,
            _key({row: getattr(item.board, row) for row in ROWS}),
        )
    )
    return candidates


def action_token(action: Action) -> str:
    """A stable key for matching one action across policy arms."""
    return action_key(action).to_token()
