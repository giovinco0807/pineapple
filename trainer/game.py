"""Training session orchestration for the regular OFC trainer.

A session is one heads-up hand: hero vs an AI opponent driven by the same
evaluator stack at fast precision.  Hero decisions are graded against the
full candidate ranking (background-evaluated while the hero thinks).
"""

from __future__ import annotations

import logging
import random
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from ofc_regular.cards import create_deck
from ofc_regular.evaluator import score_board
from ofc_regular.state import Board

logger = logging.getLogger("trainer.game")

# Deal sizes per street: T0 = 5 cards, T1..T4 = 3 cards.
STREET_DEAL = {0: 5, 1: 3, 2: 3, 3: 3, 4: 3}

# Heads-up fixed-point estimate for a 14-card regular Fantasyland, read from
# configs/fl_ev_regular_v4_selfplay.json.
from trainer.fl_ev import FL_EV_14  # noqa: E402

_executor = ThreadPoolExecutor(max_workers=3)


def board_dict(board: Board) -> Dict[str, List[str]]:
    return {"top": list(board.top), "middle": list(board.middle), "bottom": list(board.bottom)}


def sorted_rows_key(rows: Dict[str, List[str]]) -> str:
    return "|".join(",".join(sorted(rows[r])) for r in ("top", "middle", "bottom"))


def candidate_key(cand: Dict[str, Any]) -> str:
    return sorted_rows_key(cand["board"])


def rank_score(cand: Dict[str, Any]) -> float:
    """The quantity the candidate list is ordered by.

    Grade against this rather than `ev`: at T0 the engine ranks on
    `candidate_score` and `ev` is not monotone in that order, so an `ev`
    difference between two rows can carry the opposite sign to their ranks.
    The MC evaluator emits no `rank_score` and is sorted by `ev`, so `ev` is
    the fallback rather than an error.
    """
    metrics = cand.get("metrics") or {}
    value = metrics.get("rank_score")
    if value is None:
        value = metrics.get("ev")
    return float(value or 0.0)


class TrainingSession:
    """One heads-up training hand. Hero is seat 0."""

    def __init__(
        self,
        evaluate: Callable[..., Dict[str, Any]],
        *,
        position: str = "random",
        precision: str = "standard",
        mistake_threshold: float = 1.0,
        on_mistake: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_hand_done: Optional[Callable[["TrainingSession"], None]] = None,
        rng: Optional[random.Random] = None,
    ):
        self.id = uuid.uuid4().hex[:12]
        self.evaluate = evaluate
        self.precision = precision
        self.mistake_threshold = float(mistake_threshold)
        self.on_mistake = on_mistake
        self.on_hand_done = on_hand_done
        self.created_at = time.time()

        rng = rng or random.Random()
        if position == "random":
            position = rng.choice(["first", "second"])
        self.position = position  # hero acts first or second each street

        deck = create_deck(shuffle=True, rng=rng)
        self._deck = deck
        self.hero_board = Board()
        self.opp_board = Board()
        self.hero_discards: List[str] = []
        self.opp_discards: List[str] = []
        self.street = 0
        self.hero_dealt: List[str] = []
        self.opp_dealt: List[str] = []
        self.hand_no = 1

        self.phase = "dealing"  # dealing | opp_turn | hero_turn | graded | complete | error
        self.error: Optional[str] = None
        self.grading: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []
        self.result: Optional[Dict[str, Any]] = None

        self._eval_future: Optional[Future] = None
        self._eval_result: Optional[Dict[str, Any]] = None
        self._eval_error: Optional[str] = None
        self._lock = threading.RLock()

        self._deal_street()
        _executor.submit(self._advance_to_hero)

    # ------------------------------------------------------------------
    # dealing / flow
    # ------------------------------------------------------------------

    def _deal_street(self) -> None:
        n = STREET_DEAL[self.street]
        self.hero_dealt = self._deck[:n]
        self.opp_dealt = self._deck[n : 2 * n]
        self._deck = self._deck[2 * n :]

    def _advance_to_hero(self) -> None:
        """Run AI moves until it is the hero's turn, then start hero eval."""
        try:
            with self._lock:
                if self.position == "second":
                    self.phase = "opp_turn"
            if self.position == "second":
                self._opponent_move()
            with self._lock:
                self.phase = "hero_turn"
            self._start_hero_eval()
        except Exception as exc:
            logger.exception("advance_to_hero failed")
            with self._lock:
                self.phase = "error"
                self.error = str(exc)

    def _advance_after_hero(self) -> None:
        """Opponent reply (if hero acted first), then next street or finish."""
        try:
            if self.position == "first":
                with self._lock:
                    self.phase = "opp_turn"
                self._opponent_move()
            if self.street >= 4:
                self._finish_hand()
                return
            with self._lock:
                self.street += 1
                self._deal_street()
                self.phase = "dealing"
            self._advance_to_hero()
        except Exception as exc:
            logger.exception("advance_after_hero failed")
            with self._lock:
                self.phase = "error"
                self.error = str(exc)

    def _opponent_move(self) -> None:
        """AI opponent plays what the learned model plays.

        `decide` rather than a full ranking: the opponent needs one move, and
        ranking all 232 T0 candidates for it cost ~20 s per hand with nothing
        reading the ranking.  Falls back to the ranked path if the engine
        refuses, which keeps the opponent playable on a partial pin set.
        """
        opp_kwargs = dict(
            hero_board=board_dict(self.opp_board),
            opp_board=board_dict(self.hero_board),
            dealt=list(self.opp_dealt),
            dead=list(self.opp_discards),
            turn=self.street,
            position="second" if self.position == "first" else "first",
            precision="fast",
        )
        best = None
        try:
            from trainer import engine_eval

            best = engine_eval.decide_with_engine(**opp_kwargs)
        except Exception as exc:
            logger.info("opponent decide unavailable (%s); falling back to ranking", exc)
        if best is None:
            result = self.evaluate(**opp_kwargs)
            candidates = result.get("candidates") or []
            if not candidates:
                raise RuntimeError("AI opponent found no candidates")
            best = candidates[0]
        with self._lock:
            self.opp_board = self.opp_board.place(
                [(c, r) for c, r in best["action"]["placements"]]
            )
            discard = best["action"].get("discard")
            if discard:
                self.opp_discards.append(discard)
            self.opp_dealt = []

    def _start_hero_eval(self) -> None:
        with self._lock:
            self._eval_result = None
            self._eval_error = None
            self._eval_future = _executor.submit(self._run_hero_eval, self.street)

    def _run_hero_eval(self, street: int) -> Dict[str, Any]:
        try:
            result = self.evaluate(
                hero_board=board_dict(self.hero_board),
                opp_board=board_dict(self.opp_board),
                dealt=list(self.hero_dealt),
                dead=list(self.hero_discards),
                turn=street,
                position=self.position,
                precision=self.precision,
            )
            with self._lock:
                self._eval_result = result
            return result
        except Exception as exc:
            logger.exception("hero eval failed (street %s)", street)
            with self._lock:
                self._eval_error = str(exc)
            raise

    # ------------------------------------------------------------------
    # hero actions
    # ------------------------------------------------------------------

    def wait_for_eval(self, timeout: float = 300.0) -> Optional[Dict[str, Any]]:
        future = self._eval_future
        if future is None:
            return None
        try:
            return future.result(timeout=timeout)
        except Exception:
            return None

    def act(
        self,
        placements: Optional[List[List[str]]] = None,
        discard: Optional[str] = None,
        auto_best: bool = False,
    ) -> None:
        with self._lock:
            if self.phase != "hero_turn":
                raise ValueError(f"cannot act in phase {self.phase}")

        evaluation = self.wait_for_eval()
        candidates = (evaluation or {}).get("candidates") or []

        if auto_best:
            if not candidates:
                raise RuntimeError("評価が失敗したため最善手を取得できません")
            best = candidates[0]
            placements = best["action"]["placements"]
            discard = best["action"].get("discard")

        if not placements:
            raise ValueError("placements is required")

        self._validate_action(placements, discard)

        # Grade against the candidate ranking.
        next_rows = board_dict(self.hero_board.place([(c, r) for c, r in placements]))
        user_key = sorted_rows_key(next_rows)
        user_cand = None
        user_rank = None
        for i, cand in enumerate(candidates):
            if candidate_key(cand) == user_key:
                user_cand = cand
                user_rank = i + 1
                break

        best_cand = candidates[0] if candidates else None
        ev_loss = None
        is_best = False
        if user_cand is not None and best_cand is not None:
            ev_loss = rank_score(best_cand) - rank_score(user_cand)
            is_best = user_rank == 1 or ev_loss <= 1e-9
            ev_loss = max(0.0, ev_loss)

        user_action = {"placements": [[c, r] for c, r in placements], "discard": discard}
        mistake = ev_loss is not None and ev_loss >= self.mistake_threshold and not is_best

        grading = {
            "street": self.street,
            "ev_loss": ev_loss,
            "rank": user_rank,
            "is_best": is_best,
            "mistake": mistake,
            "user_key": user_key,
            "user": user_cand
            or {"action": user_action, "board": next_rows, "metrics": {"ev": None}, "key": user_key},
            "best": best_cand,
            "candidates": candidates[:12],
            "candidate_count": len(candidates),
            "evaluator": (evaluation or {}).get("evaluator", ""),
            "eval_error": self._eval_error,
            "auto_best": auto_best,
        }

        with self._lock:
            self.grading = grading
            self.phase = "graded"
            self.history.append(
                {
                    "street": self.street,
                    "ev_loss": ev_loss,
                    "is_best": is_best,
                    "rank": user_rank,
                    "mistake": mistake,
                    "auto_best": auto_best,
                }
            )

        if mistake and self.on_mistake:
            try:
                self.on_mistake(
                    {
                        "session_id": self.id,
                        "hand_no": self.hand_no,
                        "street": self.street,
                        "position": self.position,
                        "hero_board": board_dict(self.hero_board),
                        "opp_board": board_dict(self.opp_board),
                        "dealt": list(self.hero_dealt),
                        "dead_cards": list(self.hero_discards),
                        "user_action": user_action,
                        "best_action": best_cand["action"] if best_cand else {},
                        "ev_loss": ev_loss,
                        "user_rank": user_rank,
                        "candidates": candidates[:20],
                        "evaluator": grading["evaluator"],
                    }
                )
            except Exception:
                logger.exception("mistake persistence failed")

    def continue_hand(self, use: str = "user") -> None:
        with self._lock:
            if self.phase != "graded" or not self.grading:
                raise ValueError(f"cannot continue in phase {self.phase}")
            grading = self.grading
            chosen = grading["best"] if (use == "best" and grading["best"]) else grading["user"]
            action = chosen["action"]
            self.hero_board = self.hero_board.place(
                [(c, r) for c, r in action["placements"]]
            )
            if action.get("discard"):
                self.hero_discards.append(action["discard"])
            self.hero_dealt = []
            self.grading = None
            self.phase = "opp_turn" if self.position == "first" else "dealing"
        _executor.submit(self._advance_after_hero)

    def _validate_action(self, placements: List[List[str]], discard: Optional[str]) -> None:
        dealt = set(self.hero_dealt)
        need = 5 if self.street == 0 else 2
        if len(placements) != need:
            raise ValueError(f"T{self.street} は {need} 枚配置してください")
        used = [c for c, _ in placements]
        if len(set(used)) != len(used):
            raise ValueError("同じカードを複数回使っています")
        for card in used:
            if card not in dealt:
                raise ValueError(f"{card} は配られていません")
        if self.street > 0:
            leftover = dealt - set(used)
            if discard is None or set([discard]) != leftover:
                raise ValueError("捨て札が不正です")
        # capacity check
        self.hero_board.place([(c, r) for c, r in placements])

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def _finish_hand(self) -> None:
        hero = score_board(self.hero_board.top, self.hero_board.middle, self.hero_board.bottom)
        opp = score_board(self.opp_board.top, self.opp_board.middle, self.opp_board.bottom)

        line_results = [0, 0, 0]
        if not hero.busted and not opp.busted:
            pairs = [
                (hero.top_value, opp.top_value),
                (hero.middle_value, opp.middle_value),
                (hero.bottom_value, opp.bottom_value),
            ]
            for i, (hv, ov) in enumerate(pairs):
                line_results[i] = 1 if hv > ov else -1 if hv < ov else 0

        scoop = False
        if hero.busted and opp.busted:
            score = 0.0
        elif hero.busted:
            score = -(6 + opp.total_royalty)
        elif opp.busted:
            score = 6 + hero.total_royalty
        else:
            line_total = sum(line_results)
            scoop = abs(line_total) == 3
            score = line_total + (3 if line_total == 3 else -3 if line_total == -3 else 0)
            score += hero.total_royalty - opp.total_royalty

        total_ev_loss = sum(h["ev_loss"] or 0.0 for h in self.history)
        mistakes = sum(1 for h in self.history if h["mistake"])

        def fl_info(board_score):
            fl = board_score.fl_entry
            if fl.qualifies:
                return {"entry": True, "type": fl.entry_type, "cards": fl.card_count, "ev": FL_EV_14}
            return {"entry": False}

        line_names = {1: "○", 0: "△", -1: "×"}
        result = {
            "score": score,
            "line_results": line_results,
            "line_text": " ".join(
                f"{label}{line_names[v]}"
                for label, v in zip(["上", "中", "下"], line_results)
            )
            if not hero.busted and not opp.busted
            else "-",
            "scoop": scoop,
            "hero_busted": hero.busted,
            "opp_busted": opp.busted,
            "hero_royalty": hero.total_royalty,
            "opp_royalty": opp.total_royalty,
            "hero_fl": fl_info(hero),
            "opp_fl": fl_info(opp),
            "total_ev_loss": total_ev_loss,
            "mistakes": mistakes,
        }
        with self._lock:
            self.result = result
            self.phase = "complete"
        if self.on_hand_done:
            try:
                self.on_hand_done(self)
            except Exception:
                logger.exception("hand persistence failed")

    # ------------------------------------------------------------------
    # state serialization
    # ------------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        with self._lock:
            eval_status = "none"
            if self.phase == "hero_turn":
                if self._eval_error:
                    eval_status = "error"
                elif self._eval_result is not None:
                    eval_status = "ready"
                elif self._eval_future is not None:
                    eval_status = "running"
            candidate_count = None
            if self._eval_result is not None:
                candidate_count = len(self._eval_result.get("candidates") or [])
            return {
                "session_id": self.id,
                "phase": self.phase,
                "street": self.street,
                "hand_no": self.hand_no,
                "position": self.position,
                "hero_board": board_dict(self.hero_board),
                "opp_board": board_dict(self.opp_board),
                "dealt": list(self.hero_dealt) if self.phase in ("hero_turn", "graded") else [],
                "discards": list(self.hero_discards),
                "eval_status": eval_status,
                "candidate_count": candidate_count,
                "error": self.error,
                "history": list(self.history),
                "grading": self.grading,
                "result": self.result,
                "hero_fl": False,
                "opp_fl": False,
            }
