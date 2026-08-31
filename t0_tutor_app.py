import copy
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.engine.action_space import get_initial_actions
from ai.engine.encoding import Board, Observation, encode_state
from ai.tutor.exact_late import (
    board_to_dict,
    evaluate_late_position,
    normalize_position_payload,
    stable_cache_key,
)

try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth
    from firebase_admin import firestore
except Exception:  # pragma: no cover - exercised only when optional deps are absent
    firebase_admin = None
    firebase_auth = None
    firestore = None

try:
    import stripe
except Exception:  # pragma: no cover - exercised only when optional deps are absent
    stripe = None


ROOT_DIR = Path(__file__).parent
TRAINING_PRESETS_PATH = ROOT_DIR / "training_presets.json"
ROUTE10_REVIEW_PATH = Path(
    os.environ.get(
        "ROUTE10_REVIEW_PATH",
        str(ROOT_DIR / "ai" / "data" / "tutor_route10_20260522" / "tutor_route10_review.json"),
    )
)
ON_DEMAND_CACHE_MAX = int(os.environ.get("OFC_TUTOR_ON_DEMAND_CACHE_MAX", "1024"))
T2_SAFE_CANDIDATE_BUDGET = int(os.environ.get("OFC_TUTOR_T2_SAFE_CANDIDATE_BUDGET", "24"))

FREE_DAILY_TRAINING_LIMIT = 10
STARTER_DAILY_TRAINING_LIMIT = 30
STARTER_DAILY_CUSTOM_LIMIT = 15
PREMIUM_DAILY_TRAINING_LIMIT = 50
PREMIUM_DAILY_CUSTOM_LIMIT = 50

PAID_PLANS = {"starter", "premium"}
ACTIVE_SUBSCRIPTION_STATUSES = {"active", "trialing"}

STRIPE_API_VERSION = "2026-02-25.clover"
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_IDS = {
    "starter": os.environ.get("STRIPE_STARTER_PRICE_ID", ""),
    "premium": os.environ.get("STRIPE_PREMIUM_PRICE_ID", ""),
}

FIREBASE_PUBLIC_CONFIG_KEYS = {
    "apiKey": "FIREBASE_API_KEY",
    "authDomain": "FIREBASE_AUTH_DOMAIN",
    "projectId": "FIREBASE_PROJECT_ID",
    "storageBucket": "FIREBASE_STORAGE_BUCKET",
    "messagingSenderId": "FIREBASE_MESSAGING_SENDER_ID",
    "appId": "FIREBASE_APP_ID",
    "measurementId": "FIREBASE_MEASUREMENT_ID",
}

TRAINING_PRESETS: Dict[str, Any] = {}
if TRAINING_PRESETS_PATH.exists():
    with TRAINING_PRESETS_PATH.open("r", encoding="utf-8") as f:
        TRAINING_PRESETS = json.load(f)

ON_DEMAND_EVAL_CACHE: Dict[str, Dict[str, Any]] = {}


class MemoryDocumentSnapshot:
    def __init__(self, doc_id: str, data: Optional[Dict[str, Any]]):
        self.id = doc_id
        self._data = copy.deepcopy(data) if data is not None else None
        self.exists = data is not None

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data or {})


class MemoryDocumentReference:
    def __init__(self, store: Dict[str, Dict[str, Any]], doc_id: str):
        self._store = store
        self.id = doc_id

    def get(self) -> MemoryDocumentSnapshot:
        return MemoryDocumentSnapshot(self.id, self._store.get(self.id))

    def set(self, data: Dict[str, Any]) -> None:
        self._store[self.id] = copy.deepcopy(data)

    def update(self, data: Dict[str, Any]) -> None:
        existing = self._store.setdefault(self.id, {})
        existing.update(copy.deepcopy(data))


class MemoryQuery:
    def __init__(self, store: Dict[str, Dict[str, Any]], field: str, expected: Any):
        self._store = store
        self._field = field
        self._expected = expected
        self._limit: Optional[int] = None

    def limit(self, n: int) -> "MemoryQuery":
        self._limit = n
        return self

    def stream(self) -> Iterable[MemoryDocumentSnapshot]:
        yielded = 0
        for doc_id, data in self._store.items():
            if data.get(self._field) == self._expected:
                yield MemoryDocumentSnapshot(doc_id, data)
                yielded += 1
                if self._limit is not None and yielded >= self._limit:
                    return


class MemoryCollectionReference:
    def __init__(self, store: Dict[str, Dict[str, Any]]):
        self._store = store

    def document(self, doc_id: str) -> MemoryDocumentReference:
        return MemoryDocumentReference(self._store, doc_id)

    def where(self, field: str, op: str, expected: Any) -> MemoryQuery:
        if op != "==":
            raise ValueError("MemoryFirestore only supports equality filters")
        return MemoryQuery(self._store, field, expected)


class MemoryFirestore:
    def __init__(self):
        self._collections: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def collection(self, name: str) -> MemoryCollectionReference:
        return MemoryCollectionReference(self._collections.setdefault(name, {}))


def init_db():
    if os.environ.get("OFC_TUTOR_MEMORY_DB") == "1":
        return MemoryFirestore()
    if firebase_admin is None or firestore is None:
        print("[Tutor] firebase-admin unavailable; using in-memory user store")
        return MemoryFirestore()
    try:
        try:
            firebase_admin.get_app()
        except ValueError:
            firebase_admin.initialize_app()
        return firestore.client()
    except Exception as exc:
        print(f"[Tutor] Firebase initialization failed; using in-memory user store: {exc}")
        return MemoryFirestore()


db = init_db()

if stripe is not None:
    stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
    stripe.api_version = STRIPE_API_VERSION


app = FastAPI(title="OFC T0 Tutor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = ROOT_DIR / "tutor_static"
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
async def startup():
    print("Initializing AI models...")
    try:
        from backend.ai_player import init_ai

        init_ai()
    except Exception as exc:
        print(f"[Tutor] AI initialization skipped/failed: {exc}")
    print("AI startup complete")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = static_dir / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>HTML file not found in tutor_static/index.html</h1>"


@app.get("/api/config")
async def get_config():
    return {
        "firebase": {
            public_key: os.environ.get(env_key, "")
            for public_key, env_key in FIREBASE_PUBLIC_CONFIG_KEYS.items()
        },
        "dev_auth": os.environ.get("OFC_TUTOR_DEV_AUTH") == "1",
        "plans": {
            "free": {
                "training_daily_limit": FREE_DAILY_TRAINING_LIMIT,
                "custom_daily_limit": 0,
            },
            "starter": {
                "training_daily_limit": STARTER_DAILY_TRAINING_LIMIT,
                "custom_daily_limit": STARTER_DAILY_CUSTOM_LIMIT,
            },
            "premium": {
                "training_daily_limit": PREMIUM_DAILY_TRAINING_LIMIT,
                "custom_daily_limit": PREMIUM_DAILY_CUSTOM_LIMIT,
            },
        },
    }


async def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = authorization.split("Bearer ", 1)[1]

    if os.environ.get("OFC_TUTOR_DEV_AUTH") == "1" and token.startswith("dev:"):
        uid = token.split("dev:", 1)[1] or "dev-user"
        return {"uid": uid, "email": f"{uid}@dev.local"}

    if firebase_auth is None:
        raise HTTPException(status_code=503, detail="Firebase auth is not configured")

    try:
        return firebase_auth.verify_id_token(token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_user_data(email: str) -> Dict[str, Any]:
    today = utc_today()
    return {
        "email": email,
        "plan": "free",
        "subscription_status": "inactive",
        "daily_training_usage": 0,
        "daily_custom_usage": 0,
        "last_usage_date": today,
        "stripe_customer_id": None,
        "stripe_subscription_id": None,
        "price_id": None,
        "last_checkout_plan": None,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }


def ensure_user_defaults(data: Dict[str, Any], email: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    defaults = default_user_data(email)
    updates = {}
    for key, value in defaults.items():
        if key not in data:
            data[key] = value
            updates[key] = value

    today = utc_today()
    if data.get("last_usage_date") != today:
        data["daily_training_usage"] = 0
        data["daily_custom_usage"] = 0
        data["last_usage_date"] = today
        updates.update(
            {
                "daily_training_usage": 0,
                "daily_custom_usage": 0,
                "last_usage_date": today,
            }
        )
    return data, updates


def get_user_doc(uid: str, email: str):
    doc_ref = db.collection("users").document(uid)
    doc = doc_ref.get()

    if not doc.exists:
        data = default_user_data(email)
        doc_ref.set(data)
        return doc_ref, data

    data = doc.to_dict()
    data, updates = ensure_user_defaults(data, email)
    if updates:
        updates["updated_at"] = now_iso()
        doc_ref.update(updates)
    return doc_ref, data


def effective_plan(user_data: Dict[str, Any]) -> str:
    plan = user_data.get("plan", "free")
    if plan not in PAID_PLANS:
        return "free"

    status = user_data.get("subscription_status")
    has_subscription = bool(user_data.get("stripe_subscription_id"))
    if status in ACTIVE_SUBSCRIPTION_STATUSES:
        return plan
    if not has_subscription and status in (None, "", "inactive"):
        # Allows manual/admin plan overrides while keeping canceled Stripe users gated.
        return plan
    return "free"


def training_limit_for(plan: str) -> int:
    if plan == "premium":
        return PREMIUM_DAILY_TRAINING_LIMIT
    if plan == "starter":
        return STARTER_DAILY_TRAINING_LIMIT
    return FREE_DAILY_TRAINING_LIMIT


def custom_limit_for(plan: str) -> int:
    if plan == "premium":
        return PREMIUM_DAILY_CUSTOM_LIMIT
    if plan == "starter":
        return STARTER_DAILY_CUSTOM_LIMIT
    return 0


def price_id_to_plan(price_id: Optional[str]) -> Optional[str]:
    for plan, configured_price_id in STRIPE_PRICE_IDS.items():
        if configured_price_id and configured_price_id == price_id:
            return plan
    return None


def find_user_doc_by_field(field: str, value: Optional[str]):
    if not value:
        return None, None
    query = db.collection("users").where(field, "==", value).limit(1)
    for snapshot in query.stream():
        return db.collection("users").document(snapshot.id), snapshot.to_dict()
    return None, None


def load_route10_review() -> Dict[str, Any]:
    if not ROUTE10_REVIEW_PATH.exists():
        raise HTTPException(status_code=404, detail="Route10 review data not found")
    with ROUTE10_REVIEW_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def route10_decision_summary(decision: Dict[str, Any], locked: bool) -> Dict[str, Any]:
    return {
        "seat": decision.get("seat"),
        "position": decision.get("position"),
        "dealt": decision.get("dealt", []),
        "objective": decision.get("objective"),
        "settings": decision.get("settings", {}),
        "best": decision.get("best", {}),
        "metric_source": decision.get("metric_source", "estimated"),
        "locked": locked,
    }


def find_route10_decision(review: Dict[str, Any], hand: int, seat: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    normalized_seat = seat.lower()
    for hand_data in review.get("hands", []):
        if int(hand_data.get("hand")) != hand:
            continue
        for decision in hand_data.get("decisions", []):
            if str(decision.get("seat", "")).lower() == normalized_seat:
                return hand_data, decision
    raise HTTPException(status_code=404, detail="Route10 decision not found")


def route10_access_for(user_data: Dict[str, Any]) -> Dict[str, Any]:
    plan = effective_plan(user_data)
    return {"plan": plan, "paid": plan in PAID_PLANS}


def _on_demand_sims(precision: str) -> int:
    if precision == "high":
        return 300
    if precision == "fast":
        return 20
    return 100


def _candidate_budget_for_turn(turn: int, top_n: int) -> int:
    budget = max(1, int(top_n))
    if turn == 2 and T2_SAFE_CANDIDATE_BUDGET > 0:
        budget = max(budget, T2_SAFE_CANDIDATE_BUDGET)
    return min(budget, 50)


def _cache_get(cache_key: str) -> Optional[Dict[str, Any]]:
    cached = ON_DEMAND_EVAL_CACHE.get(cache_key)
    return copy.deepcopy(cached) if cached is not None else None


def _cache_set(cache_key: str, value: Dict[str, Any]) -> None:
    if ON_DEMAND_CACHE_MAX <= 0:
        return
    if len(ON_DEMAND_EVAL_CACHE) >= ON_DEMAND_CACHE_MAX:
        oldest = next(iter(ON_DEMAND_EVAL_CACHE))
        ON_DEMAND_EVAL_CACHE.pop(oldest, None)
    ON_DEMAND_EVAL_CACHE[cache_key] = copy.deepcopy(value)


def _opponent_cards(board: Board) -> list[str]:
    return list(board.top) + list(board.middle) + list(board.bottom)


def _normalize_engine_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    mc = candidate.get("mc") or {}
    fl_types = mc.get("fl_type_rates") or {}
    metrics = {
        "ev": float(candidate.get("ev", mc.get("avg_score", 0.0)) or 0.0),
        "score": float(candidate.get("target_score", candidate.get("ev", mc.get("avg_score", 0.0))) or 0.0),
        "bust_rate": float(candidate.get("bust_prob", mc.get("bust_rate", 0.0)) or 0.0),
        "fl_rate": float(candidate.get("fl_rate", mc.get("fl_rate", 0.0)) or 0.0),
        "fl_type_rates": {
            "qq": float(fl_types.get("qq", 0.0) or 0.0),
            "kk": float(fl_types.get("kk", 0.0) or 0.0),
            "aa": float(fl_types.get("aa", 0.0) or 0.0),
            "trips": float(fl_types.get("trips", 0.0) or 0.0),
        },
        "sims": int(mc.get("simulations", 0) or 0),
        "source": "estimated",
    }
    return {
        "action": {
            "placements": candidate.get("placements", []),
            "discard": candidate.get("discard"),
        },
        "board": candidate.get("board") or candidate.get("next_board"),
        "metrics": metrics,
    }


def evaluate_position_payload(req: Dict[str, Any], plan: str) -> Dict[str, Any]:
    turn = int(req.get("turn", 0))
    precision = str(req.get("precision") or "standard").lower()
    if precision not in {"fast", "standard", "high", "exact"}:
        return {"error": "INVALID_PRECISION", "message": "precision must be fast, standard, high, or exact."}

    board, opponent_board, dealt, exclude = normalize_position_payload(req)
    expected_cards = 5 if turn == 0 else 3
    if len(dealt) != expected_cards:
        return {"error": "INVALID_CARDS", "message": f"Turn {turn} requires exactly {expected_cards} dealt cards."}

    requested_top_n = int(req.get("top_n", 0) or 0)
    top_n = requested_top_n or 20
    top_n = max(1, min(top_n, 50))
    candidate_budget = _candidate_budget_for_turn(turn, top_n)
    response_top_n = candidate_budget if turn == 2 and requested_top_n <= 0 else top_n
    normalized_req = {
        "turn": turn,
        "precision": precision,
        "top_n": top_n,
        "candidate_budget": candidate_budget,
        "board": board_to_dict(board),
        "opponent_board": board_to_dict(opponent_board),
        "dealt": dealt,
        "exclude": exclude,
    }
    cache_key = stable_cache_key(normalized_req)
    cached = _cache_get(cache_key)
    if cached is not None:
        cached["cached"] = True
        cached["cache_key"] = cache_key
        return cached

    if turn in (3, 4):
        result = evaluate_late_position(
            board=board,
            dealt=dealt,
            turn=turn,
            opponent_board=opponent_board,
            exclude=exclude,
            top_n=top_n,
        )
        result["precision"] = "exact"
    else:
        try:
            from ai.prob_engine_wrapper import evaluate_mc_candidates, evaluate_mc_t0
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Probability engine wrapper is unavailable: {exc}")

        sims = _on_demand_sims(precision)
        engine_exclude = exclude + _opponent_cards(opponent_board)
        try:
            if turn == 0:
                engine_result = evaluate_mc_t0(
                    dealt=dealt,
                    exclude=engine_exclude,
                    sims=sims,
                    candidate_limit=max(top_n, 20),
                    candidate_filter="all",
                )
            elif turn in (1, 2):
                engine_result = evaluate_mc_candidates(
                    top=board.top,
                    mid=board.middle,
                    bot=board.bottom,
                    dealt=dealt,
                    exclude=engine_exclude,
                    turn=turn,
                    sims=sims,
                    candidate_limit=candidate_budget if turn == 2 else 0,
                )
            else:
                return {"error": "INVALID_TURN", "message": "turn must be 0 through 4."}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Probability engine evaluation failed: {exc}")

        raw_candidates = engine_result.get("candidates") or []
        candidates = [_normalize_engine_candidate(candidate) for candidate in raw_candidates]
        result = {
            "turn": turn,
            "board": board_to_dict(board),
            "dealt": dealt,
            "legal_actions": len(raw_candidates),
            "chosen_action": candidates[0]["action"] if candidates else None,
            "best": candidates[0] if candidates else None,
            "candidates": candidates[:response_top_n],
            "candidate_count": len(candidates),
            "requested_top_n": top_n,
            "candidate_budget": candidate_budget,
            "safety_budget_applied": candidate_budget > top_n,
            "source": "estimated",
            "precision": precision,
            "sims": sims,
        }

    result["plan"] = plan
    result["cached"] = False
    result["cache_key"] = cache_key
    _cache_set(cache_key, result)
    return result


@app.get("/api/me")
async def get_me(decoded_token: dict = Depends(verify_token)):
    uid = decoded_token.get("uid")
    email = decoded_token.get("email") or ""
    _, user_data = get_user_doc(uid, email)
    response = copy.deepcopy(user_data)
    response["effective_plan"] = effective_plan(user_data)
    response["limits"] = {
        "training_daily_limit": training_limit_for(response["effective_plan"]),
        "custom_daily_limit": custom_limit_for(response["effective_plan"]),
    }
    return response


@app.get("/api/training_puzzle")
async def get_training_puzzle(decoded_token: dict = Depends(verify_token)):
    uid = decoded_token.get("uid")
    email = decoded_token.get("email") or ""
    _, user_data = get_user_doc(uid, email)

    plan = effective_plan(user_data)
    training_usage = user_data.get("daily_training_usage", 0)
    limit = training_limit_for(plan)

    if training_usage >= limit:
        return {
            "error": "DAILY_LIMIT_REACHED",
            "message": f"Training limit of {limit} reached for your plan.",
        }

    if not TRAINING_PRESETS:
        return {"error": "NO_TRAINING_PRESETS", "message": "No training presets available."}

    puzzle_id = random.choice(list(TRAINING_PRESETS.keys()))
    cards = TRAINING_PRESETS[puzzle_id]["cards"]

    return {
        "puzzle_id": puzzle_id,
        "cards": cards,
        "remaining": max(limit - training_usage, 0),
        "limit": limit,
    }


@app.post("/api/evaluate_training")
async def evaluate_training(req: dict, decoded_token: dict = Depends(verify_token)):
    uid = decoded_token.get("uid")
    email = decoded_token.get("email") or ""
    doc_ref, user_data = get_user_doc(uid, email)

    plan = effective_plan(user_data)
    training_usage = user_data.get("daily_training_usage", 0)
    limit = training_limit_for(plan)

    if training_usage >= limit:
        return {"error": "DAILY_LIMIT_REACHED", "message": f"Training limit of {limit} reached."}

    puzzle_id = req.get("puzzle_id")
    user_placement = req.get("placement")

    if puzzle_id not in TRAINING_PRESETS:
        return {"error": "INVALID_PUZZLE_ID", "message": "Invalid puzzle ID."}

    puzzle_data = TRAINING_PRESETS[puzzle_id]
    results = puzzle_data["top_placements"]

    user_result = None
    if user_placement:
        for result in results:
            placement = result["placement"]
            if (
                sorted(placement["top"]) == sorted(user_placement.get("top", []))
                and sorted(placement["middle"]) == sorted(user_placement.get("middle", []))
                and sorted(placement["bottom"]) == sorted(user_placement.get("bottom", []))
            ):
                user_result = result
                break

    response_data = {
        "top_placements": results[:10],
        "optimal_ev": puzzle_data["optimal_ev"],
    }

    if user_result:
        response_data["user_placement"] = user_result
        response_data["ev_diff"] = user_result["ev"] - puzzle_data["optimal_ev"]
        response_data["is_optimal"] = user_result == results[0]

    doc_ref.update({"daily_training_usage": training_usage + 1, "updated_at": now_iso()})
    return response_data


@app.post("/api/evaluate_t0")
async def evaluate_t0(req: dict, decoded_token: dict = Depends(verify_token)):
    uid = decoded_token.get("uid")
    email = decoded_token.get("email") or ""
    doc_ref, user_data = get_user_doc(uid, email)

    plan = effective_plan(user_data)
    if plan == "free":
        return {"error": "CUSTOM_LOCKED", "message": "Custom mode is not available on the free plan."}

    custom_usage = user_data.get("daily_custom_usage", 0)
    limit = custom_limit_for(plan)
    if custom_usage >= limit:
        return {"error": "DAILY_LIMIT_REACHED", "message": f"Custom plan limit of {limit} reached."}

    cards = req.get("cards")
    if not cards or len(cards) != 5:
        return {"error": "INVALID_CARDS", "message": "Must provide exactly 5 cards."}

    try:
        from backend import ai_player

        if not ai_player.init_ai() or ai_player._ai_re1 is None or ai_player._ai_re1.value_net is None:
            raise RuntimeError("AI value model is not available")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"AI evaluator is unavailable: {exc}")

    user_placement = req.get("placement")
    valid_actions = get_initial_actions(cards, Board())
    if not valid_actions:
        return {"error": "NO_VALID_ACTIONS", "message": "No valid actions."}

    states = []
    placements = []
    for action in valid_actions:
        board = Board()
        for card, position in action.placements:
            getattr(board, position).append(card)

        obs = Observation(
            board_self=board,
            board_opponent=Board(),
            dealt_cards=[],
            known_discards_self=[],
            turn=1,
            is_btn=True,
        )
        states.append(encode_state(obs))
        placements.append(
            {
                "top": list(board.top),
                "middle": list(board.middle),
                "bottom": list(board.bottom),
            }
        )

    tensor = torch.FloatTensor(np.array(states))
    evaluator = ai_player._ai_re1
    value_net = evaluator.value_net
    mean = evaluator.score_mean
    std = evaluator.score_std

    with torch.no_grad():
        model_result = value_net(tensor)

    values = model_result["value"].squeeze(-1).numpy()
    values = values * std + mean

    results = []
    for index in range(len(valid_actions)):
        results.append(
            {
                "placement": placements[index],
                "ev": float(values[index]),
                "bust_prob": float(model_result["bust_prob"][index].item())
                if "bust_prob" in model_result
                else 0.0,
                "fl_prob": float(model_result["fl_prob"][index].item())
                if "fl_prob" in model_result
                else 0.0,
                "royalty_ev": float(model_result["royalty_ev"][index].item())
                if "royalty_ev" in model_result
                else 0.0,
            }
        )

    results.sort(key=lambda item: item["ev"], reverse=True)

    user_result = None
    if user_placement:
        for result in results:
            placement = result["placement"]
            if (
                sorted(placement["top"]) == sorted(user_placement.get("top", []))
                and sorted(placement["middle"]) == sorted(user_placement.get("middle", []))
                and sorted(placement["bottom"]) == sorted(user_placement.get("bottom", []))
            ):
                user_result = result
                break

    response_data = {
        "top_placements": results[:10],
        "optimal_ev": results[0]["ev"],
    }

    if user_result:
        response_data["user_placement"] = user_result
        response_data["ev_diff"] = user_result["ev"] - results[0]["ev"]
        response_data["is_optimal"] = user_result == results[0]

    doc_ref.update({"daily_custom_usage": custom_usage + 1, "updated_at": now_iso()})
    return response_data


@app.post("/api/evaluate_position")
async def evaluate_position(req: dict, decoded_token: dict = Depends(verify_token)):
    uid = decoded_token.get("uid")
    email = decoded_token.get("email") or ""
    doc_ref, user_data = get_user_doc(uid, email)

    plan = effective_plan(user_data)
    if plan == "free":
        return {"error": "CUSTOM_LOCKED", "message": "On-demand position evaluation is not available on the free plan."}

    custom_usage = user_data.get("daily_custom_usage", 0)
    limit = custom_limit_for(plan)
    if custom_usage >= limit:
        return {"error": "DAILY_LIMIT_REACHED", "message": f"Custom plan limit of {limit} reached."}

    result = evaluate_position_payload(req, plan)
    if "error" in result:
        return result

    if not result.get("cached"):
        doc_ref.update({"daily_custom_usage": custom_usage + 1, "updated_at": now_iso()})
    return result


@app.get("/api/reviews/route10")
async def get_route10_review(decoded_token: dict = Depends(verify_token)):
    uid = decoded_token.get("uid")
    email = decoded_token.get("email") or ""
    _, user_data = get_user_doc(uid, email)
    access = route10_access_for(user_data)
    review = load_route10_review()

    hands = []
    for hand_data in review.get("hands", []):
        hand_number = int(hand_data.get("hand"))
        decisions = []
        for decision in hand_data.get("decisions", []):
            seat = str(decision.get("seat", "")).lower()
            is_free_sample = hand_number == 1 and seat == "bb"
            decisions.append(route10_decision_summary(decision, locked=not access["paid"] and not is_free_sample))
        hands.append(
            {
                "hand": hand_number,
                "bb_dealt": hand_data.get("bb_dealt", []),
                "btn_dealt": hand_data.get("btn_dealt", []),
                "decisions": decisions,
            }
        )

    response = {
        "run": review.get("run"),
        "generated_records": review.get("generated_records"),
        "data_quality": review.get(
            "data_quality",
            {
                "t0_candidates": "estimated",
                "t0_sims": 300,
                "turn_traces": "estimated",
                "turn_trace_sims": 96,
                "exact_terminal_available": False,
            },
        ),
        "access": "paid" if access["paid"] else "free",
        "plan": access["plan"],
        "hands": hands,
    }

    if not access["paid"]:
        _, sample_decision = find_route10_decision(review, 1, "bb")
        response["sample"] = copy.deepcopy(sample_decision)

    return response


@app.get("/api/reviews/route10/hands/{hand}/decisions/{seat}")
async def get_route10_decision(hand: int, seat: str, decoded_token: dict = Depends(verify_token)):
    uid = decoded_token.get("uid")
    email = decoded_token.get("email") or ""
    _, user_data = get_user_doc(uid, email)
    access = route10_access_for(user_data)
    normalized_seat = seat.lower()

    if not access["paid"] and not (hand == 1 and normalized_seat == "bb"):
        raise HTTPException(
            status_code=403,
            detail="Route10 full review is available on Starter and Premium plans.",
        )

    review = load_route10_review()
    hand_data, decision = find_route10_decision(review, hand, normalized_seat)
    return {
        "run": review.get("run"),
        "data_quality": review.get(
            "data_quality",
            {
                "t0_candidates": "estimated",
                "t0_sims": 300,
                "turn_traces": "estimated",
                "turn_trace_sims": 96,
                "exact_terminal_available": False,
            },
        ),
        "hand": hand_data.get("hand"),
        "bb_dealt": hand_data.get("bb_dealt", []),
        "btn_dealt": hand_data.get("btn_dealt", []),
        "decision": decision,
        "access": "paid" if access["paid"] else "free_sample",
    }


@app.post("/api/create-checkout-session")
async def create_checkout_session(req: dict, decoded_token: dict = Depends(verify_token)):
    if stripe is None:
        raise HTTPException(status_code=503, detail="Stripe is not installed")

    uid = decoded_token.get("uid")
    email = decoded_token.get("email") or ""
    doc_ref, user_data = get_user_doc(uid, email)

    requested_plan = req.get("plan")
    if requested_plan not in PAID_PLANS:
        return {"error": "INVALID_PLAN", "message": "Plan must be starter or premium."}

    price_id = STRIPE_PRICE_IDS.get(requested_plan)
    if not price_id:
        return {
            "error": "PRICE_NOT_CONFIGURED",
            "message": f"Stripe price id for {requested_plan} is not configured.",
        }

    success_url = req.get("success_url") or "http://localhost:8081/?success=true"
    cancel_url = req.get("cancel_url") or "http://localhost:8081/?canceled=true"
    customer_id = user_data.get("stripe_customer_id")

    session_kwargs = {
        "payment_method_types": ["card"],
        "line_items": [{"price": price_id, "quantity": 1}],
        "mode": "subscription",
        "success_url": success_url,
        "cancel_url": cancel_url,
        "client_reference_id": uid,
        "metadata": {"uid": uid, "plan": requested_plan},
        "subscription_data": {"metadata": {"uid": uid, "plan": requested_plan}},
        "allow_promotion_codes": True,
    }
    if customer_id:
        session_kwargs["customer"] = customer_id
    elif email:
        session_kwargs["customer_email"] = email

    try:
        session = stripe.checkout.Session.create(**session_kwargs)
    except Exception as exc:
        return {"error": "CHECKOUT_FAILED", "message": str(exc)}

    doc_ref.update({"last_checkout_plan": requested_plan, "updated_at": now_iso()})
    return {"url": session.url}


@app.post("/api/create-billing-portal-session")
async def create_billing_portal_session(req: dict, decoded_token: dict = Depends(verify_token)):
    if stripe is None:
        raise HTTPException(status_code=503, detail="Stripe is not installed")

    uid = decoded_token.get("uid")
    email = decoded_token.get("email") or ""
    _, user_data = get_user_doc(uid, email)
    customer_id = user_data.get("stripe_customer_id")
    if not customer_id:
        return {
            "error": "NO_STRIPE_CUSTOMER",
            "message": "No Stripe customer is attached to this account yet.",
        }

    return_url = req.get("return_url") or "http://localhost:8081/"
    try:
        session = stripe.billing_portal.Session.create(customer=customer_id, return_url=return_url)
    except Exception as exc:
        return {"error": "BILLING_PORTAL_FAILED", "message": str(exc)}
    return {"url": session.url}


def subscription_price_id(subscription: Dict[str, Any]) -> Optional[str]:
    items = subscription.get("items", {}).get("data", [])
    if not items:
        return None
    price = items[0].get("price") or {}
    return price.get("id")


def update_user_from_subscription(subscription: Dict[str, Any]) -> None:
    metadata = subscription.get("metadata") or {}
    uid = metadata.get("uid")
    customer_id = subscription.get("customer")
    subscription_id = subscription.get("id")
    price_id = subscription_price_id(subscription)
    plan = metadata.get("plan") or price_id_to_plan(price_id) or "free"
    status = subscription.get("status", "inactive")

    doc_ref = None
    if uid:
        doc_ref = db.collection("users").document(uid)
    if doc_ref is None:
        doc_ref, _ = find_user_doc_by_field("stripe_subscription_id", subscription_id)
    if doc_ref is None:
        doc_ref, _ = find_user_doc_by_field("stripe_customer_id", customer_id)
    if doc_ref is None:
        return

    if status not in ACTIVE_SUBSCRIPTION_STATUSES:
        plan = "free"

    doc_ref.update(
        {
            "plan": plan if plan in PAID_PLANS else "free",
            "subscription_status": status,
            "stripe_customer_id": customer_id,
            "stripe_subscription_id": subscription_id,
            "price_id": price_id,
            "updated_at": now_iso(),
        }
    )


@app.post("/api/webhook")
async def stripe_webhook(request: Request):
    if stripe is None:
        raise HTTPException(status_code=503, detail="Stripe is not installed")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event["type"]
    obj = event["data"]["object"]

    if event_type == "checkout.session.completed":
        uid = obj.get("client_reference_id") or (obj.get("metadata") or {}).get("uid")
        requested_plan = (obj.get("metadata") or {}).get("plan") or "premium"
        customer_id = obj.get("customer")
        subscription_id = obj.get("subscription")
        if uid:
            doc_ref = db.collection("users").document(uid)
            doc_ref.update(
                {
                    "plan": requested_plan if requested_plan in PAID_PLANS else "premium",
                    "subscription_status": "active",
                    "stripe_customer_id": customer_id,
                    "stripe_subscription_id": subscription_id,
                    "updated_at": now_iso(),
                }
            )
    elif event_type in {"customer.subscription.created", "customer.subscription.updated"}:
        update_user_from_subscription(obj)
    elif event_type == "customer.subscription.deleted":
        update_user_from_subscription({**obj, "status": obj.get("status") or "canceled"})
    elif event_type == "invoice.payment_failed":
        subscription_id = obj.get("subscription")
        customer_id = obj.get("customer")
        doc_ref, _ = find_user_doc_by_field("stripe_subscription_id", subscription_id)
        if doc_ref is None:
            doc_ref, _ = find_user_doc_by_field("stripe_customer_id", customer_id)
        if doc_ref is not None:
            doc_ref.update({"subscription_status": "past_due", "plan": "free", "updated_at": now_iso()})

    return {"status": "success"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)
