from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import t0_tutor_app as tutor
from ai.engine.encoding import ALL_CARDS, Board
from ai.tutor.exact_late import default_rust_t3_exact_solver_path, exact_t4_draw_distribution


def auth_override(uid="user_1", email="user@example.com"):
    return {"uid": uid, "email": email}


def make_client(uid="user_1", email="user@example.com"):
    tutor.app.dependency_overrides[tutor.verify_token] = lambda: auth_override(uid, email)
    return TestClient(tutor.app)


def user_doc(uid="user_1", email="user@example.com", **updates):
    data = tutor.default_user_data(email)
    data.update(updates)
    tutor.db.collection("users").document(uid).set(data)
    return data


@pytest.fixture(autouse=True)
def reset_tutor_state(monkeypatch):
    tutor.db = tutor.MemoryFirestore()
    tutor.TRAINING_PRESETS = {
        "puzzle_1": {
            "cards": ["Ah", "Kh", "Qh", "Jh", "Th"],
            "top_placements": [
                {
                    "placement": {"top": ["Ah"], "middle": ["Kh", "Qh"], "bottom": ["Jh", "Th"]},
                    "ev": 1.5,
                    "bust_prob": 0.1,
                    "fl_prob": 0.3,
                    "royalty_ev": 0.2,
                }
            ],
            "optimal_ev": 1.5,
        }
    }
    tutor.STRIPE_PRICE_IDS["starter"] = "price_starter"
    tutor.STRIPE_PRICE_IDS["premium"] = "price_premium"
    tutor.ON_DEMAND_EVAL_CACHE.clear()
    tutor.app.dependency_overrides.clear()
    yield
    tutor.app.dependency_overrides.clear()


def test_me_requires_auth():
    client = TestClient(tutor.app)
    response = client.get("/api/me")
    assert response.status_code == 401


def test_training_puzzle_and_evaluation_increment_usage():
    client = make_client()

    puzzle_response = client.get("/api/training_puzzle")
    assert puzzle_response.status_code == 200
    assert puzzle_response.json()["puzzle_id"] == "puzzle_1"

    placement = tutor.TRAINING_PRESETS["puzzle_1"]["top_placements"][0]["placement"]
    eval_response = client.post(
        "/api/evaluate_training",
        json={"puzzle_id": "puzzle_1", "placement": placement},
    )
    assert eval_response.status_code == 200
    body = eval_response.json()
    assert body["is_optimal"] is True

    _, stored = tutor.get_user_doc("user_1", "user@example.com")
    assert stored["daily_training_usage"] == 1


def test_training_daily_limit_is_enforced():
    user_doc(daily_training_usage=tutor.FREE_DAILY_TRAINING_LIMIT)
    client = make_client()

    response = client.get("/api/training_puzzle")

    assert response.status_code == 200
    assert response.json()["error"] == "DAILY_LIMIT_REACHED"


def test_custom_t0_is_locked_for_free_users():
    client = make_client()

    response = client.post(
        "/api/evaluate_t0",
        json={
            "cards": ["Ah", "Kh", "Qh", "Jh", "Th"],
            "placement": {"top": ["Ah"], "middle": ["Kh", "Qh"], "bottom": ["Jh", "Th"]},
        },
    )

    assert response.status_code == 200
    assert response.json()["error"] == "CUSTOM_LOCKED"


def test_route10_free_preview_and_locked_details():
    client = make_client()

    summary = client.get("/api/reviews/route10")
    assert summary.status_code == 200
    body = summary.json()
    assert body["access"] == "free"
    assert len(body["hands"]) == 10
    assert body["sample"]["seat"] == "bb"

    sample = client.get("/api/reviews/route10/hands/1/decisions/bb")
    assert sample.status_code == 200
    assert sample.json()["access"] == "free_sample"

    locked = client.get("/api/reviews/route10/hands/2/decisions/bb")
    assert locked.status_code == 403


def test_route10_paid_users_get_full_details():
    user_doc(
        plan="starter",
        subscription_status="active",
        stripe_subscription_id="sub_123",
        stripe_customer_id="cus_123",
    )
    client = make_client()

    response = client.get("/api/reviews/route10/hands/2/decisions/btn")

    assert response.status_code == 200
    decision = response.json()["decision"]
    assert decision["seat"] == "btn"
    assert len(decision["top_candidates"]) > 0
    assert len(decision["turn_traces"]) == 4


def test_checkout_session_uses_requested_plan_price(monkeypatch):
    calls = {}

    def fake_checkout_create(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(url="https://stripe.test/checkout")

    monkeypatch.setattr(tutor.stripe.checkout.Session, "create", fake_checkout_create)
    client = make_client()

    response = client.post(
        "/api/create-checkout-session",
        json={"plan": "starter", "success_url": "https://app.test/s", "cancel_url": "https://app.test/c"},
    )

    assert response.status_code == 200
    assert response.json()["url"] == "https://stripe.test/checkout"
    assert calls["line_items"][0]["price"] == "price_starter"
    assert calls["metadata"] == {"uid": "user_1", "plan": "starter"}

    _, stored = tutor.get_user_doc("user_1", "user@example.com")
    assert stored["last_checkout_plan"] == "starter"


def test_billing_portal_requires_stripe_customer(monkeypatch):
    user_doc(stripe_customer_id="cus_123")
    calls = {}

    def fake_portal_create(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(url="https://stripe.test/portal")

    monkeypatch.setattr(tutor.stripe.billing_portal.Session, "create", fake_portal_create)
    client = make_client()

    response = client.post("/api/create-billing-portal-session", json={"return_url": "https://app.test"})

    assert response.status_code == 200
    assert response.json()["url"] == "https://stripe.test/portal"
    assert calls == {"customer": "cus_123", "return_url": "https://app.test"}


def test_subscription_webhook_updates_and_cancels_plan(monkeypatch):
    user_doc()
    events = [
        {
            "type": "customer.subscription.updated",
            "data": {
                "object": {
                    "id": "sub_123",
                    "customer": "cus_123",
                    "status": "active",
                    "metadata": {"uid": "user_1"},
                    "items": {"data": [{"price": {"id": "price_premium"}}]},
                }
            },
        },
        {
            "type": "customer.subscription.deleted",
            "data": {
                "object": {
                    "id": "sub_123",
                    "customer": "cus_123",
                    "status": "canceled",
                    "metadata": {"uid": "user_1"},
                    "items": {"data": [{"price": {"id": "price_premium"}}]},
                }
            },
        },
    ]

    def fake_construct_event(payload, sig_header, secret):
        return events.pop(0)

    monkeypatch.setattr(tutor.stripe.Webhook, "construct_event", fake_construct_event)
    client = TestClient(tutor.app)

    updated = client.post("/api/webhook", content=b"{}", headers={"stripe-signature": "sig"})
    assert updated.status_code == 200
    _, stored = tutor.get_user_doc("user_1", "user@example.com")
    assert stored["plan"] == "premium"
    assert stored["subscription_status"] == "active"

    deleted = client.post("/api/webhook", content=b"{}", headers={"stripe-signature": "sig"})
    assert deleted.status_code == 200
    _, stored = tutor.get_user_doc("user_1", "user@example.com")
    assert stored["plan"] == "free"
    assert stored["subscription_status"] == "canceled"


def test_exact_t4_distribution_replaces_sampled_fl_rate_for_late_state():
    board = Board(
        top=["Ad"],
        middle=["2h", "2s", "4c", "4s", "5s"],
        bottom=["3c", "3s", "6c", "6h", "Jc"],
    )
    opponent = Board(top=["X1"], middle=["8s", "Jh"], bottom=["Qc", "Qh"])

    result = exact_t4_draw_distribution(board, opponent_board=opponent, exclude=["Ts", "Qs", "7h", "X2"])

    assert result["source"] == "exact"
    assert result["remaining_deck_size"] == 34
    assert result["samples"] == 5984
    assert result["fl_rate"] == pytest.approx(1654 / 5984)
    assert result["fl_type_rates"]["aa"] == pytest.approx(1488 / 5984)
    assert result["fl_type_rates"]["kk"] == pytest.approx(166 / 5984)


def test_evaluate_position_t4_exact_uses_cache_and_does_not_double_charge():
    user_doc(
        plan="starter",
        subscription_status="active",
        stripe_subscription_id="sub_123",
        stripe_customer_id="cus_123",
    )
    client = make_client()
    payload = {
        "turn": 4,
        "precision": "exact",
        "board": {
            "top": ["Ad"],
            "middle": ["2h", "2s", "4c", "4s", "5s"],
            "bottom": ["3c", "3s", "6c", "6h", "Jc"],
        },
        "dealt": ["Ah", "Kd", "9s"],
        "known_discards": ["Ts", "Qs", "7h", "X2"],
    }

    first = client.post("/api/evaluate_position", json=payload)
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["source"] == "exact"
    assert first_body["exact_scope"] == "t4_terminal_self_board"
    assert first_body["hu_exact"] is False
    assert first_body["cached"] is False
    assert first_body["legal_actions"] > 0

    second = client.post("/api/evaluate_position", json=payload)
    assert second.status_code == 200
    assert second.json()["cached"] is True

    _, stored = tutor.get_user_doc("user_1", "user@example.com")
    assert stored["daily_custom_usage"] == 1


def test_evaluate_position_t3_exact_uses_rust_when_available():
    solver = default_rust_t3_exact_solver_path()
    if not Path(solver).exists():
        pytest.skip(f"Rust T3 exact solver is not built: {solver}")

    result = tutor.evaluate_position_payload(
        {
            "turn": 3,
            "precision": "exact",
            "top_n": 27,
            "board": {
                "top": ["Qd"],
                "middle": ["2c", "Kc", "3s", "Ah"],
                "bottom": ["8h", "Jc", "8s", "Js"],
            },
            "opponent_board": {
                "top": ["Qc", "X2", "Ad"],
                "middle": ["2d", "7d", "6d"],
                "bottom": ["8c", "9h", "8d"],
            },
            "dealt": ["Td", "3h", "6c"],
        },
        plan="starter",
    )

    assert result["source"] == "rust_exact"
    assert result["precision"] == "exact"
    assert result["exact_scope"] == "t3_self_board_all_t4_draws_best_t4"
    assert result["hu_exact"] is False
    assert result["legal_actions"] == 21
    assert result["chosen_action"] == result["best"]["action"]


def test_evaluate_position_t4_bb_uses_exact_btn_response_when_available():
    solver = default_rust_t3_exact_solver_path()
    if not Path(solver).exists():
        pytest.skip(f"Rust late exact solver is not built: {solver}")

    board = {
        "top": ["2h", "3d"],
        "middle": ["4h", "4d", "5c", "6s"],
        "bottom": ["8h", "8d", "9c", "Ts", "Js"],
    }
    opponent = {
        "top": ["2c", "3c"],
        "middle": ["5h", "5d", "6c", "7s"],
        "bottom": ["9h", "9d", "Tc", "Jd", "Qs"],
    }
    dealt = ["Qh", "Kd", "Ac"]
    reply_draw = ["X1", "Ah", "Ad"]
    live = {
        *board["top"],
        *board["middle"],
        *board["bottom"],
        *opponent["top"],
        *opponent["middle"],
        *opponent["bottom"],
        *dealt,
        *reply_draw,
    }
    exclude = [card for card in ALL_CARDS if card not in live]

    result = tutor.evaluate_position_payload(
        {
            "turn": 4,
            "precision": "exact",
            "top_n": 50,
            "board": board,
            "opponent_board": opponent,
            "dealt": dealt,
            "exclude": exclude,
        },
        plan="starter",
    )

    assert result["source"] == "rust_exact"
    assert result["precision"] == "exact"
    assert result["exact_scope"] == "t4_all_opponent_draws_best_response_given_exclude"
    assert result["hu_exact"] is False
    assert result["best"]["metrics"]["source"] == "exact_hu_response"
    assert result["best"]["metrics"]["samples"] == 1
