from ofc_regular.action_space import generate_actions
from ofc_regular.policy import RegularAiPolicy
from ofc_regular.state import Board


class OpeningModel:
    def predict_sample(self, sample):
        return [float(len(sample["actions"]) - index) for index in range(len(sample["actions"]))]

    def choose_action_index(self, sample):
        return 0


class HuCandidateModel:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def predict_sample(self, sample):
        self.calls += 1
        if self.fail:
            raise RuntimeError("prediction failed")
        values = [0.0 for _action in sample["actions"]]
        values[1] = 2.0
        return values


class SafeSelector:
    def __init__(self, score):
        self.score = score

    def predict_proba(self, features):
        return [[1.0 - self.score, self.score] for _ in range(len(features))]


def _safe_selector_payload(score):
    return {
        "model_kind": "hu_turn0_safe_override_selector_sklearn",
        "feature_mode": "meta_only",
        "estimator": SafeSelector(score),
    }


def _choose(policy, *, dead_cards=()):
    board = Board.from_rows()
    dealt = ("As", "Kh", "Qd", "Jc", "Ts")
    opponent = Board.from_rows()
    return policy.choose_action(
        board,
        dealt,
        dead_cards=dead_cards,
        opponent_board=opponent,
        hand_id="hand-1",
        game_id="game-1",
        decision_seed=123,
        street="T0",
    )


def test_hu_turn0_selective_override_fires_inside_opening_topk():
    logs = []
    policy = RegularAiPolicy(
        opening_model=OpeningModel(),
        hu_turn0_model=HuCandidateModel(),
        hu_turn0_candidate_topk=3,
        hu_turn0_min_margin=1.0,
        hu_turn0_decision_log=logs,
        seat="first",
    )
    actions = generate_actions(Board.from_rows(), ("As", "Kh", "Qd", "Jc", "Ts"))

    chosen = _choose(policy, dead_cards=("2c",))

    assert chosen == actions[1]
    assert len(logs) == 1
    assert logs[0]["override_fired"] is True
    assert logs[0]["fallback_action_index"] == 0
    assert logs[0]["candidate_action_index"] == 1
    assert logs[0]["candidate_pool_count"] == 3
    assert logs[0]["dead_cards"] == ["2c"]
    assert logs[0]["hu_turn0_predicted_margin"] == 2.0


def test_hu_turn0_below_margin_keeps_opening_baseline():
    logs = []
    policy = RegularAiPolicy(
        opening_model=OpeningModel(),
        hu_turn0_model=HuCandidateModel(),
        hu_turn0_candidate_topk=3,
        hu_turn0_min_margin=3.0,
        hu_turn0_decision_log=logs,
        seat="first",
    )
    actions = generate_actions(Board.from_rows(), ("As", "Kh", "Qd", "Jc", "Ts"))

    assert _choose(policy) == actions[0]
    assert logs[0]["override_fired"] is False
    assert logs[0]["no_override_reason"] == "below_hu_turn0_margin"


def test_hu_turn0_prediction_failure_falls_back_to_opening():
    logs = []
    policy = RegularAiPolicy(
        opening_model=OpeningModel(),
        hu_turn0_model=HuCandidateModel(fail=True),
        hu_turn0_candidate_topk=3,
        hu_turn0_min_margin=0.0,
        hu_turn0_decision_log=logs,
        seat="first",
    )
    actions = generate_actions(Board.from_rows(), ("As", "Kh", "Qd", "Jc", "Ts"))

    assert _choose(policy) == actions[0]
    assert logs[0]["override_fired"] is False
    assert logs[0]["no_override_reason"] == "prediction_failed"


def test_hu_turn0_seat_gate_skips_candidate_model():
    model = HuCandidateModel()
    logs = []
    policy = RegularAiPolicy(
        opening_model=OpeningModel(),
        hu_turn0_model=model,
        hu_turn0_candidate_topk=3,
        hu_turn0_allowed_seats=("second",),
        hu_turn0_decision_log=logs,
        seat="first",
    )
    actions = generate_actions(Board.from_rows(), ("As", "Kh", "Qd", "Jc", "Ts"))

    assert _choose(policy) == actions[0]
    assert model.calls == 0
    assert logs[0]["no_override_reason"] == "seat_disabled"


def test_hu_turn0_seat_specific_threshold_is_used():
    logs = []
    policy = RegularAiPolicy(
        opening_model=OpeningModel(),
        hu_turn0_model=HuCandidateModel(),
        hu_turn0_candidate_topk=3,
        hu_turn0_min_margin=0.0,
        hu_turn0_min_margin_by_seat={"first": 3.0, "second": 1.0},
        hu_turn0_decision_log=logs,
        seat="first",
    )

    _choose(policy)

    assert logs[0]["hu_turn0_min_margin"] == 3.0
    assert logs[0]["override_fired"] is False


def test_hu_turn0_safe_selector_can_fire_after_margin_gate():
    logs = []
    policy = RegularAiPolicy(
        opening_model=OpeningModel(),
        hu_turn0_model=HuCandidateModel(),
        hu_turn0_candidate_topk=3,
        hu_turn0_min_margin=1.0,
        hu_turn0_safe_selector_enabled=True,
        hu_turn0_safe_selector_model=_safe_selector_payload(0.9),
        hu_turn0_safe_selector_threshold=0.8,
        hu_turn0_decision_log=logs,
        decision_context={
            "runtime_profile": "stage19_p0",
            "runtime_status": "p0_fixed",
            "selective_override_only": True,
            "full_replacement_enabled": False,
            "fallback_policy": "stage18_p1",
            "t1_continuation": "stage18_p1",
            "t2_continuation": "stage9f_p2",
            "t3_continuation": "stage7_m5_r10",
        },
        seat="first",
    )

    _choose(policy)

    assert logs[0]["override_fired"] is True
    assert logs[0]["hu_turn0_safe_selector_score"] == 0.9
    assert logs[0]["hu_turn0_safe_selector_threshold"] == 0.8
    assert logs[0]["runtime_profile"] == "stage19_p0"
    assert logs[0]["runtime_status"] == "p0_fixed"
    assert logs[0]["fallback_policy"] == "stage18_p1"
    assert logs[0]["t1_continuation"] == "stage18_p1"
    assert logs[0]["t2_continuation"] == "stage9f_p2"
    assert logs[0]["t3_continuation"] == "stage7_m5_r10"


def test_hu_turn0_safe_selector_below_threshold_falls_back():
    logs = []
    policy = RegularAiPolicy(
        opening_model=OpeningModel(),
        hu_turn0_model=HuCandidateModel(),
        hu_turn0_candidate_topk=3,
        hu_turn0_min_margin=1.0,
        hu_turn0_safe_selector_enabled=True,
        hu_turn0_safe_selector_model=_safe_selector_payload(0.7),
        hu_turn0_safe_selector_threshold=0.8,
        hu_turn0_decision_log=logs,
        seat="first",
    )

    _choose(policy)

    assert logs[0]["override_fired"] is False
    assert logs[0]["no_override_reason"] == "below_hu_turn0_safe_selector"


def test_hu_turn0_missing_safe_selector_falls_back():
    logs = []
    policy = RegularAiPolicy(
        opening_model=OpeningModel(),
        hu_turn0_model=HuCandidateModel(),
        hu_turn0_safe_selector_enabled=True,
        hu_turn0_safe_selector_model=None,
        hu_turn0_decision_log=logs,
        seat="first",
    )

    _choose(policy)

    assert logs[0]["override_fired"] is False
    assert logs[0]["no_override_reason"] == "safe_selector_unavailable"


def test_hu_turn0_nan_safe_selector_falls_back():
    logs = []
    policy = RegularAiPolicy(
        opening_model=OpeningModel(),
        hu_turn0_model=HuCandidateModel(),
        hu_turn0_safe_selector_enabled=True,
        hu_turn0_safe_selector_model=_safe_selector_payload(float("nan")),
        hu_turn0_decision_log=logs,
        seat="first",
    )

    _choose(policy)

    assert logs[0]["override_fired"] is False
    assert logs[0]["no_override_reason"] == "safe_selector_failed"
