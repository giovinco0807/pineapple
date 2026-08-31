import json

from ai.tutor.convert_rust_t3_exact_teacher import convert_record


def test_convert_record_merges_rust_result_with_source_context() -> None:
    source = {
        "turn": 3,
        "board": {"top": ["Ad"], "middle": ["2h", "3h", "4h", "5h"], "bottom": ["6h", "7h", "8h", "9h"]},
        "opponent_board": {"top": ["Kd"], "middle": [], "bottom": []},
        "dealt": ["Ah", "Qs", "2c"],
        "known_discards": ["3c"],
        "exclude": ["4c"],
        "is_btn": False,
    }
    result = {
        "record_index": 7,
        "turn": 3,
        "legal_actions": 1,
        "elapsed_ms": 12.5,
        "candidates": [
            {
                "action": {"placements": [["Ah", "top"], ["Qs", "middle"]], "discard": "2c"},
                "board": {"top": ["Ad", "Ah"], "middle": ["2h", "3h", "4h", "5h", "Qs"], "bottom": ["6h"]},
                "metrics": {
                    "score": 9.5,
                    "ev": 9.5,
                    "raw_score": 1.5,
                    "royalty": 1.5,
                    "bust_rate": 0.2,
                    "fl_rate": 0.3,
                    "fl_type_rates": {"aa": 0.3},
                    "samples": 42,
                },
            }
        ],
    }

    converted = convert_record(json.loads(json.dumps(result)), source)

    assert converted["board"] == source["board"]
    assert converted["opponent_board"] == source["opponent_board"]
    assert converted["dealt"] == source["dealt"]
    assert converted["position"] == "bb"
    assert converted["elapsed_s"] == 0.0125
    assert converted["exact_scope"] == "t3_self_board_all_t4_draws_best_t4"
    assert converted["total_exact_samples"] == 42
    assert converted["candidates"][0]["placements"] == [["Ah", "top"], ["Qs", "middle"]]
    assert converted["candidates"][0]["bust_prob"] == 0.2
    assert converted["candidates"][0]["fl_type_rates"]["aa"] == 0.3


def test_convert_record_marks_t4_opponent_response_scope() -> None:
    source = {
        "turn": 4,
        "board": {"top": ["2c", "3c"], "middle": [], "bottom": []},
        "opponent_board": {"top": ["Qh", "Qs"], "middle": [], "bottom": []},
        "dealt": ["4c", "Kc", "5d"],
        "is_btn": False,
    }
    result = {
        "record_index": 0,
        "turn": 4,
        "legal_actions": 1,
        "candidates": [
            {
                "action": {"placements": [["4c", "top"], ["Kc", "bottom"]], "discard": "5d"},
                "board": {},
                "metrics": {
                    "score": -20.5,
                    "raw_score": -20.5,
                    "royalty": 0.0,
                    "bust_rate": 0.0,
                    "fl_rate": 0.0,
                    "fl_type_rates": {},
                    "samples": 4,
                    "source": "exact_hu_response",
                    "opponent_response": True,
                },
            }
        ],
    }

    converted = convert_record(result, source)

    assert converted["exact_scope"] == "t4_all_opponent_draws_best_response_given_exclude"
    assert converted["hu_exact"] is False
    assert converted["total_exact_samples"] == 4
