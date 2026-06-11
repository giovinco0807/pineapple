from ofc_regular import (
    ALL_CARDS,
    check_fl_entry,
    check_fl_stay,
    create_deck,
    get_bottom_royalty,
    get_middle_royalty,
    get_top_royalty,
    score_board,
)


def test_regular_deck_has_no_jokers():
    deck = create_deck(shuffle=False)
    assert len(deck) == 52
    assert len(set(deck)) == 52
    assert "X1" not in deck
    assert "X2" not in deck
    assert "JK" not in deck
    assert tuple(deck) == ALL_CARDS


def test_fl_entry_conditions_all_deal_14_cards():
    assert check_fl_entry(["Qh", "Qs", "4d"]).entry_type == "qq"
    assert check_fl_entry(["Qh", "Qs", "4d"]).card_count == 14
    assert check_fl_entry(["Kh", "Ks", "4d"]).entry_type == "kk"
    assert check_fl_entry(["Kh", "Ks", "4d"]).card_count == 14
    assert check_fl_entry(["Ah", "As", "4d"]).entry_type == "aa"
    assert check_fl_entry(["Ah", "As", "4d"]).card_count == 14
    assert check_fl_entry(["2h", "2s", "2d"]).entry_type == "trips"
    assert check_fl_entry(["2h", "2s", "2d"]).card_count == 14
    assert not check_fl_entry(["Jh", "Js", "4d"]).qualifies


def test_fl_stay_always_deals_14_cards():
    top_trips = check_fl_stay(["7h", "7s", "7d"], ["Ah", "Kh", "Qh", "Jh", "Th"])
    assert top_trips.qualifies
    assert top_trips.card_count == 14

    bottom_quads = check_fl_stay(["Ah", "Ks", "Qd"], ["9h", "9s", "9d", "9c", "2h"])
    assert bottom_quads.qualifies
    assert bottom_quads.card_count == 14

    no_stay = check_fl_stay(["Ah", "As", "Qd"], ["2h", "3h", "4h", "5h", "7h"])
    assert not no_stay.qualifies


def test_royalties_match_existing_tables():
    assert get_top_royalty(["6h", "6s", "4d"]) == 1
    assert get_top_royalty(["Ah", "As", "4d"]) == 9
    assert get_top_royalty(["2h", "2s", "2d"]) == 10
    assert get_top_royalty(["Ah", "As", "Ad"]) == 22

    assert get_middle_royalty(["Ah", "Ad", "Ac", "Kh", "Kd"]) == 12
    assert get_middle_royalty(["Ah", "Ad", "Ac", "As", "Kh"]) == 20
    assert get_middle_royalty(["Ah", "Kh", "Qh", "Jh", "Th"]) == 50

    assert get_bottom_royalty(["Ah", "Ad", "Ac", "Kh", "Kd"]) == 6
    assert get_bottom_royalty(["Ah", "Ad", "Ac", "As", "Kh"]) == 10
    assert get_bottom_royalty(["Ah", "Kh", "Qh", "Jh", "Th"]) == 25


def test_score_board_reports_regular_fl_entry():
    score = score_board(
        top=["Qc", "Qs", "4d"],
        middle=["2h", "3h", "4h", "5h", "7h"],
        bottom=["Ah", "Kh", "Qh", "Jh", "Th"],
    )
    assert not score.busted
    assert score.fl_entry.qualifies
    assert score.fl_entry.entry_type == "qq"
    assert score.fl_entry.card_count == 14
    assert score.total_royalty == 7 + 8 + 25
