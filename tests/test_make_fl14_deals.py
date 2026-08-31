from ai.tutor.make_fl14_deals import card_names, deal


def test_first_development_deal_matches_the_recorded_lap3_stream():
    assert card_names(deal(0xC0FFEE21, 0)) == [
        "4d", "2d", "Ac", "Qd", "3d", "Jc", "8d",
        "Ah", "8h", "Qs", "Ks", "8c", "8s", "Kc",
    ]


def test_offset_changes_the_deal_without_duplicate_cards():
    first = card_names(deal(0xC0FFEE21, 0))
    later = card_names(deal(0xC0FFEE21, 5000))
    assert first != later
    assert len(later) == len(set(later)) == 14
