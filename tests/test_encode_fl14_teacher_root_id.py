from ai.tutor.encode_fl14_teacher import split_of, stable_root_id


def test_stable_root_id_uses_global_record_id_not_local_root():
    first = {"id": "3237998113", "root": 7}
    resumed = {"id": "3237999999", "root": 7}

    assert stable_root_id(first) == 3237998113
    assert stable_root_id(resumed) == 3237999999
    assert stable_root_id(first) != stable_root_id(resumed)


def test_stable_root_id_handles_non_numeric_ids_deterministically():
    record = {"id": "deal-A/seat-1", "root": 0}

    value = stable_root_id(record)
    assert value == stable_root_id(record)
    assert 0 <= value < 2**63
    assert split_of(value, "t2") in {"fit", "dev", "test"}
