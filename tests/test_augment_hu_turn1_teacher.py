import pytest

from ofc_regular.augment_hu_turn1_teacher import augment_records


def _sample(key: str, *, label_source: str = "base", schema: str = "source") -> dict:
    return {
        "schema": schema,
        "state_key": key,
        "label_source": label_source,
        "actions": [{"score": 1.0}],
    }


def test_augment_appends_addition_rows_with_new_label_source():
    rows, summary = augment_records(
        [_sample("a", label_source="base_refinement")],
        [_sample("b", label_source="old_label", schema="runtime_relabel")],
        addition_label_source="stage9f_p2_runtime_relabel",
        duplicate_policy="error",
    )

    assert [row["state_key"] for row in rows] == ["a", "b"]
    assert rows[1]["schema"] == "hu_turn1_stage1_augmented_teacher_v1"
    assert rows[1]["source_schema"] == "runtime_relabel"
    assert rows[1]["label_source"] == "stage9f_p2_runtime_relabel"
    assert summary["appended_records"] == 1
    assert summary["label_source_counts"] == {
        "base_refinement": 1,
        "stage9f_p2_runtime_relabel": 1,
    }


def test_augment_rejects_duplicate_base_key_by_default():
    with pytest.raises(ValueError, match="already exists"):
        augment_records(
            [_sample("a")],
            [_sample("a")],
            addition_label_source="runtime",
            duplicate_policy="error",
        )


def test_augment_can_skip_duplicate_addition_rows():
    rows, summary = augment_records(
        [_sample("a")],
        [_sample("a"), _sample("b")],
        addition_label_source="runtime",
        duplicate_policy="skip",
    )

    assert [row["state_key"] for row in rows] == ["a", "b"]
    assert summary["skipped_duplicate_records"] == 1
    assert summary["appended_records"] == 1


def test_augment_can_replace_duplicate_base_rows():
    rows, summary = augment_records(
        [_sample("a", label_source="base")],
        [_sample("a", label_source="old_runtime")],
        addition_label_source="runtime",
        duplicate_policy="replace",
    )

    assert len(rows) == 1
    assert rows[0]["state_key"] == "a"
    assert rows[0]["label_source"] == "runtime"
    assert summary["replaced_duplicate_records"] == 1
