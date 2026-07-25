from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as subject


RUN_NAME = "regular-hu-m31-c02-f100wv2-20260722-003"
SALT = "0123456789abcdef0123456789abcdef"
PACKAGE_SHA = "2" * 64
IMAGE_DIGEST = "sha256:" + "3" * 64
PROJECT = "ofc-project-123"
ZONE = "asia-northeast1-b"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def wave_plan() -> dict:
    return subject.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt=SALT,
        package_sha256=PACKAGE_SHA,
        image_digest=IMAGE_DIGEST,
    )


def _baseline(plan: dict) -> dict:
    return subject.build_observed_transition(
        plan,
        project_id=PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-22T00:00:00Z",
        previous_transition_digest=None,
        attempt_history=subject.empty_attempt_history(plan),
    )


def _history_by_job(transition: dict) -> dict[str, dict]:
    return {row["job_id"]: row for row in deepcopy(transition["attempt_history"])}


def _root_hash(hand: int) -> str:
    # Roots are common between candidate/reference by contract.
    return _sha(f"root-hand-{hand:03d}")


def _root_digest(work: list[int]) -> str:
    return subject.canonical_sha256(
        [{"hand_index": hand, "sha256": _root_hash(hand)} for hand in work]
    )


def _advance(
    plan: dict,
    previous: dict,
    outcomes: dict[str, str],
    *,
    observed_at: str,
) -> dict:
    """Append one terminal attempt for each listed job."""

    histories = _history_by_job(previous)
    done = deepcopy(previous["done_objects"])
    acceptances = deepcopy(previous["acceptance_records"])
    jobs = {row["job_id"]: row for row in plan["full100_plan"]["jobs"]}
    role_by_job = {row["job_id"]: row["source_role"] for row in histories.values()}
    instance_by_job: dict[str, dict[str, str]] = {}
    for wave in plan["waves"]:
        for pair in wave["candidate_reference_pairs"]:
            instance_by_job[pair["candidate_job_id"]] = pair[
                "candidate_attempt_instance_ids"
            ]
            instance_by_job[pair["reference_job_id"]] = pair[
                "reference_attempt_instance_ids"
            ]
    ordinal = {job: index for index, job in enumerate(plan["coverage"]["job_ids"])}
    for job, outcome in outcomes.items():
        attempts = histories[job]["attempts"]
        attempt_id = subject.ATTEMPT_IDS[len(attempts)]
        attempts.append(
            {
                "attempt_id": attempt_id,
                "instance_id": instance_by_job[job][attempt_id],
                "launch_receipt_sha256": _sha(f"receipt-{job}-{attempt_id}"),
                "terminal_status": outcome,
            }
        )
        if outcome == "accepted":
            role = role_by_job[job]
            root_digest = _root_digest(list(jobs[job]["work_hand_indices"]))
            generation = 1000 + ordinal[job] * 10 + int(attempt_id[-1])
            done_sha = _sha(f"done-{job}-{attempt_id}")
            prefix = plan["artifact_contract"]["attempt_path_template"].format(
                job_id=job, attempt_id=attempt_id
            )
            done.append(
                {
                    "job_id": job,
                    "source_role": role,
                    "attempt_id": attempt_id,
                    "path": f"{prefix}/DONE.json",
                    "generation": generation,
                    "bytes": 200 + ordinal[job],
                    "sha256": done_sha,
                    "done_identity_sha256": subject.expected_done_identity_sha256(
                        plan,
                        job_id=job,
                        attempt_id=attempt_id,
                        root_digest=root_digest,
                    ),
                    "package_sha256": plan["runtime_binding"]["package_sha256"],
                    "image_digest": plan["runtime_binding"]["image_digest"],
                    "binary_sha256": plan["runtime_binding"][
                        "binary_sha256_by_role"
                    ][role],
                    "allocation_digest": plan["runtime_binding"][
                        "allocation_digest"
                    ],
                    "root_digest": root_digest,
                }
            )
            acceptances.append(
                {
                    "job_id": job,
                    "source_role": role,
                    "attempt_id": attempt_id,
                    "path": plan["artifact_contract"][
                        "job_acceptance_path_template"
                    ].format(job_id=job),
                    "generation": generation + 1,
                    "bytes": 100 + ordinal[job],
                    "sha256": _sha(f"accepted-{job}-{attempt_id}"),
                    "done_generation": generation,
                    "done_sha256": done_sha,
                    "create_only": True,
                }
            )
    return subject.build_observed_transition(
        plan,
        project_id=PROJECT,
        zone=ZONE,
        observed_at_utc=observed_at,
        previous_transition_digest=previous["transition_digest"],
        attempt_history=[histories[job] for job in plan["coverage"]["job_ids"]],
        done_objects=done,
        acceptance_records=acceptances,
    )


def _ledger(plan: dict, transitions: list[dict], consumed: list[str]) -> dict:
    return subject.build_attempt_ledger(
        plan,
        transitions=transitions,
        consumed_transition_digests=consumed,
    )


def _complete_ledger(plan: dict) -> dict:
    transitions = [_baseline(plan)]
    consumed: list[str] = []
    for wave_index, stamp in enumerate(("01", "02", "03")):
        consumed.append(transitions[-1]["transition_digest"])
        outcomes = {job: "accepted" for job in plan["waves"][wave_index]["job_ids"]}
        transitions.append(
            _advance(
                plan,
                transitions[-1],
                outcomes,
                observed_at=f"2026-07-22T00:00:{stamp}Z",
            )
        )
    return _ledger(plan, transitions, consumed)


def _observed_objects(expected: dict) -> list[dict]:
    result: list[dict] = []
    generation = 10000
    for record in expected["records"]:
        common = {
            "job_id": record["job_id"],
            "source_role": record["source_role"],
            "attempt_id": record["accepted_attempt_id"],
        }
        for hand, path in zip(
            record["work_hand_indices"], record["root_paths"], strict=True
        ):
            generation += 1
            result.append(
                {
                    "path": path,
                    **common,
                    "object_kind": "root",
                    "hand_index": hand,
                    "generation": generation,
                    "bytes": 500 + hand,
                    "sha256": _root_hash(hand),
                    "done_identity_sha256": None,
                    "package_sha256": None,
                    "image_digest": None,
                    "binary_sha256": None,
                    "allocation_digest": None,
                    "root_digest": None,
                }
            )
        for hand, path in zip(
            record["work_hand_indices"], record["source_hand_paths"], strict=True
        ):
            generation += 1
            result.append(
                {
                    "path": path,
                    **common,
                    "object_kind": "source_hand",
                    "hand_index": hand,
                    "generation": generation,
                    "bytes": 700 + hand,
                    "sha256": _sha(f"source-{record['job_id']}-{hand}"),
                    "done_identity_sha256": None,
                    "package_sha256": None,
                    "image_digest": None,
                    "binary_sha256": None,
                    "allocation_digest": None,
                    "root_digest": None,
                }
            )
        result.append(
            {
                "path": record["done_path"],
                **common,
                "object_kind": "done",
                "hand_index": None,
                "generation": record["done_generation"],
                "bytes": record["done_bytes"],
                "sha256": record["done_sha256"],
                "done_identity_sha256": record["done_identity_sha256"],
                "package_sha256": record["package_sha256"],
                "image_digest": record["image_digest"],
                "binary_sha256": record["binary_sha256"],
                "allocation_digest": record["allocation_digest"],
                "root_digest": record["root_digest"],
            }
        )
        result.append(
            {
                "path": record["acceptance_path"],
                **common,
                "object_kind": "acceptance",
                "hand_index": None,
                "generation": record["acceptance_generation"],
                "bytes": record["acceptance_bytes"],
                "sha256": record["acceptance_sha256"],
                "done_identity_sha256": None,
                "package_sha256": None,
                "image_digest": None,
                "binary_sha256": None,
                "allocation_digest": None,
                "root_digest": None,
            }
        )
    return result


def _rehash_inventory(value: dict) -> None:
    value["object_count"] = len(value["objects"])
    value["inventory_sha256"] = subject.canonical_sha256(
        {key: item for key, item in value.items() if key != "inventory_sha256"}
    )


def test_frozen_science_is_8_8_4_and_identity_salt_is_hash_only(
    wave_plan: dict,
) -> None:
    assert subject.validate_wave_plan(wave_plan) == wave_plan
    # The established normal-scope identity and complete plan remain bit-exact.
    assert wave_plan["execution_identity_sha256"] == (
        "18885a46e2a5125be972d9fb5b250209e344fa02f6597068ab6ddb5d8284d6f8"
    )
    assert wave_plan["schedule_sha256"] == (
        "8b6dea07202290ae41293ecd2c3e76307a5298848ad4eee3133d9565ce97c8c5"
    )
    assert wave_plan["scope"] == subject.FULL100_EXECUTION_SCOPE
    assert wave_plan["wave_vm_counts"] == [8, 8, 4]
    assert [wave["shard_indices"] for wave in wave_plan["waves"]] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]
    assert wave_plan["coverage"]["paired_hand_indices"] == list(range(100))
    assert wave_plan["coverage"]["source_root_count"] == 200
    assert all("salt" not in key and "nonce" not in key for key in wave_plan)
    assert SALT not in subject.canonical_bytes(wave_plan).decode("ascii")
    assert wave_plan["cloud_launcher_ready"] is False
    assert wave_plan["cloud_launch_authorized"] is False
    assert wave_plan["authorization_blockers"] == list(
        subject.AUTHORIZATION_BLOCKERS
    )

    different_salt = subject.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt="fedcba9876543210fedcba9876543210",
        package_sha256=PACKAGE_SHA,
        image_digest=IMAGE_DIGEST,
    )
    assert different_salt["execution_identity_sha256"] != wave_plan[
        "execution_identity_sha256"
    ]
    assert different_salt["artifact_contract"]["prefix"] != wave_plan[
        "artifact_contract"
    ]["prefix"]
    first_a = wave_plan["waves"][0]["candidate_reference_pairs"][0][
        "candidate_attempt_instance_ids"
    ]["a00"]
    first_b = different_salt["waves"][0]["candidate_reference_pairs"][0][
        "candidate_attempt_instance_ids"
    ]["a00"]
    assert first_a != first_b
    with pytest.raises(ValueError, match="nonzero"):
        subject.build_wave_plan(
            run_name=RUN_NAME,
            identity_salt="0" * 32,
            package_sha256=PACKAGE_SHA,
            image_digest=IMAGE_DIGEST,
        )


def test_startup_canary_is_scope_bound_single_a00_and_never_scientific(
    wave_plan: dict,
) -> None:
    canary = subject.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt=SALT,
        package_sha256=PACKAGE_SHA,
        image_digest=IMAGE_DIGEST,
        execution_scope=subject.STARTUP_CANARY_SCOPE,
    )
    assert subject.validate_startup_canary_plan(canary) == canary
    assert canary["execution_identity_sha256"] != wave_plan[
        "execution_identity_sha256"
    ]
    assert canary["artifact_contract"]["prefix"] != wave_plan[
        "artifact_contract"
    ]["prefix"]
    assert canary["artifact_contract"]["prefix"].endswith(
        "/full100-wave-v2/startup-canary"
    )

    # The canary narrows only the launch selection, never the frozen full100
    # scientific schedule or its 8+8+4 inventory.
    assert canary["wave_vm_counts"] == [8, 8, 4]
    assert [wave["job_ids"] for wave in canary["waves"]] == [
        wave["job_ids"] for wave in wave_plan["waves"]
    ]
    assert canary["coverage"] == wave_plan["coverage"]
    assert canary["full100_plan"] == wave_plan["full100_plan"]
    assert canary["resume_contract"]["a01_authorized"] is False
    assert canary["resume_contract"]["other_jobs_authorized"] is False
    assert canary["resume_contract"]["retry_authorized"] is False
    assert canary["artifact_contract"]["performance_evaluation_eligible"] is False
    assert canary["artifact_contract"]["scientific_merge_eligible"] is False
    assert canary["artifact_contract"]["training_eligible"] is False

    baseline = _baseline(canary)
    ledger = _ledger(canary, [baseline], [])
    resume = subject.build_resume_plan(canary, attempt_ledger=ledger)
    assert resume["selected_attempts"] == [
        {
            "job_id": "candidate-shard-00",
            "source_role": "candidate",
            "attempt_id": "a00",
            "instance_id": canary["waves"][0]["candidate_reference_pairs"][0][
                "candidate_attempt_instance_ids"
            ]["a00"],
            "artifact_prefix": canary["artifact_contract"][
                "attempt_path_template"
            ].format(job_id="candidate-shard-00", attempt_id="a00"),
        }
    ]

    failed = _advance(
        canary,
        baseline,
        {"candidate-shard-00": "failed"},
        observed_at="2026-07-22T00:00:01Z",
    )
    failed_ledger = _ledger(
        canary, [baseline, failed], [baseline["transition_digest"]]
    )
    with pytest.raises(PermissionError, match="single-attempt"):
        subject.build_resume_plan(canary, attempt_ledger=failed_ledger)

    other = _advance(
        canary,
        baseline,
        {"reference-shard-00": "failed"},
        observed_at="2026-07-22T00:00:01Z",
    )
    other_ledger = _ledger(
        canary, [baseline, other], [baseline["transition_digest"]]
    )
    with pytest.raises(PermissionError, match="single-attempt"):
        subject.build_resume_plan(canary, attempt_ledger=other_ledger)

    with pytest.raises(PermissionError, match="ineligible"):
        subject.expected_artifact_inventory(canary, attempt_ledger=ledger)
    with pytest.raises(ValueError, match="startup canary"):
        subject.validate_startup_canary_plan(wave_plan)


def test_attempt_ledger_selects_a00_then_only_failed_job_a01_and_never_a02(
    wave_plan: dict,
) -> None:
    baseline = _baseline(wave_plan)
    initial = _ledger(wave_plan, [baseline], [])
    resume = subject.build_resume_plan(wave_plan, attempt_ledger=initial)
    assert [row["attempt_id"] for row in resume["selected_attempts"]] == [
        "a00"
    ] * 8
    assert all("/attempts/a00" in row["artifact_prefix"] for row in resume["selected_attempts"])
    assert resume["cloud_launch_authorized"] is False

    wave0 = wave_plan["waves"][0]["job_ids"]
    failed_job = wave0[0]
    first = _advance(
        wave_plan,
        baseline,
        {job: ("failed" if job == failed_job else "accepted") for job in wave0},
        observed_at="2026-07-22T00:00:01Z",
    )
    after_first = _ledger(
        wave_plan, [baseline, first], [baseline["transition_digest"]]
    )
    retry = subject.build_resume_plan(wave_plan, attempt_ledger=after_first)
    assert retry["selected_attempts"] == [
        {
            "job_id": failed_job,
            "source_role": "candidate",
            "attempt_id": "a01",
            "instance_id": wave_plan["waves"][0]["candidate_reference_pairs"][0][
                "candidate_attempt_instance_ids"
            ]["a01"],
            "artifact_prefix": wave_plan["artifact_contract"][
                "attempt_path_template"
            ].format(job_id=failed_job, attempt_id="a01"),
        }
    ]

    second_failure = _advance(
        wave_plan,
        first,
        {failed_job: "failed"},
        observed_at="2026-07-22T00:00:02Z",
    )
    exhausted = _ledger(
        wave_plan,
        [baseline, first, second_failure],
        [baseline["transition_digest"], first["transition_digest"]],
    )
    with pytest.raises(ValueError, match="exhausted"):
        subject.build_resume_plan(wave_plan, attempt_ledger=exhausted)


def test_same_transition_reuse_and_unclaimed_history_fail_closed(
    wave_plan: dict,
) -> None:
    baseline = _baseline(wave_plan)
    ledger = _ledger(wave_plan, [baseline], [])
    consumed = subject.mark_latest_transition_consumed(wave_plan, ledger)
    with pytest.raises(ValueError, match="already consumed"):
        subject.build_resume_plan(wave_plan, attempt_ledger=consumed)

    first_job = wave_plan["waves"][0]["job_ids"][0]
    advanced = _advance(
        wave_plan,
        baseline,
        {first_job: "failed"},
        observed_at="2026-07-22T00:00:01Z",
    )
    with pytest.raises(ValueError, match="chain or observation scope"):
        _ledger(wave_plan, [baseline, advanced], [])


def test_observed_vm_readback_not_caller_active_list_controls_quiescence(
    wave_plan: dict,
) -> None:
    baseline = _baseline(wave_plan)
    pair = wave_plan["waves"][0]["candidate_reference_pairs"][0]
    active = subject.build_observed_transition(
        wave_plan,
        project_id=PROJECT,
        zone=ZONE,
        observed_at_utc="2026-07-22T00:00:01Z",
        previous_transition_digest=baseline["transition_digest"],
        attempt_history=baseline["attempt_history"],
        owned_vms=[
            {
                "instance_id": pair["candidate_attempt_instance_ids"]["a00"],
                "job_id": pair["candidate_job_id"],
                "source_role": "candidate",
                "attempt_id": "a00",
                "status": "RUNNING",
            }
        ],
    )
    ledger = _ledger(
        wave_plan, [baseline, active], [baseline["transition_digest"]]
    )
    with pytest.raises(ValueError, match="quiescence"):
        subject.build_resume_plan(wave_plan, attempt_ledger=ledger)


def test_both_done_multiple_acceptance_and_transition_scope_tamper_fail(
    wave_plan: dict,
) -> None:
    baseline = _baseline(wave_plan)
    wave0 = wave_plan["waves"][0]["job_ids"]
    accepted = _advance(
        wave_plan,
        baseline,
        {job: "accepted" for job in wave0},
        observed_at="2026-07-22T00:00:01Z",
    )
    duplicate_done = [*accepted["done_objects"], deepcopy(accepted["done_objects"][0])]
    duplicate_done[-1]["attempt_id"] = "a01"
    duplicate_done[-1]["path"] = duplicate_done[-1]["path"].replace(
        "/a00/", "/a01/"
    )
    with pytest.raises(ValueError, match="both attempts have DONE"):
        subject.build_observed_transition(
            wave_plan,
            project_id=PROJECT,
            zone=ZONE,
            observed_at_utc="2026-07-22T00:00:02Z",
            previous_transition_digest=accepted["transition_digest"],
            attempt_history=accepted["attempt_history"],
            done_objects=duplicate_done,
            acceptance_records=accepted["acceptance_records"],
        )
    duplicate_acceptance = [
        *accepted["acceptance_records"],
        deepcopy(accepted["acceptance_records"][0]),
    ]
    with pytest.raises(ValueError, match="multiple accepted"):
        subject.build_observed_transition(
            wave_plan,
            project_id=PROJECT,
            zone=ZONE,
            observed_at_utc="2026-07-22T00:00:02Z",
            previous_transition_digest=accepted["transition_digest"],
            attempt_history=accepted["attempt_history"],
            done_objects=accepted["done_objects"],
            acceptance_records=duplicate_acceptance,
        )
    tampered = deepcopy(accepted)
    tampered["project_id"] = "other-project-123"
    with pytest.raises(ValueError):
        subject.validate_observed_transition(wave_plan, tampered)


def test_expected_and_observed_inventory_are_exact_and_runtime_bound(
    wave_plan: dict,
) -> None:
    ledger = _complete_ledger(wave_plan)
    expected = subject.expected_artifact_inventory(
        wave_plan, attempt_ledger=ledger
    )
    assert expected["job_count"] == 20
    assert expected["source_root_count"] == 200
    assert expected["source_hand_count"] == 200
    assert expected["object_count"] == 440
    assert all(
        f"/attempts/{row['accepted_attempt_id']}" in row["attempt_prefix"]
        and row["acceptance_create_only"] is True
        for row in expected["records"]
    )

    observed = subject.build_observed_artifact_inventory(
        wave_plan,
        attempt_ledger=ledger,
        observed_at_utc="2026-07-22T00:01:00Z",
        objects=_observed_objects(expected),
    )
    assert (
        subject.validate_observed_artifact_inventory(wave_plan, ledger, observed)
        == observed
    )

    missing = deepcopy(observed)
    missing["objects"].pop()
    _rehash_inventory(missing)
    with pytest.raises(ValueError, match="missing or extra"):
        subject.validate_observed_artifact_inventory(wave_plan, ledger, missing)

    extra = deepcopy(observed)
    extra["objects"].append(deepcopy(extra["objects"][0]))
    extra["objects"][-1]["path"] += ".extra"
    _rehash_inventory(extra)
    with pytest.raises(ValueError, match="missing or extra|extra path"):
        subject.validate_observed_artifact_inventory(wave_plan, ledger, extra)

    duplicate = deepcopy(observed)
    duplicate["objects"][1] = deepcopy(duplicate["objects"][0])
    _rehash_inventory(duplicate)
    with pytest.raises(ValueError, match="duplicate path"):
        subject.validate_observed_artifact_inventory(wave_plan, ledger, duplicate)

    wrong_generation = deepcopy(observed)
    done_index = next(
        index
        for index, row in enumerate(wrong_generation["objects"])
        if row["object_kind"] == "done"
    )
    wrong_generation["objects"][done_index]["generation"] += 1
    _rehash_inventory(wrong_generation)
    with pytest.raises(ValueError, match="metadata changed"):
        subject.validate_observed_artifact_inventory(
            wave_plan, ledger, wrong_generation
        )

    wrong_binary = deepcopy(observed)
    wrong_binary["objects"][done_index]["binary_sha256"] = "f" * 64
    _rehash_inventory(wrong_binary)
    with pytest.raises(ValueError, match="DONE binding"):
        subject.validate_observed_artifact_inventory(wave_plan, ledger, wrong_binary)


def test_wave_plan_and_ledgers_are_create_only_local_artifacts(
    tmp_path: Path, wave_plan: dict
) -> None:
    target = tmp_path / "wave_plan.json"
    subject.write_once(target, wave_plan)
    assert target.read_bytes() == subject.canonical_bytes(wave_plan)
    with pytest.raises(FileExistsError, match="immutable"):
        subject.write_once(target, wave_plan)
