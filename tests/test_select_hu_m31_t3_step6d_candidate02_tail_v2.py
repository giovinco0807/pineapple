from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ofc_regular import select_hu_m31_t3_step6d_candidate02_tail_v2 as subject


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "candidate02_development"
    / "tail_reselection_v2"
    / "roots"
)


@pytest.fixture(scope="module")
def frozen_roots() -> list[dict]:
    return subject.load_frozen_roots(ROOT_DIR)


def test_frozen_all100_roots_and_topology_match_preregistered_digests(
    frozen_roots: list[dict],
) -> None:
    root_digest = subject.canonical_sha256(
        [subject.canonical_sha256(root) for root in frozen_roots]
    )
    topology = subject.topology_rows(frozen_roots)
    assert root_digest == subject.ALL100_ROOT_SHA256
    assert subject.canonical_sha256(topology) == subject.TOPOLOGY_SHA256
    assert len(topology) == 100
    assert set(topology[0]) == {
        "hand_index",
        "profile",
        "first_fingerprint",
        "hero_legal_actions",
        "opponent_response_legal_actions",
    }


def test_selector_uses_topology_and_excludes_prior_runtime_exposure(
    frozen_roots: list[dict],
) -> None:
    manifest = subject.build_selection_manifest(frozen_roots)
    assert manifest["heavy_hand_indices"] == [0, 5, 12, 16, 17, 23, 41, 43]
    assert manifest["random_hand_indices"] == [4, 14]
    assert manifest["tail_hand_indices"] == [0, 4, 5, 12, 14, 16, 17, 23, 41, 43]
    assert not set(manifest["tail_hand_indices"]) & set(
        subject.PRIOR_RUNTIME_EXPOSED_HAND_INDICES
    )
    assert manifest["heavy_hand_indices_by_profile"] == {
        "stage19_p0": [0, 5],
        "stage9f_p2": [16, 41],
        "stage7_m5_r10": [12, 17],
        "stage3_baseline": [23, 43],
    }
    assert subject.canonical_sha256(manifest) == subject.SELECTION_MANIFEST_SHA256
    assert all(
        manifest[field] is False
        for field in (
            "runtime_results_used",
            "timing_used",
            "memory_used",
            "teacher_values_used",
            "training_eligible",
            "quality_evidence",
            "promotion_evidence",
            "current_profile_changed",
        )
    )


def test_each_heavy_pick_is_first_eligible_per_profile(
    frozen_roots: list[dict],
) -> None:
    rows = subject.topology_rows(frozen_roots)
    manifest = subject.build_selection_manifest(frozen_roots)
    exposed = set(subject.PRIOR_RUNTIME_EXPOSED_HAND_INDICES)
    for profile, selected in manifest["heavy_hand_indices_by_profile"].items():
        eligible = [
            row["hand_index"]
            for row in rows
            if row["profile"] == profile
            and row["hand_index"] not in exposed
            and row["hero_legal_actions"] == 21
            and row["opponent_response_legal_actions"] == 21
        ]
        assert selected == eligible[:2]


def test_manifest_and_topology_tamper_fail_closed(
    frozen_roots: list[dict],
) -> None:
    manifest = subject.build_selection_manifest(frozen_roots)
    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["random_hand_indices"] = [4, 19]
    with pytest.raises(ValueError, match="selection manifest changed"):
        subject.validate_selection_manifest(tampered_manifest)

    tampered_roots = copy.deepcopy(frozen_roots)
    tampered_roots[0]["observations"][0]["observation_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="first observation changed"):
        subject.topology_rows(tampered_roots)


def test_loader_rejects_noncanonical_or_incomplete_root_sets(tmp_path: Path) -> None:
    (tmp_path / "hand_000.json").write_text(
        json.dumps({"schema": subject.ROOT_SCHEMA}, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly roots 000..099"):
        subject.load_frozen_roots(tmp_path)


def test_cli_writes_canonical_manifest_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "selection_manifest.json"
    assert (
        subject.main(
            [
                "--root-dir",
                str(ROOT_DIR),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert output.read_bytes() == subject.canonical_bytes(manifest)
    assert subject.canonical_sha256(manifest) == subject.SELECTION_MANIFEST_SHA256
    assert json.loads(capsys.readouterr().out) == manifest
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        subject.main(
            [
                "--root-dir",
                str(ROOT_DIR),
                "--output",
                str(output),
            ]
        )
