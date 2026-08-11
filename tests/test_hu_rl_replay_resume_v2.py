from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ofc_regular.hu_rl_replay_resume_v2 as replay_v2
from ofc_regular.action_key import ActionKey
from ofc_regular.hu_rl_native import (
    MAX_LEGAL_ACTIONS,
    PACKED_ACTION_BYTES,
    PACKED_OBSERVATION_RECORD_BYTES,
    PACKED_STEP_RECORD_BYTES,
    native_available,
)
from ofc_regular.hu_rl_replay_resume_v2 import (
    ACCEPTANCE_PAIR_TARGET,
    PROVENANCE_SCHEMA,
    _SOURCE_PATHS,
    HuRlReplayResumeV2Error,
    _decode_shard,
    prepare_replay_resume_run,
    run_or_resume_replay,
    validate_replay_resume_run,
)


_ACTIONS = (
    ActionKey(0, 0, 0, 0),
    ActionKey(1, 0, 0, 0),
)


def _provenance() -> dict[str, Any]:
    return {
        "schema": PROVENANCE_SCHEMA,
        "package_name": "test-native",
        "package_version": "0.0",
        "wheel_filename": "test.whl",
        "wheel_sha256": "1" * 64,
        "native_extension_filename": "test.pyd",
        "native_extension_sha256": "2" * 64,
        "source_sha256": {path: "3" * 64 for path in _SOURCE_PATHS},
    }


class _FakeLegal:
    def __init__(self, lanes: int, decision: int) -> None:
        self._lanes = lanes
        self._decision = decision
        self.action_set_digests = b"".join(
            hashlib.sha256(f"set:{decision}:{lane}".encode()).digest()
            for lane in range(lanes)
        )
        self.action_order_digests = b"".join(
            hashlib.sha256(f"order:{decision}:{lane}".encode()).digest()
            for lane in range(lanes)
        )

    def action_count(self, lane: int) -> int:
        if not 0 <= lane < self._lanes:
            raise IndexError(lane)
        return len(_ACTIONS)

    def select(self, indices: tuple[int, ...]) -> bytes:
        if len(indices) != self._lanes:
            raise ValueError("lane count")
        output = bytearray()
        for index in indices:
            output.extend(struct.pack("<4Q", *_ACTIONS[index].masks))
        return bytes(output)


class _FakeStep:
    def __init__(self, *, payload: bytes, rewards: list[tuple[float, float]], done: bool) -> None:
        self.payload = payload
        self._rewards = rewards
        self._done = done

    def lane(self, lane: int) -> Any:
        return SimpleNamespace(rewards=self._rewards[lane], done=self._done)


class _FakeEnv:
    def __init__(self, deck_digests: list[bytes], *, chunk_width: int, thread_count: int) -> None:
        del chunk_width, thread_count
        self._deck_digests = deck_digests
        self._decision = 0

    @classmethod
    def from_paired_seed_range(
        cls,
        *,
        seed_base: int,
        global_pair_start: int,
        pair_count: int,
        seed_stride: int,
        chunk_width: int,
        thread_count: int,
    ) -> "_FakeEnv":
        digests: list[bytes] = []
        for local_pair in range(pair_count):
            pair_seed = seed_base + (global_pair_start + local_pair) * seed_stride
            for leg in ("ab", "ba"):
                digests.append(
                    hashlib.sha256(f"native-seed-test:{pair_seed}:{leg}".encode()).digest()
                )
        return cls(digests, chunk_width=chunk_width, thread_count=thread_count)

    @property
    def lane_count(self) -> int:
        return len(self._deck_digests)

    @property
    def all_done(self) -> bool:
        return self._decision == 10

    def actor_decision_batch_packed(self) -> Any:
        payload = bytearray()
        for lane, digest in enumerate(self._deck_digests):
            seed = bytes([self._decision, lane & 255]) + digest
            payload.extend((seed * ((PACKED_OBSERVATION_RECORD_BYTES // len(seed)) + 1))[:PACKED_OBSERVATION_RECORD_BYTES])
        return SimpleNamespace(
            observations=SimpleNamespace(payload=bytes(payload)),
            legal_actions=_FakeLegal(len(self._deck_digests), self._decision),
        )

    def step_batch_packed(self, selected: bytes) -> _FakeStep:
        assert len(selected) == len(self._deck_digests) * PACKED_ACTION_BYTES
        done = self._decision == 9
        rewards = []
        payload = bytearray()
        for lane, digest in enumerate(self._deck_digests):
            score = float((digest[0] % 11) - 5) if done else 0.0
            rewards.append((score, -score))
            prefix = bytes([self._decision, lane & 255, 1 if done else 0]) + digest
            payload.extend((prefix * ((PACKED_STEP_RECORD_BYTES // len(prefix)) + 1))[:PACKED_STEP_RECORD_BYTES])
        self._decision += 1
        return _FakeStep(payload=bytes(payload), rewards=rewards, done=done)


def _prepare(path: Path, *, paired_hands: int = 3, shard_pairs: int = 1) -> dict[str, Any]:
    return prepare_replay_resume_run(
        path,
        run_id="unit-replay-v2",
        paired_hand_start=7,
        paired_hand_count=paired_hands,
        shard_pair_count=shard_pairs,
        seed_base=12345,
        seed_stride=17,
        policy_a_id="policy-a@001",
        policy_b_id="policy-b@002",
        chunk_width=8,
        thread_count=2,
        _provenance_provider=_provenance,
    )


def test_clean_and_interrupted_resume_are_byte_identical(tmp_path: Path) -> None:
    interrupted = tmp_path / "interrupted"
    clean = tmp_path / "clean"
    _prepare(interrupted)
    _prepare(clean)

    paused = run_or_resume_replay(
        interrupted,
        seed_base=12345,
        max_new_shards=1,
        _env_factory=_FakeEnv,
        _provenance_provider=_provenance,
    )
    assert paused == {
        "status": "paused",
        "completed_shards": 1,
        "total_shards": 3,
        "new_shards": 1,
        "aggregate_sha256": None,
    }
    prefix = validate_replay_resume_run(
        interrupted,
        seed_base=12345,
        require_complete=False,
        _env_factory=_FakeEnv,
        _provenance_provider=_provenance,
    )
    assert prefix["status"] == "validated_prefix"
    assert prefix["reconstructed_records"] == 20
    assert prefix["replay_reconstruction_complete"] is False

    resumed = run_or_resume_replay(
        interrupted,
        seed_base=12345,
        _env_factory=_FakeEnv,
        _provenance_provider=_provenance,
    )
    uninterrupted = run_or_resume_replay(
        clean,
        seed_base=12345,
        _env_factory=_FakeEnv,
        _provenance_provider=_provenance,
    )
    assert resumed["status"] == uninterrupted["status"] == "complete"
    assert resumed["aggregate_sha256"] == uninterrupted["aggregate_sha256"]
    assert (interrupted / "aggregate.json").read_bytes() == (clean / "aggregate.json").read_bytes()
    aggregate = json.loads((interrupted / "aggregate.json").read_text(encoding="ascii"))
    assert aggregate["acceptance"] == {
        "paired_hand_target": 100_000,
        "target_scale_evaluated": False,
        "target_scale_pass": None,
        "parquet_gate_evaluated": False,
    }
    if replay_v2.os.name == "posix":
        assert aggregate["durability"] == {
            "mode": "posix_parent_directory_fsync",
            "parent_directory_fsync_enabled": True,
            "hard_power_loss_primitives_enabled": True,
            "fault_injection_evaluated": False,
            "production_spot_eligible": True,
        }
    else:
        assert aggregate["durability"] == {
            "mode": "local_process_interruption_only",
            "parent_directory_fsync_enabled": False,
            "hard_power_loss_primitives_enabled": False,
            "fault_injection_evaluated": False,
            "production_spot_eligible": False,
        }
    for directory in ("shards", "checkpoints"):
        left = sorted((interrupted / directory).iterdir())
        right = sorted((clean / directory).iterdir())
        assert [path.name for path in left] == [path.name for path in right]
        assert [path.read_bytes() for path in left] == [path.read_bytes() for path in right]

    validation = validate_replay_resume_run(
        interrupted,
        seed_base=12345,
        _env_factory=_FakeEnv,
        _provenance_provider=_provenance,
    )
    assert validation["checkpoint_reconstruction_complete"] is True
    assert validation["replay_reconstruction_complete"] is True
    assert validation["return_reconstruction_complete"] is True
    assert validation["reconstructed_records"] == validation["expected_records"] == 60


@pytest.mark.skipif(not native_available(), reason="optional PyO3 extension not built")
def test_real_native_seeded_replay_resume_is_clean_byte_identical(tmp_path: Path) -> None:
    interrupted = tmp_path / "native-interrupted"
    clean = tmp_path / "native-clean"
    _prepare(interrupted, paired_hands=2, shard_pairs=1)
    _prepare(clean, paired_hands=2, shard_pairs=1)

    paused = run_or_resume_replay(
        interrupted,
        seed_base=12345,
        max_new_shards=1,
        _provenance_provider=_provenance,
    )
    assert paused["status"] == "paused"
    resumed = run_or_resume_replay(
        interrupted,
        seed_base=12345,
        _provenance_provider=_provenance,
    )
    uninterrupted = run_or_resume_replay(
        clean,
        seed_base=12345,
        _provenance_provider=_provenance,
    )
    assert resumed["status"] == uninterrupted["status"] == "complete"
    assert resumed["aggregate_sha256"] == uninterrupted["aggregate_sha256"]
    for relative in ("aggregate.json", "shards/shard-000000.json.gz", "shards/shard-000001.json.gz", "checkpoints/checkpoint-000001.json", "checkpoints/checkpoint-000002.json"):
        assert (interrupted / relative).read_bytes() == (clean / relative).read_bytes()
    validation = validate_replay_resume_run(
        interrupted,
        seed_base=12345,
        _provenance_provider=_provenance,
    )
    assert validation["reconstructed_records"] == validation["expected_records"] == 40
    assert validation["checkpoint_reconstruction_complete"] is True
    assert validation["replay_reconstruction_complete"] is True
    assert validation["return_reconstruction_complete"] is True


def test_trajectory_is_actor_safe_and_keeps_actionkey_rng_provenance(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _prepare(output, paired_hands=1)
    run_or_resume_replay(
        output,
        seed_base=12345,
        _env_factory=_FakeEnv,
        _provenance_provider=_provenance,
    )
    shard = _decode_shard((output / "shards" / "shard-000000.json.gz").read_bytes())
    assert shard["hidden_truth_exposed"] is False
    assert shard["opponent_private_discard_exposed"] is False
    assert shard["raw_seed_exposed"] is False
    assert len(shard["records"]) == 20
    for row in shard["records"]:
        assert row["actor_private_observation_persisted"] is False
        assert row["chosen_action_key_persisted"] is False
        assert row["action_key_reconstruction"].endswith("fail_closed")
        assert row["rng"]["namespace"].endswith("behavior_action_v2")
        assert row["rng"]["coupling"] == "paired_seat_swap_common_tape"
        assert len(row["rng"]["draw_sha256"]) == 64
        assert math.isfinite(float.fromhex(row["behavior_logprob_hex"]))
        assert "seed_base" not in row
    records_json = json.dumps(shard["records"], sort_keys=True).lower()
    for forbidden in (
        "opponent_private_discard", "deck_tail", "world_state", "explicit_deck",
        "actor_observation_b64", "chosen_action_key\"", "rak1:",
    ):
        assert forbidden not in records_json

    # Each population role receives the exact same random tape at the same
    # street in AB and BA.  Seat/leg and legal action count are not hash input.
    coupled: dict[tuple[int, str, str], list[tuple[int, str]]] = {}
    for row in shard["records"]:
        key = (row["pair_index"], row["population_role"], row["street"])
        coupled.setdefault(key, []).append(
            (row["rng"]["counter"], row["rng"]["draw_sha256"])
        )
    assert len(coupled) == 10
    assert all(len(values) == 2 and values[0] == values[1] for values in coupled.values())


def test_tampered_shard_fails_before_resume(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _prepare(output, paired_hands=2, shard_pairs=1)
    run_or_resume_replay(
        output,
        seed_base=12345,
        max_new_shards=1,
        _env_factory=_FakeEnv,
        _provenance_provider=_provenance,
    )
    path = output / "shards" / "shard-000000.json.gz"
    value = _decode_shard(path.read_bytes())
    value["records"][0]["returns_hex"] = [0.0.hex(), 0.0.hex()]
    # A noncanonical/corrupt replacement must not be accepted even if gzip is valid.
    import gzip

    path.write_bytes(gzip.compress((json.dumps(value) + "\n").encode(), mtime=0))
    with pytest.raises(HuRlReplayResumeV2Error):
        run_or_resume_replay(
            output,
            seed_base=12345,
            _env_factory=_FakeEnv,
            _provenance_provider=_provenance,
        )


def test_resume_reconstructs_one_missing_trailing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    _prepare(output, paired_hands=2, shard_pairs=1)
    run_or_resume_replay(
        output,
        seed_base=12345,
        max_new_shards=1,
        _env_factory=_FakeEnv,
        _provenance_provider=_provenance,
    )
    checkpoint = output / "checkpoints" / "checkpoint-000001.json"
    checkpoint.unlink()
    synced_directories: list[Path] = []
    original_directory_fsync = replay_v2._fsync_parent_directory

    def recording_directory_fsync(directory: Path) -> None:
        synced_directories.append(Path(directory))
        original_directory_fsync(directory)

    monkeypatch.setattr(
        replay_v2, "_fsync_parent_directory", recording_directory_fsync
    )
    result = run_or_resume_replay(
        output,
        seed_base=12345,
        max_new_shards=0,
        _env_factory=_FakeEnv,
        _provenance_provider=_provenance,
    )
    assert result["status"] == "paused"
    assert result["completed_shards"] == 1
    assert checkpoint.is_file()
    assert synced_directories == [
        output,
        output / "shards",
        output / "checkpoints",
        output / "shards",
        output / "checkpoints",
    ]


def test_create_only_fsyncs_parent_after_final_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, Path]] = []
    real_link = replay_v2.os.link

    def recording_link(source: Path, destination: Path) -> None:
        real_link(source, destination)
        events.append(("link", Path(destination)))

    def recording_directory_fsync(directory: Path) -> None:
        events.append(("directory_fsync", Path(directory)))

    monkeypatch.setattr(replay_v2.os, "link", recording_link)
    monkeypatch.setattr(
        replay_v2, "_fsync_parent_directory", recording_directory_fsync
    )

    target = tmp_path / "shards" / "shard-000000.json.gz"
    replay_v2._write_bytes_atomic_create_only(target, b"durable")

    assert target.read_bytes() == b"durable"
    assert events == [("link", target), ("directory_fsync", target.parent)]


def test_cross_platform_durability_is_readable_but_not_mutable(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    _prepare(output, paired_hands=1)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if replay_v2.os.name == "posix":
        other = {
            "mode": "local_process_interruption_only",
            "parent_directory_fsync_enabled": False,
            "hard_power_loss_primitives_enabled": False,
            "fault_injection_evaluated": False,
            "production_spot_eligible": False,
        }
    else:
        other = {
            "mode": "posix_parent_directory_fsync",
            "parent_directory_fsync_enabled": True,
            "hard_power_loss_primitives_enabled": True,
            "fault_injection_evaluated": False,
            "production_spot_eligible": True,
        }
    manifest["durability"] = other
    manifest["manifest_sha256"] = None
    manifest["manifest_sha256"] = replay_v2._self_digest(
        manifest, "manifest_sha256"
    )

    replay_v2.validate_manifest(manifest, expected_provenance=_provenance())
    manifest_path.write_text(
        replay_v2._canonical_json(manifest) + "\n", encoding="ascii", newline=""
    )
    with pytest.raises(
        HuRlReplayResumeV2Error,
        match="mutation runtime does not match",
    ):
        run_or_resume_replay(
            output,
            seed_base=12345,
            _env_factory=_FakeEnv,
            _provenance_provider=_provenance,
        )


def test_retry_resyncs_checkpoint_directory_after_post_link_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    _prepare(output, paired_hands=2, shard_pairs=1)
    original_directory_fsync = replay_v2._fsync_parent_directory
    checkpoint_sync_count = 0

    def fail_after_checkpoint_link(directory: Path) -> None:
        nonlocal checkpoint_sync_count
        if directory == output / "checkpoints":
            checkpoint_sync_count += 1
            if checkpoint_sync_count == 2:
                raise HuRlReplayResumeV2Error("injected checkpoint fsync failure")
        original_directory_fsync(directory)

    monkeypatch.setattr(
        replay_v2, "_fsync_parent_directory", fail_after_checkpoint_link
    )
    with pytest.raises(
        HuRlReplayResumeV2Error, match="injected checkpoint fsync failure"
    ):
        run_or_resume_replay(
            output,
            seed_base=12345,
            max_new_shards=1,
            _env_factory=_FakeEnv,
            _provenance_provider=_provenance,
        )
    assert (output / "checkpoints" / "checkpoint-000001.json").is_file()

    retry_syncs: list[Path] = []

    def record_retry_sync(directory: Path) -> None:
        retry_syncs.append(Path(directory))
        original_directory_fsync(directory)

    monkeypatch.setattr(replay_v2, "_fsync_parent_directory", record_retry_sync)
    result = run_or_resume_replay(
        output,
        seed_base=12345,
        max_new_shards=0,
        _env_factory=_FakeEnv,
        _provenance_provider=_provenance,
    )

    assert result["status"] == "paused"
    assert retry_syncs[:3] == [
        output,
        output / "shards",
        output / "checkpoints",
    ]


def test_wrong_seed_and_unknown_manifest_field_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _prepare(output, paired_hands=1)
    with pytest.raises(HuRlReplayResumeV2Error, match="seed_base commitment"):
        run_or_resume_replay(
            output,
            seed_base=12346,
            _env_factory=_FakeEnv,
            _provenance_provider=_provenance,
        )
    manifest_path = output / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="ascii"))
    value["opponent_private_discard"] = "leak"
    manifest_path.write_bytes(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    )
    with pytest.raises(HuRlReplayResumeV2Error, match="manifest fields"):
        run_or_resume_replay(
            output,
            seed_base=12345,
            _env_factory=_FakeEnv,
            _provenance_provider=_provenance,
        )


def test_plan_supports_100k_without_claiming_it_was_run(tmp_path: Path) -> None:
    manifest = _prepare(
        tmp_path / "scale-contract",
        paired_hands=ACCEPTANCE_PAIR_TARGET,
        shard_pairs=2_048,
    )
    assert manifest["paired_hand_count"] == 100_000
    assert manifest["shard_count"] == 49
    assert manifest["acceptance"] == {
        "paired_hand_target": 100_000,
        "target_scale_evaluated": False,
        "parquet_gate_evaluated": False,
    }
    assert "seed_base" not in manifest


def test_100k_plan_rejects_pathological_checkpoint_count(tmp_path: Path) -> None:
    with pytest.raises(HuRlReplayResumeV2Error, match="maximum 256"):
        _prepare(
            tmp_path / "too-many-shards",
            paired_hands=ACCEPTANCE_PAIR_TARGET,
            shard_pairs=1,
        )


def test_gzip_expansion_limit_is_enforced_while_streaming(monkeypatch) -> None:
    import gzip

    monkeypatch.setattr(replay_v2, "MAX_UNCOMPRESSED_SHARD_BYTES", 1_024)
    encoded = gzip.compress(b"x" * 2_048, mtime=0)
    with pytest.raises(HuRlReplayResumeV2Error, match="uncompressed shard"):
        replay_v2._decode_shard(encoded)


def test_manifest_pins_rust_seed_generator_and_never_allows_python_decks(
    tmp_path: Path,
) -> None:
    manifest = _prepare(tmp_path / "native-seed-contract", paired_hands=1)
    assert manifest["rng"] == {
        "deck_namespace": replay_v2.DECK_RNG_NAMESPACE,
        "policy_namespace": replay_v2.POLICY_RNG_NAMESPACE,
        "seed_base_commitment_sha256": replay_v2._seed_base_commitment(12345),
        "seed_stride": 17,
        "pair_seed_formula": "seed_base_plus_global_pair_index_times_seed_stride",
        "native_deck_generator": "rust_cpython_mt19937_splitmix64_shuffle_v1",
        "paired_lane_order": "pair_major_ab_then_ba",
        "seat_swap_formula": "swap_0_5_5_10_10_13_13_16_16_19_19_22_22_25_25_28_28_31_31_34",
        "hidden_tail_coupling": "positions_34_through_51_identical",
        "deck_materialized_in_python": False,
        "policy_counter_formula": "global_pair_index_times_10_plus_population_role_times_5_plus_street",
        "raw_seed_persisted_in_trajectory": False,
    }
    source = Path(replay_v2.__file__).read_text(encoding="utf-8")
    execute_source = source[source.index("def _execute_shard"):source.index("def validate_record")]
    assert "generate_benchmark_decks" not in source
    assert "explicit_deck" not in execute_source
