from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ofc_regular import hu_turn3_stage3_feature_rust as subject


def _library(tmp_path: Path, name: str, payload: bytes) -> tuple[Path, str]:
    path = (tmp_path / name).resolve()
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_exact_library_binding_is_scoped_nested_and_restored(tmp_path: Path):
    first, first_sha = _library(tmp_path, "first.so", b"accepted-feature-one")
    second, second_sha = _library(tmp_path, "second.so", b"accepted-feature-two")
    original = subject._library_path()

    with subject.pinned_feature_encoder_library(
        first,
        expected_sha256=first_sha,
    ):
        assert subject._library_path() == first
        with subject.pinned_feature_encoder_library(
            second,
            expected_sha256=second_sha,
        ):
            assert subject._library_path() == second
        assert subject._library_path() == first

    assert subject._library_path() == original


def test_binding_is_restored_when_materialization_body_raises(tmp_path: Path):
    library, digest = _library(tmp_path, "feature.so", b"accepted-feature")
    original = subject._library_path()

    with pytest.raises(RuntimeError, match="synthetic materialization failure"):
        with subject.pinned_feature_encoder_library(
            library,
            expected_sha256=digest,
        ):
            assert subject._library_path() == library
            raise RuntimeError("synthetic materialization failure")

    assert subject._library_path() == original


def test_wrong_sha_relative_path_and_post_bind_tamper_fail_closed(tmp_path: Path):
    library, digest = _library(tmp_path, "feature.so", b"accepted-feature")

    with pytest.raises(PermissionError, match="SHA-256 changed"):
        with subject.pinned_feature_encoder_library(
            library,
            expected_sha256="f" * 64,
        ):
            pytest.fail("wrong-SHA feature encoder was accepted")

    with pytest.raises(ValueError, match="absolute non-symlink"):
        with subject.pinned_feature_encoder_library(
            Path("relative-feature.so"),
            expected_sha256=digest,
        ):
            pytest.fail("relative feature encoder was accepted")

    with subject.pinned_feature_encoder_library(
        library,
        expected_sha256=digest,
    ):
        library.write_bytes(b"changed-after-binding")
        with pytest.raises(PermissionError, match="SHA-256 changed"):
            subject._library_path()


def test_load_library_honors_explicit_path_and_rechecks_pin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    library, digest = _library(tmp_path, "feature.so", b"accepted-feature")
    loaded_paths: list[str] = []

    class FakeFunction:
        argtypes = None
        restype = None

    class FakeLibrary:
        ofc_stage3_encode = FakeFunction()

    def fake_cdll(path: str):
        loaded_paths.append(path)
        return FakeLibrary()

    monkeypatch.setattr(subject.ctypes, "CDLL", fake_cdll)
    with subject.pinned_feature_encoder_library(
        library,
        expected_sha256=digest,
    ):
        loaded = subject._load_library()

    assert isinstance(loaded, FakeLibrary)
    assert loaded_paths == [str(library)]
