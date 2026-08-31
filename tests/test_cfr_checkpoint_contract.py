import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from ai.cfr.ofc_cfr import (
    CFR_ACTION_SCHEMA,
    CFR_INFO_MODEL,
    CFR_RULES_VERSION,
    OFC_CFR,
)
from ai.engine.turn_order import POSITION_CONTRACT_VERSION


def test_save_checkpoint_writes_mandatory_contract_metadata(tmp_path):
    checkpoint = tmp_path / "cfr.pkl"
    solver = OFC_CFR()

    solver.save_checkpoint(str(checkpoint))

    metadata = json.loads(
        checkpoint.with_suffix(".pkl.meta.json").read_text(encoding="utf-8")
    )
    repo_fl_config = (
        Path(__file__).resolve().parents[1]
        / "ai"
        / "config"
        / "fl_ev.json"
    )
    assert metadata["position_contract_version"] == POSITION_CONTRACT_VERSION == "bb_first_v1"
    assert metadata["rules_version"] == CFR_RULES_VERSION
    assert metadata["action_schema"] == CFR_ACTION_SCHEMA
    assert metadata["info_model"] == CFR_INFO_MODEL
    assert metadata["fl_config_sha256"] == hashlib.sha256(repo_fl_config.read_bytes()).hexdigest()


def test_load_rejects_legacy_before_loading_pickle_store(tmp_path):
    checkpoint = tmp_path / "legacy.pkl"
    checkpoint.with_suffix(".pkl.meta.json").write_text(
        json.dumps({"iteration": 2000, "max_cfr_depth": 3}),
        encoding="utf-8",
    )
    solver = OFC_CFR()
    solver.store.load = Mock(side_effect=AssertionError("store must not be loaded"))

    with pytest.raises(ValueError, match="Legacy CFR checkpoint rejected"):
        solver.load_checkpoint(str(checkpoint))

    solver.store.load.assert_not_called()


def test_load_allows_explicit_legacy_diagnostic_override(tmp_path):
    checkpoint = tmp_path / "legacy.pkl"
    source = OFC_CFR()
    source.store.get("legacy-key", 2).cumulative_regret[1] = 3.5
    source.store.save(str(checkpoint))

    loaded = OFC_CFR()
    with pytest.warns(RuntimeWarning, match="Legacy CFR checkpoint rejected"):
        loaded.load_checkpoint(str(checkpoint), allow_legacy=True)

    assert loaded.store.data["legacy-key"].cumulative_regret[1] == 3.5


def test_load_rejects_contract_mismatch_even_with_legacy_override(tmp_path):
    checkpoint = tmp_path / "mismatch.pkl"
    OFC_CFR().save_checkpoint(str(checkpoint))
    meta_path = checkpoint.with_suffix(".pkl.meta.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["rules_version"] = "different_rules"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    solver = OFC_CFR()
    solver.store.load = Mock(side_effect=AssertionError("store must not be loaded"))
    with pytest.raises(ValueError, match="contract mismatch.*rules_version"):
        solver.load_checkpoint(str(checkpoint), allow_legacy=True)

    solver.store.load.assert_not_called()
