import pytest

from ofc_regular import fl_ev


def test_legacy_fl_ev_output_requires_explicit_flag(monkeypatch, tmp_path):
    output = tmp_path / "fl_ev.json"
    monkeypatch.setattr("sys.argv", ["fl_ev", "--trials", "1", "--output", str(output)])

    with pytest.raises(SystemExit) as excinfo:
        fl_ev.main()

    assert "--allow-legacy-chain-output" in str(excinfo.value)
    assert not output.exists()
