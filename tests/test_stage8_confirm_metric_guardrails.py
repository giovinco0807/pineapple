from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src" / "ofc_regular"

CONFIRM_AGGREGATE_TOKENS = (
    "confirm_delta_mean",
    "confirm_delta_mean_on_fired",
    "confirm_delta_z_mean",
    "confirm_mean",
    "mean_confirm_delta",
)

CONFIRM_ROLE_TOKENS = (
    "CONFIRM_DELTA_METRIC_ROLE",
    "confirm_delta_metric_role",
    "confirm_delta_performance_claim_allowed",
    "performance_claim_allowed_from_score_mean",
    "RAW_CONFIRM_SCORE_ROLE",
)


def test_confirm_aggregate_outputs_have_metric_role_guardrails():
    offenders: list[str] = []
    for path in sorted(SRC_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if not any(token in text for token in CONFIRM_AGGREGATE_TOKENS):
            continue
        if any(token in text for token in CONFIRM_ROLE_TOKENS):
            continue
        offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []
