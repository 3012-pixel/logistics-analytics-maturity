"""Shared fixtures. All responses used in tests are SYNTHETIC and built in code."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lamf.framework import load_framework

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework" / "lamf_framework.yaml"

NEUTRAL = {  # a middle-of-the-road respondent: every Likert item 3
    "q24": "diagnostic", "q22": "ask_around", "q23": "review_meeting;thresholds", "q31": "persuasion",
    **{q: "3" for q in ["q18", "q19", "q20", "q21", "q26", "q27", "q28", "q29", "q30", "q32", "q33", "q34",
                        "q13", "q14", "q35", "q36", "q37"]},
    "q10": "partially",
}


@pytest.fixture(scope="session")
def fw():
    return load_framework(FRAMEWORK)


def make_csv(tmp_path: Path, rows: list[dict[str, str]], name: str = "responses.csv", header: str = "") -> Path:
    """Write rows (dicts of item -> raw answer) as a responses CSV."""
    df = pd.DataFrame(rows).fillna("")
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(header)
        df.to_csv(fh, index=False)
    return path


def respondent(org: str = "org-a", rid: str = "r1", **overrides: str) -> dict[str, str]:
    """A neutral respondent with selected answers overridden."""
    return {"org_id": org, "respondent_id": rid, **NEUTRAL, **overrides}
