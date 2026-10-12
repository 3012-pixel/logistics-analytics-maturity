"""Scoring, flags, recommendations, synthetic data and the CLI."""

from __future__ import annotations

import pandas as pd
import pytest

from lamf.cli import main
from lamf.recommend import recommend
from lamf.responses import load_responses
from lamf.scoring import respondent_scores, results_frame, score_organisations
from lamf.synthetic import generate
from tests.conftest import make_csv, respondent


def _score(fw, tmp_path, rows):
    return score_organisations(load_responses(make_csv(tmp_path, rows), fw), fw)


def _dim(result, dim_id):
    return next(d for d in result.dimensions if d.dimension == dim_id)


def test_dimension_mean_by_hand(fw, tmp_path) -> None:
    # culture: q26..q29 = 4,5,3,4 and q30 = 2 (reverse -> 4): mean (4+5+3+4+4)/5 = 4.0
    r = respondent(q26="4", q27="5", q28="3", q29="4", q30="2")
    res = _score(fw, tmp_path, [r])[0]
    assert _dim(res, "culture").score == pytest.approx(4.0)
    assert _dim(res, "culture").level.name == "Prescriptive"


def test_minimum_items_rule(fw, tmp_path) -> None:
    r = respondent(q18="5", q19="5", q20="", q21="")  # literacy needs 3 of 4 answered
    coded = load_responses(make_csv(tmp_path, [r]), fw)
    per = respondent_scores(coded, fw)
    assert pd.isna(per.loc[0, "literacy"]) and per.loc[0, "literacy__answered"] == 2
    res = score_organisations(coded, fw)[0]
    assert _dim(res, "literacy").level is None
    assert "literacy" not in res.binding


def test_org_aggregation_mean_sd_n(fw, tmp_path) -> None:
    rows = [respondent(rid="r1", q32="2", q33="2", q34="2"), respondent(rid="r2", q32="4", q33="4", q34="4")]
    d = _dim(_score(fw, tmp_path, rows)[0], "autonomy")
    assert (d.score, d.n) == (3.0, 2)
    assert d.sd == pytest.approx(pd.Series([2.0, 4.0]).std(ddof=1))


def test_overall_is_lowest_level_not_mean(fw, tmp_path) -> None:
    high = {q: "5" for q in ["q18", "q19", "q20", "q21", "q26", "q27", "q28", "q29", "q32", "q33", "q34", "q13"]}
    r = respondent(**high, q30="1", q14="1", q10="first_glance", q24="descriptive",
                   q22="procedure", q23="review_meeting;thresholds;dashboard;procedures;post_incident", q31="test")
    res = _score(fw, tmp_path, [r])[0]
    assert res.mean_score > 4  # nearly everything is top-level...
    assert res.overall.name == "Descriptive"  # ...but analytics maturity binds
    assert res.binding == ("maturity",) and res.primary == "maturity"


def test_separate_organisations(fw, tmp_path) -> None:
    rows = [respondent(org="a"), respondent(org="b", rid="r2", q24="prescriptive")]
    results = {r.org_id: r for r in _score(fw, tmp_path, rows)}
    assert _dim(results["a"], "maturity").score == 2.0 and _dim(results["b"], "maturity").score == 4.0


# --- flags -------------------------------------------------------------------

def _flags(fw, tmp_path, **overrides):
    return {f.id for f in _score(fw, tmp_path, [respondent(**overrides)])[0].flags}


def test_neutral_respondent_only_self_report_flag(fw, tmp_path) -> None:
    assert _flags(fw, tmp_path) == {"self_report"}


@pytest.mark.parametrize("overrides,flag", [
    ({"q24": "prescriptive"}, "firm_ahead_of_people"),                      # maturity 4 > literacy level 3
    ({"q37": "5"}, "explainability_demand"),
    ({"q35": "4", "q36": "SA"}, "accountability_exposure"),
    ({"q18": "5", "q19": "5", "q20": "5", "q21": "3"}, "contradicting_figures_gap"),
])
def test_each_flag_triggers(fw, tmp_path, overrides, flag) -> None:
    assert flag in _flags(fw, tmp_path, **overrides)


def test_accountability_needs_both_items(fw, tmp_path) -> None:
    assert "accountability_exposure" not in _flags(fw, tmp_path, q35="5", q36="3")


# --- recommendations ------------------------------------------------------------

def test_recommendations_target_next_level_binding_first(fw, tmp_path) -> None:
    res = _score(fw, tmp_path, [respondent(q24="descriptive")])[0]
    recs = recommend(res, fw)
    assert recs[0].dimension == "maturity" and recs[0].binding
    assert (recs[0].current, recs[0].target) == ("Descriptive", "Diagnostic")
    assert recs[0].steps == fw.next_steps["maturity"]["Descriptive"]


def test_top_level_gets_sustain_steps(fw, tmp_path) -> None:
    res = _score(fw, tmp_path, [respondent(q24="prescriptive")])[0]
    rec = next(r for r in recommend(res, fw) if r.dimension == "maturity")
    assert rec.target == "Sustain" and rec.steps == fw.next_steps["maturity"]["Prescriptive"]


# --- synthetic data and CLI ---------------------------------------------------------

def test_synthetic_is_deterministic_valid_and_varied(fw, tmp_path) -> None:
    a, b = generate(fw, seed=7), generate(fw, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(generate(fw, seed=8))
    path = make_csv(tmp_path, a.to_dict("records"))
    results = score_organisations(load_responses(path, fw), fw)  # loads => all values valid
    assert len({r.overall.name for r in results}) == 3  # personas land at different levels


def test_cli_end_to_end(tmp_path, capsys) -> None:
    data, out = tmp_path / "synth.csv", tmp_path / "out"
    assert main(["synth", "--out", str(data), "--seed", "42"]) == 0
    assert data.read_text(encoding="utf-8").startswith("# SYNTHETIC")
    assert main(["assess", str(data), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "SYNTHETIC" in printed and "not validated" in printed
    scores = pd.read_csv(out / "scores.csv")
    assert set(scores["dimension"]) == {"maturity", "routines", "literacy", "culture", "autonomy", "interpretation"}
    pngs = {p.name for p in out.glob("*.png")}
    assert "radar_comparison.png" in pngs and len(pngs) == 4
    report = (out / "report_3pl-growing.md").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" in report and "Not empirically validated" in report and "Start with" in report
    assert "DRAFT FRAMEWORK" not in report


def test_cli_template_and_bad_input(tmp_path, capsys) -> None:
    assert main(["template", "--out", str(tmp_path / "t.csv")]) == 0
    assert (tmp_path / "t_codebook.csv").exists()
    bad = make_csv(tmp_path, [respondent(q18="9")])
    assert main(["assess", str(bad), "--out", str(tmp_path / "o")]) == 2
    assert "column 'q18'" in capsys.readouterr().err


def test_results_frame_marks_insufficient(fw, tmp_path) -> None:
    res = _score(fw, tmp_path, [respondent(q32="", q33="", q34="")])
    row = results_frame(res).set_index("dimension").loc["autonomy"]
    assert row["level"] == "insufficient data" and row["n"] == 0
