"""Framework validation, response coding and templates."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from lamf.framework import FrameworkError, load_framework
from lamf.responses import ResponseError, code_item, load_responses, write_template
from tests.conftest import FRAMEWORK, make_csv, respondent


def test_bundled_framework_loads_as_draft(fw) -> None:
    assert [l.name for l in fw.levels] == ["Descriptive", "Diagnostic", "Predictive", "Prescriptive"]
    assert len(fw.dimensions) == 6
    assert fw.is_draft
    assert "Aditya" in fw.meta["citation"]


def _broken(tmp_path, mutate) -> str:
    data = yaml.safe_load(FRAMEWORK.read_text(encoding="utf-8"))
    mutate(data)
    p = tmp_path / "fw.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(p)


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["levels"][2].update(min_score=1.5), "strictly increasing"),
    (lambda d: d["dimensions"][2]["items"][1].update(id="q18"), "duplicate item ids"),
    (lambda d: d["next_steps"]["literacy"].pop("Predictive"), "no next steps"),
    (lambda d: d["dimensions"][0].update(min_items=5), "min_items"),
    (lambda d: d["flags"].update(made_up={"title": "x", "message": "y"}), "unknown flags"),
])
def test_invalid_framework_rejected(tmp_path, mutate, match) -> None:
    with pytest.raises(FrameworkError, match=match):
        load_framework(_broken(tmp_path, mutate))


def test_level_thresholds_at_edges(fw) -> None:
    assert fw.level_for(1.0).name == "Descriptive"
    assert fw.level_for(1.99).name == "Descriptive"
    assert fw.level_for(2.0).name == "Diagnostic"
    assert fw.level_for(3.999).name == "Predictive"
    assert fw.level_for(4.0).name == "Prescriptive"
    assert fw.level_for(5.0).name == "Prescriptive"


def _item(fw, item_id):
    return next(i for i in fw.all_items if i.id == item_id)


def test_likert_numeric_label_and_reverse(fw) -> None:
    q18, q30 = _item(fw, "q18"), _item(fw, "q30")
    assert code_item(q18, "4", fw.likert_labels) == 4.0
    assert code_item(q18, "sa", fw.likert_labels) == 5.0
    assert code_item(q30, "5", fw.likert_labels) == 1.0   # hierarchy reverse-coded: 6 - 5
    assert code_item(q30, "SD", fw.likert_labels) == 5.0
    assert code_item(q18, "", fw.likert_labels) is None
    for bad in ["0", "6", "2.5", "maybe"]:
        with pytest.raises(ValueError):
            code_item(q18, bad, fw.likert_labels)


def test_choice_and_multi(fw) -> None:
    q24, q23 = _item(fw, "q24"), _item(fw, "q23")
    assert code_item(q24, "predictive", fw.likert_labels) == 3.0
    assert code_item(q23, "none", fw.likert_labels) == 1.0
    assert code_item(q23, "thresholds;dashboard", fw.likert_labels) == pytest.approx(1 + 4 * 2 / 5)
    assert code_item(q23, "thresholds;thresholds", fw.likert_labels) == pytest.approx(1 + 4 * 1 / 5)
    with pytest.raises(ValueError, match="expected one of"):
        code_item(q24, "advanced", fw.likert_labels)
    with pytest.raises(ValueError, match="unknown option"):
        code_item(q23, "robots", fw.likert_labels)


def test_load_errors_name_row_and_column(fw, tmp_path) -> None:
    path = make_csv(tmp_path, [respondent(rid="ok"), respondent(rid="bad", q19="7")])
    with pytest.raises(ResponseError, match=r"row 2 \(respondent 'bad'\), column 'q19'"):
        load_responses(path, fw)


def test_load_rejects_unknown_and_missing_columns(fw, tmp_path) -> None:
    with pytest.raises(ResponseError, match="unknown column"):
        load_responses(make_csv(tmp_path, [{**respondent(), "q99": "3"}]), fw)
    with pytest.raises(ResponseError, match="missing required"):
        load_responses(make_csv(tmp_path, [{"respondent_id": "r1", "q18": "3"}]), fw)


def test_comment_header_and_partial_columns(fw, tmp_path) -> None:
    path = make_csv(tmp_path, [{"org_id": "o", "respondent_id": "r", "q18": "A", "q19": "4"}],
                    header="# SYNTHETIC test file\n")
    coded = load_responses(path, fw)
    assert coded.loc[0, "q18"] == 4.0 and list(coded.columns) == ["org_id", "respondent_id", "q18", "q19"]


def test_template_round_trip(fw, tmp_path) -> None:
    csv_path, codebook = write_template(fw, tmp_path / "t.csv", tmp_path / "t_codebook.csv")
    header = pd.read_csv(csv_path).columns.tolist()
    assert header[:2] == ["org_id", "respondent_id"] and "q37" in header
    book = pd.read_csv(codebook)
    assert set(book["column"]) == {i.id for i in fw.all_items}
    assert "reverse-scored" in book.set_index("column").loc["q30", "allowed_values"]
