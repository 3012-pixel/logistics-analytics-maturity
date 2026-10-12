"""Read questionnaire responses (CSV) and code them onto the 1-5 scale.

CSV layout: one row per respondent with ``org_id``, ``respondent_id`` and one
column per item id (``q18``, ``q24``...). Lines starting with ``#`` are comments.

- Likert items: 1-5 or the labels SD / D / N / A / SA.
- Choice items: an option code from the framework (e.g. ``predictive``).
- Multi-select items: option codes separated by ``;`` (``none`` = nothing selected).
- Blank = not answered.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd

from lamf.framework import Framework, Item

ID_COLUMNS = ["org_id", "respondent_id"]


class ResponseError(ValueError):
    """Raised for invalid response data; the message names the row and column."""


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == ""


def code_item(item: Item, value: object, likert_labels: dict[str, int]) -> float | None:
    """Convert one raw answer to a 1-5 score (``None`` if unanswered).

    Raises:
        ValueError: for values outside what the item allows.
    """
    if _blank(value):
        return None
    text = str(value).strip()
    if item.type == "likert":
        if text.upper() in likert_labels:
            score = float(likert_labels[text.upper()])
        else:
            try:
                score = float(text)
            except ValueError:
                raise ValueError(f"expected 1-5 or one of {sorted(likert_labels)}, got {text!r}") from None
            if score not in {1.0, 2.0, 3.0, 4.0, 5.0}:
                raise ValueError(f"Likert value must be 1-5, got {text!r}")
        return 6.0 - score if item.reverse else score
    if item.type == "choice":
        if text not in item.options:
            raise ValueError(f"expected one of {sorted(item.options)}, got {text!r}")
        return item.options[text].score
    # multi-select: score = 1 + 4 * share of options in place
    codes = [c.strip() for c in text.split(";") if c.strip()]
    if codes == ["none"]:
        return 1.0
    bad = [c for c in codes if c not in item.options]
    if bad:
        raise ValueError(f"unknown option(s) {bad}; expected codes from {sorted(item.options)} or 'none'")
    return 1.0 + 4.0 * len(set(codes)) / len(item.options)


def load_responses(path: str | Path, fw: Framework) -> pd.DataFrame:
    """Load and code a responses CSV.

    Returns:
        DataFrame with ``org_id``, ``respondent_id`` and one coded (1-5 or NaN)
        column per item present in the file.
    """
    path = Path(path)
    if not path.exists():
        raise ResponseError(f"responses file not found: {path}")
    raw = pd.read_csv(path, dtype=str, comment="#", skip_blank_lines=True, keep_default_na=False)
    missing = [c for c in ID_COLUMNS if c not in raw.columns]
    if missing:
        raise ResponseError(f"missing required column(s): {missing}")
    items = {i.id: i for i in fw.all_items}
    unknown = [c for c in raw.columns if c not in items and c not in ID_COLUMNS]
    if unknown:
        raise ResponseError(f"unknown column(s) not in the framework: {unknown}")
    if raw.empty:
        raise ResponseError("responses file has no rows")

    coded = raw[ID_COLUMNS].copy()
    for col in [c for c in raw.columns if c in items]:
        values = []
        for idx, value in raw[col].items():
            try:
                values.append(code_item(items[col], value, fw.likert_labels))
            except ValueError as exc:
                raise ResponseError(f"row {idx + 1} (respondent {raw.at[idx, 'respondent_id']!r}), "
                                    f"column {col!r}: {exc}") from None
        coded[col] = pd.to_numeric(pd.Series(values, index=raw.index), errors="coerce")
    blank_ids = coded[ID_COLUMNS].apply(lambda s: s.str.strip() == "").any(axis=1)
    if blank_ids.any():
        raise ResponseError(f"row {int(blank_ids.idxmax()) + 1}: org_id and respondent_id are required")
    return coded


def write_template(fw: Framework, csv_path: str | Path, codebook_path: str | Path) -> tuple[Path, Path]:
    """Write a blank responses CSV and a codebook describing every column."""
    csv_path, codebook_path = Path(csv_path), Path(codebook_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ID_COLUMNS + [i.id for i in fw.all_items]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(columns)
    rows = []
    dim_of = {i.id: d.name for d in fw.dimensions for i in d.items}
    for item in fw.all_items:
        if item.type == "likert":
            allowed = "1-5 or " + "/".join(fw.likert_labels) + (" (reverse-scored)" if item.reverse else "")
        elif item.type == "choice":
            allowed = "; ".join(f"{c} = {o.label}" for c, o in item.options.items())
        else:
            allowed = "codes separated by ';' (or 'none'): " + "; ".join(f"{c} = {o.label}" for c, o in item.options.items())
        rows.append({"column": item.id, "dimension": dim_of.get(item.id, "context (flags only)"),
                     "type": item.type, "question": item.text, "allowed_values": allowed})
    pd.DataFrame(rows).to_csv(codebook_path, index=False)
    return csv_path, codebook_path
