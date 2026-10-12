"""Scoring: respondent dimension scores, organisation aggregation, levels, flags.

Conventions follow the thesis calculation workbook: each dimension is the mean
of its coded 1-5 items, reported with the number of items answered, and
reverse-coded items use 6 - value. A respondent's dimension is "insufficient"
if fewer than ``min_items`` were answered.

Overall readiness is the LOWEST dimension level (the binding constraint), not
the mean: the thesis argues a firm is only as ready for autonomous data as its
weakest layer (maturity x literacy x design, inside culture).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from lamf.framework import FlagDef, Framework, Level


def respondent_scores(coded: pd.DataFrame, fw: Framework) -> pd.DataFrame:
    """One row per respondent: dimension scores (NaN when insufficient) and items answered."""
    out = coded[["org_id", "respondent_id"]].copy()
    for d in fw.dimensions:
        cols = [i.id for i in d.items if i.id in coded.columns]
        values = coded[cols] if cols else pd.DataFrame(index=coded.index)
        answered = values.notna().sum(axis=1) if cols else pd.Series(0, index=coded.index)
        mean = values.mean(axis=1) if cols else pd.Series(np.nan, index=coded.index)
        out[d.id] = mean.where(answered >= d.min_items)
        out[f"{d.id}__answered"] = answered
    return out


@dataclass(frozen=True)
class DimensionResult:
    """Organisation-level result for one dimension."""

    dimension: str
    name: str
    score: float | None
    sd: float | None
    n: int
    level: Level | None


@dataclass(frozen=True)
class OrgResult:
    """Everything known about one organisation."""

    org_id: str
    n_respondents: int
    dimensions: tuple[DimensionResult, ...]
    overall: Level | None
    mean_score: float | None
    binding: tuple[str, ...]
    flags: tuple[FlagDef, ...]

    @property
    def primary(self) -> str | None:
        """The binding dimension with the lowest score: where to start."""
        scores = {d.dimension: d.score for d in self.dimensions if d.dimension in self.binding}
        return min(scores, key=lambda k: scores[k]) if scores else None


def _mean(coded: pd.DataFrame, *cols: str) -> float | None:
    present = [c for c in cols if c in coded.columns]
    if not present:
        return None
    v = coded[present].stack().mean()  # stack() drops NaN
    return None if pd.isna(v) else float(v)


FlagRule = Callable[[pd.DataFrame, dict[str, DimensionResult]], bool]


def _ahead(c: pd.DataFrame, dims: dict[str, DimensionResult]) -> bool:
    m, l = dims.get("maturity"), dims.get("literacy")
    return bool(m and l and m.level and l.level and m.level.id > l.level.id)


def _explain(c: pd.DataFrame, dims: dict[str, DimensionResult]) -> bool:
    v = _mean(c, "q37")
    return v is not None and v >= 4


def _accountability(c: pd.DataFrame, dims: dict[str, DimensionResult]) -> bool:
    a, b = _mean(c, "q35"), _mean(c, "q36")
    return a is not None and b is not None and a >= 4 and b >= 4


def _contradicting(c: pd.DataFrame, dims: dict[str, DimensionResult]) -> bool:
    q21, rest = _mean(c, "q21"), _mean(c, "q18", "q19", "q20")
    return q21 is not None and rest is not None and q21 <= rest - 1


def _self_report(c: pd.DataFrame, dims: dict[str, DimensionResult]) -> bool:
    d = dims.get("interpretation")
    return bool(d and d.score is not None)


FLAG_RULES: dict[str, FlagRule] = {
    "firm_ahead_of_people": _ahead,
    "explainability_demand": _explain,
    "accountability_exposure": _accountability,
    "contradicting_figures_gap": _contradicting,
    "self_report": _self_report,
}


def score_organisations(coded: pd.DataFrame, fw: Framework) -> list[OrgResult]:
    """Aggregate respondents to organisations and evaluate levels and flags."""
    per_resp = respondent_scores(coded, fw)
    results = []
    for org_id, grp in per_resp.groupby("org_id", sort=True):
        dims: dict[str, DimensionResult] = {}
        for d in fw.dimensions:
            vals = grp[d.id].dropna()
            score = float(vals.mean()) if len(vals) else None
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else None
            dims[d.id] = DimensionResult(d.id, d.name, score, sd, int(len(vals)),
                                         fw.level_for(score) if score is not None else None)
        scored = [r for r in dims.values() if r.level is not None]
        overall = min((r.level for r in scored), key=lambda l: l.id) if scored else None
        binding = tuple(r.dimension for r in scored if overall and r.level.id == overall.id)
        mean_score = float(np.mean([r.score for r in scored])) if scored else None
        org_coded = coded[coded["org_id"] == org_id]
        flags = tuple(f for fid, f in fw.flags.items() if FLAG_RULES[fid](org_coded, dims))
        results.append(OrgResult(str(org_id), len(grp), tuple(dims.values()), overall, mean_score, binding, flags))
    return results


def results_frame(results: list[OrgResult]) -> pd.DataFrame:
    """Long-format table: one row per organisation x dimension."""
    rows = []
    for r in results:
        for d in r.dimensions:
            rows.append({"org_id": r.org_id, "dimension": d.dimension, "dimension_name": d.name,
                         "score": None if d.score is None else round(d.score, 2),
                         "sd": None if d.sd is None else round(d.sd, 2), "n": d.n,
                         "level": d.level.name if d.level else "insufficient data",
                         "overall_level": r.overall.name if r.overall else "insufficient data",
                         "binding_constraint": d.dimension in r.binding})
    return pd.DataFrame(rows)
