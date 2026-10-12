"""Load and validate the framework definition (YAML).

The YAML is the single source of truth: dimensions, items, answer scoring,
level thresholds, flags and next steps. Code never hardcodes framework content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ITEM_TYPES = {"likert", "choice", "multi"}
KNOWN_FLAGS = {"firm_ahead_of_people", "explainability_demand", "accountability_exposure",
               "contradicting_figures_gap", "self_report"}


class FrameworkError(ValueError):
    """Raised when the framework file is inconsistent."""


@dataclass(frozen=True)
class Option:
    """One answer option of a choice/multi item."""

    code: str
    label: str
    score: float | None = None


@dataclass(frozen=True)
class Item:
    """One questionnaire item."""

    id: str
    text: str
    type: str
    reverse: bool = False
    options: dict[str, Option] = field(default_factory=dict)
    score_rule: str | None = None
    source: str = ""


@dataclass(frozen=True)
class Dimension:
    """A scored dimension made of items."""

    id: str
    name: str
    construct: str
    unit: str
    min_items: int
    items: tuple[Item, ...]
    self_report_caveat: bool = False


@dataclass(frozen=True)
class Level:
    """A maturity level and the minimum 1-5 score that reaches it."""

    id: int
    name: str
    min_score: float
    summary: str


@dataclass(frozen=True)
class Step:
    """A recommended next step."""

    text: str
    source: str


@dataclass(frozen=True)
class FlagDef:
    """Display text for a diagnostic flag (logic lives in ``scoring.FLAG_RULES``)."""

    id: str
    title: str
    rule: str
    message: str
    source: str


@dataclass(frozen=True)
class Framework:
    """A validated framework."""

    meta: dict[str, Any]
    levels: tuple[Level, ...]
    likert_labels: dict[str, int]
    dimensions: tuple[Dimension, ...]
    context_items: tuple[Item, ...]
    flags: dict[str, FlagDef]
    next_steps: dict[str, dict[str, tuple[Step, ...]]]

    @property
    def is_draft(self) -> bool:
        """True while the framework has not been reviewed by its author."""
        return str(self.meta.get("status", "")).startswith("draft")

    @property
    def not_validated(self) -> bool:
        """True until the thresholds have been calibrated on real data."""
        return bool(self.meta.get("not_validated"))

    def level_for(self, score: float) -> Level:
        """Highest level whose ``min_score`` the score reaches."""
        chosen = self.levels[0]
        for level in self.levels:
            if score >= level.min_score - 1e-9:
                chosen = level
        return chosen

    def next_level(self, level: Level) -> Level | None:
        """The level above ``level``, or None at the top."""
        later = [l for l in self.levels if l.id > level.id]
        return later[0] if later else None

    def dimension(self, dim_id: str) -> Dimension:
        """Look up a dimension by id."""
        for d in self.dimensions:
            if d.id == dim_id:
                return d
        raise KeyError(dim_id)

    @property
    def all_items(self) -> list[Item]:
        """Scored items followed by context items."""
        return [i for d in self.dimensions for i in d.items] + list(self.context_items)


def _item(raw: dict[str, Any]) -> Item:
    try:
        options = {code: Option(code=str(code), label=str(o.get("label", code)),
                                score=None if o.get("score") is None else float(o["score"]))
                   for code, o in (raw.get("options") or {}).items()}
        return Item(id=str(raw["id"]), text=str(raw["text"]), type=str(raw["type"]),
                    reverse=bool(raw.get("reverse", False)), options=options,
                    score_rule=raw.get("score_rule"), source=str(raw.get("source", "")))
    except KeyError as exc:
        raise FrameworkError(f"item is missing field {exc}: {raw}") from exc


def load_framework(path: str | Path) -> Framework:
    """Load ``path`` and validate it. Raises :class:`FrameworkError`."""
    path = Path(path)
    if not path.exists():
        raise FrameworkError(f"framework file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        fw = Framework(
            meta=dict(raw.get("meta", {})),
            levels=tuple(Level(int(l["id"]), str(l["name"]), float(l["min_score"]), str(l.get("summary", "")))
                         for l in raw["levels"]),
            likert_labels={str(k): int(v) for k, v in raw["likert"]["labels"].items()},
            dimensions=tuple(Dimension(id=str(d["id"]), name=str(d["name"]), construct=str(d.get("construct", "")),
                                       unit=str(d.get("unit", "")), min_items=int(d.get("min_items", 1)),
                                       items=tuple(_item(i) for i in d["items"]),
                                       self_report_caveat=bool(d.get("self_report_caveat", False)))
                             for d in raw["dimensions"]),
            context_items=tuple(_item(i) for i in raw.get("context_items", [])),
            flags={fid: FlagDef(fid, str(f["title"]), str(f.get("rule", "")), " ".join(str(f["message"]).split()),
                                str(f.get("source", "")))
                   for fid, f in (raw.get("flags") or {}).items()},
            next_steps={dim: {lvl: tuple(Step(str(s["text"]), str(s.get("source", "author"))) for s in steps)
                              for lvl, steps in by_level.items()}
                        for dim, by_level in (raw.get("next_steps") or {}).items()},
        )
    except (KeyError, TypeError) as exc:
        raise FrameworkError(f"framework file is missing a required section or field: {exc}") from exc
    validate(fw)
    return fw


def validate(fw: Framework) -> None:
    """Check internal consistency; raise :class:`FrameworkError` listing every problem."""
    problems: list[str] = []
    mins = [l.min_score for l in fw.levels]
    if len(fw.levels) < 2 or mins != sorted(mins) or len(set(mins)) != len(mins):
        problems.append("level min_score values must be strictly increasing (at least 2 levels)")
    ids = [i.id for i in fw.all_items]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        problems.append(f"duplicate item ids: {dupes}")
    for item in fw.all_items:
        if item.type not in ITEM_TYPES:
            problems.append(f"{item.id}: unknown type {item.type!r}")
        if item.type in {"choice", "multi"} and not item.options:
            problems.append(f"{item.id}: {item.type} item needs options")
        if item.type == "choice" and any(o.score is None for o in item.options.values()):
            problems.append(f"{item.id}: every choice option needs a score")
    for d in fw.dimensions:
        if not 1 <= d.min_items <= len(d.items):
            problems.append(f"{d.id}: min_items must be between 1 and {len(d.items)}")
        steps = fw.next_steps.get(d.id, {})
        missing = [l.name for l in fw.levels if not steps.get(l.name)]
        if missing:
            problems.append(f"{d.id}: no next steps for levels {missing}")
    unknown_flags = set(fw.flags) - KNOWN_FLAGS
    if unknown_flags:
        problems.append(f"unknown flags (no rule implemented): {sorted(unknown_flags)}")
    if problems:
        raise FrameworkError("invalid framework:\n  - " + "\n  - ".join(problems))
