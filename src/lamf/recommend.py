"""Turn scores into prioritised next steps."""

from __future__ import annotations

from dataclasses import dataclass

from lamf.framework import Framework, Step
from lamf.scoring import OrgResult


@dataclass(frozen=True)
class Recommendation:
    """Next steps for one dimension."""

    dimension: str
    name: str
    current: str
    target: str  # next level name, or "Sustain" at the top level
    binding: bool
    steps: tuple[Step, ...]


def recommend(result: OrgResult, fw: Framework) -> list[Recommendation]:
    """Steps for each scored dimension; binding constraints first, then lowest scores."""
    recs = []
    for d in result.dimensions:
        if d.level is None:
            continue
        nxt = fw.next_level(d.level)
        recs.append(Recommendation(
            dimension=d.dimension, name=d.name, current=d.level.name,
            target=nxt.name if nxt else "Sustain",
            binding=d.dimension in result.binding,
            steps=fw.next_steps[d.dimension][d.level.name],
        ))
    score = {d.dimension: d.score or 0 for d in result.dimensions}
    return sorted(recs, key=lambda r: (not r.binding, score[r.dimension]))
