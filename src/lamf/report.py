"""Markdown reports per organisation."""

from __future__ import annotations

from pathlib import Path

from lamf.framework import Framework
from lamf.recommend import recommend
from lamf.scoring import OrgResult


def banner(fw: Framework, synthetic: bool) -> str:
    """Warnings printed at the top of every report."""
    lines = []
    if synthetic:
        lines.append("> **SYNTHETIC DATA.** These responses were generated for demonstration; they describe no real organisation.")
    if fw.is_draft:
        lines.append(f"> **DRAFT FRAMEWORK ({fw.meta.get('status', 'draft')}).** Scoring thresholds and next steps are "
                     "design choices derived from the source thesis and have not been empirically validated.")
    return "\n>\n".join(lines)


def org_report(result: OrgResult, fw: Framework, radar_file: str, synthetic: bool) -> str:
    """Markdown report for one organisation."""
    overall = result.overall.name if result.overall else "insufficient data"
    parts = [f"# Analytics maturity assessment: {result.org_id}", "", banner(fw, synthetic), "",
             f"**Overall readiness: {overall}**, set by the lowest-level dimension(s): "
             f"{', '.join(fw.dimension(b).name for b in result.binding) or 'n/a'}. "
             f"**Start with: {fw.dimension(result.primary).name if result.primary else 'n/a'}** (lowest score among them). "
             f"Mean score {result.mean_score:.2f} / 5, shown for information only. Respondents: {result.n_respondents}.", "",
             f"![Radar chart]({radar_file})", "",
             "## Scores by dimension", "",
             "| Dimension | Score (1-5) | Level | Respondents scored | Spread (sd) |",
             "|---|---:|---|---:|---:|"]
    for d in result.dimensions:
        score = "-" if d.score is None else f"{d.score:.2f}"
        sd = "-" if d.sd is None else f"{d.sd:.2f}"
        level = d.level.name if d.level else "insufficient data"
        if d.dimension in result.binding:
            level = f"**{level}** (binding)"
        parts.append(f"| {d.name} | {score} | {level} | {d.n} | {sd} |")

    if result.flags:
        parts += ["", "## Diagnostic flags", ""]
        for f in result.flags:
            parts.append(f"- **{f.title}.** {f.message} _(source: {f.source})_")

    parts += ["", "## Recommended next steps", "", "Ordered by priority: binding constraints first, then lowest scores.", ""]
    for rec in recommend(result, fw):
        tag = " **(binding constraint)**" if rec.binding else ""
        parts.append(f"### {rec.name}: {rec.current} → {rec.target}{tag}")
        for s in rec.steps:
            parts.append(f"- {s.text} _({'from thesis' if s.source == 'thesis' else 'draft'})_")
        parts.append("")

    parts += ["---", f"Framework: {fw.meta.get('name')} v{fw.meta.get('version')}. "
              f"Based on: {' '.join(str(fw.meta.get('citation', '')).split())}"]
    return "\n".join(parts) + "\n"


def write_reports(results: list[OrgResult], fw: Framework, out_dir: Path, radar_files: dict[str, str],
                  synthetic: bool) -> list[Path]:
    """Write one report per organisation plus an index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    index = ["# Assessment summary", "", banner(fw, synthetic), "",
             "| Organisation | Overall readiness | Start with | Binding dimensions | Flags |", "|---|---|---|---|---|"]
    for r in results:
        p = out_dir / f"report_{r.org_id}.md"
        p.write_text(org_report(r, fw, radar_files[r.org_id], synthetic), encoding="utf-8")
        paths.append(p)
        start = fw.dimension(r.primary).name if r.primary else "n/a"
        index.append(f"| [{r.org_id}]({p.name}) | {r.overall.name if r.overall else 'n/a'} | {start} | "
                     f"{', '.join(fw.dimension(b).name for b in r.binding)} | {len(r.flags)} |")
    idx = out_dir / "README.md"
    idx.write_text("\n".join(index) + "\n\n![Comparison](radar_comparison.png)\n", encoding="utf-8")
    return [idx, *paths]
