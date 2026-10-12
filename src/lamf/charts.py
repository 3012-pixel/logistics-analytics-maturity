"""Radar charts of dimension scores with maturity-level rings."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lamf.framework import Framework  # noqa: E402
from lamf.scoring import OrgResult  # noqa: E402

PALETTE = ["#2a6f97", "#e07a1f", "#2e8b57", "#8e44ad", "#c0392b"]
RING = "#b8b8b8"


def _axes(fw: Framework, title: str):
    labels = [textwrap.fill(d.name, 14) for d in fw.dimensions]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=150, subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(1, 5)
    ax.set_xticks(angles, labels, fontsize=9)
    ax.set_yticks([l.min_score for l in fw.levels[1:]] + [5])
    ax.set_yticklabels([])
    ax.grid(color=RING, linewidth=0.6)
    for level in fw.levels:  # label each ring with the level it starts
        ax.text(np.deg2rad(8), level.min_score + 0.5 if level.min_score < 5 else 4.6, level.name,
                fontsize=7, color="#777", ha="left", va="center")
    ax.set_title(title, fontsize=11, pad=22)
    return fig, ax, angles


def _plot(ax, angles: list[float], values: list[float], colour: str, label: str) -> None:
    closed_a = angles + angles[:1]
    closed_v = values + values[:1]
    ax.plot(closed_a, closed_v, color=colour, linewidth=2, label=label)
    ax.fill(closed_a, closed_v, color=colour, alpha=0.18)
    ax.scatter(angles, values, color=colour, s=18, zorder=3)


def radar(result: OrgResult, fw: Framework, path: str | Path) -> Path:
    """Radar for one organisation. Dimensions with insufficient data are drawn at 1 and marked."""
    path = Path(path)
    overall = result.overall.name if result.overall else "n/a"
    fig, ax, angles = _axes(fw, f"{result.org_id}: overall readiness {overall}")
    values = [d.score if d.score is not None else 1.0 for d in result.dimensions]
    _plot(ax, angles, values, PALETTE[0], result.org_id)
    for a, d in zip(angles, result.dimensions):
        if d.score is None:
            ax.text(a, 1.4, "no data", fontsize=7, color="#c0392b", ha="center")
        elif d.dimension in result.binding:
            ax.scatter([a], [d.score], s=90, facecolors="none", edgecolors="#c0392b", linewidths=1.6, zorder=4)
    caption = "Red circle = binding constraint (lowest level)"
    if fw.is_draft:
        caption += " | DRAFT framework, not validated"
    fig.text(0.5, 0.02, caption, ha="center", fontsize=7, color="#555")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def comparison_radar(results: list[OrgResult], fw: Framework, path: str | Path) -> Path:
    """All organisations on one radar."""
    path = Path(path)
    fig, ax, angles = _axes(fw, "Maturity profile by organisation")
    for i, r in enumerate(results):
        _plot(ax, angles, [d.score if d.score is not None else 1.0 for d in r.dimensions],
              PALETTE[i % len(PALETTE)], r.org_id)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=min(3, len(results)), frameon=False, fontsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
