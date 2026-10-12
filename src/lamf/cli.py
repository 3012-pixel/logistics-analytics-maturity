"""Command-line interface.

    lamf template --out responses_template.csv
    lamf synth --out data/synthetic_responses.csv --seed 42
    lamf assess data/synthetic_responses.csv --out outputs/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from lamf.charts import comparison_radar, radar
from lamf.framework import FrameworkError, load_framework
from lamf.report import write_reports
from lamf.responses import ResponseError, load_responses, write_template
from lamf.scoring import results_frame, score_organisations
from lamf.synthetic import HEADER, write_synthetic

DEFAULT_FRAMEWORK = "framework/lamf_framework.yaml"


def _is_synthetic(path: Path) -> bool:
    with path.open(encoding="utf-8") as fh:
        return "SYNTHETIC" in fh.readline().upper()


def cmd_assess(args: argparse.Namespace) -> int:
    """Score responses and write scores, radars and reports."""
    fw = load_framework(args.framework)
    path = Path(args.responses)
    coded = load_responses(path, fw)
    results = score_organisations(coded, fw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    synthetic = _is_synthetic(path)

    results_frame(results).to_csv(out / "scores.csv", index=False)
    radar_files = {}
    for r in results:
        p = radar(r, fw, out / f"radar_{r.org_id}.png")
        radar_files[r.org_id] = p.name
    comparison_radar(results, fw, out / "radar_comparison.png")
    write_reports(results, fw, out, radar_files, synthetic)

    if synthetic:
        print("NOTE: input is SYNTHETIC data.")
    if fw.is_draft:
        print(f"NOTE: framework status '{fw.meta.get('status')}'. Thresholds and next steps are not validated.")
    print(f"\n{'Organisation':<18} {'Overall':<13} {'Start with':<34} Flags")
    for r in results:
        overall = r.overall.name if r.overall else "n/a"
        primary = fw.dimension(r.primary).name if r.primary else "n/a"
        others = f" (+{len(r.binding) - 1} tied)" if len(r.binding) > 1 else ""
        print(f"{r.org_id:<18} {overall:<13} {primary + others:<34} {len(r.flags)}")
    print(f"\nWrote {out / 'scores.csv'}, radar PNGs and reports (start at {out / 'README.md'})")
    return 0


def cmd_template(args: argparse.Namespace) -> int:
    """Write a blank CSV and codebook."""
    fw = load_framework(args.framework)
    out = Path(args.out)
    csv_path, codebook = write_template(fw, out, out.with_name(out.stem + "_codebook.csv"))
    print(f"Wrote {csv_path} and {codebook}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    """Write synthetic responses."""
    fw = load_framework(args.framework)
    path = write_synthetic(fw, args.out, args.seed)
    print(f"Wrote {path} ({HEADER.strip()[:60]}...)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser."""
    load_dotenv()
    p = argparse.ArgumentParser(prog="lamf", description="Logistics Analytics Maturity Framework assessment.")
    p.add_argument("--framework", default=DEFAULT_FRAMEWORK, help=f"framework YAML (default {DEFAULT_FRAMEWORK})")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("assess", help="score a responses CSV")
    a.add_argument("responses")
    a.add_argument("--out", default=os.getenv("LAMF_OUTPUT_DIR", "outputs"))
    a.set_defaults(func=cmd_assess)

    t = sub.add_parser("template", help="write a blank responses CSV and codebook")
    t.add_argument("--out", default="responses_template.csv")
    t.set_defaults(func=cmd_template)

    s = sub.add_parser("synth", help="generate synthetic responses")
    s.add_argument("--out", default="data/synthetic_responses.csv")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_synth)
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FrameworkError, ResponseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
