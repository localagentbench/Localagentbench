#!/usr/bin/env python3
"""Task-bootstrap 95% CI of the v3 total and tie groups, per tier.
Method (lab_stats.bootstrap_ci): for each model, resample tasks with replacement within each category (B=10,000, seed 12345);
a resampled task keeps all its trials (trials pooled); JSON and Schema share the same draw (same 25 tasks). The CI is the
2.5-97.5 percentile of the resampled totals. Tie groups (lab_stats.tie_groups): walking down the ranking, a model joins the
current group while its CI overlaps the CI of the group's top model. import_results.py stores ci_lo/ci_hi/tie_group in
leaderboard.json; this script recomputes them from the raw results and prints a report (no file written).
Later the variance-rerun trials (results/variance, temp 0.7, 5 seeds) will be pooled into the same resampling.
Usage: python3 scripts/bootstrap_ci.py [--in leaderboard.json]"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "scripts"))
import lab_stats as LS
def _lb_path(p):
    import os
    if p: return Path(p)
    if os.environ.get("LAB_LEADERBOARD_OUT"): return Path(os.environ["LAB_LEADERBOARD_OUT"])
    s = ROOT / "site/src/_data/leaderboard.json"
    return s if s.exists() else ROOT / "leaderboard.json"
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp"); a = ap.parse_args()
lb = json.loads(_lb_path(a.inp).read_text())
for t in lb["tiers"]:
    ranked = [e for e in t["entries"] if not e.get("no_native_tools")]
    rows = []
    for e in ranked:
        src = e.get("source") or "gpu"; f = ROOT / ("results/mac-m2" if src == "mac-m2" else "results/gpu") / f"{e['run_id']}.json"
        lo, hi = LS.bootstrap_ci(LS.outcomes(f)); rows.append(dict(model=e["model"], score=e["score"], ci_lo=lo, ci_hi=hi))
    print(f"== {t['id']}")
    for r, g in zip(rows, LS.tie_groups(rows)): print(f"  group {g}  {r['model']:24s} {r['score']:6.2f}  95% CI [{r['ci_lo']}, {r['ci_hi']}]")
