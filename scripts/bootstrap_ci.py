#!/usr/bin/env python3
"""Task-bootstrap 95% CI of the v3 total and tie groups, per tier.
Method (lab_stats.bootstrap_ci): for each model, resample tasks with replacement within each category (B=10,000, seed 12345);
a resampled task keeps all its trials (trials pooled); JSON and Schema share the same draw (same 25 tasks). The CI is the
2.5-97.5 percentile of the resampled totals. Tie groups (lab_stats.tie_groups): walking down the ranking, a model joins the
current group while its CI overlaps the CI of the group's top model. import_results.py stores ci_lo/ci_hi/tie_group in
leaderboard.json; this script recomputes them from the raw results and prints a report (no file written).
With --pool-variance, the variance-rerun trials (results/variance, temp 0.7, seeds 1001-1005, listed in
site/src/_data/variance.json) are pooled per task with the 3 main-suite trials (8 trials per task for GPU models), and the
pooled CI and tie groups are written into variance.json ("pooled" per model, "pooled_ties" per tier). Ranking order and
the published score are unchanged (main suite); Mac models have no variance trials and keep their main-suite CI.
Usage: python3 scripts/bootstrap_ci.py [--in leaderboard.json] [--pool-variance]"""
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
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp"); ap.add_argument("--pool-variance", action="store_true"); a = ap.parse_args()
VJ = ROOT / "site/src/_data/variance.json"; var = json.loads(VJ.read_text()) if a.pool_variance else None
def pooled(o, model):
    if not var or model not in var["models"]: return o, 0
    files = var["models"][model]["files"]; o = {c: {t: list(v) for t, v in d.items()} for c, d in o.items()}
    for fn in files:
        for c, d in LS.outcomes(ROOT / "results/variance" / fn).items():
            for t, v in d.items(): o[c].setdefault(t, []).extend(v)
    return o, len(files)
pooled_ties = {}
lb = json.loads(_lb_path(a.inp).read_text())
for t in lb["tiers"]:
    ranked = [e for e in t["entries"] if not e.get("no_native_tools")]
    rows = []
    for e in ranked:
        src = e.get("source") or "gpu"; f = ROOT / ("results/mac-m2" if src == "mac-m2" else "results/gpu") / f"{e['run_id']}.json"
        o, nv = pooled(LS.outcomes(f), e["model"]) if src != "mac-m2" else (LS.outcomes(f), 0)
        lo, hi = LS.bootstrap_ci(o); rows.append(dict(model=e["model"], score=e["score"], ci_lo=lo, ci_hi=hi, nv=nv, main_tie=e.get("tie_group")))
    print(f"== {t['id']}")
    gs = LS.tie_groups(rows); sizes = {g: gs.count(g) for g in gs}
    for r, g in zip(rows, gs):
        print(f"  group {g}  {r['model']:24s} {r['score']:6.2f}  95% CI [{r['ci_lo']}, {r['ci_hi']}]  (+{r['nv']} variance trials; main-suite group {r['main_tie']})")
        if var is not None and r["nv"]: var["models"][r["model"]].setdefault("pooled", {}).update({t["id"]: dict(ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], tie_group=g, tied=sizes[g] > 1, trials=3 + r["nv"])})
    if var is not None and any(r["nv"] for r in rows): pooled_ties[t["id"]] = [dict(model=r["model"], tie_group=g, ci_lo=r["ci_lo"], ci_hi=r["ci_hi"]) for r, g in zip(rows, gs)]
if var is not None: var["pooled_ties"] = pooled_ties; VJ.write_text(json.dumps(var, indent=1)); print("wrote pooled CIs and tie groups into", VJ)
