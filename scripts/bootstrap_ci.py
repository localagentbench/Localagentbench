#!/usr/bin/env python3
"""Task-bootstrap 95% CI of the v3 total and tie groups, per tier.
Method (lab_stats.bootstrap_ci): for each model, resample tasks with replacement within each category (B=10,000, seed 12345);
a resampled task keeps all its trials (trials pooled); JSON and Schema share the same draw (same 25 tasks). The CI is the
2.5-97.5 percentile of the resampled totals. Tie groups (lab_stats.tie_groups): walking down the ranking, a model joins the
current group while its CI overlaps the CI of the group's top model. import_results.py stores ci_lo/ci_hi/tie_group in
leaderboard.json; this script recomputes them from the raw results and prints a report (no file written).
With --pool-variance, the variance-rerun trials (results/variance, temp 0.7, seeds 1001-1005, listed in
data/variance.json) are pooled per task with the 3 main-suite trials (8 trials per task for GPU models), and the
pooled CI and tie groups are written into variance.json ("pooled" per model, "pooled_ties" per tier). Ranking order and
the published score are unchanged (main suite); Mac models have no variance trials and keep their main-suite CI.
Because the CI is pooled over 3+5 trials while the score is the 3-trial main-suite mean, the CI need not be centred on the
published score (the site labels it as pooled and flags scores outside or near the edge of it).
Missing or unreadable variance trial files (listed in variance.json but absent from results/variance, or with no readable
.log.jsonl) are skipped with a warning listing them; a model left with no readable variance trial falls back to its
main-suite (suite-only) CI and tie group, and any stale "pooled" entry for that tier is removed. --strict fails instead.
Inputs/outputs default to data/leaderboard.json and data/variance.json (see lab_paths.py; env $LAB_DATA_DIR).
Usage: python3 scripts/bootstrap_ci.py [--in leaderboard.json] [--pool-variance [--strict] [--variance variance.json]]"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "scripts"))
import lab_stats as LS
import lab_paths
def _lb_path(p): return Path(p) if p else lab_paths.leaderboard()
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp"); ap.add_argument("--pool-variance", action="store_true")
ap.add_argument("--strict", action="store_true", help="with --pool-variance: fail on a missing/unreadable variance trial file instead of skipping it")
ap.add_argument("--variance", default=None, help="with --pool-variance: variance.json to read and update (default: data/variance.json)")
a = ap.parse_args()
if a.strict and not a.pool_variance: ap.error("--strict only applies with --pool-variance")
VJ = Path(a.variance) if a.variance else lab_paths.data("variance.json"); var = json.loads(VJ.read_text()) if a.pool_variance else None
def _outcomes(f):
    """LS.outcomes() with a reason instead of None/exception (missing result or log, or unparsable log)."""
    if not Path(f).exists(): return None, "file not found"
    try: o = LS.outcomes(f)
    except (OSError, ValueError, KeyError, TypeError) as ex: return None, f"unreadable ({type(ex).__name__}: {ex})"
    return (o, None) if o is not None else (None, "no readable .log.jsonl next to it")
skipped = {}; fallback = []
def pooled(o, model):
    if not var or model not in var["models"]: return o, 0
    files = var["models"][model].get("files") or []; o = {c: {t: list(v) for t, v in d.items()} for c, d in o.items()}; n = 0
    for fn in files:
        vo, why = _outcomes(ROOT / "results/variance" / fn)
        if vo is None:
            if a.strict: sys.exit(f"error: --strict: variance trial for {model} listed in {VJ.name} is {why}: results/variance/{fn}")
            skipped.setdefault(model, []).append(f"{fn} ({why})"); continue
        for c, d in vo.items():
            for t, v in d.items(): o[c].setdefault(t, []).extend(v)
        n += 1
    if files and not n: fallback.append(model)
    return o, n
pooled_ties = {}
lb = json.loads(_lb_path(a.inp).read_text())
for t in lb["tiers"]:
    ranked = [e for e in t["entries"] if not e.get("no_native_tools")]
    rows = []
    for e in ranked:
        src = e.get("source") or "gpu"; f = ROOT / ("results/mac-m2" if src == "mac-m2" else "results/gpu") / f"{e['run_id']}.json"
        mo, why = _outcomes(f)
        if mo is None: sys.exit(f"error: main-suite result for {e['model']} ({t['id']}) is {why}: {f.relative_to(ROOT)}")
        o, nv = pooled(mo, e["model"]) if src != "mac-m2" else (mo, 0)
        lo, hi = LS.bootstrap_ci(o); rows.append(dict(model=e["model"], score=e["score"], ci_lo=lo, ci_hi=hi, nv=nv, main_tie=e.get("tie_group"), fb=src != "mac-m2" and e["model"] in fallback))
    print(f"== {t['id']}")
    gs = LS.tie_groups(rows); sizes = {g: gs.count(g) for g in gs}
    for r, g in zip(rows, gs):
        fb = " [suite-only fallback: no readable variance trials]" if r["fb"] else ""
        print(f"  group {g}  {r['model']:24s} {r['score']:6.2f}  95% CI [{r['ci_lo']}, {r['ci_hi']}]  (+{r['nv']} variance trials; main-suite group {r['main_tie']}){fb}")
        vm = var["models"].get(r["model"]) if var is not None else None
        if vm is not None and r["nv"]: vm.setdefault("pooled", {}).update({t["id"]: dict(ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], tie_group=g, tied=sizes[g] > 1, trials=3 + r["nv"])})
        elif vm is not None and "pooled" in vm: vm["pooled"].pop(t["id"], None)  # fallback: drop a stale pooled CI so the site shows the suite-only one
    if var is not None and any(r["nv"] for r in rows): pooled_ties[t["id"]] = [dict(model=r["model"], tie_group=g, ci_lo=r["ci_lo"], ci_hi=r["ci_hi"]) for r, g in zip(rows, gs)]
if var is not None:
    for m, vm in var["models"].items():
        if "pooled" in vm and not vm["pooled"]: del vm["pooled"]
        if skipped.get(m): vm["pooled_skipped_files"] = skipped[m]
        else: vm.pop("pooled_skipped_files", None)
    if skipped:
        print(f"WARNING: skipped {sum(map(len, skipped.values()))} missing/unreadable variance trial file(s) (use --strict to fail instead):", file=sys.stderr)
        for m, fs in skipped.items():
            for x in fs: print(f"  {m}: results/variance/{x}", file=sys.stderr)
    if fallback: print("WARNING: no readable variance trials, fell back to the suite-only (main-suite) CI for: " + ", ".join(sorted(set(fallback))), file=sys.stderr)
    var["pooled_ties"] = pooled_ties; VJ.write_text(json.dumps(var, indent=1)); print("wrote pooled CIs and tie groups into", VJ)
