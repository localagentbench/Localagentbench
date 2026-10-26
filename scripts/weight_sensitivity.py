#!/usr/bin/env python3
"""Weight sensitivity of the ranking, per tier (lab_stats.weight_sensitivity).
Samples N=10,000 weight vectors from Dirichlet(c * w0), w0 = current v3 weights (JSON .15, Schema .15, Tool .20,
Multi-step .30, Injection useful&safe .20), concentration c=50 (seed 2026). With c=50 each weight's SD is about
sqrt(w(1-w)/51): ~5 points for the 15-20% weights and ~6.4 points for the 30% weight, i.e. "reasonable people could
pick weights a few points different". For each draw the tier is re-ranked; we report the 5th-95th percentile rank.
Usage: python3 scripts/weight_sensitivity.py [--conc 50] [--n 10000]"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "scripts"))
import lab_stats as LS
ap = argparse.ArgumentParser(); ap.add_argument("--conc", type=float, default=50); ap.add_argument("--n", type=int, default=10000); ap.add_argument("--in", dest="inp"); a = ap.parse_args()
import lab_paths
def _lb_path(p): return Path(p) if p else lab_paths.leaderboard()
lb = json.loads(_lb_path(a.inp).read_text())
for t in lb["tiers"]:
    ranked = [e for e in t["entries"] if not e.get("no_native_tools")]
    ents = [dict(cats=dict(json=e["json"], schema=e["schema"], tool=e["tool"], multistep=e["multistep"], inj_useful_safe=e["inj_useful_safe"])) for e in ranked]
    print(f"== {t['id']}")
    for i, (e, r) in enumerate(zip(ranked, LS.weight_sensitivity(ents, n=a.n, conc=a.conc))):
        print(f"  #{i+1:<2} {e['model']:24s} p5-p95 #{r['p5']}-#{r['p95']}  (min #{r['min']}, max #{r['max']})")
