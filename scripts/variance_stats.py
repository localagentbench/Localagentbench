#!/usr/bin/env python3
"""Variance rerun stats -> data/variance.json (see lab_paths.py; env $LAB_DATA_DIR). Reads results/variance/*.json (single-trial runs, temp 0.7,
seeds 1001-1005), scores each trial with the same scoring as the leaderboard (scripts/import_results.py, v2.1), and computes
per-model mean, SD and 95% CI (Student t, n-1 df) of the total and of each category. Only models with >=2 completed trials."""
import json, math, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "scripts"))
import import_results as IR
import lab_stats as LS
import lab_paths
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
CATS = {"json": "json_valid_pct", "schema": "schema_ok_pct", "tool": "tool_correct_pct", "multistep": "multistep_success_pct", "injection": "injection_resist_pct"}
def ci(xs):
    n = len(xs); m = st.mean(xs); sd = st.stdev(xs) if n > 1 else 0.0
    h = T975.get(n - 1, 1.96) * sd / math.sqrt(n) if n > 1 else None
    return {"mean": round(m, 2), "sd": round(sd, 2), "ci95": round(h, 2) if h is not None else None, "lo": round(m - h, 2) if h is not None else None, "hi": round(m + h, 2) if h is not None else None, "n": n}
runs = {}
for f in sorted((ROOT / "results/variance").glob("*.json")):
    if f.name.endswith(".tps.json"): continue
    r = json.loads(f.read_text()); s = r.get("settings", {})
    if r.get("n_trials") != 1 or s.get("temp") != 0.7 or r.get("x_subset"): continue
    rate = IR.tool_call_rate(f)
    if rate is None: continue
    met = r["metrics"]; runs.setdefault(r["model"]["name"], {})[s["seed"]] = dict(
        score=LS.score_v3(LS.cat_pct(LS.outcomes(f))), score_v21=IR.score_v2(met, rate), rate=rate, sha=r["model"].get("gguf_sha256"), file=f.name, **{k: met[v] for k, v in CATS.items()})
lb = json.loads(lab_paths.leaderboard().read_text())
main = {e["model"]: e for t in lb["tiers"] if t["id"] in ("8gb", "16gb", "24gb") for e in t["entries"]}
out = {"temp": 0.7, "seeds": [1001, 1002, 1003, 1004, 1005], "scoring_version": IR.SCORING_VERSION, "models": {}}
for m, by in sorted(runs.items()):
    tr = [by[k] for k in sorted(by)]
    if len(tr) < 2: continue
    e = {"seeds": sorted(by), "trial_scores": [t["score"] for t in tr], "score": ci([t["score"] for t in tr]),
         "categories": {c: ci([t[c] for t in tr]) for c in CATS}, "tool_call_rate": ci([t["rate"] for t in tr]),
         "sha_matches_main": all(t["sha"] == main.get(m, {}).get("gguf_sha256") for t in tr), "files": [t["file"] for t in tr]}
    out["models"][m] = e
VJ = lab_paths.data("variance.json"); VJ.parent.mkdir(parents=True, exist_ok=True); VJ.write_text(json.dumps(out, indent=1))
print(f"wrote {VJ}: {len(out['models'])} models")
for m, e in out["models"].items(): print(f"  {m:24s} n={e['score']['n']} mean={e['score']['mean']} ±{e['score']['ci95']} sha_ok={e['sha_matches_main']} trials={e['trial_scores']}")
