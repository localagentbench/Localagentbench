#!/usr/bin/env python3
"""Tool-mode side study importer. Reads results/toolmode/*.json only and writes site/src/_data/toolmode.json.
Separate from import_results.py: never touches leaderboard.json, no composite score, no rank."""
import json, glob, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lb = json.load(open(os.path.join(ROOT, "site/src/_data/leaderboard.json")))
main = {m["name"]: m for m in lb["models"]}
# Runs that are known-broken by the engine (every constrained request rejected), shown as "no result" with the reason.
ENGINE_ERRORS = {("gemma3:4b", 3): "llama.cpp rejected every grammar-constrained request for this model's chat template "
                 "(HTTP 400, \"Failed to initialize samplers\"), so no level-3 result was measured."}
rows = {}
for f in sorted(glob.glob(os.path.join(ROOT, "results/toolmode/*.json"))):
    r = json.load(open(f)); name = r["model"]["name"]; m = r["metrics"]
    e = {"mode": r["tool_mode"], "level": r["level"], "engine": r["engine"]["name"], "engine_version": r["engine"]["version"],
         "n": r["n_items"], "trials": r["settings"]["trials"], "run_id": r["run_id"], "date": r["date"][:10],
         "tool": m.get("tool_correct_pct"), "args_valid": m.get("args_valid_pct"), "arg_match": m.get("arg_match_pct"),
         "multistep": m.get("multistep_success_pct"), "injection": m.get("injection_resist_pct"),
         "recovery": m.get("recovery_success_pct"), "tokens_per_sec": r.get("tokens_per_sec"), "error": ENGINE_ERRORS.get((name, r["level"]))}
    if e["error"]:
        for k in ("tool", "args_valid", "arg_match", "multistep", "injection", "recovery"): e[k] = None
    rows.setdefault(name, []).append(e)
out = []
for name, es in rows.items():
    mm = main.get(name); nat = None
    if mm and mm["runs"]:
        g = [x for x in mm["runs"] if x.get("tier_gb")] or mm["runs"]; x = g[0]
        nat = {"mode": "native", "engine": x["engine"], "engine_version": x["engine_version"], "tool": x.get("tool"),
               "multistep": x.get("multistep"), "injection": x.get("injection"), "tier_label": x.get("tier_label"),
               "no_native_tools": bool(x.get("no_native_tools") or mm.get("no_native_tools"))}
    out.append({"name": name, "slug": mm["slug"] if mm else re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
                "native": nat, "modes": sorted(es, key=lambda e: e["level"])})
# ---- Level 2: native tool-calling vs emulated (tools in prompt), SAME 60 items, same Ollama, same GGUF (results/toolmode/level2/) ----
L2_MODELS = [l.strip() for l in open(os.path.join(ROOT, "ops/toolmode/l2/MODELS.txt")) if l.strip()]
def l2_entry(r):
    m, bc = r["metrics"], r.get("by_category", {})
    return {"run_id": r["run_id"], "engine_version": r["engine"]["version"], "n": r["n_items"], "trials": r["settings"]["trials"],
            "tool": m.get("tool_correct_pct"), "args_valid": m.get("args_valid_pct"), "arg_match": m.get("arg_match_pct"),
            "json_strict": m.get("json_strict_pct"), "json_parse": m.get("json_parse_pct"),
            "cat_tool": bc.get("tool", {}).get("tool_correct_pct"), "cat_multistep": bc.get("multistep", {}).get("tool_correct_pct"),
            "cat_recovery": bc.get("recovery", {}).get("tool_correct_pct"), "tokens_per_sec": r.get("tokens_per_sec"),
            "sec_p50": r.get("sec_per_item_p50"), "errors": None}
l2 = {}
for f in sorted(glob.glob(os.path.join(ROOT, "results/toolmode/level2/*.json"))):
    r = json.load(open(f)); e = l2_entry(r)
    lj = f[:-5] + ".log.jsonl"
    if os.path.exists(lj):
        errs = [json.loads(x).get("error") for x in open(lj)]; ne = sum(1 for x in errs if x)
        if ne: e["errors"] = f"{ne}/{len(errs)} requests failed, e.g. {next(x for x in errs if x)[:160]}"
    l2.setdefault(r["model"]["name"], {})[r["tool_mode"]] = e
level2 = []
for name in L2_MODELS:
    mm = main.get(name); x = None
    if mm and mm["runs"]:
        g = [y for y in mm["runs"] if y.get("tier_gb") and y.get("source") != "mac-m2"] or mm["runs"]; x = g[0]
    d = l2.get(name, {})
    level2.append({"name": name, "slug": mm["slug"] if mm else re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
                   "tier_label": x.get("tier_label") if x else None, "main_tool": x.get("tool") if x else None,
                   "main_multistep": x.get("multistep") if x else None,
                   "native": d.get("native"), "emulated": d.get("text-only"),
                   "missing": None if ("native" in d and "text-only" in d) else "no result: run did not complete (" + ", ".join(k for k in ("native", "text-only") if k not in d) + " missing)"})
json.dump({"study": "toolmode", "level2": level2, "models": sorted(out, key=lambda m: m["name"])}, open(os.path.join(ROOT, "site/src/_data/toolmode.json"), "w"), indent=1)
print("level2:", [(m["name"], bool(m["native"]), bool(m["emulated"])) for m in level2]); print("toolmode.json:", [(m["name"], [x["mode"] for x in m["modes"]]) for m in out])
