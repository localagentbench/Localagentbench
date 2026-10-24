#!/usr/bin/env python3
"""Convert runner output into leaderboard.json (the site's data file; see --out for the default path).

Usage: python3 scripts/import_results.py [--src gpu --src mac-m2 | --src <dir>] [--merge]
                                        [--out PATH | env LAB_LEADERBOARD_OUT] [--allow-partial]

Sources (see SOURCES below; default = all of them, so a plain re-run includes GPU *and* Mac data):
  gpu     results/gpu     -> GPU tiers 8/16/24 GB VRAM, by hardware.tier_gb. Runs tagged with a
                             non-GPU hardware.slug (e.g. mac-m2) are skipped here.
  mac-m2  results/mac-m2  -> one tier "Apple M2 16GB unified memory, macOS" (hardware.slug == mac-m2),
                             whatever the per-model tier_gb says.
--merge: only rebuild the tiers/runs of the given --src sources and keep every other tier and model run
         exactly as it is in the existing --out file (e.g. `--src mac-m2 --merge` adds the Mac tier
         without re-reading results/gpu).
Dev/CPU runs (x_channel=dev, tier 0, gpu none), subset runs and runs with fewer than 3 trials are skipped
unless --allow-partial is given (dev never imported). Best run per (model, quant, engine) is kept per tier,
sorted by score. Overrides in scripts/overrides.json apply to every source."""
import argparse, json, os, re, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GPU_TIERS = [(8, "8gb", "Fits in 8 GB VRAM"), (16, "16gb", "Fits in 16 GB VRAM"), (24, "24gb", "Fits in 24 GB VRAM")]
SOURCES = {
    "gpu": {"dir": "results/gpu", "kind": "gpu"},
    "mac-m2": {"dir": "results/mac-m2", "kind": "fixed", "hw_slug": "mac-m2", "tier_id": "mac-m2",
               "label": "Apple M2 16GB unified memory, macOS", "hardware": "mac",
               "note": "Same suite, Hermes Agent harness, Q4_K_M quants and Ollama 0.40.6 (native macOS, Metal) as the GPU tiers, "
                       "run on a rented Apple M2 Mac mini with 16 GB unified memory (about 11.8 GB usable by the GPU). "
                       "tok/s is single-stream decode speed (256-token generation, median of 3).",
               "skipped": [
                   {"model": "phi4:14b", "reason": "ran out of time: a 14B model takes about 5 h on this Mac (time cap)",
                    "tool_support": "no native tool support in Ollama"},
                   {"model": "mistral-small:24b", "reason": "doesn't fit in 16 GB unified memory (Q4_K_M weights about 14 GB)",
                    "tool_support": "native"},
                   {"model": "qwen3:30b-a3b", "reason": "doesn't fit in 16 GB unified memory (Q4_K_M weights about 18-19 GB)",
                    "tool_support": "native"}]},
}
W = {"json_valid_pct": .15, "schema_ok_pct": .15, "tool_correct_pct": .20, "multistep_success_pct": .30, "injection_resist_pct": .20}
SCORING_VERSION = "2.1"   # 2026-10-23: injection sub-score weighted by tool-call rate (see methodology)
                          # 2.1: tool-call rate computed only over tool-requiring tasks (tool, multi, inj)
TOOL_CATS = ("tool", "multi", "inj")

def tool_call_rate(result_json):
    """Fraction of tool-requiring task trials (categories tool, multi, inj; all trials) in which the harness executed at least one tool call
    parsed from the model's output, i.e. a non-empty `trace` in the run's .log.jsonl. None if the log is missing."""
    log = result_json.with_name(result_json.name[:-5] + ".log.jsonl")
    if not log.exists(): return None
    n = hit = 0
    for line in log.read_text().splitlines():
        if not line.strip(): continue
        x = json.loads(line)
        if str(x.get("task", "")).rsplit("-", 1)[0] not in TOOL_CATS: continue
        n += 1; hit += bool(x.get("trace"))
    return round(hit / n, 4) if n else None

def score_v2(met, rate):
    """v2 total: like v1 but injection_resist_pct is multiplied by the tool-call rate, so a model that never calls
    tools gets ~0 injection credit instead of a free 100."""
    return round(sum(met[k] * w * (rate if k == "injection_resist_pct" else 1) for k, w in W.items()), 2)

slug = lambda s: re.sub(r"[^a-z0-9.]+", "-", s.lower()).strip("-")

def source_of(x):
    """Source name of an existing tier or run in leaderboard.json (older GPU data carries no tag)."""
    return x.get("source") or "gpu"

def resolve_src(s):
    if s in SOURCES: return s, dict(SOURCES[s], dir=ROOT / SOURCES[s]["dir"])
    p = Path(s).resolve()
    for n, c in SOURCES.items():
        if (ROOT / c["dir"]).resolve() == p: return n, dict(c, dir=p)
    return p.name, {"dir": p, "kind": "gpu"}   # unknown dir: legacy behaviour (GPU tiers)

def load_runs(name, cfg, a, no_tools):
    if "dev" in Path(cfg["dir"]).resolve().parts: sys.exit("refusing to import dev results")
    allf = sorted(Path(cfg["dir"]).glob("*.json")) if Path(cfg["dir"]).exists() else []
    tps = {}
    for f in allf:
        if f.name.endswith(".tps.json"):
            try:
                t = json.loads(f.read_text())
                if cfg["kind"] == "fixed" and t.get("hardware") not in (None, cfg["hw_slug"]): continue
                tps[t["model"]] = t.get("tokens_per_sec_decode")
            except Exception as e: print(f"  [{name}] bad tps file {f.name}: {e}")
    files = [f for f in allf if not f.name.endswith(".tps.json")]
    gpu_labels = {t: l for t, _, l in GPU_TIERS}
    runs, skipped = [], []
    for f in files:
        try: r = json.loads(f.read_text())
        except Exception as e: skipped.append((f.name, f"bad json: {e}")); continue
        m, hw, met = r.get("model", {}), r.get("hardware", {}), r.get("metrics", {})
        tier = hw.get("tier_gb")
        if cfg["kind"] == "gpu":
            why = ("dev channel" if r.get("x_channel") == "dev" else
                   f"hardware {hw['slug']} belongs to its own source" if hw.get("slug") else
                   "no GPU" if not tier or str(hw.get("gpu", "none")).lower() == "none" else
                   f"tier {tier} not in 8/16/24" if tier not in gpu_labels else None)
            tier_id = dict((t, i) for t, i, _ in GPU_TIERS).get(tier); label = gpu_labels.get(tier)
        else:
            why = ("dev channel" if r.get("x_channel") == "dev" else
                   f"hardware slug {hw.get('slug')!r} != {cfg['hw_slug']}" if hw.get("slug") != cfg["hw_slug"] else None)
            tier_id, label = cfg["tier_id"], cfg["label"]
        if not why and not all(k in met for k in W): why = "missing metrics"
        if not why and not a.allow_partial:
            if r.get("x_subset"): why = "subset run"
            elif (r.get("n_trials") or 0) < 3: why = "fewer than 3 trials"
        if why: skipped.append((f.name, why)); continue
        score_v1 = r.get("score_total")
        if score_v1 is None: score_v1 = round(sum(met[k] * w for k, w in W.items()), 2)
        rate = tool_call_rate(f)
        if rate is None and not a.allow_partial: skipped.append((f.name, "no .log.jsonl (tool-call rate unknown)")); continue
        score = score_v2(met, rate if rate is not None else 0.0)
        # Hermes Agent sends no temperature/seed to Ollama, so settings.temp/seed in result files were never applied -> null + "sampling"
        s = r.get("settings", {}); eng = r.get("engine", {}); ag = r.get("agent", {})
        e = {"run_id": r.get("run_id"), "date": r.get("date"), "model": m.get("name"), "model_slug": slug(m.get("name", "")),
             "family": m.get("family"), "params_b": m.get("params_b"), "quant": m.get("quant"), "gguf_sha256": m.get("gguf_sha256"),
             "engine": eng.get("name"), "engine_version": eng.get("version"), "agent": ag.get("name"), "agent_version": ag.get("version"),
             "tier_gb": tier, "tier_label": label, "gpu": hw.get("gpu"), "vram_gb": hw.get("vram_gb"), "os": hw.get("os"),
             "temp": None, "ctx": s.get("ctx"), "seed": None,
             "sampling": "ollama-default (harness sets no temperature/seed)", "n_trials": r.get("n_trials"),
             "score": score, "json": met["json_valid_pct"], "schema": met["schema_ok_pct"], "tool": met["tool_correct_pct"],
             "multistep": met["multistep_success_pct"], "injection": met["injection_resist_pct"],
             "tool_call_rate": rate, "injection_weighted": round(met["injection_resist_pct"] * (rate or 0), 2),
             "score_v1": score_v1, "scoring_version": SCORING_VERSION,
             "tokens_per_sec": met.get("tokens_per_sec"), "p50_s": met.get("sec_per_task_p50"), "p95_s": met.get("sec_per_task_p95"),
             "verified": bool(r.get("verified")), "sponsor_disclosure": r.get("sponsor_disclosure", "none")}
        if e["tokens_per_sec"] is None and tps.get(e["model"]) is not None: e["tokens_per_sec"] = tps[e["model"]]
        e["no_native_tools"] = e["model"] in no_tools
        e["note"] = no_tools.get(e["model"])
        if e["no_native_tools"]: e["raw_score"], e["score"] = e["score"], None
        if name != "gpu": e["source"], e["tier_id"], e["hardware_label"] = name, tier_id, label
        e["_tier_id"] = tier_id
        runs.append(e)
    return runs, skipped, len(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", help="source name (%s) or results dir; repeatable; default: all" % ", ".join(SOURCES))
    # The site (11ty) lives in a separate, private repo. Default output: $LAB_LEADERBOARD_OUT, else
    # site/src/_data/leaderboard.json if a site/ checkout sits next to scripts/, else ./leaderboard.json in the repo root.
    default_out = os.environ.get("LAB_LEADERBOARD_OUT") or (ROOT / "site/src/_data/leaderboard.json" if (ROOT / "site").is_dir() else ROOT / "leaderboard.json")
    ap.add_argument("--out", default=default_out, type=Path, help="output JSON (default: $LAB_LEADERBOARD_OUT, site/src/_data/leaderboard.json if site/ exists, else leaderboard.json)")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--merge", action="store_true", help="keep tiers/runs of sources not listed in --src from the existing --out")
    ap.add_argument("--overrides", default=ROOT / "scripts/overrides.json", type=Path)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ov = json.loads(a.overrides.read_text()) if a.overrides.exists() else {}
    no_tools = ov.get("no_native_tools", {})
    srcs = dict(resolve_src(s) for s in (a.src or list(SOURCES)))
    old = json.loads(a.out.read_text()) if a.merge and a.out.exists() else {"tiers": [], "models": []}

    # tier skeleton, in display order: GPU tiers, then fixed-hardware tiers in SOURCES order
    out_tiers, new_by_tier = [], {}
    order = [n for n in SOURCES if n in srcs] + [n for n in srcs if n not in SOURCES]
    if a.merge: order = list(dict.fromkeys([n for n in SOURCES] + [source_of(t) for t in old["tiers"]] + order))
    total, models_new = 0, {}
    for n in order:
        if n in srcs:
            cfg = srcs[n]
            runs, skipped, nf = load_runs(n, cfg, a, no_tools); total += len(runs)
            print(f"[{n}] imported {len(runs)} run(s) from {nf} file(s) in {cfg['dir']}")
            for f, w in skipped: print(f"  [{n}] skipped {f}: {w}")
            if cfg["kind"] == "gpu":
                specs = [{"id": i, "label": l, "vram_gb": t} for t, i, l in GPU_TIERS]
            else:
                specs = [{"id": cfg["tier_id"], "label": cfg["label"], "vram_gb": None, "source": n, "hardware": cfg.get("hardware"),
                          "note": cfg.get("note"), "skipped_models": cfg.get("skipped", [])}]
            for sp in specs:
                best = {}
                for e in runs:
                    if e["_tier_id"] != sp["id"]: continue
                    k = (e["model"], e["quant"], e["engine"])
                    if k not in best or (e["score"] or -1) > (best[k]["score"] or -1): best[k] = e
                sp["entries"] = sorted(best.values(), key=lambda e: (e["no_native_tools"], -(e["score"] or 0)))
                out_tiers.append(sp)
            for e in runs: models_new.setdefault(n, []).append(e)
        else:
            out_tiers += [t for t in old["tiers"] if source_of(t) == n]   # kept verbatim
    tier_pos = {t["id"]: i for i, t in enumerate(out_tiers)}

    models = {}
    for mm in old["models"]:   # merge mode: keep runs of untouched sources verbatim
        kept = [r for r in mm["runs"] if source_of(r) not in srcs]
        if kept:
            models[mm["slug"]] = dict(mm, runs=kept)
            if not any(r.get("no_native_tools") for r in kept): models[mm["slug"]].pop("no_native_tools", None); models[mm["slug"]].pop("note", None)
    for n in order:
        for e in models_new.get(n, []):
            mm = models.setdefault(e["model_slug"], {"slug": e["model_slug"], "name": e["model"], "family": e["family"], "params_b": e["params_b"], "runs": []})
            if e["no_native_tools"]: mm["no_native_tools"], mm["note"] = True, e["note"]
            mm["runs"].append(e)
    run_tier = lambda r: r.get("tier_id") or {t: i for t, i, _ in GPU_TIERS}.get(r.get("tier_gb"))
    for mm in models.values(): mm["runs"].sort(key=lambda e: (tier_pos.get(run_tier(e), 99), -(e["score"] or 0)))
    for t in out_tiers:
        for e in t["entries"]: e.pop("_tier_id", None)
    for mm in models.values():
        for e in mm["runs"]: e.pop("_tier_id", None)
    sources = list(dict.fromkeys([str(srcs[n]["dir"].relative_to(ROOT)) if n in srcs and srcs[n]["dir"].is_relative_to(ROOT) else
                                  SOURCES.get(n, {}).get("dir", n) for n in order if any(source_of(t) == n for t in out_tiers)]))
    data = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if models else None,
            "suite_version": "0.1", "scoring_version": SCORING_VERSION, "source": ", ".join(sources), "tiers": out_tiers, "models": sorted(models.values(), key=lambda m: m["name"])}
    a.out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {a.out}: {len(out_tiers)} tier(s), {len(models)} model(s), {total} newly imported run(s)" + (" (merge)" if a.merge else ""))

if __name__ == "__main__": main()
