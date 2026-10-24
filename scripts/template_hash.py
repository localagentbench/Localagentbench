#!/usr/bin/env python3
"""Chat-template hash comparison -> results/template_hash.json (consumed by import_results.py).

For each leaderboard model, compares the GGUF chat template (tokenizer.chat_template read from the GGUF header of the
exact registry blob we tested) with the provider's upstream reference template on Hugging Face, after whitespace
normalisation (every whitespace run -> one space, trimmed), using SHA-256. Inputs are the files already fetched by the
2026-10-23 template-integrity check (results/template_integrity_2026-10-23_raw/{gguf,hf}/ and ollama.json); nothing is
downloaded. Labels are only: "matches" (normalised hashes equal), "differs" (both exist, hashes differ),
"no reference" (no upstream Jinja reference available, e.g. gated repo or the provider ships only a system prompt).
A difference is NOT a safety verdict: legitimate re-uploads and Ollama conversions often carry fixed or older templates,
and Ollama may render prompts with a built-in Go renderer instead of the GGUF Jinja (recorded as `ollama_renderer`).

Usage: python3 scripts/template_hash.py [--raw DIR] [--out PATH]"""
import argparse, hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def norm(t): return re.sub(r"\s+", " ", t).strip()
def h(t): return hashlib.sha256(norm(t).encode()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results/template_integrity_2026-10-23_raw")
    ap.add_argument("--out", type=Path, default=ROOT / "results/template_hash.json")
    a = ap.parse_args()
    ol = json.loads((a.raw / "ollama.json").read_text())
    hfmeta = json.loads((a.raw / "hf.json").read_text())
    out = {"method": "sha256 of whitespace-normalised template (re.sub(r'\\s+',' ').strip()); GGUF tokenizer.chat_template vs upstream HF chat template",
           "source": str(a.raw.relative_to(ROOT)), "labels": ["matches", "differs", "no reference"], "models": {}}
    for g in sorted((a.raw / "gguf").glob("*.json")):
        slug = g.stem; gt = json.loads(g.read_text()).get("tokenizer.chat_template")
        ref = a.raw / "hf" / f"{slug}.jinja"; o = ol.get(slug, {})
        rt = ref.read_text() if ref.exists() else None
        e = {"model": o.get("name"), "gguf_template_sha256": h(gt) if gt else None,
             "reference_sha256": h(rt) if rt else None, "reference_source": (hfmeta.get(slug) or {}).get("src") if rt else None,
             "ollama_template_layer": o.get("template_digest"), "ollama_renderer": o.get("renderer")}
        e["label"] = "no reference" if not (gt and rt) else ("matches" if e["gguf_template_sha256"] == e["reference_sha256"] else "differs")
        e["detail"] = ("GGUF has no Jinja chat template (prompt built by Ollama's Go template), so there is nothing to hash-compare" if not gt else
                       "no upstream Jinja reference available (gated repo or provider publishes no Jinja template)" if not rt else
                       "GGUF template equals the upstream reference after whitespace normalisation" if e["label"] == "matches" else
                       "GGUF template differs from the current upstream reference (can be an older revision or a legitimate fix)")
        if e["ollama_renderer"]: e["detail"] += f"; Ollama renders this model with its built-in Go renderer '{e['ollama_renderer']}', not the GGUF Jinja"
        out["models"][slug] = e
    c = {}
    for e in out["models"].values(): c[e["label"]] = c.get(e["label"], 0) + 1
    out["counts"] = c
    a.out.write_text(json.dumps(out, indent=2) + "\n"); print(f"wrote {a.out}: {c}")
if __name__ == "__main__": main()
