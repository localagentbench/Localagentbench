#!/usr/bin/env python3
"""Why do some models score ~0 on tool selection? Classifies every tool-needed task response in the EXISTING
main-suite traces (no GPU, no new runs) and checks each model's Ollama chat template.

Inputs (read-only):
  results/{gpu,mac-m2}/<run>.log.jsonl   main-suite traces (Hermes Agent harness, MCP mock tools)
  results/template_integrity_2026-10-23_raw/{ol,hf}/  Ollama registry templates and provider (HF) Jinja
  results/template_integrity_2026-10-23_raw/ollama.json  Ollama renderer / parser per model
  results/toolmode/level2/*.json          tool-mode study (native vs emulated), when the model was included
Outputs:
  research/tool_failure_counts.json  (all counts + snippets)
  markdown tables on stdout

What the trace can and cannot tell us: in the Hermes harness `answer` is Hermes' final stdout (quiet mode) and `trace`
is what the MCP server actually executed. An empty trace means no tool call reached the tool server. Hermes does not
print hidden reasoning, so <think> content and token counts are NOT in the traces; only wall-clock seconds are.

Usage: python3 scripts/tool_failure_analysis.py [--json OUT]"""
import argparse, glob, json, re, statistics, sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))
from lab.tasks import all_tasks  # noqa: E402
from lab.tools import TOOL_SPECS  # noqa: E402

FAILING = ["deepseek-r1:8b", "deepseek-r1:14b", "mistral:7b", "qwen2.5-coder:7b", "qwen2.5-coder:14b"]
REFERENCE = ["qwen2.5:7b", "qwen2.5:14b"]  # same harness, tool calls work: latency baseline + sanity check
TOOL_CATS = {"tool": "tool", "multistep": "multistep", "recovery": "recovery", "injection": "injection"}
REAL = {n for n, _, _ in TOOL_SPECS}
TASKS = {t["id"]: t for t in all_tasks()}
RAW = ROOT / "results/template_integrity_2026-10-23_raw"

CATEGORIES = OrderedDict([
    ("empty", "Empty answer (nothing printed)"),
    ("think_leak", "Reasoning text (<think>) in the answer"),
    ("harness_error", "Engine/harness error instead of an answer"),
    ("text_call_real_tool", "Tool call written as text in content, real tool name (not parsed by Ollama)"),
    ("text_call_unknown_tool", "Call-like text (JSON, tags, function syntax, shell) with a tool name that does not exist"),
    ("refuse_or_clarify", "No call; refuses / says it has no access / asks a question"),
    ("plain_answer", "No call; answers in prose (often with an invented result)"),
    ("other", "Other"),
])

def norm_name(n):
    n = (n or "").strip().strip("`'\"")
    for p in ("mcp__lab__", "mcp_lab_", "functions.", "lab."):
        if n.startswith(p): n = n[len(p):]
    return n

def json_objects(text):
    """All parseable top-level JSON objects/arrays found anywhere in text."""
    out, dec, i = [], json.JSONDecoder(), 0
    while i < len(text):
        if text[i] in "{[":
            try:
                obj, end = dec.raw_decode(text, i); out.append(obj); i = end; continue
            except ValueError: pass
        i += 1
    return out

def calls_from_json(obj):
    items = obj if isinstance(obj, list) else [obj]
    res = []
    for it in items:
        if isinstance(it, dict):
            if isinstance(it.get("steps"), list): res += calls_from_json(it["steps"]); continue
            name = it.get("name") or it.get("tool") or it.get("function_name") or it.get("function") or it.get("command") or it.get("tool_name")
            if isinstance(name, dict): name = name.get("name")
            if isinstance(name, str) and any(k in it for k in ("arguments", "args", "parameters", "params", "input")) or \
               (isinstance(name, str) and ("tool" in it or "command" in it)):
                res.append(name)
    return res

TAG_RE = re.compile(r"<\s*tool[ _]?call\s*>|<\s*tool\s*>|<\s*(?:web_search|search|calculator|function_call)\s*>|\[TOOL_CALLS\]|<\s*function[ =]|<\|python_tag\|>", re.I)
LOOSE_NAME_RE = re.compile(r'"(?:name|tool)"\s*:\s*"?([A-Za-z_][\w.]*)')
MCP_CALL_RE = re.compile(r"\b(mcp_{1,2}lab_{1,2}[a-z_]+)\s*(?:\(|\{|,|\s+\{|\s+[A-Za-z_]+=)")
FUNC_CALL_RE = re.compile(r"\b([a-z][a-z0-9_]{2,})\(\s*(?:[a-z_]+\s*=|['\"\d])")
FENCE_RE = re.compile(r"```(\w*)\n(.*?)(?:```|$)", re.S)
REFUSE_RE = re.compile(r"\b(i(?:'m| am)? (?:unable|not able)|i (?:can(?:no|')t|cannot|do not|don't) (?:have|access|directly|send|open|read|fetch|browse|perform|provide|retrieve|check|execute|run|schedule|create|look)|(?:can(?:no|')t|cannot) (?:access|directly|send|open|proceed)|don't have (?:access|the ability|a tool|real-time)|do not have (?:access|the ability|a tool|real-time)|no (?:access|tool) (?:to|for)|not have the capability|unable to (?:access|fetch|send|open|retrieve|browse|provide))", re.I)
CLARIFY_RE = re.compile(r"(please (?:provide|specify|confirm|clarify|let me know|share)|could you (?:please )?(?:provide|specify|confirm|clarify|tell)|would you like|do you want me|can you (?:provide|confirm|tell))", re.I)
FAKE_USER_RE = re.compile(r"\[\s*out-of-band user message", re.I)
CLAIM_RE = re.compile(r"(\bresult(?:s)?\s*(?:was|is)?\s*:|\boutput(?: from the tool)?\s*:|\bresponse\s*:|here (?:is|are) (?:the|its) (?:content|result|output)|\bi(?:'ve| have)? (?:used|executed|ran|called|checked|read|sent)\b|\bexecuted\b|according to the (?:search|tool|latest)|email (?:has been |was )?sent|\bi(?:'ve| have)? (?:created|scheduled|added|converted|translated|looked up|searched)\b|\bhere(?:'s| is) (?:its|the file'?s?) content)", re.I)

def classify(task, ans):
    a = ans or ""
    s = a.strip()
    flags = set()
    if FAKE_USER_RE.search(a): flags.add("fake_user_turn")
    if CLAIM_RE.search(a): flags.add("claims_execution_or_result")
    if re.search(r"mcp_{1,2}lab_{1,2}(%s)\b" % "|".join(REAL), a): flags.add("mentions_exact_exposed_name")
    if "```" in a: flags.add("code_fence")
    if re.match(r'\s*\[?\s*\{\s*"(?:name|function_name)"\s*:', a): flags.add("starts_with_bare_json_call")  # what a tag-less parser fallback would need
    if TAG_RE.search(a): flags.add("wrapper_tag")
    names = []
    js = json_objects(a)
    for o in js: names += calls_from_json(o)
    if names: flags.add("json_call")
    if not names:
        names += LOOSE_NAME_RE.findall(a) if re.search(r'"(?:arguments|args|parameters)"', a) else []
        if names: flags.add("broken_json_call")
    if not names: names += MCP_CALL_RE.findall(a)
    if not names and TAG_RE.search(a):
        m = re.search(r"(?:<\s*tool[ _]?call\s*>|<\s*tool\s*>)\s*([A-Za-z_][\w.]*)", a)
        names.append(m.group(1) if m else "?")
    if not names:
        for lang, body in FENCE_RE.findall(a):
            names += FUNC_CALL_RE.findall(body)
            if not names and (lang in ("bash", "sh", "shell", "powershell", "console") or re.match(r"\s*[a-z][\w.-]*\s+(?:-|'|\"|[A-Za-z0-9/])", body)):
                first = body.strip().split()
                if first: names.append("shell:" + first[0]); flags.add("shell_command")
    if not names:
        fc = [n for n in FUNC_CALL_RE.findall(a) if n not in ("e.g", "i.e")]
        if fc: names += fc; flags.add("func_syntax")
    if not names:  # `tool_name args` in single backticks / "tool: name(...)" / bare one-liner command
        m = re.search(r"`(mcp_{1,2}lab_{1,2}[a-z_]+)[ ,]", a) or re.search(r"\btool(?:\s*call)?\s*:\s*([a-z_]+)", a, re.I) \
            or re.fullmatch(r"([a-z][a-z0-9_]+)(?:\s+(?:\"|'|[A-Z0-9]|-)\S*.*)", s)
        if m: names.append(m.group(1)); flags.add("pseudo_call")
    names = [n[6:] if n.startswith("shell:") and norm_name(n[6:]) in REAL else n for n in names]  # `mcp__lab__x args` in a fence
    real_names = [norm_name(n) for n in names if norm_name(n) in REAL]
    if not s: cat = "empty"
    elif "<think>" in a or "</think>" in a: cat = "think_leak"
    elif re.search(r"HTTP \d{3}|Provider said|rejected this request|does not support tools", a): cat = "harness_error"
    elif names and real_names: cat = "text_call_real_tool"
    elif names: cat = "text_call_unknown_tool"
    elif REFUSE_RE.search(a) or CLARIFY_RE.search(a) or (s.endswith("?") and len(s) < 600): cat = "refuse_or_clarify"
    elif s: cat = "plain_answer"
    else: cat = "other"
    exp = task.get("expect_tool") or (task.get("steps") or [[None]])[0][0] or (task.get("expect") or [None])[0]
    first_real = real_names[0] if real_names else None
    return cat, sorted(flags), [norm_name(n) for n in names], exp, first_real

def load_runs(model):
    runs = []
    for f in sorted(glob.glob(str(ROOT / "results/*/*.json"))):
        if "/toolmode/" in f or "/dev/" in f or f.endswith(".tps.json") or "template" in f: continue
        try: d = json.load(open(f))
        except Exception: continue
        if not isinstance(d, dict) or d.get("model", {}).get("name") != model: continue
        log = f[:-5] + ".log.jsonl"
        if Path(log).exists(): runs.append((d, log))
    return runs

def snip(t, n=220):
    t = re.sub(r"\[OUT-OF-BAND USER MESSAGE[^\]]*\]", "[OUT-OF-BAND USER MESSAGE …]", t or "")
    t = t.strip().replace("\n", " ⏎ ")
    return t if len(t) <= n else t[:n].rstrip() + " …"

def template_facts(slug):
    ol = json.loads((RAW / "ollama.json").read_text()).get(slug, {})
    tp, hp = RAW / "ol" / f"{slug}.template", RAW / "hf" / f"{slug}.jinja"
    t = tp.read_text() if tp.exists() else None
    h = hp.read_text() if hp.exists() else None
    f = {"ollama_template_file": str(tp.relative_to(ROOT)) if t else None, "template_digest": ol.get("template_digest"),
         "renderer": ol.get("renderer"), "parser": ol.get("parser")}
    if t is not None:
        f["renders_tools"] = ".Tools" in t
        f["renders_assistant_tool_calls"] = ".ToolCalls" in t
        f["renders_tool_role"] = 'eq .Role "tool"' in t
        f["tool_call_wrapper"] = next((w for w in ("<tool_call>", "[TOOL_CALLS]", "<|python_tag|>") if w in t), None)
        f["template_instruction"] = ("NO other text" in t)
    if h is not None:
        f["hf_renders_tool_list"] = bool(re.search(r"\btools\b", h))  # does the provider template ever print the tool schemas?
        f["hf_renders_tool_call_history"] = "tool_calls" in h
        f["hf_template_tool_markers"] = sorted(set(re.findall(r"<｜tool[^｜]*｜>|<tool_call>|\[TOOL_CALLS\]|\[AVAILABLE_TOOLS\]", h)))
    return f

def toolmode(model):
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results/toolmode/level2/*.json"))):
        if f.endswith(".log.jsonl"): continue
        d = json.load(open(f))
        if d.get("model", {}).get("name") != model: continue
        out[d["tool_mode"]] = {"run_id": d["run_id"], "tool_correct_pct": d["metrics"]["tool_correct_pct"],
                               "by_category_tool_correct": {k: v["tool_correct_pct"] for k, v in d["by_category"].items()},
                               "n_items": d["n_items"], "trials": d["settings"]["trials"]}
        # also classify the native-mode answers of the tool category with the same rules
        log = f[:-5] + ".log.jsonl"
        if Path(log).exists():
            c = Counter()
            for line in open(log):
                r = json.loads(line)
                if not r["task"].startswith("tool-"): continue
                c[classify(TASKS[r["task"]], r["answer"])[0]] += 1
            out[d["tool_mode"]]["tool_category_answer_classes"] = dict(c)
    return out

def analyse(model):
    res = []
    for meta, log in load_runs(model):
        rows = [json.loads(l) for l in open(log)]
        per = {}
        for r in rows:
            t = TASKS[r["task"]]
            if t["category"] not in TOOL_CATS: continue
            cat, flags, names, exp, first_real = classify(t, r["answer"])
            p = per.setdefault(t["category"], {"n": 0, "trace_nonempty": 0, "cats": Counter(), "flags": Counter(),
                                                 "right_tool_as_text": 0, "secs": [], "lens": [], "snips": {}, "unknown_names": Counter()})
            p["n"] += 1; p["trace_nonempty"] += bool(r["trace"]); p["cats"][cat] += 1; p["flags"].update(flags)
            p["secs"].append(r["sec"]); p["lens"].append(len(r["answer"] or ""))
            if cat == "text_call_real_tool" and first_real == exp: p["right_tool_as_text"] += 1
            if cat == "text_call_unknown_tool": p["unknown_names"].update(n for n in names if n not in REAL)
            sl = p["snips"].setdefault(cat, [])
            if len(sl) < 2 and (not sl or sl[0]["task"] != r["task"]):
                sl.append({"task": r["task"], "trial": r["trial"], "text": snip(r["answer"])})
        for p in per.values():
            p["sec_p50"] = round(statistics.median(p.pop("secs")), 2); p["answer_len_p50"] = int(statistics.median(p.pop("lens")))
            p["cats"] = dict(p["cats"]); p["flags"] = dict(p["flags"]); p["unknown_names"] = dict(p["unknown_names"].most_common(8))
        res.append({"run_id": meta["run_id"], "hardware": meta["hardware"].get("gpu") or meta["hardware"].get("cpu"),
                    "engine": f'{meta["engine"]["name"]} {meta["engine"]["version"]}', "harness": meta["agent"]["version"],
                    "tool_correct_pct": meta["metrics"]["tool_correct_pct"], "multistep_success_pct": meta["metrics"]["multistep_success_pct"],
                    "per_category": per})
    return res

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", default=str(ROOT / "research/tool_failure_counts.json"))
    a = ap.parse_args()
    out = {"generated_by": "scripts/tool_failure_analysis.py", "categories": CATEGORIES, "models": {}, "reference": {}}
    for m in FAILING:
        slug = m.replace(":", "-")
        out["models"][m] = {"runs": analyse(m), "template": template_facts(slug), "toolmode": toolmode(m)}
    for m in REFERENCE:
        out["reference"][m] = {"runs": [{"run_id": r["run_id"], "tool_correct_pct": r["tool_correct_pct"],
                                        "tool_sec_p50": r["per_category"]["tool"]["sec_p50"],
                                        "tool_trace_nonempty": r["per_category"]["tool"]["trace_nonempty"],
                                        "tool_n": r["per_category"]["tool"]["n"]} for r in analyse(m)]}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    # ---- markdown ----
    keys = list(CATEGORIES)
    short = {"empty": "empty", "think_leak": "think", "harness_error": "error", "text_call_real_tool": "call-as-text (real name)",
             "text_call_unknown_tool": "call-like, unknown name", "refuse_or_clarify": "refuse/ask", "plain_answer": "prose answer", "other": "other"}
    for catname in ("tool", "multistep", "recovery", "injection"):
        print(f"\n### Category `{catname}`\n")
        print("| Model | Run | n | trace≠∅ | " + " | ".join(short[k] for k in keys) + " | right tool as text | starts with bare JSON call | fake user turn | claims result | p50 s |")
        print("|---|---|---|---|" + "---|" * len(keys) + "---|---|---|---|---|")
        for m, d in out["models"].items():
            for r in d["runs"]:
                p = r["per_category"][catname]
                print(f"| {m} | {r['run_id']} | {p['n']} | {p['trace_nonempty']} | " + " | ".join(str(p["cats"].get(k, 0)) for k in keys) +
                      f" | {p['right_tool_as_text']} | {p['flags'].get('starts_with_bare_json_call', 0)} | {p['flags'].get('fake_user_turn', 0)} | {p['flags'].get('claims_execution_or_result', 0)} | {p['sec_p50']} |")
    print("\n### Templates\n\n| Model | Ollama renders .Tools | tool-call wrapper in Ollama template | renders tool role | renderer/parser | provider (HF) template renders tool list | provider template renders tool-call history |\n|---|---|---|---|---|---|---|")
    for m, d in out["models"].items():
        t = d["template"]
        print(f"| {m} | {t.get('renders_tools')} | {t.get('tool_call_wrapper')} | {t.get('renders_tool_role')} | {t.get('renderer')}/{t.get('parser')} | {t.get('hf_renders_tool_list')} | {t.get('hf_renders_tool_call_history')} |")
    print("\n### Tool-mode study level 2 (tool_correct %: tool / multistep / recovery categories; overall)\n\n| Model | native | emulated (text-only) | native tool-category answer classes |\n|---|---|---|---|")
    for m, d in out["models"].items():
        tm = d["toolmode"]
        if not tm: print(f"| {m} | not in study | not in study | |"); continue
        fmt = lambda x: "—" if not x else " / ".join(str(x["by_category_tool_correct"].get(k)) for k in ("tool", "multistep", "recovery")) + f"; overall {x['tool_correct_pct']}"
        print(f"| {m} | {fmt(tm.get('native'))} | {fmt(tm.get('text-only'))} | {tm.get('native', {}).get('tool_category_answer_classes')} |")
    print("\n### Reference models (same harness)\n")
    for m, d in out["reference"].items(): print(f"- {m}: {d['runs']}")
    print(f"\nJSON written to {a.json}")

if __name__ == "__main__":
    main()
