"""Hermes Agent (NousResearch/hermes-agent) harness adapter.
Runs every task as a real `hermes chat --oneshot` process against a local OpenAI-compatible endpoint.
The deterministic mock tools are served to Hermes over MCP (lab/mcp_server.py) and the session is
restricted with `-t mcp-lab`, so Hermes' built-in tools (terminal, files, web) are not offered.
Each task gets a fresh HERMES_HOME (no memory/session carry-over between tasks)."""
import json, os, re, shutil, subprocess, sys, tempfile, time

PREFIX = "mcp_lab_"
JSON_HINT = "Respond with ONLY a single valid JSON object. No prose, no markdown, no code fences.\n\n"

class HermesHarness:
    def __init__(self, hermes_bin, base_url, model, api_key, ctx, score_fn):
        self.bin, self.base_url, self.model, self.api_key, self.score = hermes_bin, base_url, model, api_key, score_fn
        self.ctx = max(ctx or 0, 65536)  # Hermes refuses endpoints serving < 64K context
        self.state = tempfile.mkdtemp(prefix="lab-hermes-state-")
        self.server_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
        self.server_python = os.environ.get("LAB_MCP_PYTHON", sys.executable)  # needs `mcp` installed
        try: self.version = subprocess.run([self.bin, "--version"], capture_output=True, text=True, timeout=120).stdout.splitlines()[0].strip()
        except Exception: self.version = "unknown"

    def _home(self):
        h = tempfile.mkdtemp(prefix="lab-hermes-home-")
        open(os.path.join(h, "config.yaml"), "w").write(f"""model:
  default: {json.dumps(self.model)}
  provider: custom
  base_url: {json.dumps(self.base_url)}
  api_key: {json.dumps(self.api_key)}
  ollama_num_ctx: {self.ctx}
memory:
  memory_enabled: false
  user_profile_enabled: false
tools:
  tool_search:
    enabled: "off"
mcp_servers:
  lab:
    command: {json.dumps(self.server_python)}
    args: [{json.dumps(self.server_py)}]
    env: {{LAB_STATE_DIR: {json.dumps(self.state)}}}
""")
        return h

    def run_task(self, task, timeout=900):
        json.dump({"task_id": task["id"]}, open(os.path.join(self.state, "current.json"), "w"))
        tp = os.path.join(self.state, "trace.jsonl"); open(tp, "w").close()
        prompt = (JSON_HINT if task["category"] == "json" else "") + task["prompt"]
        home = self._home()
        args = [self.bin, "chat", "-q", prompt, "--oneshot", "-Q", "-t", "mcp-lab", "--yolo", "--max-turns", "10"]
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=dict(os.environ, HERMES_HOME=home), cwd=home).stdout
        finally:
            # oneshot spawns a local gateway (--idle-exit); stop it so the next task starts clean
            subprocess.run(["pkill", "-f", "hermes_cli.main --profile default gateway run"], capture_output=True)
            shutil.rmtree(home, ignore_errors=True)
        answer = re.sub(r"\s*session_id: \S+\s*$", "", out).strip()
        trace, sent = [], []
        for line in open(tp):
            c = json.loads(line)
            trace.append({"name": c["name"].removeprefix(PREFIX), "args": c["args"], "result": c["result"]}); sent = c["sent"]
        return self.score(task, trace, answer, sent)
