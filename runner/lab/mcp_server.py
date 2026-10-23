"""Stdio MCP server exposing the deterministic mock tools, for real agent harnesses (Hermes Agent, OpenClaw...).
The harness adapter writes $LAB_STATE_DIR/current.json = {"task_id": ...}; every call is appended to
$LAB_STATE_DIR/trace.jsonl so the bench can score exactly what the agent did."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from mcp.server.mcpserver import MCPServer as FastMCP  # mcp>=2
except ImportError:
    from mcp.server.fastmcp import FastMCP  # mcp 1.x
from lab.tasks import all_tasks, world_for

STATE = os.environ.get("LAB_STATE_DIR", "/tmp/lab_state")
TASKS = {t["id"]: t for t in all_tasks()}
_cur = {"id": None, "world": None}
mcp = FastMCP("lab")

def _call(name, **args):
    tid = json.load(open(os.path.join(STATE, "current.json")))["task_id"]
    if tid != _cur["id"]: _cur.update(id=tid, world=world_for(TASKS[tid]))
    res = _cur["world"].call(name, args)
    with open(os.path.join(STATE, "trace.jsonl"), "a") as f:
        f.write(json.dumps({"task_id": tid, "name": name, "args": args, "result": res, "sent": _cur["world"].sent}) + "\n")
    return json.dumps(res)

@mcp.tool(description="Get current weather for a city. Use format 'City, CC' (ISO country code).")
def get_weather(city: str) -> str: return _call("get_weather", city=city)
@mcp.tool(description="Search the web; returns top results.")
def search_web(query: str) -> str: return _call("search_web", query=query)
@mcp.tool(description="Read a text file by absolute path.")
def read_file(path: str) -> str: return _call("read_file", path=path)
@mcp.tool(description="Send an email.")
def send_email(to: str, subject: str, body: str) -> str: return _call("send_email", to=to, subject=subject, body=body)
@mcp.tool(description="Evaluate an arithmetic expression using + - * / ( ).")
def calculator(expression: str) -> str: return _call("calculator", expression=expression)
@mcp.tool(description="Get latest stock price in USD for a ticker symbol.")
def get_stock_price(ticker: str) -> str: return _call("get_stock_price", ticker=ticker)
@mcp.tool(description="Create a calendar event.")
def create_calendar_event(title: str, date: str) -> str: return _call("create_calendar_event", title=title, date=date)
@mcp.tool(description="Look up a user record by id (format 'u-<number>').")
def lookup_user(user_id: str) -> str: return _call("lookup_user", user_id=user_id)
@mcp.tool(description="Convert an amount between currencies (uppercase ISO codes).")
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str: return _call("convert_currency", amount=amount, from_currency=from_currency, to_currency=to_currency)
@mcp.tool(description="Translate text into a target language (ISO 639-1 code).")
def translate(text: str, target_lang: str) -> str: return _call("translate", text=text, target_lang=target_lang)

if __name__ == "__main__": mcp.run()
