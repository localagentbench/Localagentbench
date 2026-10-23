"""Deterministic mock tools. Same input -> same output, always. No network."""
import ast, json, operator

TOOL_SPECS = [
 ("get_weather", "Get current weather for a city. Use format 'City, CC' (ISO country code).", {"city": "string"}),
 ("search_web", "Search the web; returns top results.", {"query": "string"}),
 ("read_file", "Read a text file by absolute path.", {"path": "string"}),
 ("send_email", "Send an email.", {"to": "string", "subject": "string", "body": "string"}),
 ("calculator", "Evaluate an arithmetic expression using + - * / ( ).", {"expression": "string"}),
 ("get_stock_price", "Get latest stock price in USD for a ticker symbol.", {"ticker": "string"}),
 ("create_calendar_event", "Create a calendar event.", {"title": "string", "date": "string"}),
 ("lookup_user", "Look up a user record by id (format 'u-<number>').", {"user_id": "string"}),
 ("convert_currency", "Convert an amount between currencies (uppercase ISO codes).", {"amount": "number", "from_currency": "string", "to_currency": "string"}),
 ("translate", "Translate text into a target language (ISO 639-1 code).", {"text": "string", "target_lang": "string"}),
]

def openai_tools():
    out = []
    for name, desc, props in TOOL_SPECS:
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": {
            "type": "object", "properties": {k: {"type": v} for k, v in props.items()}, "required": list(props)}}})
    return out

WEATHER = {"paris, fr": 14, "madrid, es": 22, "tokyo, jp": 18, "lima, pe": 19, "oslo, no": 3, "cairo, eg": 31,
           "berlin, de": 11, "toronto, ca": 7, "sydney, au": 24, "mexico city, mx": 21, "paris, us": 9,
           "bogota, co": 15, "rome, it": 20, "seoul, kr": 12, "nairobi, ke": 25}
STOCKS = {"AAPL": 190.5, "MSFT": 410.0, "NVDA": 120.25, "TSLA": 250.0, "AMZN": 180.0, "GOOG": 165.5, "META": 500.0, "IBM": 175.0}
FX = {"USD": 1.0, "EUR": 0.9, "GBP": 0.8, "JPY": 150.0, "MXN": 18.0, "CAD": 1.35}
USERS = {f"u-{100+i}": {"name": n, "email": f"{n.lower()}@example.com", "city": c, "manager": f"u-{100+(i+1)%10}"}
         for i, (n, c) in enumerate([("Ana", "Madrid, ES"), ("Bruno", "Lima, PE"), ("Chen", "Tokyo, JP"), ("Dana", "Oslo, NO"),
                                     ("Eli", "Cairo, EG"), ("Fatima", "Berlin, DE"), ("Gus", "Toronto, CA"), ("Hana", "Seoul, KR"),
                                     ("Ivan", "Rome, IT"), ("Juno", "Nairobi, KE")])}
TRANSLATIONS = {("hello", "es"): "hola", ("thank you", "fr"): "merci", ("good morning", "de"): "guten Morgen",
                ("goodbye", "it"): "arrivederci", ("cat", "ja"): "neko", ("water", "es"): "agua", ("friend", "pt"): "amigo"}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.USub: operator.neg}
def _calc(expr):
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported")
    return round(ev(ast.parse(expr, mode="eval")), 4)

class World:
    """files / search index / one-shot errors are per-task overrides."""
    def __init__(self, files=None, search=None, errors=None, inject=None):
        self.files = dict(files or {})
        self.search = dict(search or {})
        self.errors = dict(errors or {})   # tool_name -> (predicate(args)->bool, error_msg)
        self.inject = dict(inject or {})  # tool_name -> extra fields merged into its result (injection carriers)
        self.sent = []

    def call(self, name, args):
        if not isinstance(args, dict): return {"error": "arguments must be a JSON object"}
        if name in self.errors:
            pred, msg = self.errors[name]
            if pred(args): return {"error": msg}
        try:
            res = getattr(self, "t_" + name)(**args)
            return dict(res, **self.inject[name]) if name in self.inject and "error" not in res else res
        except AttributeError:
            return {"error": f"unknown tool {name}"}
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}

    def t_get_weather(self, city):
        t = WEATHER.get(str(city).strip().lower())
        return {"city": city, "temp_c": t, "condition": "clear"} if t is not None else {"error": f"city '{city}' not found; use 'City, CC'"}
    def t_search_web(self, query):
        q = str(query).lower()
        for k, v in self.search.items():
            if k in q: return {"results": v}
        return {"results": [{"title": "No relevant results", "snippet": ""}]}
    def t_read_file(self, path):
        return {"path": path, "content": self.files[path]} if path in self.files else {"error": f"file not found: {path}"}
    def t_send_email(self, to, subject, body):
        self.sent.append({"to": to, "subject": subject, "body": body})
        return {"status": "sent", "message_id": f"msg-{len(self.sent):03d}"}
    def t_calculator(self, expression):
        try: return {"result": _calc(str(expression))}
        except Exception: return {"error": "invalid expression; use only numbers and + - * / ( )"}
    def t_get_stock_price(self, ticker):
        p = STOCKS.get(str(ticker).upper().strip())
        return {"ticker": ticker, "price_usd": p} if p is not None else {"error": f"unknown ticker {ticker}"}
    def t_create_calendar_event(self, title, date):
        h = sum(ord(c) for c in f"{title}|{date}") % 9000 + 1000
        return {"status": "created", "event_id": f"evt-{h}"}
    def t_lookup_user(self, user_id):
        u = USERS.get(str(user_id).strip())
        return dict(u, user_id=user_id) if u else {"error": f"user {user_id} not found; ids look like 'u-123'"}
    def t_convert_currency(self, amount, from_currency, to_currency):
        f, t = str(from_currency), str(to_currency)
        if f not in FX or t not in FX: return {"error": "unknown currency; use uppercase ISO codes like USD, EUR"}
        return {"amount": round(float(amount) / FX[f] * FX[t], 2), "currency": t}
    def t_translate(self, text, target_lang):
        r = TRANSLATIONS.get((str(text).lower().strip(), str(target_lang).lower()))
        return {"translation": r} if r else {"error": "unsupported text/language pair"}
