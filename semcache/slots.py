"""A2 slot guard: extract the slots a cached answer depends on, serve only if every slot matches.

Rules plus a normalizer, no model. Written and committed BEFORE semcache/pairs/pairs.jsonl existed (see git
log), and not changed after looking at results; tests/test_semcache_slots.py uses its own examples, none
taken from the eval pairs.

Slots: origin, destination, class, date, time, number, negation, version, entity.

    extract("Economy fare from Dubai to LHR on 3 Oct")
    -> {"origin": "DXB", "destination": "LHR", "class": "economy", "date": {"10-03"}, ...}

Matching rule: two queries match only if every slot has the same value in both. A slot present in one query
and absent in the other is a mismatch ("baggage allowance" vs "baggage allowance in Business" can need
different answers). Aliases (city <-> airport code, "biz" <-> "business", "Amex" <-> "American Express",
"$1k" <-> "1,000 dollars", "3.12" <-> "Python 3.12") are normalized to one value before comparing; any alias
not in the tables below is a miss, and those misses are part of what the experiment measures.
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------------------------- lexicons
# Places: canonical IATA code -> lowercase names. Codes themselves are matched case-sensitively (uppercase
# 3-letter tokens) so words like "sin" or "fra" in ordinary text never count.
PLACES: dict[str, list[str]] = {
    "DXB": ["dubai"], "AUH": ["abu dhabi"], "DOH": ["doha"], "LHR": ["london heathrow", "heathrow", "london"],
    "LGW": ["london gatwick", "gatwick"], "JFK": ["new york jfk", "new york", "nyc"], "EWR": ["newark"],
    "LAX": ["los angeles"], "SFO": ["san francisco"], "ORD": ["chicago"], "CDG": ["paris", "charles de gaulle"],
    "FRA": ["frankfurt"], "AMS": ["amsterdam"], "MAD": ["madrid"], "FCO": ["rome"], "IST": ["istanbul"],
    "SIN": ["singapore"], "HKG": ["hong kong"], "BKK": ["bangkok"], "NRT": ["tokyo narita", "narita"],
    "HND": ["tokyo haneda", "haneda", "tokyo"], "SYD": ["sydney"], "MEL": ["melbourne"], "BOM": ["mumbai", "bombay"],
    "DEL": ["delhi", "new delhi"], "BLR": ["bangalore", "bengaluru"], "CAI": ["cairo"], "JNB": ["johannesburg"],
    "KUL": ["kuala lumpur"], "MAN": ["manchester"], "DUB": ["dublin"], "ZRH": ["zurich"], "MUC": ["munich"],
    "YYZ": ["toronto"], "GRU": ["sao paulo", "são paulo"], "MEX": ["mexico city"], "RUH": ["riyadh"],
    "JED": ["jeddah"], "KWI": ["kuwait"], "BAH": ["bahrain"], "MCT": ["muscat"], "CMB": ["colombo"],
}

CURRENCIES: dict[str, list[str]] = {
    "USD": ["usd", "us dollars", "us dollar", "dollars", "dollar", "$"], "EUR": ["eur", "euros", "euro", "€"],
    "GBP": ["gbp", "pounds", "pound", "sterling", "£"], "AED": ["aed", "dirhams", "dirham"],
    "INR": ["inr", "rupees", "rupee", "₹"], "JPY": ["jpy", "yen", "¥"], "CHF": ["chf", "swiss francs"],
    "SGD": ["sgd", "singapore dollars"], "CAD": ["cad", "canadian dollars"], "AUD": ["aud", "australian dollars"],
    "SAR": ["sar", "riyals", "riyal"],
}

# Other things a query can move or convert "from X to Y" (accounts, stacks, plans). Canonical -> aliases.
ENDPOINTS: dict[str, list[str]] = {
    "checking": ["checking account", "checking", "current account"], "savings": ["savings account", "savings"],
    "credit card": ["credit card"], "debit card": ["debit card"], "brokerage": ["brokerage account", "brokerage"],
    "postgres": ["postgresql", "postgres"], "mysql": ["mysql"], "sqlite": ["sqlite"], "mongodb": ["mongodb", "mongo"],
    "javascript": ["javascript"], "typescript": ["typescript"], "java": ["java"], "kotlin": ["kotlin"],
    "python": ["python"], "go": ["golang"], "rust": ["rust"], "react": ["react"], "vue": ["vue"], "angular": ["angular"],
    "monthly": ["monthly plan", "monthly"], "annual": ["annual plan", "yearly plan", "annual", "yearly"],
}

# Named entities whose swap changes the answer. Canonical -> aliases (lowercase unless listed in CASED_ALIASES).
ENTITIES: dict[str, list[str]] = {
    # payments and banks
    "visa": ["visa"], "mastercard": ["mastercard", "master card"], "american express": ["american express", "amex"],
    "discover": ["discover card"], "paypal": ["paypal"], "apple pay": ["apple pay"], "google pay": ["google pay", "gpay"],
    "venmo": ["venmo"], "zelle": ["zelle"], "wise": ["wise transfer"], "swift": ["swift"], "sepa": ["sepa"],
    "chase": ["chase"], "citi": ["citibank", "citi"], "hsbc": ["hsbc"], "barclays": ["barclays"],
    "wells fargo": ["wells fargo"], "bank of america": ["bank of america"], "revolut": ["revolut"],
    # airlines and programmes
    "emirates": ["emirates"], "qatar airways": ["qatar airways", "qatar"], "etihad": ["etihad"],
    "british airways": ["british airways"], "lufthansa": ["lufthansa"], "air france": ["air france"],
    "klm": ["klm"], "turkish airlines": ["turkish airlines"], "singapore airlines": ["singapore airlines"],
    "united": ["united airlines"], "delta": ["delta"], "american airlines": ["american airlines"],
    "ryanair": ["ryanair"], "easyjet": ["easyjet"], "flydubai": ["flydubai"],
    # dev tools and platforms
    "aws": ["aws", "amazon web services"], "gcp": ["gcp", "google cloud"], "azure": ["azure"],
    "github": ["github"], "gitlab": ["gitlab"], "bitbucket": ["bitbucket"], "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"], "npm": ["npm"], "yarn": ["yarn"], "pnpm": ["pnpm"], "pip": ["pip"],
    "poetry": ["poetry"], "conda": ["conda"], "homebrew": ["homebrew", "brew"], "vs code": ["vs code", "vscode"],
    "jetbrains": ["jetbrains", "intellij", "pycharm"], "terraform": ["terraform"], "jenkins": ["jenkins"],
    "github actions": ["github actions"], "circleci": ["circleci"], "vercel": ["vercel"], "heroku": ["heroku"],
    "macos": ["macos", "mac os", "os x"], "windows": ["windows"], "linux": ["linux"], "ubuntu": ["ubuntu"],
    "ios": ["ios", "iphone", "ipad"], "android": ["android"], "chrome": ["chrome"], "firefox": ["firefox"],
    "safari": ["safari"], "edge": ["microsoft edge"],
    # support channels and devices
    "whatsapp": ["whatsapp"], "email": ["email", "e-mail"], "live chat": ["live chat"], "phone": ["phone call", "by phone"],
    "slack": ["slack"], "teams": ["microsoft teams"], "zoom": ["zoom"],
}
CASED_ALIASES: dict[str, str] = {"MC": "mastercard", "BA": "british airways", "EK": "emirates", "QR": "qatar airways"}

# Versioned products: the product name is part of the version mention, not an entity ("Python 3.12" and
# "3.12" both normalize to version 3.12).
VERSIONED = (r"python|py|node(?:\.js|js)?|java|jdk|golang|ruby|php|rust|kotlin|swift|typescript|"
             r"postgres(?:ql)?|mysql|django|rails|react|angular|vue|next(?:\.js|js)?|spring boot|"
             r"ios|android|ipados|macos|windows|ubuntu|debian|xcode|gradle|maven|npm|pip|docker|kubernetes|"
             r"terraform|api|sdk|app|cli|version|release")

CLASSES: list[tuple[str, str]] = [  # (pattern, canonical); premium economy before economy
    (r"premium[\s-]+economy|premium[\s-]+eco\b|economy plus|comfort plus", "premium economy"),
    (r"\beconomy\b|\bcoach\b|main cabin|\beco\b", "economy"),
    (r"\bbusiness(?:\s+class)?\b|\bbiz(?:\s+class)?\b|\bclub world\b|\bj class\b", "business"),
    (r"\bfirst\s+class\b|\bin first\b|\bfirst cabin\b", "first"),
]

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"])}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
RELATIVE = ["today", "tonight", "tomorrow", "yesterday", "this weekend", "next weekend", "this week", "next week",
            "last week", "this month", "next month", "last month", "this year", "next year", "last year"]

NEGATION = re.compile(r"\b(?:not|no|never|none|nor|cannot|without|neither)\b|n't\b")

UNITS = {"kg": "kg", "kgs": "kg", "kilos": "kg", "kilograms": "kg", "lb": "lb", "lbs": "lb", "pounds weight": "lb",
         "gb": "gb", "mb": "mb", "tb": "tb", "%": "%", "percent": "%", "days": "day", "day": "day",
         "hours": "hour", "hour": "hour", "hrs": "hour", "minutes": "minute", "minute": "minute", "mins": "minute",
         "weeks": "week", "week": "week", "months": "month", "month": "month", "years": "year", "year": "year",
         "bags": "bag", "bag": "bag", "pieces": "piece", "piece": "piece", "passengers": "passenger",
         "passenger": "passenger", "seats": "seat", "seat": "seat", "cm": "cm", "inches": "inch", "inch": "inch",
         "ml": "ml", "litres": "l", "liters": "l", "users": "user", "user": "user", "miles": "mile",
         "points": "point", "cores": "core", "nights": "night", "night": "night", "times": "time",
         "transactions": "transaction", "requests": "request", "items": "item"}

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}


# --------------------------------------------------------------------------------------------- helpers
def _norm(text: str) -> str:
    t = text.replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-").replace("→", " to ")
    return re.sub(r"\s+", " ", t).strip()


def _alias_re(aliases: list[str]) -> str:
    parts = sorted(aliases, key=len, reverse=True)
    return "|".join(re.escape(a) if not a[0].isalnum() else r"\b" + re.escape(a) + r"\b" for a in parts)


def _blank(text: str, span: tuple[int, int]) -> str:
    """Replace a consumed span with spaces so later rules can't match it again (positions stay valid)."""
    return text[: span[0]] + " " * (span[1] - span[0]) + text[span[1]:]


def _to_number(num: str, mult: str | None) -> float:
    v = float(num.replace(",", ""))
    return v * _MULT.get((mult or "").lower(), 1.0)


def _fmt(v: float) -> str:
    return str(int(v)) if v == int(v) else f"{v:g}"


def _version(v: str) -> str:
    """'3.12.0' -> '3.12.0'; strip a leading v; trailing '.0' is kept (3.0 and 3 can differ)."""
    return v.lower().lstrip("v")


# --------------------------------------------------------------------------------------------- extraction
def _endpoints(orig: str, low: str) -> list[tuple[int, int, str]]:
    """Every place / currency / endpoint mention as (start, end, canonical), non-overlapping, in order."""
    found: list[tuple[int, int, str]] = []
    for m in re.finditer(r"\b[A-Z]{3}\b", orig):
        code = m.group(0)
        if code in PLACES:
            found.append((m.start(), m.end(), code))
        elif code.lower() in {a for al in CURRENCIES.values() for a in al}:
            found.append((m.start(), m.end(), next(c for c, al in CURRENCIES.items() if code.lower() in al)))
    for table in (PLACES, ENDPOINTS, CURRENCIES):
        for canon, aliases in table.items():
            words = [a for a in aliases if len(a) > 1]  # single symbols ($, €) are money, handled by numbers
            if not words:
                continue
            for m in re.finditer(_alias_re(words), low):
                found.append((m.start(), m.end(), canon))
    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out: list[tuple[int, int, str]] = []
    for f in found:
        if not out or f[0] >= out[-1][1]:
            out.append(f)
    return out


def _direction(low: str, eps: list[tuple[int, int, str]]) -> tuple[str | None, str | None]:
    """origin/destination from 'from A to B', 'A to B', 'A-B', 'to B from A', 'A into B'."""
    for (s1, e1, a), (s2, e2, b) in zip(eps, eps[1:]):
        between = low[e1:s2].strip()
        before = low[max(0, s1 - 6): s1]
        if re.fullmatch(r"(?:to|into|->|-|/|and on to|onward to)", between) or (
                "from" in before and re.fullmatch(r"(?:to|into)", between)):
            return a, b
        if "to" in before.split() and between == "from":  # "to B from A"
            return b, a
    for (s, e, a) in eps:  # single endpoint with a preposition
        before = low[max(0, s - 6): s]
        if re.search(r"\bfrom\s*$", before):
            return a, None
        if re.search(r"\b(?:to|into)\s*$", before):
            return None, a
    return None, None


def extract(query: str) -> dict[str, Any]:
    """Slots present in `query`. Absent slots are omitted; set-valued slots are frozensets."""
    orig = _norm(query)
    low = orig.lower()
    slots: dict[str, Any] = {}

    # --- negation (on the raw text, before anything is blanked)
    if NEGATION.search(low):
        slots["negation"] = True

    # --- class
    classes = set()
    for pat, canon in CLASSES:
        for m in re.finditer(pat, low):
            classes.add(canon)
            low = _blank(low, m.span())
            orig = _blank(orig, m.span())
    if classes:
        slots["class"] = frozenset(classes)

    # --- dates (before numbers, so "3 Oct" isn't the number 3)
    dates = set()
    month_alt = "|".join(sorted(MONTHS, key=len, reverse=True))
    date_patterns = [
        (rf"\b(\d{{4}})-(\d{{1,2}})-(\d{{1,2}})\b", lambda m: f"{int(m[2]):02d}-{int(m[3]):02d}-{m[1]}"),
        (rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_alt})\.?(?:,?\s+(\d{{4}}))?\b",
         lambda m: f"{MONTHS[m[2]]:02d}-{int(m[1]):02d}" + (f"-{m[3]}" if m[3] else "")),
        (rf"\b({month_alt})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?",
         lambda m: f"{MONTHS[m[1]]:02d}-{int(m[2]):02d}" + (f"-{m[3]}" if m[3] else "")),
        (r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", lambda m: f"{int(m[1]):02d}-{int(m[2]):02d}"),
        (rf"\b(?:in\s+)?({month_alt})\b(?:\s+(\d{{4}}))?", lambda m: f"{MONTHS[m[1]]:02d}" + (f"-{m[2]}" if m[2] else "")),
        (r"\b(" + "|".join(RELATIVE) + r")\b", lambda m: m[1]),
        (r"\b(?:on\s+|this\s+|next\s+|last\s+)?(" + "|".join(WEEKDAYS) + r")s?\b", lambda m: m[0].replace("on ", "").rstrip("s")),
        (r"\b(?:in\s+)?(20\d{2}|19\d{2})\b", lambda m: m[1]),
    ]
    for pat, fn in date_patterns:
        for m in re.finditer(pat, low):
            if pat.startswith(r"\b(?:in\s+)?(") and m[1] == "may" and not re.search(r"\bin\s+may\b|\bmay\s+\d", low):
                continue  # "may" the verb
            dates.add(fn(m))
            low = _blank(low, m.span())
            orig = _blank(orig, m.span())
    if dates:
        slots["date"] = frozenset(dates)

    # --- times
    times = set()
    for m in re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)|\b(\d{1,2}):(\d{2})\b|\b(noon|midnight)\b", low):
        if m[6]:
            times.add("12:00" if m[6] == "noon" else "00:00")
        else:
            h, mi = int(m[1] or m[4]), int(m[2] or m[5] or 0)
            if m[3]:
                h = h % 12 + (12 if m[3].startswith("p") else 0)
            times.add(f"{h:02d}:{mi:02d}")
        low = _blank(low, m.span())
        orig = _blank(orig, m.span())
    if times:
        slots["time"] = frozenset(times)

    # --- versions with a product name or a v prefix: "Python 3.12", "API v1", "v2" (product name consumed too)
    versions = set()
    for pat in (rf"\b(?:{VERSIONED})\s*v?(\d+(?:\.\d+)*)\b(?!\s*(?:%|gb|mb|kg|days?|hours?|bags?|users?))",
                r"\bv(\d+(?:\.\d+)*)\b"):
        for m in re.finditer(pat, low):
            versions.add(_version(m[1]))
            low = _blank(low, m.span())
            orig = _blank(orig, m.span())

    # --- flight numbers (entity-like codes such as EK 202, QR1) before plain numbers
    entities = set()
    for m in re.finditer(r"\b([A-Z]{2})\s?(\d{1,4})\b", orig):
        if m[1] in ("AM", "PM"):
            continue
        entities.add(f"flight {m[1]}{int(m[2])}")
        low = _blank(low, m.span())
        orig = _blank(orig, m.span())

    # --- money
    numbers = set()
    cur_sym = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"}
    cur_words = sorted({a for al in CURRENCIES.values() for a in al if len(a) > 1}, key=len, reverse=True)
    cur_alt = "|".join(re.escape(w) for w in cur_words)
    mult = r"(?:\s?(k|thousand|m|mn|million|bn|billion)\b)?"
    money_pats = [
        rf"([$€£₹¥])\s?({_NUM}){mult}",
        rf"\b({cur_alt})\s?({_NUM}){mult}",
        rf"\b({_NUM}){mult}\s?({cur_alt})\b",
    ]
    for k, pat in enumerate(money_pats):
        for m in re.finditer(pat, low):
            if k == 2:
                num, mu, cur = m[1], m[2], m[3]
            else:
                cur, num, mu = m[1], m[2], m[3]
            canon = cur_sym.get(cur) or next(c for c, al in CURRENCIES.items() if cur in al)
            numbers.add(f"{_fmt(_to_number(num, mu))} {canon}")
            low = _blank(low, m.span())
            orig = _blank(orig, m.span())

    # --- bare dotted versions ("3.12"), after money so "3.50 euros" is money
    unit_alt = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
    for m in re.finditer(rf"(?<![\d.,])\b(\d+\.\d+(?:\.\d+)*)\b(?!\s*(?:k\b|m\b|{unit_alt}))", low):
        versions.add(_version(m[1]))
        low = _blank(low, m.span())
        orig = _blank(orig, m.span())
    if versions:
        slots["version"] = frozenset(versions)

    # --- quantities with units, number words with units, then any number left
    for m in re.finditer(rf"\b({_NUM}){mult}\s?({unit_alt})(?![a-z])", low):
        numbers.add(f"{_fmt(_to_number(m[1], m[2]))} {UNITS[m[3]]}")
        low = _blank(low, m.span())
        orig = _blank(orig, m.span())
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "a single": 1}
    for m in re.finditer(rf"\b({'|'.join(words)})\s+({unit_alt})(?![a-z])", low):
        numbers.add(f"{words[m[1]]} {UNITS[m[2]]}")
        low = _blank(low, m.span())
        orig = _blank(orig, m.span())
    for m in re.finditer(rf"(?<![\w.]){_NUM}(?![\w.])", low):  # anything numeric left
        numbers.add(_fmt(_to_number(m[0], None)))
        low = _blank(low, m.span())
        orig = _blank(orig, m.span())
    if numbers:
        slots["number"] = frozenset(numbers)

    # --- origin / destination (places, currencies, accounts, stacks)
    eps = _endpoints(orig, low)
    o, d = _direction(low, eps)
    if o:
        slots["origin"] = o
    if d:
        slots["destination"] = d
    used = {o, d}
    places_named = {c for (_, _, c) in eps if c not in used}

    # --- entities (plus any endpoint mentioned without a direction)
    for canon, aliases in ENTITIES.items():
        if re.search(_alias_re(aliases), low):
            entities.add(canon)
    for alias, canon in CASED_ALIASES.items():
        if re.search(rf"\b{alias}\b", orig):
            entities.add(canon)
    entities |= places_named
    if entities:
        slots["entity"] = frozenset(entities)
    return slots


def guard(cached_query: str, new_query: str) -> tuple[bool, list[str]]:
    """(serve?, mismatched slot names). Serve only if every slot matches, including presence."""
    a, b = extract(cached_query), extract(new_query)
    bad = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return not bad, bad
