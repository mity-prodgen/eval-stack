"""Post 2's Claude judge prompt and parser, verbatim. Used as the baseline (P0) Claude prompt.

The judge replies with one line of JSON, {"hallucinated": true|false, "confidence": 0-100}, parsed with a
regex + json.loads so a malformed reply becomes an explicit, countable parse error instead of a silent
mismatch.
"""

from __future__ import annotations

import json
import re

JUDGE_SYSTEM_PROMPT = (
    "You are a strict fact-checking judge. You will be given a source passage "
    "and a response. Decide whether every factual claim in the response is "
    "directly supported by the source passage. A claim is supported only if "
    "the source states it explicitly or it follows with no added assumptions. "
    "Answer with hallucinated=true if any claim contradicts the source, states "
    "a number, date, or name not present in the source, or adds a specific "
    "detail the source does not support.\n\n"
    "Your entire reply must be exactly one line matching this pattern and "
    'nothing else: {"hallucinated": true|false, "confidence": <integer 0-100>}. '
    '"confidence" is how certain you are in that verdict, from 0 (a coin flip) '
    "to 100 (certain). Do not explain your reasoning. Do not list issues. Do "
    "not add any text, markdown, or code fences before or after the JSON - "
    "the JSON object is the complete response, first character to last."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judgment(text: str) -> tuple[bool, float]:
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"no JSON object found in judge output: {text!r}")
    data = json.loads(match.group(0))
    return bool(data["hallucinated"]), float(data["confidence"]) / 100.0
