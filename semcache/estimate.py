"""Dry-run token and cost estimates for semcache, and the --max-usd gate (shared with persuasion/).

Token counts use tiktoken cl100k_base (the text-embedding-3 tokenizer) times a safety factor; every factor
errs high because the estimate gates --max-usd:

  OpenAI embeddings   x1.05  (same tokenizer; small margin)
  Cohere, Voyage,     x1.30  (own tokenizers, not public in a form we can run offline; no calibration yet -
  Gemini                      the smoke test gives one)
  Jev 1.13 / 1.14     x1.54  (p95 of actual/cl100k on post 2's logs for 1.13, persuasion/estimate.py; 1.14
                              assumed the same tokenizer until the smoke test measures it)
  Luna                x1.20  on o200k_base, input billed once per question (one question here)

The Voyage free allowance (200M tokens) and Gemini batch discount are ignored, so hosted estimates are
upper bounds. Local models cost $0 in API fees.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import tiktoken

from harness.prices import price

from .common import GEMINI_SIMILARITY_PREFIX, Embedder

_CL100K = tiktoken.get_encoding("cl100k_base")
_O200K = tiktoken.get_encoding("o200k_base")
EMBED_FACTOR = {"openai": 1.05, "cohere": 1.30, "voyage": 1.30, "gemini": 1.30}
JEV_FACTOR, LUNA_FACTOR = 1.54, 1.20


@dataclass
class Est:
    n: int = 0
    tokens: int = 0
    usd: float = 0.0

    def add(self, n: int, tokens: int, usd: float) -> None:
        self.n += n
        self.tokens += tokens
        self.usd += usd


def embed_tokens(e: Embedder, text: str) -> int:
    if e.provider == "local":
        return 0
    if e.provider == "gemini":
        text = GEMINI_SIMILARITY_PREFIX + text
    return round(len(_CL100K.encode(text)) * EMBED_FACTOR[e.provider])


def estimate_embed(e: Embedder, texts: list[str]) -> Est:
    est = Est()
    if e.provider == "local":
        est.add(len(texts), 0, 0.0)
        return est
    tok = sum(embed_tokens(e, t) for t in texts)
    est.add(len(texts), tok, tok * price(e.model_id).input / 1e6)
    return est


def decision_tokens(d) -> int:
    if d.decider.startswith("jev"):
        return round(len(_CL100K.encode(json.dumps(d.payload["state"]) + json.dumps(d.payload["questions"]))) * JEV_FACTOR)
    q = d.payload["questions"]
    return round(len(_O200K.encode(d.payload["input"] + json.dumps(q))) * LUNA_FACTOR) * len(q)


def estimate_decisions(calls) -> dict[str, Est]:
    by: dict[str, Est] = defaultdict(Est)
    for d in calls:
        t = decision_tokens(d)
        by[d.decider].add(1, t, t * price(d.model).input / 1e6)
    return dict(by)


class BudgetExceeded(SystemExit):
    pass


def check_budget(estimated_usd: float, max_usd: float) -> None:
    if estimated_usd > max_usd:
        raise BudgetExceeded(
            f"ABORT: estimated cost ${estimated_usd:.4f} exceeds --max-usd ${max_usd:.2f}. No API calls were made.")


def print_table(rows: list[tuple[str, Est]], title: str) -> float:
    print(f"=== {title} ===")
    print(f"{'step':<54}{'calls/texts':>12}{'est. input tok':>16}{'est. USD':>11}")
    total = Est()
    for name, e in rows:
        print(f"{name:<54}{e.n:>12}{e.tokens:>16,}{'$' + format(e.usd, '.4f'):>11}")
        total.add(e.n, e.tokens, e.usd)
    print("-" * 93)
    print(f"{'TOTAL':<54}{total.n:>12}{total.tokens:>16,}{'$' + format(total.usd, '.4f'):>11}")
    return total.usd
