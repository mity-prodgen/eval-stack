"""Dry-run token and cost estimates for post 3, and the --max-usd gate (shared with semcache/).

Input tokens: tiktoken times a factor measured in earlier posts, rounded up; every choice leans high because the
estimate gates --max-usd.

  Claude Haiku 4.5        cl100k x 1.18           (post 1, measured on post 2's logs: max 1.18)
  Claude Haiku/Sonnet 5.5 cl100k x 1.59           (1.18 x 1.35 for the newer tokenizer, post 1)
  gpt-6.1-sol             o200k  x 1.20
  Qwen3.8-27B             o200k  x 1.30           (own tokenizer; no calibration yet - the pilot gives one)
  Jev 1.13.0              cl100k x 1.54 + 300     (post 1 p95 factor + post 2's measured per-request overhead)
  GPT-6 Luna              (o200k x 1.10 + 150) x questions, priced at the cache-write rate ($0.125, not $0.10)

Output tokens (guesses, high): answers 150 (the RAG prompt asks for three sentences at most), labels 30, Claude judge
verdicts 20. Reasoning on top: Sol at effort "low" 200 per call (post 1's rewrites at low: median 0, max 49; post 2's
generation at medium: p90 633 on much longer outputs). Sonnet 5.5 judges: the cacheable prefix is priced as a cache
write on every request (no hits assumed). Batch discount (50%) applies to Anthropic and OpenAI requests unless
--sync. Before answers exist, labels and judges are estimated on a placeholder answer of typical length, and F2 is
estimated as if the pre-check fired on every context (upper bound).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import tiktoken

from harness.prices import price
from semcache.estimate import BudgetExceeded, check_budget  # noqa: F401 - one gate for every post

_CL100K = tiktoken.get_encoding("cl100k_base")
_O200K = tiktoken.get_encoding("o200k_base")

IN_FACTOR = {"claude-haiku-4-5-20251001": 1.18, "claude-haiku-5-5": 1.59, "claude-sonnet-5-5": 1.59,
             "gpt-6.1-sol": 1.20, "Qwen/Qwen3.8-27B": 1.30}
JEV_FACTOR, JEV_OVERHEAD = 1.54, 300
LUNA_FACTOR, LUNA_OVERHEAD = 1.10, 150
OUT = {"gen": 150, "label": 30, "judge": 20}
SOL_REASONING = 200
PLACEHOLDER_ANSWER = ("The refund window is 21 days from delivery for unused items in their original packaging, "
                      "according to the newest version of the policy (v4.0). An older version listed 30 days, but it "
                      "has been replaced.")


@dataclass
class Est:
    n: int = 0
    input: int = 0
    output: int = 0
    usd: float = 0.0

    def add(self, o: "Est") -> None:
        self.n += o.n
        self.input += o.input
        self.output += o.output
        self.usd += o.usd


def _enc(model: str):
    return _O200K if model.startswith(("gpt-", "Qwen/")) else _CL100K


def _text_of(payload: dict) -> str:
    parts = [payload.get("system", ""), payload.get("instructions", "")]
    inp = payload.get("input")
    if isinstance(inp, str):
        parts.append(inp)
    for m in payload.get("messages", []):
        c = m["content"]
        parts.append(c if isinstance(c, str) else "".join(b["text"] for b in c))
    return "".join(p for p in parts if isinstance(p, str))


def estimate_call(c, use_batch: bool = True) -> Est:
    p = price(c.model)
    if c.provider == "jev":
        t = round(len(_CL100K.encode(json.dumps(c.payload["state"]) + json.dumps(c.payload["questions"]))) * JEV_FACTOR) \
            + JEV_OVERHEAD
        return Est(1, t, 0, t * p.input / 1e6)
    if c.provider == "luna":
        q = c.payload["questions"]
        t = (round(len(_O200K.encode(c.payload["input"] + json.dumps(q))) * LUNA_FACTOR) + LUNA_OVERHEAD) * len(q)
        return Est(1, t, 0, t * (p.cache_write_5m or p.input) / 1e6)
    t_in = round(len(_enc(c.model).encode(_text_of(c.payload))) * IN_FACTOR[c.model])
    t_out = OUT[c.step] + (SOL_REASONING if c.provider == "openai" else 0)
    m = p.batch_discount if (use_batch and c.provider in ("anthropic", "openai")) else 1.0
    in_rate = p.input
    if c.provider == "anthropic" and c.step == "judge" and isinstance(c.payload["messages"][0]["content"], list):
        in_rate = p.cache_write_5m   # Sonnet judge: whole prompt priced as a cache write (upper bound)
    return Est(1, t_in, t_out, (t_in * in_rate + t_out * p.output) * m / 1e6)


def estimate(calls, use_batch: bool = True) -> dict[str, Est]:
    by: dict[str, Est] = defaultdict(Est)
    for c in calls:
        by[f"{c.step} {c.who} ({c.model})"].add(estimate_call(c, use_batch))
    return dict(by)


def print_table(rows: dict[str, Est], title: str) -> float:
    print(f"=== {title} ===")
    print(f"{'step':<52}{'calls':>8}{'est. input tok':>16}{'est. output tok':>17}{'est. USD':>11}")
    total = Est()
    for name in sorted(rows, key=lambda k: ("precheck gen label judge".split().index(k.split()[0]), k)):
        e = rows[name]
        print(f"{name:<52}{e.n:>8}{e.input:>16,}{e.output:>17,}{'$' + format(e.usd, '.4f'):>11}")
        total.add(e)
    print("-" * 104)
    print(f"{'TOTAL':<52}{total.n:>8}{total.input:>16,}{total.output:>17,}{'$' + format(total.usd, '.4f'):>11}")
    return total.usd
