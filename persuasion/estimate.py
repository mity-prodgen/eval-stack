"""Dry-run cost estimates and the --max-usd guard.

Token counts use tiktoken as a proxy, scaled by factors measured on post 2's real usage logs
(post 2's judge_results_*.json, 600 calls each):

    Haiku 4.5   actual / cl100k estimate: mean 1.108, p95 1.149, max 1.180   -> 1.18
    Jev 1.13.0  actual / cl100k estimate: mean 1.363, p95 1.535, max 1.687   -> 1.54 (p95)
    Haiku output tokens: 16 on every call                                     -> 20

Sonnet 5.5 uses Anthropic's newer tokenizer; the models overview gives ~750k words per 1M tokens
before it and ~555k after, a ratio of 1.35, so Sonnet 5.5 and Haiku 5.5 (same tokenizer; Anthropic says ~30% more than
Haiku 4.5) = 1.18 x 1.35 = 1.59. OpenAI models are
estimated with o200k_base and a 1.2 safety factor (no measured calibration yet - the pilot will
give one). Every choice errs toward over-estimating, because the estimate gates --max-usd:

  * Luna: input is assumed billed once per question (unknown for /v1/decisions), so P0+G
    requests are priced at 2x input.
  * Sonnet / Haiku 5.5: every request whose cacheable prefix (system + passage) clears the 512-token minimum
    is priced as a cache *write* (1.25x); no cache hits are assumed.
  * gpt-6.1-sol: 1,500 reasoning tokens per call at effort=low (undocumented), and the
    worst case of 3 attempts for every item.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import tiktoken

from common import HAIKU55_MODEL, HAIKU_MODEL, JEV_MODEL, LUNA_MODEL, REWRITE_MODEL, SONNET_MODEL, passage_text
from harness.prices import price

_CL100K = tiktoken.get_encoding("cl100k_base")
_O200K = tiktoken.get_encoding("o200k_base")

FACTOR = {JEV_MODEL: 1.54, HAIKU_MODEL: 1.18, SONNET_MODEL: 1.18 * 1.35, HAIKU55_MODEL: 1.18 * 1.35,
          LUNA_MODEL: 1.2, REWRITE_MODEL: 1.2}
CLAUDE_OUT_TOKENS = 20
SOL_REASONING_TOKENS = 1500
SOL_OUTPUT_FACTOR = 1.3  # rewrite length vs original
SOL_MAX_ATTEMPTS = 3


def _tok(text: str, model: str) -> int:
    enc = _O200K if model in (LUNA_MODEL, REWRITE_MODEL) else _CL100K
    return round(len(enc.encode(text)) * FACTOR[model])


@dataclass
class Est:
    input: int = 0
    output: int = 0
    cache_write: int = 0
    usd: float = 0.0
    n: int = 0

    def add(self, other: "Est") -> None:
        self.input += other.input
        self.output += other.output
        self.cache_write += other.cache_write
        self.usd += other.usd
        self.n += other.n


def estimate_call(call, batch: bool = True) -> Est:
    """Estimated tokens and USD for one judges.Call (payload already built)."""
    p = price(call.model)
    if call.judge == "jev":
        t = _tok(json.dumps(call.payload["state"]) + json.dumps(call.payload["questions"]), call.model)
        return Est(input=t, usd=t * p.input / 1e6, n=1)
    if call.judge == "luna":
        q = call.payload["questions"]
        t = _tok(call.payload["input"] + json.dumps(q), call.model) * len(q)
        return Est(input=t, usd=t * p.input / 1e6, n=1)
    # Claude
    m = p.batch_discount if batch else 1.0
    system = call.payload["system"]
    msgs = call.payload["messages"][0]["content"]
    if isinstance(msgs, str):
        t = _tok(system + msgs, call.model)
        usd = (t * p.input + CLAUDE_OUT_TOKENS * p.output) * m / 1e6
        return Est(input=t, output=CLAUDE_OUT_TOKENS, usd=usd, n=1)
    prefix = _tok(system + msgs[0]["text"], call.model)
    rest = _tok(msgs[1]["text"], call.model)
    write = prefix if p.min_cacheable_tokens and prefix >= p.min_cacheable_tokens else 0
    plain = rest + (0 if write else prefix)
    usd = (plain * p.input + write * p.cache_write_5m + CLAUDE_OUT_TOKENS * p.output) * m / 1e6
    return Est(input=plain, output=CLAUDE_OUT_TOKENS, cache_write=write, usd=usd, n=1)


def estimate_rewrite(instructions: str, user_text: str, response: str, attempts: int = SOL_MAX_ATTEMPTS) -> Est:
    p = price(REWRITE_MODEL)
    t_in = _tok(instructions + user_text, REWRITE_MODEL)
    t_out = round(_tok(response, REWRITE_MODEL) * SOL_OUTPUT_FACTOR) + SOL_REASONING_TOKENS
    return Est(input=t_in * attempts, output=t_out * attempts,
               usd=attempts * (t_in * p.input + t_out * p.output) / 1e6, n=attempts)


class BudgetExceeded(SystemExit):
    pass


def check_budget(estimated_usd: float, max_usd: float) -> None:
    """Abort before any call if the estimate for the calls about to be made exceeds the cap."""
    if estimated_usd > max_usd:
        raise BudgetExceeded(
            f"ABORT: estimated cost ${estimated_usd:.4f} exceeds --max-usd ${max_usd:.2f}. No API calls were made. "
            "Raise --max-usd or narrow the run (--judge/--arm/--limit)."
        )


def print_table(rows: list[tuple[str, Est]], title: str) -> float:
    print(f"=== {title} ===")
    print(f"{'step':<30}{'calls':>7}{'est. input tok':>16}{'cache-write tok':>17}{'est. output tok':>17}{'est. USD':>11}")
    total = Est()
    for name, e in rows:
        print(f"{name:<30}{e.n:>7}{e.input:>16,}{e.cache_write:>17,}{e.output:>17,}{'$' + format(e.usd, '.4f'):>11}")
        total.add(e)
    print("-" * 98)
    print(f"{'TOTAL':<30}{total.n:>7}{total.input:>16,}{total.cache_write:>17,}{total.output:>17,}{'$' + format(total.usd, '.4f'):>11}")
    return total.usd


def passage_tokens(passage) -> int:
    return len(_CL100K.encode(passage_text(passage)))
