"""Verified model IDs and prices, one place, each with its source and check date.

Every number here was read from the vendor's own docs on the date in
`checked`; nothing is copied from the task prompt. cost.py (dry-run estimates)
and the judge clients (actual cost from logged usage) both read from here, so
an estimate and a bill can never use different rates.

All prices are USD per million tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

CHECKED = "2026-10-13"


@dataclass(frozen=True)
class ModelPrice:
    model_id: str            # exactly what goes in the request
    display: str
    input: float             # standard input
    output: float
    cache_write_5m: float | None = None
    cache_read: float | None = None
    batch_discount: float = 1.0      # multiplier on input/output when sent through a batch API
    min_cacheable_tokens: int | None = None
    source: str = ""
    checked: str = CHECKED
    notes: str = ""
    # Prompt-length pricing (Haiku 5.5): a request whose whole prompt (incl. cache reads/writes) exceeds
    # this many tokens pays the `long` rates instead. None = flat pricing.
    long_context_over: int | None = None
    long: "ModelPrice | None" = None


MODELS: dict[str, ModelPrice] = {
    # --- TypeSafe Jev ---
    "jev-1.13.0": ModelPrice(
        model_id="jev-1.13.0",
        display="Jev 1.13.0 (TypeSafe)",
        input=0.042,
        output=0.0,
        source="https://docs.typesafe.ai/models",
        notes="$42 per billion input tokens, output free. jev-latest and jev-preview both alias jev-1.13.0; "
        "we pin the versioned ID.",
    ),
    # --- OpenAI ---
    "gpt-6-luna": ModelPrice(
        model_id="gpt-6-luna",
        display="GPT-6 Luna on /v1/decisions (public beta)",
        input=0.10,
        output=0.0,
        source="https://developers.openai.com/api/docs/guides/decisions",
        notes="Decisions API rate: input only, 'no cache-read, cache-write, or output-token charges'. "
        "The model page lists $0.10/$0.01/$0.125/$0.50 (input/cached/cache-write/output) for other endpoints - "
        "not used here. Batch is not documented for /v1/decisions, so no discount is assumed. "
        "Only snapshot listed is `gpt-6-luna` (no dated pin exists).",
    ),
    "gpt-6.1-sol": ModelPrice(
        model_id="gpt-6.1-sol",
        display="GPT-6.1 Sol (rewrite generator, not a judge)",
        input=2.00,
        output=10.00,
        cache_read=0.10,
        batch_discount=0.5,
        source="https://developers.openai.com/api/docs/models/gpt-6.1-sol",
        notes="Reasoning model: effort low/medium(default)/high/xhigh/max; none/minimal unsupported. "
        "Temperature not documented. Only snapshot listed is `gpt-6.1-sol`.",
    ),
    # --- Anthropic ---
    "claude-haiku-4-5-20251001": ModelPrice(
        model_id="claude-haiku-4-5-20251001",
        display="Claude Haiku 4.5",
        input=1.00,
        output=5.00,
        cache_write_5m=1.25,
        cache_read=0.10,
        batch_discount=0.5,
        min_cacheable_tokens=4096,
        source="https://platform.claude.com/docs/en/about-claude/pricing",
        notes="Legacy, still active; retirement 'not sooner than October 15, 2026', no deprecation notice yet "
        "(Anthropic gives >=60 days notice). Pinned dated snapshot = what post 2 ran "
        "(post 2 logged model claude-haiku-4-5-20251001). Min cacheable prompt 4,096 tokens, so caching "
        "never triggers for these short prompts.",
    ),
    "claude-haiku-5-5": ModelPrice(
        model_id="claude-haiku-5-5",
        display="Claude Haiku 5.5",
        input=0.10,
        output=0.50,
        cache_write_5m=0.125,
        cache_read=0.01,
        batch_discount=0.5,
        min_cacheable_tokens=512,
        source="https://platform.claude.com/docs/en/models/haiku-5-5/overview",
        notes="Released 2026-10-07; dateless ID is a pinned snapshot. Prices shown are for prompts <= 100,000 "
        "tokens (ours are ~1-3k). Sampling parameters return 400 - omit temperature. Adaptive thinking is on by "
        "default; thinking {'type': 'disabled'} is accepted at high effort or below (default effort medium). "
        "Newer tokenizer: ~30% more tokens than Haiku 4.5 for the same text.",
        long_context_over=100_000,
        long=None,  # filled below
    ),
    "claude-sonnet-5-5": ModelPrice(
        model_id="claude-sonnet-5-5",
        display="Claude Sonnet 5.5",
        input=2.00,
        output=10.00,
        cache_write_5m=2.50,
        cache_read=0.10,
        batch_discount=0.5,
        min_cacheable_tokens=512,
        source="https://platform.claude.com/docs/en/models/sonnet-5-5/overview",
        notes="Dateless ID is a pinned snapshot. temperature/top_p/top_k at non-default values return 400, "
        "so temperature 0 is impossible on this model. thinking 'disabled' returns 400; lowest setting is "
        "{'type': 'between_tools'} (no up-front thinking; tool-free requests return text only).",
    ),
}


object.__setattr__(MODELS["claude-haiku-5-5"], "long", ModelPrice(
    model_id="claude-haiku-5-5", display="Claude Haiku 5.5 (prompt > 100k tokens)", input=0.50, output=2.50,
    cache_write_5m=0.625, cache_read=0.05, batch_discount=0.5,
    source="https://platform.claude.com/docs/en/about-claude/pricing"))


def price(model: str) -> ModelPrice:
    try:
        return MODELS[model]
    except KeyError:
        raise KeyError(f"no verified price for {model!r}; add it to jev_harness/prices.py with a source") from None


def usage_cost(
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    batch: bool = False,
) -> float:
    """Cost in USD of one call from the API's own usage numbers.

    input_tokens means uncached input (Anthropic's `input_tokens` already excludes
    cache reads and writes). Batch discount stacks with the cache multipliers,
    per Anthropic's pricing page.
    """
    p = price(model)
    if p.long_context_over and input_tokens + cache_write_tokens + cache_read_tokens > p.long_context_over:
        p = p.long
    m = p.batch_discount if batch else 1.0
    total = input_tokens * p.input + output_tokens * p.output
    if cache_write_tokens:
        total += cache_write_tokens * (p.cache_write_5m if p.cache_write_5m is not None else p.input)
    if cache_read_tokens:
        total += cache_read_tokens * (p.cache_read if p.cache_read is not None else p.input)
    return total * m / 1_000_000
