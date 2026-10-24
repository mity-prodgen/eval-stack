"""Verified model IDs and prices, one place, each with its source and check date.

Every number here was read from the vendor's own docs on the date in
`checked`; nothing is copied from the task prompt. persuasion/estimate.py (dry-run
estimates) and the judge clients (actual cost from logged usage) both read from here, so
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
    "jev-1.14.0": ModelPrice(
        model_id="jev-1.14.0",
        display="Jev 1.14.0 (TypeSafe)",
        input=0.042,
        output=0.0,
        source="https://docs.typesafe.ai/models",
        checked="2026-10-24",
        notes="Released 2026-10-14 as jev-preview; on 2026-10-24 both jev-latest and jev-preview point to it. "
        "Same price as 1.13.0. Pinned versioned ID.",
    ),
    # --- OpenAI ---
    "gpt-6-luna": ModelPrice(
        model_id="gpt-6-luna",
        display="GPT-6 Luna on /v1/decisions (public beta)",
        input=0.10,
        output=0.0,
        cache_write_5m=0.125,
        cache_read=0.01,
        source="https://developers.openai.com/api/docs/guides/decisions",
        checked="2026-10-22",
        notes="Decisions API: uncached input $0.10/M, output not charged. On 2026-10-13 the guide said 'no cache-read, "
        "cache-write, or output-token charges'; on 2026-10-22 it says cache hits get discounted input pricing and cache "
        "writes are charged at 'the standard gpt-6-luna prompt caching rates' (model page: $0.01 cached, $0.125 cache "
        "write). Estimates price all Luna input at the cache-write rate (upper bound). Batch is not documented for "
        "/v1/decisions, so no discount is assumed. Only snapshot listed is `gpt-6-luna` (no dated pin exists).",
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
        "Temperature not documented. Only snapshot listed is `gpt-6.1-sol`. Re-checked 2026-10-22 for post 3: "
        "unchanged ($2 / $0.10 cached / $2.50 cache write / $10; Batch API 50% off; Responses and Batch supported).",
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
    # --- post 3 (freshness/): open generator and decision models, checked 2026-10-24 ---
    "google/gemma-4-31B-it": ModelPrice(
        model_id="google/gemma-4-31B-it",
        display="Gemma 4 31B-it on DeepInfra",
        input=0.20,
        output=0.40,
        source="https://deepinfra.com/google/gemma-4-31B-it",
        checked="2026-10-24",
        notes="Open weights, Apache-2.0 (huggingface.co/google/gemma-4-31B-it, created 2026-03-11; OpenRouter lists "
        "it from 2026-04-02). Served fp8 on DeepInfra; no cached-input price listed. Thinking is off unless the "
        "system prompt starts with <|think|>; with it off the 31B still emits an empty thought block. Model card "
        "sampling: temperature 1.0, top_p 0.95, top_k 64. No batch API.",
    ),
    "pplx-decider-v1.2": ModelPrice(
        model_id="pplx-decider-v1.2",
        display="Perplexity pplx-decider-v1.2 (Decisions API)",
        input=0.02,
        output=0.0,
        source="https://docs.perplexity.ai/docs/decisions/quickstart",
        checked="2026-10-24",
        notes="POST https://api.perplexity.ai/v1/decisions. Released 2026-10-15 (Hugging Face perplexity-ai/"
        "pplx-decider-v1.2, Apache-2.0, Qwen3.8-27B base). The task named pplx-decider-v1-27b; the API routes that "
        "legacy name (and v1.1-27b) to v1.2, so v1.2 is pinned explicitly. Input only, no per-request fee. 10 req/s per "
        "organization shared by both deciders.",
    ),
    "pplx-decider-flash-v1.2": ModelPrice(
        model_id="pplx-decider-flash-v1.2",
        display="Perplexity pplx-decider-flash-v1.2 (Decisions API)",
        input=0.01,
        output=0.0,
        source="https://docs.perplexity.ai/docs/decisions/quickstart",
        checked="2026-10-24",
        notes="Substitute for Microsoft-Decision-1 (not reachable: Azure quota). Released 2026-10-15, Apache-2.0, "
        "Qwen3.5-4B base (Hugging Face perplexity-ai/pplx-decider-flash-v1.2).",
    ),
    "clef": ModelPrice(
        model_id="clef",
        display="Cloudflare Clef (@cf/cloudflare/clef, Workers AI)",
        input=0.24,
        output=0.0,
        source="https://developers.cloudflare.com/workers-ai/models/clef/",
        checked="2026-10-24",
        notes="27B, Qwen3.8-27B base, Apache-2.0, released 2026-10-01. System One request body; 65,536-token context; "
        "long text state is truncated to fit. Workers AI's free daily Neuron allowance is ignored (upper bound).",
    ),
    "clef-flash": ModelPrice(
        model_id="clef-flash",
        display="Cloudflare Clef-flash (@cf/cloudflare/clef-flash, Workers AI)",
        input=0.038,
        output=0.0,
        source="https://developers.cloudflare.com/workers-ai/models/clef-flash/",
        checked="2026-10-24",
        notes="9B, Qwen3.5-9B base, Apache-2.0, released 2026-10-01. 24,576-token context.",
    ),
    # --- Embedding models (semcache/, checked 2026-10-19). Input-only pricing. ---
    "text-embedding-3-small": ModelPrice(
        model_id="text-embedding-3-small",
        display="OpenAI text-embedding-3-small",
        input=0.02,
        output=0.0,
        source="https://developers.openai.com/api/docs/pricing",
        checked="2026-10-19",
        notes="1536 dims by default. Pricing page lists no batch price for embeddings, so none is assumed.",
    ),
    "text-embedding-3-large": ModelPrice(
        model_id="text-embedding-3-large",
        display="OpenAI text-embedding-3-large",
        input=0.13,
        output=0.0,
        source="https://developers.openai.com/api/docs/pricing",
        checked="2026-10-19",
        notes="3072 dims by default.",
    ),
    "embed-v4.0": ModelPrice(
        model_id="embed-v4.0",
        display="Cohere Embed v4",
        input=0.12,
        output=0.0,
        source="https://docs.cohere.com/docs/cohere-embed (model); price NOT on cohere.com/pricing",
        checked="2026-10-19",
        notes="UNVERIFIED PRICE. cohere.com/pricing lists only Model Vault hourly rates (Embed 5) and no per-token "
        "rate for embed-v4.0; $0.12/M is the rate third-party trackers and the Bedrock/Azure listings show. Used "
        "for estimates only; actual cost = billed_units.input_tokens x this rate. 1536 dims by default. Newer "
        "embed-v5.0-pro / embed-v5.0-fast exist; the task names v4.",
    ),
    "voyage-4": ModelPrice(
        model_id="voyage-4",
        display="Voyage 4",
        input=0.06,
        output=0.0,
        source="https://docs.voyageai.com/docs/pricing",
        checked="2026-10-19",
        notes="'Optimized for general-purpose and multilingual retrieval quality' (docs.voyageai.com/docs/embeddings). "
        "First 200M tokens free per account, so the bill is likely $0; the estimate ignores the allowance. "
        "1024 dims by default.",
    ),
    "gemini-embedding-2": ModelPrice(
        model_id="gemini-embedding-2",
        display="Gemini Embedding 2",
        input=0.20,
        output=0.0,
        batch_discount=0.5,
        source="https://ai.google.dev/gemini-api/docs/pricing",
        checked="2026-10-19",
        notes="Paid tier text input; free-tier data is used to improve Google products, so run on the paid tier. "
        "Stable model code per ai.google.dev/gemini-api/docs/models/gemini-embedding-2. No task_type; similarity "
        "uses the documented prefix 'task: sentence similarity | query: '. 3072 dims by default.",
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
        raise KeyError(f"no verified price for {model!r}; add it to harness/prices.py with a source") from None


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
