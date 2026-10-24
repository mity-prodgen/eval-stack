"""Request builders, parsers and executors for every post 3 call: generation, labels, baseline judge, decisions.

Builders are pure (tests check request shapes with no network). Every result is one record:

    ok, status   "ok" | "refusal" | "parse_error" | "empty" | "api_error" (api_error is never cached)
    usage        {"input", "output", "cache_write", "cache_read", "reasoning"} from the API's own usage field
    cost_usd     harness.prices at the pinned model, batch discount applied when sent through a batch
    latency_ms   wall clock on our side, request sent -> response parsed (decisions: one attempt, see below)
  + per step:    text (gen) | label, conflict_noted (label) | p_pass (baseline) | p_yes (decision)

Decision calls are timed per attempt: the executor makes a single attempt and retries itself only on 429/5xx,
logging `attempts`, so a retried call is never mistaken for a slow one. The latency table uses a separate
sequential sample (run.py --step latency), not these throughput calls.

Cache keys hash the request body (+ a repeat tag for the noise and latency re-runs).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable

from harness.cache import DiskCache, cache_key
from harness.post1_judge import judge_user_text, p_from_verdict, passage_text
from harness.post2_judge import parse_judgment
from harness.prices import usage_cost

from .common import (BASELINE_MAX_TOKENS, BASELINE_MODEL, BASELINE_SYSTEM, DECIDER, DEEPINFRA_BASE_URL, GEMMA_EXTRA_BODY,
                     GEMMA_SAMPLING, GEN_MAX_TOKENS, GENERATORS, HAIKU_GEN_SETTINGS, LABEL_MAX_TOKENS, LABELERS, LABELS,
                     SOL_LABEL_EFFORT, SOL_MAX_OUTPUT_TOKENS, SONNET_SETTINGS, WIKI_LABELS, context_text, gen_prompt,
                     job1_state, job2_state, judge_passage, label_system, label_user_text, luna_questions,
                     noul_questions)


def chunks_for(item: dict, order: str) -> list[str]:
    """Key chunks at ranks 1 and 2 (newer first or older first), distractors at 3-6 in a fixed order."""
    c = item["chunks"]
    keys = [c["new"], c["old"]] if order == "new_first" else [c["old"], c["new"]]
    return keys + c["distractors"]


def context_for(item: dict, order: str) -> str:
    return context_text(chunks_for(item, order))


@dataclass
class Call:
    step: str          # gen | label | baseline | job1 | job2
    who: str           # generator, labeler, "baseline" or decision-model short name
    model: str
    provider: str      # anthropic | openai | deepinfra | jev | luna | systemone
    item_id: str
    order: str
    gen: str = ""      # label/baseline/job2: which generator wrote the answer
    repeat: str = ""   # "" = main run; "noise1"; "latency1".. for the re-runs
    payload: dict[str, Any] = field(default_factory=dict)
    key: str = ""

    @property
    def meta(self) -> dict[str, Any]:
        return {"step": self.step, "who": self.who, "model": self.model, "provider": self.provider}


def _key(kind: str, payload: dict, repeat: str = "") -> str:
    return cache_key(kind, payload) if not repeat else cache_key(kind, payload, {"repeat": repeat})


def provider_of(model: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gpt-6.1"):
        return "openai"
    if model.startswith("google/"):
        return "deepinfra"
    raise ValueError(model)


# =====================================================================================================
# Builders
# =====================================================================================================
def build_gen(item: dict, order: str, gen: str) -> Call:
    model = GENERATORS[gen]
    prompt = gen_prompt(item["question"], context_for(item, order))
    c = Call("gen", gen, model, provider_of(model), item["id"], order, gen=gen)
    if c.provider == "anthropic":
        c.payload = {"model": model, "max_tokens": GEN_MAX_TOKENS, **HAIKU_GEN_SETTINGS,
                     "messages": [{"role": "user", "content": prompt}]}
    else:
        c.payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": GEN_MAX_TOKENS,
                     **GEMMA_SAMPLING, "extra_body": GEMMA_EXTRA_BODY}
    c.key = _key("freshness-gen", c.payload)
    return c


def build_label(item: dict, answer: str, labeler: str, *, order: str = "", gen: str = "") -> Call:
    model = LABELERS[labeler]
    system, user = label_system(item), label_user_text(item, answer)
    c = Call("label", labeler, model, provider_of(model), item["id"], order, gen=gen)
    if c.provider == "anthropic":
        c.payload = {"model": model, "max_tokens": LABEL_MAX_TOKENS, **SONNET_SETTINGS, "system": system,
                     "messages": [{"role": "user", "content": user}]}
    else:
        c.payload = {"model": model, "instructions": system, "input": user, "reasoning": {"effort": SOL_LABEL_EFFORT},
                     "max_output_tokens": SOL_MAX_OUTPUT_TOKENS}
    c.key = _key("freshness-label", c.payload)
    return c


def build_baseline(item: dict, order: str, answer: str, *, gen: str = "") -> Call:
    """Post 1's P0 Claude faithfulness prompt (post 2's system prompt) on Sonnet 5.5, passage block cached."""
    passage = judge_passage(item, context_for(item, order))
    c = Call("baseline", "baseline", BASELINE_MODEL, "anthropic", item["id"], order, gen=gen)
    c.payload = {"model": BASELINE_MODEL, "max_tokens": BASELINE_MAX_TOKENS, **SONNET_SETTINGS,
                 "system": BASELINE_SYSTEM,
                 "messages": [{"role": "user", "content": [
                     {"type": "text", "text": f"SOURCE PASSAGE:\n{passage_text(passage)}\n\n",
                      "cache_control": {"type": "ephemeral"}},
                     {"type": "text", "text": f"RESPONSE:\n{answer}"}]}]}
    c.key = _key("freshness-baseline", c.payload)
    return c


def build_decision(item: dict, order: str, decider: str, job: str, answer: str | None = None, *, gen: str = "",
                   repeat: str = "") -> Call:
    d = DECIDER[decider]
    ctx = context_for(item, order)
    state = job1_state(item["question"], ctx) if job == "job1" else job2_state(item["question"], ctx, answer or "")
    c = Call(job, decider, d.model, d.api, item["id"], order, gen=gen, repeat=repeat)
    if d.api == "luna":
        c.payload = {"model": d.model, "input": state, "questions": luna_questions(job)}
    else:
        c.payload = {"state": state, "questions": noul_questions(job)}
        if d.send_model:
            c.payload = {"model": d.model, **c.payload}
    key_body = c.payload if (d.api == "luna" or d.send_model) else {"model": d.model, "revision": d.revision, **c.payload}
    c.key = _key(f"freshness-{job}-{decider}", key_body, repeat)
    return c


# =====================================================================================================
# Parsers
# =====================================================================================================
def _usage(input_=0, output=0, cache_write=0, cache_read=0, reasoning=0) -> dict[str, int]:
    return {"input": input_ or 0, "output": output or 0, "cache_write": cache_write or 0,
            "cache_read": cache_read or 0, "reasoning": reasoning or 0}


def _rec(status: str, **kw) -> dict[str, Any]:
    return {"ok": status != "api_error", "status": status, **kw}


def usage_and_text(raw: dict, provider: str, model: str, batch: bool) -> tuple[dict, float, str, str | None]:
    """(usage, cost, text, stop) for the three text-generating APIs."""
    if provider == "anthropic":
        u = raw.get("usage") or {}
        usage = _usage(u.get("input_tokens"), u.get("output_tokens"), u.get("cache_creation_input_tokens"),
                       u.get("cache_read_input_tokens"))
        text = "".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")
        stop = raw.get("stop_reason")
    elif provider == "openai":
        u = raw.get("usage") or {}
        cached = (u.get("input_tokens_details") or {}).get("cached_tokens") or 0
        usage = _usage((u.get("input_tokens") or 0) - cached, u.get("output_tokens"), 0, cached,
                       (u.get("output_tokens_details") or {}).get("reasoning_tokens"))
        text = "".join(c.get("text", "") for o in raw.get("output", []) if o.get("type") == "message"
                       for c in o.get("content", []) if c.get("type") == "output_text")
        stop = (raw.get("incomplete_details") or {}).get("reason") or raw.get("status")
    else:  # deepinfra chat completions
        u = raw.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        usage = _usage((u.get("prompt_tokens") or 0) - cached, u.get("completion_tokens"), 0, cached)
        ch = (raw.get("choices") or [{}])[0]
        text = (ch.get("message") or {}).get("content") or ""
        stop = ch.get("finish_reason")
    cost = usage_cost(model, usage["input"], usage["output"], usage["cache_write"], usage["cache_read"], batch=batch)
    return usage, cost, text, stop


_EMPTY_THOUGHT = re.compile(r"^\s*<\|channel>thought\s*<channel\|>\s*")


def parse_gen(raw: dict, meta: dict) -> dict:
    usage, cost, text, stop = usage_and_text(raw, meta["provider"], meta["model"], meta.get("batch", False))
    text = _EMPTY_THOUGHT.sub("", text)   # Gemma 4 31B emits an empty thought block with thinking off
    common = {"usage": usage, "cost_usd": cost, "raw": raw, "model": raw.get("model"), "stop": stop,
              "batch": meta.get("batch", False)}
    if stop == "refusal":
        return _rec("refusal", text=text, **common)
    if not text.strip():
        return _rec("empty", text="", **common)
    return _rec("ok", text=text.strip(), **common)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_label_text(text: str, wiki: bool) -> tuple[str, bool]:
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"no JSON object in label output: {text!r}")
    d = json.loads(m.group(0))
    label = str(d["label"]).strip().lower()
    if label not in (WIKI_LABELS if wiki else LABELS):
        raise ValueError(f"unknown label {label!r}")
    noted = d["conflict_noted"]
    if not isinstance(noted, bool):
        raise ValueError(f"conflict_noted not a bool: {noted!r}")
    return label, noted


def parse_label(raw: dict, meta: dict) -> dict:
    usage, cost, text, stop = usage_and_text(raw, meta["provider"], meta["model"], meta.get("batch", False))
    common = {"usage": usage, "cost_usd": cost, "raw": raw, "model": raw.get("model"), "batch": meta.get("batch", False)}
    if stop == "refusal":
        return _rec("refusal", **common)
    try:
        label, noted = parse_label_text(text, meta.get("wiki", False))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        return _rec("parse_error", error=str(e), **common)
    return _rec("ok", label=label, conflict_noted=noted, **common)


def parse_baseline(raw: dict, meta: dict) -> dict:
    usage, cost, text, stop = usage_and_text(raw, "anthropic", meta["model"], meta.get("batch", False))
    common = {"usage": usage, "cost_usd": cost, "raw": raw, "model": raw.get("model"), "batch": meta.get("batch", False)}
    if stop == "refusal":
        return _rec("refusal", **common)
    try:
        hallucinated, conf = parse_judgment(text)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        return _rec("parse_error", error=str(e), **common)
    return _rec("ok", p_pass=p_from_verdict(not hallucinated, conf), **common)


def parse_decision(raw: dict, meta: dict) -> dict:
    """P(yes) from a System One noul (Jev, Perplexity, Clef, Strands) or a Luna predicate. Cloudflare's REST API
    wraps the body in {"result": ..., "success": ...}; both shapes are read."""
    job, model = meta["step"], meta["model"]
    body = raw.get("result") if isinstance(raw.get("result"), dict) and "answers" in raw["result"] else raw
    tok = (body.get("usage") or {}).get("input_tokens") or 0
    cost = 0.0 if meta.get("local") else (usage_cost(model, tok) if tok else None)
    common = {"usage": _usage(tok), "cost_usd": cost, "raw": raw, "model": body.get("model"),
              "server_latency_ms": body.get("latency_ms")}
    answers = body.get("answers")
    if isinstance(answers, list):   # Luna
        a = {x.get("name"): x for x in answers}.get(job) or {}
        if a.get("type") == "refusal":
            return _rec("refusal", p_yes=None, **common)
        p = a.get("probability")
    else:
        p = ((answers or {}).get(job) or {}).get("noul")
    if p is None:
        return _rec("parse_error", p_yes=None, error=f"no P(yes) for {job}", **common)
    return _rec("ok", p_yes=float(p), **common)


PARSERS: dict[str, Callable[[dict, dict], dict]] = {"gen": parse_gen, "label": parse_label,
                                                    "baseline": parse_baseline, "job1": parse_decision,
                                                    "job2": parse_decision}


# =====================================================================================================
# Executors
# =====================================================================================================
def _meta(c: Call, items_by_id: dict) -> dict:
    m = c.meta
    if c.step == "label":
        m["wiki"] = items_by_id[c.item_id]["kind"] == "wiki"
    if c.step in ("job1", "job2"):
        m["local"] = DECIDER[c.who].local
    return m


def systemone_url(decider: str) -> str:
    d = DECIDER[decider]
    return d.url.replace("{CLOUDFLARE_ACCOUNT_ID}", os.environ.get("CLOUDFLARE_ACCOUNT_ID", "{CLOUDFLARE_ACCOUNT_ID}"))


RETRY_STATUS = (429, 500, 502, 503, 504)


def run_calls(calls: list[Call], cache: DiskCache, items_by_id: dict, batch_dir, *, use_batch: bool = True,
              max_workers: int = 8, log: Callable[[str], None] = print, clients: dict | None = None,
              sequential: bool = False) -> dict[str, dict]:
    """Run every uncached call, grouped by provider (and by decision model). Returns {key: record (+ "cached")}.
    sequential=True: one call at a time, no automatic retry (the latency sample)."""
    from harness.batch import anthropic_batch, openai_batch
    clients = clients or {}
    uniq = list({c.key: c for c in calls}.values())
    out: dict[str, dict] = {}
    groups: dict[tuple, list[Call]] = {}
    for c in uniq:
        groups.setdefault((c.provider, c.who if c.step in ("job1", "job2") else ""), []).append(c)
    for (provider, who), cs in groups.items():
        if provider in ("anthropic", "openai") and use_batch and not sequential:
            reqs = [(c.key, c.payload, _meta(c, items_by_id)) for c in cs]
            parse = lambda raw, m: PARSERS[m["step"]](raw, m)  # noqa: E731
            fn = anthropic_batch if provider == "anthropic" else openai_batch
            out.update(fn(reqs, cache, batch_dir, parse, client=clients.get(provider), log=log))
            continue
        workers = 1 if (sequential or (who and DECIDER[who].local)) else max_workers
        out.update(_run_sync(provider, who, cs, cache, items_by_id, workers, clients, sequential))
    return out


def _run_sync(provider, who, cs, cache, items_by_id, max_workers, clients, sequential) -> dict[str, dict]:
    local, made, lock = threading.local(), [], threading.Lock()
    retries = 0 if sequential else 4

    def jev(model):
        from harness.jev_client import JevClient
        by = local.__dict__.setdefault("by_model", {})
        if model not in by:
            cl = JevClient(model=model, max_retries=retries, timeout=120.0)
            cl.__enter__()
            by[model] = cl
            with lock:
                made.append(cl)
        return by[model]

    client = clients.get(provider)
    if client is None and provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(max_retries=retries)
    elif client is None and provider in ("openai", "luna"):
        from openai import OpenAI
        client = OpenAI(max_retries=retries, timeout=120.0)
    elif client is None and provider == "deepinfra":
        from openai import OpenAI
        client = OpenAI(base_url=DEEPINFRA_BASE_URL, api_key=os.environ.get("DEEPINFRA_API_KEY"),
                        max_retries=retries, timeout=120.0)
    elif client is None and provider == "systemone":
        import httpx
        client = httpx.Client(timeout=120.0)
        made.append(client)

    def systemone(c: Call) -> tuple[dict, int]:
        d = DECIDER[c.who]
        headers = {"content-type": "application/json"}
        if d.key_env:
            headers["authorization"] = f"Bearer {os.environ.get(d.key_env, '')}"
        attempts = 0
        while True:
            attempts += 1
            r = client.post(systemone_url(c.who), json=c.payload, headers=headers)
            if r.status_code in RETRY_STATUS and attempts <= retries:
                time.sleep(float(r.headers.get("retry-after") or 2 ** attempts))
                continue
            r.raise_for_status()
            return r.json(), attempts

    def one(c: Call) -> tuple[str, dict]:
        def do() -> dict:
            m = _meta(c, items_by_id)
            attempts = 1
            start = monotonic()
            try:
                if provider == "jev":
                    r = jev(c.model).ask(c.payload["state"], c.payload["questions"])
                    if not r.ok:
                        return _rec("api_error", error=r.error, latency_ms=r.latency_ms)
                    raw = r.raw_response
                elif provider == "luna":
                    raw = client.decisions.create(**c.payload).model_dump()
                elif provider == "systemone":
                    raw, attempts = systemone(c)
                elif provider == "anthropic":
                    raw = client.messages.create(**c.payload).to_dict()
                elif provider == "openai":
                    raw = client.responses.create(**c.payload).model_dump()
                else:
                    raw = client.chat.completions.create(**c.payload).model_dump()
            except Exception as e:  # noqa: BLE001 - logged, not cached, retried next run
                return _rec("api_error", error=f"{type(e).__name__}: {e}", latency_ms=(monotonic() - start) * 1000)
            elapsed = (monotonic() - start) * 1000
            res = PARSERS[c.step](raw, m)
            res.update({"latency_ms": elapsed, "attempts": attempts, "started_at": time.time() - elapsed / 1000})
            return res
        res, cached = cache.get_or_call(c.key, do)
        return c.key, {**res, "cached": cached}

    try:
        if max_workers == 1:
            return dict(one(c) for c in cs)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return dict(ex.map(one, cs))
    finally:
        for cl in made:
            cl.__exit__(None, None, None)


__all__ = ["Call", "build_baseline", "build_decision", "build_gen", "build_label", "chunks_for", "context_for",
           "run_calls"]
