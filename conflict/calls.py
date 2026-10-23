"""Request builders, parsers and executors for every post 3 call: pre-check, generation, labels, judges.

Builders are pure (tests check request shapes with no network). Every result is one record:

    ok, status   "ok" | "refusal" | "parse_error" | "empty" | "api_error" (api_error is never cached)
    usage        {"input", "output", "cache_write", "cache_read", "reasoning"} from the API's own usage field
    cost_usd     harness.prices at the pinned model, batch discount applied when sent through a batch
    raw          the full response body
  + per step:    text (gen) | p_conflict, newest (precheck) | label, conflict_noted (label) | p_pass (judge)

Cache keys hash the request body (+ the repeat index for the noise re-run), so an F2 prompt that gets no notice is
the F0 prompt and reuses the F0 answer, and an answer that two arms produce word for word is labeled once.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable

from harness.cache import DiskCache, cache_key
from harness.post1_judge import judge_user_text, p_from_verdict, passage_text
from harness.post2_judge import parse_judgment
from harness.prices import usage_cost

from .common import (CLAUDE_GEN_SETTINGS, CLAUDE_J0_SYSTEM, CLAUDE_J1_SYSTEM, CLAUDE_JUDGE_SETTINGS,
                     CLAUDE_LABEL_SETTINGS, DEEPINFRA_BASE_URL, GEN_MAX_TOKENS, GENERATORS, JEV_J0, JEV_J1,
                     JEV_PRECHECK, JUDGE_MAX_TOKENS, JUDGES, LABEL_MAX_TOKENS, LABELERS, LABELS, LUNA_J0, LUNA_J1,
                     LUNA_PRECHECK, PRECHECKERS, QWEN_EXTRA_BODY, QWEN_SAMPLING, SOL_GEN_EFFORT, SOL_LABEL_EFFORT,
                     WIKI_LABELS, context_text, gen_prompt, judge_passage, label_system, label_user_text,
                     precheck_input, precheck_state)

SOL_MAX_OUTPUT_TOKENS = 4000   # includes reasoning tokens; a cap against runaway reasoning, not a length target


def chunks_for(item: dict, order: str) -> list[str]:
    """Key chunks at ranks 1 and 2 (newer first or older first), distractors at 3-6 in a fixed order."""
    c = item["chunks"]
    keys = [c["new"], c["old"]] if order == "new_first" else [c["old"], c["new"]]
    return keys + c["distractors"]


def context_for(item: dict, order: str) -> str:
    return context_text(chunks_for(item, order))


@dataclass
class Call:
    step: str          # precheck | gen | label | judge
    who: str           # short name: generator, labeler, judge or pre-checker
    model: str
    provider: str      # anthropic | openai | deepinfra | jev | luna
    item_id: str
    order: str
    arm: str = ""      # gen: F0/F1/F2jev/F2luna; label/judge: the arm of the answer
    gen: str = ""      # label/judge: which generator wrote the answer
    rubric: str = ""   # judge: J0/J1
    repeat: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    key: str = ""

    @property
    def meta(self) -> dict[str, Any]:
        return {"step": self.step, "who": self.who, "model": self.model, "provider": self.provider}


def _key(kind: str, payload: dict, repeat: int = 0) -> str:
    return cache_key(kind, payload) if not repeat else cache_key(kind, payload, {"repeat": repeat})


def provider_of(model: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gpt-6.1"):
        return "openai"
    if model.startswith("jev-"):
        return "jev"
    if model == "gpt-6-luna":
        return "luna"
    if model.startswith("Qwen/"):
        return "deepinfra"
    raise ValueError(model)


# =====================================================================================================
# Builders
# =====================================================================================================
def build_precheck(item: dict, order: str, who: str) -> Call:
    model = PRECHECKERS[who]
    ctx = context_for(item, order)
    c = Call("precheck", who, model, provider_of(model), item["id"], order)
    if who == "jev":
        c.payload = {"state": precheck_state(item["question"], ctx), "questions": JEV_PRECHECK}
        c.key = _key("conflict-precheck-jev", {"model": model, **c.payload})
    else:
        c.payload = {"model": model, "input": precheck_input(item["question"], ctx), "questions": LUNA_PRECHECK}
        c.key = _key("conflict-precheck-luna", c.payload)
    return c


def build_gen(item: dict, order: str, gen: str, arm: str, notice: str | None = None, repeat: int = 0) -> Call:
    model = GENERATORS[gen]
    prompt = gen_prompt(item["question"], context_for(item, order), arm, notice)
    c = Call("gen", gen, model, provider_of(model), item["id"], order, arm=arm, gen=gen, repeat=repeat)
    if c.provider == "anthropic":
        c.payload = {"model": model, "max_tokens": GEN_MAX_TOKENS, **CLAUDE_GEN_SETTINGS[model],
                     "messages": [{"role": "user", "content": prompt}]}
    elif c.provider == "openai":
        c.payload = {"model": model, "input": prompt, "reasoning": {"effort": SOL_GEN_EFFORT},
                     "max_output_tokens": SOL_MAX_OUTPUT_TOKENS}
    else:
        c.payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": GEN_MAX_TOKENS,
                     **QWEN_SAMPLING, "extra_body": QWEN_EXTRA_BODY}
    c.key = _key("conflict-gen", c.payload, repeat)
    return c


def build_label(item: dict, answer: str, labeler: str, *, order: str = "", arm: str = "", gen: str = "",
                repeat: int = 0) -> Call:
    model = LABELERS[labeler]
    system, user = label_system(item), label_user_text(item, answer)
    c = Call("label", labeler, model, provider_of(model), item["id"], order, arm=arm, gen=gen, repeat=repeat)
    if c.provider == "anthropic":
        c.payload = {"model": model, "max_tokens": LABEL_MAX_TOKENS, **CLAUDE_LABEL_SETTINGS, "system": system,
                     "messages": [{"role": "user", "content": user}]}
    else:
        c.payload = {"model": model, "instructions": system, "input": user, "reasoning": {"effort": SOL_LABEL_EFFORT},
                     "max_output_tokens": SOL_MAX_OUTPUT_TOKENS}
    c.key = _key("conflict-label", c.payload)
    return c


def build_judge(item: dict, order: str, answer: str, judge: str, rubric: str, *, arm: str = "F0", gen: str = "") -> Call:
    model = JUDGES[judge]
    passage = judge_passage(item, context_for(item, order))
    c = Call("judge", judge, model, provider_of(model), item["id"], order, arm=arm, gen=gen, rubric=rubric)
    if judge == "jev":
        c.payload = {"state": {"source_passage": passage, "response": answer},
                     "questions": JEV_J0 if rubric == "J0" else JEV_J1}
        c.key = _key("conflict-judge-jev", {"model": model, **c.payload})
    elif judge == "luna":
        c.payload = {"model": model, "input": judge_user_text(passage, answer),
                     "questions": LUNA_J0 if rubric == "J0" else LUNA_J1}
        c.key = _key("conflict-judge-luna", c.payload)
    else:
        s = CLAUDE_JUDGE_SETTINGS[model]
        p: dict[str, Any] = {"model": model, "max_tokens": JUDGE_MAX_TOKENS,
                             "system": CLAUDE_J0_SYSTEM if rubric == "J0" else CLAUDE_J1_SYSTEM}
        if "thinking" in s:
            p["thinking"] = s["thinking"]
        if "temperature" in s:
            p["temperature"] = s["temperature"]
        if s.get("cache_passage"):
            p["messages"] = [{"role": "user", "content": [
                {"type": "text", "text": f"SOURCE PASSAGE:\n{passage_text(passage)}\n\n",
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": f"RESPONSE:\n{answer}"}]}]
        else:
            p["messages"] = [{"role": "user", "content": judge_user_text(passage, answer)}]
        c.payload = p
        c.key = _key("conflict-judge-claude", p)
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
        stop = raw.get("status") if raw.get("status") != "completed" else "completed"
        if (raw.get("incomplete_details") or {}).get("reason"):
            stop = raw["incomplete_details"]["reason"]
    else:  # deepinfra chat completions
        u = raw.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        usage = _usage((u.get("prompt_tokens") or 0) - cached, u.get("completion_tokens"), 0, cached)
        ch = (raw.get("choices") or [{}])[0]
        text = (ch.get("message") or {}).get("content") or ""
        stop = ch.get("finish_reason")
    cost = usage_cost(model, usage["input"], usage["output"], usage["cache_write"], usage["cache_read"], batch=batch)
    return usage, cost, text, stop


def parse_gen(raw: dict, meta: dict) -> dict:
    usage, cost, text, stop = usage_and_text(raw, meta["provider"], meta["model"], meta.get("batch", False))
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


def parse_precheck(raw: dict, meta: dict) -> dict:
    model = meta["model"]
    if meta["provider"] == "jev":
        ans = raw.get("answers") or {}
        p = (ans.get("conflict") or {}).get("noul")
        newest = (ans.get("newest") or {}).get("choice")
        probs = (ans.get("newest") or {}).get("probabilities")
    else:
        ans = {a.get("name"): a for a in raw.get("answers", [])}
        p = (ans.get("conflict") or {}).get("probability")
        newest = (ans.get("newest") or {}).get("choice")
        probs = (ans.get("newest") or {}).get("probabilities")
        if any((a or {}).get("type") == "refusal" for a in ans.values()):
            tok = (raw.get("usage") or {}).get("input_tokens") or 0
            return _rec("refusal", raw=raw, usage=_usage(tok), cost_usd=usage_cost(model, tok), model=raw.get("model"))
    tok = (raw.get("usage") or {}).get("input_tokens") or 0
    common = {"usage": _usage(tok), "cost_usd": usage_cost(model, tok), "raw": raw, "model": raw.get("model")}
    if p is None:
        return _rec("parse_error", error="no conflict probability", p_conflict=None, newest=newest, **common)
    return _rec("ok", p_conflict=float(p), newest=newest, newest_probs=probs, **common)


def parse_judge(raw: dict, meta: dict) -> dict:
    model, provider = meta["model"], meta["provider"]
    if provider == "jev":
        p = ((raw.get("answers") or {}).get("supported") or {}).get("noul")
        tok = (raw.get("usage") or {}).get("input_tokens") or 0
        common = {"usage": _usage(tok), "cost_usd": usage_cost(model, tok), "raw": raw, "model": raw.get("model")}
        return _rec("ok", p_pass=float(p), **common) if p is not None else _rec("parse_error", error="no noul", **common)
    if provider == "luna":
        a = {x.get("name"): x for x in raw.get("answers", [])}.get("supported") or {}
        tok = (raw.get("usage") or {}).get("input_tokens") or 0
        common = {"usage": _usage(tok), "cost_usd": usage_cost(model, tok), "raw": raw, "model": raw.get("model")}
        if a.get("type") == "refusal":
            return _rec("refusal", **common)
        p = a.get("probability")
        return _rec("ok", p_pass=float(p), **common) if p is not None else _rec("parse_error", error="no probability", **common)
    usage, cost, text, stop = usage_and_text(raw, "anthropic", model, meta.get("batch", False))
    common = {"usage": usage, "cost_usd": cost, "raw": raw, "model": raw.get("model"), "batch": meta.get("batch", False)}
    if stop == "refusal":
        return _rec("refusal", **common)
    try:
        hallucinated, conf = parse_judgment(text)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        return _rec("parse_error", error=str(e), **common)
    return _rec("ok", p_pass=p_from_verdict(not hallucinated, conf), **common)


PARSERS: dict[str, Callable[[dict, dict], dict]] = {"gen": parse_gen, "label": parse_label, "precheck": parse_precheck,
                                                    "judge": parse_judge}


# =====================================================================================================
# Executors
# =====================================================================================================
def _meta(c: Call, items_by_id: dict) -> dict:
    m = c.meta
    if c.step == "label":
        m["wiki"] = items_by_id[c.item_id]["kind"] == "wiki"
    return m


def _sync_kwargs(params: dict) -> dict:
    """anthropic>=1.0 removed `temperature` from messages.create; Haiku 4.5 still accepts it, so it goes in
    extra_body (the pinned SDK sends anthropic-version 2023-06-01, where it is valid)."""
    p = dict(params)
    return {**p, "extra_body": {"temperature": p.pop("temperature")}} if "temperature" in p else p


def _batch_params(params: dict) -> dict:
    return params   # batch requests carry raw API params, temperature included


def run_calls(calls: list[Call], cache: DiskCache, items_by_id: dict, batch_dir, *, use_batch: bool = True,
              max_workers: int = 8, log: Callable[[str], None] = print, clients: dict | None = None) -> dict[str, dict]:
    """Run every uncached call, grouped by provider. Returns {key: record (+ "cached")}."""
    from harness.batch import anthropic_batch, openai_batch
    clients = clients or {}
    uniq = list({c.key: c for c in calls}.values())
    out: dict[str, dict] = {}
    by_provider: dict[str, list[Call]] = {}
    for c in uniq:
        by_provider.setdefault(c.provider, []).append(c)

    def parse_for(c: Call):
        return PARSERS[c.step]

    for provider, cs in by_provider.items():
        if provider in ("anthropic", "openai") and use_batch:
            reqs = [(c.key, _batch_params(c.payload), _meta(c, items_by_id)) for c in cs]
            parse = lambda raw, m: PARSERS[m["step"]](raw, m)  # noqa: E731
            fn = anthropic_batch if provider == "anthropic" else openai_batch
            out.update(fn(reqs, cache, batch_dir, parse, client=clients.get(provider), log=log))
            continue
        out.update(_run_sync(provider, cs, cache, items_by_id, max_workers, clients, parse_for))
    return out


def _run_sync(provider, cs, cache, items_by_id, max_workers, clients, parse_for) -> dict[str, dict]:
    local, made, lock = threading.local(), [], threading.Lock()

    def jev(model):
        from harness.jev_client import JevClient
        by = local.__dict__.setdefault("by_model", {})
        if model not in by:
            cl = JevClient(model=model, max_retries=6, timeout=120.0)
            cl.__enter__()
            by[model] = cl
            with lock:
                made.append(cl)
        return by[model]

    client = clients.get(provider)
    if client is None and provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(max_retries=4)
    elif client is None and provider in ("openai", "luna"):
        from openai import OpenAI
        client = OpenAI(max_retries=4, timeout=120.0)
    elif client is None and provider == "deepinfra":
        import os
        from openai import OpenAI
        client = OpenAI(base_url=DEEPINFRA_BASE_URL, api_key=os.environ.get("DEEPINFRA_API_KEY"), max_retries=4,
                        timeout=120.0)

    def one(c: Call) -> tuple[str, dict]:
        def do() -> dict:
            start = monotonic()
            m = _meta(c, items_by_id)
            try:
                if provider == "jev":
                    r = jev(c.model).ask(c.payload["state"], c.payload["questions"])
                    if not r.ok:
                        return _rec("api_error", error=r.error, latency_ms=r.latency_ms)
                    raw = r.raw_response
                elif provider == "luna":
                    raw = client.decisions.create(**c.payload).model_dump()
                elif provider == "anthropic":
                    raw = client.messages.create(**_sync_kwargs(c.payload)).to_dict()
                elif provider == "openai":
                    raw = client.responses.create(**c.payload).model_dump()
                else:
                    p = dict(c.payload)
                    raw = client.chat.completions.create(**p).model_dump()
            except Exception as e:  # noqa: BLE001 - logged, not cached, retried next run
                return _rec("api_error", error=f"{type(e).__name__}: {e}", latency_ms=(monotonic() - start) * 1000)
            res = parse_for(c)(raw, m)
            res["latency_ms"] = (monotonic() - start) * 1000
            return res
        res, cached = cache.get_or_call(c.key, do)
        return c.key, {**res, "cached": cached}

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return dict(ex.map(one, cs))
    finally:
        for cl in made:
            cl.__exit__(None, None, None)


__all__ = ["Call", "build_gen", "build_judge", "build_label", "build_precheck", "chunks_for", "context_for",
           "run_calls", "GENERATORS", "JUDGES", "LABELERS", "PRECHECKERS"]
