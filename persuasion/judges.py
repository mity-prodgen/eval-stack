"""Request builders, response parsers and cached executors for the four judges.

Every judge's result is normalised to one record shape:

    status     "ok" | "refusal" | "parse_error" | "api_error"
    p_pass     P(answer passes) in [0, 1], or None
    p_guard    P(guard question true) in [0, 1], or None (guard requests / P0G only)
    usage      {"input", "output", "cache_write", "cache_read"} tokens from the API's own usage field
    cost_usd   from harness.prices and the logged usage
    raw        the full response body (always logged)

Refusals and parse errors are records like any other - nothing is dropped here; analysis
counts them per judge/arm/condition. Only status "ok"/"refusal"/"parse_error" results are
cached (an api_error is transient by assumption and is retried on the next run, the same rule
harness.cache.DiskCache already applies via result["ok"]).

Builders are pure functions so tests can check request shapes without any network.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable

from common import (CLAUDE_MAX_TOKENS, CLAUDE_SETTINGS, JEV_MODEL, LUNA_MODEL, claude_system,
                    jev_questions, jev_state, judge_user_text, luna_questions, p_from_verdict, passage_text)
from harness.cache import DiskCache, cache_key
from harness.post2_judge import parse_judgment as parse_post2_judgment
from harness.prices import usage_cost


@dataclass
class Call:
    """One planned API request. `kind` is "judge" (P0/P1/P0G judgment) or "guard" (Claude guard)."""
    judge: str
    model: str
    arm: str
    kind: str
    item_id: str
    condition: str
    passage: Any
    response: str
    key: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def rid(self) -> str:
        return f"{self.judge}|{self.arm}|{self.kind}|{self.item_id}|{self.condition}"


def _usage(input_=0, output=0, cache_write=0, cache_read=0) -> dict[str, int]:
    return {"input": input_ or 0, "output": output or 0, "cache_write": cache_write or 0, "cache_read": cache_read or 0}


def _result(status: str, *, p_pass=None, p_guard=None, usage=None, cost=None, raw=None, error=None,
            model=None, latency_ms=None, batch=False) -> dict[str, Any]:
    return {"ok": status != "api_error", "status": status, "p_pass": p_pass, "p_guard": p_guard,
            "usage": usage or _usage(), "cost_usd": cost, "raw": raw, "error": error, "model": model,
            "latency_ms": latency_ms, "batch": batch}


# =====================================================================================
# Jev
# =====================================================================================
def build_jev(call: Call) -> Call:
    call.payload = {"state": jev_state(call.passage, call.response), "questions": jev_questions(call.arm)}
    call.key = cache_key("jev", call.model, call.payload["state"], call.payload["questions"])
    return call


def parse_jev(raw: dict[str, Any] | None, arm: str) -> dict[str, Any]:
    answers = (raw or {}).get("answers", {})
    sup = (answers.get("supported") or {}).get("noul")
    usage = _usage(input_=((raw or {}).get("usage") or {}).get("input_tokens"))
    cost = usage_cost(JEV_MODEL, usage["input"]) if usage["input"] else None
    if sup is None:
        return _result("parse_error", usage=usage, cost=cost, raw=raw, error="no noul for 'supported'",
                       model=(raw or {}).get("model"))
    p_guard = None
    if arm == "P0G":
        p_guard = (answers.get("guard") or {}).get("noul")
        if p_guard is None:
            return _result("parse_error", p_pass=sup, usage=usage, cost=cost, raw=raw,
                           error="no noul for 'guard'", model=raw.get("model"))
    return _result("ok", p_pass=float(sup), p_guard=None if p_guard is None else float(p_guard),
                   usage=usage, cost=cost, raw=raw, model=raw.get("model"))


def run_jev(calls: list[Call], cache: DiskCache, max_workers: int = 8) -> dict[str, dict]:
    from harness.jev_client import JevClient

    local = threading.local()
    clients: list[JevClient] = []
    lock = threading.Lock()

    def client() -> JevClient:
        c = getattr(local, "c", None)
        if c is None:
            c = JevClient(model=JEV_MODEL, max_retries=6, timeout=120.0)
            c.__enter__()
            local.c = c
            with lock:
                clients.append(c)
        return c

    def one(call: Call) -> tuple[str, dict]:
        def do() -> dict:
            r = client().ask(call.payload["state"], call.payload["questions"])
            if not r.ok:
                return _result("api_error", error=r.error, latency_ms=r.latency_ms)
            out = parse_jev(r.raw_response, call.arm)
            out["latency_ms"] = r.latency_ms
            return out
        res, cached = cache.get_or_call(call.key, do)
        return call.key, {**res, "cached": cached}

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return dict(ex.map(one, calls))
    finally:
        for c in clients:
            c.__exit__(None, None, None)


# =====================================================================================
# GPT-6 Luna (OpenAI /v1/decisions)
# =====================================================================================
def build_luna(call: Call) -> Call:
    call.payload = {"model": call.model, "input": judge_user_text(call.passage, call.response),
                    "questions": luna_questions(call.arm)}
    call.key = cache_key("luna", call.payload)
    return call


def parse_luna(raw: dict[str, Any], arm: str) -> dict[str, Any]:
    answers = {a.get("name"): a for a in raw.get("answers", [])}
    u = raw.get("usage") or {}
    usage = _usage(input_=u.get("input_tokens"), output=u.get("output_tokens"))
    cost = usage_cost(LUNA_MODEL, usage["input"])  # decisions: input-only pricing
    model = raw.get("model")

    def prob(name: str) -> tuple[str, float | None]:
        a = answers.get(name)
        if a is None:
            return "parse_error", None
        if a.get("type") == "refusal":
            return "refusal", None
        if a.get("type") != "predicate" or a.get("probability") is None:
            return "parse_error", None
        return "ok", float(a["probability"])

    s_status, p_pass = prob("supported")
    p_guard, g_status = None, "ok"
    if arm == "P0G":
        g_status, p_guard = prob("guard")
    status = s_status if s_status != "ok" else g_status
    err = None if status == "ok" else f"supported={s_status}" + (f", guard={g_status}" if arm == "P0G" else "")
    return _result(status, p_pass=p_pass, p_guard=p_guard, usage=usage, cost=cost, raw=raw, error=err, model=model)


def run_luna(calls: list[Call], cache: DiskCache, max_workers: int = 8, client: Any = None) -> dict[str, dict]:
    if client is None:
        from openai import OpenAI
        client = OpenAI(max_retries=4, timeout=60.0)

    def one(call: Call) -> tuple[str, dict]:
        def do() -> dict:
            start = monotonic()
            try:
                d = client.decisions.create(**call.payload)
            except Exception as e:  # noqa: BLE001 - logged, not raised, like JevClient.ask()
                return _result("api_error", error=f"{type(e).__name__}: {e}", latency_ms=(monotonic() - start) * 1000)
            out = parse_luna(d.model_dump(), call.arm)
            out["latency_ms"] = (monotonic() - start) * 1000
            return out
        res, cached = cache.get_or_call(call.key, do)
        return call.key, {**res, "cached": cached}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return dict(ex.map(one, calls))


# =====================================================================================
# Claude (Haiku 4.5, Sonnet 5.5) - sync or Message Batches
# =====================================================================================
def build_claude(call: Call) -> Call:
    """Haiku 4.5: post-2 request shape (single user string) + temperature 0.
    Haiku 5.5 / Sonnet 5.5: no temperature (400), up-front thinking off (see CLAUDE_SETTINGS), and the
    source passage is its own cache_control block so repeated passages can hit the prompt cache."""
    settings = CLAUDE_SETTINGS[call.model]
    system = claude_system("guard" if call.kind == "guard" else call.arm)
    params: dict[str, Any] = {"model": call.model, "max_tokens": CLAUDE_MAX_TOKENS, "system": system}
    if "thinking" in settings:
        params["thinking"] = settings["thinking"]
    if "temperature" in settings:
        params["temperature"] = settings["temperature"]
    if settings.get("cache_passage"):
        params["messages"] = [{"role": "user", "content": [
            {"type": "text", "text": f"SOURCE PASSAGE:\n{passage_text(call.passage)}\n\n",
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"RESPONSE:\n{call.response}"},
        ]}]
    else:
        params["messages"] = [{"role": "user", "content": judge_user_text(call.passage, call.response)}]
    call.payload = params
    call.key = cache_key("claude", params)
    return call


_GUARD_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_guard_json(text: str) -> float:
    m = _GUARD_JSON_RE.search(text)
    if not m:
        raise ValueError(f"no JSON object found in guard output: {text!r}")
    d = json.loads(m.group(0))
    return p_from_verdict(bool(d["guard"]), float(d["confidence"]) / 100.0)


def parse_claude(raw: dict[str, Any], kind: str, batch: bool, requested_model: str) -> dict[str, Any]:
    """requested_model prices the call: the model we pinned, not a guess from the response's model string."""
    model = raw.get("model")
    u = raw.get("usage") or {}
    usage = _usage(input_=u.get("input_tokens"), output=u.get("output_tokens"),
                   cache_write=u.get("cache_creation_input_tokens"), cache_read=u.get("cache_read_input_tokens"))
    cost = usage_cost(requested_model, usage["input"], usage["output"], usage["cache_write"], usage["cache_read"], batch=batch)
    common = {"usage": usage, "cost": cost, "raw": raw, "model": model, "batch": batch}
    if raw.get("stop_reason") == "refusal":
        return _result("refusal", error=f"stop_reason=refusal {raw.get('stop_details')}", **common)
    text = "".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")
    try:
        if kind == "guard":
            return _result("ok", p_guard=parse_guard_json(text), **common)
        hallucinated, conf = parse_post2_judgment(text)
        return _result("ok", p_pass=p_from_verdict(not hallucinated, conf), **common)
    except (ValueError, KeyError, TypeError) as e:
        return _result("parse_error", error=f"parse_error: {e}", **common)


def _claude_client():
    import anthropic
    return anthropic.Anthropic(max_retries=4)


def _sync_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """anthropic>=1.0 removed `temperature` from messages.create; it is still a valid API field for
    Haiku 4.5, so it travels in extra_body."""
    p = dict(params)
    extra = {}
    if "temperature" in p:
        extra["temperature"] = p.pop("temperature")
    return {**p, **({"extra_body": extra} if extra else {})}


def run_claude_sync(calls: list[Call], cache: DiskCache, max_workers: int = 4, client: Any = None) -> dict[str, dict]:
    client = client or _claude_client()

    def one(call: Call) -> tuple[str, dict]:
        def do() -> dict:
            start = monotonic()
            try:
                msg = client.messages.create(**_sync_kwargs(call.payload))
            except Exception as e:  # noqa: BLE001
                return _result("api_error", error=f"{type(e).__name__}: {e}", latency_ms=(monotonic() - start) * 1000)
            out = parse_claude(msg.to_dict(), call.kind, batch=False, requested_model=call.model)
            out["latency_ms"] = (monotonic() - start) * 1000
            return out
        res, cached = cache.get_or_call(call.key, do)
        return call.key, {**res, "cached": cached}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return dict(ex.map(one, calls))


def run_claude_batch(calls: list[Call], cache: DiskCache, batch_dir, poll_s: float = 30.0, client: Any = None,
                     log: Callable[[str], None] = print) -> dict[str, dict]:
    """Submit all uncached calls as one Message Batch (50% off), poll, cache results.

    The batch id is written to batch_dir before polling, so an interrupted run resumes the same
    batch on the next start instead of paying twice.
    """
    client = client or _claude_client()
    batch_dir.mkdir(parents=True, exist_ok=True)
    by_key = {c.key: c for c in calls}
    out: dict[str, dict] = {}

    for c in calls:
        hit = cache.get(c.key)
        if hit is not None:
            out[c.key] = {**hit, "cached": True}

    # resume any pending batch that covers some of these keys
    for f in sorted(batch_dir.glob("*.json")):
        meta = json.loads(f.read_text())
        if meta.get("collected") or not set(meta["keys"]) & (set(by_key) - set(out)):
            continue
        log(f"resuming batch {meta['id']}")
        _collect(client, meta, f, by_key, cache, out, poll_s, log)

    todo = [c for c in calls if c.key not in out]
    if todo:
        batch = client.messages.batches.create(
            requests=[{"custom_id": c.key, "params": c.payload} for c in todo])
        meta = {"id": batch.id, "keys": [c.key for c in todo], "created": time.time(), "collected": False,
                "requests": {c.key: {"model": c.model, "kind": c.kind} for c in todo}}
        f = batch_dir / f"{batch.id}.json"
        f.write_text(json.dumps(meta))
        log(f"submitted batch {batch.id} with {len(todo)} requests")
        _collect(client, meta, f, by_key, cache, out, poll_s, log)
    return out


def _collect(client, meta, meta_file, by_key, cache, out, poll_s, log) -> None:
    while True:
        b = client.messages.batches.retrieve(meta["id"])
        if b.processing_status == "ended":
            break
        log(f"  batch {meta['id']}: {b.processing_status} {b.request_counts}")
        time.sleep(poll_s)
    for entry in client.messages.batches.results(meta["id"]):
        key = entry.custom_id
        req = meta["requests"][key]  # recorded at submit time, so a resumed batch prices and parses correctly
        r = entry.result
        if r.type == "succeeded":
            res = parse_claude(r.message.to_dict(), req["kind"], batch=True, requested_model=req["model"])
            cache.set(key, res)  # ok / refusal / parse_error are all deterministic outcomes worth keeping
            out[key] = {**res, "cached": False}
        else:  # errored / canceled / expired: not cached, retried next run
            err = getattr(r, "error", None)
            out[key] = {**_result("api_error", error=f"batch {r.type}: {err}", batch=True), "cached": False}
    meta["collected"] = True
    meta_file.write_text(json.dumps(meta))


BUILDERS = {"jev": build_jev, "luna": build_luna, "haiku": build_claude, "haiku55": build_claude,
            "sonnet": build_claude}
