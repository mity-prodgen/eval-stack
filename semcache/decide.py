"""A3 decision-model check: Jev and GPT-6 Luna answer "Do these two queries need the same answer?"

One request per (decider, pair); P(yes) is Jev's noul for `same_answer` or Luna's predicate probability.
Results are cached like every other call (harness DiskCache, namespace "semcache"); api errors are not
cached and retry on the next run. Builders are pure so tests check request shapes with no network.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from harness.cache import DiskCache, cache_key
from harness.prices import usage_cost

from .common import DECIDER_MODELS, JEV_A3, JEV_MODEL, LUNA_MODEL, jev_state, luna_input, luna_questions


@dataclass
class Decision:
    decider: str
    model: str
    pair_id: str
    cached: str
    new: str
    key: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


def build(d: Decision) -> Decision:
    if d.decider == "jev":
        d.payload = {"state": jev_state(d.cached, d.new), "questions": JEV_A3}
        d.key = cache_key("a3-jev", d.model, d.payload)
    elif d.decider == "luna":
        d.payload = {"model": d.model, "input": luna_input(d.cached, d.new), "questions": luna_questions()}
        d.key = cache_key("a3-luna", d.payload)
    else:
        raise ValueError(d.decider)
    return d


def plan(pairs: list[dict], deciders: list[str]) -> list[Decision]:
    return [build(Decision(k, DECIDER_MODELS[k], p["id"], p["cached"], p["new"])) for p in pairs for k in deciders]


def _result(status, p_yes=None, input_tokens=None, cost=None, raw=None, error=None, model=None, latency_ms=None):
    return {"ok": status != "api_error", "status": status, "p_yes": p_yes, "input_tokens": input_tokens,
            "cost_usd": cost, "raw": raw, "error": error, "model": model, "latency_ms": latency_ms}


def parse_jev(raw: dict) -> dict:
    p = ((raw.get("answers") or {}).get("same_answer") or {}).get("noul")
    tok = (raw.get("usage") or {}).get("input_tokens")
    cost = usage_cost(JEV_MODEL, tok) if tok else None
    if p is None:
        return _result("parse_error", input_tokens=tok, cost=cost, raw=raw, error="no noul for same_answer",
                       model=raw.get("model"))
    return _result("ok", float(p), tok, cost, raw, model=raw.get("model"))


def parse_luna(raw: dict) -> dict:
    a = {x.get("name"): x for x in raw.get("answers", [])}.get("same_answer")
    tok = (raw.get("usage") or {}).get("input_tokens")
    cost = usage_cost(LUNA_MODEL, tok) if tok else None
    if a is None:
        return _result("parse_error", input_tokens=tok, cost=cost, raw=raw, error="no same_answer", model=raw.get("model"))
    if a.get("type") == "refusal":
        return _result("refusal", input_tokens=tok, cost=cost, raw=raw, model=raw.get("model"))
    if a.get("probability") is None:
        return _result("parse_error", input_tokens=tok, cost=cost, raw=raw, error="no probability", model=raw.get("model"))
    return _result("ok", float(a["probability"]), tok, cost, raw, model=raw.get("model"))


def run(calls: list[Decision], cache: DiskCache, max_workers: int = 8, luna_client: Any = None) -> dict[str, dict]:
    jev_local = threading.local()
    jev_clients, lock = [], threading.Lock()

    def jev():
        from harness.jev_client import JevClient
        c = getattr(jev_local, "c", None)
        if c is None:
            c = JevClient(model=JEV_MODEL, max_retries=6, timeout=120.0)
            c.__enter__()
            jev_local.c = c
            with lock:
                jev_clients.append(c)
        return c

    if luna_client is None and any(c.decider == "luna" for c in calls):
        from openai import OpenAI
        luna_client = OpenAI(max_retries=4, timeout=60.0)

    def one(d: Decision) -> tuple[str, dict]:
        def do() -> dict:
            start = monotonic()
            if d.decider == "jev":
                r = jev().ask(d.payload["state"], d.payload["questions"])
                if not r.ok:
                    return _result("api_error", error=r.error, latency_ms=r.latency_ms)
                out = parse_jev(r.raw_response)
            else:
                try:
                    out = parse_luna(luna_client.decisions.create(**d.payload).model_dump())
                except Exception as e:  # noqa: BLE001 - logged, not raised
                    return _result("api_error", error=f"{type(e).__name__}: {e}", latency_ms=(monotonic() - start) * 1000)
            out["latency_ms"] = (monotonic() - start) * 1000
            return out
        res, cached = cache.get_or_call(d.key, do)
        return d.key, {**res, "cached": cached}

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return dict(ex.map(one, list({c.key: c for c in calls}.values())))
    finally:
        for c in jev_clients:
            c.__exit__(None, None, None)
