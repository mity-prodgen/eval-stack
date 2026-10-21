#!/usr/bin/env python3
"""Outside-model pairs: gpt-6.1-sol and Claude Sonnet 5.5 write traps and controls from the spec alone.

Why: the authored set (templates + handwritten, pairs.jsonl ids p000-p402) and the slot guard share one author.
These generators never see slots.py, the alias tables, or any authored pair - only the task's slot-type
descriptions - so results per source show how much the guard depends on its author's phrasing.

Three jobs per generator model (same prompts for both):
  traps     one request per slot type: 12 near-identical pairs that need different answers
            (class: airline only, 12; other slot types: 3 per domain)
  controls  one request per domain: 15 pairs that need the same answer (about half alias/format changes)
  banking77 20 real BANKING77 test queries (seeded; 40 in total, split between the models): for each, one
            minimal-edit trap and one same-answer rewording, the real query on the cached side

Every response is cached (harness DiskCache, namespace "semcache") and logged raw to results/generate_raw.jsonl.
Valid pairs are deduped against the authored set and each other and frozen to pairs/generated.jsonl; build_pairs
then merges them into pairs.jsonl with ids g000+ (authored ids do not change). Dry run by default.

    python -m semcache.generate_pairs                  # dry run: estimate
    python -m semcache.generate_pairs --live --max-usd 3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import tiktoken

from harness.cache import DiskCache, cache_key
from harness.prices import usage_cost

from .common import CACHE, PAIRS_DIR, RESULTS, ROOT, SEED

GENERATORS = {"sol": "gpt-6.1-sol", "sonnet": "claude-sonnet-5-5"}
SOL_EFFORT = "medium"           # the model's default
SONNET_MAX_TOKENS = 16000       # default adaptive thinking (effort high) left on: generation quality matters here
BANKING77 = ROOT / "data" / "banking77" / "test.csv"
N_BANKING77 = 40
GENERATED = PAIRS_DIR / "generated.jsonl"
DOMAINS = ("airline", "banking", "dev-tools", "support")
DOMAIN_TEXT = {"airline": "airline", "banking": "banking", "dev-tools": "developer tools",
               "support": "customer support for a software product"}

# Slot descriptions: the task's own wording and examples, nothing from the authored pairs or slots.py.
SLOTS = {
    "class": "the cabin class (for example Economy vs Premium Economy)",
    "direction": "the direction: origin and destination, or what is moved/converted from and to, are swapped "
                 "(for example DXB to LHR vs LHR to DXB)",
    "negation": "a negation (for example can vs can't)",
    "version": "a software or product version (for example Python 3.11 vs 3.12)",
    "date_time": "a date or a time",
    "number": "a number or amount",
    "entity": "a named entity, such as a brand, company, product or provider (for example Visa vs Mastercard)",
}

SYSTEM = (
    "You write test data for evaluating semantic caches in customer-facing assistants. A semantic cache stores an "
    "answer for a query and serves it again when a new query looks similar. Queries must read like real messages "
    "users type to an assistant: short, natural, varied in wording and length. Output only JSON, no prose."
)


def trap_prompt(slot: str) -> str:
    where = ("all 12 in the airline domain" if slot == "class"
             else "3 in each domain: airline, banking, developer tools, customer support for a software product")
    return (
        f"Write 12 pairs of user queries, {where}. In each pair the two queries are worded almost identically, but "
        f"they differ in {SLOTS[slot]}, so a correct answer to the first query would be wrong or incomplete for the "
        "second. Change only that one detail. Vary the topics and phrasing across pairs.\n\n"
        'Return {"pairs": [{"domain": "...", "cached": "...", "new": "..."}]}.'
    )


def control_prompt(domain: str) -> str:
    return (
        f"Write 15 pairs of user queries in the {DOMAIN_TEXT[domain]} domain where both queries need exactly the "
        "same answer. In about half, the queries differ only in how a detail is written: a city name vs an airport "
        "code, an abbreviation or nickname vs the full name, a different format for a number, date or time, a "
        "product name with or without its version prefix, a contraction vs the full words. The rest are ordinary "
        "rewordings. Every pair must truly need the same answer.\n\n"
        'Return {"pairs": [{"kind": "format" | "rewording", "cached": "...", "new": "..."}]}.'
    )


def banking77_prompt(queries: list[str]) -> str:
    kinds = ", ".join(SLOTS)
    listing = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))
    return (
        "Below are real banking customer queries. For each one write two new queries:\n"
        "- trap: a minimal edit of the original that changes one detail so the answer is different. Name the kind "
        f"of change, one of: {kinds}.\n"
        "- same: a rewording that needs exactly the same answer as the original.\n"
        "Keep the user's voice. If a query has no detail that can be changed, still make the smallest edit that "
        "changes the answer and pick the closest kind.\n\n"
        f"{listing}\n\n"
        'Return {"items": [{"n": 1, "trap": "...", "trap_kind": "...", "same": "..."}]} with one item per query, '
        "in order."
    )


# ----------------------------------------------------------------------------------- requests
def banking77_sample() -> list[str]:
    rows = list(csv.DictReader(BANKING77.open()))
    return [r["text"].strip() for r in random.Random(SEED + 4).sample(rows, N_BANKING77)]


def plan(bank: list[str] | None) -> list[dict]:
    jobs = []
    for g in GENERATORS:
        for s in SLOTS:
            jobs.append({"gen": g, "job": "trap", "slot": s, "prompt": trap_prompt(s)})
        for d in DOMAINS:
            jobs.append({"gen": g, "job": "control", "domain": d, "prompt": control_prompt(d)})
    if bank:
        half = len(bank) // 2
        for g, qs in zip(GENERATORS, (bank[:half], bank[half:])):
            jobs.append({"gen": g, "job": "banking77", "queries": qs, "prompt": banking77_prompt(qs)})
    for j in jobs:
        j["request"] = request(j["gen"], j["prompt"])
        j["key"] = cache_key("generate", j["request"])
    return jobs


def request(gen: str, prompt: str) -> dict:
    if gen == "sol":
        return {"model": GENERATORS[gen], "instructions": SYSTEM, "input": prompt, "reasoning": {"effort": SOL_EFFORT}}
    return {"model": GENERATORS[gen], "max_tokens": SONNET_MAX_TOKENS, "system": SYSTEM,
            "messages": [{"role": "user", "content": prompt}]}


# Estimates lean high: o200k / cl100k x1.35 for input; output = JSON size guess + reasoning/thinking budget guess.
_ENC = tiktoken.get_encoding("o200k_base")
OUT_JSON = {"trap": 900, "control": 1100, "banking77": 1800}
REASONING = {"sol": 4000, "sonnet": 6000}


def estimate(job: dict) -> tuple[int, int, float]:
    t_in = round(len(_ENC.encode(SYSTEM + job["prompt"])) * 1.35)
    t_out = OUT_JSON[job["job"]] + REASONING[job["gen"]]
    return t_in, t_out, usage_cost(GENERATORS[job["gen"]], t_in, t_out)


def call(job: dict, sol_client=None, claude_client=None) -> dict:
    start = time.monotonic()
    try:
        if job["gen"] == "sol":
            r = sol_client.responses.create(**job["request"])
            raw = r.model_dump()
            text = r.output_text or ""
            u = raw.get("usage") or {}
            cached = ((u.get("input_tokens_details") or {}).get("cached_tokens")) or 0
            cost = usage_cost(GENERATORS["sol"], (u.get("input_tokens") or 0) - cached, u.get("output_tokens") or 0,
                              cache_read_tokens=cached)
        else:
            m = claude_client.messages.create(**job["request"])
            raw = m.to_dict()
            text = "".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")
            u = raw.get("usage") or {}
            cost = usage_cost(GENERATORS["sonnet"], u.get("input_tokens") or 0, u.get("output_tokens") or 0)
    except Exception as e:  # noqa: BLE001 - logged, not cached, retried next run
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "latency_ms": (time.monotonic() - start) * 1000}
    return {"ok": True, "text": text, "raw": raw, "cost_usd": cost, "model": raw.get("model"),
            "latency_ms": (time.monotonic() - start) * 1000}


# ----------------------------------------------------------------------------------- parsing
def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in output")
    return json.loads(m.group(0))


def to_pairs(job: dict, text: str) -> tuple[list[dict], list[str]]:
    """Validated pair rows from one response, plus the reasons anything was dropped."""
    model = GENERATORS[job["gen"]]
    out, problems = [], []
    try:
        d = parse_json(text)
    except (ValueError, json.JSONDecodeError) as e:
        return [], [f"parse_error: {e}"]

    def ok_text(*xs) -> bool:
        return all(isinstance(x, str) and 3 <= len(x.strip()) <= 300 for x in xs)

    if job["job"] in ("trap", "control"):
        for p in d.get("pairs", []):
            if not ok_text(p.get("cached"), p.get("new")):
                problems.append("bad_text")
                continue
            if job["job"] == "trap":
                dom = {"developer tools": "dev-tools", "customer support for a software product": "support",
                       "customer support": "support"}.get(p.get("domain", ""), p.get("domain", ""))
                if job["slot"] == "class":
                    dom = "airline"
                if dom not in DOMAINS:
                    problems.append(f"bad_domain:{p.get('domain')}")
                    continue
                out.append({"cached": p["cached"].strip(), "new": p["new"].strip(), "label": "different",
                            "slot_type": job["slot"], "domain": dom, "source": f"llm:{model}"})
            else:
                kind = p.get("kind")
                out.append({"cached": p["cached"].strip(), "new": p["new"].strip(), "label": "same",
                            "slot_type": "format" if kind == "format" else "paraphrase", "domain": job["domain"],
                            "source": f"llm:{model}"})
    else:
        items = d.get("items", [])
        if len(items) != len(job["queries"]):
            problems.append(f"item_count {len(items)} != {len(job['queries'])}")
        for q, it in zip(job["queries"], items):
            if ok_text(it.get("trap")) and it.get("trap_kind") in SLOTS:
                out.append({"cached": q, "new": it["trap"].strip(), "label": "different", "slot_type": it["trap_kind"],
                            "domain": "banking", "source": f"banking77+llm:{model}"})
            else:
                problems.append("bad_trap")
            if ok_text(it.get("same")):
                out.append({"cached": q, "new": it["same"].strip(), "label": "same", "slot_type": "paraphrase",
                            "domain": "banking", "source": f"banking77+llm:{model}"})
            else:
                problems.append("bad_same")
    return out, problems


# ----------------------------------------------------------------------------------- main
def main() -> None:
    from .build_pairs import authored_rows, dedupe
    from .estimate import check_budget

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--max-usd", type=float, default=3.0)
    ap.add_argument("--refreeze", action="store_true")
    args = ap.parse_args()

    bank = banking77_sample() if BANKING77.exists() else None
    jobs = plan(bank)
    cache = DiskCache(CACHE, "semcache")
    todo = [j for j in jobs if cache.get(j["key"]) is None]
    print(f"{len(jobs)} requests ({len(todo)} uncached); BANKING77 seeds: {len(bank) if bank else 'MISSING - run scripts/download_banking77.sh'}")
    by = {}
    for j in todo:
        k = f"{j['gen']} {j['job']}"
        t_in, t_out, usd = estimate(j)
        n, a, b, c = by.get(k, (0, 0, 0, 0.0))
        by[k] = (n + 1, a + t_in, b + t_out, c + usd)
    print(f"{'step':<22}{'calls':>6}{'est. input':>12}{'est. output':>13}{'est. USD':>11}")
    for k, (n, a, b, c) in sorted(by.items()):
        print(f"{k:<22}{n:>6}{a:>12,}{b:>13,}{'$' + format(c, '.4f'):>11}")
    total = sum(v[3] for v in by.values())
    print(f"{'TOTAL':<22}{len(todo):>6}{'':>12}{'':>13}{'$' + format(total, '.4f'):>11}")
    print("(output includes a reasoning/thinking guess: sol 4,000, sonnet 6,000 tokens per call)")
    check_budget(total, args.max_usd)
    print(f"within --max-usd ${args.max_usd:.2f}")
    if not args.live:
        print("Dry run: no API calls made. Pass --live to run.")
        return
    if bank is None:
        raise SystemExit("BANKING77 missing - run scripts/download_banking77.sh")
    if GENERATED.exists() and not args.refreeze:
        raise SystemExit(f"{GENERATED} is frozen; pass --refreeze to regenerate it")

    from anthropic import Anthropic
    from openai import OpenAI
    sol, claude = OpenAI(max_retries=4, timeout=600.0), Anthropic(max_retries=4)

    def one(j):
        res, cached = cache.get_or_call(j["key"], lambda: call(j, sol, claude))
        return j, res, cached

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, problems, spent = [], {}, 0.0
    with ThreadPoolExecutor(max_workers=6) as ex, (RESULTS / "generate_raw.jsonl").open("a") as log:
        for j, res, cached in ex.map(one, jobs):
            tag = f"{j['gen']}|{j['job']}|{j.get('slot') or j.get('domain') or 'banking77'}"
            log.write(json.dumps({"tag": tag, "cached": cached, **{k: v for k, v in res.items() if k != "raw"},
                                  "raw": res.get("raw")}, default=str) + "\n")
            if not res.get("ok"):
                problems[tag] = [res.get("error")]
                continue
            spent += 0 if cached else (res.get("cost_usd") or 0)
            got, bad = to_pairs(j, res["text"])
            rows += got
            if bad:
                problems[tag] = bad
    authored = authored_rows()
    kept, dropped = dedupe(authored + rows)
    new = kept[len(authored):] if kept[: len(authored)] == authored else [r for r in kept if r not in authored]
    failed = [t for t, v in problems.items() if any(str(x).startswith(("api_error", "BadRequest", "APIError")) or
                                                    "Error" in str(x)[:40] for x in v)]
    out = GENERATED if not failed else RESULTS / "generated_partial.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new))
    summary = {"requests": len(jobs), "candidates": len(rows), "kept": len(new), "dropped": dict(dropped),
               "problems": problems, "actual_usd_this_run": round(spent, 4)}
    (RESULTS / "generate_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if failed:
        raise SystemExit(f"{len(failed)} requests failed ({', '.join(failed)}); NOT frozen. Partial pairs in {out}. "
                         "Successful responses are cached, so a re-run only retries the failed ones.")
    print(f"wrote {GENERATED}; now run: python -m semcache.build_pairs --refreeze")


if __name__ == "__main__":
    main()
