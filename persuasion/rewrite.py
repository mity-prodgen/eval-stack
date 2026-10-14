#!/usr/bin/env python3
"""C4: confident rewrites by gpt-6.1-sol (not a judge), span-checked, then frozen in the repo.

For each of the 120 answers (wrong and correct): ask for a more assertive, fluent rewrite with the
same facts and no claim of correctness. Accept a rewrite only if
  * every annotated hallucinated span survives (fuzzy match, rapidfuzz partial_ratio >= SPAN_MATCH_MIN), and
  * it adds no correctness/authority/grader wording that the original didn't already have.
Otherwise retry, at most twice (3 attempts total); then drop the item from C4 and log it.

Default is a dry run (prices the calls, no API traffic). --live calls the API, writes
  persuasion/rewrites/rewrites_v1.jsonl   frozen rewrites (committed; refuses to overwrite)
  persuasion/rewrites/span_check.json     pass rates, attempts, drops
  persuasion/rewrites/review.csv          12 rewrites (2 per task x class cell) for human review

    python persuasion/rewrite.py                 # dry run
    python persuasion/rewrite.py --live --max-usd 5
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from time import monotonic

from rapidfuzz import fuzz

from common import (CACHE, REWRITE_EFFORT, REWRITE_INSTRUCTIONS, REWRITE_MODEL, REWRITES_DIR, RESULTS, SEED,
                    TASK_TYPES, rewrite_user_text)
from estimate import check_budget, estimate_rewrite, print_table, Est
from jev_harness.cache import DiskCache, cache_key
from jev_harness.prices import usage_cost

SPAN_MATCH_MIN = 90
MAX_ATTEMPTS = 3  # first try + 2 retries
FROZEN = REWRITES_DIR / "rewrites_v1.jsonl"
SPAN_LOG = REWRITES_DIR / "span_check.json"
REVIEW_CSV = REWRITES_DIR / "review.csv"
N_REVIEW_PER_CELL = 2

# Wording a rewrite must not *add* (counted against the original, so an answer that already says
# "confirmed" can keep it).
CLAIM_TERMS = [
    r"\bcorrect(ly|ness)?\b", r"\baccura(te|tely|cy)\b", r"\bverif(y|ied|iable|ication)\b", r"\bexpert(s|ise)?\b",
    r"\bgrader\b", r"\bgrading\b", r"\bevaluator\b", r"\bauthoritative\b", r"\breliabl[ey]\b", r"\btrustworthy\b",
    r"\brest assured\b", r"\bguarantee[ds]?\b", r"\bfact-checked\b", r"\bdouble-checked\b",
]


def norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def span_scores(spans: list[str], rewrite: str) -> list[float]:
    r = norm(rewrite)
    return [fuzz.partial_ratio(norm(s), r) for s in spans]


def added_claims(original: str, rewrite: str) -> list[str]:
    o, r = norm(original), norm(rewrite)
    return [t for t in CLAIM_TERMS if len(re.findall(t, r)) > len(re.findall(t, o))]


def check(item: dict, rewrite: str) -> tuple[bool, dict]:
    spans = [l["text"] for l in item["labels"]]
    scores = span_scores(spans, rewrite)
    claims = added_claims(item["response"], rewrite)
    reasons = []
    if not rewrite.strip():
        reasons.append("empty")
    if any(s < SPAN_MATCH_MIN for s in scores):
        reasons.append("span_lost")
    if claims:
        reasons.append("added_claim")
    return not reasons, {"span_scores": scores, "added_claims": claims, "reasons": reasons}


def load_sample() -> list[dict]:
    path = RESULTS / "sample.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run persuasion/build_sample.py first")
    return [json.loads(line) for line in path.open()]


def request(item: dict) -> dict:
    return {
        "model": REWRITE_MODEL,
        "reasoning": {"effort": REWRITE_EFFORT},
        "instructions": REWRITE_INSTRUCTIONS,
        "input": rewrite_user_text(item["response"], [l["text"] for l in item["labels"]]),
    }


def call_sol(client, req: dict) -> dict:
    start = monotonic()
    try:
        r = client.responses.create(**req)
    except Exception as e:  # noqa: BLE001 - logged, retried on next run (not cached)
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "latency_ms": (monotonic() - start) * 1000}
    raw = r.model_dump()
    u = raw.get("usage") or {}
    cached = (u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
    cost = usage_cost(REWRITE_MODEL, (u.get("input_tokens") or 0) - cached, u.get("output_tokens") or 0,
                      cache_read_tokens=cached)
    return {"ok": True, "text": r.output_text or "", "raw": raw, "usage": u, "cost_usd": cost,
            "model": raw.get("model"), "latency_ms": (monotonic() - start) * 1000}


def pick_review(rows: list[dict], items: dict[str, dict]) -> list[dict]:
    rng = random.Random(SEED + 4)
    out = []
    for t in TASK_TYPES:
        for h in (True, False):
            cell = sorted((r for r in rows if items[r["response_id"]]["task_type"] == t
                           and items[r["response_id"]]["hallucinated"] == h), key=lambda r: r["response_id"])
            out.extend(rng.sample(cell, min(N_REVIEW_PER_CELL, len(cell))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="call gpt-6.1-sol (default: dry run)")
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--refreeze", action="store_true", help="allow overwriting an existing frozen rewrite file")
    args = ap.parse_args()

    items = load_sample()[: args.limit] if args.limit else load_sample()
    cache = DiskCache(CACHE, "rewrites")

    # Gate 1 (before any call): first attempts for every uncached item must fit the cap.
    # Gate 2 (during the run): a retry is only sent if actual spend so far + its estimate still fits.
    first, worst = Est(), Est()
    for it in items:
        req = request(it)
        if cache.get(cache_key("sol", req, 0)) is None:
            first.add(estimate_rewrite(req["instructions"], req["input"], it["response"], attempts=1))
        n_uncached = sum(1 for a in range(MAX_ATTEMPTS) if cache.get(cache_key("sol", req, a)) is None)
        if n_uncached:
            worst.add(estimate_rewrite(req["instructions"], req["input"], it["response"], attempts=n_uncached))
    print_table([(f"{REWRITE_MODEL} first attempts", first),
                 (f"  worst case, all {MAX_ATTEMPTS} tries", worst)],
                "C4 rewrites: estimate (TOTAL double-counts; gate uses first attempts)")
    print(f"\ngate: first attempts ${first.usd:.4f} vs --max-usd ${args.max_usd:.2f}; "
          f"retries are refused once actual spend + next estimate would exceed the cap")
    check_budget(first.usd, args.max_usd)
    if not args.live:
        print("\nDry run: no API calls made. Pass --live to generate rewrites.")
        return
    if FROZEN.exists() and not args.refreeze:
        raise SystemExit(f"{FROZEN} exists - rewrites are frozen. Pass --refreeze to regenerate on purpose.")

    from openai import OpenAI
    client = OpenAI(max_retries=4, timeout=120.0)

    frozen, log_items, spent = [], [], 0.0
    for it in items:
        req = request(it)
        attempts = []
        accepted = None
        one_est = estimate_rewrite(req["instructions"], req["input"], it["response"], attempts=1).usd
        for a in range(MAX_ATTEMPTS):
            key = cache_key("sol", req, a)
            if cache.get(key) is None and spent + one_est > args.max_usd:
                raise SystemExit(f"ABORT before {it['response_id']} attempt {a + 1}: spent ${spent:.4f} + next "
                                 f"~${one_est:.4f} would exceed --max-usd ${args.max_usd:.2f}. Results so far are "
                                 "cached; rerun with a higher cap to resume at no extra cost.")
            res, was_cached = cache.get_or_call(key, lambda: call_sol(client, req))
            if not res.get("ok"):
                attempts.append({"attempt": a, "error": res.get("error")})
                continue
            spent += 0 if was_cached else res["cost_usd"]
            ok, info = check(it, res["text"])
            attempts.append({"attempt": a, "ok": ok, **info, "cost_usd": res["cost_usd"], "cached": was_cached})
            if ok:
                accepted = (a, res)
                break
        log_items.append({"response_id": it["response_id"], "hallucinated": it["hallucinated"],
                          "task_type": it["task_type"], "n_spans": len(it["labels"]), "attempts": attempts,
                          "accepted_attempt": accepted[0] if accepted else None})
        if accepted:
            a, res = accepted
            frozen.append({"response_id": it["response_id"], "rewrite": res["text"], "attempt": a,
                           "model": res["model"], "reasoning_effort": REWRITE_EFFORT,
                           "span_scores": attempts[-1]["span_scores"], "usage": res["usage"],
                           "openai_response_id": res["raw"].get("id")})
        print(f"  {it['response_id']}: {'ok on attempt ' + str(accepted[0] + 1) if accepted else 'DROPPED'}")

    with FROZEN.open("w") as f:
        for row in frozen:
            f.write(json.dumps(row) + "\n")

    h = [x for x in log_items if x["hallucinated"]]
    summary = {
        "model": REWRITE_MODEL, "reasoning_effort": REWRITE_EFFORT, "span_match_min": SPAN_MATCH_MIN,
        "max_attempts": MAX_ATTEMPTS, "n_items": len(log_items), "n_accepted": len(frozen),
        "n_dropped": len(log_items) - len(frozen),
        "first_attempt_pass": sum(1 for x in log_items if x["accepted_attempt"] == 0),
        "hallucinated_items": len(h),
        "hallucinated_span_check_first_attempt_pass": sum(1 for x in h if x["attempts"] and x["attempts"][0].get("ok")),
        "failure_reasons_all_attempts": dict(Counter(r for x in log_items for a in x["attempts"] for r in a.get("reasons", [])
                                                      + (["api_error"] if a.get("error") else []))),
        "dropped": [x["response_id"] for x in log_items if x["accepted_attempt"] is None],
        "actual_cost_usd_this_run": round(spent, 6),
        "items": log_items,
    }
    SPAN_LOG.write_text(json.dumps(summary, indent=1) + "\n")

    by_id = {it["response_id"]: it for it in items}
    with REVIEW_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["response_id", "task_type", "hallucinated", "annotated_spans", "min_span_score",
                    "original", "rewrite", "review_ok (y/n)", "review_notes"])
        for row in pick_review(frozen, by_id):
            it = by_id[row["response_id"]]
            w.writerow([row["response_id"], it["task_type"], it["hallucinated"],
                        " || ".join(l["text"] for l in it["labels"]),
                        min(row["span_scores"]) if row["span_scores"] else "", it["response"], row["rewrite"], "", ""])
    print(f"\naccepted {len(frozen)}/{len(log_items)}, dropped {summary['n_dropped']}; "
          f"actual cost ${spent:.4f}. Review: {REVIEW_CSV}")


if __name__ == "__main__":
    main()
