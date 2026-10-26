#!/usr/bin/env python3
"""Post 3 runner. Dry run by default.

Plan (full run):
  gen       Claude Haiku 5.5 and Gemma 4 31B answer every item x 2 chunk orders with the default RAG prompt
  label     every answer x both labelers (Sonnet 5.5, gpt-6.1-sol)
  baseline  Sonnet 5.5 faithfulness judge (post 1's question) on every stale/current and control answer
  job1      7 decision models x every item x 2 orders: "Do these chunks give different answers to the question?"
  job2      7 decision models x every stale/current and control answer: "Does this answer use the most recent
            source in the context?" (WikiContradict has no newer source, so no Job 2 there)
  noise     both jobs again on N_NOISE conflict items (repeat tag "noise1")
  latency   (only when asked) N_LATENCY sequential Job 1 calls per decision model after warm-up, no retries

Spending is capped for the whole post: the ledger (results/spend.json) records what every live command spent,
and each command's estimate must fit in min(--max-usd, $5 - spent so far) before any call, and again before each
step. Strands runs locally: start its server first (README), or leave it out with --decider.

    python -m freshness.run                              # dry run, full plan
    python -m freshness.run --pilot                      # STOP 2: 10 items x 2 orders, every model, both jobs
    python -m freshness.run --pilot --sync --live        # pilot with synchronous requests
    python -m freshness.run --live --step gen label
    python -m freshness.run --live --step latency
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict

from harness.cache import DiskCache
from harness.ledger import Ledger

from .build_items import has, load_items
from .calls import Call, build_baseline, build_decision, build_gen, build_label, run_calls
from .common import (CACHE, CACHE_NS, DECIDER, DECIDERS, GENERATORS, HANDCHECK, LABELERS, N_HANDCHECK, N_LATENCY,
                     N_LATENCY_WARMUP, N_NOISE, ORDERS, RESULTS, SEED, TOTAL_USD_CAP)
from .estimate import PLACEHOLDER_ANSWER, check_budget, estimate, print_table

STEPS = ("gen", "label", "baseline", "job1", "job2", "noise", "latency")
DEFAULT_STEPS = STEPS[:-1]
PILOT_LATENCY = 10


# ------------------------------------------------------------------------------------------- item sets
def pilot_items(items: list[dict]) -> list[dict]:
    """6 conflict items (one per domain), 2 controls, 2 wiki. Seeded."""
    rng = random.Random(SEED + 10)
    by_dom = defaultdict(list)
    for i in sorted((i for i in items if i["kind"] == "conflict"), key=lambda i: i["id"]):
        by_dom[i["domain"]].append(i)
    pick = [rng.choice(v) for _, v in sorted(by_dom.items())]
    pick += rng.sample(sorted((i for i in items if i["kind"] == "control"), key=lambda i: i["id"]), 2)
    pick += rng.sample(sorted((i for i in items if i["kind"] == "wiki"), key=lambda i: i["id"]), 2)
    return pick


def noise_items(items: list[dict]) -> list[dict]:
    rng = random.Random(SEED + 11)
    return rng.sample(sorted((i for i in items if i["kind"] == "conflict"), key=lambda i: i["id"]), N_NOISE)


def latency_contexts(items: list[dict], n: int) -> list[tuple[dict, str]]:
    rng = random.Random(SEED + 13)
    pool = [(it, o) for it in sorted(items, key=lambda i: i["id"]) for o in ORDERS]
    return rng.sample(pool, min(n + N_LATENCY_WARMUP, len(pool)))


# ------------------------------------------------------------------------------------------- planning
def plan_gens(items, gens) -> list[Call]:
    return [build_gen(it, o, g) for it in items for o in ORDERS for g in gens]


def plan_labels(answers, by_id, labelers=LABELERS) -> list[Call]:
    return [build_label(by_id[a["item_id"]], a["text"], lb, order=a["order"], gen=a["gen"])
            for a in answers for lb in labelers]


def judged(answers, by_id) -> list[dict]:
    """Baseline and Job 2 run on stale/current and control answers (not WikiContradict)."""
    return [a for a in answers if by_id[a["item_id"]]["kind"] != "wiki"]


def plan_baseline(answers, by_id) -> list[Call]:
    return [build_baseline(by_id[a["item_id"]], a["order"], a["text"], gen=a["gen"]) for a in judged(answers, by_id)]


def plan_job1(items, deciders, repeat="") -> list[Call]:
    return [build_decision(it, o, d, "job1", repeat=repeat) for it in items for o in ORDERS for d in deciders]


def plan_job2(answers, by_id, deciders, repeat="") -> list[Call]:
    return [build_decision(by_id[a["item_id"]], a["order"], d, "job2", a["text"], gen=a["gen"], repeat=repeat)
            for a in judged(answers, by_id) for d in deciders]


def plan_latency(items, deciders, n) -> list[Call]:
    """Sequential sample: warm-up calls first (tagged, dropped in analysis), then n timed calls per model."""
    ctx = latency_contexts(items, n)
    return [build_decision(it, o, d, "job1", repeat=f"latency{'-warmup' if i < N_LATENCY_WARMUP else ''}1")
            for d in deciders for i, (it, o) in enumerate(ctx)]


def answer_rows_for(gen_calls, cache) -> list[dict]:
    """Answers that exist (cached) or a unique placeholder (dry-run estimate only)."""
    rows = []
    for c in gen_calls:
        hit = cache.get(c.key) or {}
        rows.append({"item_id": c.item_id, "order": c.order, "gen": c.gen,
                     "text": hit.get("text") or f"{PLACEHOLDER_ANSWER} ({c.key[:6]})", "real": bool(hit.get("text"))})
    return rows


# ------------------------------------------------------------------------------------------- outputs
def _load(path, key) -> dict:
    if not path.exists():
        return {}
    return {tuple(r[k] for k in key): r for r in map(json.loads, path.open())}


def _write(path, rows: dict) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for _, r in sorted(rows.items(), key=lambda x: str(x[0]))))


SLIM = ("status", "error", "usage", "cost_usd", "model", "cached", "batch", "latency_ms", "attempts",
        "server_latency_ms", "started_at")


def slim(rec: dict) -> dict:
    return {k: rec.get(k) for k in SLIM if k in rec}


def string_match(item: dict, text: str) -> str:
    """Free sanity check for the LLM labels: which of the two values appear in the answer, as whole tokens."""
    def found(v):
        return has(v, text) or v.lower() in text.lower()
    if item["kind"] == "control":
        return "current" if found(item["current"]) else "neither"
    cur, old = found(item["current"]), found(item["stale"])
    return {(True, True): "both", (True, False): "current", (False, True): "stale", (False, False): "neither"}[(cur, old)]


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(labels: dict, keys: list[tuple]) -> dict:
    pairs = [(labels.get(k + ("sonnet",)), labels.get(k + ("sol",))) for k in keys]
    pairs = [(a, b) for a, b in pairs if a and b and a["status"] == "ok" and b["status"] == "ok"]
    la, lb = [a["label"] for a, _ in pairs], [b["label"] for _, b in pairs]
    na, nb = [str(a["conflict_noted"]) for a, _ in pairs], [str(b["conflict_noted"]) for _, b in pairs]
    n = max(len(pairs), 1)
    return {"n": len(pairs), "label_agree": round(sum(x == y for x, y in zip(la, lb)) / n, 3),
            "label_kappa": round(kappa(la, lb), 3),
            "conflict_noted_agree": round(sum(x == y for x, y in zip(na, nb)) / n, 3),
            "disagreements (sonnet->sol)": dict(Counter(f"{x}->{y}" for x, y in zip(la, lb) if x != y).most_common())}


def write_handcheck(answers: list[dict], labels: dict, by_id: dict) -> None:
    rng = random.Random(SEED + 12)
    pool = sorted(answers, key=lambda a: (a["item_id"], a["order"], a["gen"]))
    rows = []
    for i, a in enumerate(sorted(rng.sample(pool, min(N_HANDCHECK, len(pool))), key=lambda a: (a["item_id"], a["gen"]))):
        it = by_id[a["item_id"]]
        k = (a["item_id"], a["order"], a["gen"])
        ls, lo = labels.get(k + ("sonnet",)) or {}, labels.get(k + ("sol",)) or {}
        rows.append({"n": i + 1, "item_id": a["item_id"], "kind": it["kind"], "generator": a["gen"], "order": a["order"],
                     "question": it["question"], "current_value / answer_a": it["current"],
                     "outdated_value / answer_b": it["stale"], "answer": a["text"],
                     "your_label": "", "your_conflict_noted": "", "your_note": "",
                     "sonnet_label": ls.get("label"), "sonnet_conflict_noted": ls.get("conflict_noted"),
                     "sol_label": lo.get("label"), "sol_conflict_noted": lo.get("conflict_noted")})
    with HANDCHECK.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {HANDCHECK} ({len(rows)} answers; labeler columns are at the right so you can hide them)")


def strands_up() -> bool:
    import httpx
    try:
        return httpx.get(DECIDER["strands"].url.replace("/v1/systemone", "/health"), timeout=5).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def percentile(xs: list[float], p: float) -> float | None:
    s = sorted(xs)
    return s[min(int(p * len(s)), len(s) - 1)] if s else None


# ------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--max-usd", type=float, default=TOTAL_USD_CAP, help="per-command cap (the ledger caps the post)")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--sync", action="store_true", help="synchronous requests instead of batch APIs (2x price)")
    ap.add_argument("--step", nargs="+", choices=STEPS, default=None)
    ap.add_argument("--gen", nargs="+", choices=list(GENERATORS), default=list(GENERATORS))
    ap.add_argument("--decider", nargs="+", choices=list(DECIDER), default=[d.name for d in DECIDERS])
    args = ap.parse_args()
    steps = args.step or (list(DEFAULT_STEPS) + (["latency"] if args.pilot else []))
    use_batch = not args.sync

    all_items = load_items()
    by_id = {i["id"]: i for i in all_items}
    items = pilot_items(all_items) if args.pilot else all_items
    noise = [] if args.pilot else noise_items(all_items)
    stem = "pilot" if args.pilot else "full"
    cache = DiskCache(CACHE, CACHE_NS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(RESULTS / "spend.json", TOTAL_USD_CAP)
    batch_dir = RESULTS / "batches"
    n_lat = PILOT_LATENCY if args.pilot else N_LATENCY

    # ---- whole-plan estimate (placeholders where answers don't exist yet)
    gen_calls = plan_gens(items, args.gen)
    answers_est = answer_rows_for(gen_calls, cache)
    noise_ids = {i["id"] for i in noise}
    plan = {"gen": gen_calls, "label": plan_labels(answers_est, by_id), "baseline": plan_baseline(answers_est, by_id),
            "job1": plan_job1(items, args.decider), "job2": plan_job2(answers_est, by_id, args.decider),
            "noise": plan_job1(noise, args.decider, "noise1")
            + plan_job2([a for a in answers_est if a["item_id"] in noise_ids], by_id, args.decider, "noise1"),
            "latency": plan_latency(items, args.decider, n_lat)}
    plan = {k: v for k, v in plan.items() if k in steps}
    every = list({c.key: c for cs in plan.values() for c in cs}.values())
    todo = [c for c in every if cache.get(c.key) is None]

    kinds = dict(Counter(i["kind"] for i in items))
    print(f"{len(items)} items ({kinds}) x {len(ORDERS)} orders; generators {args.gen}; decision models "
          f"{args.decider}; steps {steps}; noise items {len(noise)}; {'batch APIs' if use_batch else 'synchronous'}\n")
    print_table(estimate(every, use_batch), "Estimate, whole plan")
    print()
    total_new = print_table(estimate(todo, use_batch), "Estimate, uncached requests only (what --live would spend)")
    per_k = {}
    for c in [c for c in every if c.step in ("job1", "job2")]:
        e = estimate([c], use_batch)
        per_k.setdefault(c.who, []).append(sum(x.usd for x in e.values()))
    print("\nEstimated cost per 1,000 decisions: " + ", ".join(f"{k} ${1000 * sum(v) / len(v):.4f}"
                                                            for k, v in per_k.items()))
    allowance = ledger.allowance(args.max_usd)
    print(f"\nledger: spent ${ledger.spent():.4f} of ${TOTAL_USD_CAP:.2f}; this command may spend ${allowance:.4f}")
    check_budget(total_new, allowance)
    print("within budget")
    if not args.live:
        print("Dry run: no API calls made. Pass --live to run.")
        return
    if "strands" in args.decider and any(s in steps for s in ("job1", "job2", "noise", "latency")) and not strands_up():
        sys.exit("Strands server not reachable at 127.0.0.1:8099 - start it (README) or pass --decider without strands")

    command = " ".join(sys.argv[1:])
    est_by_step, spent = {}, {}

    def go(step: str, calls: list[Call], sequential: bool = False) -> dict[str, dict]:
        new = [c for c in {c.key: c for c in calls}.values() if cache.get(c.key) is None]
        est = sum(e.usd for e in estimate(new, use_batch and not sequential).values())
        check_budget(est, ledger.allowance(args.max_usd - sum(spent.values())))
        print(f"[{step}] {len(calls)} calls, {len(new)} uncached, est ${est:.4f}", flush=True)
        res = run_calls(calls, cache, by_id, batch_dir, use_batch=use_batch, sequential=sequential)
        usd = sum((r.get("cost_usd") or 0) for r in res.values() if not r.get("cached"))
        est_by_step[step], spent[step] = est, usd
        ledger.record(command, step, usd, est)
        print(f"[{step}] statuses {dict(Counter(r.get('status') for r in res.values()))}; spent ${usd:.4f}", flush=True)
        return res

    f_ans, f_lab = RESULTS / f"answers_{stem}.jsonl", RESULTS / f"labels_{stem}.jsonl"
    f_base, f_dec = RESULTS / f"baseline_{stem}.jsonl", RESULTS / f"decisions_{stem}.jsonl"
    ans_rows = _load(f_ans, ("item_id", "order", "gen"))
    if "gen" in steps:
        res = go("gen", gen_calls)
        for c in gen_calls:
            r = res[c.key]
            ans_rows[(c.item_id, c.order, c.gen)] = {"item_id": c.item_id, "kind": by_id[c.item_id]["kind"],
                                                     "order": c.order, "gen": c.gen, "text": r.get("text"),
                                                     "stop": r.get("stop"), "key": c.key, **slim(r)}
        _write(f_ans, ans_rows)
    item_ids = {i["id"] for i in items}
    answers = [a for a in ans_rows.values() if a.get("text") and a["item_id"] in item_ids and a["gen"] in args.gen]

    labels = _load(f_lab, ("item_id", "order", "gen", "labeler"))
    if "label" in steps:
        calls = plan_labels(answers, by_id)
        res = go("label", calls)
        for c in calls:
            r, it = res[c.key], by_id[c.item_id]
            a = ans_rows[(c.item_id, c.order, c.gen)]
            labels[(c.item_id, c.order, c.gen, c.who)] = {
                "item_id": c.item_id, "kind": it["kind"], "order": c.order, "gen": c.gen, "labeler": c.who,
                "label": r.get("label"), "conflict_noted": r.get("conflict_noted"),
                "string_match": string_match(it, a["text"]) if it["kind"] != "wiki" else None, **slim(r)}
        _write(f_lab, labels)

    if "baseline" in steps:
        calls = plan_baseline(answers, by_id)
        res = go("baseline", calls)
        rows = _load(f_base, ("item_id", "order", "gen"))
        for c in calls:
            r = res[c.key]
            rows[(c.item_id, c.order, c.gen)] = {"item_id": c.item_id, "kind": by_id[c.item_id]["kind"],
                                                 "order": c.order, "gen": c.gen, "p_pass": r.get("p_pass"), **slim(r)}
        _write(f_base, rows)

    dec_rows = _load(f_dec, ("job", "decider", "item_id", "order", "gen", "repeat"))

    def keep(calls, res):
        for c in calls:
            r = res[c.key]
            dec_rows[(c.step, c.who, c.item_id, c.order, c.gen, c.repeat)] = {
                "job": c.step, "decider": c.who, "model": c.model, "item_id": c.item_id,
                "kind": by_id[c.item_id]["kind"], "date_format": by_id[c.item_id]["date_format"], "order": c.order,
                "gen": c.gen, "repeat": c.repeat, "p_yes": r.get("p_yes"), **slim(r)}
        _write(f_dec, dec_rows)

    if "job1" in steps:
        calls = plan_job1(items, args.decider)
        keep(calls, go("job1", calls))
    if "job2" in steps:
        calls = plan_job2(answers, by_id, args.decider)
        keep(calls, go("job2", calls))
    if "noise" in steps and noise:
        calls = plan_job1(noise, args.decider, "noise1") + plan_job2(
            [a for a in answers if a["item_id"] in noise_ids], by_id, args.decider, "noise1")
        keep(calls, go("noise", calls))
    lat_summary = {}
    if "latency" in steps:
        calls = plan_latency(items, args.decider, n_lat)
        res = go("latency", calls, sequential=True)
        keep(calls, res)
        for d in args.decider:
            ms = [res[c.key]["latency_ms"] for c in calls if c.who == d and "warmup" not in c.repeat
                  and res[c.key].get("status") == "ok" and not res[c.key].get("cached")]
            lat_summary[d] = {"n": len(ms), "median_ms": percentile(ms, 0.5), "p95_ms": percentile(ms, 0.95),
                              "where": "local M5" if DECIDER[d].local else "hosted, measured from this machine",
                              "vendor_claim": DECIDER[d].vendor_latency}

    summary = {"stem": stem, "command": command,
               "estimated_usd": {k: round(v, 4) for k, v in est_by_step.items()},
               "actual_usd": {k: round(v, 4) for k, v in spent.items()},
               "ledger_spent_usd": round(ledger.spent(), 4), "latency": lat_summary}
    if args.pilot and labels:
        keys = sorted({k[:3] for k in labels if k[0] in item_ids})
        summary["labeler_agreement"] = {"all": agreement(labels, keys)}
        for kind in ("conflict", "control", "wiki"):
            summary["labeler_agreement"][kind] = agreement(labels, [k for k in keys if by_id[k[0]]["kind"] == kind])
        write_handcheck(answers, labels, by_id)
    (RESULTS / f"run_summary_{stem}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
