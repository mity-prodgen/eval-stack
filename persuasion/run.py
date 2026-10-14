#!/usr/bin/env python3
"""Run the judges over items x conditions x arms. Dry run by default.

Plan per item and condition (C0-C4):
  Jev, Luna   P0, P1, P0G  -> 3 requests (P0G = judge + guard questions in one request)
  Haiku, Sonnet  P0, P1, guard -> 3 requests (P0G = the P0 judgment + the separate guard request;
                 see CLAUDE_GUARD_SEPARATE_REQUEST in common.py for why)

Before any API call the estimated cost of every *uncached* request in the plan is compared with
--max-usd (default 5); over the cap, the run aborts with zero calls. Every response is cached
(cache/), so reruns and re-analysis are free. Claude judges go through the Message Batches API
(50% off) unless --claude-mode sync.

    python persuasion/run.py                         # dry run, full plan
    python persuasion/run.py --pilot --live          # 5 items x 5 conditions x every arm, Claude sync
    python persuasion/run.py --live                  # full run
    python persuasion/run.py --live --judge jev --arm P0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from common import (ARMS, CACHE, CONDITIONS, JUDGE_MODELS, JUDGES, RESULTS, SEED, TASK_TYPES, apply_condition)
from estimate import Est, check_budget, estimate_call, print_table
from judges import (BUILDERS, Call, run_claude_batch, run_claude_sync, run_jev, run_luna)
from jev_harness.cache import DiskCache
from rewrite import FROZEN

N_PILOT = 5


def load_sample() -> list[dict]:
    path = RESULTS / "sample.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run persuasion/build_sample.py first")
    return [json.loads(line) for line in path.open()]


def load_rewrites(path: Path = FROZEN) -> dict[str, str]:
    if not path.exists():
        return {}
    return {r["response_id"]: r["rewrite"] for r in map(json.loads, path.open())}


def pilot_items(items: list[dict], n: int = N_PILOT) -> list[dict]:
    """One hallucinated item per task type + clean items, seeded: covers every task type and both classes."""
    rng = random.Random(SEED + 5)
    out = []
    for t in TASK_TYPES:
        out.append(rng.choice(sorted((i for i in items if i["task_type"] == t and i["hallucinated"]),
                                     key=lambda i: i["response_id"])))
    clean = sorted((i for i in items if not i["hallucinated"]), key=lambda i: i["response_id"])
    out.extend(rng.sample(clean, n - len(out)))
    return out


def plan(items, judges, arms, conditions, rewrites, c4_proxy: bool) -> tuple[list[Call], list[str]]:
    """Build every request. If C4 rewrites aren't frozen yet and c4_proxy is set (dry run only),
    the original answer stands in for the rewrite so the cost estimate covers C4."""
    calls, notes = [], []
    missing_c4 = 0
    for it in items:
        for cond in conditions:
            if cond == "C4" and it["response_id"] not in rewrites:
                if not c4_proxy:
                    missing_c4 += 1
                    continue
                text = it["response"]
            else:
                text = apply_condition(it["response"], cond, rewrites, it["response_id"])
            for j in judges:
                model = JUDGE_MODELS[j]
                kinds = []
                for arm in arms:
                    if j in ("haiku", "sonnet") and arm == "P0G":
                        kinds.append(("guard", "guard"))
                        if "P0" not in arms:
                            kinds.append(("P0", "judge"))
                    else:
                        kinds.append((arm, "judge"))
                for arm, kind in dict.fromkeys(kinds):
                    calls.append(BUILDERS[j](Call(j, model, arm, kind, it["response_id"], cond,
                                                  it["source_passage"], text)))
    if missing_c4:
        notes.append(f"{missing_c4} item-condition pairs skipped: no frozen C4 rewrite (dropped by span check or not generated)")
    if c4_proxy and "C4" in conditions and not rewrites:
        notes.append("C4 rewrites not generated yet: original answers used as a length proxy for C4 in this estimate")
    return calls, notes


def estimate_plan(calls: list[Call], cache: DiskCache, claude_batch: bool, only_uncached: bool) -> tuple[float, list]:
    by: dict[str, Est] = defaultdict(Est)
    for c in calls:
        if only_uncached and cache.get(c.key) is not None:
            continue
        by[f"{c.judge} ({c.model})"].add(estimate_call(c, batch=claude_batch))
    rows = sorted(by.items())
    return sum(e.usd for _, e in rows), rows


def per_1000(calls: list[Call], claude_batch: bool, n_arms: int) -> dict[str, float]:
    """Estimated cost per 1,000 judgments per judge. A judgment = one item x condition x arm verdict,
    so every judge yields the same number; for Claude the P0G verdict costs one guard request."""
    usd: dict[str, float] = defaultdict(float)
    pairs: dict[str, set] = defaultdict(set)
    for c in calls:
        usd[c.judge] += estimate_call(c, batch=claude_batch).usd
        pairs[c.judge].add((c.item_id, c.condition))
    return {j: 1000 * usd[j] / (len(pairs[j]) * n_arms) for j in usd}


def execute(calls: list[Call], cache: DiskCache, claude_mode: str) -> dict[str, dict]:
    results: dict[str, dict] = {}
    by_judge: dict[str, list[Call]] = defaultdict(list)
    for c in calls:
        by_judge[c.judge].append(c)
    for j, cs in by_judge.items():
        uniq = list({c.key: c for c in cs}.values())
        t0 = time.time()
        print(f"[{j}] {len(uniq)} requests ...", flush=True)
        if j == "jev":
            results.update(run_jev(uniq, cache))
        elif j == "luna":
            results.update(run_luna(uniq, cache))
        elif claude_mode == "sync":
            results.update(run_claude_sync(uniq, cache))
        else:
            results.update(run_claude_batch(uniq, cache, RESULTS / "batches"))
        print(f"[{j}] done in {time.time() - t0:.0f}s", flush=True)
    return results


ROW_FIELDS = ["judge", "model", "arm", "kind", "item_id", "task_type", "hallucinated", "condition", "status",
              "p_pass", "p_guard", "error", "input_tokens", "output_tokens", "cache_write_tokens",
              "cache_read_tokens", "cost_usd", "batch", "cached", "latency_ms", "api_model"]


def rows_for(calls: list[Call], results: dict[str, dict], items: dict[str, dict]) -> list[dict]:
    rows = []
    for c in calls:
        r = results.get(c.key, {})
        u = r.get("usage") or {}
        it = items[c.item_id]
        rows.append({
            "judge": c.judge, "model": c.model, "arm": c.arm, "kind": c.kind, "item_id": c.item_id,
            "task_type": it["task_type"], "hallucinated": it["hallucinated"], "condition": c.condition,
            "status": r.get("status", "missing"), "p_pass": r.get("p_pass"), "p_guard": r.get("p_guard"),
            "error": r.get("error"), "input_tokens": u.get("input"), "output_tokens": u.get("output"),
            "cache_write_tokens": u.get("cache_write"), "cache_read_tokens": u.get("cache_read"),
            "cost_usd": r.get("cost_usd"), "batch": r.get("batch"), "cached": r.get("cached"),
            "latency_ms": r.get("latency_ms"), "api_model": r.get("model"),
        })
    return rows


def write_rows(rows: list[dict], stem: str) -> None:
    (RESULTS / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with (RESULTS / f"{stem}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="make API calls (default: dry run)")
    ap.add_argument("--max-usd", type=float, default=5.0, help="abort before any call if the estimate is higher")
    ap.add_argument("--pilot", action="store_true", help=f"{N_PILOT} items x all conditions x all arms")
    ap.add_argument("--judge", nargs="+", choices=JUDGES, default=list(JUDGES))
    ap.add_argument("--arm", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--condition", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    ap.add_argument("--limit", type=int, default=None, help="first N sample items")
    ap.add_argument("--claude-mode", choices=["batch", "sync"], default=None,
                    help="default: sync for --pilot, batch otherwise")
    args = ap.parse_args()

    claude_mode = args.claude_mode or ("sync" if args.pilot else "batch")
    sample = load_sample()
    items = pilot_items(sample) if args.pilot else sample[: args.limit] if args.limit else sample
    rewrites = load_rewrites()
    calls, notes = plan(items, args.judge, args.arm, args.condition, rewrites, c4_proxy=not args.live)
    cache = DiskCache(CACHE, "persuasion")

    total_all, rows_all = estimate_plan(calls, cache, claude_mode == "batch", only_uncached=False)
    print_table(rows_all, f"Estimate, whole plan ({len(items)} items, Claude {claude_mode})")
    total_new, rows_new = estimate_plan(calls, cache, claude_mode == "batch", only_uncached=True)
    print()
    print_table(rows_new, "Estimate, uncached requests only (what --live would spend)")
    print("\nEstimated cost per 1,000 judgments:")
    for j, v in per_1000(calls, claude_mode == "batch", len(args.arm)).items():
        print(f"  {j:<8} ${v:.4f}")
    for n in notes:
        print(f"NOTE: {n}")
    check_budget(total_new, args.max_usd)
    print(f"\nwithin --max-usd ${args.max_usd:.2f}")

    if not args.live:
        print("Dry run: no API calls made. Pass --live to run.")
        return

    results = execute(calls, cache, claude_mode)
    rows = rows_for(calls, results, {i["response_id"]: i for i in sample})
    stem = "judgments_pilot" if args.pilot else "judgments"
    write_rows(rows, stem)

    actual = defaultdict(float)
    for r in rows:
        if not r["cached"] and r["cost_usd"]:
            actual[r["judge"]] += r["cost_usd"]
    est_by = {k.split(" ")[0]: e.usd for k, e in rows_new}
    status = defaultdict(lambda: defaultdict(int))
    for r in rows:
        status[r["judge"]][r["status"]] += 1
    summary = {"estimated_usd_uncached": est_by, "actual_usd_this_run": dict(actual),
               "status_counts": {k: dict(v) for k, v in status.items()}, "claude_mode": claude_mode,
               "n_items": len(items), "notes": notes}
    (RESULTS / f"{stem}_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
