#!/usr/bin/env python3
"""Embed every pair with every model, then ask the A3 deciders about pairs that pass A0. Dry run by default.

Plan:
  1. embeddings   every distinct text in pairs.jsonl + the PAWS sample, per embedding model (cached per text)
  2. A3           Jev and Luna, one request per pair that passes A0 (cosine >= 0.90, RedisVL's default) on at
                  least one embedding model. Before embeddings exist that set is unknown, so the dry run and the
                  --max-usd gate assume every pair (an upper bound).

Before any API call, the estimate for every uncached request (embeddings + the A3 upper bound) is compared
with --max-usd (default 3); over the cap, nothing runs. Local models make no API calls and cost $0.

    python -m semcache.run                          # dry run, full plan
    python -m semcache.run --smoke --live           # 20 pairs per model (10 traps, 10 controls) + A3 on them
    python -m semcache.run --live                   # full run
    python -m semcache.run --live --embedder minilm bge-m3 langcache-v1 --no-decide
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from .build_paws import load_paws
from .common import DECIDERS, DEFAULT_COSINE, EMBEDDER, EMBEDDERS, PAIRS, RESULTS, SEED
from .decide import plan as plan_decisions, run as run_decisions
from .embed import cosine_scores, semcache_cache, uncached
from .estimate import Est, check_budget, estimate_decisions, estimate_embed, print_table

N_SMOKE = 20


def load_pairs() -> list[dict]:
    return [json.loads(l) for l in PAIRS.open()]


def smoke_pairs(pairs: list[dict], n: int = N_SMOKE) -> list[dict]:
    """10 traps (at least one per slot type) + 10 controls (aliases and paraphrases), seeded."""
    rng = random.Random(SEED + 3)
    traps = sorted((p for p in pairs if p["label"] == "different"), key=lambda p: p["id"])
    by_slot = defaultdict(list)
    for p in traps:
        by_slot[p["slot_type"]].append(p)
    pick = [rng.choice(v) for _, v in sorted(by_slot.items())]
    rest = [p for p in traps if p not in pick]
    pick += rng.sample(rest, n // 2 - len(pick))
    controls = sorted((p for p in pairs if p["label"] == "same"), key=lambda p: p["id"])
    pick += rng.sample(controls, n - len(pick))
    return pick


def _load_rows(path, key: tuple[str, str]) -> dict:
    if not path.exists():
        return {}
    return {(r[key[0]], r[key[1]]): r for r in map(json.loads, path.open())}


def _write_rows(path, rows: dict) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for _, r in sorted(rows.items())))


def texts_of(pairs: list[dict]) -> list[str]:
    return list(dict.fromkeys(t for p in pairs for t in (p["cached"], p["new"])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="make API calls and run local models (default: dry run)")
    ap.add_argument("--max-usd", type=float, default=3.0)
    ap.add_argument("--smoke", action="store_true", help=f"{N_SMOKE} pairs from pairs.jsonl, no PAWS")
    ap.add_argument("--embedder", nargs="+", choices=list(EMBEDDER), default=[e.name for e in EMBEDDERS])
    ap.add_argument("--decider", nargs="+", choices=DECIDERS, default=list(DECIDERS))
    ap.add_argument("--no-decide", action="store_true")
    ap.add_argument("--no-paws", action="store_true")
    args = ap.parse_args()

    pairs = load_pairs()
    if args.smoke:
        pairs = smoke_pairs(pairs)
    elif not args.no_paws:
        pairs = pairs + load_paws()
    texts = texts_of(pairs)
    cache = semcache_cache()
    embedders = [EMBEDDER[n] for n in args.embedder]
    deciders = [] if args.no_decide else args.decider

    rows_all, rows_new = [], []
    for e in embedders:
        rows_all.append((f"embed {e.name} ({e.model_id})", estimate_embed(e, texts)))
        rows_new.append((f"embed {e.name} ({e.model_id})", estimate_embed(e, uncached(e, texts, cache))))
    calls = plan_decisions(pairs, deciders)
    new_calls = [d for d in calls if cache.get(d.key) is None]
    for k, est in estimate_decisions(calls).items():
        rows_all.append((f"A3 {k} (upper bound: every pair)", est))
    for k, est in estimate_decisions(new_calls).items():
        rows_new.append((f"A3 {k} (upper bound: every pair)", est))

    n_t = sum(p["label"] == "different" for p in pairs)
    print(f"{len(pairs)} pairs ({n_t} different, {len(pairs) - n_t} same), {len(texts)} distinct texts, "
          f"{len(embedders)} embedding models, deciders: {', '.join(deciders) or 'none'}\n")
    print_table(rows_all, "Estimate, whole plan")
    print()
    total_new = print_table(rows_new, "Estimate, uncached requests only (what --live would spend)")
    print("\nEstimated API cost per 1,000 cache lookups (one new-query embedding per lookup; A3 adds one "
          "decision per lookup that passes A0):")
    new_texts = [p["new"] for p in pairs]
    for e in embedders:
        est = estimate_embed(e, new_texts)
        print(f"  embed {e.name:<14} ${1000 * est.usd / len(new_texts):.4f}")
    for k, est in estimate_decisions(calls).items():
        print(f"  A3 {k:<17} ${1000 * est.usd / est.n:.4f} per 1,000 decisions")
    check_budget(total_new, args.max_usd)
    print(f"\nwithin --max-usd ${args.max_usd:.2f}")
    if not args.live:
        print("Dry run: no API calls made, no local model run. Pass --live to run.")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = "smoke" if args.smoke else "full"
    scores: dict[str, dict[str, float]] = {}
    for e in embedders:
        print(f"[{e.name}] embedding {len(texts)} texts ...", flush=True)
        scores[e.name] = cosine_scores(e, pairs, cache)
    # merge with earlier runs, so a model added later (e.g. after a quota reset) does not erase the others
    rows = _load_rows(RESULTS / f"scores_{stem}.jsonl", ("embedder", "pair_id"))
    for p in pairs:
        for name, s in scores.items():
            rows[(name, p["id"])] = {"pair_id": p["id"], "embedder": name, "cosine": s[p["id"]]}
    _write_rows(RESULTS / f"scores_{stem}.jsonl", rows)

    if deciders:
        passing = {p["id"] for p in pairs if any(s[p["id"]] >= DEFAULT_COSINE for s in scores.values())}
        todo = [d for d in calls if d.pair_id in passing]
        print(f"A3: {len(passing)}/{len(pairs)} pairs pass A0 at cosine >= {DEFAULT_COSINE} on some model; "
              f"{len(todo)} decisions ...", flush=True)
        res = run_decisions(todo, cache)
        rows = _load_rows(RESULTS / f"decisions_{stem}.jsonl", ("decider", "pair_id"))
        for d in todo:
            r = res[d.key]
            rows[(d.decider, d.pair_id)] = {"pair_id": d.pair_id, "decider": d.decider, "model": d.model,
                                            "status": r["status"], "p_yes": r["p_yes"], "input_tokens": r["input_tokens"],
                                            "cost_usd": r["cost_usd"], "cached": r["cached"], "error": r["error"],
                                            "api_model": r["model"], "latency_ms": r["latency_ms"]}
        _write_rows(RESULTS / f"decisions_{stem}.jsonl", rows)
        status = defaultdict(lambda: defaultdict(int))
        for d in todo:
            status[d.decider][res[d.key]["status"]] += 1
        print(json.dumps({k: dict(v) for k, v in status.items()}, indent=1))
    print(f"wrote {RESULTS}/scores_{stem}.jsonl" + (f" and decisions_{stem}.jsonl" if deciders else ""))


if __name__ == "__main__":
    main()
