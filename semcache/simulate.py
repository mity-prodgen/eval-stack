#!/usr/bin/env python3
"""Full-cache simulation: every lookup searches ALL stored queries, as a real semantic cache does. Offline.

The main experiment compares each new query with one stored query (its labelled partner). Here the store holds
every distinct `cached` query in pairs.jsonl (one entry per text), and each pair's `new` query is looked up
against the whole store: the nearest stored query is served if its cosine clears the threshold. Embeddings come
from the cache the full run filled; a missing vector raises instead of calling an API.

Each lookup ends in one of:
  miss        nearest cosine below the threshold
  identical   the new query's exact text is stored (cosine 1): serving it is correct by definition
  partner     the nearest entry is the pair's own cached query: the pair's label applies
  other       the nearest entry is some other stored query: NOT labelled

So for traps (label "different") the wrong-hit rate is a range: low = partner hits only, high = partner + other
hits. For controls, a correct hit = partner or identical; "other" hits are unknown. review_sim.csv samples 30
"other" hits (seeded, stratified by lookup label) for a human label, like pairs/review.csv.

A2 in the simulation: serve the nearest entry only if the slot guard passes for (that entry, new query).

    python -m semcache.simulate
"""

from __future__ import annotations

import csv
import json
import random

import numpy as np

from .common import DEFAULT_COSINE, EMBEDDERS, PAIRS_DIR, RESULTS, SEED
from .embed import embed_texts, semcache_cache
from .run import load_pairs
from .slots import guard

THRESHOLDS_SIM = (0.80, 0.85, 0.90, 0.95)
N_REVIEW = 30
REVIEW_SIM = PAIRS_DIR / "review_sim.csv"


def _no_api(e, texts):
    raise RuntimeError(f"{e.name}: {len(texts)} texts not in the cache - run semcache.run first (no API calls here)")


def nearest(pairs: list[dict], vec: dict[str, np.ndarray]) -> tuple[list[str], list[tuple[int, float]]]:
    store = list(dict.fromkeys(p["cached"] for p in pairs))
    S = np.stack([vec[t] for t in store])
    Q = np.stack([vec[p["new"]] for p in pairs])
    sims = Q @ S.T
    j = sims.argmax(axis=1)
    return store, [(int(j[i]), float(sims[i, j[i]])) for i in range(len(pairs))]


def outcome(p: dict, store: list[str], j: int, cos: float, t: float, use_guard: bool = False) -> str:
    if p["new"] in store:  # exact text stored: a cache serves it (cosine 1 against itself)
        return "identical"
    if cos < t or (use_guard and not guard(store[j], p["new"])[0]):
        return "miss"
    return "partner" if store[j] == p["cached"] else "other"


def simulate() -> tuple[dict, list[dict]]:
    pairs = load_pairs()
    cache = semcache_cache()
    texts = list(dict.fromkeys(t for p in pairs for t in (p["cached"], p["new"])))
    out, others = {}, []
    for e in EMBEDDERS:
        vec = embed_texts(e, texts, cache, log=lambda _: None, call=_no_api)
        store, nn = nearest(pairs, vec)
        single = {p["id"]: float(vec[p["cached"]] @ vec[p["new"]]) for p in pairs}
        for arm in ("A0", "A2"):
            for t in THRESHOLDS_SIM:
                res = {lab: {"n": 0, "miss": 0, "identical": 0, "partner": 0, "other": 0, "single_pair_served": 0}
                       for lab in ("different", "same")}
                for p, (j, cos) in zip(pairs, nn):
                    o = outcome(p, store, j, cos, t, use_guard=arm == "A2")
                    r = res[p["label"]]
                    r["n"] += 1
                    r[o] += 1
                    served_single = single[p["id"]] >= t and (arm == "A0" or guard(p["cached"], p["new"])[0])
                    r["single_pair_served"] += served_single
                    if arm == "A0" and t == DEFAULT_COSINE and o == "other":
                        others.append({"model": e.name, "lookup_id": p["id"], "lookup_label": p["label"],
                                       "stored_query": store[j], "new_query": p["new"], "cosine": round(cos, 4),
                                       "partner_query": p["cached"], "partner_cosine": round(single[p["id"]], 4)})
                tr, ct = res["different"], res["same"]
                out[f"{e.name}|{arm}|{t:.2f}"] = {
                    **res, "store_size": len(store),
                    "trap_wrong_low": tr["partner"] / tr["n"], "trap_wrong_high": (tr["partner"] + tr["other"]) / tr["n"],
                    "trap_single_pair": tr["single_pair_served"] / tr["n"],
                    "control_correct": (ct["partner"] + ct["identical"]) / ct["n"],
                    "control_any_hit": (ct["partner"] + ct["identical"] + ct["other"]) / ct["n"],
                    "control_single_pair": ct["single_pair_served"] / ct["n"]}
    return out, others


def write_review(others: list[dict]) -> int:
    """30 distinct (stored, new) texts from 'other' hits at 0.90, half from trap lookups, half from controls."""
    if REVIEW_SIM.exists():
        return 0  # never redraw a sheet that may hold the reviewer's answers
    uniq = {}
    for o in others:
        uniq.setdefault((o["stored_query"], o["new_query"]), o)
    rng = random.Random(SEED + 5)
    pick = []
    for lab in ("different", "same"):
        pool = sorted((o for o in uniq.values() if o["lookup_label"] == lab), key=lambda o: (o["stored_query"], o["new_query"]))
        pick += rng.sample(pool, min(N_REVIEW // 2, len(pool)))
    with REVIEW_SIM.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "stored_query", "new_query", "same_answer (y/n)", "notes"])
        for i, o in enumerate(sorted(pick, key=lambda o: o["new_query"])):
            w.writerow([i + 1, o["stored_query"], o["new_query"], "", ""])
    return len(pick)


def main() -> None:
    out, others = simulate()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "simulation.json").write_text(json.dumps(out, indent=1))
    (RESULTS / "simulation_other_hits.jsonl").write_text("".join(json.dumps(o) + "\n" for o in others))
    n = write_review(others)
    print(f"{'model':<14}{'arm':<5}{'t':>5}  {'trap single-pair':>16}  {'trap wrong (low-high)':>22}  "
          f"{'ctrl single-pair':>16}  {'ctrl correct':>12}  {'ctrl other':>10}")
    for k, v in out.items():
        m, arm, t = k.split("|")
        if t not in ("0.90", "0.95"):
            continue
        print(f"{m:<14}{arm:<5}{t:>5}  {100 * v['trap_single_pair']:>15.1f}%  "
              f"{100 * v['trap_wrong_low']:>9.1f}% - {100 * v['trap_wrong_high']:>5.1f}%     "
              f"{100 * v['control_single_pair']:>15.1f}%  {100 * v['control_correct']:>11.1f}%  "
              f"{100 * v['same']['other'] / v['same']['n']:>9.1f}%")
    print(f"store: {next(iter(out.values()))['store_size']} distinct stored queries; review sheet: "
          f"{n or 'kept existing'} rows -> {REVIEW_SIM}")


if __name__ == "__main__":
    main()
