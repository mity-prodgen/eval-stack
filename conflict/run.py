#!/usr/bin/env python3
"""Post 3 runner: pre-check -> generation -> labels -> judges. Dry run by default.

Plan (full run):
  precheck  Jev 1.13.0 and GPT-6 Luna on every item x chunk order: "Do these chunks give different answers to the
            question?" + which source is newest. Needed before F2 prompts exist.
  gen       4 generators x arms F0, F1, F2jev, F2luna x every item x 2 chunk orders, plus the noise re-run
            (F0, repeat 1) on N_NOISE conflict items x 2 orders. An F2 context whose pre-check does not fire sends
            the F0 prompt, which is a cache hit on the F0 answer.
  label     every answer x both labelers (Sonnet 5.5, gpt-6.1-sol)
  judge     F0 answers (repeat 0) x 4 judges x rubrics J0, J1

Before any call, the estimate for every uncached request is compared with --max-usd (default 5); over the cap,
nothing runs. Before answers exist, later steps are estimated on a placeholder answer, and F2 as if every pre-check
fired (upper bound). Each step is gated again on its refined estimate just before it runs.

    python -m conflict.run                          # dry run, full plan
    python -m conflict.run --pilot                  # STOP 2 pilot: 10 items x 2 orders x 4 arms x 4 generators
    python -m conflict.run --pilot --live --sync    # pilot, synchronous requests (minutes, not batch hours)
    python -m conflict.run --live --step precheck gen --max-usd 5
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict

from harness.cache import DiskCache

from .build_items import has, load_items
from .calls import (Call, build_gen, build_judge, build_label, build_precheck, run_calls)
from .common import (ARMS, CACHE, CACHE_NS, DATA_DIR, GENERATORS, JUDGES, LABELERS, N_HANDCHECK, N_NOISE, ORDERS,
                     PRECHECK_THRESHOLD, PRECHECKERS, PRIMARY_LABELER, RESULTS, RUBRICS, SEED, f2_notice)
from .estimate import PLACEHOLDER_ANSWER, check_budget, estimate, print_table

STEPS = ("precheck", "gen", "label", "judge")
HANDCHECK = DATA_DIR / "handcheck.csv"


# ------------------------------------------------------------------------------------------- item sets
def pilot_items(items: list[dict]) -> list[dict]:
    """6 conflict items (one per domain, one per date format twice), 2 controls, 2 wiki. Seeded."""
    rng = random.Random(SEED + 10)
    conflict = sorted((i for i in items if i["kind"] == "conflict"), key=lambda i: i["id"])
    by_dom = defaultdict(list)
    for i in conflict:
        by_dom[i["domain"]].append(i)
    pick = [rng.choice(v) for _, v in sorted(by_dom.items())]
    pick += rng.sample(sorted((i for i in items if i["kind"] == "control"), key=lambda i: i["id"]), 2)
    pick += rng.sample(sorted((i for i in items if i["kind"] == "wiki"), key=lambda i: i["id"]), 2)
    return pick


def noise_items(items: list[dict]) -> list[dict]:
    rng = random.Random(SEED + 11)
    return rng.sample(sorted((i for i in items if i["kind"] == "conflict"), key=lambda i: i["id"]), N_NOISE)


# ------------------------------------------------------------------------------------------- planning
def plan_prechecks(items, who=PRECHECKERS) -> list[Call]:
    return [build_precheck(it, o, w) for it in items for o in ORDERS for w in who]


def notice_for(rec: dict | None, assume: str | None) -> str | None:
    """assume: the source number to name when no pre-check result exists yet (dry-run upper bound), else None."""
    if rec is None:
        return f2_notice(assume) if assume else None
    if rec.get("status") != "ok" or rec["p_conflict"] < PRECHECK_THRESHOLD:
        return None
    return f2_notice(rec.get("newest"))


def plan_gens(items, gens, arms, pre: dict, noise: list[dict], assume_fire: bool) -> list[Call]:
    calls = []
    for it in items:
        for o in ORDERS:
            for g in gens:
                for a in arms:
                    notice = None
                    if a.startswith("F2"):
                        # distinct placeholder per pre-checker, so the estimate never merges F2jev and F2luna
                        notice = notice_for(pre.get((a[2:], it["id"], o)),
                                            ("1" if a == "F2jev" else "2") if assume_fire else None)
                    calls.append(build_gen(it, o, g, a, notice))
    for it in noise:
        for o in ORDERS:
            for g in gens:
                if "F0" in arms:
                    calls.append(build_gen(it, o, g, "F0", repeat=1))
    return calls


def plan_labels(answers: list[dict], by_id: dict, labelers=LABELERS) -> list[Call]:
    return [build_label(by_id[a["item_id"]], a["text"], lb, order=a["order"], arm=a["arm"], gen=a["gen"],
                        repeat=a["repeat"]) for a in answers for lb in labelers]


def plan_judges(answers: list[dict], by_id: dict, judges=JUDGES, rubrics=RUBRICS) -> list[Call]:
    return [build_judge(by_id[a["item_id"]], a["order"], a["text"], j, r, arm=a["arm"], gen=a["gen"])
            for a in answers if a["arm"] == "F0" and a["repeat"] == 0 for j in judges for r in rubrics]


def placeholder_answers(gen_calls: list[Call]) -> list[dict]:
    return [{"item_id": c.item_id, "order": c.order, "gen": c.gen, "arm": c.arm, "repeat": c.repeat,
             "text": PLACEHOLDER_ANSWER} for c in gen_calls]


# ------------------------------------------------------------------------------------------- outputs
def _load(path, key) -> dict:
    if not path.exists():
        return {}
    return {tuple(r[k] for k in key): r for r in map(json.loads, path.open())}


def _write(path, rows: dict) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for _, r in sorted(rows.items(), key=lambda x: str(x[0]))))


SLIM = ("status", "error", "usage", "cost_usd", "model", "cached", "batch")


def slim(rec: dict) -> dict:
    return {k: rec.get(k) for k in SLIM}


def string_match(item: dict, text: str) -> str:
    """Free sanity check for the LLM labels: which of the two values appear in the answer, as whole tokens."""
    if item["kind"] == "control":
        return "current" if has(item["current"], text) or item["current"].lower() in text.lower() else "neither"
    cur = has(item["current"], text) or item["current"].lower() in text.lower()
    old = has(item["stale"], text) or item["stale"].lower() in text.lower()
    return {(True, True): "both", (True, False): "current", (False, True): "stale", (False, False): "neither"}[(cur, old)]


# ------------------------------------------------------------------------------------------- pilot report
def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(labels: dict, keys: list[tuple]) -> dict:
    """Labeler agreement on answers both labelers labeled ok."""
    pairs = [(labels.get(k + ("sonnet",)), labels.get(k + ("sol",))) for k in keys]
    pairs = [(a, b) for a, b in pairs if a and b and a["status"] == "ok" and b["status"] == "ok"]
    la, lb = [a["label"] for a, _ in pairs], [b["label"] for _, b in pairs]
    na, nb = [str(a["conflict_noted"]) for a, _ in pairs], [str(b["conflict_noted"]) for _, b in pairs]
    disagree = Counter(f"{x}->{y}" for x, y in zip(la, lb) if x != y)
    return {"n": len(pairs), "label_agree": round(sum(x == y for x, y in zip(la, lb)) / max(len(pairs), 1), 3),
            "label_kappa": round(kappa(la, lb), 3),
            "conflict_noted_agree": round(sum(x == y for x, y in zip(na, nb)) / max(len(pairs), 1), 3),
            "conflict_noted_kappa": round(kappa(na, nb), 3), "disagreements (sonnet->sol)": dict(disagree.most_common())}


def write_handcheck(answers: list[dict], labels: dict, by_id: dict) -> None:
    rng = random.Random(SEED + 12)
    pool = sorted((a for a in answers if a.get("text")), key=lambda a: (a["item_id"], a["order"], a["gen"], a["arm"]))
    pick = rng.sample(pool, min(N_HANDCHECK, len(pool)))
    rows = []
    for i, a in enumerate(sorted(pick, key=lambda a: (a["item_id"], a["gen"], a["arm"], a["order"]))):
        it = by_id[a["item_id"]]
        k = (a["item_id"], a["order"], a["gen"], a["arm"], a["repeat"])
        ls, lo = labels.get(k + ("sonnet",)) or {}, labels.get(k + ("sol",)) or {}
        rows.append({"n": i + 1, "item_id": a["item_id"], "kind": it["kind"], "generator": a["gen"], "arm": a["arm"],
                     "order": a["order"], "question": it["question"],
                     "current_value" if it["kind"] != "wiki" else "answer_a": it["current"],
                     "outdated_value" if it["kind"] != "wiki" else "answer_b": it["stale"], "answer": a["text"],
                     "your_label": "", "your_conflict_noted": "", "your_note": "",
                     "sonnet_label": ls.get("label"), "sonnet_conflict_noted": ls.get("conflict_noted"),
                     "sol_label": lo.get("label"), "sol_conflict_noted": lo.get("conflict_noted"),
                     "primary_labeler": PRIMARY_LABELER[a["gen"]]})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with HANDCHECK.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {HANDCHECK} ({len(rows)} answers; label columns are at the right so you can hide them)")


# ------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--sync", action="store_true", help="synchronous requests instead of batch APIs (2x price)")
    ap.add_argument("--step", nargs="+", choices=STEPS, default=list(STEPS))
    ap.add_argument("--gen", nargs="+", choices=list(GENERATORS), default=list(GENERATORS))
    ap.add_argument("--arm", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--judge", nargs="+", choices=list(JUDGES), default=list(JUDGES))
    ap.add_argument("--labeler", nargs="+", choices=list(LABELERS), default=list(LABELERS))
    ap.add_argument("--no-noise", action="store_true")
    args = ap.parse_args()
    use_batch = not args.sync

    all_items = load_items()
    by_id = {i["id"]: i for i in all_items}
    items = pilot_items(all_items) if args.pilot else all_items
    noise = [] if (args.pilot or args.no_noise) else noise_items(all_items)
    stem = "pilot" if args.pilot else "full"
    cache = DiskCache(CACHE, CACHE_NS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    batch_dir = RESULTS / "batches"

    f_pre, f_ans = RESULTS / f"prechecks_{stem}.jsonl", RESULTS / f"answers_{stem}.jsonl"
    f_lab, f_jud = RESULTS / f"labels_{stem}.jsonl", RESULTS / f"judgments_{stem}.jsonl"
    pre_rows = _load(f_pre, ("who", "item_id", "order"))
    ans_rows = _load(f_ans, ("item_id", "order", "gen", "arm", "repeat"))

    # ---- whole-plan estimate (upper bound where outputs don't exist yet)
    pre_calls = plan_prechecks(items)
    pre_known = {k: cache.get(c.key) for k, c in ((("" + c.who, c.item_id, c.order), c) for c in pre_calls)}
    pre_known = {k: v for k, v in pre_known.items() if v is not None}
    gen_calls = plan_gens(items, args.gen, args.arm, pre_known, noise, assume_fire=True)
    answers_est = []
    for c in gen_calls:
        hit = cache.get(c.key)
        answers_est.append({"item_id": c.item_id, "order": c.order, "gen": c.gen, "arm": c.arm, "repeat": c.repeat,
                            # unique per answer, so placeholder labels/judgments are not merged by cache key
                            "text": (hit or {}).get("text") or f"{PLACEHOLDER_ANSWER} ({c.key[:6]})"})
    lab_calls = plan_labels(answers_est, by_id, args.labeler)
    jud_calls = plan_judges(answers_est, by_id, args.judge)
    steps = {"precheck": pre_calls, "gen": gen_calls, "label": lab_calls, "judge": jud_calls}
    steps = {k: v for k, v in steps.items() if k in args.step}

    n_kind = Counter(i["kind"] for i in items)
    print(f"{len(items)} items ({dict(n_kind)}) x {len(ORDERS)} orders; generators {args.gen}; arms {args.arm}; "
          f"noise items {len(noise)}; {'batch APIs' if use_batch else 'synchronous'}\n")
    every = [c for cs in steps.values() for c in cs]
    print_table(estimate(list({c.key: c for c in every}.values()), use_batch), "Estimate, whole plan (upper bound)")
    todo = [c for c in {c.key: c for c in every}.values() if cache.get(c.key) is None]
    print()
    total_new = print_table(estimate(todo, use_batch), "Estimate, uncached requests only (what --live would spend)")
    est_by_step = {s: sum(e.usd for e in estimate([c for c in todo if c.step == s], use_batch).values()) for s in STEPS}
    check_budget(total_new, args.max_usd)
    print(f"\nwithin --max-usd ${args.max_usd:.2f}")
    if not args.live:
        print("Dry run: no API calls made. Pass --live to run.")
        return

    spent = {}

    def go(step: str, calls: list[Call]) -> dict[str, dict]:
        new = [c for c in {c.key: c for c in calls}.values() if cache.get(c.key) is None]
        est = sum(e.usd for e in estimate(new, use_batch).values())
        check_budget(est, args.max_usd - sum(spent.values()))
        print(f"[{step}] {len(calls)} calls, {len(new)} uncached, est ${est:.4f}", flush=True)
        res = run_calls(calls, cache, by_id, batch_dir, use_batch=use_batch)
        spent[step] = sum((r.get("cost_usd") or 0) for k, r in res.items() if not r.get("cached"))
        st = Counter(r.get("status") for r in res.values())
        print(f"[{step}] statuses {dict(st)}; spent ${spent[step]:.4f}", flush=True)
        return res

    pre = {}
    if "precheck" in args.step or any(a.startswith("F2") for a in args.arm):
        res = go("precheck", pre_calls) if "precheck" in args.step else {c.key: cache.get(c.key) or {} for c in pre_calls}
        for c in pre_calls:
            r = res.get(c.key) or {}
            if r.get("status"):
                pre[(c.who, c.item_id, c.order)] = r
                pre_rows[(c.who, c.item_id, c.order)] = {"who": c.who, "item_id": c.item_id, "order": c.order,
                                                         "p_conflict": r.get("p_conflict"), "newest": r.get("newest"),
                                                         **slim(r)}
        _write(f_pre, pre_rows)

    missing_pre = [a for a in args.arm if a.startswith("F2") and
                   any((a[2:], it["id"], o) not in pre for it in items for o in ORDERS)]
    if "gen" in args.step:
        arms = [a for a in args.arm if a not in missing_pre]
        if missing_pre:
            print(f"skipping {missing_pre}: pre-check results missing (run --step precheck first)")
        calls = plan_gens(items, args.gen, arms, pre, noise, assume_fire=False)
        res = go("gen", calls)
        for c in calls:
            r = res[c.key]
            ans_rows[(c.item_id, c.order, c.gen, c.arm, c.repeat)] = {
                "item_id": c.item_id, "kind": by_id[c.item_id]["kind"], "order": c.order, "gen": c.gen, "arm": c.arm,
                "repeat": c.repeat, "notice": None if not c.arm.startswith("F2") else
                notice_for(pre.get((c.arm[2:], c.item_id, c.order)), None), "text": r.get("text"),
                "stop": r.get("stop"), "key": c.key, **slim(r)}
        _write(f_ans, ans_rows)

    wanted = {i["id"] for i in items + noise}
    answers = [a for a in ans_rows.values() if a.get("text") and a["item_id"] in wanted and a["gen"] in args.gen
               and a["arm"] in args.arm]
    labels = {}
    if "label" in args.step:
        calls = plan_labels(answers, by_id, args.labeler)
        res = go("label", calls)
        lab_rows = _load(f_lab, ("item_id", "order", "gen", "arm", "repeat", "labeler"))
        for c in calls:
            r = res[c.key]
            it = by_id[c.item_id]
            a = ans_rows[(c.item_id, c.order, c.gen, c.arm, c.repeat)]
            lab_rows[(c.item_id, c.order, c.gen, c.arm, c.repeat, c.who)] = {
                "item_id": c.item_id, "kind": it["kind"], "order": c.order, "gen": c.gen, "arm": c.arm,
                "repeat": c.repeat, "labeler": c.who, "primary": PRIMARY_LABELER[c.gen] == c.who,
                "label": r.get("label"), "conflict_noted": r.get("conflict_noted"),
                "string_match": string_match(it, a["text"]) if it["kind"] != "wiki" else None, **slim(r)}
        _write(f_lab, lab_rows)
        labels = lab_rows

    if "judge" in args.step:
        calls = plan_judges(answers, by_id, args.judge)
        res = go("judge", calls)
        jud_rows = _load(f_jud, ("item_id", "order", "gen", "judge", "rubric"))
        for c in calls:
            r = res[c.key]
            jud_rows[(c.item_id, c.order, c.gen, c.who, c.rubric)] = {
                "item_id": c.item_id, "kind": by_id[c.item_id]["kind"], "order": c.order, "gen": c.gen,
                "judge": c.who, "rubric": c.rubric, "p_pass": r.get("p_pass"), **slim(r)}
        _write(f_jud, jud_rows)

    summary = {"stem": stem, "estimated_usd_uncached": {k: round(v, 4) for k, v in est_by_step.items()},
               "actual_usd_this_run": {k: round(v, 4) for k, v in spent.items()}}
    if args.pilot and labels:
        keys = sorted({k[:5] for k in labels})
        summary["labeler_agreement"] = {"all": agreement(labels, keys)}
        for kind in ("conflict", "control", "wiki"):
            summary["labeler_agreement"][kind] = agreement(labels, [k for k in keys if by_id[k[0]]["kind"] == kind])
        write_handcheck(answers, labels, by_id)
    (RESULTS / f"run_summary_{stem}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()


__all__ = ["main", "pilot_items", "noise_items", "plan_gens", "plan_judges", "plan_labels", "plan_prechecks",
           "string_match", "kappa", "agreement", "re"]
