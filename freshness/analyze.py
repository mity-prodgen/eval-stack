#!/usr/bin/env python3
"""Metrics, bootstrap CIs, ROC/AUC, figures and RESULTS.md from freshness/results/*_{stem}.jsonl.

Definitions (n with every rate; CIs are seeded 95% bootstrap over the unit counted, 2,000 resamples):
  verdict         a decision model says "yes" when P(yes) >= 0.5 (THRESHOLD); ROC/AUC use the raw P(yes).
  Job 1           "Do these chunks give different answers to the question?" per item x chunk order.
                  truth: stale/current and WikiContradict = yes (conflict), agreeing controls = no.
                  catch = yes on a conflict; false alarm = yes on a control.
  Job 2           "Does this answer use the most recent source in the context?" per generated answer.
                  The model CATCHES a stale answer by saying no. truth (common.JOB2_TRUTH): current = yes (fresh);
                  stale, blends = no (stale); flags, refuses, other left out.
                  Final label of an answer: the human's label (data/timeline_labels.csv, TIMELINE_RULE) for a
                  timeline answer (either labeler marks timeline); otherwise the labelers' label where both agree;
                  "stale"/"blends" count as the same truth when the labelers split between them.
                  catch = no on a stale answer; false alarm = no on a fresh (current) answer.
                  Headline: stale/current items. Controls (all fresh) and the labelers-only truth are separate rows.
  baseline        Sonnet 5.5 faithfulness judge (post 1's question) passes an answer when P(pass) >= 0.5.
  noise           the same request sent twice (repeat "" and "noise1"): share whose verdict changes, mean |dP|.
  latency         the sequential sample (repeat "latency1", warm-up dropped), measured from this machine.
  cost / 1,000    mean billed cost per decision (from the APIs' usage) x 1,000. Strands: $0 API cost.

    python -m freshness.analyze [--pilot]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from harness.metrics import auc, bootstrap_ci, percentile

from .build_items import load_items
from .common import (DECIDER, DECIDERS, FIGURES, GENERATORS, JOB2_TRUTH, NOT_REACHABLE, ORDERS, PKG, RESULTS,
                     RESULTS_MD, SEED, THRESHOLD, TIMELINE_SHEET, TOTAL_USD_CAP)
from .run import timeline_status

N_BOOT = 2000
DATE_FORMATS = ("header", "body", "version")


# ------------------------------------------------------------------------------------------- loading
def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()] if path.exists() else []


def load(stem: str) -> dict:
    items = {i["id"]: i for i in load_items()}
    answers = {(a["item_id"], a["order"], a["gen"]): a for a in _jsonl(RESULTS / f"answers_{stem}.jsonl")
               if a.get("status") == "ok" and a.get("text")}
    labels = {(r["item_id"], r["order"], r["gen"], r["labeler"]): r for r in _jsonl(RESULTS / f"labels_{stem}.jsonl")}
    baseline = {(r["item_id"], r["order"], r["gen"]): r for r in _jsonl(RESULTS / f"baseline_{stem}.jsonl")}
    decisions = _jsonl(RESULTS / f"decisions_{stem}.jsonl")
    human = {}
    if TIMELINE_SHEET.exists() and stem == "full":
        for h in csv.DictReader(TIMELINE_SHEET.open(newline="")):
            if h.get("your_label", "").strip():
                human[(h["item_id"], h["order"], h["generator"])] = h["your_label"].strip().lower()
    return {"items": items, "answers": answers, "labels": labels, "baseline": baseline, "decisions": decisions,
            "human": human, "stem": stem}


# ------------------------------------------------------------------------------------------- truth
def final_label(D: dict, k: tuple) -> tuple[str | None, str]:
    """(label, source) for answer k: source is human | labelers | timeline-unlabeled | split | failed."""
    ls, lo = D["labels"].get(k + ("sonnet",)) or {}, D["labels"].get(k + ("sol",)) or {}
    if ls.get("status") != "ok" or lo.get("status") != "ok":
        return None, "failed"
    if D["items"][k[0]]["kind"] == "conflict" and timeline_status(ls, lo) in ("timeline", "split"):
        if k in D["human"]:
            return D["human"][k], "human"
        return None, "timeline-unlabeled"
    if ls["label"] == lo["label"]:
        return ls["label"], "labelers"
    if {ls["label"], lo["label"]} <= {"stale", "blends"}:
        return "stale/blends", "labelers"
    return None, "split"


def labeler_truth(D: dict, k: tuple) -> int | None:
    """Sensitivity: the labelers' agreed truth value everywhere, timeline answers included."""
    ls, lo = D["labels"].get(k + ("sonnet",)) or {}, D["labels"].get(k + ("sol",)) or {}
    a, b = JOB2_TRUTH.get(ls.get("label")), JOB2_TRUTH.get(lo.get("label"))
    return a if a is not None and a == b else None


def truth_of(label: str | None) -> int | None:
    return 0 if label == "stale/blends" else JOB2_TRUTH.get(label)


# ------------------------------------------------------------------------------------------- stats helpers
def _ci(vals: list[float], seed: int):
    return bootstrap_ci(vals, n_boot=N_BOOT, seed=seed) if vals else None


def rate(vals: list[float], seed: int) -> dict:
    return {"n": len(vals), "k": int(sum(vals)), "ci": _ci(vals, seed)}


def auc_ci(pos: list[float], neg: list[float], seed: int, n_boot: int = 1000) -> tuple | None:
    """AUC (P(score_pos > score_neg)) with a bootstrap that resamples positives and negatives independently."""
    if not pos or not neg:
        return None
    rng = random.Random(seed)
    stats = sorted(auc([pos[rng.randrange(len(pos))] for _ in pos], [neg[rng.randrange(len(neg))] for _ in neg])
                   for _ in range(n_boot))
    return auc(pos, neg), stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)]


def roc_points(pos: list[float], neg: list[float]) -> list[tuple[float, float, float]]:
    """(threshold, false-alarm rate, catch rate) at every distinct score, plus the ends."""
    ts = sorted(set(pos) | set(neg) | {0.0, 1.0001}, reverse=True)
    return [(t, sum(x >= t for x in neg) / len(neg), sum(x >= t for x in pos) / len(pos)) for t in ts] if pos and neg else []


def seed_of(*parts) -> int:
    return SEED + sum(ord(c) for c in "|".join(map(str, parts))) % 10_000


# ------------------------------------------------------------------------------------------- metrics
def main_decisions(D: dict, job: str) -> list[dict]:
    return [d for d in D["decisions"] if d["job"] == job and d["repeat"] == "" and d.get("status") == "ok"
            and d.get("p_yes") is not None]


def job1(D: dict) -> dict:
    out = {}
    rows = main_decisions(D, "job1")
    for dec in DECIDERS:
        r = [d for d in rows if d["decider"] == dec.name]
        if not r:
            continue
        s = lambda *p: seed_of("job1", dec.name, *p)  # noqa: E731
        yes = lambda d: float(d["p_yes"] >= THRESHOLD)  # noqa: E731
        conf, ctrl, wiki = ([d for d in r if d["kind"] == k] for k in ("conflict", "control", "wiki"))
        m = {"catch": rate([yes(d) for d in conf], s("c")), "false_alarm": rate([yes(d) for d in ctrl], s("f")),
             "catch_wiki": rate([yes(d) for d in wiki], s("w")),
             "auc": auc_ci([d["p_yes"] for d in conf], [d["p_yes"] for d in ctrl], s("auc")),
             "roc": roc_points([d["p_yes"] for d in conf], [d["p_yes"] for d in ctrl])}
        for o in ORDERS:
            m[f"catch_{o}"] = rate([yes(d) for d in conf if d["order"] == o], s("c", o))
            m[f"false_alarm_{o}"] = rate([yes(d) for d in ctrl if d["order"] == o], s("f", o))
        for f in DATE_FORMATS:
            m[f"catch_{f}"] = rate([yes(d) for d in conf if d["date_format"] == f], s("c", f))
            m[f"false_alarm_{f}"] = rate([yes(d) for d in ctrl if d["date_format"] == f], s("f", f))
        out[dec.name] = m
    return out


def job2(D: dict) -> dict:
    """Job 2 on stale/current items (headline, human truth for timeline answers), with the labelers-only truth and
    the controls as separate rows, plus splits."""
    rows = main_decisions(D, "job2")
    out = {}
    for dec in DECIDERS:
        r = [d for d in rows if d["decider"] == dec.name]
        if not r:
            continue
        s = lambda *p: seed_of("job2", dec.name, *p)  # noqa: E731
        no = lambda d: float(d["p_yes"] < THRESHOLD)  # noqa: E731
        head, sens, ctrl = [], [], []
        _label = lambda d: final_label(D, (d["item_id"], d["order"], d["gen"]))[0]  # noqa: E731
        for d in r:
            k = (d["item_id"], d["order"], d["gen"])
            lab, src = final_label(D, k)
            t = truth_of(lab)
            if d["kind"] == "control":
                if t is not None:
                    ctrl.append((d, t))
                continue
            if t is not None:
                head.append((d, t, src))
            lt = labeler_truth(D, k)
            if lt is not None:
                sens.append((d, lt))
        stale = [d for d, t, _ in head if t == 0]
        fresh = [d for d, t, _ in head if t == 1]
        m = {"catch": rate([no(d) for d in stale], s("c")), "false_alarm": rate([no(d) for d in fresh], s("f")),
             "catch_labelers_truth": rate([no(d) for d, t in sens if t == 0], s("cl")),
             "false_alarm_labelers_truth": rate([no(d) for d, t in sens if t == 1], s("fl")),
             "false_alarm_controls": rate([no(d) for d, t in ctrl if t == 1], s("fc")),
             "catch_timeline": rate([no(d) for d, t, src in head if t == 0 and src == "human"], s("ct")),
             "catch_non_timeline": rate([no(d) for d, t, src in head if t == 0 and src != "human"], s("cn")),
             "auc": auc_ci([1 - d["p_yes"] for d in stale], [1 - d["p_yes"] for d in fresh], s("auc")),
             # Within the human-labeled timeline answers only: every answer here mentions both values, so this AUC
             # can't come from "the answer mentions the old value". Stale vs blends asks whether a model tells a
             # future-change answer from an unresolved one.
             "auc_timeline": auc_ci([1 - d["p_yes"] for d, t, src in head if t == 0 and src == "human"],
                                    [1 - d["p_yes"] for d, t, src in head if t == 1 and src == "human"], s("auct")),
             "auc_stale_vs_blends": auc_ci([1 - d["p_yes"] for d, _, src in head if src == "human" and _label(d) == "stale"],
                                           [1 - d["p_yes"] for d, _, src in head if src == "human" and _label(d) == "blends"],
                                           s("aucsb")),
             "roc": roc_points([1 - d["p_yes"] for d in stale], [1 - d["p_yes"] for d in fresh])}
        for o in ORDERS:
            m[f"catch_{o}"] = rate([no(d) for d in stale if d["order"] == o], s("c", o))
            m[f"false_alarm_{o}"] = rate([no(d) for d in fresh if d["order"] == o], s("f", o))
        for f in DATE_FORMATS:
            m[f"catch_{f}"] = rate([no(d) for d in stale if d["date_format"] == f], s("c", f))
        for g in GENERATORS:
            m[f"catch_{g}"] = rate([no(d) for d in stale if d["gen"] == g], s("c", g))
        out[dec.name] = m
    return out


def truth_counts(D: dict) -> dict:
    c = Counter()
    for k, a in D["answers"].items():
        if D["items"][k[0]]["kind"] == "wiki":
            continue
        lab, src = final_label(D, k)
        c[(D["items"][k[0]]["kind"], src, {0: "stale/blends", 1: "fresh", None: "left out"}[truth_of(lab)])] += 1
    return {"|".join(k): v for k, v in sorted(c.items())}


def baseline(D: dict) -> dict:
    by = defaultdict(list)
    for k, b in D["baseline"].items():
        if b.get("status") != "ok" or b.get("p_pass") is None or D["items"][k[0]]["kind"] != "conflict":
            continue
        lab, _ = final_label(D, k)
        if lab is None:
            continue
        group = "stale" if lab in ("stale", "stale/blends") else lab
        by[group].append(float(b["p_pass"] >= THRESHOLD))
    out = {g: rate(v, seed_of("base", g)) for g, v in sorted(by.items())}
    out["stale_or_blends"] = rate(by["stale"] + by["blends"], seed_of("base", "sb"))
    return out


def answer_mix(D: dict) -> dict:
    out = defaultdict(Counter)
    for k in D["answers"]:
        it = D["items"][k[0]]
        lab, src = final_label(D, k)
        name = lab if lab else src
        out[f"{k[2]}|{it['kind']}|{k[1]}"][name] += 1
        if it["kind"] == "conflict":
            out[f"{k[2]}|conflict|{it['date_format']}"][name] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def labeler_agreement(D: dict) -> dict:
    from .run import agreement
    keys = sorted({k[:3] for k in D["labels"] if k[:3] in D["answers"]})
    out = {"all": agreement(D["labels"], keys)}
    for kind in ("conflict", "control", "wiki"):
        out[kind] = agreement(D["labels"], [k for k in keys if D["items"][k[0]]["kind"] == kind])
    hv = defaultdict(lambda: [0, 0])
    for k, h in D["human"].items():
        for who in ("sonnet", "sol"):
            r = D["labels"].get(k + (who,)) or {}
            hv[who][0] += r.get("label") == h
            hv[who][1] += 1
    out["human_vs_labeler_on_timeline"] = {w: f"{a}/{n}" for w, (a, n) in hv.items()}
    return out


def noise(D: dict) -> dict:
    by = defaultdict(dict)
    for d in D["decisions"]:
        if d["repeat"] in ("", "noise1") and d.get("status") == "ok" and d.get("p_yes") is not None:
            by[(d["job"], d["decider"], d["item_id"], d["order"], d["gen"])][d["repeat"]] = d["p_yes"]
    out = {}
    for job in ("job1", "job2"):
        for dec in DECIDERS:
            pairs = [(v[""], v["noise1"]) for k, v in by.items() if k[0] == job and k[1] == dec.name and len(v) == 2]
            if not pairs:
                continue
            flips = [float((a >= THRESHOLD) != (b >= THRESHOLD)) for a, b in pairs]
            out[f"{job}|{dec.name}"] = {**rate(flips, seed_of("noise", job, dec.name)),
                                        "mean_abs_dp": round(statistics.mean(abs(a - b) for a, b in pairs), 4)}
    return out


def latency(D: dict) -> dict:
    out = {}
    for dec in DECIDERS:
        ms = [d["latency_ms"] for d in D["decisions"] if d["decider"] == dec.name and d["repeat"] == "latency1"
              and d.get("status") == "ok" and d.get("latency_ms") is not None and not d.get("cached")]
        tp = [d["latency_ms"] for d in D["decisions"] if d["decider"] == dec.name and d["repeat"] == ""
              and d.get("status") == "ok" and d.get("latency_ms") is not None and (d.get("attempts") or 1) == 1
              and not d.get("cached")]
        out[dec.name] = {"n": len(ms), "median_ms": percentile(ms, 0.5) if ms else None,
                         "p95_ms": percentile(ms, 0.95) if ms else None,
                         "throughput_median_ms": percentile(tp, 0.5) if tp else None,
                         "where": "local M5" if dec.local else "hosted, from this machine",
                         "vendor_claim": dec.vendor_latency}
    return out


def costs(D: dict) -> dict:
    out = {}
    for dec in DECIDERS:
        c = [d["cost_usd"] for d in D["decisions"] if d["decider"] == dec.name and d["repeat"] == ""
             and d.get("status") == "ok" and d.get("cost_usd") is not None]
        out[dec.name] = {"n": len(c), "per_1000_usd": round(1000 * statistics.mean(c), 4) if c else None}
    return out


def harness_checks(D: dict) -> dict:
    st = Counter()
    for d in D["decisions"]:
        st[(d["job"], d["decider"], d.get("status"))] += 1
    retried = Counter(d["decider"] for d in D["decisions"] if (d.get("attempts") or 1) > 1)
    lab_status = Counter((r["labeler"], r.get("status")) for r in D["labels"].values())
    return {"decision_status": {"|".join(map(str, k)): v for k, v in sorted(st.items()) if k[2] != "ok"},
            "decisions_retried": dict(retried), "label_status": {"|".join(map(str, k)): v for k, v in lab_status.items()},
            "answers_ok": len(D["answers"])}


def hypotheses(M: dict) -> dict:
    b = M["baseline"].get("stale_or_blends") or {}
    h1 = {"pass_rate_on_stale_or_blends": b, "supported": bool(b.get("ci")) and b["ci"][0] > 0.5}
    h2 = {}
    for name in M["job1"]:
        c1, c2 = M["job1"][name]["catch"], (M["job2"].get(name) or {}).get("catch")
        if c1.get("ci") and c2 and c2.get("ci"):
            h2[name] = {"job1_catch": c1["ci"][0], "job2_catch": c2["ci"][0],
                        "ci_separate": c1["ci"][1] > c2["ci"][2]}
    h3 = {}
    for job in ("job1", "job2"):
        for name, m in M[job].items():
            cs = {f: m.get(f"catch_{f}", {}).get("ci") for f in DATE_FORMATS}
            if all(cs.values()):
                low = min(cs[f][0] for f in DATE_FORMATS)
                lows = [f for f in DATE_FORMATS if cs[f][0] == low]
                h3[f"{job}|{name}"] = {"lowest": lows[0] if len(lows) == 1 else "tie: " + ", ".join(lows),
                                       **{f: round(cs[f][0], 3) for f in DATE_FORMATS}}
    return {"H1 faithfulness judge passes most stale answers": h1,
            "H2 decision models catch conflicts more often than stale answers": h2,
            "H3 version-only dates are missed most": h3}


def compute(stem: str) -> tuple[dict, dict]:
    D = load(stem)
    M = {"job1": job1(D), "job2": job2(D), "truth_counts": truth_counts(D),
         "human_labels": dict(Counter(D["human"].values()).most_common()), "baseline": baseline(D),
         "answer_mix": answer_mix(D), "labeler_agreement": labeler_agreement(D), "noise": noise(D),
         "latency": latency(D), "costs": costs(D), "harness": harness_checks(D)}
    M["hypotheses"] = hypotheses(M)
    spend = RESULTS / "spend.json"
    M["spend_usd"] = round(sum(e["usd"] for e in json.loads(spend.read_text())["entries"]), 4) if spend.exists() else None
    return D, M


# ------------------------------------------------------------------------------------------- output helpers
def pct(m: dict | None) -> str:
    if not m or not m.get("ci"):
        return "- (n=0)"
    p, lo, hi = m["ci"]
    return f"{100 * p:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}] (n={m['n']})"


def fmt_auc(a) -> str:
    return f"{a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}]" if a else "-"


def ms(x) -> str:
    return f"{x:,.0f}" if x is not None else "-"


# ------------------------------------------------------------------------------------------- figures
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, GREY = "#2a78d6", "#d9622b", "#9a9893"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7, width=0.6)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figures(M: dict, out_dir: Path, suffix: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. catch vs false alarm per model, one panel per job
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE)
    for ax, job, title in ((axes[0], "job1", "Job 1: conflict check (before generation)"),
                           (axes[1], "job2", "Job 2: freshness check (after generation)")):
        _style(ax)
        pts = []
        for name, m in M[job].items():
            c, f = m["catch"].get("ci"), m["false_alarm"].get("ci")
            if not c or not f:
                continue
            color = GREY if DECIDER[name].local else BLUE
            ax.errorbar(100 * f[0], 100 * c[0], xerr=[[100 * (f[0] - f[1])], [100 * (f[2] - f[0])]],
                        yerr=[[100 * (c[0] - c[1])], [100 * (c[2] - c[0])]], fmt="o", color=color, ms=5,
                        elinewidth=0.8, capsize=2)
            pts.append((100 * c[0], 100 * f[0], name))
        # The points cluster in a corner, so the labels go in a column toward the middle with a leader line each.
        pts.sort(key=lambda t: (-t[0], t[1]))
        top = sum(y for y, _, _ in pts) / len(pts) > 50 if pts else True
        for i, (y, x, name) in enumerate(pts if top else pts[::-1]):
            ty = 88 - 7 * i if top else 12 + 7 * i
            ax.annotate(f"{name} ({y:.0f}%, {x:.0f}%)", (x, y), xytext=(22, ty), textcoords="data", fontsize=7,
                        color=INK, va="center", ha="left",
                        arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.6, "relpos": (0, 0.5)})
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103)
        ax.set_xlabel("false-alarm rate (%)", fontsize=8, color=INK2)
        ax.set_ylabel("catch rate (%)", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.suptitle("Catch vs false alarm at P(yes) >= 0.5, 95% bootstrap CIs; labels read (catch, false alarm); grey = local Strands", fontsize=8,
                 color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    p = out_dir / f"catch_vs_false_alarm{suffix}.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)

    # 2. latency (median, p95) and cost per 1,000 decisions per model
    names = [d.name for d in DECIDERS if M["latency"].get(d.name, {}).get("n")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    for i, n in enumerate(names):
        L = M["latency"][n]
        color = GREY if DECIDER[n].local else BLUE
        ax.plot([L["median_ms"], L["p95_ms"]], [i, i], color=color, lw=1.2)
        ax.plot(L["median_ms"], i, "o", color=color, ms=5)
        ax.plot(L["p95_ms"], i, "|", color=color, ms=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n + (" (local M5)" if DECIDER[n].local else "") for n in names], fontsize=7)
    ax.set_xscale("log")
    ticks = [t for t in (50, 100, 200, 300, 500, 1000, 2000) if t <= 1.2 * max(M["latency"][n]["p95_ms"] for n in names)]
    ax.set_xticks(ticks, [str(t) for t in ticks], fontsize=7)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("latency per call, ms (dot = median, bar = p95; log scale)", fontsize=8, color=INK2)
    ax.set_title("Latency from this machine (sequential sample)", fontsize=9, color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    vals = [M["costs"].get(n, {}).get("per_1000_usd") or 0 for n in names]
    ax.barh(range(len(names)), vals, color=[GREY if DECIDER[n].local else BLUE for n in names], height=0.55)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    for i, v in enumerate(vals):
        ax.annotate(f"${v:.3f}", (v, i), textcoords="offset points", xytext=(4, -3), fontsize=7, color=INK)
    ax.set_xlabel("USD per 1,000 decisions (list price; Strands = $0 API cost)", fontsize=8, color=INK2)
    ax.set_title("Cost per 1,000 decisions", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    p = out_dir / f"latency_cost{suffix}.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)

    # 3. faithfulness baseline pass rate by answer label
    groups = [g for g in ("current", "stale", "blends", "flags") if M["baseline"].get(g, {}).get("ci")]
    fig, ax = plt.subplots(figsize=(6.2, 3.4), facecolor=SURFACE)
    _style(ax)
    for i, g in enumerate(groups):
        p_, lo, hi = M["baseline"][g]["ci"]
        ax.bar(i, 100 * p_, color=ORANGE if g in ("stale", "blends") else BLUE, width=0.55)
        ax.errorbar(i, 100 * p_, yerr=[[100 * (p_ - lo)], [100 * (hi - p_)]], color=INK, elinewidth=0.8, capsize=3)
        ax.annotate(f"n={M['baseline'][g]['n']}", (i, 2), ha="center", fontsize=7, color=SURFACE)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("passed by the faithfulness judge (%)", fontsize=8, color=INK2)
    ax.set_title("Sonnet 5.5 faithfulness judge (post 1's question) on stale/current answers", fontsize=9,
                 color=INK, loc="left")
    fig.tight_layout()
    p = out_dir / f"baseline_passes{suffix}.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)

    # 4. ROC per model, per job
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE)
    for ax, job in zip(axes, ("job1", "job2")):
        _style(ax)
        for name, m in M[job].items():
            if not m["roc"]:
                continue
            xs, ys = [100 * r[1] for r in m["roc"]], [100 * r[2] for r in m["roc"]]
            ax.plot(xs, ys, lw=1, label=f"{name} (AUC {m['auc'][0]:.2f})" if m["auc"] else name,
                    ls="--" if DECIDER[name].local else "-")
        ax.plot([0, 100], [0, 100], color=GRID, lw=0.8)
        ax.set_xlabel("false-alarm rate (%)", fontsize=8, color=INK2)
        ax.set_ylabel("catch rate (%)", fontsize=8, color=INK2)
        ax.set_title(f"{'Job 1' if job == 'job1' else 'Job 2'}: ROC over the threshold", fontsize=9, color=INK,
                     loc="left")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    p = out_dir / f"roc{suffix}.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)
    return paths


# ------------------------------------------------------------------------------------------- RESULTS.md
LIMITATIONS = """\
- **Authored data.** The 101 stale/current items and 40 controls are fictional policy documents written by Claude
  Opus 5.5 (not tested here). They are clean by design (one changed value, dated versions next to each other); real
  knowledge bases are messier. WikiContradict (20 items) is the real-data check and is reported separately.
- **Few stale answers, one generator, one shape.** Job 2's catch rate rests on the 95 stale or blended answers the
  two generators happened to write: 94 from Gemma, 1 from Haiku 5.5. 91 of them are \"timeline\" answers (both values
  with their dates), and their truth comes from one human (the post's author) under a rule fixed before the full run.
- **Job 2's wording has a literal reading.** A timeline answer that treats the newer date as a future change does
  mention the most recent source, so a decision model can say \"yes\" to \"Does this answer use the most recent
  source in the context?\" and be literally right while the answer is stale.
- **Labels are model labels outside the timeline set.** Sonnet 5.5 and gpt-6.1-sol agreed with a human on 38 and 35
  of 40 pilot answers (all disagreements were timeline answers); the labelers-only truth is shown as a sensitivity
  row.
- **One threshold, one wording, default settings.** Every model gets the same question text with no criteria and is
  read at P(yes) >= 0.5; vendors' own phrasing or calibrated thresholds could change the ranking. The ROC/AUC columns
  show how much a different threshold would help, but Job 2's all-answers AUC is inflated by a shortcut (answers that
  mention the old value score lower); the within-timeline AUCs are the fairer test, and they rest on 8 current
  answers.
- **Latency is from one machine and one network**, on short inputs (about 1,000 tokens), 99 timed sequential calls per
  model (the 100th was already cached by the pilot and has no fresh timing); vendor figures were measured on other setups and are not comparable. Strands ran on an Apple M5 (MPS) and
  is not comparable with the hosted models or with AWS's RTX 3090 figure.
- **Model versions.** pplx-decider-v1.2 stands in for the v1 the task named (the API routes v1 to v1.2);
  pplx-decider-flash-v1.2 stands in for Microsoft-Decision-1, which was not reachable. Hosted models can change
  behind the same ID; the cache keeps the responses this report used.
- **Cloudflare ran on two days and two plans.** Clef and Clef-flash ran Job 1 and Job 2 on Cloudflare's free plan on
  2026-10-27 (UTC); their noise re-run and latency sample ran on Workers Paid on 2026-10-28. Costs everywhere are
  list prices; the free-plan calls were not actually billed.
"""


def write_md(M: dict, figs: list[Path], path: Path, pilot: bool) -> None:
    L = [f"# Post 3: can the new decision models catch stale RAG answers?{' (PILOT)' if pilot else ''}", "",
         "Regenerated by `freshness/analyze.py`; not hand-edited (findings come from `freshness/findings.md`).", ""]
    findings = PKG / "findings.md"
    if findings.exists() and not pilot:
        L += ["## Top findings", "", findings.read_text().strip(), ""]
    tc = M["truth_counts"]
    L += ["## Setup", "",
          "Seven decision models were asked the same two yes/no questions with their native yes/no type, and read at "
          "P(yes) >= 0.5. Job 1 (before generation): *\"Do these chunks give different answers to the question?\"* "
          "Job 2 (after generation): *\"Does this answer use the most recent source in the context?\"* on answers "
          "written by Claude Haiku 5.5 and Gemma 4 31B with LangChain's default RAG prompt. Every item ran twice: "
          "newer chunk first and older chunk first. Design, data, model IDs and prices: README.md.", "",
          f"Not reachable: {', '.join(f'**{k}** ({v})' for k, v in NOT_REACHABLE.items())}", "",
          f"Spend for the whole post: ${M['spend_usd']} of the ${TOTAL_USD_CAP:.0f} cap (ledger).", "",
          "Job 2 truth counts (kind | truth source | truth): " + ", ".join(f"{k} {v}" for k, v in tc.items()), ""]

    L += ["## Job 1: conflict check", "",
          "| Model | Catch (stale/current) | False alarm (controls) | AUC | Catch, newer first | Catch, older first | "
          "Catch, WikiContradict |", "|---|---|---|---|---|---|---|"]
    for n, m in M["job1"].items():
        L.append(f"| {n} | {pct(m['catch'])} | {pct(m['false_alarm'])} | {fmt_auc(m['auc'])} | "
                 f"{pct(m['catch_new_first'])} | {pct(m['catch_old_first'])} | {pct(m['catch_wiki'])} |")
    L += ["", "By date format (catch on stale/current items):", "",
          "| Model | Header | Body | Version only |", "|---|---|---|---|"]
    for n, m in M["job1"].items():
        L.append(f"| {n} | " + " | ".join(pct(m[f"catch_{f}"]) for f in DATE_FORMATS) + " |")

    L += ["", "## Job 2: freshness check", "",
          "A model catches a stale answer by saying **no**. Headline truth: the human's label for timeline answers, "
          "the labelers' agreed label elsewhere.", "",
          "| Model | Catch (stale/blends) | False alarm (current) | AUC | Catch, timeline answers | "
          "Catch, other answers | False alarm, controls |", "|---|---|---|---|---|---|---|"]
    for n, m in M["job2"].items():
        L.append(f"| {n} | {pct(m['catch'])} | {pct(m['false_alarm'])} | {fmt_auc(m['auc'])} | "
                 f"{pct(m['catch_timeline'])} | {pct(m['catch_non_timeline'])} | {pct(m['false_alarm_controls'])} |")
    L += ["", "Sensitivity, the labelers' agreed truth everywhere (timeline answers included):", "",
          "| Model | Catch | False alarm |", "|---|---|---|"]
    for n, m in M["job2"].items():
        L.append(f"| {n} | {pct(m['catch_labelers_truth'])} | {pct(m['false_alarm_labelers_truth'])} |")
    L += ["", "Post hoc (added after seeing the full-run numbers): AUC within the human-labeled timeline answers only, where "
          "every answer mentions both values, so a model can't separate them by \"mentions the old value\". Human truth: "
          f"{M['human_labels']}", "",
          "| Model | AUC, stale/blends vs current | AUC, stale vs blends |", "|---|---|---|"]
    for n, m in M["job2"].items():
        L.append(f"| {n} | {fmt_auc(m['auc_timeline'])} | {fmt_auc(m['auc_stale_vs_blends'])} |")
    L += ["", "Splits (catch rate):", "",
          "| Model | Newer first | Older first | Header | Body | Version only | Haiku 5.5 answers | Gemma answers |",
          "|---|---|---|---|---|---|---|---|"]
    for n, m in M["job2"].items():
        L.append(f"| {n} | {pct(m['catch_new_first'])} | {pct(m['catch_old_first'])} | "
                 + " | ".join(pct(m[f"catch_{f}"]) for f in DATE_FORMATS) + f" | {pct(m['catch_haiku55'])} | "
                 f"{pct(m['catch_gemma'])} |")

    L += ["", "## Faithfulness baseline (Sonnet 5.5, post 1's question)", "",
          "| Answer label | Passed |", "|---|---|"]
    for g, m in M["baseline"].items():
        L.append(f"| {g} | {pct(m)} |")

    L += ["", "## Latency and cost", "",
          "Measured from this machine: 100 sequential Job 1 calls per model after 5 warm-up calls, no automatic "
          "retries. Strands is local and kept apart.", "",
          "| Model | Where | Median ms | p95 ms | n | Vendor claim | Cost per 1,000 decisions |", "|---|---|---|---|---|---|---|"]
    for n, Lm in M["latency"].items():
        c = M["costs"].get(n, {}).get("per_1000_usd")
        L.append(f"| {n} | {Lm['where']} | {ms(Lm['median_ms'])} | {ms(Lm['p95_ms'])} | {Lm['n']} | "
                 f"{Lm['vendor_claim']} | {'-' if c is None else f'${c:.4f}'} |")

    L += ["", "## Noise (the same request twice)", "", "| Job | Model | Verdict changed | Mean abs. change in P(yes) |",
          "|---|---|---|---|"]
    for k, m in M["noise"].items():
        job, n = k.split("|")
        L.append(f"| {job} | {n} | {pct(m)} | {m['mean_abs_dp']} |")

    L += ["", "## Generators: answer mix", "", "Final label per answer (human for timeline answers).", "",
          "| Generator \\| kind \\| split | Labels |", "|---|---|"]
    for k, c in M["answer_mix"].items():
        L.append(f"| {k} | " + ", ".join(f"{a} {b}" for a, b in sorted(c.items(), key=lambda x: -x[1])) + " |")

    la = M["labeler_agreement"]
    L += ["", "## Labelers", "",
          f"All answers: label agreement {la['all']['label_agree']} (kappa {la['all']['label_kappa']}), "
          f"conflict_noted {la['all']['conflict_noted_agree']}, timeline {la['all'].get('timeline_agree')} "
          f"(n={la['all']['n']}). Human vs labeler on timeline answers: {la['human_vs_labeler_on_timeline']}.", ""]

    L += ["## Hypotheses", "", "```", json.dumps(M["hypotheses"], indent=1, default=str), "```", ""]
    L += ["## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in figs] + [""]
    L += ["## Limitations", "", LIMITATIONS, "## Harness checks", "", "```", json.dumps(M["harness"], indent=1), "```",
          ""]
    path.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    stem = "pilot" if args.pilot else "full"
    _, M = compute(stem)
    (RESULTS / f"metrics_{stem}.json").write_text(json.dumps(M, indent=1, default=str))
    out_dir = RESULTS / "figures_pilot" if args.pilot else FIGURES
    figs = figures(M, out_dir, "")
    md = RESULTS / "pilot_RESULTS.md" if args.pilot else RESULTS_MD
    write_md(M, figs, md, args.pilot)
    print(f"wrote {md}, {RESULTS / f'metrics_{stem}.json'} and {len(figs)} figures in {out_dir}")


if __name__ == "__main__":
    main()
